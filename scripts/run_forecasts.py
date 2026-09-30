"""Produce forecasts, record them, and close the ones that have matured.

This is the step that makes the learning loop turn. L1 built the ledger and
the closure machinery; MEASURED, nothing wrote to it — `record_forecast` had
no caller, and no production path built a forecast at all. The whole F3–F8
engine was exercised only by gates and tests.

    record   for each ticker and horizon, build a forecast and append it
    close    for every OPEN row whose window has since elapsed, score it
    report   publish the scoreboard with its coverage

**Refusals are recorded, and that is the point.** MEASURED over 40 real
probes, 75% of forecasts refuse (30 INSUFFICIENT, 10 INTERVAL). Recording only
the 25% that make a claim would report `scored_share = 100%` while thirty
refusals went unseen — which is precisely the gaming L1 was built to expose,
arriving by the back door: not refusing to SCORE, but refusing to RECORD. The
ledger's job is the denominator. A refusal row costs ~325 bytes; the full
77-ticker portfolio over 252 sessions is ~24 MB/year.

**Recording is separated from closing.** A forecast is written when it is
made and scored only once its window has elapsed, so the two never run
against the same row in one pass. `--record` and `--close` can be run
independently.

**A live run uses TODAY'S chart, and refuses rather than guessing.** The
chart state is the retrieval key for analogs, so anchoring it to whenever a
ticker last produced a memory does not make the forecast slightly stale — it
looks up a different history. MEASURED, that path was a median 144 days behind
(max 535) and retrieved analog sets overlapping the live ones by a Jaccard of
0.205, calling VOO and CIBR bearish while both were bullish. States are now
built from bars ending at `as_of` by the canonical builder in
`core.chart_features`; a ticker whose bars cannot support one is refused with
its reason, never fallen back to memory.

**Idempotent throughout.** Re-recording a forecast is a no-op (deduplicated by
`forecast_id`), and a CLOSED row is terminal. Running this twice in a day
changes nothing, which is what lets it be scheduled without care.

Usage:
    python scripts/run_forecasts.py                 # record, then close
    python scripts/run_forecasts.py --record-only
    python scripts/run_forecasts.py --close-only
    python scripts/run_forecasts.py --dry-run
    python scripts/run_forecasts.py --report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CLOSURE_CLOSED,
    CLOSURE_OPEN,
    EVENT_FORECAST_HORIZONS,
    LIVE_CHART_MAX_STALENESS_DAYS,
    LIVE_CHART_MIN_HISTORY,
)
from core.outcome_closure import (  # noqa: E402
    LEDGER_PATH,
    current_ledger,
    build_ledger_row,
    close_forecast,
    closure_report,
    evaluate_calibration,
    record_forecast,
)

LOGGER = logging.getLogger("run_forecasts")

# The target every recorded forecast is about. One target keeps the ledger
# interpretable while there is no trained model: P(up) is the only thing F4
# and F5 actually produce today, and recording five more that never emit a
# value would fill the store with rows that can only ever close NOT_SCORED.
TARGET = "probability_up"


def _tickers(explicit: str) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    from fetch_data import PORTFOLIO_TICKERS

    return sorted({str(t).upper() for t in PORTFOLIO_TICKERS if str(t).strip()})


def live_chart_states(
    tickers, as_of: str, period: str = "3y"
) -> tuple[dict[str, dict], dict[str, str]]:
    """Today's chart state per ticker, built from bars ending at `as_of`.

    Returns `(states, refusals)`, where a refusal carries the REASON the state
    could not be built. Nothing falls back to the most recent memory's state:
    that fallback is precisely the stale-anchor bug this function replaces, and
    it would restore it while reporting success.

    Point-in-time by construction. The frame is truncated to bars at or before
    `as_of` BEFORE the state is built, so a run replayed for a past date sees
    only what was available then — without it, `--as-of` would silently read
    today's bars and every backfilled forecast would be look-ahead.
    """
    import pandas as pd

    from core.chart_features import ChartFeatureError, chart_state
    from fetch_data import fetch_price_history

    cutoff = pd.Timestamp(as_of)
    states: dict[str, dict] = {}
    refusals: dict[str, str] = {}

    for ticker in tickers:
        try:
            frame = fetch_price_history(ticker, period=period, interval="1d")
        except Exception as exc:  # noqa: BLE001 - a fetch failure is a refusal
            refusals[ticker] = f"price history unavailable ({exc})"
            continue
        if frame is None or frame.empty:
            refusals[ticker] = "price history unavailable (empty frame)"
            continue

        index = pd.to_datetime(frame.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        frame = frame.copy()
        frame.index = index
        frame = frame.loc[frame.index <= cutoff]
        if frame.empty:
            refusals[ticker] = f"no bars at or before {as_of}"
            continue

        last_bar = frame.index[-1]
        staleness = (cutoff - last_bar).days
        if staleness > LIVE_CHART_MAX_STALENESS_DAYS:
            # A weekend or a holiday is fine; a month-old last bar means the
            # feed stopped, and forecasting from it is the stale anchor again.
            refusals[ticker] = (
                f"last bar {last_bar.date()} is {staleness} days before "
                f"{as_of} (limit {LIVE_CHART_MAX_STALENESS_DAYS})"
            )
            continue

        try:
            state = chart_state(frame)
        except ChartFeatureError as exc:
            refusals[ticker] = f"chart state refused ({exc})"
            continue
        if not state:
            refusals[ticker] = (
                f"only {len(frame)} bars at {as_of}; "
                f"{LIVE_CHART_MIN_HISTORY} are required"
            )
            continue
        states[ticker] = state

    return states, refusals


def record_run(tickers, as_of: str, dry_run: bool, path: Path | None) -> dict:
    """Build one forecast per (ticker, horizon) and append it to the ledger."""
    from core.event_memory import find_analogs, load_memory_objects
    from core.forecast_confidence import assess_event_forecast
    from core.forecast_event import build_event_forecast
    from core.forecast_snapshot import forecast_versions

    memories = load_memory_objects()
    if not memories:
        return {
            "recorded": 0, "refused": 0, "attempted": 0,
            "reason": (
                "the event-memory store is empty, so no forecast can be "
                "conditioned on anything — run scripts/build_event_memory.py"
            ),
        }

    scored = [m for m in memories if (m.response_at("20d") or None) is not None]
    base_rate = (
        sum(1 for m in scored if (m.response_at("20d") or 0) > 0) / len(scored)
        if scored
        else None
    )

    # RETRIEVAL NEEDS TODAY'S CHART STATE, NOT THE LAST ONE ON RECORD. This
    # used to read the most recent MEMORY's chart state, which is correct for a
    # backfill (the memory IS the point being forecast from) and wrong for a
    # live run. MEASURED across the 73 tickers holding a memory, that state was
    # a median 144 days old (max 535), and because the chart state is the
    # RETRIEVAL KEY the staleness did not shift the forecast slightly - the
    # stale and live keys retrieved analog sets overlapping by a mean Jaccard
    # of 0.205, and the stale key called VOO and CIBR BEARISH while both were
    # bullish.
    #
    # So the state is built from bars ending at as_of, by the same canonical
    # builder the backfill uses. A ticker whose bars cannot support one is
    # REFUSED, never fallen back to memory: the fallback is the bug.
    states, chart_refusals = live_chart_states(tickers, as_of)

    recorded = refused = attempted = 0
    no_chart: list[str] = sorted(chart_refusals)

    for ticker in tickers:
        state = states.get(ticker)
        if not state:
            if ticker not in chart_refusals:
                no_chart.append(ticker)
            continue

        similarities = [
            entry["similarity"]
            for entry in find_analogs(state, "earnings", memories)
        ]

        for horizon in EVENT_FORECAST_HORIZONS:
            attempted += 1
            event = {
                "event_id": f"scheduled-{ticker}-{as_of}",
                "entity": ticker,
                "event_type": "earnings",
                "direction": "neutral",
                "published_time": as_of,
                "effective_time": as_of,
            }
            forecast = build_event_forecast(
                event, as_of, chart_state=state,
                regime=(state or {}).get("market_regime"),
                memories=memories, base_rate=base_rate, horizon=horizon,
            )
            confidence = assess_event_forecast(forecast, similarities=similarities)

            claim = forecast.get("claim") or "INSUFFICIENT"
            stages = forecast.get("stages") or {}
            matches = stages.get("matches") or {}
            row = build_ledger_row(
                ticker, as_of, horizon, TARGET,
                claim=claim,
                # The shape rule: a value only where the claim carried one.
                value=forecast.get("value") if claim == "POINT" else None,
                interval=forecast.get("interval"),
                direction=(stages.get("regime") or {}).get("direction"),
                samples=int(forecast.get("samples") or 0),
                confidence=confidence.get("confidence"),
                forecast_version=forecast_versions(),
                # L2's POINT-IN-TIME dimensions, captured HERE because they
                # cannot be recovered later. MEASURED, 7 of 8 retrieval probes
                # returned a different analog count as the store grew, and a
                # recomputed regime depends on whichever classifier version
                # runs at closing time.
                regime=(state or {}).get("market_regime"),
                event_type=event["event_type"],
                observed_share=matches.get("observed_share"),
                volatility=(state or {}).get("volatility"),
            )
            if claim == "INSUFFICIENT":
                refused += 1
            if dry_run:
                recorded += 1
                continue
            record_forecast(row, path)
            recorded += 1

    return {
        "recorded": recorded,
        "refused": refused,
        "attempted": attempted,
        "no_chart_state": sorted(set(no_chart)),
        "chart_refusals": chart_refusals,
        "base_rate": base_rate,
        "reason": "",
    }


def close_run(as_of: str, dry_run: bool, path: Path | None) -> dict:
    """Score every OPEN forecast whose window has since elapsed."""
    from core.labels import build_outcome_labels

    # The collapsed view: a forecast already CLOSED must not be scored again,
    # and the append-only store holds both its OPEN and CLOSED rows.
    rows = current_ledger(path)
    open_rows = [r for r in rows if r.get("state") == CLOSURE_OPEN]
    if not open_rows:
        return {"closed": 0, "still_open": 0, "examined": len(rows), "rows": []}

    label_cache: dict[tuple[str, str], dict] = {}
    closed_rows: list[dict] = []
    closed = still_open = 0

    for row in open_rows:
        key = (row.get("ticker"), row.get("as_of"))
        if key not in label_cache:
            try:
                label_cache[key] = build_outcome_labels(key[0], key[1])
            except Exception as exc:
                LOGGER.warning("labels unavailable for %s: %s", key, exc)
                label_cache[key] = {}
        result = close_forecast(row, label_cache[key])
        closed_rows.append(result)
        if result.get("state") == CLOSURE_CLOSED:
            closed += 1
        else:
            still_open += 1

    if not dry_run and closed:
        # The ledger is append-only, so a closure is a NEW row carrying the
        # outcome. The original forecast is never rewritten — that is what
        # makes "what did we believe at the time?" answerable later.
        store = Path(path) if path is not None else LEDGER_PATH
        store.parent.mkdir(parents=True, exist_ok=True)
        with store.open("a", encoding="utf-8") as handle:
            for result in closed_rows:
                if result.get("state") == CLOSURE_CLOSED:
                    handle.write(
                        json.dumps(result, sort_keys=True, default=str) + "\n"
                    )

    return {
        "closed": closed,
        "still_open": still_open,
        "examined": len(rows),
        "rows": closed_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--close-only", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="print the scoreboard and exit")
    parser.add_argument("--performance", default="",
                        help="print the L2 breakdown for one dimension and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    path = Path(args.ledger) if args.ledger else None

    if args.performance:
        from core.config import PERFORMANCE_DIMENSIONS  # noqa: PLC0415
        from core.performance_ledger import (
            breakdown,
            render_breakdown,
        )

        if args.performance not in PERFORMANCE_DIMENSIONS:
            print(f"unknown dimension {args.performance!r}")
            print(f"known: {', '.join(PERFORMANCE_DIMENSIONS)}")
            return 2
        closed = [
            r for r in current_ledger(path) if r.get("state") == CLOSURE_CLOSED
        ]
        table = breakdown(closed, args.performance)
        print(f"performance by {args.performance} [{table['forecasts']} closed]")
        print(f"  {'level':<20} {'n':>5} {'scored':>7} {'brier':>8} {'cover':>7}")
        for row in render_breakdown(table):
            brier = f"{row['mean_brier']:.4f}" if row["mean_brier"] is not None else "-"
            cover = (
                f"{row['coverage_rate']:.0%}"
                if row["coverage_rate"] is not None
                else "-"
            )
            print(
                f"  {row['level']:<20} {row['forecasts']:>5} {row['scored']:>7} "
                f"{brier:>8} {cover:>7}"
                + ("" if row["sufficient"] else "   (thin)")
            )
        if table["thin_levels"]:
            print(
                f"  {len(table['thin_levels'])} level(s) below the "
                f"{table['min_cell']}-forecast floor publish no metric."
            )
        return 0

    if args.report:
        rows = current_ledger(path)
        closed = [r for r in rows if r.get("state") == CLOSURE_CLOSED]
        report = closure_report(closed)
        print(f"forecast scoreboard [{len(rows)} ledger rows]")
        print(f"  scored   : {report['scored']}")
        print(f"  refused  : {report['refused']}")
        print(f"  pending  : {len(rows) - len(closed)}")
        print(f"  coverage : {report['scored_share']:.0%}  reliable={report['reliable']}")
        if report["mean_brier"] is not None:
            print(f"  brier    : {report['mean_brier']:.4f} over {report['brier_count']}")
        if report["coverage_rate"] is not None:
            print(
                f"  interval coverage: {report['coverage_rate']:.0%} over "
                f"{report['coverage_count']}"
            )
        if not report["reliable"]:
            print(f"  NOTE: {report['reliability_reason']}")
        calibration = evaluate_calibration(closed)
        print(f"  calibration: {calibration['status']} "
              f"({calibration.get('observations', 0)} observations)")
        return 0

    tickers = _tickers(args.tickers)
    print(f"forecast run [{args.as_of}]" + ("  (dry run)" if args.dry_run else ""))

    if not args.close_only:
        recorded = record_run(tickers, args.as_of, args.dry_run, path)
        if recorded.get("reason"):
            print(f"  record : SKIPPED — {recorded['reason']}")
        else:
            print(
                f"  record : {recorded['recorded']} row(s) over "
                f"{len(tickers)} ticker(s) x {len(EVENT_FORECAST_HORIZONS)} "
                f"horizon(s); {recorded['refused']} refused"
            )
            if recorded["no_chart_state"]:
                names = recorded["no_chart_state"]
                print(
                    f"      {len(names)} ticker(s) had no live chart state and "
                    f"were skipped: {', '.join(names[:6])}"
                    + (" ..." if len(names) > 6 else "")
                )
                # The REASON matters operationally: a short history is a
                # permanent property of a young listing, while a stale last bar
                # means the feed stopped and someone has to look.
                reasons = recorded.get("chart_refusals") or {}
                for ticker in sorted(reasons)[:6]:
                    print(f"        {ticker}: {reasons[ticker]}")
            print(
                "      refusals ARE recorded: the ledger is the denominator, "
                "and dropping them is how a scoreboard gets gamed."
            )

    if not args.record_only:
        closed = close_run(args.as_of, args.dry_run, path)
        print(
            f"  close  : {closed['closed']} matured and scored, "
            f"{closed['still_open']} still open, of {closed['examined']} row(s)"
        )
        if closed["closed"]:
            report = closure_report(
                [r for r in closed["rows"] if r.get("state") == CLOSURE_CLOSED]
            )
            print(
                f"      scored {report['scored']}, refused {report['refused']}, "
                f"coverage {report['scored_share']:.0%}, "
                f"reliable={report['reliable']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
