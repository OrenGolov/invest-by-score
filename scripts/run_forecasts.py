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
)
from core.outcome_closure import (  # noqa: E402
    LEDGER_PATH,
    current_ledger,
    build_ledger_row,
    close_forecast,
    closure_report,
    evaluate_calibration,
    load_ledger,
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


def record_run(tickers, as_of: str, dry_run: bool, path: Path | None) -> dict:
    """Build one forecast per (ticker, horizon) and append it to the ledger."""
    from core.event_memory import find_analogs, load_memory_objects
    from core.forecast_confidence import assess_event_forecast
    from core.forecast_event import build_event_forecast

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

    # Retrieval needs a chart state per ticker. The most recent memory for a
    # ticker carries one that is already PIT-correct for its own as_of; for a
    # live run the caller would supply today's. Absent that, a ticker with no
    # memory is skipped rather than forecast from nothing.
    latest: dict[str, dict] = {}
    for memory in memories:
        key = str(memory.ticker).upper()
        if key not in latest or memory.published_time > latest[key]["when"]:
            latest[key] = {
                "when": memory.published_time,
                "chart_state": memory.chart_state,
            }

    recorded = refused = attempted = 0
    no_chart: list[str] = []

    for ticker in tickers:
        state = (latest.get(ticker) or {}).get("chart_state")
        if not state:
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
            row = build_ledger_row(
                ticker, as_of, horizon, TARGET,
                claim=claim,
                # The shape rule: a value only where the claim carried one.
                value=forecast.get("value") if claim == "POINT" else None,
                interval=forecast.get("interval"),
                direction=(forecast.get("stages") or {})
                .get("regime", {})
                .get("direction"),
                samples=int(forecast.get("samples") or 0),
                confidence=confidence.get("confidence"),
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
        "no_chart_state": sorted(no_chart),
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    path = Path(args.ledger) if args.ledger else None

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
                    f"      {len(names)} ticker(s) had no chart state and were "
                    f"skipped: {', '.join(names[:6])}"
                    + (" ..." if len(names) > 6 else "")
                )
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
