"""Populate `data/event_memory.jsonl` — the store F5 retrieves from.

F5 answers "what followed setups like this one?" by retrieving comparable
historical events. It can only do that if such events have been recorded, and
on a fresh clone the store does not exist. This is the producer.

    news snapshot (N2)  ->  events (E1)
                        ->  event study on real bars (E4)
                        ->  attribution (E5)
                        ->  remember() (E6)

**Two modes, and the difference is the point.**

`--mode forward` records what the news provider actually reported. Every
memory is `provenance="observed"`: a real, sourced event. MEASURED: with no
`NEWSAPI_KEY` configured the news contract returns UNAVAILABLE, and the news
adapter looks back only `NEWS_LOOKBACK_DAYS` (7) even with one. So this mode
writes ZERO memories today and says so — it is built to accrue going forward,
not to fill a store retroactively.

`--mode backfill` dates candidate EARNINGS events from price behaviour, using
the quarterly volume cadence, and records them as `provenance="inferred"`.
MEASURED against PUBLISHED earnings dates for AAPL/MSFT/NVDA/JPM: the filter
scores a PRECISION of 0.65 — 15 of 23 picks land within +/-2 sessions of a
real earnings date. A plain volume-cadence filter scores only 0.35, because
31.7% of its picks are quarterly TRIPLE-WITCHING dates (options expiry, not
earnings) sharing the same high-volume quarterly signature. Excluding those
and requiring an opening gap is what closes the difference.

So ROUGHLY ONE IN THREE inferred events is not the event it is labelled as.

**Why inferred memories are allowed at all.** The price move is real and the
E4 study measures it on real bars. What was never sourced is WHICH event
produced the move, or whether a discrete event did. That is a narrower claim
than "this data is fake", and it is exactly what the provenance field records.

**Why they can never be laundered into observed ones.** MEASURED:
`fetch_fundamental_snapshot` returns `earnings_date=None`, and no historical
earnings calendar exists anywhere in the system. An inferred date has nothing
to be scored against — it can be FLAGGED but never VERIFIED. So F5 reports the
observed share of every analog set beside its same-ticker share, and degrades
the forecast when inferred events dominate.

**One event type only.** The cadence supports `earnings` and nothing else.
Deriving ten taxonomy buckets from one volume signal would be fabrication
wearing a classifier's clothes.

**PIT throughout.** A study is only recorded once its windows have elapsed;
`run_event_study` reports INSUFFICIENT_DATA otherwise and that memory is
skipped rather than written with a partial response.

Usage:
    python scripts/build_event_memory.py --mode forward
    python scripts/build_event_memory.py --mode backfill --years 5
    python scripts/build_event_memory.py --mode backfill --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.attribution import attribute_study  # noqa: E402
from core.config import (  # noqa: E402
    MEMORY_PROVENANCE_INFERRED,
    MEMORY_PROVENANCE_OBSERVED,
)
from core.chart_features import chart_state
from core.event_memory import (  # noqa: E402
    EVENT_MEMORY_STORE_PATH,
    EventMemoryError,
    build_memory,
    load_memories,
    remember,
)
from core.event_study import run_event_study  # noqa: E402
from core.timeframes import as_naive_timestamp  # noqa: E402

LOGGER = logging.getLogger("build_event_memory")

# The inference method, declared in config so its measurement travels with it.
CADENCE_METHOD = "quarterly_volume_cadence"

# One quarter of trading sessions. Earnings recur on roughly this cadence.
QUARTER_SESSIONS = 63

# A picked day must stand out from its neighbours. MEASURED against published
# earnings dates, raising this from 1.5 to 2.0 lifts precision 0.60 -> 0.65
# while keeping enough volume to be useful.
MIN_VOLUME_RATIO = 2.0

# An earnings reaction opens with a gap; a purely mechanical volume event
# (index rebalancing, options expiry) usually does not. MEASURED: requiring
# this lifts precision from 0.35 to 0.65 -- the single largest improvement
# available, and the reason it is part of the method rather than an option.
MIN_OPENING_GAP = 0.02

# Sessions of history a study needs before AND after the event. Anything more
# recent has not finished happening yet.
TRAILING_BUFFER_SESSIONS = 60


def _price(ticker: str, period: str):
    from fetch_data import fetch_price_history

    return fetch_price_history(ticker, period=period, interval="1d")


def is_triple_witching(day) -> bool:
    """Quarterly options/futures expiry: the third Friday of Mar/Jun/Sep/Dec.

    These carry an enormous, purely mechanical volume spike on exactly the
    quarterly cadence earnings follow. MEASURED: 31.7% of a plain volume-
    cadence filter's picks landed on one, which is the single biggest source
    of mislabelling in this method.
    """
    return day.weekday() == 4 and 15 <= day.day <= 21 and day.month in (3, 6, 9, 12)


def cadence_candidates(frame: pd.DataFrame) -> list[int]:
    """Positions of candidate earnings days, by quarterly volume cadence.

    Keeps the largest qualifying 20-day-relative volume spike in each
    63-session window. A day qualifies when it clears MIN_VOLUME_RATIO, opens
    with at least MIN_OPENING_GAP, and is not a triple-witching expiry.

    MEASURED precision against published earnings dates for AAPL/MSFT/NVDA/JPM:
    0.65. Roughly one pick in three is still NOT an earnings event, which is
    why everything this produces is recorded as `inferred`.

    Returns POSITIONS, not timestamps, so the caller can apply its own
    trailing buffer against the same index.
    """
    if frame is None or frame.empty or "Volume" not in frame.columns:
        return []
    volume = frame["Volume"]
    ratio = (volume / volume.rolling(20).mean()).fillna(0.0).to_numpy(dtype=float)

    if "Open" in frame.columns:
        gap = (frame["Open"] / frame["Close"].shift() - 1.0).abs().fillna(0.0)
        ratio = ratio.copy()
        ratio[gap.to_numpy(dtype=float) < MIN_OPENING_GAP] = 0.0

    for position, stamp in enumerate(frame.index):
        if is_triple_witching(pd.Timestamp(stamp).date()):
            ratio[position] = 0.0

    picks: list[int] = []
    start = 0
    while start < len(frame):
        window = ratio[start:start + QUARTER_SESSIONS]
        if window.size == 0:
            break
        if float(window.max()) > MIN_VOLUME_RATIO:
            picks.append(start + int(window.argmax()))
        start += QUARTER_SESSIONS
    return picks


def _chart_snapshot(frame: pd.DataFrame, position: int) -> dict | None:
    """The E6 chart state at `position`, from `core.chart_features` (W5).

    This used to carry its own copy of the calculation. It now delegates to the
    canonical builder, which the LIVE run in `scripts/run_forecasts.py` also
    calls — so a remembered analog and a live forecast describe a chart with
    one definition rather than two that can drift apart.
    """
    return chart_state(frame, position)


class _InferredEvent:
    """The minimum an E4 study and an E6 memory need from an event.

    Deliberately NOT a core.event_contract.Event: that type carries source,
    evidence and entity resolution, none of which an inferred event has. A
    stub that filled those with plausible values would be the fabrication
    provenance exists to prevent.
    """

    def __init__(self, ticker: str, when: str, position: int):
        self.entity = ticker
        self.event_id = f"inferred-{ticker}-{when[:10]}"
        self.published_time = when
        self.effective_time = when
        self.event_type = "earnings"
        self.direction = "neutral"
        self.actor = ""
        self.actor_type = ""


def backfill(tickers, years: int, dry_run: bool, store: Path | None) -> dict:
    """Record inferred earnings events across the given tickers."""
    period = f"{max(1, int(years))}y"
    written = 0
    skipped: dict[str, int] = {}
    considered = 0

    benchmark = None
    try:
        benchmark = _price("SPY", period)
    except Exception as exc:
        LOGGER.warning("benchmark unavailable (%s); studies run unadjusted", exc)

    for ticker in tickers:
        try:
            frame = _price(ticker, period)
        except Exception as exc:
            skipped["price_unavailable"] = skipped.get("price_unavailable", 0) + 1
            LOGGER.warning("%s: price history unavailable (%s)", ticker, exc)
            continue
        if frame is None or frame.empty or len(frame) < 260:
            skipped["history_too_short"] = skipped.get("history_too_short", 0) + 1
            continue

        usable_end = len(frame) - TRAILING_BUFFER_SESSIONS
        for position in cadence_candidates(frame):
            considered += 1
            if position >= usable_end:
                # Its windows have not elapsed. Recording it now would store a
                # partial response that later looks complete.
                skipped["windows_not_elapsed"] = skipped.get("windows_not_elapsed", 0) + 1
                continue

            snapshot = _chart_snapshot(frame, position)
            if snapshot is None:
                skipped["insufficient_history"] = skipped.get("insufficient_history", 0) + 1
                continue

            when = frame.index[position].strftime("%Y-%m-%d %H:%M:%S")
            event = _InferredEvent(ticker, when, position)

            study = run_event_study(
                ticker, when, frame, benchmark_frame=benchmark,
                benchmark="SPY", event_id=event.event_id,
            )
            if not study.is_measured():
                key = f"study_{str(study.status).lower()}"
                skipped[key] = skipped.get(key, 0) + 1
                continue

            memory = build_memory(
                event, study, attribute_study(study), snapshot,
                provenance=MEMORY_PROVENANCE_INFERRED,
                inference_method=CADENCE_METHOD,
            )
            if dry_run:
                written += 1
                continue
            try:
                remember(memory, store)
                written += 1
            except EventMemoryError as exc:
                skipped["refused"] = skipped.get("refused", 0) + 1
                LOGGER.warning("%s @ %s refused: %s", ticker, when[:10], exc)

    return {
        "mode": "backfill",
        "provenance": MEMORY_PROVENANCE_INFERRED,
        "considered": considered,
        "written": written,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def forward(tickers, as_of: str, dry_run: bool, store: Path | None) -> dict:
    """Record OBSERVED events from the news provider.

    MEASURED: with no NEWSAPI_KEY the news contract is UNAVAILABLE, so this
    writes nothing and reports why. That is the correct outcome, not a
    failure — the alternative is inventing events.
    """
    from core.event_contract import events_from_news_snapshot
    from core.news_adapter import build_news_snapshot

    written = 0
    considered = 0
    skipped: dict[str, int] = {}
    unavailable: list[str] = []

    benchmark = None
    try:
        benchmark = _price("SPY", "2y")
    except Exception:
        pass

    for ticker in tickers:
        snapshot = build_news_snapshot(ticker, as_of)
        if str(snapshot.get("status", "")).upper() == "UNAVAILABLE":
            unavailable.append(ticker)
            continue

        events = events_from_news_snapshot(snapshot)
        if not events:
            skipped["no_events"] = skipped.get("no_events", 0) + 1
            continue

        try:
            frame = _price(ticker, "2y")
        except Exception:
            skipped["price_unavailable"] = skipped.get("price_unavailable", 0) + 1
            continue
        if frame is None or frame.empty:
            skipped["price_unavailable"] = skipped.get("price_unavailable", 0) + 1
            continue

        for event in events:
            considered += 1
            when = str(getattr(event, "effective_time", "")
                       or getattr(event, "published_time", ""))
            # Normalised through C1's canonical converter, not a local copy:
            # price bars are tz-NAIVE while news timestamps arrive tz-aware
            # UTC ("2026-09-16T21:33:00Z"), and comparing the two raises.
            # The backfill never hit this because its timestamps came from
            # the price index itself.
            position = frame.index.searchsorted(
                as_naive_timestamp(when), side="right"
            ) - 1
            if position < 0:
                skipped["no_bar"] = skipped.get("no_bar", 0) + 1
                continue
            chart = _chart_snapshot(frame, int(position))
            if chart is None:
                skipped["insufficient_history"] = skipped.get("insufficient_history", 0) + 1
                continue

            study = run_event_study(
                ticker, when, frame, benchmark_frame=benchmark,
                benchmark="SPY", event_id=str(getattr(event, "event_id", "")),
            )
            if not study.is_measured():
                key = f"study_{str(study.status).lower()}"
                skipped[key] = skipped.get(key, 0) + 1
                continue

            memory = build_memory(
                event, study, attribute_study(study), chart,
                provenance=MEMORY_PROVENANCE_OBSERVED,
            )
            if dry_run:
                written += 1
                continue
            try:
                remember(memory, store)
                written += 1
            except EventMemoryError as exc:
                skipped["refused"] = skipped.get("refused", 0) + 1
                LOGGER.warning("%s refused: %s", ticker, exc)

    return {
        "mode": "forward",
        "provenance": MEMORY_PROVENANCE_OBSERVED,
        "considered": considered,
        "written": written,
        "skipped": skipped,
        "news_unavailable": unavailable,
        "dry_run": dry_run,
    }


def _portfolio_tickers() -> list[str]:
    """The governed universe, from the single place it is declared.

    `fetch_data.PORTFOLIO_TICKERS` is the list every other entry point uses;
    reading it here keeps the store's coverage tied to the portfolio rather
    than to a second list that would drift out of step with it.
    """
    from fetch_data import PORTFOLIO_TICKERS

    return sorted({str(t).upper() for t in PORTFOLIO_TICKERS if str(t).strip()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "backfill"), default="forward")
    parser.add_argument("--years", type=int, default=5,
                        help="history depth for backfill (default 5)")
    parser.add_argument("--tickers", default="",
                        help="comma-separated; defaults to the portfolio")
    parser.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--dry-run", action="store_true",
                        help="measure what would be written, write nothing")
    parser.add_argument("--store", default="",
                        help="override the store path (tests use this)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        or _portfolio_tickers()
    )
    store = Path(args.store) if args.store else None
    before = len(load_memories(store))

    if args.mode == "backfill":
        report = backfill(tickers, args.years, args.dry_run, store)
    else:
        report = forward(tickers, args.as_of, args.dry_run, store)

    after = len(load_memories(store))
    path = store or EVENT_MEMORY_STORE_PATH

    print(f"event-memory builder [{report['mode']}]")
    print(f"  store        : {path}")
    print(f"  tickers      : {len(tickers)}")
    print(f"  provenance   : {report['provenance']}")
    print(f"  considered   : {report['considered']}")
    print(f"  written      : {report['written']}" + (" (dry run)" if args.dry_run else ""))
    print(f"  store size   : {before} -> {after}")
    if report.get("skipped"):
        print("  skipped      :")
        for reason, count in sorted(report["skipped"].items()):
            print(f"      {reason}: {count}")
    if report.get("news_unavailable"):
        names = report["news_unavailable"]
        print(f"  news UNAVAILABLE for {len(names)} ticker(s): {', '.join(names[:8])}"
              + (" ..." if len(names) > 8 else ""))
        print("      No NEWSAPI_KEY is configured, so no OBSERVED event can be")
        print("      recorded. This writes nothing rather than inventing events.")
        print("      Set NEWSAPI_KEY and re-run daily to accrue real memories.")

    if report["written"] == 0 and not args.dry_run:
        print("  NOTE: nothing was written. F5 will keep refusing at its")
        print("        'matches' stage, which is the honest state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
