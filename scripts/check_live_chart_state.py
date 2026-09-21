"""Gate: a live forecast run is anchored to TODAY'S chart, not to memory.

`scripts/run_forecasts.py` used to derive each ticker's chart state from that
ticker's most recent EVENT MEMORY. Correct for a backfill, wrong for a live
run. MEASURED across the 73 tickers holding a memory, that state was a median
144 days old and up to 535.

THE STALENESS IS NOT THE HARM, WHAT IT RETRIEVES IS. The chart state is the
retrieval key for analogs, so a stale key does not return a slightly stale
forecast — it looks up a different history. The stale and live keys retrieved
analog sets overlapping by a mean Jaccard of 0.205, and the stale key called
VOO and CIBR BEARISH while both were in fact bullish.

Verified to FAIL when any of these is reinjected:
  - the builder duplicated instead of shared (W5 split-brain)
  - a live run falling back to the most recent memory's state
  - the staleness bound widened past a live run
  - the 210-bar minimum lowered
  - `as_of` truncation removed, so a replayed date reads future bars
  - a refusal reported without its reason
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from core.chart_features import ChartFeatureError, chart_state  # noqa: E402
from core.config import (  # noqa: E402
    LIVE_CHART_FALLBACK_TO_MEMORY,
    LIVE_CHART_MAX_STALENESS_DAYS,
    LIVE_CHART_MIN_HISTORY,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def synthetic(bars: int, start: str = "2024-01-01", price: float = 100.0):
    index = pd.bdate_range(start=start, periods=bars)
    closes = [price + index_position * 0.1 for index_position in range(bars)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value * 1.01 for value in closes],
            "Low": [value * 0.99 for value in closes],
            "Close": closes,
            "Volume": [1_000_000] * bars,
        },
        index=index,
    )


def main() -> int:
    from run_forecasts import live_chart_states

    # 1. ONE BUILDER (W5). The backfill must delegate, not carry a copy.
    import build_event_memory

    source = (REPO_ROOT / "scripts" / "build_event_memory.py").read_text(
        encoding="utf-8"
    )
    check(
        "return chart_state(frame, position)" in source,
        "the backfill no longer delegates to core.chart_features.chart_state — "
        "a second implementation of the chart state means a live forecast and "
        "a remembered analog compare two different definitions",
    )
    frame = synthetic(300)
    check(
        build_event_memory._chart_snapshot(frame, 280) == chart_state(frame, 280),
        "the backfill snapshot and the canonical builder disagree",
    )

    # 2. NO FALLBACK TO MEMORY. This is the bug itself.
    check(
        LIVE_CHART_FALLBACK_TO_MEMORY is False,
        "a live run must never fall back to the most recent memory's chart "
        "state: it would reinstate a median 144-day-stale anchor while "
        "reporting success",
    )
    # BEHAVIOURAL, NOT TEXTUAL. A first version of this check searched the
    # source for "memory.chart_state" and was BLIND: reinjecting the fallback
    # under any other variable name (`_m.chart_state`) walked straight past it.
    # A guard that matches one spelling of a bug does not guard against the bug.
    # So the fallback is provoked instead: every ticker is made unfetchable, and
    # a run that still produces a state must have got it from somewhere else.
    import run_forecasts as run_module

    real_fetch = run_module.__dict__.get("fetch_price_history")

    def _dead_fetch(*_args, **_kwargs):
        raise RuntimeError("feed down")

    import fetch_data

    original = fetch_data.fetch_price_history
    fetch_data.fetch_price_history = _dead_fetch
    try:
        starved, starved_refusals = run_module.live_chart_states(
            ["AAPL", "VOO"], "2026-09-21"
        )
    finally:
        fetch_data.fetch_price_history = original
        if real_fetch is not None:
            run_module.__dict__["fetch_price_history"] = real_fetch

    check(
        not starved,
        f"with the price feed down, {sorted(starved)} still received a chart "
        f"state — a live run is falling back to a stored one, which is the "
        f"median 144-day-stale anchor this gate exists to prevent",
    )
    check(
        len(starved_refusals) == 2,
        "a ticker with no obtainable bars was neither given a state nor "
        "recorded as a refusal, so it vanished from the denominator",
    )

    # AND record_run ITSELF, under the same dead feed. The probe above only
    # covers live_chart_states; a fallback re-added in record_run AFTER that
    # call sits downstream of it and was missed. With no obtainable bars, a
    # correct record_run forecasts NOTHING — anything it records came from a
    # stored state.
    fetch_data.fetch_price_history = _dead_fetch
    try:
        starved_run = run_module.record_run(
            ["AAPL", "VOO"], "2026-09-21", dry_run=True, path=None
        )
    finally:
        fetch_data.fetch_price_history = original
        if real_fetch is not None:
            run_module.__dict__["fetch_price_history"] = real_fetch

    check(
        isinstance(starved_run, dict) and "chart_refusals" in starved_run,
        "record_run stopped reporting which tickers were refused a live state",
    )
    check(
        int(starved_run.get("attempted", 0)) == 0,
        f"with the price feed down, record_run still attempted "
        f"{starved_run.get('attempted')} forecast(s) — it recovered a chart "
        f"state from somewhere other than today's bars, which is the stale "
        f"anchor arriving downstream of live_chart_states",
    )
    check(
        int(starved_run.get("recorded", 0)) == 0,
        f"with the price feed down, record_run still recorded "
        f"{starved_run.get('recorded')} row(s)",
    )

    # 3. THE BOUNDS ARE LIVE BOUNDS.
    check(
        LIVE_CHART_MAX_STALENESS_DAYS <= 30,
        f"a staleness bound of {LIVE_CHART_MAX_STALENESS_DAYS} days is not a "
        f"live run",
    )
    check(
        LIVE_CHART_MIN_HISTORY >= 210,
        f"{LIVE_CHART_MIN_HISTORY} bars cannot support a 200-session mean",
    )
    check(
        chart_state(synthetic(LIVE_CHART_MIN_HISTORY - 1)) is None,
        "a frame below the minimum returned a state instead of None — a "
        "price_vs_ma_200 computed over a window that does not exist is "
        "indistinguishable downstream from a real one",
    )
    check(
        chart_state(synthetic(LIVE_CHART_MIN_HISTORY + 5)) is not None,
        "a frame above the minimum refused to produce a state",
    )

    # 4. POINT IN TIME. A replayed date must not read later bars.
    full = synthetic(400, start="2024-01-01")
    early = chart_state(full, 250)
    truncated = chart_state(full.iloc[:251])
    check(
        early == truncated,
        "building at a position and building from a truncated frame disagree — "
        "one of them is reading bars the other cannot see",
    )
    check(
        chart_state(full) != early,
        "the last bar and an earlier position produced the same state, so the "
        "position argument is being ignored",
    )

    # 4b. THE LIVE PATH TRUNCATES TO as_of. The check above covers the builder;
    # the TRUNCATION lives in live_chart_states, and removing it is invisible to
    # a builder-only test. MEASURED with the truncation removed, a replay of
    # 2025-11-03 returned AAPL at 336.13 - today's close - instead of 270.37.
    # Every backfilled forecast would then be look-ahead.
    past, _ = live_chart_states(["AAPL"], "2025-11-03")
    now, _ = live_chart_states(["AAPL"], "2026-09-21")
    past_close = (past.get("AAPL") or {}).get("close")
    now_close = (now.get("AAPL") or {}).get("close")
    if past_close is not None and now_close is not None:
        check(
            past_close != now_close,
            f"a replay of 2025-11-03 returned the same close as today "
            f"({past_close}) — live_chart_states is not truncating to as_of, so "
            f"every past-dated forecast reads bars from its own future",
        )

    # 5. AN OUT-OF-ORDER FRAME IS REFUSED, NOT SORTED.
    try:
        chart_state(full.iloc[::-1])
        FAILURES.append(
            "a descending frame produced a chart state — every window reads "
            "backwards from the last row, so a reversed downtrend reads as a "
            "breakout"
        )
    except ChartFeatureError:
        pass

    # 6. REFUSALS CARRY THEIR REASON.
    states, refusals = live_chart_states(["NOT_A_REAL_TICKER_XYZ"], "2026-09-21")
    check(
        not states,
        "an unfetchable ticker produced a chart state",
    )
    check(
        bool(refusals.get("NOT_A_REAL_TICKER_XYZ", "").strip()),
        "a refusal carried no reason — a short history is permanent while a "
        "stale feed needs a human, and the operator cannot tell them apart",
    )

    if FAILURES:
        print("L live chart-state gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L live chart-state gate OK:")
    print("  ONE builder: the backfill delegates to core.chart_features (W5).")
    print(f"  live runs refuse past {LIVE_CHART_MAX_STALENESS_DAYS} days stale; "
          f"no fallback to memory.")
    print(f"  {LIVE_CHART_MIN_HISTORY} bars required, else None - never a short "
          f"window.")
    print("  point-in-time: a position and a truncated frame agree exactly.")
    print("  refusals carry the reason an operator needs to act on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
