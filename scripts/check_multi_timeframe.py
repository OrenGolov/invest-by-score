"""CI drift gate for the C1 multi-timeframe representation.

Proves, on every push, that the alignment guarantee Sprint C depends on holds:

1. every timeframe in TIMEFRAME_ORDER answers, for the SAME as_of;
2. a still-forming coarse bar is EXCLUDED — the C1 leak: a weekly bar is
   labelled at period start, so at a mid-week as_of its Close is Friday's,
   and a naive `index <= as_of` filter would keep future data;
3. excluded bars are COUNTED, so a quietly shrinking timeframe is visible;
4. a timeframe below its minimum bar count is INCOMPLETE with NO trend;
5. the snapshot status is the WORST timeframe, never an average;
6. yearly is RESAMPLED from monthly bars, not fetched as a second price
   truth (W5);
7. moving as_of earlier never reveals more bars (PIT monotonicity);
8. the build is deterministic.

Synthetic only, so it runs in well under a second and needs no network.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.config import (  # noqa: E402
    TIMEFRAME_MIN_BARS,
    TIMEFRAME_ORDER,
    TIMEFRAME_SPECS,
)
from core.timeframes import (  # noqa: E402
    STATUS_INCOMPLETE,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    build_timeframe_snapshot,
    build_timeframe_state,
    eligible_bars,
    snapshot_problems,
)

AS_OF = "2026-09-15"


def _frame(dates, closes=None):
    index = pd.to_datetime(dates)
    closes = closes if closes is not None else [100.0 + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(index),
        },
        index=index,
    )


def _healthy_frames():
    return {
        "1h": _frame(pd.date_range("2026-09-01", periods=100, freq="h")),
        "1d": _frame(pd.date_range("2026-01-01", periods=200, freq="D")),
        "1wk": _frame(pd.date_range("2025-01-06", periods=80, freq="W-MON")),
        "1mo": _frame(pd.date_range("2018-01-01", periods=90, freq="MS")),
    }


def _fetcher(frames):
    def fetch(ticker, period, interval):
        if interval not in frames:
            raise RuntimeError(f"no fixture for interval {interval}")
        return frames[interval]
    return fetch


def main() -> int:
    failures: list[str] = []

    snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=_fetcher(_healthy_frames()))

    # 1. every clock answers, on one as_of
    missing = [name for name in TIMEFRAME_ORDER if name not in snapshot["timeframes"]]
    if missing:
        failures.append(f"timeframes absent from the snapshot: {missing}")
    if snapshot["status"] != STATUS_OK:
        failures.append(f"healthy fixture did not read OK (got {snapshot['status']})")
    for problem in snapshot_problems(snapshot):
        failures.append(f"contract problem: {problem}")

    # 2. the leak: a bar labelled in the past that closes in the future
    leaking = _frame(["2026-08-31", "2026-09-07", "2026-09-14"], [100.0, 200.0, 999.0])
    eligible, unclosed, _ = eligible_bars(leaking, AS_OF, "7D")
    if 999.0 in list(eligible["Close"]):
        failures.append(
            "a still-forming weekly bar survived the filter — future data is "
            "reaching the representation"
        )

    # 3. exclusions are counted
    if unclosed != 1:
        failures.append(f"excluded bars are not being counted (unclosed={unclosed})")

    # 4. below minimum -> INCOMPLETE, no trend
    thin = build_timeframe_state("daily", _frame(["2026-09-01", "2026-09-02"]), AS_OF)
    if thin.status != STATUS_INCOMPLETE or thin.trend is not None:
        failures.append(
            f"a thin timeframe must be INCOMPLETE with no trend "
            f"(got {thin.status}/{thin.trend})"
        )

    # 5. worst-wins status
    degraded_frames = _healthy_frames()
    degraded_frames["1wk"] = _frame(["2026-09-14"])  # only an unclosed bar
    degraded = build_timeframe_snapshot("MSFT", AS_OF, fetcher=_fetcher(degraded_frames))
    if degraded["status"] != STATUS_INCOMPLETE:
        failures.append(
            f"snapshot status must be the worst timeframe (got {degraded['status']})"
        )

    # 6. yearly is derived, not a second price truth
    if TIMEFRAME_SPECS["yearly"]["interval"] != "1mo" or not TIMEFRAME_SPECS["yearly"]["resample"]:
        failures.append("yearly must be resampled from monthly bars (W5: one price truth)")

    # 7. PIT monotonicity
    fetcher = _fetcher(_healthy_frames())
    late = build_timeframe_snapshot("MSFT", "2026-09-15", fetcher=fetcher)
    early = build_timeframe_snapshot("MSFT", "2026-06-15", fetcher=fetcher)
    for name in TIMEFRAME_ORDER:
        if early["timeframes"][name]["bar_count"] > late["timeframes"][name]["bar_count"]:
            failures.append(f"{name}: an earlier as_of revealed MORE bars — PIT violation")

    # 8. determinism
    if build_timeframe_snapshot("MSFT", AS_OF, fetcher=fetcher) != late:
        failures.append("the snapshot is not deterministic for identical inputs")

    # provider failure degrades rather than raising. The module logs a warning
    # per timeframe by design; silence it here so a PASSING gate does not print
    # five scary lines about a failure it induced on purpose.
    def failing(ticker, period, interval):
        raise RuntimeError("provider down")

    timeframe_logger = logging.getLogger("core.timeframes")
    previous_level = timeframe_logger.level
    timeframe_logger.setLevel(logging.ERROR)
    try:
        unavailable = build_timeframe_snapshot("MSFT", AS_OF, fetcher=failing)
    finally:
        timeframe_logger.setLevel(previous_level)
    if unavailable["status"] != STATUS_UNAVAILABLE:
        failures.append("a total provider failure must read UNAVAILABLE")

    if failures:
        print("C1 multi-timeframe gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C1 multi-timeframe gate OK:")
    print(f"  all {len(TIMEFRAME_ORDER)} timeframes answer for one as_of: {', '.join(TIMEFRAME_ORDER)}.")
    print("  still-forming coarse bars excluded and counted; no future Close leaks in.")
    print(f"  below-minimum timeframes are INCOMPLETE with no trend (mins: {TIMEFRAME_MIN_BARS}).")
    print("  snapshot status is the worst clock; yearly resampled, not re-fetched.")
    print("  PIT monotonic across as_of; build deterministic; failures degrade, never raise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
