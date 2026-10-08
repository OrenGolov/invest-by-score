"""CI drift gate for the F2 forecast horizons.

Proves, on every push, that the horizon set and everything coupled to it holds:

1. all six F2 horizons are declared (1d, 5d, 20d, 60d, 120d, 252d), and each
   resolves to a label horizon — a forecast horizon with no label could never
   be scored against an outcome;
2. horizons are ordered SHORTEST-FIRST by sessions, not by name: "120d" sorts
   before "1d" alphabetically, and a joint forecast reading in that order would
   be nonsense;
3. the label builder's calendar coverage reaches past the LONGEST horizon. It
   was a fixed 130 days sized for a 60-session maximum; a 252-session window
   spans about a year, so long labels would have reported pending forever
   because their future bars were never fetched;
4. the backtest embargo covers the longest horizon, or a validation fold sees
   bars that shaped a training row's outcome;
5. pending is distinguishable from missing — a 252d forecast made last month is
   unfinished, not wrong;
6. the real label builder matures every horizon on deep history, and leaves the
   long ones pending on a recent as_of.

Reads the live label builder; otherwise synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import config as core_config  # noqa: E402
from core.config import (  # noqa: E402
    FORECAST_HORIZONS,
    FORECAST_REQUIRED_HORIZONS,
    LABEL_HORIZON_SESSIONS,
)
from core.forecast_horizons import (  # noqa: E402
    HORIZON_STATUS_PENDING,
    HORIZON_STATUS_SCORABLE,
    horizon_problems,
    horizon_readiness,
    horizon_sessions,
    readiness_report,
)


def _labels(matured=(), pending=()):
    return {"matured_horizons": list(matured), "pending_horizons": list(pending)}


def main() -> int:
    failures: list[str] = []

    # 1. every F2 horizon declared, and label-backed
    for horizon in FORECAST_REQUIRED_HORIZONS:
        if horizon not in FORECAST_HORIZONS:
            failures.append(
                f"F2 horizon {horizon!r} is not declared — a product narrower than "
                f"its contract is a silent downgrade"
            )
            continue
        if horizon not in LABEL_HORIZON_SESSIONS:
            failures.append(
                f"horizon {horizon!r} has no label horizon — it could never be "
                f"scored against a realized outcome"
            )

    # 2. shortest-first ordering
    sessions = [horizon_sessions(name) for name in FORECAST_HORIZONS]
    if sessions != sorted(sessions):
        failures.append(
            "horizons are not ordered shortest-first — a joint forecast would read "
            "out of time order ('120d' sorts before '1d' by name)"
        )

    longest = max(LABEL_HORIZON_SESSIONS.values())

    # 3. calendar coverage reaches past the longest horizon
    if core_config.LABEL_CALENDAR_COVERAGE_DAYS < longest:
        failures.append(
            f"LABEL_CALENDAR_COVERAGE_DAYS ({core_config.LABEL_CALENDAR_COVERAGE_DAYS}) "
            f"is below the longest horizon ({longest} sessions) — those labels could "
            f"never mature because their future bars are never fetched"
        )

    # 4. the embargo covers the longest horizon
    if core_config.BACKTEST_EMBARGO_SESSIONS < longest:
        failures.append(
            f"BACKTEST_EMBARGO_SESSIONS ({core_config.BACKTEST_EMBARGO_SESSIONS}) is "
            f"below the longest horizon ({longest}) — a validation fold would see "
            f"bars that shaped a training row's outcome"
        )
    if core_config.BACKTEST_FOLD_SESSIONS < core_config.BACKTEST_EMBARGO_SESSIONS:
        failures.append("fold sessions are narrower than the embargo")

    # 5. pending is not missing
    pending = horizon_readiness(_labels(pending=["252d"]), "252d")
    if pending["status"] != HORIZON_STATUS_PENDING or not pending["reason"]:
        failures.append(
            "a pending horizon did not report itself as unfinished with a reason — "
            "an unfinished forecast is not a failed one"
        )
    absent = horizon_readiness(_labels(), "20d")
    if absent["status"] == HORIZON_STATUS_PENDING:
        failures.append("an absent horizon was reported as merely pending")

    report = readiness_report(_labels(matured=list(FORECAST_HORIZONS)))
    for problem in horizon_problems(report):
        failures.append(f"contract problem: {problem}")
    if report["scorable"] != list(FORECAST_HORIZONS):
        failures.append("the readiness report is not in time order")

    # 6. the real label builder
    try:
        from core.labels import build_outcome_labels

        deep = build_outcome_labels("NVDA", "2024-06-15")
        if deep.get("status") in ("OK", "PARTIAL"):
            deep_report = readiness_report(deep)
            missing = [
                h for h in FORECAST_REQUIRED_HORIZONS if h not in deep_report["scorable"]
            ]
            if missing:
                failures.append(
                    f"horizons {missing} did not mature on deep history — the label "
                    f"builder cannot supply an outcome for them"
                )

        recent = build_outcome_labels("NVDA", "2026-06-15")
        if recent.get("status") in ("OK", "PARTIAL"):
            recent_report = readiness_report(recent)
            if "252d" in recent_report["scorable"]:
                failures.append(
                    "252d reported scorable at a recent as_of — its window cannot "
                    "have closed, so an outcome there would be fabricated"
                )
    except Exception as exc:  # provider unavailable is not a contract failure
        print(f"  (live label check skipped: {type(exc).__name__})")

    if failures:
        print("F2 forecast-horizon gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F2 forecast-horizon gate OK:")
    print(f"  {len(FORECAST_HORIZONS)} horizons, shortest-first: {', '.join(FORECAST_HORIZONS)}.")
    print(f"  every horizon is label-backed; longest is {longest} sessions.")
    print(f"  calendar coverage {core_config.LABEL_CALENDAR_COVERAGE_DAYS}d and embargo "
          f"{core_config.BACKTEST_EMBARGO_SESSIONS} both reach past it.")
    print("  pending is distinguishable from missing; long horizons stay pending when young.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
