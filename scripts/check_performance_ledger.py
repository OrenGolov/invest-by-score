"""CI drift gate for the L2 performance ledger.

Three properties carry this sprint:

  - the point-in-time dimensions are CAPTURED, never recomputed;
  - breakdowns are MARGINAL, never a cross-tab;
  - a thin cell publishes its count and no metric.

1.  the four perishable dimensions are read off the row, not recomputed.
    MEASURED, 7 of 8 retrieval probes returned a different analog count as
    the store grew, so `source` cannot be reconstructed later; a recomputed
    regime depends on whichever classifier version runs at closing time;
2.  a captured dimension that was never written reads `unknown`, not a
    plausible default — inventing a level would attribute performance to a
    regime nobody forecast under;
3.  BREAKDOWNS ARE MARGINAL. MEASURED: nine dimensions cross-multiply to
    4,065,600 cells against 77,616 forecasts a year — 0.0191 per cell, so a
    cross-tab is empty almost everywhere while appearing thorough;
4.  A THIN CELL PUBLISHES NO METRIC, only its count. A performance number
    from three forecasts is the F4 stress cell in analytics clothing;
5.  ...and the floor is F4's, not a second sample-size policy;
6.  coverage travels with every per-cell error, for the same measured reason
    it travels with the global one;
7.  the shape rule holds: a metric key exists only where it was computed, or
    `mean_brier: None` renders a cell that scored nothing as perfect;
8.  every declared dimension is reported, in order;
9.  a sufficient cell still publishes, so the floor is not a blanket refusal.

Synthetic and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    CLOSURE_CLOSED,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    PERF_EVENT_TYPE,
    PERF_REGIME,
    PERF_SOURCE,
    PERF_VOLATILITY_REGIME,
    PERFORMANCE_CAPTURED_AT_RECORD,
    PERFORMANCE_CROSS_TABULATE,
    PERFORMANCE_DIMENSIONS,
    PERFORMANCE_MIN_CELL,
    PERFORMANCE_SPARSITY_EVIDENCE,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_NOT_SCORED,
)
from core.performance_ledger import (  # noqa: E402
    UNKNOWN,
    PerformanceLedgerError,
    breakdown,
    cell_metrics,
    dimension_value,
    performance_problems,
    performance_report,
    render_breakdown,
)

_METRICS = ("mean_brier", "coverage_rate", "direction_accuracy")


def _scored(predicted, actual, **extra):
    row = {
        "state": CLOSURE_CLOSED, "ticker": "NVDA", "horizon": "20d",
        "regime": "bullish", "event_type": "earnings",
        "observed_share": 1.0, "volatility": 0.22, "confidence": 0.8,
        "forecast_version": {"snapshot": "forecast-snapshot-v1"},
        "score": {
            "method": SCORE_METHOD_BRIER, "scored": True,
            "predicted": float(predicted), "actual": float(actual),
            "brier": float((predicted - actual) ** 2),
        },
    }
    row.update(extra)
    return row


def _refused(**extra):
    row = _scored(0.5, 1.0)
    row["score"] = {
        "method": SCORE_METHOD_NOT_SCORED, "scored": False, "reason": "no claim",
    }
    row.update(extra)
    return row


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    for dimension in (PERF_REGIME, PERF_EVENT_TYPE, PERF_SOURCE,
                      PERF_VOLATILITY_REGIME):
        if dimension not in PERFORMANCE_CAPTURED_AT_RECORD:
            failures.append(
                f"{dimension!r} is no longer captured at record time — it is a "
                f"point-in-time fact, and MEASURED, 7 of 8 retrieval probes "
                f"returned a different analog set as the store grew"
            )

    # A captured dimension is READ, not recomputed. Planting a regime that
    # disagrees with the ticker's real one proves the row wins.
    planted = _scored(0.6, 1.0, regime="stress")
    if dimension_value(planted, PERF_REGIME) != "stress":
        failures.append(
            "the regime was not read from the row — recomputing it later "
            "would re-attribute past performance to whichever classifier "
            "version runs at closing time"
        )
    if dimension_value(_scored(0.6, 1.0, event_type="m_and_a"), PERF_EVENT_TYPE) != "m_and_a":
        failures.append("the event type was not read from the row")

    # ---------------------------------------------------------------- 2
    bare = {"state": CLOSURE_CLOSED, "ticker": "NVDA", "score": {"scored": False}}
    for dimension in PERFORMANCE_CAPTURED_AT_RECORD:
        if dimension_value(bare, dimension) != UNKNOWN:
            failures.append(
                f"{dimension!r} invented a level for a row that never recorded "
                f"it — that attributes performance to a condition nobody "
                f"forecast under"
            )

    # ---------------------------------------------------------------- 3
    if PERFORMANCE_CROSS_TABULATE:
        failures.append(
            "the ledger cross-tabulates. " + PERFORMANCE_SPARSITY_EVIDENCE
        )
    if "MEASURED" not in PERFORMANCE_SPARSITY_EVIDENCE:
        failures.append("the sparsity evidence lost its measurement")

    report = performance_report([_scored(0.6, 1.0) for _ in range(20)])
    if report.get("cross_tabulated"):
        failures.append("a built report claims to cross-tabulate")
    for name, table in (report.get("breakdowns") or {}).items():
        if any(isinstance(level, tuple) for level in table.get("levels") or ()):
            failures.append(
                f"{name}: a level is a tuple — that is a cross-tab, and "
                f"MEASURED it would be empty almost everywhere"
            )

    # ---------------------------------------------------------------- 4 + 5
    thin = cell_metrics([_scored(0.95, 0.0) for _ in range(3)])
    if thin.get("sufficient"):
        failures.append(
            f"3 scored forecasts were treated as sufficient (floor "
            f"{PERFORMANCE_MIN_CELL})"
        )
    for metric in _METRICS:
        if metric in thin:
            failures.append(
                f"a thin cell published {metric!r} over 3 forecasts — that is "
                f"the F4 stress cell in analytics clothing"
            )
    if not thin.get("reason"):
        failures.append("a thin cell does not explain itself")
    if thin.get("scored") != 3:
        failures.append("a thin cell does not report how thin it was")

    # The floor must line up with F4's, not be a fresh invention.
    if PERFORMANCE_MIN_CELL != CONDITIONAL_MIN_SAMPLES_INTERVAL:
        failures.append(
            f"the cell floor ({PERFORMANCE_MIN_CELL}) no longer matches F4's "
            f"INTERVAL floor ({CONDITIONAL_MIN_SAMPLES_INTERVAL}) — one "
            f"sample-size policy must govern a forecast and its evaluation"
        )

    # A thin cell with the WORST record must still publish nothing: this is
    # the case where a metric would be most eye-catching and least earned.
    worst = breakdown(
        [_scored(0.6, 1.0) for _ in range(20)]
        + [_scored(0.95, 0.0, ticker="TINY") for _ in range(3)],
        "ticker",
    )
    tiny = (worst.get("cells") or {}).get("TINY") or {}
    if any(metric in tiny for metric in _METRICS):
        failures.append(
            "a 3-forecast ticker with a 0.90 Brier published a metric — the "
            "worst cells are exactly the ones a thin floor must silence"
        )
    if "TINY" not in worst.get("thin_levels", []):
        failures.append("a thin level was not reported as thin")

    # ---------------------------------------------------------------- 9
    healthy = (worst.get("cells") or {}).get("NVDA") or {}
    if not healthy.get("sufficient"):
        failures.append("a 20-forecast cell was treated as thin")
    if "mean_brier" not in healthy:
        failures.append(
            "a sufficient cell published no metric — the floor must be a "
            "threshold, not a blanket refusal"
        )

    # ---------------------------------------------------------------- 6 + 7
    for name, table in (report.get("breakdowns") or {}).items():
        for level, cell in (table.get("cells") or {}).items():
            if any(metric in cell for metric in _METRICS):
                for required in ("scored", "refused", "forecasts", "scored_share"):
                    if cell.get(required) is None:
                        failures.append(
                            f"{name}[{level}]: a metric without {required!r} — "
                            f"refusing remains the cheapest way to look accurate"
                        )
            for metric in _METRICS:
                if metric in cell and cell[metric] is None:
                    failures.append(
                        f"{name}[{level}]: {metric} is present but None — it "
                        f"would coalesce to 0.0 and render a cell that scored "
                        f"nothing as perfectly accurate"
                    )

    # A cell with NO Brier at all: every scored row used a different method.
    # The fixtures above always carry Briers, so the `None` path is never
    # exercised through them — and a metric key present but None is exactly
    # what coalesces to 0.0 and renders an unscored cell as perfect.
    coverage_only = [
        {
            "state": CLOSURE_CLOSED, "ticker": "NVDA", "horizon": "20d",
            "score": {
                "method": "coverage", "scored": True,
                "actual": 1.0, "lower": 0.3, "upper": 1.0, "midpoint": 0.65,
                "brier": 0.1225, "coverage_is_group_property": True,
            },
        }
        for _ in range(PERFORMANCE_MIN_CELL + 2)
    ]
    interval_cell = cell_metrics(coverage_only)
    if interval_cell.get("coverage_rate") is None:
        failures.append("an all-interval cell published no coverage rate")
    # COVERAGE IS COMPUTED ACROSS THE CELL, not per forecast. MEASURED over
    # real data, the per-forecast version was False 42/42 because a binary
    # outcome cannot land inside a probability range.
    if not interval_cell.get("coverage_is_group_property"):
        failures.append(
            "the cell does not declare coverage as a group property — asking "
            "per forecast returns False every time and measures nothing"
        )
    if interval_cell.get("realised_rate") is None:
        failures.append(
            "an all-interval cell publishes no realised rate, so its coverage "
            "claim cannot be checked"
        )
    # A cell whose realised rate sits OUTSIDE its interval must report 0.
    outside = cell_metrics([
        {
            "state": CLOSURE_CLOSED, "ticker": "NVDA", "horizon": "20d",
            "score": {
                "method": "coverage", "scored": True, "actual": 0.0,
                "lower": 0.6, "upper": 0.9, "midpoint": 0.75,
                "brier": 0.5625, "coverage_is_group_property": True,
            },
        }
        for _ in range(PERFORMANCE_MIN_CELL + 2)
    ])
    if outside.get("coverage_rate") != 0.0:
        failures.append(
            f"a cell whose realised rate (0.0) sits outside its published "
            f"interval [0.6, 0.9] reported coverage "
            f"{outside.get('coverage_rate')} — coverage must be able to FAIL"
        )

    mixed = cell_metrics(
        [_scored(0.6, 1.0) for _ in range(10)] + [_refused() for _ in range(5)]
    )
    if mixed.get("refused") != 5:
        failures.append("refusals were not counted inside a cell")
    if mixed.get("scored_share") is None:
        failures.append("a cell publishes no scored share")

    # ---------------------------------------------------------------- 8
    if list(report.get("breakdowns") or {}) != list(PERFORMANCE_DIMENSIONS):
        failures.append(
            "not every declared dimension is reported in order — a missing "
            "breakdown reads as an oversight where an empty one reads as a fact"
        )
    if len(PERFORMANCE_DIMENSIONS) != 9:
        failures.append(
            f"the ledger now measures {len(PERFORMANCE_DIMENSIONS)} dimensions, "
            f"not the nine the sprint names"
        )
    try:
        breakdown([], "not_a_dimension")
        failures.append("an unknown dimension was accepted")
    except PerformanceLedgerError:
        pass

    for problem in performance_problems(report):
        failures.append(f"contract problem: {problem}")
    empty = performance_report([])
    for problem in performance_problems(empty):
        failures.append(f"empty report problem: {problem}")

    for row in render_breakdown(worst):
        if not row["sufficient"] and not row["reason"]:
            failures.append(f"{row['level']}: renders blank for a reader")

    if failures:
        print("L2 performance-ledger gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("L2 performance-ledger gate OK:")
    print(
        f"  {len(PERFORMANCE_DIMENSIONS)} dimensions, sliced MARGINALLY; a "
        f"cross-tab would be 4,065,600 cells against 77,616 forecasts a year."
    )
    print(
        f"  {len(PERFORMANCE_CAPTURED_AT_RECORD)} point-in-time dimensions are "
        f"read off the row, never recomputed."
    )
    print(
        f"  a cell below {PERFORMANCE_MIN_CELL} scored forecasts publishes its "
        f"count and no metric — F4's floor, not a second policy."
    )
    print("  coverage travels with every per-cell error; unknown is a real level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
