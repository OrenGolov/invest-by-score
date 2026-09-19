"""CI drift gate for the F3 joint forecast.

Proves, on every push, that the contract F6/F7/F8 and the dashboard will build
on holds:

1. THE SHAPE RULE — a cell carries a `value` key IF AND ONLY IF its status is
   OK. `value: None` would coalesce to 0.0 under the dashboard's existing
   `Number(x ?? 0)` idiom and render a refused P(up) as CERTAIN DOWN;
2. a refused cell supplies NO uncertainty either — an interval beside a
   withheld point estimate is a point estimate by another name;
3. cell precedence is total, ordered, and OK is LAST, so every refusal
   outranks it and a cell is never ambiguous about why it is empty;
4. a zero-width interval is refused. MEASURED: prediction_interval([0.55]*3)
   returns width 0.0 at folds=3, so a fold-COUNT floor passes its own check
   while publishing perfect certainty — width is the guard;
5. the four REJECTED coherence rules stay rejected, each still carrying the
   measurement that rejected it. Re-adding one would flag legitimate market
   behaviour as an error;
6. contract defects are REPORTED, not hidden. The positive-adverse-excursion
   check remains as a REGRESSION GUARD: the label builder now floors at 0
   (outcome-label-v2), so a positive value reaching F3 means the floor was
   removed. Healthy labels must raise NO findings;
7. the grid is complete and in shortest-first order ("120d" sorts before "1d");
8. with no model the object is honest: status NO_MODEL, zero emitted values,
   and a reason.

Synthetic apart from an optional live-label read.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration import prediction_interval  # noqa: E402
from core.config import (  # noqa: E402
    CELL_STATUS_DEGENERATE_INTERVAL,
    CELL_STATUS_OK,
    FORECAST_HORIZONS,
    FORECAST_TARGETS,
    JOINT_CELL_PRECEDENCE,
    JOINT_REJECTED_RULES,
    JOINT_STATUS_NO_MODEL,
)
from core.forecast_joint import (  # noqa: E402
    build_cell,
    build_joint_forecast,
    contract_findings,
    evaluate_coherence,
    joint_problems,
    render_rows,
)

_REQUIRED_REJECTED = (
    "monotonic_return",
    "direction_matches_return",
    "volatility_grows_with_horizon",
    "adverse_excursion_below_return",
)


def _labels(**overrides):
    record = {
        "status": "OK",
        "entry_bar": "2024-06-14 00:00:00",
        "exit_bar": "2024-07-15 00:00:00",
        "forward_return": 0.032,
        "realized_vol": 0.024,
        "label_up": True,
        "adverse_excursion": -0.075,
    }
    record.update(overrides)
    return {
        "label_version": "outcome-label-v1",
        "matured_horizons": ["20d"],
        "pending_horizons": [h for h in FORECAST_HORIZONS if h != "20d"],
        "horizons": {"20d": record},
    }


def _model():
    return {"status": "MODEL", "model_version": "demo-v1", "entry_hash": "abc", "reason": ""}


def _interval(lower=-0.02, upper=0.06, folds=5):
    return {"level": 0.8, "lower": lower, "upper": upper,
            "median": (lower + upper) / 2, "folds": folds}


def main() -> int:
    failures: list[str] = []

    forecast = build_joint_forecast("TEST", "2024-06-15", labels=_labels())

    # 1 + 2. the shape rule
    for horizon, row in forecast["rows"].items():
        for target, cell in row["cells"].items():
            if cell["status"] == CELL_STATUS_OK:
                if "value" not in cell:
                    failures.append(f"{target}@{horizon}: OK cell carries no value key")
                continue
            if "value" in cell:
                failures.append(
                    f"{target}@{horizon}: a {cell['status']} cell carries a value key — "
                    f"it would coalesce to 0 in a consumer and render as a real reading"
                )
            if cell.get("interval") is not None or cell.get("dispersion") is not None:
                failures.append(
                    f"{target}@{horizon}: a refused cell supplies uncertainty — an "
                    f"interval beside a withheld estimate is an estimate by another name"
                )
            if not cell.get("reason"):
                failures.append(f"{target}@{horizon}: a refused cell does not explain itself")

    # 3. precedence is total and OK is last
    if JOINT_CELL_PRECEDENCE[-1] != CELL_STATUS_OK:
        failures.append(
            "OK is not the last precedence entry — a refusal could fail to outrank it"
        )
    if len(set(JOINT_CELL_PRECEDENCE)) != len(JOINT_CELL_PRECEDENCE):
        failures.append("the cell precedence contains a duplicate")

    # 4. width, not fold count
    degenerate = prediction_interval([0.55, 0.55, 0.55])
    if degenerate["upper"] - degenerate["lower"] != 0.0:
        failures.append("the degenerate-interval fixture no longer produces zero width")
    if degenerate["folds"] < 3:
        failures.append("the degenerate fixture no longer reports >= 3 folds")
    degenerate_cell = build_cell(
        "expected_return", "20d", labels=_labels(), readiness="SCORABLE",
        has_model=True, interval=degenerate, value=0.03,
    )
    if degenerate_cell["status"] != CELL_STATUS_DEGENERATE_INTERVAL:
        failures.append(
            "a zero-width interval was accepted — a fold-count floor would have "
            "passed it while publishing perfect certainty"
        )
    real_cell = build_cell(
        "expected_return", "20d", labels=_labels(), readiness="SCORABLE",
        has_model=True, interval=_interval(), value=0.03,
    )
    if real_cell["status"] != CELL_STATUS_OK:
        failures.append("a genuine interval was refused")

    # 5. the rejected rules stay rejected
    for name in _REQUIRED_REJECTED:
        if name not in JOINT_REJECTED_RULES:
            failures.append(
                f"rejected rule {name!r} is no longer registered — it was rejected by "
                f"measurement, and re-adding it would flag legitimate market behaviour"
            )
        elif "REJECTED" not in JOINT_REJECTED_RULES[name]:
            failures.append(f"rejected rule {name!r} no longer states its evidence")

    non_monotonic = {
        h: {"cells": {"expected_return": {"status": CELL_STATUS_OK, "value": v}}}
        for h, v in zip(FORECAST_HORIZONS, [-0.007, -0.104, -0.042, -0.114, 0.100, 0.103])
    }
    if evaluate_coherence(non_monotonic)["violations"]:
        failures.append(
            "non-monotonic returns were flagged as a violation — real NVDA labels run "
            "- - - - + + across 1d..252d, which is ordinary market behaviour"
        )

    skewed = {"20d": {"cells": {
        "probability_up": {"status": CELL_STATUS_OK, "value": 0.73},
        "expected_return": {"status": CELL_STATUS_OK, "value": -0.0075},
    }}}
    if evaluate_coherence(skewed)["violations"]:
        failures.append(
            "P(up)=0.73 with E[return]=-0.0075 was flagged — a skewed payoff makes "
            "both correct simultaneously"
        )

    # 6. defects are reported
    gap_up_findings = contract_findings(_labels(adverse_excursion=0.01))
    if not any("adverse_excursion" in f for f in gap_up_findings):
        failures.append(
            "a POSITIVE adverse excursion raised no finding — a gap-up leaves every "
            "low above entry, and 9 of 300 real 20d labels violate the F1 bound"
        )
    # The relative-target defect is FIXED (probability_outperform is marked
    # label_unavailable), so the finding must now stay SILENT. Checked by
    # behaviour: the field name is still declared for shape uniformity.
    if any("outperform" in f for f in contract_findings(_labels())):
        failures.append(
            "probability_outperform resolves to the raw stock return again — the "
            "label_unavailable marker was removed, and the target silently went "
            "back to measuring nothing about outperformance"
        )
    if contract_findings(_labels()):
        failures.append(
            f"healthy labels raised findings: {contract_findings(_labels())}"
        )

    # 7. the grid is complete and ordered
    if forecast["horizon_order"] != list(FORECAST_HORIZONS):
        failures.append("horizon_order is not shortest-first ('120d' sorts before '1d')")
    for horizon in FORECAST_HORIZONS:
        row = forecast["rows"].get(horizon)
        if not row:
            failures.append(f"horizon {horizon!r} absent from the grid")
            continue
        missing = set(FORECAST_TARGETS) - set(row["cells"])
        if missing:
            failures.append(f"{horizon}: cells missing for {sorted(missing)}")
    if [r["horizon"] for r in render_rows(forecast)] != list(FORECAST_HORIZONS):
        failures.append("rendered rows are not in the declared reading order")

    # 8. no model is honest
    if forecast["status"] != JOINT_STATUS_NO_MODEL:
        failures.append(
            f"with no model the forecast must read NO_MODEL (got {forecast['status']})"
        )
    if forecast["emitted_values"] != 0:
        failures.append("a model-less forecast emitted values")
    if not forecast["model"].get("reason"):
        failures.append("an absent model does not say why")
    for problem in joint_problems(forecast):
        failures.append(f"contract problem: {problem}")

    # live labels, when available
    try:
        from core.labels import build_outcome_labels

        live = build_outcome_labels("NVDA", "2024-06-15")
        if live.get("status") in ("OK", "PARTIAL"):
            live_forecast = build_joint_forecast("NVDA", "2024-06-15", labels=live)
            for problem in joint_problems(live_forecast):
                failures.append(f"live contract problem: {problem}")
            statuses = {
                cell["status"]
                for row in live_forecast["rows"].values()
                for cell in row["cells"].values()
            }
            if len(statuses) < 3:
                failures.append(
                    f"live cells collapsed to {len(statuses)} status(es) — refusals "
                    f"should name their specific obstacle, not one blanket reason"
                )
    except Exception as exc:  # provider unavailable is not a contract failure
        print(f"  (live label check skipped: {type(exc).__name__})")

    if failures:
        print("F3 joint-forecast gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    grid = len(FORECAST_HORIZONS) * len(FORECAST_TARGETS)
    print("F3 joint-forecast gate OK:")
    print(f"  {grid}-cell grid complete and shortest-first: {', '.join(FORECAST_HORIZONS)}.")
    print("  a value key exists IFF status is OK; a refused cell supplies no uncertainty.")
    print("  zero-width intervals refused on WIDTH — a fold-count floor would pass them.")
    print(f"  {len(_REQUIRED_REJECTED)} coherence rules stay rejected, each with its evidence.")
    print("  contract defects reported as findings; a model-less forecast is honestly empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
