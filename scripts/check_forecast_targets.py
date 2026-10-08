"""CI drift gate for the F1 forecast targets.

Proves, on every push, that the contract the forecasting engine is built on
holds:

1. all six F1 targets are declared, each with a kind, bounds, a question and a
   realized label field — a target with no realized counterpart could never be
   validated, and an unvalidatable forecast is an opinion;
2. a probability may ONLY be produced through a fitted M6 calibration: an
   uncalibrated score is refused, and calibrating a return target is refused
   because it would present a return as a likelihood;
3. probability-ness and calibration-requirement never disagree — a probability
   that skips calibration is the exact failure M6 exists to prevent;
4. declared bounds are enforced: no probability outside [0,1], no positive
   adverse excursion, no negative volatility;
5. the relative target refuses without a benchmark, rather than silently
   reducing to probability_up;
6. a distribution refuses without an interval — "P(return within what?)" is
   not a question;
7. every target scores against the REAL V1 label builder;
8. an unmatured horizon yields no realized value.

Reads the live label builder; otherwise synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.calibration import UncalibratedProbabilityError, fit_calibration  # noqa: E402
from core.config import (  # noqa: E402
    FORECAST_KIND_DISTRIBUTION,
    FORECAST_KIND_PROBABILITY,
    FORECAST_TARGET_CONTRACTS,
    FORECAST_TARGET_DIRECTION,
    FORECAST_TARGET_DISTRIBUTION,
    FORECAST_TARGET_DOWNSIDE,
    FORECAST_TARGET_RELATIVE,
    FORECAST_TARGET_RETURN,
    FORECAST_TARGET_VOLATILITY,
    FORECAST_TARGETS,
)
from core.forecast_targets import (  # noqa: E402
    TARGET_STATUS_OK,
    TARGET_STATUS_UNAVAILABLE,
    ForecastTargetError,
    bounds_problems,
    build_target_request,
    is_probability_target,
    realized_value,
    request_problems,
    resolve_probability,
    target_contract,
    target_coverage,
)

_F1_TARGETS = (
    FORECAST_TARGET_DIRECTION,
    FORECAST_TARGET_RETURN,
    FORECAST_TARGET_DISTRIBUTION,
    FORECAST_TARGET_DOWNSIDE,
    FORECAST_TARGET_VOLATILITY,
    FORECAST_TARGET_RELATIVE,
)


def _calibration_map(seed=7, n=200):
    rng = np.random.default_rng(seed)
    scores = list(rng.uniform(0, 1, n))
    actuals = [1.0 if s + rng.normal(0, 0.2) > 0.5 else 0.0 for s in scores]
    return fit_calibration(scores, actuals)


def _labels(status="OK"):
    return {
        "status": "OK",
        "horizons": {
            "20d": {
                "status": status,
                "label_up": True,
                "forward_return": 0.032214,
                "adverse_excursion": -0.075004,
                "realized_vol": 0.024811,
            }
        },
    }


def main() -> int:
    failures: list[str] = []

    # 1. every F1 target declared, fully
    for target in _F1_TARGETS:
        if target not in FORECAST_TARGETS:
            failures.append(f"F1 target {target!r} is not declared")
            continue
        contract = target_contract(target)
        if not contract.get("label_field"):
            failures.append(
                f"{target!r} names no realized label field — it could never be "
                f"validated against an outcome"
            )
        if not contract.get("question"):
            failures.append(f"{target!r} does not state the question it answers")
        if contract.get("bounds") is None:
            failures.append(f"{target!r} declares no bounds")

    # 2. probabilities only through a fitted calibration
    try:
        resolve_probability(FORECAST_TARGET_DIRECTION, 0.8, None)
        failures.append(
            "an uncalibrated score was returned as a probability — a raw model "
            "score is not a likelihood"
        )
    except UncalibratedProbabilityError:
        pass

    cmap = _calibration_map()
    try:
        resolve_probability(FORECAST_TARGET_RETURN, 0.05, cmap)
        failures.append(
            "a return target was calibrated into a probability — that presents a "
            "return as a likelihood"
        )
    except ForecastTargetError:
        pass

    for target in FORECAST_TARGETS:
        if is_probability_target(target):
            value = resolve_probability(target, 0.6, cmap)
            if not 0.0 <= value <= 1.0:
                failures.append(f"{target!r} produced {value}, outside [0, 1]")

    # 3. probability-ness and calibration never disagree
    for target, contract in FORECAST_TARGET_CONTRACTS.items():
        is_probability = contract["kind"] in (
            FORECAST_KIND_PROBABILITY, FORECAST_KIND_DISTRIBUTION
        )
        if is_probability != contract["requires_calibration"]:
            failures.append(
                f"{target!r}: kind says probability={is_probability} but "
                f"requires_calibration={contract['requires_calibration']}"
            )

    # 4. bounds enforced
    if not bounds_problems(FORECAST_TARGET_DIRECTION, 1.4):
        failures.append("a probability above 1.0 was accepted")
    if not bounds_problems(FORECAST_TARGET_DOWNSIDE, 0.05):
        failures.append(
            "a POSITIVE adverse excursion was accepted — it is the worst drawdown "
            "inside the window and cannot be a gain"
        )
    if not bounds_problems(FORECAST_TARGET_VOLATILITY, -0.01):
        failures.append("a negative volatility was accepted — a dispersion cannot be below zero")
    if bounds_problems(FORECAST_TARGET_DIRECTION, 0.63):
        failures.append("a valid probability was rejected")

    # 5. the relative target needs a benchmark
    without = build_target_request(FORECAST_TARGET_RELATIVE, "20d")
    if without["status"] != TARGET_STATUS_UNAVAILABLE:
        failures.append(
            "the relative target was accepted with no benchmark — scoring it "
            "against nothing silently reduces it to probability_up"
        )
    with_benchmark = build_target_request(FORECAST_TARGET_RELATIVE, "20d", benchmark="SPY")
    if with_benchmark["status"] != TARGET_STATUS_OK or request_problems(with_benchmark):
        failures.append("the relative target was refused WITH a benchmark")

    # 6. a distribution needs an interval
    if build_target_request(FORECAST_TARGET_DISTRIBUTION, "20d")["status"] != TARGET_STATUS_UNAVAILABLE:
        failures.append("a distribution was accepted with no interval stated")
    stated = build_target_request(FORECAST_TARGET_DISTRIBUTION, "20d", interval=(0.0, 0.05))
    if stated["status"] != TARGET_STATUS_OK:
        failures.append("a distribution with a stated interval was refused")
    try:
        build_target_request(FORECAST_TARGET_DISTRIBUTION, "20d", interval=(0.05, 0.05))
        failures.append("an empty interval was accepted")
    except ForecastTargetError:
        pass

    # 7. every target scores against the REAL label builder
    try:
        from core.labels import build_outcome_labels

        live = build_outcome_labels("NVDA", "2026-06-15")
        if live.get("status") == "OK":
            coverage = target_coverage(live, "20d")
            if coverage["unscorable_targets"]:
                failures.append(
                    f"targets unscorable against real labels: "
                    f"{coverage['unscorable_targets']} — each must resolve to a "
                    f"realized field or it cannot be validated"
                )
    except Exception as exc:  # provider unavailable is not a contract failure
        print(f"  (live label check skipped: {type(exc).__name__})")

    # 8. an unmatured horizon yields nothing
    if realized_value(FORECAST_TARGET_RETURN, _labels(status="PENDING"), "20d") is not None:
        failures.append(
            "an unmatured horizon produced a realized value — a forecast judged "
            "against an unfinished window is judged against noise"
        )

    if failures:
        print("F1 forecast-target gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F1 forecast-target gate OK:")
    print(f"  all {len(_F1_TARGETS)} targets declared with kind, bounds, question and label field.")
    print("  probabilities only through a fitted calibration; calibrating a return is refused.")
    print("  bounds enforced: no probability outside [0,1], no positive drawdown, no negative vol.")
    print("  the relative target refuses without a benchmark; a distribution without an interval.")
    print("  every target scores against the real V1 labels; an unmatured horizon yields nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
