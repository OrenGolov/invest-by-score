"""Forecast targets (Sprint F1) — what may be predicted, and what it means.

F1 defines the six things the forecasting engine can be asked for:

    probability_up          P(return > 0)
    expected_return         E(return)
    return_distribution     P(return within a stated interval)
    adverse_excursion       expected worst drawdown inside the window
    expected_volatility     expected realized volatility
    probability_outperform  P(stock return > benchmark return)

and, for each, a contract: what kind of quantity it is, its bounds, whether it
must pass through a fitted calibration, and the realized label field it is
scored against.

**Every target resolves to a realized label.** The V1 label builder already
produces `label_up`, `forward_return`, `adverse_excursion` and `realized_vol`,
so each forecast can be compared to what actually happened. A target with no
realized counterpart could never be validated, and an unvalidatable forecast is
an opinion — so `target_contract` refuses one.

**A probability may only be emitted through a fitted calibration.** Targets
whose kind is probability or distribution carry `requires_calibration: True`,
and `resolve_probability` routes them through
`core.calibration.calibrated_probability`, which raises without a map. A raw
model score is not a probability; this is where that stops being a comment.

**Bounds are checked, not assumed.** An adverse excursion is the worst drawdown
INSIDE the window, so it is never positive. A volatility is a dispersion, so it
is never negative. A forecast violating its own declared bounds is malformed and
is refused rather than rendered.

**The relative target needs a benchmark.** `probability_outperform` is the one
target that cannot be computed from the stock alone. Without a benchmark it is
UNAVAILABLE — never silently scored against nothing, which would quietly turn it
into `probability_up`.

F1 defines and validates targets. It fits no models and makes no predictions:
those are F2 onward, and they consume this contract.
"""

from __future__ import annotations

import logging

from core.calibration import UncalibratedProbabilityError, calibrated_probability
from core.config import (
    FORECAST_KIND_DISTRIBUTION,
    FORECAST_KIND_PROBABILITY,
    FORECAST_TARGET_CONTRACT_VERSION,
    FORECAST_TARGET_CONTRACTS,
    FORECAST_TARGET_RELATIVE,
    FORECAST_TARGETS,
)

LOGGER = logging.getLogger("core.forecast_targets")

TARGET_STATUS_OK = "OK"
TARGET_STATUS_UNAVAILABLE = "UNAVAILABLE"
TARGET_STATUS_INVALID = "INVALID"

_PROBABILITY_KINDS = (FORECAST_KIND_PROBABILITY, FORECAST_KIND_DISTRIBUTION)


class ForecastTargetError(ValueError):
    """Raised when a forecast-target request violates the F1 contract."""


def known_targets() -> tuple[str, ...]:
    """Every target the engine may be asked for."""
    return tuple(FORECAST_TARGETS)


def target_contract(target: str) -> dict:
    """The full contract for one target.

    Raises on an unknown name rather than returning a default: a typo'd target
    silently falling back to expected_return would mean a model is scored
    against a question nobody asked.
    """
    contract = FORECAST_TARGET_CONTRACTS.get(target)
    if contract is None:
        raise ForecastTargetError(
            f"unknown forecast target {target!r} (known: {sorted(FORECAST_TARGETS)})"
        )
    return dict(contract)


def is_probability_target(target: str) -> bool:
    """Whether this target's value must be a calibrated probability."""
    return target_contract(target)["kind"] in _PROBABILITY_KINDS


def requires_benchmark(target: str) -> bool:
    """Whether this target is uncomputable without a benchmark."""
    return bool(target_contract(target)["requires_benchmark"])


def label_field_for(target: str) -> str:
    """The realized label field this target is scored against."""
    return str(target_contract(target)["label_field"])


def bounds_problems(target: str, value: float | None) -> list[str]:
    """Why a value violates its target's declared bounds, if it does.

    Returns reasons rather than raising, so a caller assembling several targets
    can report every malformed one instead of dying on the first.
    """
    contract = target_contract(target)
    problems: list[str] = []
    if value is None:
        return problems
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return [f"{target}: value {value!r} is not numeric"]
    if numeric != numeric:  # NaN
        return [f"{target}: value is NaN"]

    low, high = contract["bounds"]
    if low is not None and numeric < low:
        problems.append(f"{target}: {numeric} is below the declared minimum {low}")
    if high is not None and numeric > high:
        problems.append(f"{target}: {numeric} is above the declared maximum {high}")
    return problems


def resolve_probability(target: str, score: float, calibration_map=None) -> float:
    """Turn a raw model score into this target's calibrated probability.

    Refuses for a non-probability target — calling this on expected_return
    would dress a return up as a likelihood — and refuses without a fitted map,
    because an uncalibrated score presented as a probability is the exact
    failure M6 exists to prevent.
    """
    contract = target_contract(target)
    if contract["kind"] not in _PROBABILITY_KINDS:
        raise ForecastTargetError(
            f"target {target!r} is a {contract['kind']}, not a probability — "
            f"calibrating it would present a {contract['unit']} as a likelihood"
        )
    probability = calibrated_probability(score, calibration_map)
    problems = bounds_problems(target, probability)
    if problems:
        raise ForecastTargetError("; ".join(problems))
    return probability


def realized_value(target: str, labels: dict, horizon: str) -> float | None:
    """What actually happened, for scoring a forecast of this target.

    Reads the V1 label builder's output. A horizon that has not matured yields
    None rather than a partial outcome, because a forecast judged against an
    unfinished window is judged against noise.
    """
    contract = target_contract(target)
    horizons = (labels or {}).get("horizons") or {}
    entry = horizons.get(horizon)
    if not entry or entry.get("status") != "OK":
        return None
    value = entry.get(contract["label_field"])
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_target_request(
    target: str,
    horizon: str,
    benchmark: str | None = None,
    interval: tuple[float, float] | None = None,
) -> dict:
    """A validated request for one target at one horizon.

    This is what F2+ hands to a model: it fixes the question before any value
    exists, so a prediction can never be re-labelled as an answer to something
    easier after the fact.
    """
    contract = target_contract(target)
    if not str(horizon or "").strip():
        raise ForecastTargetError("a horizon is required")

    if contract["requires_benchmark"] and not benchmark:
        return {
            "target": target,
            "horizon": horizon,
            "status": TARGET_STATUS_UNAVAILABLE,
            "reason": (
                f"{target} compares the stock to a benchmark, and none was supplied; "
                f"scoring it against nothing would silently reduce it to probability_up"
            ),
            "contract_version": FORECAST_TARGET_CONTRACT_VERSION,
        }

    if contract["kind"] == FORECAST_KIND_DISTRIBUTION:
        if interval is None:
            return {
                "target": target,
                "horizon": horizon,
                "status": TARGET_STATUS_UNAVAILABLE,
                "reason": (
                    "a return distribution answers P(return within an interval), so "
                    "the interval must be stated; without it there is no question"
                ),
                "contract_version": FORECAST_TARGET_CONTRACT_VERSION,
            }
        low, high = interval
        if low >= high:
            raise ForecastTargetError(
                f"interval ({low}, {high}) is empty — P(return in an empty set) is zero "
                f"by construction and tells a reader nothing"
            )

    return {
        "target": target,
        "horizon": horizon,
        "status": TARGET_STATUS_OK,
        "kind": contract["kind"],
        "unit": contract["unit"],
        "question": contract["question"],
        "requires_calibration": contract["requires_calibration"],
        "label_field": contract["label_field"],
        "benchmark": benchmark,
        "interval": list(interval) if interval else None,
        "reason": "",
        "contract_version": FORECAST_TARGET_CONTRACT_VERSION,
    }


def request_problems(request: dict) -> list[str]:
    """Validate a built request against the F1 contract."""
    problems: list[str] = []
    if not isinstance(request, dict):
        return ["request must be a dict"]
    for field in ("target", "horizon", "status", "contract_version"):
        if not request.get(field):
            problems.append(f"request field {field!r} missing/empty")
    if problems:
        return problems

    target = request["target"]
    if target not in FORECAST_TARGETS:
        problems.append(f"request names unknown target {target!r}")
        return problems

    contract = target_contract(target)
    if request["status"] == TARGET_STATUS_OK:
        if request.get("requires_calibration") != contract["requires_calibration"]:
            problems.append(
                f"{target}: request disagrees with the contract about calibration — "
                f"a probability that skips it is not a probability"
            )
        if contract["requires_benchmark"] and not request.get("benchmark"):
            problems.append(f"{target}: OK request carries no benchmark")
        if contract["kind"] == FORECAST_KIND_DISTRIBUTION and not request.get("interval"):
            problems.append(f"{target}: OK distribution request states no interval")
    elif not request.get("reason"):
        problems.append(f"{target}: a non-OK request must explain itself")
    return problems


def target_coverage(labels: dict, horizon: str) -> dict:
    """Which targets can be scored against a given label set.

    Reported rather than assumed: a horizon that has not matured supports no
    target at all, and saying so is more useful than six Nones.
    """
    covered: dict[str, bool] = {}
    for target in FORECAST_TARGETS:
        covered[target] = realized_value(target, labels, horizon) is not None
    scorable = sorted(name for name, ok in covered.items() if ok)
    return {
        "horizon": horizon,
        "covered": covered,
        "scorable_targets": scorable,
        "unscorable_targets": sorted(set(FORECAST_TARGETS) - set(scorable)),
        "contract_version": FORECAST_TARGET_CONTRACT_VERSION,
    }
