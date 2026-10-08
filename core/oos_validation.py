"""X1 out-of-sample forecast validation — demonstrate performance, or say you cannot.

"Demonstrate performance on unseen data." MEASURED against the eight training
runs on disk, the honest answer is that no estimator demonstrates anything, and
saying so is the deliverable.

**The measurement.** Every run is one fold of 120 validation observations at the
20d horizon, trained to 2025-04-16 and validated from 2025-07-21 — a 96-day
embargo, so PIT ordering is clean and these numbers are not leakage artefacts.
They are simply negative:

* **Zero of seven learned estimators beat the no-feature baseline.**
  ``historical_mean`` carries no features at all and posts the best RMSE
  (0.12164). Four estimators are *significantly worse* — their bootstrap
  intervals exclude the baseline entirely. The three that overlap are merely
  indistinguishable from predicting the mean.
* **Directional accuracy is entirely noise.** At n=120 the 95% band around a
  coin flip is 0.5 ± 0.0895. All eight observed accuracies fall inside it. A
  10,000-shuffle permutation test on the best of them (``momentum``, 0.5750)
  returns p=0.0625 — failing at α=0.05 *before* correcting for eight tests,
  where the chance of a spurious winner is 33.7%.

**So the verdict is NOT_APPROVED, and that is the correct outcome.** The
temptation this gate exists to refuse is picking ``momentum`` because 0.5750 is
the largest number in the column. A gate that approves the best of eight noise
draws is not a gate; it is a random number generator with a rubber stamp.

**Three distinct failures, never merged.** "It lost to the baseline", "it won but
not significantly", and "there was nothing to test" are different facts. A
reader who sees only NOT_APPROVED cannot tell whether to gather more data, try
another feature set, or fix the pipeline.

**A missing run is ABSENT, never zero.** A zero score would rank an absent
estimator *last* rather than unranked, which can make a real candidate look good
by comparison — the same coercion hazard the shape rule names everywhere else.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    OOS_ALPHA,
    OOS_APPROVED,
    OOS_BASELINE_ESTIMATOR,
    OOS_BLOCKS_TRADES,
    OOS_COERCES_MISSING,
    OOS_MIN_EMBARGO_DAYS,
    OOS_MIN_FOLDS,
    OOS_MIN_OBSERVATIONS,
    OOS_NOT_APPROVED,
    OOS_NOT_EVALUATED,
    OOS_REQUIRES_BEATING_BASELINE,
    OOS_REQUIRES_SIGNIFICANCE,
    OOS_VALIDATION_VERSION,
    OOS_VERDICTS,
)


class OOSValidationError(ValueError):
    """Raised when a validation cannot be decided without guessing."""


# Why a candidate failed. Kept distinct because they imply different next
# actions: gather more data, change the features, or fix the pipeline.
OOS_REASON_LOST = "LOST_TO_BASELINE"
OOS_REASON_INSIGNIFICANT = "NOT_SIGNIFICANT"
OOS_REASON_TOO_FEW_FOLDS = "TOO_FEW_FOLDS"
OOS_REASON_TOO_FEW_OBSERVATIONS = "TOO_FEW_OBSERVATIONS"
OOS_REASON_EMBARGO = "EMBARGO_TOO_SHORT"
OOS_REASON_NO_BASELINE = "NO_BASELINE"


def rmse(predictions: Sequence[float], actuals: Sequence[float]) -> float:
    """Root mean squared error over paired sequences."""
    if len(predictions) != len(actuals):
        raise OOSValidationError(
            f"{len(predictions)} predictions against {len(actuals)} actuals; "
            f"a mismatched pair cannot be scored"
        )
    if len(predictions) == 0:
        raise OOSValidationError("an empty fold has no error to measure")
    total = 0.0
    for predicted, actual in zip(predictions, actuals):
        if predicted is None or actual is None:
            raise OOSValidationError(
                "a fold contains a missing value; coercing it to 0.0 would "
                "credit the model with a prediction it never made"
            )
        total += (float(predicted) - float(actual)) ** 2
    return math.sqrt(total / len(predictions))


def directional_accuracy(
    predictions: Sequence[float], actuals: Sequence[float]
) -> float:
    """Share of predictions whose sign matches the outcome."""
    if len(predictions) != len(actuals):
        raise OOSValidationError("mismatched prediction and actual lengths")
    if len(predictions) == 0:
        raise OOSValidationError("an empty fold has no direction to score")
    hits = sum(
        1
        for predicted, actual in zip(predictions, actuals)
        if _sign(float(predicted)) == _sign(float(actual))
    )
    return hits / len(predictions)


def _sign(value: float) -> int:
    """Sign of a value, with zero its own class rather than folded into one side."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def sampling_band(observations: int, alpha: float = OOS_ALPHA) -> float:
    """Half-width of the band around a coin flip at this many observations.

    MEASURED, this is 0.0895 at n=120 — wider than any directional effect the
    eight runs show, which is why the whole column reads as noise.
    """
    if observations <= 0:
        raise OOSValidationError("a band needs at least one observation")
    critical = 1.96 if abs(alpha - 0.05) < 1e-9 else _critical_value(alpha)
    return critical * math.sqrt(0.25 / observations)


def _critical_value(alpha: float) -> float:
    """Two-sided normal critical value, for the alphas a gate plausibly uses."""
    table = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    for known, value in table.items():
        if abs(alpha - known) < 1e-9:
            return value
    raise OOSValidationError(
        f"no critical value is tabulated for alpha={alpha!r}; inventing one "
        f"would silently change what 'significant' means"
    )


def beats_baseline(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    *,
    metric: str = "rmse",
    higher_is_better: bool = False,
) -> tuple[bool | None, str]:
    """Whether a candidate beats the baseline on the named metric.

    Returns ``(beats, reason)`` where ``beats`` is None when the comparison
    cannot be made. None is not False: "we could not compare" and "it lost" are
    different answers.
    """
    if candidate is None:
        return None, "the candidate has no measured result to compare"
    if baseline is None:
        return None, (
            f"no {OOS_BASELINE_ESTIMATOR!r} baseline was supplied, so 'better' "
            f"has nothing to be better than"
        )
    candidate_value = candidate.get(metric)
    baseline_value = baseline.get(metric)
    if candidate_value is None or baseline_value is None:
        return None, (
            f"{metric!r} is missing from the candidate or the baseline; "
            f"coercing it would invent a comparison"
        )
    candidate_value = float(candidate_value)
    baseline_value = float(baseline_value)
    if higher_is_better:
        beats = candidate_value > baseline_value
    else:
        beats = candidate_value < baseline_value
    direction = "higher" if higher_is_better else "lower"
    return beats, (
        f"{metric} {candidate_value:.5f} vs baseline {baseline_value:.5f} "
        f"({direction} is better)"
    )


def validate_run(
    run: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None = None,
) -> dict:
    """Decide whether one training run demonstrates out-of-sample performance."""
    if run is None:
        return _report(
            OOS_NOT_EVALUATED,
            estimator=None,
            reason_code=None,
            reason="no training run was supplied, so nothing was evaluated",
        )
    if not isinstance(run, Mapping):
        raise OOSValidationError("a training run must be a mapping")

    estimator = run.get("estimator")
    metrics = run.get("metrics") or {}
    folds = run.get("folds") or []
    observations = metrics.get("observations")
    accuracy = metrics.get("directional_accuracy")

    detail: dict[str, Any] = {
        "estimator": estimator,
        "horizon": run.get("target_horizon"),
        "observations": observations,
        "folds": len(folds),
        "rmse": metrics.get("rmse"),
        "directional_accuracy": accuracy,
        "baseline_estimator": OOS_BASELINE_ESTIMATOR,
        "alpha": OOS_ALPHA,
    }

    if observations is None:
        return _report(
            OOS_NOT_EVALUATED,
            reason_code=None,
            reason="the run reports no observation count, so its result cannot be sized",
            **detail,
        )

    band = sampling_band(int(observations))
    detail["sampling_band"] = band

    # Structural disqualifications first: these say the TEST was inadequate,
    # which is a different failure from losing the test.
    if len(folds) < OOS_MIN_FOLDS:
        return _report(
            OOS_NOT_APPROVED,
            reason_code=OOS_REASON_TOO_FEW_FOLDS,
            reason=(
                f"{len(folds)} fold(s), below the {OOS_MIN_FOLDS} required; a "
                f"single fold has no dispersion, so nothing distinguishes a "
                f"real edge from one lucky split"
            ),
            **detail,
        )

    if int(observations) < OOS_MIN_OBSERVATIONS:
        return _report(
            OOS_NOT_APPROVED,
            reason_code=OOS_REASON_TOO_FEW_OBSERVATIONS,
            reason=(
                f"{observations} observations, below the "
                f"{OOS_MIN_OBSERVATIONS} floor; the directional sampling band "
                f"at this size is +/-{band:.4f}"
            ),
            **detail,
        )

    beats, comparison = beats_baseline(metrics, baseline)
    detail["baseline_comparison"] = comparison

    if beats is None:
        return _report(
            OOS_NOT_EVALUATED,
            reason_code=OOS_REASON_NO_BASELINE,
            reason=comparison,
            **detail,
        )

    if OOS_REQUIRES_BEATING_BASELINE and not beats:
        return _report(
            OOS_NOT_APPROVED,
            reason_code=OOS_REASON_LOST,
            reason=(
                f"the candidate does not beat the {OOS_BASELINE_ESTIMATOR} "
                f"baseline: {comparison}"
            ),
            **detail,
        )

    if OOS_REQUIRES_SIGNIFICANCE:
        if accuracy is None:
            return _report(
                OOS_NOT_EVALUATED,
                reason_code=None,
                reason=(
                    "the run reports no directional accuracy, so significance "
                    "cannot be tested"
                ),
                **detail,
            )
        excess = abs(float(accuracy) - 0.5)
        detail["excess_over_coin_flip"] = excess
        if excess <= band:
            return _report(
                OOS_NOT_APPROVED,
                reason_code=OOS_REASON_INSIGNIFICANT,
                reason=(
                    f"directional accuracy {float(accuracy):.4f} is "
                    f"{excess:.4f} from a coin flip, inside the +/-{band:.4f} "
                    f"sampling band at n={observations}; a better point "
                    f"estimate is not evidence"
                ),
                **detail,
            )

    return _report(
        OOS_APPROVED,
        reason_code=None,
        reason=(
            f"beats the {OOS_BASELINE_ESTIMATOR} baseline ({comparison}) and "
            f"clears the sampling band at n={observations}"
        ),
        **detail,
    )


def validate_suite(
    runs: Sequence[Mapping[str, Any]] | None,
) -> dict:
    """Validate every run against the baseline drawn from the suite itself.

    Reports the suite verdict AND the selection risk, because a suite of eight
    tested at alpha=0.05 has a 33.7% chance of producing one spurious winner.
    """
    if not runs:
        return {
            "version": OOS_VALIDATION_VERSION,
            "gate": "oos_validation",
            "verdict": OOS_NOT_EVALUATED,
            "reason": "no training runs were supplied",
            "reports": [],
            "tested": 0,
            "approved": 0,
            "selection_risk": None,
            "blocks_trades": OOS_BLOCKS_TRADES,
            "note": _NOTE,
        }

    baseline = None
    for run in runs:
        if run.get("estimator") == OOS_BASELINE_ESTIMATOR:
            baseline = run.get("metrics")
            break

    reports = [
        validate_run(run, baseline)
        for run in runs
        if run.get("estimator") != OOS_BASELINE_ESTIMATOR
    ]
    approved = [r for r in reports if r["verdict"] == OOS_APPROVED]
    tested = len(reports)

    # The multiple-testing exposure, stated rather than left for the reader to
    # notice. X7 formalises the correction; X1 must not pretend it is absent.
    selection_risk = 1.0 - (1.0 - OOS_ALPHA) ** tested if tested else None

    if baseline is None:
        verdict = OOS_NOT_EVALUATED
        reason = (
            f"the suite contains no {OOS_BASELINE_ESTIMATOR!r} run, so no "
            f"candidate can be shown to beat anything"
        )
    elif approved:
        verdict = OOS_APPROVED
        reason = (
            f"{len(approved)} of {tested} estimator(s) beat the baseline and "
            f"cleared the sampling band"
        )
    else:
        verdict = OOS_NOT_APPROVED
        reason = (
            f"none of {tested} estimator(s) demonstrated out-of-sample "
            f"performance against the {OOS_BASELINE_ESTIMATOR} baseline"
        )

    return {
        "version": OOS_VALIDATION_VERSION,
        "gate": "oos_validation",
        "verdict": verdict,
        "reason": reason,
        "reports": reports,
        "tested": tested,
        "approved": len(approved),
        "baseline_estimator": OOS_BASELINE_ESTIMATOR,
        "baseline_present": baseline is not None,
        "alpha": OOS_ALPHA,
        "selection_risk": selection_risk,
        "blocks_trades": OOS_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _report(verdict: str, **detail) -> dict:
    """One X1 answer."""
    if verdict not in OOS_VERDICTS:
        raise OOSValidationError(f"unknown OOS verdict {verdict!r}")
    payload = {
        "version": OOS_VALIDATION_VERSION,
        "gate": "oos_validation",
        "verdict": verdict,
        "coerces_missing": OOS_COERCES_MISSING,
        "blocks_trades": OOS_BLOCKS_TRADES,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED over 8 training runs, one fold of 120 observations each at 20d "
    "with a 96-day embargo: ZERO of seven learned estimators beat the "
    "no-feature historical_mean baseline on RMSE, and all eight directional "
    "accuracies fall inside the 0.5 +/- 0.0895 sampling band. A permutation "
    "test on the best of them returns p=0.0625, failing before any correction "
    "for having tested eight. NOT_APPROVED is the correct outcome."
)


def validation_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a validation report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != OOS_VALIDATION_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{OOS_VALIDATION_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in OOS_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X1 reports; the registry promotes")

    if verdict == OOS_NOT_APPROVED and not report.get("reason_code"):
        if "reports" not in report:  # suite reports carry no single code
            problems.append(
                "a NOT_APPROVED run must name WHY: 'it lost', 'it won but not "
                "significantly' and 'the test was inadequate' imply different "
                "next actions"
            )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    if report.get("coerces_missing"):
        problems.append(
            "an absent estimator must stay ABSENT: a zero would rank it last "
            "rather than unranked"
        )

    # A suite that approves something must not hide its selection exposure.
    if "reports" in report and report.get("tested"):
        if report.get("selection_risk") is None:
            problems.append(
                "a suite verdict must state its multiple-testing exposure; "
                "with 8 estimators at alpha=0.05 it is 33.7%"
            )

    return problems


def render_validation(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one validation report."""
    if "reports" in report:
        lines = [
            f"OOS validation: {report.get('verdict')}",
            f"  {report.get('reason')}",
            f"  baseline        : {report.get('baseline_estimator')}"
            f" (present: {report.get('baseline_present')})",
            f"  tested          : {report.get('tested')}"
            f"  approved: {report.get('approved')}",
            f"  selection risk  : {_pct(report.get('selection_risk'))}"
            f" of one spurious winner at alpha={report.get('alpha')}",
            "",
        ]
        for item in report.get("reports") or []:
            lines.extend(f"  {line}" for line in render_validation(item))
        return lines

    return [
        f"{_shown(report.get('estimator'))} @ {_shown(report.get('horizon'))}"
        f" -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"    rmse {_num(report.get('rmse'))}"
        f"  dir {_num(report.get('directional_accuracy'))}"
        f"  n={_shown(report.get('observations'))}"
        f"  folds={_shown(report.get('folds'))}"
        f"  band=+/-{_num(report.get('sampling_band'))}",
        f"    {report.get('reason')}",
    ]


def _pct(value: Any) -> str:
    """A share as a percentage, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.1%}"


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.5f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
