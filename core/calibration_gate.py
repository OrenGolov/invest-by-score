"""X2 calibration gate — an in-sample ECE of zero is not calibration.

"Probabilities must be demonstrably calibrated." MEASURED, the number this
system currently reports is 0.0000 for every estimator, and that number is
worthless — not optimistic, but *structurally incapable* of being anything else.

**The deciding measurement.** Fitting the isotonic map on 120 observations and
scoring it on the same 120 gives a perfect ECE every time. Refitting on the
first 60 and scoring the held-out 60 gives the honest number:

===================  =============  ===========  ===========
estimator            ECE in-sample  ECE holdout  MCE holdout
===================  =============  ===========  ===========
elastic_net                 0.0000       0.1894       0.1920
ridge                       0.0000       0.1689       0.1689
logistic                    0.0000       0.1669       0.1669
momentum                    0.0000       0.1669       0.1669
historical_mean             0.0000       0.1667       0.1667
gradient_boosting           0.0000       0.1660       0.1660
===================  =============  ===========  ===========

**Why in-sample ECE is exactly zero, not merely small.** Isotonic regression on
this data collapses to *two knots*. Every observation is mapped to the base rate
of its own group, so each reliability bin reproduces its own observed frequency
by construction. An in-sample ECE cannot detect miscalibration; it can only
report the arithmetic identity it was built from.

**What the 0.1667 actually is** — and it is not a fitting artefact — is
**base-rate drift**. The training half is 63.3% up-days and the holdout half
46.7%. The map learned a base rate that had already changed by the time it was
applied. That is a live point-in-time failure, exactly what a calibration gate
exists to catch, and the system currently reports it as perfect.

**And it is real, not sampling noise.** A perfectly calibrated constant
predictor at n=60 posts a median ECE of 0.0500 and a p95 of 0.1167–0.1333. The
observed 0.1667 sits above that.

**Below the noise floor is not a pass.** An ECE under the floor means the
holdout is too small to tell, not that the model is calibrated — which is why
the gate demands 200 observations, the first size whose p95 floor (0.0700) falls
under the 0.10 bar.

**MCE is carried alongside ECE because ECE is count-weighted.** A badly wrong
region holding few observations hides inside an acceptable average; the worst
single bin cannot hide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.calibration import (
    CalibrationError,
    expected_calibration_error,
    max_calibration_error,
    reliability_curve,
)
from core.config import (
    CALIBRATION_GATE_APPROVED,
    CALIBRATION_GATE_BLOCKS_TRADES,
    CALIBRATION_GATE_COERCES_MISSING,
    CALIBRATION_GATE_MAX_ECE,
    CALIBRATION_GATE_MAX_MCE,
    CALIBRATION_GATE_MIN_HOLDOUT,
    CALIBRATION_GATE_NOISE_FLOOR_N,
    CALIBRATION_GATE_NOISE_FLOOR_P95,
    CALIBRATION_GATE_NOT_APPROVED,
    CALIBRATION_GATE_NOT_EVALUATED,
    CALIBRATION_GATE_REQUIRES_HOLDOUT,
    CALIBRATION_GATE_VERDICTS,
    CALIBRATION_GATE_VERSION,
)


class CalibrationGateError(ValueError):
    """Raised when calibration cannot be judged without guessing."""


# Why a candidate failed. Distinct because each implies a different next action.
CAL_REASON_IN_SAMPLE = "IN_SAMPLE_ONLY"
CAL_REASON_ECE = "ECE_TOO_HIGH"
CAL_REASON_MCE = "WORST_BIN_TOO_HIGH"
CAL_REASON_THIN_HOLDOUT = "HOLDOUT_TOO_SMALL"
CAL_REASON_NO_PROBABILITIES = "NO_PROBABILITIES"


def is_in_sample(fit_indices: Any, score_indices: Any) -> bool | None:
    """Whether a calibration was scored on data it was fitted on.

    Returns None when the provenance is unknown. None is NOT False: a
    calibration whose origin nobody recorded has not been shown to be
    out-of-sample, and assuming it was is how an ECE of 0.0000 gets believed.
    """
    if fit_indices is None or score_indices is None:
        return None
    fit = set(fit_indices)
    scored = set(score_indices)
    if not scored:
        return None
    return bool(fit & scored)


def noise_floor(observations: int) -> float:
    """The p95 ECE a PERFECTLY calibrated predictor still posts at this size.

    MEASURED by simulation, not derived: 0.1333 at n=60, 0.1000 at n=100,
    0.0800 at n=150, 0.0700 at n=200, 0.0440 at n=500. Scaled from the measured
    n=60 anchor as 1/sqrt(n), which reproduces the measured table to within
    0.01 at every simulated size.
    """
    if observations <= 0:
        raise CalibrationGateError("a noise floor needs at least one observation")
    scale = (CALIBRATION_GATE_NOISE_FLOOR_N / observations) ** 0.5
    return CALIBRATION_GATE_NOISE_FLOOR_P95 * scale


def evaluate_calibration(
    probabilities: Sequence[float] | None,
    actuals: Sequence[float] | None,
    *,
    fit_indices: Any = None,
    score_indices: Any = None,
    estimator: str | None = None,
) -> dict:
    """Judge one set of calibrated probabilities against their outcomes."""
    detail: dict[str, Any] = {
        "estimator": estimator,
        "max_ece": CALIBRATION_GATE_MAX_ECE,
        "max_mce": CALIBRATION_GATE_MAX_MCE,
        "min_holdout": CALIBRATION_GATE_MIN_HOLDOUT,
    }

    if probabilities is None or actuals is None:
        return _report(
            CALIBRATION_GATE_NOT_EVALUATED,
            reason_code=CAL_REASON_NO_PROBABILITIES,
            reason=(
                "no probabilities or outcomes were supplied, so calibration "
                "was not measured; that is not the same as calibrated"
            ),
            **detail,
        )

    if len(probabilities) != len(actuals):
        raise CalibrationGateError(
            f"{len(probabilities)} probabilities against {len(actuals)} "
            f"outcomes; a mismatched pair cannot be scored"
        )

    for value in probabilities:
        if value is None:
            raise CalibrationGateError(
                "a probability is missing; coercing it to 0.5 would turn 'the "
                "model declined to predict' into 'the model predicted a coin "
                "flip', manufacturing calibration evidence from silence"
            )

    observations = len(probabilities)
    detail["observations"] = observations
    detail["floor"] = noise_floor(observations) if observations else None

    # PROVENANCE FIRST. An in-sample ECE is not a weak result; it is not a
    # result. Judging it on its value would rank 0.0000 as the best of all.
    in_sample = is_in_sample(fit_indices, score_indices)
    detail["in_sample"] = in_sample
    if CALIBRATION_GATE_REQUIRES_HOLDOUT and in_sample is not False:
        return _report(
            CALIBRATION_GATE_NOT_EVALUATED,
            reason_code=CAL_REASON_IN_SAMPLE,
            reason=(
                f"provenance is {'IN-SAMPLE' if in_sample else 'UNKNOWN'}; "
                f"MEASURED, an in-sample isotonic ECE is 0.0000 for every "
                f"estimator because each bin reproduces its own base rate by "
                f"construction, so it cannot detect miscalibration at all"
            ),
            **detail,
        )

    try:
        ece = expected_calibration_error(probabilities, actuals)
        mce = max_calibration_error(probabilities, actuals)
        curve = reliability_curve(probabilities, actuals)
    except CalibrationError as exc:
        return _report(
            CALIBRATION_GATE_NOT_EVALUATED,
            reason_code=CAL_REASON_NO_PROBABILITIES,
            reason=f"calibration error could not be computed: {exc}",
            **detail,
        )

    detail["ece"] = ece
    detail["mce"] = mce
    detail["populated_bins"] = len(curve)

    # A holdout too small to decide is NOT_EVALUATED, never a pass. Below the
    # floor means the sample cannot tell, not that the model is calibrated.
    if observations < CALIBRATION_GATE_MIN_HOLDOUT:
        return _report(
            CALIBRATION_GATE_NOT_EVALUATED,
            reason_code=CAL_REASON_THIN_HOLDOUT,
            reason=(
                f"{observations} holdout observations, below the "
                f"{CALIBRATION_GATE_MIN_HOLDOUT} required; at this size a "
                f"perfectly calibrated model still posts an ECE up to "
                f"{detail['floor']:.4f}, so an ECE of {ece:.4f} decides nothing"
            ),
            **detail,
        )

    if ece > CALIBRATION_GATE_MAX_ECE:
        return _report(
            CALIBRATION_GATE_NOT_APPROVED,
            reason_code=CAL_REASON_ECE,
            reason=(
                f"ECE {ece:.4f} exceeds the {CALIBRATION_GATE_MAX_ECE} bar; "
                f"MEASURED, the shipped estimators post 0.166-0.189 out of "
                f"sample from base-rate drift"
            ),
            **detail,
        )

    if mce > CALIBRATION_GATE_MAX_MCE:
        return _report(
            CALIBRATION_GATE_NOT_APPROVED,
            reason_code=CAL_REASON_MCE,
            reason=(
                f"the worst bin is off by {mce:.4f}, above the "
                f"{CALIBRATION_GATE_MAX_MCE} bar; ECE {ece:.4f} passed because "
                f"it is count-weighted and averaged that region away"
            ),
            **detail,
        )

    return _report(
        CALIBRATION_GATE_APPROVED,
        reason_code=None,
        reason=(
            f"ECE {ece:.4f} and worst bin {mce:.4f} clear their bars on "
            f"{observations} out-of-sample observations"
        ),
        **detail,
    )


def _report(verdict: str, **detail) -> dict:
    """One X2 answer."""
    if verdict not in CALIBRATION_GATE_VERDICTS:
        raise CalibrationGateError(f"unknown calibration verdict {verdict!r}")
    payload = {
        "version": CALIBRATION_GATE_VERSION,
        "gate": "calibration",
        "verdict": verdict,
        "requires_holdout": CALIBRATION_GATE_REQUIRES_HOLDOUT,
        "coerces_missing": CALIBRATION_GATE_COERCES_MISSING,
        "blocks_trades": CALIBRATION_GATE_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED: in-sample isotonic ECE is 0.0000 for all 8 estimators because "
    "the map collapses to two knots and each bin reproduces its own base rate. "
    "Refit on the first 60 and scored on the held-out 60, the honest ECE is "
    "0.166-0.189 - BASE-RATE DRIFT, the training half being 63.3% up-days and "
    "the holdout 46.7%. A perfectly calibrated predictor at n=60 posts up to "
    "0.1333, so the observed error is real, and 200 observations are required "
    "before an ECE can decide anything."
)


def calibration_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a calibration report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != CALIBRATION_GATE_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{CALIBRATION_GATE_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in CALIBRATION_GATE_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X2 reports; the registry promotes")

    if report.get("coerces_missing"):
        problems.append(
            "a missing probability must never become 0.5; that manufactures "
            "calibration evidence from silence"
        )

    if not report.get("requires_holdout"):
        problems.append(
            "in-sample calibration is never evidence: an in-sample isotonic "
            "ECE is 0.0000 by construction"
        )

    # THE CENTRAL INVARIANT: an in-sample result may never be APPROVED.
    if verdict == CALIBRATION_GATE_APPROVED and report.get("in_sample") is not False:
        problems.append(
            "a calibration APPROVED without confirmed out-of-sample "
            "provenance; MEASURED, in-sample ECE is 0.0000 for every "
            "estimator and means nothing"
        )

    if verdict == CALIBRATION_GATE_APPROVED:
        ece = report.get("ece")
        mce = report.get("mce")
        if ece is None or float(ece) > CALIBRATION_GATE_MAX_ECE:
            problems.append(
                f"APPROVED with ECE {ece!r}, above the "
                f"{CALIBRATION_GATE_MAX_ECE} bar"
            )
        if mce is None or float(mce) > CALIBRATION_GATE_MAX_MCE:
            problems.append(
                f"APPROVED with worst bin {mce!r}, above the "
                f"{CALIBRATION_GATE_MAX_MCE} bar"
            )
        observations = report.get("observations")
        if observations is None or int(observations) < CALIBRATION_GATE_MIN_HOLDOUT:
            problems.append(
                f"APPROVED on {observations!r} observations, below the "
                f"{CALIBRATION_GATE_MIN_HOLDOUT} needed to distinguish "
                f"calibration from sampling noise"
            )

    if verdict == CALIBRATION_GATE_NOT_APPROVED and not report.get("reason_code"):
        problems.append(
            "a NOT_APPROVED calibration must name WHY: a high average and one "
            "bad region imply different fixes"
        )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_calibration(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one calibration report."""
    return [
        f"Calibration: {_shown(report.get('estimator'))}"
        f" -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"  ECE          : {_num(report.get('ece'))}"
        f"  (bar {report.get('max_ece')})",
        f"  worst bin    : {_num(report.get('mce'))}"
        f"  (bar {report.get('max_mce')})",
        f"  observations : {_shown(report.get('observations'))}"
        f"  (need {report.get('min_holdout')})",
        f"  noise floor  : {_num(report.get('floor'))}"
        f"  - an ECE below this decides nothing",
        f"  out of sample: {_provenance(report.get('in_sample'))}",
        f"  reason       : {report.get('reason')}",
    ]


def _provenance(in_sample: Any) -> str:
    """Provenance shown as three states, never two."""
    if in_sample is False:
        return "YES"
    if in_sample is True:
        return "NO - scored on data it was fitted on"
    return "UNKNOWN - which is not the same as yes"


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.4f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
