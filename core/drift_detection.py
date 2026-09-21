"""L6 concept drift — four detectors, because there are four questions.

"Feature distribution drift, relationship drift, calibration drift,
event-response drift."

**The task names four things, and they really are four.** MEASURED over four
scenarios each breaking exactly one thing, scored by the feature detector
alone: it catches ONE. The other three are invisible to it because the
features did not move — the world did.

    scenario                 feature PSI   caught?
    1 feature drift               1.2197   YES
    2 relationship drift          0.0360   no
    3 calibration drift           0.0133   no
    4 event-response drift        0.0274   no

**M5's `score_drift_psi` is not reusable here**, and that is a measurement
rather than a preference: it bins on a fixed [0, 10] score scale and refuses
anything outside it. MEASURED on a feature in [-0.3, 0.3] shifted by 3.2
standard deviations, fixed bins score **0.0000** while quantile bins score
6.9450. The PSI formula is shared; the binning is the whole difference between
seeing an enormous drift and reporting zero.

**The window floor is load-bearing.** PSI between two identical distributions
is not zero — it grows as the window shrinks, and the conventional PSI>0.25
rule fires on 77% of clean comparisons at window 50.

**Drift is reported, never acted on.** A model that retrains itself on an
alert is the uncontrolled self-modification Sprint L exists to prevent. Drift
is evidence *for* the L5 chain, not a substitute for it.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from core.config import (
    DRIFT_ALERT,
    DRIFT_CALIBRATION,
    DRIFT_CALIBRATION_GAP,
    DRIFT_CORRELATION_SHIFT,
    DRIFT_DETECTION_VERSION,
    DRIFT_FEATURE,
    DRIFT_MIN_WINDOW,
    DRIFT_NOT_EVALUATED,
    DRIFT_PSI_ALERT,
    DRIFT_PSI_BINS,
    DRIFT_PSI_WARN,
    DRIFT_RELATIONSHIP,
    DRIFT_RESPONSE,
    DRIFT_RESPONSE_LOG2_ALERT,
    DRIFT_RESPONSE_LOG2_WARN,
    DRIFT_STABLE,
    DRIFT_TRIGGERS_RETRAIN,
    DRIFT_TYPES,
    DRIFT_VERDICTS,
    DRIFT_WARN,
)


class DriftDetectionError(ValueError):
    """Raised when a drift request is structurally invalid."""


def _clean(values: Iterable[Any], label: str) -> list[float]:
    out: list[float] = []
    for value in values or []:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DriftDetectionError(f"{label} must be numeric, got {value!r}")
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise DriftDetectionError(f"{label} must be finite, got {value!r}")
        out.append(number)
    return out


def _result(drift_type: str, verdict: str, reason: str, **detail) -> dict:
    if drift_type not in DRIFT_TYPES:
        raise DriftDetectionError(f"{drift_type!r} is not a declared drift type")
    if verdict not in DRIFT_VERDICTS:
        raise DriftDetectionError(f"{verdict!r} is not a drift verdict")
    record = {"drift_type": drift_type, "verdict": verdict, "reason": reason}
    record.update(detail)
    return record


def quantile_edges(reference: Sequence[float], bins: int = DRIFT_PSI_BINS) -> list[float]:
    """Interior bin edges at the reference quantiles.

    QUANTILE, NOT FIXED. A fixed scale is blind to any feature that does not
    happen to live on it — MEASURED, a [0,10] binning scores 0.0000 on a
    3.2-sigma shift in a feature ranged [-0.3, 0.3].
    """
    ordered = sorted(reference)
    if not ordered:
        return []
    return [ordered[int(len(ordered) * index / bins)] for index in range(1, bins)]


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], bins: int = DRIFT_PSI_BINS
) -> float:
    """PSI between two samples, binned on the REFERENCE quantiles."""
    edges = quantile_edges(reference, bins)

    def proportions(values: Sequence[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            counts[sum(1 for edge in edges if value >= edge)] += 1
        total = len(values) or 1
        return [count / total for count in counts]

    epsilon = 1e-4
    total = 0.0
    for ref_share, cur_share in zip(proportions(reference), proportions(current)):
        ref_share = max(ref_share, epsilon)
        cur_share = max(cur_share, epsilon)
        total += (cur_share - ref_share) * math.log(cur_share / ref_share)
    return round(total, 6)


def _psi_verdict(psi: float) -> str:
    if psi >= DRIFT_PSI_ALERT:
        return DRIFT_ALERT
    if psi >= DRIFT_PSI_WARN:
        return DRIFT_WARN
    return DRIFT_STABLE


def feature_drift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    name: str = "",
    min_window: int = DRIFT_MIN_WINDOW,
) -> dict:
    """Has the distribution of one feature moved?"""
    reference = _clean(reference, "reference")
    current = _clean(current, "current")
    if len(reference) < min_window or len(current) < min_window:
        return _result(
            DRIFT_FEATURE,
            DRIFT_NOT_EVALUATED,
            f"need >= {min_window} observations per window (reference="
            f"{len(reference)}, current={len(current)}). MEASURED with no "
            f"drift present, PSI exceeds 0.25 on 77% of comparisons at window "
            f"50 and 20.5% at window 100",
            feature=name,
            reference_count=len(reference),
            current_count=len(current),
        )
    psi = population_stability_index(reference, current)
    verdict = _psi_verdict(psi)
    return _result(
        DRIFT_FEATURE,
        verdict,
        f"PSI {psi:.4f} against warn {DRIFT_PSI_WARN} / alert {DRIFT_PSI_ALERT}",
        feature=name,
        psi=psi,
        reference_count=len(reference),
        current_count=len(current),
        reference_mean=round(statistics.mean(reference), 6),
        current_mean=round(statistics.mean(current), 6),
    )


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in xs) * sum((b - mean_y) ** 2 for b in ys)
    )
    if denominator == 0:
        return None
    return numerator / denominator


def relationship_drift(
    reference_features: Sequence[float],
    reference_outcomes: Sequence[float],
    current_features: Sequence[float],
    current_outcomes: Sequence[float],
    *,
    name: str = "",
    min_window: int = DRIFT_MIN_WINDOW,
    max_shift: float = DRIFT_CORRELATION_SHIFT,
) -> dict:
    """Has the feature→outcome relationship changed, with features unmoved?

    MEASURED, a sign flip moves the correlation 0.683 (+0.368 → −0.315) while
    the feature distribution is IDENTICAL and its PSI reads 0.0360. This is
    the drift a distribution detector cannot see.
    """
    reference_features = _clean(reference_features, "reference features")
    reference_outcomes = _clean(reference_outcomes, "reference outcomes")
    current_features = _clean(current_features, "current features")
    current_outcomes = _clean(current_outcomes, "current outcomes")

    if len(reference_features) != len(reference_outcomes) or len(
        current_features
    ) != len(current_outcomes):
        raise DriftDetectionError(
            "features and outcomes must be paired — a mismatched length means "
            "the pairing is already wrong"
        )
    if len(reference_features) < min_window or len(current_features) < min_window:
        return _result(
            DRIFT_RELATIONSHIP,
            DRIFT_NOT_EVALUATED,
            f"need >= {min_window} paired observations per window",
            feature=name,
            reference_count=len(reference_features),
            current_count=len(current_features),
        )

    before = _correlation(reference_features, reference_outcomes)
    after = _correlation(current_features, current_outcomes)
    if before is None or after is None:
        return _result(
            DRIFT_RELATIONSHIP,
            DRIFT_NOT_EVALUATED,
            "a correlation could not be computed (zero variance in one window)",
            feature=name,
        )

    shift = abs(after - before)
    if shift >= max_shift * 2:
        verdict = DRIFT_ALERT
    elif shift >= max_shift:
        verdict = DRIFT_WARN
    else:
        verdict = DRIFT_STABLE
    return _result(
        DRIFT_RELATIONSHIP,
        verdict,
        f"correlation moved {before:+.4f} -> {after:+.4f} (shift {shift:.4f} "
        f"against {max_shift})",
        feature=name,
        reference_correlation=round(before, 6),
        current_correlation=round(after, 6),
        shift=round(shift, 6),
        sign_flip=bool(before * after < 0),
    )


def calibration_drift(
    predictions: Sequence[float],
    outcomes: Sequence[float],
    *,
    min_window: int = DRIFT_MIN_WINDOW,
    max_gap: float = DRIFT_CALIBRATION_GAP,
) -> dict:
    """Do stated probabilities still match observed frequencies?

    A model whose RANKING is intact can be badly miscalibrated. MEASURED,
    inflating probabilities by 1.6x moved the calibration gap 0.000 → +0.251
    while the correlation barely moved (0.448 → 0.430) — so the relationship
    detector is blind to this by construction.
    """
    predictions = _clean(predictions, "predictions")
    outcomes = _clean(outcomes, "outcomes")
    if len(predictions) != len(outcomes):
        raise DriftDetectionError("predictions and outcomes must be paired")
    if len(predictions) < min_window:
        return _result(
            DRIFT_CALIBRATION,
            DRIFT_NOT_EVALUATED,
            f"need >= {min_window} scored predictions, got {len(predictions)}",
            count=len(predictions),
        )
    for value in predictions:
        if not 0.0 <= value <= 1.0:
            raise DriftDetectionError(
                f"a calibration prediction must be a probability, got {value!r}"
            )

    predicted = statistics.mean(predictions)
    observed = statistics.mean(outcomes)
    gap = predicted - observed
    magnitude = abs(gap)
    if magnitude >= max_gap * 2:
        verdict = DRIFT_ALERT
    elif magnitude >= max_gap:
        verdict = DRIFT_WARN
    else:
        verdict = DRIFT_STABLE
    return _result(
        DRIFT_CALIBRATION,
        verdict,
        f"mean predicted {predicted:.4f} vs observed {observed:.4f} "
        f"(gap {gap:+.4f} against {max_gap})",
        predicted_mean=round(predicted, 6),
        observed_mean=round(observed, 6),
        gap=round(gap, 6),
        # NAMED, NOT SIGNED. "+0.25" means nothing without knowing which way.
        direction="OVERCONFIDENT" if gap > 0 else "UNDERCONFIDENT",
        count=len(predictions),
    )


def event_response_drift(
    reference_responses: Sequence[float],
    current_responses: Sequence[float],
    *,
    event_type: str = "",
    min_window: int = DRIFT_MIN_WINDOW,
    warn_log2: float = DRIFT_RESPONSE_LOG2_WARN,
    alert_log2: float = DRIFT_RESPONSE_LOG2_ALERT,
) -> dict:
    """Do events still move price the way they used to?

    Compared as a LOG2 FACTOR, not a percentage. A percentage is the wrong
    scale for a two-sided comparison and my own gate caught it: a decline is
    bounded at −100% while an increase is unbounded, so a symmetric rule on
    |ratio| is not symmetric. MEASURED under the first version, an event type
    that had stopped moving price entirely (−99.1%) read WARN while only a
    mathematically exact zero reached ALERT. On a log scale halving is −1 and
    doubling is +1, so the bound is a FACTOR that reads the same both ways.

    MEASURED, a response collapsing +0.0404 → +0.0001 leaves features,
    relationship and calibration all unchanged — this detector is the only
    one that sees it.
    """
    reference_responses = _clean(reference_responses, "reference responses")
    current_responses = _clean(current_responses, "current responses")
    if len(reference_responses) < min_window or len(current_responses) < min_window:
        return _result(
            DRIFT_RESPONSE,
            DRIFT_NOT_EVALUATED,
            f"need >= {min_window} measured responses per window "
            f"(reference={len(reference_responses)}, "
            f"current={len(current_responses)})",
            event_type=event_type,
            reference_count=len(reference_responses),
            current_count=len(current_responses),
        )

    before = statistics.mean(abs(value) for value in reference_responses)
    after = statistics.mean(abs(value) for value in current_responses)
    if before == 0:
        return _result(
            DRIFT_RESPONSE,
            DRIFT_NOT_EVALUATED,
            "the reference response is zero, so a relative change is undefined",
            event_type=event_type,
            reference_magnitude=0.0,
            current_magnitude=round(after, 6),
        )

    ratio = (after - before) / before
    if after <= 0:
        # A response that has gone to ZERO is the strongest possible form of
        # this drift, and log2(0) is undefined. Naming it explicitly is more
        # honest than letting a floor decide.
        log2_factor = -math.inf
        verdict = DRIFT_ALERT
    else:
        log2_factor = math.log2(after / before)
        magnitude = abs(log2_factor)
        if magnitude >= alert_log2:
            verdict = DRIFT_ALERT
        elif magnitude >= warn_log2:
            verdict = DRIFT_WARN
        else:
            verdict = DRIFT_STABLE
    shown = "-inf" if log2_factor == -math.inf else f"{log2_factor:+.2f}"
    return _result(
        DRIFT_RESPONSE,
        verdict,
        f"mean absolute response {before:.4f} -> {after:.4f} "
        f"(log2 factor {shown}, warn {warn_log2}, alert {alert_log2})",
        event_type=event_type,
        reference_magnitude=round(before, 6),
        current_magnitude=round(after, 6),
        ratio=round(ratio, 6),
        log2_factor=(None if log2_factor == -math.inf else round(log2_factor, 6)),
        response_vanished=bool(after <= 0),
        direction="WEAKER" if after < before else "STRONGER",
        reference_count=len(reference_responses),
        current_count=len(current_responses),
    )


def _rank(verdict: str) -> int:
    return DRIFT_VERDICTS.index(verdict) if verdict in DRIFT_VERDICTS else 0


def drift_report(
    *,
    features: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    relationships: Mapping[str, tuple] | None = None,
    calibration: tuple[Sequence[float], Sequence[float]] | None = None,
    responses: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> dict:
    """All four detectors, reported together.

    Every declared drift type appears, even when it could not be evaluated —
    a scan that silently covers three of four and reports clean is exactly
    the failure the type list exists to prevent.
    """
    findings: list[dict] = []

    for name, pair in sorted((features or {}).items()):
        findings.append(feature_drift(pair[0], pair[1], name=name))
    for name, quad in sorted((relationships or {}).items()):
        findings.append(relationship_drift(*quad, name=name))
    if calibration is not None:
        findings.append(calibration_drift(calibration[0], calibration[1]))
    for name, pair in sorted((responses or {}).items()):
        findings.append(event_response_drift(pair[0], pair[1], event_type=name))

    covered = {finding["drift_type"] for finding in findings}
    for drift_type in DRIFT_TYPES:
        if drift_type not in covered:
            findings.append(
                _result(
                    drift_type,
                    DRIFT_NOT_EVALUATED,
                    "no data was supplied for this drift type, so it has found "
                    "nothing rather than found the system stable",
                )
            )

    by_type: dict[str, str] = {}
    for drift_type in DRIFT_TYPES:
        verdicts = [f["verdict"] for f in findings if f["drift_type"] == drift_type]
        by_type[drift_type] = (
            max(verdicts, key=_rank) if verdicts else DRIFT_NOT_EVALUATED
        )

    alerts = [f for f in findings if f["verdict"] == DRIFT_ALERT]
    warnings = [f for f in findings if f["verdict"] == DRIFT_WARN]
    unevaluated = [f for f in findings if f["verdict"] == DRIFT_NOT_EVALUATED]

    return {
        "version": DRIFT_DETECTION_VERSION,
        "findings": findings,
        "by_type": by_type,
        "alerts": len(alerts),
        "warnings": len(warnings),
        "not_evaluated": len(unevaluated),
        "worst": max((f["verdict"] for f in findings), key=_rank, default=DRIFT_NOT_EVALUATED),
        # DRIFT NEVER RETRAINS BY ITSELF.
        "triggers_retrain": DRIFT_TRIGGERS_RETRAIN,
        "note": (
            "Drift is evidence for the L5 chain, not a substitute for it: a "
            "model that replaces itself on an alert is the uncontrolled "
            "self-modification Sprint L exists to prevent."
        ),
        "coverage_note": (
            "Every declared drift type is reported. NOT_EVALUATED means the "
            "detector could not run, which is not the same as stable."
        ),
    }


def drift_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a drift report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    by_type = report.get("by_type") or {}
    missing = [name for name in DRIFT_TYPES if name not in by_type]
    if missing:
        problems.append(
            f"the report omits {missing} — the feature detector catches only "
            f"one of the four drift types, so a partial scan that reports "
            f"clean is a false assurance"
        )
    for finding in report.get("findings") or []:
        if finding.get("verdict") not in DRIFT_VERDICTS:
            problems.append(f"{finding.get('drift_type')}: unknown verdict")
        if not str(finding.get("reason") or "").strip():
            problems.append(f"{finding.get('drift_type')}: no reason given")
    if report.get("triggers_retrain"):
        problems.append(
            "the report claims drift triggers a retrain — drift is evidence "
            "for the L5 chain, never a substitute for it"
        )
    return problems


def render_drift(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per finding."""
    lines: list[str] = []
    for finding in report.get("findings") or []:
        label = finding.get("feature") or finding.get("event_type") or ""
        suffix = f" [{label}]" if label else ""
        lines.append(
            f"  {str(finding.get('drift_type')):22s} "
            f"{str(finding.get('verdict')):14s} {finding.get('reason')}{suffix}"
        )
    return lines
