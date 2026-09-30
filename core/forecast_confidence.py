"""Forecast confidence (Sprint F7) — how reliable, not how bullish.

`assess_confidence(...)` scores a forecast across the nine factors the sprint
names — sample size, calibration, model agreement, feature completeness,
source quality, regime similarity, event similarity, model drift, uncertainty
— and returns a confidence bounded by its **weakest** factor, never its mean.

**Confidence is not P(up), and that is measured rather than asserted.**

    case                        P(up)      N      interval   width
    coin flip, huge sample      0.500   1000   [0.47,0.53]   0.062
    near-certain, tiny sample   1.000      2   [0.34,1.00]   0.658

`P(up) = 0.500` from a thousand observations is a HIGH-confidence statement.
`P(up) = 1.000` from two is a NEAR-ZERO-confidence one. They are orthogonal,
so a dashboard that renders the probability as a confidence bar inverts the
meaning exactly where it matters. Every payload says so, and a confidence
value never travels without the factor that bound it.

**Why not a weighted sum.** `core.score_engine._compute_confidence` averages
six factors for the SCORE, and that shape does not survive here. MEASURED, on
forecast factors:

    everything strong        weighted sum 0.950   min 0.950
    N=2 fatal, rest strong   weighted sum 0.717   min 0.020
    uniformly mediocre       weighted sum 0.550   min 0.550

A forecast built on TWO observations scores 0.717 — higher than a uniformly
mediocre one at 0.550. N=2 is the F4 stress cell, the least trustworthy state
in the system, and averaging buries it behind five strong factors.

Pure MIN fails the other way: "good N, everything else weak" scores 0.300,
indistinguishable from a single weak factor. So the **weakest factor caps the
confidence, and the weighted sum may only lower it further**. That is also how
the rest of the system already reasons: F3, F4 and F5 each refuse on the
single most fundamental obstacle rather than averaging obstacles together.

**Three factors are UNMEASURABLE, not zero.** Calibration, model agreement and
model drift all require a trained model, and MEASURED, `build_joint_forecast`
returns NO_MODEL — none is registered. Scoring them 1.0 ("nothing wrong") or
0.0 ("everything wrong") would both be false. They report UNMEASURABLE, carry
no number, and are excluded from the weighting rather than silently counted.

**Not the score's confidence (W5).** That object exists, is wired into the
risk policy, and is not this. Every payload carries
`assessed_object: "forecast"` so the two can never be read as one.

**Sample size is scored through F4's tiers**, not a second sample-size policy
that would drift from the first.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from core.config import (
    CONDITIONAL_MAX_INTERVAL_WIDTH,
    CONDITIONAL_MIN_SAMPLES_DIRECTIONAL,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    EVENT_MEMORY_MIN_SIMILARITY,
    FCONF_BAND_NONE,
    FCONF_CALIBRATION,
    FCONF_EVENT_SIMILARITY,
    FCONF_FEATURE_COMPLETENESS,
    FCONF_MODEL_AGREEMENT,
    FCONF_MODEL_DRIFT,
    FCONF_REGIME_SIMILARITY,
    FCONF_SAMPLE_SIZE,
    FCONF_SOURCE_QUALITY,
    FCONF_STATUS_MEASURED,
    FCONF_STATUS_UNAVAILABLE,
    FCONF_STATUS_UNMEASURABLE,
    FCONF_UNCERTAINTY,
    FORECAST_CONFIDENCE_AGGREGATION,
    FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE,
    FORECAST_CONFIDENCE_BANDS,
    FORECAST_CONFIDENCE_CONTRACT_VERSION,
    FORECAST_CONFIDENCE_DISCLAIMER,
    FORECAST_CONFIDENCE_FACTORS,
    FORECAST_CONFIDENCE_MEASURABLE,
    FORECAST_CONFIDENCE_VERSION,
    FORECAST_CONFIDENCE_WEIGHTS,
)

LOGGER = logging.getLogger("core.forecast_confidence")

# The object being assessed. Named in every payload so this can never be
# mistaken for core.score_engine's confidence in the SCORE.
ASSESSED_OBJECT = "forecast"

# Why each model-dependent factor cannot be measured. Stated per factor rather
# than as one blanket reason, so wiring one later is visibly a different job.
_UNMEASURABLE_REASONS = {
    FCONF_CALIBRATION: (
        "no fitted calibration map exists; M6 needs a trained model, and "
        "build_joint_forecast reports NO_MODEL. Scoring this 1.0 would claim "
        "the forecast is calibrated, and 0.0 would claim it is miscalibrated "
        "— both assert a measurement that was never made"
    ),
    FCONF_MODEL_AGREEMENT: (
        "there is one rule-based baseline and no trained model, so nothing "
        "can agree or disagree with anything. Agreement among one is not 1.0, "
        "it is undefined"
    ),
    FCONF_MODEL_DRIFT: (
        "nothing is deployed, so nothing can have drifted. A drift score of "
        "0.0 would claim a stable deployed model, which does not exist"
    ),
}


class ForecastConfidenceError(ValueError):
    """Raised when a confidence request violates the F7 contract."""


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _factor(
    name: str,
    status: str,
    *,
    value: float | None = None,
    reason: str = "",
    **detail,
) -> dict:
    """One factor row. A non-MEASURED factor never carries a value."""
    row = {
        "factor": name,
        "status": status,
        "measurable": name in FORECAST_CONFIDENCE_MEASURABLE,
        "weight": FORECAST_CONFIDENCE_WEIGHTS.get(name),
        "reason": reason,
    }
    if status == FCONF_STATUS_MEASURED:
        if value is None:
            raise ForecastConfidenceError(
                f"{name}: a MEASURED factor must carry a value"
            )
        row["value"] = round(_clip01(value), 6)
        row.update(detail)
    return row


# ---------------------------------------------------------------------------
# The six measurable factors
# ---------------------------------------------------------------------------


def sample_size_factor(samples: int, effective: int | None = None) -> dict:
    """How much evidence the forecast rests on, scored through F4's tiers.

    Uses the EFFECTIVE sample where one is supplied (F5 discounts analogs
    drawn from a single ticker), because 45 observations of one stock are not
    45 independent observations.
    """
    count = int(effective if effective is not None else samples)
    if count <= 0:
        return _factor(
            FCONF_SAMPLE_SIZE, FCONF_STATUS_UNAVAILABLE,
            reason="no observations underlie this forecast",
        )

    # Anchored on F4's MEASURED floors rather than a fresh curve: the point
    # floor is where |error| > 0.15 falls to 1.6% of draws.
    if count >= CONDITIONAL_MIN_SAMPLES_POINT:
        value = _clip01(0.75 + 0.25 * min(1.0, count / (CONDITIONAL_MIN_SAMPLES_POINT * 3)))
        note = f"{count} observations clear the point floor of {CONDITIONAL_MIN_SAMPLES_POINT}"
    elif count >= CONDITIONAL_MIN_SAMPLES_DIRECTIONAL:
        span = CONDITIONAL_MIN_SAMPLES_POINT - CONDITIONAL_MIN_SAMPLES_DIRECTIONAL
        value = 0.50 + 0.25 * ((count - CONDITIONAL_MIN_SAMPLES_DIRECTIONAL) / span)
        note = f"{count} observations support a direction but not a point estimate"
    elif count >= CONDITIONAL_MIN_SAMPLES_INTERVAL:
        span = CONDITIONAL_MIN_SAMPLES_DIRECTIONAL - CONDITIONAL_MIN_SAMPLES_INTERVAL
        value = 0.25 + 0.25 * ((count - CONDITIONAL_MIN_SAMPLES_INTERVAL) / span)
        note = f"{count} observations support a range only"
    else:
        # Below every floor. Deliberately near zero: this is the state that a
        # weighted sum was MEASURED to hide behind five strong factors.
        value = 0.25 * (count / CONDITIONAL_MIN_SAMPLES_INTERVAL)
        note = (
            f"{count} observations is below every claim floor "
            f"(weakest is {CONDITIONAL_MIN_SAMPLES_INTERVAL})"
        )

    return _factor(
        FCONF_SAMPLE_SIZE, FCONF_STATUS_MEASURED, value=value, reason=note,
        samples=int(samples), effective_samples=count,
    )


def uncertainty_factor(interval: dict | None) -> dict:
    """How tight the published interval is. Wide is honest but uninformative."""
    if not interval:
        return _factor(
            FCONF_UNCERTAINTY, FCONF_STATUS_UNAVAILABLE,
            reason="no interval was published for this forecast",
        )
    lower, upper = interval.get("lower"), interval.get("upper")
    if lower is None or upper is None:
        return _factor(
            FCONF_UNCERTAINTY, FCONF_STATUS_UNAVAILABLE,
            reason="the interval carries no bounds",
        )
    width = float(upper) - float(lower)
    # Scored against F4's measured width cap: at the cap the interval spans
    # too much of the range to constrain anything.
    value = _clip01(1.0 - width / max(CONDITIONAL_MAX_INTERVAL_WIDTH, 1e-9))
    return _factor(
        FCONF_UNCERTAINTY, FCONF_STATUS_MEASURED, value=value,
        reason=(
            f"interval width {width:.3f} against the {CONDITIONAL_MAX_INTERVAL_WIDTH} "
            f"cap above which a range constrains nothing"
        ),
        width=round(width, 6),
    )


def source_quality_factor(observed_share: float | None) -> dict:
    """Were the underlying events OBSERVED, or dated by inference?

    MEASURED: the inference method's precision is 0.65, so roughly one in
    three inferred events is not the event it is labelled as.
    """
    if observed_share is None:
        return _factor(
            FCONF_SOURCE_QUALITY, FCONF_STATUS_UNAVAILABLE,
            reason="no provenance mix was supplied",
        )
    share = _clip01(observed_share)
    # An all-inferred set is not worthless — the price move is real — but its
    # event attribution is only 65% reliable, so it caps below an observed set.
    value = _clip01(0.65 + 0.35 * share)
    return _factor(
        FCONF_SOURCE_QUALITY, FCONF_STATUS_MEASURED, value=value,
        reason=(
            f"{share:.0%} of the evidence was OBSERVED; inferred events carry "
            f"a measured precision of 0.65, so which event produced the move "
            f"was never sourced for the rest"
        ),
        observed_share=round(share, 6),
    )


def regime_similarity_factor(agreement: float | None) -> dict:
    """Did the historical evidence occur in the regime that holds now?"""
    if agreement is None:
        return _factor(
            FCONF_REGIME_SIMILARITY, FCONF_STATUS_UNAVAILABLE,
            reason="no regime agreement was established for this forecast",
        )
    value = _clip01(agreement)
    return _factor(
        FCONF_REGIME_SIMILARITY, FCONF_STATUS_MEASURED, value=value,
        reason=(
            f"{value:.0%} of the evidence occurred in the same market regime "
            f"that holds at as_of"
        ),
        agreement=round(value, 6),
    )


def event_similarity_factor(similarities) -> dict:
    """How closely the retrieved analogs actually resembled this setup.

    Scored against the retrieval bar: a set that only just cleared 0.70 is
    weaker evidence than one clustered near 1.0, even at the same count.
    """
    values = [float(s) for s in (similarities or ()) if s is not None]
    if not values:
        return _factor(
            FCONF_EVENT_SIMILARITY, FCONF_STATUS_UNAVAILABLE,
            reason="no analog similarities were supplied",
        )
    mean = sum(values) / len(values)
    headroom = max(1.0 - EVENT_MEMORY_MIN_SIMILARITY, 1e-9)
    value = _clip01((mean - EVENT_MEMORY_MIN_SIMILARITY) / headroom)
    return _factor(
        FCONF_EVENT_SIMILARITY, FCONF_STATUS_MEASURED, value=value,
        reason=(
            f"mean analog similarity {mean:.3f} against the "
            f"{EVENT_MEMORY_MIN_SIMILARITY} retrieval bar; a set that barely "
            f"cleared the bar is weaker evidence than one clustered near 1.0"
        ),
        mean_similarity=round(mean, 6),
        analogs=len(values),
    )


def feature_completeness_factor(present: int, expected: int) -> dict:
    """How much of the expected feature surface was actually available."""
    if expected <= 0:
        return _factor(
            FCONF_FEATURE_COMPLETENESS, FCONF_STATUS_UNAVAILABLE,
            reason="no expected feature set was declared",
        )
    value = _clip01(present / expected)
    return _factor(
        FCONF_FEATURE_COMPLETENESS, FCONF_STATUS_MEASURED, value=value,
        reason=f"{present} of {expected} expected features were present",
        present=int(present), expected=int(expected),
    )


def _unmeasurable(name: str) -> dict:
    return _factor(
        name, FCONF_STATUS_UNMEASURABLE,
        reason=_UNMEASURABLE_REASONS.get(
            name, "this factor requires a trained model, and none is registered"
        ),
    )


# ---------------------------------------------------------------------------
# Aggregation — limiting factor, never an average
# ---------------------------------------------------------------------------


def band_for(value: float) -> str:
    """The reading band a confidence falls in."""
    band = FCONF_BAND_NONE
    for name, floor in FORECAST_CONFIDENCE_BANDS:
        if value >= floor:
            band = name
    return band


def aggregate(factors: dict) -> dict:
    """Combine measured factors by LIMITING FACTOR.

    The weakest factor caps the confidence, and the weighted sum modulates
    that cap within half its range — so a cap alone can never be reached by a
    forecast that is weak across the board.

    MEASURED, an average rates an N=2 forecast at 0.717 — higher than a
    uniformly mediocre one at 0.550 — which inverts the ranking exactly where
    it matters. A pure cap fails the other way: one weak factor among strong
    ones scored identically to three weak factors, making the other five
    decorative.
    """
    measured = {
        name: row for name, row in factors.items()
        if row.get("status") == FCONF_STATUS_MEASURED
    }
    if not measured:
        return {
            "value": 0.0,
            "band": FCONF_BAND_NONE,
            "binding_factor": None,
            "binding_value": None,
            "weighted_sum": None,
            "measured_factors": 0,
            "aggregation": FORECAST_CONFIDENCE_AGGREGATION,
            "reason": (
                "no factor could be measured, so there is no evidence about "
                "how reliable this forecast is — which is itself a reason to "
                "treat it as unreliable"
            ),
        }

    # The cap: the weakest measured factor.
    binding_name = min(measured, key=lambda name: measured[name]["value"])
    cap = float(measured[binding_name]["value"])

    # Renormalise the weights across the factors actually measured, so an
    # UNAVAILABLE factor does not silently drag the sum toward zero.
    total_weight = sum(
        FORECAST_CONFIDENCE_WEIGHTS.get(name, 0.0) for name in measured
    )
    if total_weight <= 0:
        weighted = cap
    else:
        weighted = sum(
            FORECAST_CONFIDENCE_WEIGHTS.get(name, 0.0) * float(row["value"])
            for name, row in measured.items()
        ) / total_weight

    # The cap BINDS, and the weighted sum MODULATES it within half its range.
    #
    # `min(cap, weighted)` was the first shape and it is wrong: MEASURED, a
    # forecast with one weak factor (0.30) and two strong ones scored
    # identically to one with three weak factors (0.30 / 0.32 / 0.31), because
    # the cap dominated and the sum never bound. Broad weakness has to
    # register, or five of six factors are decorative.
    #
    # `cap * weighted` over-corrects: a uniformly mediocre forecast falls from
    # 0.55 to 0.30, penalising it for the very consistency that makes it
    # ordinary. MEASURED across the four candidate shapes, only this one
    # satisfies all three requirements at once:
    #   all-weak 0.196 < one-weak-two-strong 0.248   (weakness registers)
    #   N=2 fatal 0.018                              (the cap still dominates)
    #   everything strong 0.926                      (no undue penalty)
    value = round(cap * (0.5 + 0.5 * weighted), 6)
    return {
        "value": value,
        "band": band_for(value),
        "binding_factor": binding_name,
        "binding_value": round(cap, 6),
        "weighted_sum": round(weighted, 6),
        "measured_factors": len(measured),
        "aggregation": FORECAST_CONFIDENCE_AGGREGATION,
        "reason": (
            f"capped by {binding_name} at {cap:.3f}; the weighted sum across "
            f"{len(measured)} measured factor(s) was {weighted:.3f}"
        ),
    }


# ---------------------------------------------------------------------------
# The assessment
# ---------------------------------------------------------------------------


def confidence_of(assessment: Mapping[str, Any] | None) -> float | None:
    """The confidence SCALAR from an assessment, or None when unmeasured.

    B3. F7 returns a mapping carrying SEVERAL unrelated floats — `binding_value`,
    `weighted_sum` and every factor's own value — so a consumer that guesses
    ("take the first number") silently thresholds on a factor score instead of
    the confidence.

    MEASURED: two independent consumers hit this in one sprint and each had to
    special-case it. D1's renderer dumped the ~4,000-character mapping into a
    table cell, and A6 raised on it before learning which field named the
    quantity. Exporting the accessor means the third consumer reads it the same
    way as the first two.

    Raises rather than coercing on a malformed value: a confidence outside [0, 1]
    is a contract breach, and returning None for it would make a broken producer
    indistinguishable from an honest absence.
    """
    if assessment is None:
        return None
    if not isinstance(assessment, Mapping):
        raise ForecastConfidenceError(
            "a confidence assessment must be a mapping; a bare float has no "
            "band, no binding factor and no record of what was unmeasurable"
        )
    value = assessment.get("confidence")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ForecastConfidenceError(
            f"confidence {value!r} is not a number"
        ) from None
    if not 0.0 <= value <= 1.0:
        raise ForecastConfidenceError(
            f"confidence {value!r} lies outside [0, 1]"
        )
    return value


def band_of(assessment: Mapping[str, Any] | None) -> str | None:
    """The confidence BAND, or None when unmeasured."""
    if assessment is None:
        return None
    if not isinstance(assessment, Mapping):
        raise ForecastConfidenceError("a confidence assessment must be a mapping")
    band = assessment.get("band")
    return str(band) if band else None


def binding_factor_of(assessment: Mapping[str, Any] | None) -> str | None:
    """WHICH factor bound the confidence, or None when unmeasured.

    Distinct from `binding_value`, which is a float and is exactly the number a
    guessing consumer picks up by mistake.
    """
    if assessment is None:
        return None
    if not isinstance(assessment, Mapping):
        raise ForecastConfidenceError("a confidence assessment must be a mapping")
    factor = assessment.get("binding_factor")
    return str(factor) if factor else None


def summarise_assessment(assessment: Mapping[str, Any] | None) -> dict[str, Any]:
    """The few fields a consumer normally wants, without the factor detail.

    D1's renderer dumped the whole mapping into a table cell because nothing
    offered a short form. This is that short form: SMALL by design, so rendering
    it cannot produce a 4,000-character cell.
    """
    measured = (assessment or {}).get("measured") if assessment else None
    unmeasurable = (assessment or {}).get("unmeasurable") if assessment else None
    return {
        "confidence": confidence_of(assessment),
        "band": band_of(assessment),
        "binding_factor": binding_factor_of(assessment),
        "measured_count": len(measured) if measured is not None else None,
        "unmeasurable_count": len(unmeasurable) if unmeasurable is not None else None,
    }


def assess_confidence(
    *,
    samples: int = 0,
    effective_samples: int | None = None,
    interval: dict | None = None,
    observed_share: float | None = None,
    regime_agreement: float | None = None,
    similarities=None,
    features_present: int = 0,
    features_expected: int = 0,
    probability: float | None = None,
    horizon: str | None = None,
    ticker: str | None = None,
    as_of=None,
) -> dict:
    """Assess how reliable a forecast is — never how bullish it is."""
    factors: dict[str, dict] = {
        FCONF_SAMPLE_SIZE: sample_size_factor(samples, effective_samples),
        FCONF_FEATURE_COMPLETENESS: feature_completeness_factor(
            features_present, features_expected
        ),
        FCONF_SOURCE_QUALITY: source_quality_factor(observed_share),
        FCONF_REGIME_SIMILARITY: regime_similarity_factor(regime_agreement),
        FCONF_EVENT_SIMILARITY: event_similarity_factor(similarities),
        FCONF_UNCERTAINTY: uncertainty_factor(interval),
    }
    for name in FORECAST_CONFIDENCE_FACTORS:
        if name not in FORECAST_CONFIDENCE_MEASURABLE:
            factors[name] = _unmeasurable(name)

    summary = aggregate(factors)

    payload = {
        "assessed_object": ASSESSED_OBJECT,
        "ticker": ticker,
        "as_of": str(as_of) if as_of is not None else None,
        "horizon": horizon,
        "confidence": summary["value"],
        "band": summary["band"],
        "binding_factor": summary["binding_factor"],
        "binding_value": summary["binding_value"],
        "weighted_sum": summary["weighted_sum"],
        "aggregation": summary["aggregation"],
        "aggregation_evidence": FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE,
        "reason": summary["reason"],
        "factor_order": list(FORECAST_CONFIDENCE_FACTORS),
        "factors": {name: factors[name] for name in FORECAST_CONFIDENCE_FACTORS},
        "measured": [
            n for n, r in factors.items() if r["status"] == FCONF_STATUS_MEASURED
        ],
        "unmeasurable": [
            n for n, r in factors.items() if r["status"] == FCONF_STATUS_UNMEASURABLE
        ],
        "unavailable": [
            n for n, r in factors.items() if r["status"] == FCONF_STATUS_UNAVAILABLE
        ],
        "disclaimer": FORECAST_CONFIDENCE_DISCLAIMER,
        "calculation_version": FORECAST_CONFIDENCE_VERSION,
        "contract_version": FORECAST_CONFIDENCE_CONTRACT_VERSION,
    }

    # The probability travels ALONGSIDE the confidence, explicitly labelled as
    # a different quantity, so a consumer holding both cannot conflate them.
    if probability is not None:
        payload["probability"] = round(float(probability), 6)
        payload["probability_note"] = (
            "this is P(up), a separate quantity from the confidence above; a "
            "probability of 0.50 can be far more reliable than one of 1.00"
        )
    return payload


def assess_event_forecast(forecast: dict, similarities=None) -> dict:
    """Assess an F5 event-conditioned forecast, reading its own stages."""
    if not isinstance(forecast, dict):
        raise ForecastConfidenceError("a forecast dict is required")

    stages = forecast.get("stages") or {}
    matches = stages.get("matches") or {}
    regime = stages.get("regime") or {}
    forecast_stage = stages.get("forecast") or {}
    chart_stage = stages.get("chart") or {}

    from core.config import EVENT_MEMORY_CHART_FIELDS

    return assess_confidence(
        samples=int(forecast.get("samples") or 0),
        effective_samples=forecast_stage.get("effective_samples"),
        interval=forecast.get("interval"),
        observed_share=matches.get("observed_share"),
        regime_agreement=regime.get("agreement"),
        similarities=similarities,
        features_present=int(chart_stage.get("fields") or 0),
        features_expected=len(EVENT_MEMORY_CHART_FIELDS),
        probability=forecast.get("value"),
        horizon=forecast.get("horizon"),
        ticker=(forecast.get("event") or {}).get("entity"),
        as_of=forecast.get("as_of"),
    )


def confidence_problems(assessment: dict) -> list[str]:
    """Validate an assessment against the F7 contract."""
    problems: list[str] = []
    if not isinstance(assessment, dict):
        return ["assessment must be a dict"]

    if assessment.get("assessed_object") != ASSESSED_OBJECT:
        problems.append(
            "the assessment does not say it describes the FORECAST — it could "
            "be mistaken for the score's confidence, which is a different "
            "object wired into the risk policy"
        )
    if assessment.get("aggregation") != FORECAST_CONFIDENCE_AGGREGATION:
        problems.append(
            "the confidence was not aggregated by limiting factor. "
            + FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE
        )
    if "not P(up)" not in (assessment.get("disclaimer") or ""):
        problems.append(
            "the disclaimer no longer says confidence is not P(up) — that "
            "conflation is the specific failure this sprint prevents"
        )

    value = assessment.get("confidence")
    if value is None:
        problems.append("the assessment carries no confidence value")
    elif not 0.0 <= float(value) <= 1.0:
        problems.append(f"confidence {value!r} lies outside [0, 1]")

    # A bare scalar is what lets a reader treat confidence as a probability.
    if value and not assessment.get("binding_factor"):
        problems.append(
            "a non-zero confidence carries no binding factor — a bare scalar "
            "invites a reader to treat it as P(up)"
        )

    factors = assessment.get("factors") or {}
    if list(factors) != list(FORECAST_CONFIDENCE_FACTORS):
        problems.append(
            "not every declared factor is reported in order — a missing row "
            "reads as an oversight where UNMEASURABLE reads as a fact"
        )

    for name, row in factors.items():
        status = row.get("status")
        if status not in (
            FCONF_STATUS_MEASURED, FCONF_STATUS_UNMEASURABLE, FCONF_STATUS_UNAVAILABLE
        ):
            problems.append(f"{name}: unknown status {status!r}")
        if status != FCONF_STATUS_MEASURED and "value" in row:
            problems.append(
                f"{name}: a {status} factor carries a value — it would "
                f"coalesce to a number in a consumer and claim a measurement "
                f"that never happened"
            )
        if status != FCONF_STATUS_MEASURED and not row.get("reason"):
            problems.append(f"{name}: a {status} factor does not explain itself")
        if status == FCONF_STATUS_UNMEASURABLE and name in FORECAST_CONFIDENCE_MEASURABLE:
            problems.append(
                f"{name}: reported UNMEASURABLE although it is declared measurable"
            )
        if status == FCONF_STATUS_MEASURED:
            factor_value = row.get("value")
            if factor_value is None or not 0.0 <= float(factor_value) <= 1.0:
                problems.append(f"{name}: measured value {factor_value!r} outside [0, 1]")

    # The limiting property, checked on the output rather than trusted.
    measured = {
        n: r for n, r in factors.items() if r.get("status") == FCONF_STATUS_MEASURED
    }
    if measured and value is not None:
        weakest = min(float(r["value"]) for r in measured.values())
        if float(value) > weakest + 1e-9:
            problems.append(
                f"confidence {float(value):.6f} exceeds its weakest factor "
                f"{weakest:.6f} — an average would do this, and MEASURED, that "
                f"rates an N=2 forecast above a uniformly mediocre one"
            )

    if "probability" in assessment and not assessment.get("probability_note"):
        problems.append(
            "a probability travels with this confidence but is not labelled as "
            "a separate quantity"
        )
    return problems


def render_factors(assessment: dict) -> list[dict]:
    """One reading row per factor, in declared order. Never blank."""
    rows: list[dict] = []
    binding = assessment.get("binding_factor")
    for name in assessment.get("factor_order") or ():
        row = (assessment.get("factors") or {}).get(name) or {}
        rows.append(
            {
                "factor": name,
                "status": row.get("status"),
                "value": row.get("value"),
                "weight": row.get("weight"),
                "binding": name == binding,
                "reason": row.get("reason") or "",
            }
        )
    return rows
