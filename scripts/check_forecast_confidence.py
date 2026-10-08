"""CI drift gate for F7 forecast confidence.

Two properties carry this sprint, and both are the kind a well-meaning
refactor removes:

  - confidence is bounded by its WEAKEST factor, never its mean;
  - confidence is NOT P(up).

1.  THE LIMITING PROPERTY HOLDS. MEASURED: a weighted sum rates an N=2
    forecast at 0.717, HIGHER than a uniformly mediocre one at 0.550. N=2 is
    the F4 stress cell, so averaging inverts the ranking exactly where it
    matters;
2.  a fatal factor cannot be outvoted, however strong the others are;
3.  ...but the aggregation is not pure MIN either: broad weakness must still
    register, or five factors are being ignored;
4.  CONFIDENCE IS NOT P(up), asserted on behaviour: a high probability from a
    tiny sample must score LOWER than a middling probability from a large one;
5.  UNMEASURABLE stays distinct from a measured zero. Calibration, model
    agreement and drift all need a trained model that does not exist, and
    scoring them 0.0 or 1.0 would both be false;
6.  an unmeasurable factor carries NO value and NO weight;
7.  every declared factor is reported, in order;
8.  a confidence never travels as a bare scalar — the binding factor comes
    with it, because a lone number invites being read as a probability;
9.  the assessment names the FORECAST as its object, so it cannot be confused
    with core.score_engine's confidence in the SCORE (W5);
10. sample size is scored through F4's measured floors, not a second policy.

Synthetic and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    FCONF_BAND_NONE,
    FCONF_SAMPLE_SIZE,
    FCONF_SOURCE_QUALITY,
    FCONF_STATUS_MEASURED,
    FCONF_STATUS_UNMEASURABLE,
    FCONF_UNCERTAINTY,
    FORECAST_CONFIDENCE_AGGREGATION,
    FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE,
    FORECAST_CONFIDENCE_FACTORS,
    FORECAST_CONFIDENCE_MEASURABLE,
    FORECAST_CONFIDENCE_WEIGHTS,
)
from core.forecast_confidence import (  # noqa: E402
    ASSESSED_OBJECT,
    ForecastConfidenceError,
    _factor,
    aggregate,
    assess_confidence,
    confidence_problems,
    render_factors,
    sample_size_factor,
)

STRONG = dict(
    interval={"lower": 0.45, "upper": 0.55},
    observed_share=1.0,
    regime_agreement=1.0,
    similarities=[0.98] * 40,
    features_present=13,
    features_expected=13,
)


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1 + 2
    fatal = assess_confidence(samples=2, **STRONG)
    if fatal["binding_factor"] != FCONF_SAMPLE_SIZE:
        failures.append(
            f"an N=2 forecast with everything else strong was bound by "
            f"{fatal['binding_factor']!r} rather than sample_size"
        )
    if fatal["confidence"] > 0.25:
        failures.append(
            f"an N=2 forecast scored {fatal['confidence']:.3f} — MEASURED, a "
            f"weighted sum gives 0.717 here, higher than a uniformly mediocre "
            f"forecast at 0.550, and N=2 is the F4 stress cell"
        )
    if fatal["weighted_sum"] is not None and fatal["confidence"] > fatal["weighted_sum"]:
        failures.append(
            "the confidence exceeded its own weighted sum — the cap is meant "
            "to LOWER the result, never raise it"
        )

    mediocre = assess_confidence(
        samples=25,
        interval={"lower": 0.35, "upper": 0.62},
        observed_share=0.5, regime_agreement=0.55,
        similarities=[0.85] * 25, features_present=8, features_expected=13,
    )
    if fatal["confidence"] >= mediocre["confidence"]:
        failures.append(
            f"an N=2 forecast ({fatal['confidence']:.3f}) scored at or above a "
            f"uniformly mediocre one ({mediocre['confidence']:.3f}) — this is "
            f"the exact inversion the limiting aggregation exists to prevent"
        )

    # ---------------------------------------------------------------- 3
    # Not pure MIN: broad weakness must register beyond the single weakest.
    # REAL factor names: an unregistered name carries weight 0.0, which makes
    # total_weight zero and collapses the aggregate onto the cap. That is a
    # fixture artefact, not the behaviour under test.
    def _measured(values):
        return {
            name: _factor(name, FCONF_STATUS_MEASURED, value=value)
            for name, value in values.items()
        }

    one_weak = _measured({
        FCONF_SAMPLE_SIZE: 0.30, FCONF_UNCERTAINTY: 0.95,
        FCONF_SOURCE_QUALITY: 0.95,
    })
    all_weak = _measured({
        FCONF_SAMPLE_SIZE: 0.30, FCONF_UNCERTAINTY: 0.32,
        FCONF_SOURCE_QUALITY: 0.31,
    })
    if aggregate(all_weak)["value"] >= aggregate(one_weak)["value"]:
        failures.append(
            "three weak factors scored at or above one weak factor with two "
            "strong ones — the aggregation has collapsed to pure MIN, which "
            "ignores the other factors entirely"
        )

    # ---------------------------------------------------------------- 4
    # CONFIDENCE IS NOT P(up), asserted on behaviour.
    certain_but_tiny = assess_confidence(
        samples=2, probability=1.0,
        interval={"lower": 0.34, "upper": 1.0},
        observed_share=1.0, regime_agreement=1.0,
        similarities=[0.98] * 2, features_present=13, features_expected=13,
    )
    middling_but_large = assess_confidence(
        samples=CONDITIONAL_MIN_SAMPLES_POINT * 4, probability=0.50,
        interval={"lower": 0.47, "upper": 0.53},
        observed_share=1.0, regime_agreement=1.0,
        similarities=[0.98] * 100, features_present=13, features_expected=13,
    )
    if certain_but_tiny["confidence"] >= middling_but_large["confidence"]:
        failures.append(
            f"P(up)=1.00 from 2 observations scored "
            f"{certain_but_tiny['confidence']:.3f}, at or above P(up)=0.50 from "
            f"{CONDITIONAL_MIN_SAMPLES_POINT * 4} observations at "
            f"{middling_but_large['confidence']:.3f} — confidence is tracking "
            f"the probability instead of the evidence"
        )
    for label, assessment in (
        ("certain_but_tiny", certain_but_tiny),
        ("middling_but_large", middling_but_large),
    ):
        if "probability" in assessment and not assessment.get("probability_note"):
            failures.append(
                f"{label}: a probability travels with the confidence but is "
                f"not labelled as a separate quantity"
            )
        if "not P(up)" not in (assessment.get("disclaimer") or ""):
            failures.append(f"{label}: the disclaimer no longer denies the conflation")

    # ---------------------------------------------------------------- 5 + 6
    unmeasurable = set(FORECAST_CONFIDENCE_FACTORS) - set(FORECAST_CONFIDENCE_MEASURABLE)
    if not unmeasurable:
        failures.append(
            "every factor is now measurable — if a model really was trained, "
            "this gate must be updated deliberately, because UNMEASURABLE "
            "existing is what stops an unmeasured factor reading as a zero"
        )
    for name in unmeasurable:
        row = fatal["factors"][name]
        if row["status"] != FCONF_STATUS_UNMEASURABLE:
            failures.append(f"{name}: is unmeasurable but reports {row['status']!r}")
        if "value" in row:
            failures.append(
                f"{name}: an UNMEASURABLE factor carries a value — it claims a "
                f"measurement that never happened"
            )
        if row.get("weight") is not None:
            failures.append(
                f"{name}: an UNMEASURABLE factor carries a weight, so it would "
                f"contribute to the aggregate"
            )
        if not row.get("reason"):
            failures.append(f"{name}: does not explain why it cannot be measured")

    # The guard itself: a caller must not be able to smuggle a value in.
    smuggled = _factor("calibration", FCONF_STATUS_UNMEASURABLE, value=1.0, reason="x")
    if "value" in smuggled:
        failures.append(
            "a non-MEASURED factor kept a value that was passed to it — this "
            "is how an unmeasurable factor starts looking measured"
        )
    try:
        _factor("sample_size", FCONF_STATUS_MEASURED, reason="no value supplied")
        failures.append("a MEASURED factor without a value was accepted")
    except ForecastConfidenceError:
        pass

    # ---------------------------------------------------------------- 7
    if list(fatal["factors"]) != list(FORECAST_CONFIDENCE_FACTORS):
        failures.append("not every declared factor is reported in order")
    if len(FORECAST_CONFIDENCE_FACTORS) != 9:
        failures.append(
            f"the confidence now accounts for {len(FORECAST_CONFIDENCE_FACTORS)} "
            f"factors, not the nine the sprint names"
        )

    # ---------------------------------------------------------------- 8
    if not fatal.get("binding_factor"):
        failures.append(
            "a confidence was published with no binding factor — a bare "
            "scalar invites a reader to treat it as P(up)"
        )

    # ---------------------------------------------------------------- 9
    if fatal.get("assessed_object") != ASSESSED_OBJECT or ASSESSED_OBJECT != "forecast":
        failures.append(
            "the assessment does not name the FORECAST as its object — it "
            "could be read as the score's confidence, which is a different "
            "object wired into the risk policy"
        )
    if FORECAST_CONFIDENCE_AGGREGATION != "limiting_factor":
        failures.append(
            "the aggregation is no longer limiting_factor. "
            + FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE
        )
    if "MEASURED" not in FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE:
        failures.append("the aggregation evidence lost its measurement")

    # ---------------------------------------------------------------- 10
    below = sample_size_factor(CONDITIONAL_MIN_SAMPLES_INTERVAL - 1)
    at_point = sample_size_factor(CONDITIONAL_MIN_SAMPLES_POINT)
    if below["value"] >= 0.25:
        failures.append(
            f"a sample below F4's weakest floor scored {below['value']:.3f} — "
            f"it must sit near zero, because that is the state a weighted sum "
            f"was measured to hide"
        )
    if at_point["value"] < 0.75:
        failures.append(
            f"a sample at F4's POINT floor scored {at_point['value']:.3f} — the "
            f"floors must line up with F4's measured tiers, not a fresh curve"
        )
    if FORECAST_CONFIDENCE_WEIGHTS[FCONF_SAMPLE_SIZE] < max(
        FORECAST_CONFIDENCE_WEIGHTS.values()
    ):
        failures.append("sample_size no longer carries the largest weight")

    # Nothing measurable at all is honest, not confident.
    empty = assess_confidence()
    if empty["confidence"] != 0.0 or empty["band"] != FCONF_BAND_NONE:
        failures.append(
            f"an assessment with nothing measured scored "
            f"{empty['confidence']} / {empty['band']} rather than zero"
        )

    for label, assessment in (
        ("fatal", fatal), ("mediocre", mediocre), ("empty", empty),
        ("certain_but_tiny", certain_but_tiny),
    ):
        for problem in confidence_problems(assessment):
            failures.append(f"{label}: contract problem: {problem}")
        for row in render_factors(assessment):
            if not row.get("reason"):
                failures.append(f"{label}: {row['factor']} renders blank for a reader")

    if failures:
        print("F7 forecast-confidence gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F7 forecast-confidence gate OK:")
    print(
        f"  {len(FORECAST_CONFIDENCE_FACTORS)} factors, "
        f"{len(FORECAST_CONFIDENCE_MEASURABLE)} measurable, "
        f"{len(unmeasurable)} UNMEASURABLE (never a measured zero)."
    )
    print(
        f"  limiting factor holds: N=2 with everything else strong scores "
        f"{fatal['confidence']:.3f}, below a mediocre {mediocre['confidence']:.3f}."
    )
    print(
        f"  confidence is NOT P(up): 1.00-from-2 scores "
        f"{certain_but_tiny['confidence']:.3f} vs 0.50-from-many at "
        f"{middling_but_large['confidence']:.3f}."
    )
    print("  every value travels with the factor that bound it.")
    print("  sample size is scored through F4's measured floors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
