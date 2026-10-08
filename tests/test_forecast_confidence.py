"""F7 forecast-confidence tests.

Two properties carry this sprint: confidence is bounded by its WEAKEST factor
rather than its mean, and confidence is NOT P(up).
"""

from __future__ import annotations

import unittest

from core.config import (
    CONDITIONAL_MIN_SAMPLES_DIRECTIONAL,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    FCONF_BAND_NONE,
    FCONF_CALIBRATION,
    FCONF_MODEL_AGREEMENT,
    FCONF_MODEL_DRIFT,
    FCONF_SAMPLE_SIZE,
    FCONF_SOURCE_QUALITY,
    FCONF_STATUS_MEASURED,
    FCONF_STATUS_UNAVAILABLE,
    FCONF_STATUS_UNMEASURABLE,
    FCONF_UNCERTAINTY,
    FORECAST_CONFIDENCE_AGGREGATION,
    FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE,
    FORECAST_CONFIDENCE_BANDS,
    FORECAST_CONFIDENCE_DISCLAIMER,
    FORECAST_CONFIDENCE_FACTORS,
    FORECAST_CONFIDENCE_MEASURABLE,
    FORECAST_CONFIDENCE_WEIGHTS,
)
from core.forecast_confidence import (
    ASSESSED_OBJECT,
    ForecastConfidenceError,
    _factor,
    aggregate,
    assess_confidence,
    band_for,
    confidence_problems,
    event_similarity_factor,
    feature_completeness_factor,
    regime_similarity_factor,
    render_factors,
    sample_size_factor,
    source_quality_factor,
    uncertainty_factor,
)

STRONG = dict(
    interval={"lower": 0.45, "upper": 0.55},
    observed_share=1.0,
    regime_agreement=1.0,
    similarities=[0.98] * 40,
    features_present=13,
    features_expected=13,
)


def _measured(values):
    return {
        name: _factor(name, FCONF_STATUS_MEASURED, value=value)
        for name, value in values.items()
    }


class LimitingFactorTests(unittest.TestCase):
    """The weakest factor binds; averaging is the measured defect."""

    def test_a_fatal_factor_cannot_be_outvoted(self):
        # MEASURED: a weighted sum scores this 0.717, above a uniformly
        # mediocre forecast at 0.550. N=2 is the F4 stress cell.
        fatal = assess_confidence(samples=2, **STRONG)
        self.assertEqual(fatal["binding_factor"], FCONF_SAMPLE_SIZE)
        self.assertLess(fatal["confidence"], 0.25)

    def test_a_fatal_forecast_scores_below_a_mediocre_one(self):
        fatal = assess_confidence(samples=2, **STRONG)
        mediocre = assess_confidence(
            samples=25, interval={"lower": 0.35, "upper": 0.62},
            observed_share=0.5, regime_agreement=0.55,
            similarities=[0.85] * 25, features_present=8, features_expected=13,
        )
        self.assertLess(fatal["confidence"], mediocre["confidence"])

    def test_broad_weakness_registers_beyond_the_weakest_factor(self):
        # Pure MIN would score these identically, making five factors
        # decorative. This was a real defect in the first implementation.
        one_weak = _measured({
            FCONF_SAMPLE_SIZE: 0.30, FCONF_UNCERTAINTY: 0.95,
            FCONF_SOURCE_QUALITY: 0.95,
        })
        all_weak = _measured({
            FCONF_SAMPLE_SIZE: 0.30, FCONF_UNCERTAINTY: 0.32,
            FCONF_SOURCE_QUALITY: 0.31,
        })
        self.assertLess(aggregate(all_weak)["value"], aggregate(one_weak)["value"])

    def test_the_cap_is_never_exceeded(self):
        for values in (
            {FCONF_SAMPLE_SIZE: 0.20, FCONF_UNCERTAINTY: 0.99},
            {FCONF_SAMPLE_SIZE: 0.99, FCONF_UNCERTAINTY: 0.05},
            {FCONF_SAMPLE_SIZE: 0.50, FCONF_UNCERTAINTY: 0.50},
        ):
            summary = aggregate(_measured(values))
            self.assertLessEqual(summary["value"], summary["binding_value"] + 1e-9)

    def test_a_uniformly_strong_forecast_is_not_unduly_penalised(self):
        strong = aggregate(_measured({
            FCONF_SAMPLE_SIZE: 0.95, FCONF_UNCERTAINTY: 0.95,
            FCONF_SOURCE_QUALITY: 0.95,
        }))
        self.assertGreater(strong["value"], 0.85)

    def test_the_aggregation_is_declared_and_evidenced(self):
        self.assertEqual(FORECAST_CONFIDENCE_AGGREGATION, "limiting_factor")
        self.assertIn("MEASURED", FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE)

    def test_nothing_measurable_is_zero_not_confident(self):
        empty = assess_confidence()
        self.assertEqual(empty["confidence"], 0.0)
        self.assertEqual(empty["band"], FCONF_BAND_NONE)
        self.assertIsNone(empty["binding_factor"])


class ConfidenceIsNotProbabilityTests(unittest.TestCase):
    """The conflation this sprint exists to prevent."""

    def test_high_probability_from_a_tiny_sample_scores_low(self):
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
        self.assertLess(
            certain_but_tiny["confidence"], middling_but_large["confidence"]
        )

    def test_a_probability_travels_labelled_as_a_separate_quantity(self):
        assessment = assess_confidence(samples=50, probability=0.72, **STRONG)
        self.assertEqual(assessment["probability"], 0.72)
        self.assertIn("separate quantity", assessment["probability_note"])

    def test_no_probability_means_no_probability_key(self):
        self.assertNotIn("probability", assess_confidence(samples=50, **STRONG))

    def test_the_disclaimer_denies_the_conflation(self):
        self.assertIn("not P(up)", FORECAST_CONFIDENCE_DISCLAIMER)
        self.assertIn(
            "not P(up)", assess_confidence(samples=50, **STRONG)["disclaimer"]
        )

    def test_confidence_does_not_track_the_probability(self):
        # Same evidence, different probabilities -> same confidence.
        low = assess_confidence(samples=50, probability=0.05, **STRONG)
        high = assess_confidence(samples=50, probability=0.95, **STRONG)
        self.assertEqual(low["confidence"], high["confidence"])


class UnmeasurableIsNotZeroTests(unittest.TestCase):
    def setUp(self):
        self.assessment = assess_confidence(samples=50, **STRONG)
        self.unmeasurable = set(FORECAST_CONFIDENCE_FACTORS) - set(
            FORECAST_CONFIDENCE_MEASURABLE
        )

    def test_the_model_dependent_factors_are_unmeasurable(self):
        self.assertEqual(
            self.unmeasurable,
            {FCONF_CALIBRATION, FCONF_MODEL_AGREEMENT, FCONF_MODEL_DRIFT},
        )

    def test_they_carry_no_value_and_no_weight(self):
        for name in self.unmeasurable:
            row = self.assessment["factors"][name]
            self.assertEqual(row["status"], FCONF_STATUS_UNMEASURABLE)
            self.assertNotIn("value", row)
            self.assertIsNone(row["weight"])

    def test_they_explain_why_they_cannot_be_measured(self):
        for name in self.unmeasurable:
            self.assertTrue(self.assessment["factors"][name]["reason"], name)

    def test_a_blocked_status_discards_a_smuggled_value(self):
        row = _factor(FCONF_CALIBRATION, FCONF_STATUS_UNMEASURABLE, value=1.0)
        self.assertNotIn("value", row)

    def test_a_measured_factor_without_a_value_is_refused(self):
        with self.assertRaises(ForecastConfidenceError):
            _factor(FCONF_SAMPLE_SIZE, FCONF_STATUS_MEASURED, reason="none")

    def test_unmeasurable_factors_do_not_enter_the_aggregate(self):
        self.assertNotIn(
            FCONF_CALIBRATION, FORECAST_CONFIDENCE_WEIGHTS
        )


class FactorTests(unittest.TestCase):
    def test_sample_size_follows_f4s_measured_floors(self):
        below = sample_size_factor(CONDITIONAL_MIN_SAMPLES_INTERVAL - 1)
        interval = sample_size_factor(CONDITIONAL_MIN_SAMPLES_INTERVAL)
        directional = sample_size_factor(CONDITIONAL_MIN_SAMPLES_DIRECTIONAL)
        point = sample_size_factor(CONDITIONAL_MIN_SAMPLES_POINT)
        self.assertLess(below["value"], 0.25)
        self.assertGreaterEqual(interval["value"], 0.25)
        self.assertGreaterEqual(directional["value"], 0.50)
        self.assertGreaterEqual(point["value"], 0.75)

    def test_sample_size_uses_the_effective_count(self):
        # F5 discounts analogs drawn from one ticker; 45 of one stock are not
        # 45 independent observations.
        discounted = sample_size_factor(45, effective=5)
        raw = sample_size_factor(45)
        self.assertLess(discounted["value"], raw["value"])
        self.assertEqual(discounted["effective_samples"], 5)

    def test_zero_samples_is_unavailable_not_zero(self):
        self.assertEqual(
            sample_size_factor(0)["status"], FCONF_STATUS_UNAVAILABLE
        )

    def test_a_wide_interval_scores_lower_than_a_tight_one(self):
        tight = uncertainty_factor({"lower": 0.48, "upper": 0.52})
        wide = uncertainty_factor({"lower": 0.10, "upper": 0.90})
        self.assertGreater(tight["value"], wide["value"])

    def test_a_missing_interval_is_unavailable(self):
        self.assertEqual(
            uncertainty_factor(None)["status"], FCONF_STATUS_UNAVAILABLE
        )
        self.assertEqual(
            uncertainty_factor({"lower": None, "upper": 1.0})["status"],
            FCONF_STATUS_UNAVAILABLE,
        )

    def test_observed_sources_score_above_inferred_ones(self):
        observed = source_quality_factor(1.0)
        inferred = source_quality_factor(0.0)
        self.assertGreater(observed["value"], inferred["value"])
        # An all-inferred set is not worthless: the price move is real.
        self.assertGreater(inferred["value"], 0.5)

    def test_regime_agreement_passes_through(self):
        self.assertAlmostEqual(
            regime_similarity_factor(0.8)["value"], 0.8, places=6
        )
        self.assertEqual(
            regime_similarity_factor(None)["status"], FCONF_STATUS_UNAVAILABLE
        )

    def test_similarity_is_scored_against_the_retrieval_bar(self):
        # A set that barely cleared 0.70 is weaker than one near 1.0.
        barely = event_similarity_factor([0.71] * 10)
        strong = event_similarity_factor([0.99] * 10)
        self.assertLess(barely["value"], strong["value"])
        self.assertEqual(
            event_similarity_factor([])["status"], FCONF_STATUS_UNAVAILABLE
        )

    def test_feature_completeness_is_a_ratio(self):
        self.assertAlmostEqual(
            feature_completeness_factor(10, 20)["value"], 0.5, places=6
        )
        self.assertEqual(
            feature_completeness_factor(5, 0)["status"], FCONF_STATUS_UNAVAILABLE
        )

    def test_every_measured_value_is_inside_the_unit_range(self):
        assessment = assess_confidence(samples=50, **STRONG)
        for name, row in assessment["factors"].items():
            if row["status"] == FCONF_STATUS_MEASURED:
                self.assertGreaterEqual(row["value"], 0.0, name)
                self.assertLessEqual(row["value"], 1.0, name)


class BandTests(unittest.TestCase):
    def test_bands_ascend_and_start_at_zero(self):
        values = [value for _name, value in FORECAST_CONFIDENCE_BANDS]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0.0)

    def test_every_confidence_lands_in_a_band(self):
        for value in (0.0, 0.1, 0.25, 0.4, 0.5, 0.74, 0.75, 1.0):
            self.assertIn(
                band_for(value), [name for name, _ in FORECAST_CONFIDENCE_BANDS]
            )


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assessment = assess_confidence(samples=50, probability=0.6, **STRONG)

    def test_it_names_the_forecast_as_its_object(self):
        self.assertEqual(ASSESSED_OBJECT, "forecast")
        self.assertEqual(self.assessment["assessed_object"], "forecast")

    def test_every_factor_is_reported_in_order(self):
        self.assertEqual(
            list(self.assessment["factors"]), list(FORECAST_CONFIDENCE_FACTORS)
        )

    def test_a_healthy_assessment_raises_nothing(self):
        self.assertEqual(confidence_problems(self.assessment), [])

    def test_a_confidence_above_its_weakest_factor_is_reported(self):
        broken = dict(self.assessment)
        broken["confidence"] = 0.99
        self.assertTrue(
            any("weakest factor" in p for p in confidence_problems(broken))
        )

    def test_a_missing_binding_factor_is_reported(self):
        broken = dict(self.assessment)
        broken["binding_factor"] = None
        self.assertTrue(any("bare scalar" in p for p in confidence_problems(broken)))

    def test_a_value_on_an_unmeasurable_factor_is_reported(self):
        broken = dict(self.assessment)
        broken["factors"] = dict(broken["factors"])
        broken["factors"][FCONF_CALIBRATION] = {
            **broken["factors"][FCONF_CALIBRATION], "value": 1.0,
        }
        self.assertTrue(
            any("never happened" in p for p in confidence_problems(broken))
        )

    def test_render_never_leaves_a_factor_blank(self):
        for row in render_factors(assess_confidence()):
            self.assertTrue(row["reason"], row["factor"])

    def test_render_marks_the_binding_factor(self):
        rows = render_factors(assess_confidence(samples=2, **STRONG))
        binding = [r for r in rows if r["binding"]]
        self.assertEqual(len(binding), 1)
        self.assertEqual(binding[0]["factor"], FCONF_SAMPLE_SIZE)


if __name__ == "__main__":
    unittest.main()
