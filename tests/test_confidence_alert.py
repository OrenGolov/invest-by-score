"""A2 confidence change alert tests.

The central claim under test: a magnitude-only confidence alert is silent on
51% of the cases where the reason for the uncertainty changed. These tests
drive the real F7 assessor rather than reading any data file, so they mean the
same thing on a clean clone.
"""

from __future__ import annotations

import random
import unittest

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL,
    CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL,
    CONFIDENCE_ALERT_BLOCKS_TRADES,
    CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT,
    CONFIDENCE_ALERT_MIN_MOVE,
    CONFIDENCE_ALERT_VERSION,
    CONF_CHANGE_APPEARED,
    CONF_CHANGE_BAND,
    CONF_CHANGE_BINDING,
    CONF_CHANGE_DISAPPEARED,
    CONF_CHANGE_MOVED,
    CONF_CHANGE_NONE,
    CONF_CHANGE_NOT_EVALUATED,
)
from core.confidence_alert import (
    ConfidenceAlertError,
    classify_confidence_change,
    confidence_alert_problems,
    confidence_change_alert,
    render_confidence_alert,
)
from core.forecast_confidence import assess_confidence

BASE = dict(
    samples=800,
    interval={"low": 0.48, "high": 0.60},
    observed_share=0.95,
    regime_agreement=0.9,
    similarities=[0.9, 0.85],
    features_present=10,
    features_expected=10,
    probability=0.56,
)

UNMEASURED = {
    "assessed_object": "forecast",
    "confidence": None,
    "band": None,
    "binding_factor": None,
}


def assessment(**overrides):
    return assess_confidence(**{**BASE, **overrides})


class TheDecidingMeasurementTest(unittest.TestCase):
    """A magnitude-only alert is blind half the time."""

    def test_sub_threshold_moves_often_change_the_binding_factor(self):
        rng = random.Random(17)

        def sampled():
            return assess_confidence(
                samples=rng.choice([40, 60, 100, 200, 400, 800]),
                interval={
                    "low": 0.45,
                    "high": 0.45 + rng.choice([0.05, 0.10, 0.20, 0.40]),
                },
                observed_share=rng.choice([0.4, 0.6, 0.8, 0.95]),
                regime_agreement=rng.choice([0.3, 0.5, 0.7, 0.9]),
                similarities=[rng.uniform(0.7, 1.0) for _ in range(rng.choice([2, 3, 5]))],
                features_present=rng.choice([5, 7, 9, 10]),
                features_expected=10,
                probability=0.55,
            )

        flat = 0
        changed = 0
        for _ in range(2000):
            before, after = sampled(), sampled()
            if abs(after["confidence"] - before["confidence"]) < CONFIDENCE_ALERT_MIN_MOVE:
                flat += 1
                if before["binding_factor"] != after["binding_factor"]:
                    changed += 1
        self.assertGreater(flat, 200, "too few sub-threshold pairs to measure")
        self.assertGreater(
            changed / flat,
            0.25,
            f"only {changed}/{flat} sub-threshold moves changed the binding "
            f"factor. A2's design rests on that being common; if it has "
            f"vanished the design must be re-derived rather than kept",
        )

    def test_a_flat_move_with_a_changed_binding_factor_fires(self):
        before = assessment()
        after = assessment(samples=120, regime_agreement=0.5)
        self.assertLess(
            abs(after["confidence"] - before["confidence"]),
            CONFIDENCE_ALERT_MIN_MOVE,
            "this test needs a move no magnitude threshold would fire on",
        )
        self.assertNotEqual(before["binding_factor"], after["binding_factor"])
        alert = confidence_change_alert(before, after)
        self.assertEqual(alert["kind"], CONF_CHANGE_BINDING)
        self.assertTrue(alert["fired"])
        self.assertEqual(confidence_alert_problems(alert), [])

    def test_the_alert_names_both_binding_factors(self):
        alert = confidence_change_alert(
            assessment(), assessment(samples=120, regime_agreement=0.5)
        )
        self.assertIn(alert["previous_binding"], alert["reason"])
        self.assertIn(alert["current_binding"], alert["reason"])

    def test_a_binding_change_reported_as_none_is_caught(self):
        alert = confidence_change_alert(
            assessment(), assessment(samples=120, regime_agreement=0.5)
        )
        alert["kind"] = CONF_CHANGE_NONE
        alert["fired"] = False
        alert["severity"] = None
        self.assertTrue(
            confidence_alert_problems(alert),
            "a changed binding factor reported as NONE passed — that is the "
            "51% case this alert exists to catch",
        )

    def test_the_alert_declares_it_watches_the_binding_factor(self):
        self.assertTrue(CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL)
        alert = confidence_change_alert(assessment(), assessment())
        self.assertTrue(alert["watches_binding_factor"])

    def test_an_alert_not_watching_the_binding_factor_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["watches_binding_factor"] = False
        self.assertTrue(confidence_alert_problems(alert))


class ConfidenceIsNotTheForecastTest(unittest.TestCase):
    """Confidence moves when the forecast does not."""

    def test_confidence_moves_with_the_forecast_value_fixed(self):
        before = assessment()
        after = assessment(features_present=6)
        self.assertEqual(before["probability"], after["probability"])
        self.assertNotEqual(before["confidence"], after["confidence"])


def band_cross_only():
    """A pair crossing a band WITHOUT changing the binding factor.

    The band branch has to be exercised alone. MEASURED, without such a pair
    the band check could be disabled entirely and every other scenario still
    passed — they either change the binding factor or clear the magnitude
    threshold, so the band branch was never the one that decided.
    """
    before = assessment()
    for samples in (400, 600, 800, 1000):
        for similarities in ([0.99, 0.99, 0.99], [0.95, 0.96], [0.75, 0.78], [0.72, 0.71]):
            candidate = assessment(samples=samples, similarities=similarities)
            if (
                candidate["binding_factor"] == before["binding_factor"]
                and candidate["band"] != before["band"]
            ):
                return before, candidate
    return before, None


class BandTest(unittest.TestCase):
    def test_a_band_crossing_is_material(self):
        self.assertTrue(CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL)

    def test_a_band_crossing_alone_is_reported_as_a_band_change(self):
        before, after = band_cross_only()
        self.assertIsNotNone(
            after,
            "could not construct a band crossing with an unchanged binding "
            "factor; the band branch is never exercised alone",
        )
        alert = confidence_change_alert(before, after)
        self.assertEqual(
            alert["kind"],
            CONF_CHANGE_BAND,
            "a band crossing with an unchanged binding factor was not "
            "reported as a band change",
        )
        self.assertTrue(alert["fired"])
        self.assertNotEqual(alert["previous_band"], alert["current_band"])
        self.assertEqual(confidence_alert_problems(alert), [])

    def test_a_band_change_with_the_same_band_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["kind"] = CONF_CHANGE_BAND
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_INFO
        self.assertTrue(confidence_alert_problems(alert))

    def test_a_band_crossing_reported_as_none_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["previous_band"] = "LOW"
        alert["current_band"] = "HIGH"
        self.assertTrue(confidence_alert_problems(alert))


class AvailabilityTest(unittest.TestCase):
    """An availability change is not a magnitude."""

    def test_an_appearing_confidence_is_not_a_rise_from_zero(self):
        alert = confidence_change_alert(UNMEASURED, assessment())
        self.assertEqual(alert["kind"], CONF_CHANGE_APPEARED)
        self.assertIsNone(alert["delta"])
        self.assertIsNone(alert["previous"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_INFO)

    def test_a_disappearing_confidence_is_not_a_fall_to_zero(self):
        alert = confidence_change_alert(assessment(), UNMEASURED)
        self.assertEqual(alert["kind"], CONF_CHANGE_DISAPPEARED)
        self.assertIsNone(alert["delta"])
        self.assertIsNone(alert["current"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        self.assertIn("did not fall to zero", alert["reason"])

    def test_an_availability_change_carrying_a_delta_is_caught(self):
        alert = confidence_change_alert(UNMEASURED, assessment())
        alert["delta"] = 0.56
        self.assertTrue(confidence_alert_problems(alert))

    def test_both_unmeasured_is_not_evaluated(self):
        alert = confidence_change_alert(UNMEASURED, UNMEASURED)
        self.assertEqual(alert["kind"], CONF_CHANGE_NOT_EVALUATED)
        self.assertFalse(alert["fired"])


class MeasurabilityTest(unittest.TestCase):
    """UNMEASURABLE is not low."""

    def test_the_config_treats_measurability_as_its_own_event(self):
        self.assertTrue(CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT)

    def test_a_changed_measurable_set_is_reported_separately(self):
        before = assessment()
        after = dict(before)
        after["measured"] = [m for m in before["measured"] if m != "sample_size"]
        alert = confidence_change_alert(before, after)
        self.assertIsNotNone(alert["measurability"])
        self.assertIn("sample_size", alert["measurability"]["lost"])
        self.assertEqual(confidence_alert_problems(alert), [])

    def test_an_unchanged_measurable_set_reports_nothing(self):
        alert = confidence_change_alert(assessment(), assessment())
        self.assertIsNone(alert["measurability"])

    def test_an_empty_measurability_change_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["measurability"] = {"gained": [], "lost": []}
        self.assertTrue(confidence_alert_problems(alert))


class NoPriorTest(unittest.TestCase):
    def test_no_previous_assessment_is_not_evaluated(self):
        alert = confidence_change_alert(None, assessment())
        self.assertEqual(alert["kind"], CONF_CHANGE_NOT_EVALUATED)
        self.assertFalse(alert["fired"])

    def test_not_evaluated_is_distinct_from_none(self):
        self.assertNotEqual(CONF_CHANGE_NOT_EVALUATED, CONF_CHANGE_NONE)

    def test_a_first_observation_that_fires_is_caught(self):
        alert = confidence_change_alert(None, assessment())
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(confidence_alert_problems(alert))

    def test_a_missing_current_assessment_is_refused(self):
        with self.assertRaises(ConfidenceAlertError):
            confidence_change_alert(assessment(), None)

    def test_a_non_mapping_assessment_is_refused(self):
        with self.assertRaises(ConfidenceAlertError):
            confidence_change_alert("yesterday", assessment())

    def test_a_confidence_outside_the_unit_interval_is_refused(self):
        with self.assertRaises(ConfidenceAlertError):
            confidence_change_alert({"confidence": 1.5}, assessment())

    def test_a_non_numeric_confidence_is_refused(self):
        with self.assertRaises(ConfidenceAlertError):
            confidence_change_alert({"confidence": "high"}, assessment())

    def test_an_impossible_threshold_is_refused(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ConfidenceAlertError):
                classify_confidence_change(
                    assessment(), assessment(), threshold=bad
                )


class ContractTest(unittest.TestCase):
    def test_clean_alerts_have_no_problems(self):
        for previous, current in (
            (assessment(), assessment()),
            (UNMEASURED, assessment()),
            (assessment(), UNMEASURED),
            (assessment(), assessment(samples=120, regime_agreement=0.5)),
            (None, assessment()),
        ):
            alert = confidence_change_alert(previous, current)
            self.assertEqual(
                confidence_alert_problems(alert), [], alert["kind"]
            )
            self.assertEqual(alert["version"], CONFIDENCE_ALERT_VERSION)

    def test_the_alert_does_not_block_trades(self):
        alert = confidence_change_alert(assessment(), assessment())
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(CONFIDENCE_ALERT_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["blocks_trades"] = True
        self.assertTrue(confidence_alert_problems(alert))

    def test_an_unknown_kind_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["kind"] = "PROBABLY"
        self.assertTrue(confidence_alert_problems(alert))

    def test_a_reasonless_alert_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["reason"] = "  "
        self.assertTrue(confidence_alert_problems(alert))

    def test_an_inconsistent_delta_is_caught(self):
        alert = confidence_change_alert(
            assessment(), assessment(samples=120, regime_agreement=0.5)
        )
        alert["delta"] = 0.99
        self.assertTrue(confidence_alert_problems(alert))

    def test_a_fired_alert_without_severity_is_caught(self):
        alert = confidence_change_alert(UNMEASURED, assessment())
        alert["severity"] = None
        self.assertTrue(confidence_alert_problems(alert))

    def test_a_quiet_alert_with_severity_is_caught(self):
        alert = confidence_change_alert(assessment(), assessment())
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(confidence_alert_problems(alert))


class RenderTest(unittest.TestCase):
    def test_the_binding_factor_reaches_the_render(self):
        alert = confidence_change_alert(
            assessment(), assessment(samples=120, regime_agreement=0.5)
        )
        text = "\n".join(render_confidence_alert(alert))
        self.assertIn(alert["current_binding"], text)
        self.assertIn("FIRED", text)

    def test_an_absent_confidence_renders_as_absent(self):
        alert = confidence_change_alert(assessment(), UNMEASURED)
        self.assertIn("—", render_confidence_alert(alert)[0])

    def test_a_measurability_change_reaches_the_render(self):
        before = assessment()
        after = dict(before)
        after["measured"] = [m for m in before["measured"] if m != "sample_size"]
        alert = confidence_change_alert(before, after)
        self.assertIn("!!", "\n".join(render_confidence_alert(alert)))

    def test_lines_stay_readable(self):
        alert = confidence_change_alert(
            assessment(), assessment(samples=120, regime_agreement=0.5)
        )
        for line in render_confidence_alert(alert):
            self.assertLess(len(line), 250)


if __name__ == "__main__":
    unittest.main()
