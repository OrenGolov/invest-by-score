"""B3 tests — F7 exports the accessor, so no consumer guesses at the scalar.

Open item 10: F7 returns a mapping carrying SEVERAL unrelated floats
(`binding_value`, `weighted_sum`, every factor score), and two independent
consumers hit it in one sprint — D1's renderer dumped the ~4,000-character mapping
into a table cell, and A6 raised on it before learning which field named the
quantity. A third consumer was going to guess.
"""

from __future__ import annotations

import unittest

from core.confidence_alert import ConfidenceAlertError, classify_confidence_change
from core.forecast_confidence import (
    ForecastConfidenceError,
    band_of,
    binding_factor_of,
    confidence_of,
    summarise_assessment,
)
from core.research_view import _scalar


def assessment(**overrides):
    """An F7-shaped assessment, with the several floats that invite a guess."""
    base = {
        "assessed_object": "forecast",
        "confidence": 0.83,
        "band": "HIGH",
        "binding_factor": "sample_size",
        # THE TRAP: three more floats, any of which a guessing consumer takes.
        "binding_value": 0.42,
        "weighted_sum": 0.91,
        "factors": {"sample_size": {"value": 0.42, "status": "MEASURED"}},
        "measured": ["sample_size", "calibration"],
        "unmeasurable": ["event_similarity"],
    }
    base.update(overrides)
    return base


class TheAccessorReadsTheRightFloatTests(unittest.TestCase):
    def test_the_confidence_is_not_the_binding_value(self):
        # A consumer taking "the first number" lands on binding_value (0.42) and
        # thresholds on a factor score instead of the confidence (0.83).
        built = assessment()
        self.assertEqual(confidence_of(built), 0.83)
        self.assertNotEqual(confidence_of(built), built["binding_value"])
        self.assertNotEqual(confidence_of(built), built["weighted_sum"])

    def test_the_band_and_binding_factor_are_read_separately(self):
        self.assertEqual(band_of(assessment()), "HIGH")
        self.assertEqual(binding_factor_of(assessment()), "sample_size")

    def test_the_binding_factor_is_a_name_not_a_number(self):
        # `binding_factor` names WHICH factor bound; `binding_value` is the float
        # a guess picks up. Confusing them is the documented failure.
        self.assertIsInstance(binding_factor_of(assessment()), str)

    def test_an_absent_assessment_reads_as_none(self):
        self.assertIsNone(confidence_of(None))
        self.assertIsNone(band_of(None))
        self.assertIsNone(binding_factor_of(None))

    def test_an_unmeasured_confidence_reads_as_none(self):
        self.assertIsNone(confidence_of(assessment(confidence=None)))


class MalformedIsLoudNotCoercedTests(unittest.TestCase):
    """A broken producer must not look like an honest absence."""

    def test_a_non_numeric_confidence_raises(self):
        with self.assertRaises(ForecastConfidenceError):
            confidence_of(assessment(confidence="high"))

    def test_a_confidence_outside_the_unit_interval_raises(self):
        for value in (-0.1, 1.5, 83):
            with self.subTest(value=value):
                with self.assertRaises(ForecastConfidenceError):
                    confidence_of(assessment(confidence=value))

    def test_a_bare_float_is_refused(self):
        # A float has no band, no binding factor and no record of what was
        # unmeasurable, so it cannot stand in for an assessment.
        with self.assertRaises(ForecastConfidenceError):
            confidence_of(0.83)

    def test_a_list_is_refused(self):
        with self.assertRaises(ForecastConfidenceError):
            band_of([0.83])


class TheSummaryIsSmallTests(unittest.TestCase):
    """D1's renderer dumped the whole mapping because nothing offered a short form."""

    def test_the_summary_omits_the_factor_detail(self):
        summary = summarise_assessment(assessment())
        self.assertNotIn("factors", summary)
        self.assertNotIn("weighted_sum", summary)

    def test_the_summary_keeps_what_a_reader_wants(self):
        summary = summarise_assessment(assessment())
        self.assertEqual(summary["confidence"], 0.83)
        self.assertEqual(summary["band"], "HIGH")
        self.assertEqual(summary["binding_factor"], "sample_size")

    def test_the_summary_counts_rather_than_lists(self):
        summary = summarise_assessment(assessment())
        self.assertEqual(summary["measured_count"], 2)
        self.assertEqual(summary["unmeasurable_count"], 1)

    def test_the_summary_is_short_enough_for_a_table_cell(self):
        # The defect was a ~4,000-character cell.
        self.assertLess(len(str(summarise_assessment(assessment()))), 300)

    def test_an_absent_assessment_summarises_to_nones(self):
        summary = summarise_assessment(None)
        self.assertIsNone(summary["confidence"])
        self.assertIsNone(summary["measured_count"])


class BothConsumersReadItTheSameWayTests(unittest.TestCase):
    """The point of B3: one accessor, not three implementations."""

    def test_a6_reads_through_the_accessor(self):
        import inspect

        from core import confidence_alert

        source = inspect.getsource(confidence_alert._read)
        self.assertIn("confidence_of", source)
        self.assertIn("binding_factor_of", source)

    def test_d1_reads_an_assessment_through_the_accessor(self):
        import inspect

        from core import research_view

        source = inspect.getsource(research_view._scalar)
        self.assertIn("confidence_of", source)

    def test_a6_still_raises_its_own_error_type(self):
        # A caller catching ConfidenceAlertError must keep seeing a malformed
        # assessment, so F7's error is translated rather than leaked.
        with self.assertRaises(ConfidenceAlertError):
            classify_confidence_change(
                assessment(confidence="high"), assessment()
            )

    def test_d1_renders_the_confidence_not_another_float(self):
        self.assertEqual(_scalar(assessment()), "0.83")

    def test_d1_renders_a_malformed_assessment_as_invalid(self):
        # A view whose job is to show state must not crash on bad input.
        self.assertEqual(_scalar(assessment(confidence=42)), "{invalid}")

    def test_d1_still_handles_a_generic_mapping(self):
        # The accessor applies to F7 assessments only, identified by F7's own
        # marker. A generic mapping keeps the old behaviour.
        self.assertEqual(_scalar({"value": 7}), "7")
        self.assertEqual(_scalar({"nothing": {"nested": 1}}), "{...}")


class TheAssessmentStillCarriesEverythingTests(unittest.TestCase):
    """The accessors are a reading convention, not a narrowing of the payload."""

    def test_a_real_assessment_exposes_the_scalar_and_the_detail(self):
        from core.forecast_confidence import assess_confidence

        built = assess_confidence(
            ticker="NVDA",
            as_of="2026-09-21",
            horizon="20d",
            samples=400,
            features_present=6,
            features_expected=8,
        )
        self.assertIsNotNone(confidence_of(built))
        self.assertIn("factors", built)
        self.assertIn("factor_order", built)


if __name__ == "__main__":
    unittest.main()
