"""A6 forecast threshold alert tests.

The central claim under test: one of the three conditions has no producer, and
both obvious ways to handle that are wrong. These tests drive the real
snapshot assembly rather than reading any data file, so they mean the same
thing on a clean clone.
"""

from __future__ import annotations

import copy
import unittest

from core.config import (
    ALERT_SEVERITY_WARN,
    FCONF_BAND_MODERATE,
    FORECAST_CONFIDENCE_BANDS,
    FORECAST_THRESHOLD_ALERT_VERSION,
    FTHRESHOLD_BLOCKS_TRADES,
    FTHRESHOLD_COERCES_MISSING,
    FTHRESHOLD_CONDITION_CONFIDENCE,
    FTHRESHOLD_CONDITION_PROBABILITY,
    FTHRESHOLD_CONDITION_RETURN,
    FTHRESHOLD_CONDITIONS,
    FTHRESHOLD_FIRED,
    FTHRESHOLD_HORIZON,
    FTHRESHOLD_MIN_CONFIDENCE,
    FTHRESHOLD_MIN_PROBABILITY,
    FTHRESHOLD_MIN_RETURN,
    FTHRESHOLD_NOT_EVALUATED,
    FTHRESHOLD_NOT_MET,
    FTHRESHOLD_REQUIRES_ALL_CONDITIONS,
    FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES,
    FTHRESHOLD_VETOED,
    RESEARCH_VIEW_HORIZONS,
)
from core.forecast_confidence import assess_confidence
from core.forecast_snapshot import build_forecast_snapshot
from core.forecast_threshold_alert import (
    ForecastThresholdError,
    evaluate_conditions,
    forecast_threshold_alert,
    render_threshold_alert,
    threshold_alert_problems,
)


def live_snapshot(horizon=FTHRESHOLD_HORIZON):
    return build_forecast_snapshot("NVDA", "2026-09-23", horizon)


def strong_snapshot(probability=0.72, horizon=FTHRESHOLD_HORIZON):
    confidence = assess_confidence(
        samples=800,
        interval={"low": 0.55, "high": 0.80},
        observed_share=0.95,
        regime_agreement=0.95,
        similarities=[0.95, 0.96, 0.97],
        features_present=10,
        features_expected=10,
        probability=probability,
    )
    forecast = {
        "value": probability,
        "samples": 800,
        "interval": {"low": 0.55, "high": 0.80},
        "as_of": "2026-09-23",
        "horizon": horizon,
        "event": {"entity": "NVDA"},
    }
    return build_forecast_snapshot(
        "NVDA",
        "2026-09-23",
        horizon,
        event_forecast=forecast,
        confidence=confidence,
        regime="bullish",
    )


def with_return(snapshot, value=0.045):
    """The only way to make all three conditions evaluable today."""
    full = copy.deepcopy(snapshot)
    full["fields"]["expected_return"] = {
        "status": "PRESENT",
        "value": value,
        "reason": "supplied by a test",
    }
    return full


class TheDecidingMeasurementTest(unittest.TestCase):
    """A6 cannot fire today, and must say so."""

    def test_no_condition_input_is_present_at_any_horizon(self):
        present = 0
        total = 0
        for horizon in RESEARCH_VIEW_HORIZONS:
            snapshot = live_snapshot(horizon)
            for condition in FTHRESHOLD_CONDITIONS:
                total += 1
                if snapshot["fields"][condition]["status"] == "PRESENT":
                    present += 1
        self.assertEqual(
            present,
            0,
            f"{present} of {total} condition inputs are now PRESENT; A6's "
            f"NOT_EVALUATED design rests on them being unavailable, and must "
            f"be re-derived rather than kept if that has changed",
        )

    def test_the_live_alert_is_not_evaluated_not_quiet(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        self.assertEqual(alert["verdict"], FTHRESHOLD_NOT_EVALUATED)
        self.assertFalse(alert["fired"])
        self.assertNotEqual(
            alert["verdict"],
            FTHRESHOLD_NOT_MET,
            "a blind alert reported as tested-and-failed",
        )

    def test_the_live_alert_names_what_it_could_not_test(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        self.assertIn(FTHRESHOLD_CONDITION_RETURN, alert["unevaluable"])
        self.assertIn("cannot fire today", alert["reason"])

    def test_two_of_three_passing_does_not_fire(self):
        """The case that decides the module."""
        alert = forecast_threshold_alert(strong_snapshot(), {"veto": False})
        conditions = alert["conditions"]
        self.assertTrue(conditions[FTHRESHOLD_CONDITION_PROBABILITY]["met"])
        self.assertTrue(conditions[FTHRESHOLD_CONDITION_CONFIDENCE]["met"])
        self.assertFalse(conditions[FTHRESHOLD_CONDITION_RETURN]["evaluable"])
        self.assertEqual(
            alert["verdict"],
            FTHRESHOLD_NOT_EVALUATED,
            "two of three conditions passing produced a verdict; that "
            "silently weakens the rule to 'the ones we could check held'",
        )
        self.assertFalse(alert["fired"])


class NeverCoercesTest(unittest.TestCase):
    """A missing input is not zero."""

    def test_the_config_refuses_coercion(self):
        self.assertFalse(FTHRESHOLD_COERCES_MISSING)

    def test_an_unevaluable_condition_carries_no_value(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        entry = alert["conditions"][FTHRESHOLD_CONDITION_RETURN]
        self.assertIsNone(
            entry["value"],
            "a missing input was coerced; 0.0 never clears the return bar, "
            "which makes the alert permanently and silently dead",
        )
        self.assertIsNone(entry["met"])
        self.assertFalse(entry["evaluable"])

    def test_a_coerced_value_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["conditions"][FTHRESHOLD_CONDITION_RETURN]["value"] = 0.0
        self.assertTrue(threshold_alert_problems(alert))

    def test_an_alert_declaring_coercion_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["coerces_missing"] = True
        self.assertTrue(threshold_alert_problems(alert))

    def test_a_structured_confidence_is_read_not_guessed(self):
        """F7 stores its whole assessment, not a scalar."""
        alert = forecast_threshold_alert(strong_snapshot(), {"veto": False})
        entry = alert["conditions"][FTHRESHOLD_CONDITION_CONFIDENCE]
        self.assertTrue(entry["evaluable"])
        self.assertIsInstance(entry["value"], float)

    def test_a_mapping_without_the_named_key_is_refused(self):
        """The decoy numbers are the point.

        MEASURED: with a mapping holding no numbers at all, a version that
        guessed "the first numeric value" still passed, because there was
        nothing to guess. F7's real assessment is full of unrelated floats —
        binding_value, weighted_sum, every factor score — so thresholding on
        one of those would compare against a factor, not the confidence.
        """
        snapshot = live_snapshot()
        snapshot["fields"]["confidence"] = {
            "status": "PRESENT",
            "value": {"band": "HIGH", "binding_value": 0.87, "weighted_sum": 0.96},
            "reason": "no confidence scalar, but plenty of unrelated numbers",
        }
        with self.assertRaises(ForecastThresholdError):
            evaluate_conditions(snapshot)

    def test_a_non_numeric_present_value_is_refused(self):
        snapshot = live_snapshot()
        snapshot["fields"]["probability_up"] = {
            "status": "PRESENT",
            "value": "high",
            "reason": "test",
        }
        with self.assertRaises(ForecastThresholdError):
            evaluate_conditions(snapshot)


class AllConditionsTest(unittest.TestCase):
    def test_the_config_requires_all_conditions(self):
        self.assertTrue(FTHRESHOLD_REQUIRES_ALL_CONDITIONS)

    def test_every_roadmap_condition_is_evaluated(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        for condition in FTHRESHOLD_CONDITIONS:
            self.assertIn(condition, alert["conditions"])

    def test_all_three_present_and_clearing_fires(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()), {"veto": False}
        )
        self.assertEqual(alert["verdict"], FTHRESHOLD_FIRED)
        self.assertTrue(alert["fired"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        self.assertEqual(threshold_alert_problems(alert), [])

    def test_a_failing_condition_is_not_met_not_unevaluated(self):
        weak = with_return(strong_snapshot(probability=0.52))
        alert = forecast_threshold_alert(weak, {"veto": False})
        self.assertEqual(alert["verdict"], FTHRESHOLD_NOT_MET)
        self.assertIn(FTHRESHOLD_CONDITION_PROBABILITY, alert["failed"])
        self.assertFalse(alert["fired"])

    def test_a_missing_condition_dropped_from_the_report_is_caught(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()), {"veto": False}
        )
        del alert["conditions"][FTHRESHOLD_CONDITION_RETURN]
        self.assertTrue(threshold_alert_problems(alert))

    def test_firing_with_an_untested_condition_is_caught(self):
        alert = forecast_threshold_alert(strong_snapshot(), {"veto": False})
        alert["verdict"] = FTHRESHOLD_FIRED
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        alert["veto"] = False
        self.assertTrue(
            threshold_alert_problems(alert),
            "an alert fired with expected_return untested",
        )


class ThresholdTest(unittest.TestCase):
    def test_the_probability_bar_clears_a_coin_flip(self):
        import math

        band = 1.96 * math.sqrt(0.25 / 100)
        self.assertGreater(FTHRESHOLD_MIN_PROBABILITY, 0.5 + band * 0.9)

    def test_the_confidence_bar_is_f7s_moderate_floor(self):
        self.assertEqual(
            FTHRESHOLD_MIN_CONFIDENCE,
            dict(FORECAST_CONFIDENCE_BANDS)[FCONF_BAND_MODERATE],
        )

    def test_the_return_bar_is_positive(self):
        self.assertGreater(FTHRESHOLD_MIN_RETURN, 0.0)

    def test_a_value_exactly_at_the_bar_clears_it(self):
        snapshot = with_return(
            strong_snapshot(probability=FTHRESHOLD_MIN_PROBABILITY),
            value=FTHRESHOLD_MIN_RETURN,
        )
        conditions = evaluate_conditions(snapshot)
        self.assertTrue(conditions[FTHRESHOLD_CONDITION_RETURN]["met"])
        self.assertTrue(conditions[FTHRESHOLD_CONDITION_PROBABILITY]["met"])

    def test_a_condition_with_no_threshold_is_refused(self):
        with self.assertRaises(ForecastThresholdError):
            evaluate_conditions(
                live_snapshot(), thresholds={FTHRESHOLD_CONDITION_RETURN: None}
            )

    def test_a_mismatched_met_flag_is_caught(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()), {"veto": False}
        )
        alert["conditions"][FTHRESHOLD_CONDITION_PROBABILITY]["met"] = False
        self.assertTrue(threshold_alert_problems(alert))


class VetoTest(unittest.TestCase):
    """An unknown veto is not 'no veto active'."""

    def test_the_config_suppresses_on_an_unknown_veto(self):
        self.assertTrue(FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES)

    def test_an_active_veto_suppresses(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()),
            {"veto": True, "veto_rule_ids": ["data_quality_below_threshold"]},
        )
        self.assertEqual(alert["verdict"], FTHRESHOLD_VETOED)
        self.assertFalse(alert["fired"])

    def test_an_unknown_veto_suppresses(self):
        alert = forecast_threshold_alert(with_return(strong_snapshot()), None)
        self.assertEqual(alert["verdict"], FTHRESHOLD_VETOED)
        self.assertFalse(alert["fired"])
        self.assertIsNone(alert["veto"])
        self.assertIn("not 'no veto active'", alert["reason"])

    def test_a_risk_report_without_a_veto_field_is_unknown(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()), {"rules": []}
        )
        self.assertEqual(alert["verdict"], FTHRESHOLD_VETOED)
        self.assertIsNone(alert["veto"])

    def test_firing_with_an_unknown_veto_is_caught(self):
        alert = forecast_threshold_alert(with_return(strong_snapshot()), None)
        alert["verdict"] = FTHRESHOLD_FIRED
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(threshold_alert_problems(alert))

    def test_a_non_mapping_risk_report_is_refused(self):
        with self.assertRaises(ForecastThresholdError):
            forecast_threshold_alert(live_snapshot(), "no veto")


class ContractTest(unittest.TestCase):
    def test_clean_alerts_have_no_problems(self):
        for snapshot, risk in (
            (live_snapshot(), {"veto": False}),
            (strong_snapshot(), {"veto": False}),
            (with_return(strong_snapshot()), {"veto": False}),
            (with_return(strong_snapshot()), {"veto": True}),
            (with_return(strong_snapshot()), None),
            (with_return(strong_snapshot(probability=0.52)), {"veto": False}),
        ):
            alert = forecast_threshold_alert(snapshot, risk)
            self.assertEqual(threshold_alert_problems(alert), [], alert["verdict"])
            self.assertEqual(alert["version"], FORECAST_THRESHOLD_ALERT_VERSION)

    def test_the_alert_does_not_block_trades(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(FTHRESHOLD_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["blocks_trades"] = True
        self.assertTrue(threshold_alert_problems(alert))

    def test_an_unknown_verdict_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["verdict"] = "PROBABLY"
        self.assertTrue(threshold_alert_problems(alert))

    def test_a_reasonless_alert_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["reason"] = "  "
        self.assertTrue(threshold_alert_problems(alert))

    def test_a_mismatched_unevaluable_list_is_caught(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        alert["unevaluable"] = []
        self.assertTrue(threshold_alert_problems(alert))

    def test_a_non_mapping_snapshot_is_refused(self):
        with self.assertRaises(ForecastThresholdError):
            forecast_threshold_alert("snapshot", {"veto": False})


class RenderTest(unittest.TestCase):
    def test_every_condition_is_rendered(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        text = "\n".join(render_threshold_alert(alert))
        for condition in FTHRESHOLD_CONDITIONS:
            self.assertIn(condition, text)

    def test_an_unevaluable_condition_renders_as_absent(self):
        alert = forecast_threshold_alert(live_snapshot(), {"veto": False})
        line = [
            entry
            for entry in render_threshold_alert(alert)
            if FTHRESHOLD_CONDITION_RETURN in entry
        ][0]
        self.assertIn("—", line)
        self.assertNotIn("0.0000", line)

    def test_an_unknown_veto_renders_as_unknown(self):
        alert = forecast_threshold_alert(with_return(strong_snapshot()), None)
        self.assertIn("unknown", render_threshold_alert(alert)[0])

    def test_lines_stay_readable(self):
        alert = forecast_threshold_alert(
            with_return(strong_snapshot()), {"veto": False}
        )
        for line in render_threshold_alert(alert):
            self.assertLess(len(line), 400)


if __name__ == "__main__":
    unittest.main()
