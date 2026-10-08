"""A1 forecast change alert tests.

The central claims under test: a digest diff fires every day on an unchanged
forecast, and coalescing nulls turns an availability change into a phantom
move. These tests drive the real snapshot assembly rather than reading any data
file, so they mean the same thing on a clean clone.
"""

from __future__ import annotations

import unittest

from core.config import (
    ALERT_CHANGE_APPEARED,
    ALERT_CHANGE_DISAPPEARED,
    ALERT_CHANGE_MOVED,
    ALERT_CHANGE_NONE,
    ALERT_CHANGE_NOT_EVALUATED,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    FORECAST_ALERT_AVAILABILITY_IS_MATERIAL,
    FORECAST_ALERT_BLOCKS_TRADES,
    FORECAST_ALERT_MIN_PROBABILITY_MOVE,
    FORECAST_ALERT_USES_DIGEST,
    FORECAST_ALERT_VERSION,
)
from core.forecast_alert import (
    ForecastAlertError,
    alert_problems,
    classify_change,
    forecast_change_alert,
    render_alert,
)
from core.forecast_snapshot import build_forecast_snapshot, snapshot_digest

BASE = {
    "samples": 800,
    "interval": {"low": 0.48, "high": 0.60},
    "event": {"entity": "NVDA"},
}


def snapshot(as_of, value=None, horizon="20d"):
    kwargs = {}
    if value is not None:
        kwargs["event_forecast"] = dict(
            BASE, value=value, as_of=as_of, horizon=horizon
        )
    return build_forecast_snapshot("NVDA", as_of, horizon, **kwargs)


class TheDigestTrapTest(unittest.TestCase):
    """A digest diff fires every day on an unchanged forecast."""

    def test_identical_empty_forecasts_have_different_digests(self):
        days = ("2026-09-18", "2026-09-19", "2026-09-21", "2026-09-22")
        digests = {snapshot_digest(snapshot(d)) for d in days}
        self.assertEqual(
            len(digests),
            len(days),
            "the snapshot digest no longer changes with as_of; A1's refusal "
            "to compare digests rests on that, and must be re-derived rather "
            "than kept if it has changed",
        )

    def test_identical_empty_forecasts_raise_no_alert(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        self.assertFalse(
            alert["fired"],
            "two days of an identical unavailable forecast fired an alert",
        )
        self.assertEqual(alert["kind"], ALERT_CHANGE_NOT_EVALUATED)

    def test_the_alert_declares_it_does_not_use_digests(self):
        self.assertFalse(FORECAST_ALERT_USES_DIGEST)
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        self.assertFalse(alert["compares_digest"])

    def test_an_alert_comparing_digests_is_caught(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        alert["compares_digest"] = True
        self.assertTrue(alert_problems(alert))


class TheNullCoalescingTrapTest(unittest.TestCase):
    """An availability change is not a magnitude."""

    def test_an_appearing_forecast_is_not_a_move_from_zero(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
        )
        self.assertEqual(alert["kind"], ALERT_CHANGE_APPEARED)
        self.assertIsNone(
            alert["delta"],
            "an appearance carried a delta; float(x or 0) would have made "
            "this a +0.56 move",
        )
        self.assertIsNone(alert["previous"])

    def test_a_disappearing_forecast_is_not_a_crash(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.56), snapshot("2026-09-22")
        )
        self.assertEqual(alert["kind"], ALERT_CHANGE_DISAPPEARED)
        self.assertIsNone(
            alert["delta"],
            "a disappearance carried a delta; that reports a -0.56 crash "
            "that never happened",
        )
        self.assertIsNone(alert["current"])
        self.assertIn("nothing fell", alert["reason"])

    def test_availability_changes_are_material(self):
        self.assertTrue(FORECAST_ALERT_AVAILABILITY_IS_MATERIAL)
        for previous, current in (
            (snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)),
            (snapshot("2026-09-21", 0.56), snapshot("2026-09-22")),
        ):
            self.assertTrue(forecast_change_alert(previous, current)["fired"])

    def test_a_disappearance_warns_and_an_appearance_informs(self):
        appeared = forecast_change_alert(
            snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
        )
        disappeared = forecast_change_alert(
            snapshot("2026-09-21", 0.56), snapshot("2026-09-22")
        )
        self.assertEqual(appeared["severity"], ALERT_SEVERITY_INFO)
        self.assertEqual(disappeared["severity"], ALERT_SEVERITY_WARN)

    def test_an_availability_change_carrying_a_delta_is_caught(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
        )
        alert["delta"] = 0.56
        self.assertTrue(
            alert_problems(alert),
            "an APPEARED alert carrying a delta passed the contract check",
        )

    def test_a_present_value_is_never_coerced(self):
        """A genuine 0.0 forecast is a measurement, not an absence."""
        kind, delta = classify_change(0.0, 0.0)
        self.assertEqual(kind, ALERT_CHANGE_NONE)
        self.assertEqual(delta, 0.0)


class ThresholdTest(unittest.TestCase):
    def test_a_move_at_the_threshold_fires(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.50)
        )
        self.assertEqual(alert["kind"], ALERT_CHANGE_MOVED)
        self.assertTrue(alert["fired"])

    def test_a_move_inside_the_threshold_stays_quiet(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)
        )
        self.assertEqual(alert["kind"], ALERT_CHANGE_NONE)
        self.assertFalse(alert["fired"])
        self.assertIsNone(alert["severity"])

    def test_the_threshold_sits_above_sampling_noise(self):
        import math

        band = 1.96 * math.sqrt(0.25 / 100)
        self.assertGreaterEqual(
            FORECAST_ALERT_MIN_PROBABILITY_MOVE,
            band * 0.9,
            f"the {FORECAST_ALERT_MIN_PROBABILITY_MOVE} threshold sits inside "
            f"the ±{band:.4f} sampling band at 100 observations, so the alert "
            f"fires on resampling",
        )

    def test_the_delta_is_signed(self):
        down = forecast_change_alert(
            snapshot("2026-09-21", 0.60), snapshot("2026-09-22", 0.40)
        )
        self.assertLess(down["delta"], 0)

    def test_an_impossible_threshold_is_refused(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ForecastAlertError):
                classify_change(0.4, 0.6, threshold=bad)

    def test_a_below_threshold_move_marked_moved_is_caught(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)
        )
        alert["kind"] = ALERT_CHANGE_MOVED
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(alert_problems(alert))


class NoPriorTest(unittest.TestCase):
    """A first observation is not a change."""

    def test_no_previous_snapshot_is_not_evaluated(self):
        alert = forecast_change_alert(None, snapshot("2026-09-22", 0.56))
        self.assertEqual(alert["kind"], ALERT_CHANGE_NOT_EVALUATED)
        self.assertFalse(alert["fired"])
        self.assertIsNone(alert["delta"])

    def test_not_evaluated_is_distinct_from_none(self):
        self.assertNotEqual(ALERT_CHANGE_NOT_EVALUATED, ALERT_CHANGE_NONE)

    def test_a_first_observation_that_fires_is_caught(self):
        alert = forecast_change_alert(None, snapshot("2026-09-22", 0.56))
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(alert_problems(alert))

    def test_a_missing_current_snapshot_is_refused(self):
        with self.assertRaises(ForecastAlertError):
            forecast_change_alert(snapshot("2026-09-21"), None)

    def test_a_non_mapping_snapshot_is_refused(self):
        with self.assertRaises(ForecastAlertError):
            forecast_change_alert("yesterday", snapshot("2026-09-22"))

    def test_a_non_numeric_present_value_is_refused(self):
        broken = snapshot("2026-09-22", 0.56)
        broken["fields"]["probability_up"]["value"] = "high"
        with self.assertRaises(ForecastAlertError):
            forecast_change_alert(snapshot("2026-09-21", 0.4), broken)


class ContractTest(unittest.TestCase):
    def test_a_clean_alert_has_no_problems(self):
        for previous, current in (
            (snapshot("2026-09-21"), snapshot("2026-09-22")),
            (snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)),
            (snapshot("2026-09-21", 0.56), snapshot("2026-09-22")),
            (snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)),
            (snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)),
        ):
            alert = forecast_change_alert(previous, current)
            self.assertEqual(alert_problems(alert), [], alert["kind"])
            self.assertEqual(alert["version"], FORECAST_ALERT_VERSION)

    def test_the_alert_does_not_block_trades(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(FORECAST_ALERT_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        alert["blocks_trades"] = True
        self.assertTrue(alert_problems(alert))

    def test_an_unknown_kind_is_caught(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        alert["kind"] = "PROBABLY"
        self.assertTrue(alert_problems(alert))

    def test_a_reasonless_alert_is_caught(self):
        alert = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
        alert["reason"] = "  "
        self.assertTrue(alert_problems(alert))

    def test_a_fired_alert_without_severity_is_caught(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
        )
        alert["severity"] = None
        self.assertTrue(alert_problems(alert))

    def test_a_quiet_alert_with_severity_is_caught(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)
        )
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(alert_problems(alert))

    def test_an_inconsistent_delta_is_caught(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)
        )
        alert["delta"] = 0.99
        self.assertTrue(alert_problems(alert))


class RenderTest(unittest.TestCase):
    def test_the_kind_and_reason_are_rendered(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)
        )
        text = "\n".join(render_alert(alert))
        self.assertIn(ALERT_CHANGE_MOVED, text)
        self.assertIn("FIRED", text)

    def test_an_absent_value_renders_as_absent(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.56), snapshot("2026-09-22")
        )
        line = render_alert(alert)[0]
        self.assertIn("—", line)
        self.assertNotIn("0.0000 ", line.split("->")[1])

    def test_lines_stay_readable(self):
        alert = forecast_change_alert(
            snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)
        )
        for line in render_alert(alert):
            self.assertLess(len(line), 200)


if __name__ == "__main__":
    unittest.main()
