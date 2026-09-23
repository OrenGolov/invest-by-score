"""A3 high-impact event alert tests.

The central claim under test: impact must be measured from realized outcomes,
never from the event's own claimed magnitude — which has never been compared
against a single outcome. These tests build their memories in-process rather
than reading the gitignored store, so they mean the same thing on a clean
clone.
"""

from __future__ import annotations

import random
import statistics
import unittest

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    EVENT_IMPACT_ALERT_VERSION,
    EVENT_IMPACT_BLOCKS_TRADES,
    EVENT_IMPACT_HIGH,
    EVENT_IMPACT_HORIZON,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_MIN_ANALOGS,
    EVENT_IMPACT_MIN_MEDIAN_MOVE,
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    EVENT_IMPACT_UNKNOWN_IS_NOT_LOW,
    EVENT_IMPACT_USES_ABSOLUTE_MOVE,
    EVENT_IMPACT_USES_CLAIMED_MAGNITUDE,
    EVENT_MEMORY_MIN_ANALOGS,
)
from core.event_impact_alert import (
    EventImpactAlertError,
    event_impact_alert,
    impact_alert_problems,
    realized_impact,
    render_impact_alert,
)
from core.event_memory import EventMemory

TYPES = {
    "earnings_beat": 0.055,
    "guidance_cut": 0.070,
    "analyst_note": 0.012,
    "minor_pr": 0.006,
}


def memories(seed=23, count=400):
    """Memories whose event types genuinely differ in realized impact."""
    rng = random.Random(seed)
    built = []
    for index in range(count):
        event_type = rng.choice(list(TYPES))
        built.append(
            EventMemory(
                event_id=f"e{index}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type=event_type,
                direction="positive",
                response={"20d": {"abnormal_return": rng.gauss(0, TYPES[event_type])}},
                attribution={
                    "20d": "event_associated" if rng.random() < 0.7 else "confounded"
                },
                provenance="observed",
            )
        )
    return built


def event(event_type, ticker="NVDA"):
    return {
        "event_type": event_type,
        "ticker": ticker,
        "published_time": "2026-09-23",
    }


class NeverReadsTheClaimTest(unittest.TestCase):
    """magnitude has never been compared against an outcome."""

    def test_magnitude_is_not_an_event_memory_field(self):
        sample = EventMemory(
            event_id="x",
            ticker="NVDA",
            published_time="2025-01-10",
            event_type="earnings_beat",
            direction="positive",
        )
        self.assertNotIn(
            "magnitude",
            sample.to_dict(),
            "magnitude is now carried into memory; A3's refusal to read it "
            "rests on it never having been compared against an outcome, and "
            "must be re-derived rather than kept if that has changed",
        )

    def test_the_config_refuses_the_claimed_magnitude(self):
        self.assertFalse(EVENT_IMPACT_USES_CLAIMED_MAGNITUDE)

    def test_the_alert_declares_it_does_not_read_magnitude(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        self.assertFalse(alert["uses_claimed_magnitude"])
        self.assertIn("claim", alert["magnitude_reason"].lower())

    def test_a_claimed_magnitude_in_the_event_is_ignored(self):
        """A loud claim on a quiet event type must not fire."""
        loud = dict(event("minor_pr"), magnitude=0.99)
        alert = event_impact_alert(loud, memories())
        self.assertEqual(alert["verdict"], EVENT_IMPACT_LOW)
        self.assertFalse(
            alert["fired"],
            "a 0.99 claimed magnitude fired on an event type whose realized "
            "history is quiet",
        )

    def test_a_quiet_claim_on_a_loud_type_still_fires(self):
        quiet = dict(event("guidance_cut"), magnitude=0.01)
        alert = event_impact_alert(quiet, memories())
        self.assertEqual(alert["verdict"], EVENT_IMPACT_HIGH)
        self.assertTrue(alert["fired"])

    def test_a_smuggled_magnitude_field_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["claimed_magnitude"] = 0.9
        self.assertTrue(
            impact_alert_problems(alert),
            "a magnitude smuggled into the payload passed the contract check",
        )

    def test_an_alert_reading_magnitude_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["uses_claimed_magnitude"] = True
        self.assertTrue(impact_alert_problems(alert))


class TheDecidingMeasurementTest(unittest.TestCase):
    """Event type has a measurable, large spread in realized impact."""

    def test_event_types_differ_widely_in_realized_impact(self):
        pool = memories()
        medians = {}
        for event_type in TYPES:
            impact = realized_impact(pool, event_type)
            self.assertTrue(impact["measured"], event_type)
            medians[event_type] = impact["median_move"]
        loudest = max(medians.values())
        quietest = min(medians.values())
        self.assertGreater(
            loudest / quietest,
            4.0,
            f"event types span only {loudest / quietest:.1f}x in realized "
            f"impact ({medians}); A3 rests on type being informative, and "
            f"must be re-derived rather than kept if that has vanished",
        )

    def test_loud_types_fire_and_quiet_types_do_not(self):
        pool = memories()
        for event_type in ("earnings_beat", "guidance_cut"):
            alert = event_impact_alert(event(event_type), pool)
            self.assertEqual(alert["verdict"], EVENT_IMPACT_HIGH, event_type)
            self.assertTrue(alert["fired"])
            self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        for event_type in ("analyst_note", "minor_pr"):
            alert = event_impact_alert(event(event_type), pool)
            self.assertEqual(alert["verdict"], EVENT_IMPACT_LOW, event_type)
            self.assertFalse(alert["fired"])
            self.assertIsNone(alert["severity"])

    def test_the_threshold_separates_the_groups_with_room(self):
        pool = memories()
        loud = min(
            realized_impact(pool, t)["median_move"]
            for t in ("earnings_beat", "guidance_cut")
        )
        quiet = max(
            realized_impact(pool, t)["median_move"]
            for t in ("analyst_note", "minor_pr")
        )
        self.assertLess(quiet, EVENT_IMPACT_MIN_MEDIAN_MOVE)
        self.assertGreater(loud, EVENT_IMPACT_MIN_MEDIAN_MOVE)

    def test_a_high_impact_verdict_below_the_threshold_is_caught(self):
        alert = event_impact_alert(event("minor_pr"), memories())
        alert["verdict"] = EVENT_IMPACT_HIGH
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(impact_alert_problems(alert))


class ImpactIsNotDirectionTest(unittest.TestCase):
    """A symmetric history must not cancel itself to zero."""

    def test_the_measure_is_absolute(self):
        self.assertTrue(EVENT_IMPACT_USES_ABSOLUTE_MOVE)

    def test_a_symmetric_history_is_still_high_impact(self):
        pool = []
        for index in range(20):
            move = 0.06 if index % 2 == 0 else -0.06
            pool.append(
                EventMemory(
                    event_id=f"s{index}",
                    ticker="NVDA",
                    published_time="2025-01-10",
                    event_type="volatile_event",
                    direction="positive",
                    response={"20d": {"abnormal_return": move}},
                    attribution={"20d": "event_associated"},
                    provenance="observed",
                )
            )
        impact = realized_impact(pool, "volatile_event")
        self.assertAlmostEqual(impact["median_move"], 0.06, places=6)
        alert = event_impact_alert(event("volatile_event"), pool)
        self.assertEqual(
            alert["verdict"],
            EVENT_IMPACT_HIGH,
            "a history of ±6% moves averaged to zero and read as harmless",
        )

    def test_a_negative_median_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["impact"]["median_move"] = -0.05
        self.assertTrue(impact_alert_problems(alert))


class UnknownIsNotLowTest(unittest.TestCase):
    """An unrated event type is not a quiet one."""

    def test_an_unseen_type_is_unknown_not_low(self):
        alert = event_impact_alert(event("never_seen_before"), memories())
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)
        self.assertNotEqual(alert["verdict"], EVENT_IMPACT_LOW)

    def test_an_unknown_type_fires_for_a_human_look(self):
        self.assertTrue(EVENT_IMPACT_UNKNOWN_IS_NOT_LOW)
        alert = event_impact_alert(event("never_seen_before"), memories())
        self.assertTrue(
            alert["fired"],
            "an event type the system has never seen was silenced",
        )
        self.assertEqual(alert["severity"], ALERT_SEVERITY_INFO)

    def test_an_unknown_type_carries_no_statistics(self):
        alert = event_impact_alert(event("never_seen_before"), memories())
        for field in ("median_move", "p90_move", "event_associated_share"):
            self.assertIsNone(
                alert["impact"][field],
                f"{field} was reported for an unrated event type",
            )
        self.assertFalse(alert["impact"]["measured"])

    def test_the_analog_floor_is_e6s(self):
        self.assertEqual(EVENT_IMPACT_MIN_ANALOGS, EVENT_MEMORY_MIN_ANALOGS)

    def test_just_below_the_floor_is_unknown(self):
        pool = [
            EventMemory(
                event_id=f"f{i}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type="rare_event",
                direction="positive",
                response={"20d": {"abnormal_return": 0.08}},
                attribution={"20d": "event_associated"},
                provenance="observed",
            )
            for i in range(EVENT_IMPACT_MIN_ANALOGS - 1)
        ]
        alert = event_impact_alert(event("rare_event"), pool)
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)

    def test_at_the_floor_it_is_rated(self):
        pool = [
            EventMemory(
                event_id=f"f{i}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type="rare_event",
                direction="positive",
                response={"20d": {"abnormal_return": 0.08}},
                attribution={"20d": "event_associated"},
                provenance="observed",
            )
            for i in range(EVENT_IMPACT_MIN_ANALOGS)
        ]
        alert = event_impact_alert(event("rare_event"), pool)
        self.assertEqual(alert["verdict"], EVENT_IMPACT_HIGH)

    def test_an_unknown_carrying_statistics_is_caught(self):
        alert = event_impact_alert(event("never_seen_before"), memories())
        alert["impact"]["median_move"] = 0.0
        self.assertTrue(impact_alert_problems(alert))

    def test_a_silenced_unknown_is_caught(self):
        alert = event_impact_alert(event("never_seen_before"), memories())
        alert["fired"] = False
        alert["severity"] = None
        self.assertTrue(impact_alert_problems(alert))


class NoEventTest(unittest.TestCase):
    def test_no_event_is_its_own_verdict(self):
        alert = event_impact_alert(None, memories())
        self.assertEqual(alert["verdict"], EVENT_IMPACT_NO_EVENT)
        self.assertFalse(alert["fired"])

    def test_an_untyped_event_is_no_event(self):
        alert = event_impact_alert({"ticker": "NVDA"}, memories())
        self.assertEqual(alert["verdict"], EVENT_IMPACT_NO_EVENT)

    def test_no_event_carries_no_impact(self):
        alert = event_impact_alert(None, memories())
        self.assertIsNone(alert["impact"])
        self.assertEqual(impact_alert_problems(alert), [])

    def test_a_no_event_that_fired_is_caught(self):
        alert = event_impact_alert(None, memories())
        alert["fired"] = True
        self.assertTrue(impact_alert_problems(alert))


class ContractTest(unittest.TestCase):
    def test_clean_alerts_have_no_problems(self):
        pool = memories()
        for event_type in list(TYPES) + ["never_seen_before"]:
            alert = event_impact_alert(event(event_type), pool)
            self.assertEqual(impact_alert_problems(alert), [], event_type)
            self.assertEqual(alert["version"], EVENT_IMPACT_ALERT_VERSION)

    def test_the_alert_does_not_block_trades(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(EVENT_IMPACT_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["blocks_trades"] = True
        self.assertTrue(impact_alert_problems(alert))

    def test_an_unknown_verdict_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["verdict"] = "PROBABLY"
        self.assertTrue(impact_alert_problems(alert))

    def test_a_reasonless_alert_is_caught(self):
        alert = event_impact_alert(event("earnings_beat"), memories())
        alert["reason"] = "  "
        self.assertTrue(impact_alert_problems(alert))

    def test_a_missing_event_type_is_refused(self):
        with self.assertRaises(EventImpactAlertError):
            realized_impact(memories(), "")

    def test_a_non_mapping_event_is_refused(self):
        with self.assertRaises(EventImpactAlertError):
            event_impact_alert("earnings", memories())

    def test_an_impossible_threshold_is_refused(self):
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(EventImpactAlertError):
                event_impact_alert(event("earnings_beat"), memories(), threshold=bad)

    def test_an_empty_memory_pool_rates_nothing(self):
        alert = event_impact_alert(event("earnings_beat"), [])
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)
        self.assertEqual(impact_alert_problems(alert), [])


class RenderTest(unittest.TestCase):
    def test_the_verdict_and_median_are_rendered(self):
        alert = event_impact_alert(event("guidance_cut"), memories())
        text = "\n".join(render_impact_alert(alert))
        self.assertIn(EVENT_IMPACT_HIGH, text)
        self.assertIn("FIRED", text)

    def test_an_unrated_type_renders_as_absent(self):
        alert = event_impact_alert(event("never_seen_before"), memories())
        line = render_impact_alert(alert)[0]
        self.assertIn("median —", line)
        self.assertNotIn("0.00%", line)

    def test_lines_stay_readable(self):
        alert = event_impact_alert(event("guidance_cut"), memories())
        for line in render_impact_alert(alert):
            self.assertLess(len(line), 250)


if __name__ == "__main__":
    unittest.main()
