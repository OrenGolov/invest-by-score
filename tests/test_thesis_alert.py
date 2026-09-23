"""A5 thesis break alert tests.

The central claim under test: the score is a sum, and a sum hides a reversal.
These tests build their attributions in-process rather than reading any data
file, so they mean the same thing on a clean clone.
"""

from __future__ import annotations

import random
import unittest

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    ATTRIBUTION_BUCKETS,
    ATTRIBUTION_SUPPORT_THRESHOLD,
    THESIS_ALERT_BLOCKS_TRADES,
    THESIS_ALERT_VERSION,
    THESIS_ALERT_WATCHES_BUCKETS,
    THESIS_BREAK_APPEARED,
    THESIS_BREAK_CARRIER,
    THESIS_BREAK_DISAPPEARED,
    THESIS_BREAK_NONE,
    THESIS_BREAK_NOT_EVALUATED,
    THESIS_BREAK_REVERSAL,
    THESIS_BREAK_WITHDRAWN,
    THESIS_REVERSAL_IS_MATERIAL,
    THESIS_SUPPORT_THRESHOLD,
    THESIS_ZERO_IS_NOT_OPPOSITION,
)
from core.thesis_alert import (
    STANCE_NEUTRAL,
    STANCE_OPPOSES,
    STANCE_SUPPORTS,
    STANCE_UNMEASURED,
    ThesisAlertError,
    carrier_of,
    read_buckets,
    render_thesis_alert,
    stance_of,
    thesis_alert_problems,
    thesis_break_alert,
)


def attribution(operational, narrative, macro, *, measured=None, ticker="NVDA"):
    measured = measured or {}
    buckets = {}
    for name, total in (
        ("operational", operational),
        ("narrative", narrative),
        ("macro_shock", macro),
    ):
        entry = {"total": total}
        if name in measured:
            entry["measured"] = measured[name]
        buckets[name] = entry
    return {
        "buckets": buckets,
        "score": sum(v for v in (operational, narrative, macro) if v is not None),
        "ticker": ticker,
    }


class TheDecidingMeasurementTest(unittest.TestCase):
    """A sum hides a reversal."""

    def test_a_flat_score_often_hides_a_bucket_reversal(self):
        rng = random.Random(11)
        flat = 0
        reversed_count = 0
        for _ in range(20000):
            before = [rng.uniform(-2, 2) for _ in range(3)]
            after = [rng.uniform(-2, 2) for _ in range(3)]
            if abs(sum(after) - sum(before)) >= ATTRIBUTION_SUPPORT_THRESHOLD:
                continue
            flat += 1
            stances_before = [stance_of(v) for v in before]
            stances_after = [stance_of(v) for v in after]
            if any(
                {a, b} == {STANCE_SUPPORTS, STANCE_OPPOSES}
                for a, b in zip(stances_before, stances_after)
            ):
                reversed_count += 1
        self.assertGreater(flat, 200, "too few flat-score pairs to measure")
        share = reversed_count / flat
        self.assertGreater(
            share,
            0.30,
            f"only {share:.0%} of flat-score periods contained a bucket "
            f"reversal; A5's design rests on that being common, and must be "
            f"re-derived rather than kept if it has vanished",
        )

    def test_the_concrete_offsetting_case_fires(self):
        before = attribution(1.40, -0.10, 0.20)
        after = attribution(-1.20, 2.50, 0.20)
        alert = thesis_break_alert(before, after)
        self.assertAlmostEqual(alert["score_delta"], 0.0, places=6)
        self.assertEqual(
            alert["kind"],
            THESIS_BREAK_REVERSAL,
            "a thesis that inverted from operational to narrative, at an "
            "identical score, did not fire",
        )
        self.assertTrue(alert["fired"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        self.assertEqual(alert["reversals"], ["operational"])

    def test_the_alert_watches_buckets_not_the_score(self):
        self.assertTrue(THESIS_ALERT_WATCHES_BUCKETS)
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        self.assertTrue(alert["watches_buckets"])

    def test_an_alert_not_watching_buckets_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        alert["watches_buckets"] = False
        self.assertTrue(thesis_alert_problems(alert))

    def test_a_silenced_reversal_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        alert["kind"] = THESIS_BREAK_NONE
        alert["fired"] = False
        alert["severity"] = None
        self.assertTrue(
            thesis_alert_problems(alert),
            "a reversal reported as NONE passed — that is the case A5 exists "
            "to catch",
        )


class StanceTest(unittest.TestCase):
    def test_the_threshold_is_w1s(self):
        self.assertEqual(THESIS_SUPPORT_THRESHOLD, ATTRIBUTION_SUPPORT_THRESHOLD)

    def test_stances_follow_the_threshold(self):
        self.assertEqual(stance_of(THESIS_SUPPORT_THRESHOLD + 0.01), STANCE_SUPPORTS)
        self.assertEqual(stance_of(-THESIS_SUPPORT_THRESHOLD - 0.01), STANCE_OPPOSES)
        self.assertEqual(stance_of(0.0), STANCE_NEUTRAL)
        self.assertEqual(stance_of(THESIS_SUPPORT_THRESHOLD), STANCE_NEUTRAL)

    def test_unmeasured_evidence_has_no_stance(self):
        self.assertTrue(THESIS_ZERO_IS_NOT_OPPOSITION)
        self.assertEqual(stance_of(0.0, measured=False), STANCE_UNMEASURED)
        self.assertEqual(stance_of(None), STANCE_UNMEASURED)

    def test_a_quiet_feed_is_not_a_thesis_break(self):
        """Absence of evidence is not contradiction."""
        before = attribution(1.40, 0.90, 0.20)
        after = attribution(1.40, 0.0, 0.20, measured={"narrative": False})
        alert = thesis_break_alert(before, after)
        self.assertNotEqual(
            alert["kind"],
            THESIS_BREAK_REVERSAL,
            "a narrative feed going quiet was read as the narrative opposing "
            "the thesis",
        )
        self.assertEqual(
            alert["after"]["narrative"]["stance"], STANCE_UNMEASURED
        )

    def test_a_genuine_zero_is_still_a_measurement(self):
        buckets = read_buckets(attribution(1.40, 0.0, 0.20))
        self.assertEqual(buckets["narrative"]["stance"], STANCE_NEUTRAL)
        self.assertTrue(buckets["narrative"]["measured"])

    def test_unmeasured_evidence_given_a_stance_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.90, 0.20),
            attribution(1.40, 0.0, 0.20, measured={"narrative": False}),
        )
        alert["after"]["narrative"]["stance"] = STANCE_OPPOSES
        self.assertTrue(thesis_alert_problems(alert))


class CarrierTest(unittest.TestCase):
    def test_the_carrier_is_the_largest_supporting_bucket(self):
        buckets = read_buckets(attribution(1.40, 2.50, 0.20))
        self.assertEqual(carrier_of(buckets), "narrative")

    def test_no_supporting_bucket_has_no_carrier(self):
        self.assertIsNone(carrier_of(read_buckets(attribution(0.1, 0.1, 0.1))))

    def test_a_changed_carrier_fires(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(0.10, 1.40, 0.20)
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_CARRIER)
        self.assertEqual(alert["previous_carrier"], "operational")
        self.assertEqual(alert["current_carrier"], "narrative")
        self.assertTrue(alert["fired"])

    def test_losing_every_carrier_is_a_carrier_change(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(0.10, 0.10, 0.20)
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_CARRIER)
        self.assertIsNone(alert["current_carrier"])
        self.assertEqual(alert["withdrawn"], ["operational"])

    def test_a_carrier_change_with_the_same_carrier_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(0.10, 1.40, 0.20)
        )
        alert["current_carrier"] = alert["previous_carrier"]
        self.assertTrue(thesis_alert_problems(alert))


class WithdrawnTest(unittest.TestCase):
    def test_withdrawal_with_a_surviving_carrier_is_reported(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.90, 0.20), attribution(0.10, 0.90, 0.20)
        )
        self.assertIn("operational", alert["withdrawn"])
        self.assertTrue(alert["fired"])

    def test_a_pure_withdrawal_is_its_own_kind(self):
        """Support withdrawn while the carrier is unchanged."""
        alert = thesis_break_alert(
            attribution(2.00, 0.90, 0.20), attribution(2.00, 0.10, 0.20)
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_WITHDRAWN)
        self.assertEqual(alert["withdrawn"], ["narrative"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_INFO)

    def test_a_withdrawal_with_no_bucket_named_is_caught(self):
        alert = thesis_break_alert(
            attribution(2.00, 0.90, 0.20), attribution(2.00, 0.10, 0.20)
        )
        alert["withdrawn"] = []
        self.assertTrue(thesis_alert_problems(alert))


class AvailabilityTest(unittest.TestCase):
    def test_a_thesis_becoming_attributable_is_an_appearance(self):
        alert = thesis_break_alert(
            {"buckets": {}, "score": None}, attribution(1.40, 0.10, 0.20)
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_APPEARED)
        self.assertIsNone(alert["before"])
        self.assertTrue(alert["fired"])

    def test_a_thesis_going_away_is_a_disappearance(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), {"buckets": {}, "score": None}
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_DISAPPEARED)
        self.assertIsNone(alert["after"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        self.assertIn("did not turn", alert["reason"])

    def test_both_absent_is_not_evaluated(self):
        alert = thesis_break_alert(
            {"buckets": {}, "score": None}, {"buckets": {}, "score": None}
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_NOT_EVALUATED)
        self.assertFalse(alert["fired"])

    def test_a_first_observation_is_not_a_break(self):
        alert = thesis_break_alert(None, attribution(1.40, 0.10, 0.20))
        self.assertEqual(alert["kind"], THESIS_BREAK_NOT_EVALUATED)
        self.assertFalse(alert["fired"])

    def test_an_appearance_with_a_previous_attribution_is_caught(self):
        alert = thesis_break_alert(
            {"buckets": {}, "score": None}, attribution(1.40, 0.10, 0.20)
        )
        alert["before"] = read_buckets(attribution(1.0, 0.0, 0.0))
        self.assertTrue(thesis_alert_problems(alert))


class ContractTest(unittest.TestCase):
    def test_clean_alerts_have_no_problems(self):
        for previous, current in (
            (attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)),
            (attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)),
            (attribution(1.40, 0.10, 0.20), attribution(0.10, 1.40, 0.20)),
            (attribution(2.00, 0.90, 0.20), attribution(2.00, 0.10, 0.20)),
            ({"buckets": {}, "score": None}, attribution(1.40, 0.10, 0.20)),
            (attribution(1.40, 0.10, 0.20), {"buckets": {}, "score": None}),
            (None, attribution(1.40, 0.10, 0.20)),
        ):
            alert = thesis_break_alert(previous, current)
            self.assertEqual(thesis_alert_problems(alert), [], alert["kind"])
            self.assertEqual(alert["version"], THESIS_ALERT_VERSION)

    def test_an_unchanged_thesis_is_quiet(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        self.assertEqual(alert["kind"], THESIS_BREAK_NONE)
        self.assertFalse(alert["fired"])
        self.assertIsNone(alert["severity"])

    def test_the_alert_does_not_block_trades(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(THESIS_ALERT_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        alert["blocks_trades"] = True
        self.assertTrue(thesis_alert_problems(alert))

    def test_a_reversal_at_info_severity_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        alert["severity"] = ALERT_SEVERITY_INFO
        self.assertTrue(thesis_alert_problems(alert))

    def test_a_mismatched_reversal_list_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        alert["reversals"] = ["narrative"]
        self.assertTrue(thesis_alert_problems(alert))

    def test_an_unknown_bucket_is_caught(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
        )
        alert["withdrawn"] = ["astrology"]
        self.assertTrue(thesis_alert_problems(alert))

    def test_a_missing_current_attribution_is_refused(self):
        with self.assertRaises(ThesisAlertError):
            thesis_break_alert(attribution(1.0, 0.0, 0.0), None)

    def test_a_non_mapping_attribution_is_refused(self):
        with self.assertRaises(ThesisAlertError):
            thesis_break_alert("bullish", attribution(1.0, 0.0, 0.0))

    def test_an_impossible_threshold_is_refused(self):
        with self.assertRaises(ThesisAlertError):
            thesis_break_alert(
                attribution(1.0, 0.0, 0.0), attribution(1.0, 0.0, 0.0), threshold=0.0
            )

    def test_the_reversal_config_is_material(self):
        self.assertTrue(THESIS_REVERSAL_IS_MATERIAL)


class RenderTest(unittest.TestCase):
    def test_every_bucket_is_rendered(self):
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        text = "\n".join(render_thesis_alert(alert))
        for name in ATTRIBUTION_BUCKETS:
            self.assertIn(name, text)

    def test_the_carrier_change_is_rendered(self):
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        text = "\n".join(render_thesis_alert(alert))
        self.assertIn("operational", text)
        self.assertIn("narrative", text)
        self.assertIn("FIRED", text)

    def test_an_absent_attribution_renders_as_absent(self):
        alert = thesis_break_alert(
            attribution(1.40, 0.10, 0.20), {"buckets": {}, "score": None}
        )
        self.assertIn("—", "\n".join(render_thesis_alert(alert)))

    def test_lines_stay_readable(self):
        alert = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        for line in render_thesis_alert(alert):
            self.assertLess(len(line), 250)


if __name__ == "__main__":
    unittest.main()
