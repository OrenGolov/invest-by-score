"""Sprint V7: monitoring foundations - pure run-health metrics, no dashboard.

Hermetic: synthetic decisions only (no network, no wall-clock, no I/O).
"""

from __future__ import annotations

import unittest

from core.monitoring import (
    MONITORING_VERSION,
    confidence_drift,
    label_coverage,
    monitoring_snapshot,
    score_drift_psi,
    stale_data_rate,
    veto_rate_by_rule,
)


def _ok_decision(score=6.0, confidence=0.7, statuses=None, rules=(), label_up=True):
    return {
        "score": score,
        "confidence": confidence,
        "agent_statuses": dict(statuses) if statuses is not None else {"market_data": "OK"},
        "veto": {"rule_ids": list(rules)},
        "labels": {"horizons": {"20d": {"label_up": label_up}}},
    }


class ScoreDriftTests(unittest.TestCase):
    def test_identical_windows_show_no_change(self):
        scores = [5.0, 5.5, 6.0, 6.5] * 10
        result = score_drift_psi(scores, list(scores))
        self.assertEqual(result["verdict"], "no_significant_change")
        self.assertAlmostEqual(result["psi"], 0.0, places=6)

    def test_seeded_shift_is_detected(self):
        reference = [5.0, 5.5, 6.0, 6.5] * 10
        current = [8.0, 8.5, 9.0, 9.5] * 10
        result = score_drift_psi(reference, current)
        self.assertEqual(result["verdict"], "significant_shift")
        self.assertGreater(result["psi"], 0.25)

    def test_small_windows_degrade_explicitly(self):
        result = score_drift_psi([5.0] * 10, [9.0] * 10)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertIsNone(result["psi"])

    def test_corrupt_scores_raise(self):
        with self.assertRaises(ValueError):
            score_drift_psi([5.0] * 30, [float("nan")] * 30)
        with self.assertRaises(ValueError):
            score_drift_psi([5.0] * 30, [11.0] * 30)


class ConfidenceDriftTests(unittest.TestCase):
    def test_mean_delta_is_signed(self):
        result = confidence_drift([0.8] * 40, [0.5] * 40)
        self.assertEqual(result["verdict"], "computed")
        self.assertAlmostEqual(result["delta"], -0.3, places=6)

    def test_small_windows_degrade_explicitly(self):
        result = confidence_drift([0.8] * 5, [0.5] * 5)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertIsNone(result["delta"])


class StaleDataRateTests(unittest.TestCase):
    def test_stale_fraction_counts_status_bearing_decisions(self):
        decisions = (
            [_ok_decision(statuses={"market_data": "STALE"})] * 10
            + [_ok_decision()] * 30
        )
        result = stale_data_rate(decisions)
        self.assertEqual(result["verdict"], "computed")
        self.assertAlmostEqual(result["rate"], 0.25, places=6)

    def test_missing_statuses_degrade_not_zero_fill(self):
        decisions = [_ok_decision() for _ in range(10)]
        for decision in decisions:
            del decision["agent_statuses"]
        result = stale_data_rate(decisions)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertIsNone(result["rate"])
        self.assertEqual(result["unknown_count"], 10)


class VetoRateTests(unittest.TestCase):
    def test_per_rule_fractions_over_known_decisions(self):
        decisions = (
            [_ok_decision(rules=["score_below_threshold"])] * 10
            + [_ok_decision(rules=[])] * 30
        )
        result = veto_rate_by_rule(decisions)
        self.assertEqual(result["verdict"], "computed")
        self.assertAlmostEqual(result["rates"]["score_below_threshold"], 0.25, places=6)

    def test_missing_veto_metadata_is_unknown_not_clean(self):
        decisions = [_ok_decision() for _ in range(10)]
        for decision in decisions:
            del decision["veto"]
        result = veto_rate_by_rule(decisions)
        self.assertEqual(result["verdict"], "insufficient_data")
        self.assertEqual(result["rates"], {})
        self.assertEqual(result["unknown_count"], 10)


class LabelCoverageTests(unittest.TestCase):
    def test_pending_labels_are_uncovered(self):
        decisions = (
            [_ok_decision(label_up=True)] * 30
            + [_ok_decision(label_up=None)] * 10
        )
        result = label_coverage(decisions)
        self.assertEqual(result["verdict"], "computed")
        self.assertAlmostEqual(result["rate"], 0.75, places=6)


class SnapshotTests(unittest.TestCase):
    def test_full_snapshot_is_versioned_and_ok(self):
        decisions = [_ok_decision(score=5.0 + (i % 4) * 0.5) for i in range(80)]
        snapshot = monitoring_snapshot(decisions)
        self.assertEqual(snapshot["monitoring_version"], MONITORING_VERSION)
        self.assertEqual(snapshot["status"], "ok")
        self.assertEqual(snapshot["degraded_metrics"], [])

    def test_empty_run_degrades_every_metric(self):
        snapshot = monitoring_snapshot([])
        self.assertEqual(snapshot["status"], "degraded")
        self.assertEqual(
            sorted(snapshot["degraded_metrics"]),
            ["confidence_drift", "label_coverage", "score_drift", "stale_data", "veto_rates"],
        )


if __name__ == "__main__":
    unittest.main()

