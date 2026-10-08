"""L6 drift-detection tests.

The behaviour under test is that FOUR detectors answer four different
questions. MEASURED, the feature-distribution detector catches one of the four
scenarios; the other three are invisible to it because the features do not
move — the world does.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    DRIFT_ALERT,
    DRIFT_CALIBRATION,
    DRIFT_FEATURE,
    DRIFT_MIN_WINDOW,
    DRIFT_NOT_EVALUATED,
    DRIFT_PSI_ALERT,
    DRIFT_RELATIONSHIP,
    DRIFT_RESPONSE,
    DRIFT_RESPONSE_LOG2_ALERT,
    DRIFT_RESPONSE_LOG2_WARN,
    DRIFT_STABLE,
    DRIFT_TRIGGERS_RETRAIN,
    DRIFT_TYPES,
    DRIFT_VERDICTS,
    DRIFT_WARN,
)
from core.drift_detection import (
    DriftDetectionError,
    calibration_drift,
    drift_problems,
    drift_report,
    event_response_drift,
    feature_drift,
    population_stability_index,
    quantile_edges,
    relationship_drift,
    render_drift,
)

WINDOW = max(int(DRIFT_MIN_WINDOW), 300)


def normal(count=WINDOW, mean=0.0, sd=1.0, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(count)]


class FeatureDriftTests(unittest.TestCase):
    def test_a_shifted_feature_alerts(self):
        self.assertEqual(
            feature_drift(normal(seed=1), normal(mean=1.2, seed=2))["verdict"],
            DRIFT_ALERT,
        )

    def test_an_unshifted_feature_is_stable(self):
        self.assertEqual(
            feature_drift(normal(seed=3), normal(seed=4))["verdict"], DRIFT_STABLE
        )

    def test_a_thin_window_refuses_to_judge(self):
        # MEASURED: PSI > 0.25 fires on 77% of CLEAN comparisons at window 50.
        result = feature_drift(normal(50, seed=5), normal(50, seed=6))
        self.assertEqual(result["verdict"], DRIFT_NOT_EVALUATED)

    def test_the_window_floor_supports_the_threshold(self):
        self.assertGreaterEqual(DRIFT_MIN_WINDOW, 250)

    def test_clean_windows_rarely_alert_at_the_floor(self):
        alerts = sum(
            1
            for seed in range(30)
            if feature_drift(
                normal(seed=100 + seed), normal(seed=500 + seed)
            )["verdict"]
            == DRIFT_ALERT
        )
        self.assertLessEqual(alerts, 2, f"{alerts}/30 clean windows alerted")

    def test_a_non_numeric_value_is_refused(self):
        with self.assertRaises(DriftDetectionError):
            feature_drift(["x"] * WINDOW, normal())

    def test_an_infinite_value_is_refused(self):
        with self.assertRaises(DriftDetectionError):
            feature_drift([math.inf] * WINDOW, normal())


class QuantileBinningTests(unittest.TestCase):
    """A fixed scale is blind to any feature that does not live on it."""

    def test_a_narrow_range_shift_is_seen(self):
        # MEASURED: fixed [0,10] bins score 0.0000 on this 3.2-sigma shift.
        psi = population_stability_index(
            normal(mean=0.02, sd=0.05, seed=7), normal(mean=0.18, sd=0.05, seed=8)
        )
        self.assertGreater(psi, DRIFT_PSI_ALERT)

    def test_an_rsi_scaled_shift_is_seen(self):
        psi = population_stability_index(
            normal(mean=45, sd=12, seed=9), normal(mean=72, sd=12, seed=10)
        )
        self.assertGreater(psi, DRIFT_PSI_ALERT)

    def test_identical_samples_score_near_zero(self):
        values = normal(seed=11)
        self.assertLess(population_stability_index(values, values), 0.01)

    def test_edges_come_from_the_reference(self):
        edges = quantile_edges(list(range(100)), bins=4)
        self.assertEqual(len(edges), 3)
        self.assertLess(edges[0], edges[-1])


class RelationshipDriftTests(unittest.TestCase):
    """Features identical, the world different."""

    def setUp(self):
        rng = random.Random(21)
        self.xs = [rng.gauss(0, 1) for _ in range(WINDOW)]
        self.before = [
            1 if rng.random() < 1 / (1 + math.exp(-0.9 * v)) else 0 for v in self.xs
        ]
        self.after = [
            1 if rng.random() < 1 / (1 + math.exp(+0.9 * v)) else 0 for v in self.xs
        ]

    def test_a_sign_flip_alerts(self):
        result = relationship_drift(self.xs, self.before, self.xs, self.after)
        self.assertEqual(result["verdict"], DRIFT_ALERT)
        self.assertTrue(result["sign_flip"])

    def test_the_feature_detector_is_blind_to_it(self):
        # This is why relationship drift is a separate question.
        self.assertEqual(
            feature_drift(self.xs, self.xs)["verdict"], DRIFT_STABLE
        )

    def test_an_unchanged_relationship_is_stable(self):
        self.assertEqual(
            relationship_drift(self.xs, self.before, self.xs, self.before)["verdict"],
            DRIFT_STABLE,
        )

    def test_mismatched_pairs_are_refused(self):
        with self.assertRaises(DriftDetectionError):
            relationship_drift(self.xs, self.before[:-1], self.xs, self.after)

    def test_a_thin_window_refuses_to_judge(self):
        self.assertEqual(
            relationship_drift([1.0, 2.0], [0, 1], [1.0, 2.0], [1, 0])["verdict"],
            DRIFT_NOT_EVALUATED,
        )


class CalibrationDriftTests(unittest.TestCase):
    """Ranking intact, probabilities wrong."""

    def setUp(self):
        rng = random.Random(31)
        xs = [rng.gauss(0, 1) for _ in range(WINDOW)]
        self.probabilities = [1 / (1 + math.exp(-0.9 * v)) for v in xs]
        self.outcomes = [1 if rng.random() < p else 0 for p in self.probabilities]

    def test_a_calibrated_model_is_stable(self):
        self.assertEqual(
            calibration_drift(self.probabilities, self.outcomes)["verdict"],
            DRIFT_STABLE,
        )

    def test_inflated_probabilities_are_caught(self):
        inflated = [min(0.999, p * 1.6) for p in self.probabilities]
        result = calibration_drift(inflated, self.outcomes)
        self.assertIn(result["verdict"], (DRIFT_WARN, DRIFT_ALERT))
        self.assertEqual(result["direction"], "OVERCONFIDENT")

    def test_deflated_probabilities_are_named_correctly(self):
        deflated = [p * 0.4 for p in self.probabilities]
        result = calibration_drift(deflated, self.outcomes)
        self.assertEqual(result["direction"], "UNDERCONFIDENT")

    def test_a_non_probability_is_refused(self):
        with self.assertRaises(DriftDetectionError):
            calibration_drift([1.6] * WINDOW, self.outcomes)

    def test_a_thin_window_refuses_to_judge(self):
        self.assertEqual(
            calibration_drift([0.5, 0.6], [1, 0])["verdict"], DRIFT_NOT_EVALUATED
        )


class EventResponseDriftTests(unittest.TestCase):
    """A percentage ratio is the wrong scale, and the gate caught it."""

    def setUp(self):
        rng = random.Random(41)
        self.strong = [abs(rng.gauss(0.04, 0.02)) + 0.001 for _ in range(WINDOW)]

    def test_a_vanished_response_alerts(self):
        result = event_response_drift(self.strong, [0.0] * WINDOW)
        self.assertEqual(result["verdict"], DRIFT_ALERT)
        self.assertTrue(result["response_vanished"])

    def test_halving_and_doubling_read_the_same(self):
        # A decline is bounded at -100% while an increase is unbounded, so a
        # symmetric rule on a PERCENTAGE is not symmetric at all.
        halved = event_response_drift(self.strong, [v / 2 for v in self.strong])
        doubled = event_response_drift(self.strong, [v * 2 for v in self.strong])
        self.assertEqual(halved["verdict"], doubled["verdict"])
        self.assertAlmostEqual(
            abs(halved["log2_factor"]), abs(doubled["log2_factor"]), places=6
        )

    def test_a_fourfold_weakening_alerts(self):
        self.assertEqual(
            event_response_drift(self.strong, [v / 4 for v in self.strong])["verdict"],
            DRIFT_ALERT,
        )

    def test_an_unchanged_response_is_stable(self):
        self.assertEqual(
            event_response_drift(self.strong, self.strong)["verdict"], DRIFT_STABLE
        )

    def test_a_weakening_is_named(self):
        result = event_response_drift(self.strong, [v / 3 for v in self.strong])
        self.assertEqual(result["direction"], "WEAKER")

    def test_a_zero_reference_cannot_be_compared(self):
        result = event_response_drift([0.0] * WINDOW, self.strong)
        self.assertEqual(result["verdict"], DRIFT_NOT_EVALUATED)

    def test_the_bounds_ascend(self):
        self.assertLess(DRIFT_RESPONSE_LOG2_WARN, DRIFT_RESPONSE_LOG2_ALERT)


class ReportTests(unittest.TestCase):
    def test_every_type_appears_even_when_unsupplied(self):
        # A three-of-four scan reporting clean is a false assurance.
        report = drift_report(features={"x": (normal(seed=51), normal(seed=52))})
        self.assertEqual(set(report["by_type"]), set(DRIFT_TYPES))
        self.assertEqual(report["by_type"][DRIFT_RELATIONSHIP], DRIFT_NOT_EVALUATED)

    def test_an_empty_report_is_contract_clean(self):
        report = drift_report()
        self.assertEqual(drift_problems(report), [])
        self.assertEqual(report["worst"], DRIFT_NOT_EVALUATED)

    def test_a_full_report_surfaces_the_worst_verdict(self):
        rng = random.Random(61)
        xs = [rng.gauss(0, 1) for _ in range(WINDOW)]
        report = drift_report(features={"x": (xs, normal(mean=1.5, seed=62))})
        self.assertEqual(report["worst"], DRIFT_ALERT)

    def test_drift_never_triggers_a_retrain(self):
        self.assertFalse(DRIFT_TRIGGERS_RETRAIN)
        self.assertFalse(drift_report()["triggers_retrain"])

    def test_render_returns_one_line_per_finding(self):
        report = drift_report(features={"x": (normal(seed=71), normal(seed=72))})
        self.assertEqual(len(render_drift(report)), len(report["findings"]))

    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(DRIFT_VERDICTS[0], DRIFT_NOT_EVALUATED)
        self.assertEqual(DRIFT_VERDICTS[-1], DRIFT_ALERT)

    def test_not_evaluated_is_not_stable(self):
        self.assertNotEqual(DRIFT_NOT_EVALUATED, DRIFT_STABLE)

    def test_four_types_are_declared(self):
        self.assertEqual(len(DRIFT_TYPES), 4)


if __name__ == "__main__":
    unittest.main()
