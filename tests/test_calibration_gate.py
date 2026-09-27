"""X2 calibration gate tests.

The measurements these tests defend:

* In-sample isotonic ECE is 0.0000 for all 8 estimators — structurally, because
  the map collapses to two knots and each bin reproduces its own base rate.
* Refit on the first 60 and scored on the held-out 60, the honest ECE is
  0.166-0.189, caused by BASE-RATE DRIFT (63.3% up-days in train, 46.7% in the
  holdout).
* A perfectly calibrated predictor at n=60 still posts an ECE up to 0.1333, so
  200 observations are required before an ECE can decide anything.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.calibration_gate import (
    CAL_REASON_ECE,
    CAL_REASON_IN_SAMPLE,
    CAL_REASON_MCE,
    CAL_REASON_NO_PROBABILITIES,
    CAL_REASON_THIN_HOLDOUT,
    CalibrationGateError,
    calibration_problems,
    evaluate_calibration,
    is_in_sample,
    noise_floor,
    render_calibration,
)
from core.config import (
    CALIBRATION_GATE_APPROVED,
    CALIBRATION_GATE_MAX_ECE,
    CALIBRATION_GATE_MAX_MCE,
    CALIBRATION_GATE_MIN_HOLDOUT,
    CALIBRATION_GATE_NOT_APPROVED,
    CALIBRATION_GATE_NOT_EVALUATED,
    CALIBRATION_GATE_VERSION,
    DRIFT_CALIBRATION_GAP,
)

HOLDOUT = CALIBRATION_GATE_MIN_HOLDOUT


def well_calibrated(n=400, seed=5):
    """Probabilities whose outcomes really do occur at the stated rate."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < probs).astype(float)
    return list(probs), list(outcomes)


def out_of_sample(n):
    """Fit and score index sets that provably do not overlap."""
    return {"fit_indices": range(10_000, 10_000 + n), "score_indices": range(n)}


class NoiseFloorTests(unittest.TestCase):
    def test_floor_matches_the_measured_anchor(self):
        """MEASURED: 0.1333 at n=60."""
        self.assertAlmostEqual(noise_floor(60), 0.1333, places=4)

    def test_floor_tracks_the_measured_table(self):
        """Scaled 1/sqrt(n) reproduces the simulation within 0.01."""
        for n, measured in ((100, 0.1000), (150, 0.0800), (200, 0.0700), (500, 0.0440)):
            with self.subTest(n=n):
                self.assertLess(abs(noise_floor(n) - measured), 0.01)

    def test_floor_at_the_minimum_holdout_is_below_the_bar(self):
        """The reason 200 is the minimum: below it, the floor swamps the bar."""
        self.assertLess(noise_floor(HOLDOUT), CALIBRATION_GATE_MAX_ECE)

    def test_floor_at_60_exceeds_the_bar(self):
        """Which is why a 60-observation holdout cannot decide anything."""
        self.assertGreater(noise_floor(60), CALIBRATION_GATE_MAX_ECE)

    def test_zero_observations_raises(self):
        with self.assertRaises(CalibrationGateError):
            noise_floor(0)


class ProvenanceTests(unittest.TestCase):
    def test_overlapping_indices_are_in_sample(self):
        self.assertTrue(is_in_sample(range(100), range(50, 150)))

    def test_disjoint_indices_are_out_of_sample(self):
        self.assertFalse(is_in_sample(range(60), range(60, 120)))

    def test_unknown_provenance_is_none_not_false(self):
        """None is not False: unrecorded origin has not been shown to be OOS."""
        self.assertIsNone(is_in_sample(None, range(60)))
        self.assertIsNone(is_in_sample(range(60), None))

    def test_empty_scored_set_is_unknown(self):
        self.assertIsNone(is_in_sample(range(60), []))


class InSampleTests(unittest.TestCase):
    def test_in_sample_is_not_evaluated(self):
        """THE CENTRAL RULE: an in-sample ECE of 0.0000 is not a result."""
        probs, outcomes = well_calibrated(HOLDOUT)
        report = evaluate_calibration(
            probs, outcomes, fit_indices=range(HOLDOUT), score_indices=range(HOLDOUT)
        )
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_IN_SAMPLE)

    def test_in_sample_never_reports_an_ece(self):
        """It must not even be computed, or 0.0000 can leak into a comparison."""
        probs, outcomes = well_calibrated(HOLDOUT)
        report = evaluate_calibration(
            probs, outcomes, fit_indices=range(HOLDOUT), score_indices=range(HOLDOUT)
        )
        self.assertIsNone(report.get("ece"))

    def test_unknown_provenance_is_treated_as_in_sample(self):
        probs, outcomes = well_calibrated(HOLDOUT)
        report = evaluate_calibration(probs, outcomes)
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_IN_SAMPLE)

    def test_in_sample_can_never_be_approved(self):
        probs, outcomes = well_calibrated(HOLDOUT)
        report = evaluate_calibration(
            probs, outcomes, fit_indices=range(HOLDOUT), score_indices=range(HOLDOUT)
        )
        self.assertNotEqual(report["verdict"], CALIBRATION_GATE_APPROVED)


class EvaluateTests(unittest.TestCase):
    def test_well_calibrated_large_holdout_is_approved(self):
        """The gate must be able to APPROVE, or refusing proves nothing."""
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        self.assertEqual(report["verdict"], CALIBRATION_GATE_APPROVED)
        self.assertEqual(calibration_problems(report), [])

    def test_overconfident_probabilities_are_not_approved(self):
        probs, outcomes = well_calibrated(400)
        shifted = [min(1.0, p + 0.25) for p in probs]
        report = evaluate_calibration(shifted, outcomes, **out_of_sample(400))
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_APPROVED)
        self.assertEqual(report["reason_code"], CAL_REASON_ECE)

    def test_thin_holdout_is_not_evaluated_not_passed(self):
        """Below the floor means 'cannot tell', never 'calibrated'."""
        probs, outcomes = well_calibrated(60)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(60))
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_THIN_HOLDOUT)

    def test_a_thin_holdout_with_a_perfect_ece_is_still_not_approved(self):
        """The trap: a tiny sample can post ECE 0.0 and mean nothing."""
        report = evaluate_calibration(
            [0.5] * 20, [1.0] * 10 + [0.0] * 10, **out_of_sample(20)
        )
        self.assertNotEqual(report["verdict"], CALIBRATION_GATE_APPROVED)

    def test_a_wrong_region_hidden_by_a_passing_average_is_caught(self):
        """THE CASE MCE EXISTS FOR. ECE is count-weighted, so a small region
        that is completely wrong averages away; the worst bin cannot hide."""
        rng = np.random.default_rng(9)
        good = rng.uniform(0.4, 0.6, 395)
        probs = list(good) + [0.95] * 5
        outcomes = list((rng.random(395) < good).astype(float)) + [0.0] * 5
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        # The probe is only meaningful if the AVERAGE passes.
        self.assertLessEqual(report["ece"], CALIBRATION_GATE_MAX_ECE)
        self.assertGreater(report["mce"], CALIBRATION_GATE_MAX_MCE)
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_APPROVED)
        self.assertEqual(report["reason_code"], CAL_REASON_MCE)

    def test_absent_probabilities_are_not_evaluated(self):
        report = evaluate_calibration(None, None)
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_NO_PROBABILITIES)

    def test_a_missing_probability_raises_rather_than_becoming_half(self):
        """Coercing to 0.5 manufactures calibration evidence from silence."""
        with self.assertRaises(CalibrationGateError) as ctx:
            evaluate_calibration([0.6, None], [1.0, 0.0], **out_of_sample(2))
        self.assertIn("coin flip", str(ctx.exception))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(CalibrationGateError):
            evaluate_calibration([0.5], [1.0, 0.0], **out_of_sample(1))

    def test_the_noise_floor_is_reported_not_just_applied(self):
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        self.assertAlmostEqual(report["floor"], noise_floor(400), places=6)


class ThresholdTests(unittest.TestCase):
    def test_ece_bar_is_l6s_measured_drift_threshold(self):
        """Reused, not invented, so 'miscalibrated' means one thing."""
        self.assertEqual(CALIBRATION_GATE_MAX_ECE, DRIFT_CALIBRATION_GAP)

    def test_worst_bin_bar_exceeds_the_average_bar(self):
        self.assertGreater(CALIBRATION_GATE_MAX_MCE, CALIBRATION_GATE_MAX_ECE)


class ContractTests(unittest.TestCase):
    def test_clean_report_has_no_problems(self):
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        self.assertEqual(calibration_problems(report), [])

    def test_approved_without_confirmed_provenance_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["in_sample"] = None
        self.assertTrue(
            any("out-of-sample provenance" in p for p in calibration_problems(report))
        )

    def test_approved_above_the_ece_bar_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["ece"] = 0.5
        self.assertTrue(any("above the" in p for p in calibration_problems(report)))

    def test_approved_on_a_thin_holdout_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["observations"] = 60
        self.assertTrue(
            any("sampling noise" in p for p in calibration_problems(report))
        )

    def test_not_approved_without_a_reason_code_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        shifted = [min(1.0, p + 0.25) for p in probs]
        report = dict(evaluate_calibration(shifted, outcomes, **out_of_sample(400)))
        report["reason_code"] = None
        self.assertTrue(any("must name WHY" in p for p in calibration_problems(report)))

    def test_blocks_trades_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in calibration_problems(report)))

    def test_coerces_missing_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["coerces_missing"] = True
        self.assertTrue(
            any("never become 0.5" in p for p in calibration_problems(report))
        )

    def test_not_requiring_holdout_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["requires_holdout"] = False
        self.assertTrue(
            any("never evidence" in p for p in calibration_problems(report))
        )

    def test_approved_scored_in_sample_is_a_problem(self):
        """The contract check is the last defence when a producer is wrong."""
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["in_sample"] = True
        self.assertTrue(
            any("out-of-sample provenance" in p for p in calibration_problems(report))
        )

    def test_approved_above_the_worst_bin_bar_is_a_problem(self):
        probs, outcomes = well_calibrated(400)
        report = dict(evaluate_calibration(probs, outcomes, **out_of_sample(400)))
        report["mce"] = 0.9
        self.assertTrue(
            any("worst bin" in p for p in calibration_problems(report))
        )

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(calibration_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        self.assertEqual(report["version"], CALIBRATION_GATE_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_verdict(self):
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes, **out_of_sample(400))
        self.assertTrue(
            any(CALIBRATION_GATE_APPROVED in line for line in render_calibration(report))
        )

    def test_render_shows_unknown_provenance_as_three_states(self):
        probs, outcomes = well_calibrated(400)
        report = evaluate_calibration(probs, outcomes)
        text = "\n".join(render_calibration(report))
        self.assertIn("UNKNOWN", text)

    def test_render_shows_absent_rather_than_zero(self):
        report = evaluate_calibration(None, None)
        self.assertIn("ABSENT", "\n".join(render_calibration(report)))

    def test_render_returns_lines_not_a_blob(self):
        probs, outcomes = well_calibrated(400)
        lines = render_calibration(
            evaluate_calibration(probs, outcomes, **out_of_sample(400))
        )
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class RealRunTests(unittest.TestCase):
    """The measurement itself, against the runs the repo ships."""

    def test_in_sample_calibration_of_a_shipped_run_is_refused(self):
        from core.calibration import calibrated_probability, fit_calibration
        from core.training import load_training_runs

        runs = {r["estimator"]: r for r in load_training_runs()}
        if "historical_mean" not in runs:
            self.skipTest("no shipped training runs")
        fold = runs["historical_mean"]["folds"][0]
        predictions = np.array(fold["predictions"])
        actuals = np.array(fold["actuals"])

        fitted = fit_calibration(predictions, actuals)
        probs = [calibrated_probability(float(x), fitted) for x in predictions]
        report = evaluate_calibration(
            probs,
            actuals,
            fit_indices=range(len(predictions)),
            score_indices=range(len(predictions)),
            estimator="historical_mean",
        )
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_IN_SAMPLE)

    def test_honest_holdout_of_a_shipped_run_is_still_too_thin(self):
        """Even out of sample, n=60 cannot decide — and the gate says so."""
        from core.calibration import calibrated_probability, fit_calibration
        from core.training import load_training_runs

        runs = {r["estimator"]: r for r in load_training_runs()}
        if "historical_mean" not in runs:
            self.skipTest("no shipped training runs")
        fold = runs["historical_mean"]["folds"][0]
        predictions = np.array(fold["predictions"])
        actuals = np.array(fold["actuals"])

        fitted = fit_calibration(predictions[:60], actuals[:60])
        probs = [calibrated_probability(float(x), fitted) for x in predictions[60:]]
        report = evaluate_calibration(
            probs,
            actuals[60:],
            fit_indices=range(60),
            score_indices=range(60, 120),
            estimator="historical_mean",
        )
        self.assertEqual(report["verdict"], CALIBRATION_GATE_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], CAL_REASON_THIN_HOLDOUT)
        # And the honest ECE really is the measured one.
        self.assertGreater(report["ece"], 0.10)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
