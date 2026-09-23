"""X1 out-of-sample validation tests.

The measurements these tests defend, from the 8 training runs on disk:

* Zero of seven learned estimators beat the no-feature ``historical_mean``
  baseline on RMSE (best learned 0.13192 vs baseline 0.12164).
* All eight directional accuracies fall inside the 0.5 +/- 0.0895 sampling band
  at n=120; the best, momentum at 0.5750, returns p=0.0625 by permutation.
* Every run has exactly ONE fold, so no result has any dispersion.
"""

from __future__ import annotations

import math
import unittest

from core.config import (
    OOS_ALPHA,
    OOS_APPROVED,
    OOS_BASELINE_ESTIMATOR,
    OOS_MIN_FOLDS,
    OOS_MIN_OBSERVATIONS,
    OOS_NOT_APPROVED,
    OOS_NOT_EVALUATED,
    OOS_VALIDATION_VERSION,
)
from core.oos_validation import (
    OOSValidationError,
    OOS_REASON_INSIGNIFICANT,
    OOS_REASON_LOST,
    OOS_REASON_NO_BASELINE,
    OOS_REASON_TOO_FEW_FOLDS,
    OOS_REASON_TOO_FEW_OBSERVATIONS,
    beats_baseline,
    directional_accuracy,
    render_validation,
    rmse,
    sampling_band,
    validate_run,
    validate_suite,
    validation_problems,
)

BASELINE = {"rmse": 0.12164, "directional_accuracy": 0.5500, "observations": 120}


def run(**over):
    """A run that clears every structural check, so substantive ones decide."""
    metrics = {"observations": 120, "rmse": 0.10000, "directional_accuracy": 0.70}
    metrics.update(over.pop("metrics", {}))
    base = {
        "estimator": "candidate",
        "target_horizon": "20d",
        "folds": [{}, {}],
        "metrics": metrics,
    }
    base.update(over)
    return base


class MetricTests(unittest.TestCase):
    def test_rmse_is_zero_for_perfect_predictions(self):
        self.assertAlmostEqual(rmse([1.0, -2.0], [1.0, -2.0]), 0.0)

    def test_rmse_matches_hand_computation(self):
        self.assertAlmostEqual(rmse([0.0, 0.0], [3.0, 4.0]), math.sqrt(12.5))

    def test_rmse_rejects_mismatched_lengths(self):
        with self.assertRaises(OOSValidationError):
            rmse([1.0], [1.0, 2.0])

    def test_rmse_rejects_empty(self):
        with self.assertRaises(OOSValidationError):
            rmse([], [])

    def test_rmse_refuses_to_coerce_a_missing_value(self):
        """Coercing would credit a prediction the model never made."""
        with self.assertRaises(OOSValidationError):
            rmse([1.0, None], [1.0, 2.0])

    def test_directional_accuracy_counts_sign_matches(self):
        self.assertAlmostEqual(
            directional_accuracy([1.0, -1.0, 1.0], [1.0, -1.0, -1.0]), 2 / 3
        )

    def test_zero_is_its_own_sign_class(self):
        """Folding zero into positive would credit a non-prediction."""
        self.assertAlmostEqual(directional_accuracy([0.0], [1.0]), 0.0)


class SamplingBandTests(unittest.TestCase):
    def test_band_at_120_matches_the_measurement(self):
        """THE MEASUREMENT: 0.0895 at n=120, wider than any observed effect."""
        self.assertAlmostEqual(sampling_band(120), 0.0895, places=4)

    def test_band_narrows_with_more_observations(self):
        self.assertLess(sampling_band(1200), sampling_band(120))

    def test_untabulated_alpha_raises_rather_than_inventing(self):
        with self.assertRaises(OOSValidationError):
            sampling_band(120, alpha=0.037)

    def test_zero_observations_raises(self):
        with self.assertRaises(OOSValidationError):
            sampling_band(0)


class BeatsBaselineTests(unittest.TestCase):
    def test_lower_rmse_beats(self):
        beats, _ = beats_baseline({"rmse": 0.10}, {"rmse": 0.12})
        self.assertTrue(beats)

    def test_higher_rmse_loses(self):
        beats, _ = beats_baseline({"rmse": 0.13}, {"rmse": 0.12})
        self.assertFalse(beats)

    def test_missing_baseline_is_none_not_false(self):
        """'We could not compare' and 'it lost' are different answers."""
        beats, reason = beats_baseline({"rmse": 0.10}, None)
        self.assertIsNone(beats)
        self.assertIn("baseline", reason)

    def test_missing_metric_is_none_not_false(self):
        beats, _ = beats_baseline({"mae": 0.1}, {"rmse": 0.12})
        self.assertIsNone(beats)

    def test_higher_is_better_flips_the_comparison(self):
        beats, _ = beats_baseline(
            {"acc": 0.6}, {"acc": 0.5}, metric="acc", higher_is_better=True
        )
        self.assertTrue(beats)


class ValidateRunTests(unittest.TestCase):
    def test_a_single_fold_is_not_approved(self):
        """MEASURED: every run on disk has exactly one fold."""
        report = validate_run(run(folds=[{}]), BASELINE)
        self.assertEqual(report["verdict"], OOS_NOT_APPROVED)
        self.assertEqual(report["reason_code"], OOS_REASON_TOO_FEW_FOLDS)

    def test_too_few_observations_is_not_approved(self):
        report = validate_run(
            run(metrics={"observations": OOS_MIN_OBSERVATIONS - 1}), BASELINE
        )
        self.assertEqual(report["reason_code"], OOS_REASON_TOO_FEW_OBSERVATIONS)

    def test_losing_to_the_baseline_is_not_approved(self):
        """THE DECIDING MEASUREMENT: 0 of 7 learned estimators beat it."""
        report = validate_run(run(metrics={"rmse": 0.13192}), BASELINE)
        self.assertEqual(report["verdict"], OOS_NOT_APPROVED)
        self.assertEqual(report["reason_code"], OOS_REASON_LOST)

    def test_beating_but_inside_the_band_is_not_approved(self):
        """A better point estimate is not evidence."""
        report = validate_run(
            run(metrics={"rmse": 0.10, "directional_accuracy": 0.55}), BASELINE
        )
        self.assertEqual(report["verdict"], OOS_NOT_APPROVED)
        self.assertEqual(report["reason_code"], OOS_REASON_INSIGNIFICANT)

    def test_beating_and_clearing_the_band_is_approved(self):
        """The gate must be able to APPROVE, or it proves nothing by refusing."""
        report = validate_run(run(), BASELINE)
        self.assertEqual(report["verdict"], OOS_APPROVED)
        self.assertIsNone(report["reason_code"])
        self.assertEqual(validation_problems(report), [])

    def test_no_baseline_is_not_evaluated_not_failed(self):
        report = validate_run(run(), None)
        self.assertEqual(report["verdict"], OOS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], OOS_REASON_NO_BASELINE)

    def test_absent_run_is_not_evaluated(self):
        report = validate_run(None, BASELINE)
        self.assertEqual(report["verdict"], OOS_NOT_EVALUATED)

    def test_missing_observation_count_is_not_evaluated(self):
        report = validate_run(run(metrics={"observations": None}), BASELINE)
        self.assertEqual(report["verdict"], OOS_NOT_EVALUATED)

    def test_non_mapping_run_raises(self):
        with self.assertRaises(OOSValidationError):
            validate_run(["run"], BASELINE)

    def test_the_band_is_reported_not_just_applied(self):
        report = validate_run(run(), BASELINE)
        self.assertAlmostEqual(report["sampling_band"], 0.0895, places=4)


class ValidateSuiteTests(unittest.TestCase):
    def test_a_suite_with_no_winner_is_not_approved(self):
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]},
            run(estimator="loser", metrics={"rmse": 0.20}),
        ]
        report = validate_suite(runs)
        self.assertEqual(report["verdict"], OOS_NOT_APPROVED)
        self.assertEqual(report["approved"], 0)

    def test_the_baseline_is_not_tested_against_itself(self):
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]},
            run(estimator="a"),
        ]
        self.assertEqual(validate_suite(runs)["tested"], 1)

    def test_a_suite_without_a_baseline_is_not_evaluated(self):
        report = validate_suite([run(estimator="a")])
        self.assertEqual(report["verdict"], OOS_NOT_EVALUATED)

    def test_selection_risk_is_stated(self):
        """With 8 estimators at alpha=0.05 the exposure is 33.7%."""
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]}
        ] + [run(estimator=f"e{i}") for i in range(8)]
        report = validate_suite(runs)
        self.assertAlmostEqual(report["selection_risk"], 1 - 0.95**8, places=6)

    def test_empty_suite_is_not_evaluated(self):
        self.assertEqual(validate_suite([])["verdict"], OOS_NOT_EVALUATED)

    def test_a_suite_can_approve(self):
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]},
            run(estimator="winner"),
        ]
        report = validate_suite(runs)
        self.assertEqual(report["verdict"], OOS_APPROVED)
        self.assertEqual(validation_problems(report), [])


class ContractTests(unittest.TestCase):
    def test_clean_report_has_no_problems(self):
        self.assertEqual(validation_problems(validate_run(run(), BASELINE)), [])

    def test_not_approved_without_a_reason_code_is_a_problem(self):
        report = dict(validate_run(run(metrics={"rmse": 0.2}), BASELINE))
        report["reason_code"] = None
        self.assertTrue(any("must name WHY" in p for p in validation_problems(report)))

    def test_blocks_trades_is_a_problem(self):
        report = dict(validate_run(run(), BASELINE))
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in validation_problems(report)))

    def test_coerces_missing_is_a_problem(self):
        report = dict(validate_run(run(), BASELINE))
        report["coerces_missing"] = True
        self.assertTrue(any("stay ABSENT" in p for p in validation_problems(report)))

    def test_missing_reason_is_a_problem(self):
        report = dict(validate_run(run(), BASELINE))
        report["reason"] = ""
        self.assertTrue(any("reason" in p for p in validation_problems(report)))

    def test_suite_without_selection_risk_is_a_problem(self):
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]},
            run(estimator="a"),
        ]
        report = dict(validate_suite(runs))
        report["selection_risk"] = None
        self.assertTrue(
            any("multiple-testing" in p for p in validation_problems(report))
        )

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(validation_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        self.assertEqual(
            validate_run(run(), BASELINE)["version"], OOS_VALIDATION_VERSION
        )


class RenderTests(unittest.TestCase):
    def test_render_names_the_verdict(self):
        lines = render_validation(validate_run(run(), BASELINE))
        self.assertTrue(any(OOS_APPROVED in line for line in lines))

    def test_render_shows_absent_rather_than_zero(self):
        report = validate_run(None, BASELINE)
        self.assertIn("ABSENT", "\n".join(render_validation(report)))

    def test_suite_render_includes_each_report(self):
        runs = [
            {"estimator": OOS_BASELINE_ESTIMATOR, "metrics": BASELINE, "folds": [{}, {}]},
            run(estimator="a"),
            run(estimator="b"),
        ]
        text = "\n".join(render_validation(validate_suite(runs)))
        self.assertIn("a @", text)
        self.assertIn("b @", text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_validation(validate_run(run(), BASELINE))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class RealRunTests(unittest.TestCase):
    """The measurement itself, against the runs the repo ships."""

    def test_the_shipped_suite_is_not_approved(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        report = validate_suite(runs)
        self.assertEqual(report["verdict"], OOS_NOT_APPROVED)
        self.assertEqual(report["approved"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
