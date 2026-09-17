"""Calibration and uncertainty tests (Sprint M6 / board M4).

The M6 rule, pinned: "Never expose arbitrary probability numbers as if they
were calibrated."

Covers every M6 requirement — probability calibration, reliability curves,
Brier, log loss, calibration error, prediction intervals, uncertainty, fold
dispersion — plus the two properties that make the numbers trustworthy:

- the mapping is monotone, so a higher score never yields a lower
  probability;
- calibration is MEASURED out-of-fold, because an in-sample ECE is
  near-perfect by construction and tells you nothing.
"""

from __future__ import annotations

import math

import unittest
from types import SimpleNamespace

import numpy as np

from core.calibration import (
    CalibrationError,
    CalibrationMap,
    UncalibratedProbabilityError,
    brier_score,
    calibrate_training_run,
    calibrated_probability,
    expected_calibration_error,
    fit_calibration,
    fold_dispersion,
    log_loss,
    max_calibration_error,
    prediction_interval,
    reliability_curve,
)
from core.config import (
    CALIBRATION_METHOD_ISOTONIC,
    CALIBRATION_METHOD_PLATT,
    CALIBRATION_MIN_ISOTONIC_SAMPLES,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_VERSION,
    PREDICTION_INTERVAL_LEVEL,
)


def _signal(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores with genuine signal, and returns whose sign follows them."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, n)
    up = (scores + rng.normal(0.0, 1.0, n)) > 0
    return scores, up.astype(float) * 0.02 - 0.01


class TestFitting(unittest.TestCase):
    def test_isotonic_is_preferred_on_ample_data(self) -> None:
        scores, actuals = _signal(CALIBRATION_MIN_ISOTONIC_SAMPLES + 50)
        self.assertEqual(fit_calibration(scores, actuals).method, CALIBRATION_METHOD_ISOTONIC)

    def test_platt_is_used_on_thin_data(self) -> None:
        """Isotonic overfits small samples, so the fitter falls back."""
        scores, actuals = _signal(CALIBRATION_MIN_ISOTONIC_SAMPLES - 10)
        fitted = fit_calibration(scores, actuals)
        self.assertEqual(fitted.method, CALIBRATION_METHOD_PLATT)
        self.assertIn("isotonic overfits", fitted.fallback_reason)

    def test_fallback_reason_is_empty_when_no_fallback_happened(self) -> None:
        scores, actuals = _signal(CALIBRATION_MIN_ISOTONIC_SAMPLES + 50)
        self.assertEqual(fit_calibration(scores, actuals).fallback_reason, "")

    def test_explicit_method_is_honoured(self) -> None:
        scores, actuals = _signal(300)
        self.assertEqual(
            fit_calibration(scores, actuals, CALIBRATION_METHOD_PLATT).method,
            CALIBRATION_METHOD_PLATT,
        )

    def test_too_few_samples_is_refused(self) -> None:
        """A map nobody should trust is worse than an explicit refusal."""
        scores, actuals = _signal(CALIBRATION_MIN_SAMPLES - 1)
        with self.assertRaises(CalibrationError) as ctx:
            fit_calibration(scores, actuals)
        self.assertIn("no honest probability", str(ctx.exception))

    def test_single_outcome_class_is_refused(self) -> None:
        with self.assertRaises(CalibrationError) as ctx:
            fit_calibration(list(np.linspace(0, 1, 60)), [0.01] * 60)
        self.assertIn("single class", str(ctx.exception))

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            fit_calibration([0.1] * 50, [0.01] * 40)

    def test_unknown_method_is_refused(self) -> None:
        scores, actuals = _signal(100)
        with self.assertRaises(CalibrationError):
            fit_calibration(scores, actuals, "astrology")

    def test_sample_size_is_recorded(self) -> None:
        scores, actuals = _signal(150)
        self.assertEqual(fit_calibration(scores, actuals).sample_size, 150)

    def test_map_hash_is_deterministic(self) -> None:
        scores, actuals = _signal(200)
        self.assertEqual(
            fit_calibration(scores, actuals).canonical_hash(),
            fit_calibration(scores, actuals).canonical_hash(),
        )

    def test_different_data_changes_the_map_hash(self) -> None:
        first = fit_calibration(*_signal(200, seed=1))
        second = fit_calibration(*_signal(200, seed=2))
        self.assertNotEqual(first.canonical_hash(), second.canonical_hash())


class TestNeverExposeUncalibrated(unittest.TestCase):
    """The binding rule, enforced in code rather than in a comment."""

    def test_missing_map_refuses_rather_than_passing_through(self) -> None:
        with self.assertRaises(UncalibratedProbabilityError) as ctx:
            calibrated_probability(0.7, None)
        self.assertIn("is not a probability", str(ctx.exception))

    def test_a_fitted_map_produces_a_probability(self) -> None:
        fitted = fit_calibration(*_signal(200))
        value = calibrated_probability(0.5, fitted)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_output_is_always_within_zero_and_one(self) -> None:
        for method in (CALIBRATION_METHOD_ISOTONIC, CALIBRATION_METHOD_PLATT):
            with self.subTest(method=method):
                fitted = fit_calibration(*_signal(300), method)
                extreme = fitted.apply([-1e6, -10.0, 0.0, 10.0, 1e6])
                self.assertTrue(np.all(extreme >= 0.0))
                self.assertTrue(np.all(extreme <= 1.0))


class TestMonotonicity(unittest.TestCase):
    """A higher raw score must never yield a lower probability."""

    def test_both_methods_are_monotone(self) -> None:
        scores, actuals = _signal(300)
        for method in (CALIBRATION_METHOD_ISOTONIC, CALIBRATION_METHOD_PLATT):
            with self.subTest(method=method):
                fitted = fit_calibration(scores, actuals, method)
                probabilities = fitted.apply(np.sort(scores))
                self.assertTrue(np.all(np.diff(probabilities) >= -1e-12))

    def test_isotonic_clamps_outside_the_fitted_range(self) -> None:
        """The data says nothing about scores it never saw."""
        scores, actuals = _signal(300)
        fitted = fit_calibration(scores, actuals, CALIBRATION_METHOD_ISOTONIC)
        self.assertEqual(fitted.apply(-1e9), fitted.knots_y[0])
        self.assertEqual(fitted.apply(1e9), fitted.knots_y[-1])


class TestMetrics(unittest.TestCase):
    def test_perfect_forecasts_score_zero_brier(self) -> None:
        self.assertEqual(brier_score([1.0, 0.0, 1.0], [0.05, -0.05, 0.05]), 0.0)

    def test_worst_forecasts_score_one_brier(self) -> None:
        self.assertEqual(brier_score([0.0, 1.0], [0.05, -0.05]), 1.0)

    def test_log_loss_punishes_confident_mistakes(self) -> None:
        confident_wrong = log_loss([0.99], [-0.05])
        hedged_wrong = log_loss([0.6], [-0.05])
        self.assertGreater(confident_wrong, hedged_wrong)

    def test_log_loss_is_finite_for_a_certain_miss(self) -> None:
        """Clamping keeps a confident miss finite instead of infinite."""
        self.assertTrue(math.isfinite(log_loss([1.0], [-0.05])))

    def test_empty_inputs_are_refused(self) -> None:
        for fn in (brier_score, log_loss, reliability_curve):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(CalibrationError):
                    fn([], [])

    def test_flat_return_counts_as_not_up(self) -> None:
        """A zero move is not an up move."""
        self.assertEqual(brier_score([0.0], [0.0]), 0.0)


class TestReliabilityCurve(unittest.TestCase):
    def test_perfect_calibration_has_zero_gap(self) -> None:
        probabilities = [0.25] * 100 + [0.75] * 100
        actuals = [0.01] * 25 + [-0.01] * 75 + [0.01] * 75 + [-0.01] * 25
        for entry in reliability_curve(probabilities, actuals):
            with self.subTest(bin=entry["bin_lower"]):
                self.assertAlmostEqual(entry["gap"], 0.0, places=6)

    def test_empty_bins_are_omitted_not_zeroed(self) -> None:
        """No observations is not the same as never happened."""
        curve = reliability_curve([0.95] * 50, [0.01] * 25 + [-0.01] * 25)
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve[0]["count"], 50)

    def test_probability_of_one_is_counted(self) -> None:
        curve = reliability_curve([1.0] * 40, [0.01] * 20 + [-0.01] * 20)
        self.assertEqual(sum(entry["count"] for entry in curve), 40)

    def test_bins_below_two_are_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            reliability_curve([0.5] * 10, [0.01] * 10, bins=1)

    def test_overconfident_model_shows_a_positive_gap(self) -> None:
        """Says 90%, happens 50% of the time."""
        curve = reliability_curve([0.9] * 100, [0.01] * 50 + [-0.01] * 50)
        self.assertGreater(curve[0]["gap"], 0.3)


class TestCalibrationError(unittest.TestCase):
    def test_perfect_calibration_has_zero_ece(self) -> None:
        probabilities = [0.25] * 100 + [0.75] * 100
        actuals = [0.01] * 25 + [-0.01] * 75 + [0.01] * 75 + [-0.01] * 25
        self.assertAlmostEqual(
            expected_calibration_error(probabilities, actuals), 0.0, places=6
        )

    def test_overconfidence_produces_a_large_ece(self) -> None:
        self.assertGreater(
            expected_calibration_error([0.95] * 100, [0.01] * 50 + [-0.01] * 50), 0.4
        )

    def test_max_error_exposes_a_bad_region_ece_would_dilute(self) -> None:
        """One badly wrong bin among many good ones."""
        probabilities = [0.5] * 180 + [0.95] * 20
        actuals = ([0.01] * 90 + [-0.01] * 90) + [-0.01] * 20
        self.assertGreater(
            max_calibration_error(probabilities, actuals),
            expected_calibration_error(probabilities, actuals),
        )


class TestUncertainty(unittest.TestCase):
    def test_dispersion_separates_stable_from_erratic(self) -> None:
        """Same mean, very different trustworthiness."""
        stable = fold_dispersion([0.62, 0.61, 0.63, 0.62])
        erratic = fold_dispersion([0.30, 0.90, 0.35, 0.93])
        self.assertAlmostEqual(stable["mean"], erratic["mean"], places=1)
        self.assertLess(stable["std"], erratic["std"])
        self.assertLess(stable["range"], erratic["range"])

    def test_dispersion_requires_a_fold(self) -> None:
        with self.assertRaises(CalibrationError):
            fold_dispersion([])

    def test_interval_brackets_the_median(self) -> None:
        interval = prediction_interval([0.4, 0.5, 0.6, 0.7, 0.8])
        self.assertLessEqual(interval["lower"], interval["median"])
        self.assertLessEqual(interval["median"], interval["upper"])

    def test_interval_level_is_recorded(self) -> None:
        self.assertEqual(
            prediction_interval([0.4, 0.6])["level"], PREDICTION_INTERVAL_LEVEL
        )

    def test_wider_spread_gives_a_wider_interval(self) -> None:
        tight = prediction_interval([0.50, 0.51, 0.52, 0.53, 0.54])
        wide = prediction_interval([0.10, 0.30, 0.52, 0.75, 0.95])
        self.assertLess(
            tight["upper"] - tight["lower"], wide["upper"] - wide["lower"]
        )

    def test_invalid_level_is_refused(self) -> None:
        for level in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(level=level):
                with self.assertRaises(CalibrationError):
                    prediction_interval([0.5, 0.6], level=level)


class TestOutOfFoldMeasurement(unittest.TestCase):
    """An in-sample ECE is near-perfect by construction and means nothing."""

    def _run(self, folds: int = 5, per_fold: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        fold_objects = []
        for index in range(folds):
            scores = rng.normal(0.0, 1.0, per_fold)
            actuals = ((scores + rng.normal(0.0, 1.5, per_fold)) > 0).astype(float) * 0.02 - 0.01
            fold_objects.append(SimpleNamespace(
                fold_id=index,
                predictions=[float(v) for v in scores],
                actuals=[float(v) for v in actuals],
                metrics={"directional_accuracy": 0.5 + index * 0.02},
            ))
        return SimpleNamespace(folds=fold_objects)

    def test_measurement_is_out_of_fold_when_folds_allow(self) -> None:
        report = calibrate_training_run(self._run())
        self.assertEqual(report.measurement_basis, "out_of_fold")

    def test_in_sample_measurement_is_labelled_when_folds_are_too_few(self) -> None:
        """Rather than silently reporting flattering numbers."""
        report = calibrate_training_run(self._run(folds=2, per_fold=200))
        self.assertEqual(report.measurement_basis, "in_sample")

    def test_out_of_fold_error_exceeds_in_sample_error(self) -> None:
        """The whole reason the out-of-fold path exists."""
        run = self._run()
        honest = calibrate_training_run(run)

        pooled_predictions = [v for fold in run.folds for v in fold.predictions]
        pooled_actuals = [v for fold in run.folds for v in fold.actuals]
        in_sample_map = fit_calibration(pooled_predictions, pooled_actuals)
        in_sample_ece = expected_calibration_error(
            in_sample_map.apply(pooled_predictions), pooled_actuals
        )
        self.assertGreater(honest.expected_calibration_error, in_sample_ece)

    def test_report_carries_every_m6_requirement(self) -> None:
        report = calibrate_training_run(self._run()).to_dict()
        for key in (
            "calibration_map", "brier", "logloss", "expected_calibration_error",
            "max_calibration_error", "reliability", "uncertainty", "interval",
            "sample_size", "measurement_basis",
        ):
            with self.subTest(key=key):
                self.assertIn(key, report)

    def test_version_is_stamped(self) -> None:
        self.assertEqual(
            calibrate_training_run(self._run()).calibration_version, CALIBRATION_VERSION
        )

    def test_run_without_folds_is_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            calibrate_training_run(SimpleNamespace(folds=[]))

    def test_run_without_retained_predictions_is_refused(self) -> None:
        run = SimpleNamespace(folds=[
            SimpleNamespace(fold_id=0, predictions=[], actuals=[], metrics={})
        ])
        with self.assertRaises(CalibrationError) as ctx:
            calibrate_training_run(run)
        self.assertIn("out-of-sample", str(ctx.exception))




if __name__ == "__main__":
    unittest.main()
