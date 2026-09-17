"""Regression tests for issues found in the post-Sprint-M review pass.

Each test here corresponds to a defect that existed and was fixed, per the
house rule that a bug fix lands with the test that would have caught it.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.calibration import CalibrationError, fit_calibration
from core.training import TrainingError, train_baseline
from core.training_dataset import TrainingDatasetError, build_training_dataset
from tests._dataset_fixture import load_frame, shared_dataset


class TestThinHistoryRaisesTypedError(unittest.TestCase):
    """A recent listing or short cached frame is an ordinary condition.

    It previously raised a raw ValueError from the fold builder, so a caller
    writing `except TrainingError` would miss it entirely.
    """

    def _thin_dataset(self):
        frame = load_frame()
        times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[-80:-40]]
        return build_training_dataset({"NVDA": times}, {"NVDA": frame})

    def test_thin_history_raises_training_error(self) -> None:
        with self.assertRaises(TrainingError) as ctx:
            train_baseline(
                self._thin_dataset(), "ridge",
                fold_sessions=60, embargo_sessions=60, holdout_sessions=60,
            )
        self.assertIn("cannot be trained on", str(ctx.exception))

    def test_the_error_names_the_requirement(self) -> None:
        """Saying 'too short' without saying how short is unhelpful."""
        with self.assertRaises(TrainingError) as ctx:
            train_baseline(
                self._thin_dataset(), "ridge",
                fold_sessions=60, embargo_sessions=60, holdout_sessions=60,
            )
        self.assertIn("needs at least", str(ctx.exception))


class TestNonFiniteCalibrationInputs(unittest.TestCase):
    """NaN predictions raised a raw sklearn error that explained nothing."""

    def test_nan_predictions_raise_calibration_error(self) -> None:
        predictions = [float("nan")] * 50 + list(np.linspace(0, 1, 150))
        with self.assertRaises(CalibrationError) as ctx:
            fit_calibration(predictions, [0.01, -0.01] * 100)
        self.assertIn("NaN or infinite", str(ctx.exception))

    def test_infinite_predictions_raise_calibration_error(self) -> None:
        predictions = [float("inf")] * 50 + list(np.linspace(0, 1, 150))
        with self.assertRaises(CalibrationError):
            fit_calibration(predictions, [0.01, -0.01] * 100)

    def test_non_finite_actuals_are_refused(self) -> None:
        with self.assertRaises(CalibrationError):
            fit_calibration(list(np.linspace(0, 1, 200)), [float("nan")] * 200)

    def test_clean_inputs_still_fit(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 1, 200)
        actuals = ((scores + rng.normal(0, 1, 200)) > 0).astype(float) * 0.02 - 0.01
        self.assertTrue(fit_calibration(scores, actuals).method)


class TestDatasetSurvivorshipVerdict(unittest.TestCase):
    """A multi-ticker dataset could silently inherit survivorship bias.

    The backtest engine enforced the V6 ledger; the dataset builder did not.
    Every ticker someone types today is one that survived to be typed.
    """

    def test_single_ticker_dataset_reports_single_ticker(self) -> None:
        """A one-name set makes no universe claim."""
        verdict = shared_dataset(-700, -400).survivorship
        self.assertEqual(verdict["status"], "single_ticker")

    def _multi(self, **kwargs):
        nvda = load_frame("NVDA")
        try:
            other = pd.read_parquet("data/MSFT_5y_1d.parquet")
        except Exception:
            self.skipTest("MSFT fixture not available")
        other.index = pd.DatetimeIndex(other.index)
        times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in nvda.index[-260:-245]]
        return build_training_dataset(
            {"NVDA": times, "MSFT": times}, {"NVDA": nvda, "MSFT": other}, **kwargs
        )

    def test_multi_ticker_dataset_is_judged(self) -> None:
        verdict = self._multi().survivorship
        self.assertNotEqual(verdict["status"], "single_ticker")
        self.assertEqual(verdict["tickers"], 2)

    def test_empty_ledger_is_unverifiable_not_silently_clean(self) -> None:
        verdict = self._multi().survivorship
        self.assertEqual(verdict["status"], "unverifiable")
        self.assertIn("ledger", verdict["detail"])

    def test_fail_closed_mode_refuses_an_unsafe_build(self) -> None:
        with self.assertRaises(TrainingDatasetError) as ctx:
            self._multi(require_survivorship_safe=True)
        self.assertIn("survivorship-safe", str(ctx.exception))

    def test_verdict_travels_in_the_report(self) -> None:
        self.assertIn("survivorship", shared_dataset(-700, -400).report())


if __name__ == "__main__":
    unittest.main()
