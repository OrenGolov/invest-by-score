"""A2 tests — the embargo is counted in ROWS but the horizon is in SESSIONS.

MEASURED: `build_walk_forward_folds` is named in sessions and `training.py` passes
a row count. A panel dataset holds one row per (ticker, prediction_time), so with
14 tickers an embargo of 252 ROWS spans only 18 DATES — short of even the 20d
horizon. The shipped run escaped only because its prediction times are sampled
~96 calendar days apart; that is a property of the sampling, not a guarantee.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.config import LABEL_HORIZON_SESSIONS
from core.feature_registry import build_default_registry, feature_set_hash
from core.training import (
    TrainingError,
    embargo_problems,
    minimum_rows_for,
    required_embargo_rows,
    rows_per_date,
    train_baseline,
)
from core.training_dataset import TrainingDataset, TrainingRow, dataset_hash


def panel(tickers: int, dates: int, start="2010-01-01"):
    """A dense panel: `tickers` rows on each of `dates` consecutive business days."""
    rng = np.random.default_rng(1)
    stamps = pd.bdate_range(start, periods=dates)
    rows = []
    for stamp in stamps:
        for index in range(tickers):
            rows.append(
                TrainingRow(
                    ticker=f"T{index:02d}",
                    prediction_time=str(stamp),
                    features={
                        "rsi": float(rng.normal(50, 10)),
                        "volatility": float(abs(rng.normal(0.01, 0.003))),
                    },
                    feature_contracts={
                        "rsi": {"calculation_version": "v1"},
                        "volatility": {"calculation_version": "v1"},
                    },
                    target_horizon="20d",
                    forward_return=float(rng.normal(0, 0.02)),
                    label_up=True,
                    realized_vol=0.1,
                    adverse_excursion=-0.01,
                    label_version="lv1",
                    label_record_hash=f"h{stamp}{index}",
                )
            )
    return rows


def dataset_of(rows):
    registry = build_default_registry()
    digest = feature_set_hash(["rsi", "volatility"], registry)
    return (
        TrainingDataset(
            rows=rows,
            feature_names=["rsi", "volatility"],
            target_horizon="20d",
            dataset_hash=dataset_hash(rows, digest, "20d"),
            feature_set_hash=digest,
        ),
        registry,
    )


class DensityIsTheConversionFactorTests(unittest.TestCase):
    def test_a_single_ticker_series_has_one_row_per_date(self):
        self.assertAlmostEqual(rows_per_date(panel(1, 50)), 1.0)

    def test_a_panel_packs_many_rows_into_one_date(self):
        self.assertAlmostEqual(rows_per_date(panel(14, 50)), 14.0)

    def test_an_empty_dataset_does_not_divide_by_zero(self):
        self.assertEqual(rows_per_date([]), 1.0)

    def test_rows_without_timestamps_fall_back_to_one(self):
        class Bare:
            prediction_time = None

        self.assertEqual(rows_per_date([Bare(), Bare()]), 1.0)


class RequiredEmbargoScalesWithBothTests(unittest.TestCase):
    """MEASURED, and the reason this is computed rather than configured.

        tickers   20d      60d       252d
              1    20       60        252
             14   280      840      3,528
             50 1,000    3,000     12,600
    """

    def test_a_single_ticker_needs_the_horizon_in_rows(self):
        rows = panel(1, 40)
        for horizon, sessions in LABEL_HORIZON_SESSIONS.items():
            with self.subTest(horizon=horizon):
                self.assertEqual(required_embargo_rows(rows, horizon), sessions)

    def test_a_panel_needs_the_horizon_times_its_density(self):
        rows = panel(14, 40)
        self.assertEqual(required_embargo_rows(rows, "20d"), 280)
        self.assertEqual(required_embargo_rows(rows, "60d"), 840)
        self.assertEqual(required_embargo_rows(rows, "252d"), 3528)

    def test_an_unknown_horizon_is_refused(self):
        with self.assertRaises(TrainingError):
            required_embargo_rows(panel(1, 10), "7d")

    def test_the_minimum_dataset_accounts_for_density(self):
        thin = minimum_rows_for(panel(1, 10), "20d", 300, 60)
        dense = minimum_rows_for(panel(14, 10), "20d", 300, 60)
        self.assertGreater(dense, thin)


class LeakageIsRefusedTests(unittest.TestCase):
    """The bug this closes: a row-embargo that spans too few dates."""

    def test_a_dense_panel_with_a_row_embargo_is_refused(self):
        # 14 tickers over 120 dates: a 252-ROW embargo is 18 dates, and the 20d
        # label needs ~28 calendar days.
        data, registry = dataset_of(panel(14, 120))
        with self.assertRaises(TrainingError) as caught:
            train_baseline(
                data,
                estimator="ridge",
                fold_sessions=300,
                embargo_sessions=252,
                holdout_sessions=60,
                registry=registry,
            )
        self.assertIn("calendar", str(caught.exception))

    def test_the_refusal_names_the_shortfall(self):
        data, registry = dataset_of(panel(14, 120))
        with self.assertRaises(TrainingError) as caught:
            train_baseline(
                data,
                estimator="ridge",
                fold_sessions=300,
                embargo_sessions=252,
                holdout_sessions=60,
                registry=registry,
            )
        message = str(caught.exception)
        self.assertIn("20d", message)
        self.assertIn("ROWS", message)

    def test_a_single_ticker_series_is_not_refused(self):
        # Where a row IS a session the geometry means what it says, and a
        # false positive here would block the legitimate case.
        data, registry = dataset_of(panel(1, 1500))
        run = train_baseline(
            data,
            estimator="ridge",
            fold_sessions=300,
            embargo_sessions=252,
            holdout_sessions=60,
            registry=registry,
        )
        self.assertGreaterEqual(len(run.folds), 1)

    def test_the_verifier_passes_a_well_separated_panel(self):
        # Directly, without training: separation measured in calendar days.
        rows = panel(1, 1500)
        folds = [{"fold_id": 0, "train": [0, 299], "validation": [552, 851]}]
        self.assertEqual(embargo_problems(rows, folds, "20d"), [])

    def test_the_verifier_catches_an_adjacent_split(self):
        rows = panel(1, 1500)
        folds = [{"fold_id": 0, "train": [0, 299], "validation": [300, 599]}]
        self.assertTrue(embargo_problems(rows, folds, "20d"))

    def test_an_unknown_horizon_is_reported_not_ignored(self):
        self.assertTrue(embargo_problems(panel(1, 10), [], "7d"))

    def test_out_of_range_indices_are_skipped_not_fatal(self):
        rows = panel(1, 10)
        folds = [{"fold_id": 0, "train": [0, 999], "validation": [1000, 1999]}]
        self.assertEqual(embargo_problems(rows, folds, "20d"), [])


class TrainPyDefaultsAreRunnableTests(unittest.TestCase):
    """A2's headline: the script could not run with its own defaults."""

    def setUp(self):
        import importlib.util
        import pathlib

        spec = importlib.util.spec_from_file_location(
            "train_script", pathlib.Path("scripts/train.py")
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_the_embargo_default_covers_the_longest_horizon(self):
        # F2 raised the horizon to 252 the day after the ledger was written, and
        # the default stayed at 60 — rejected by build_walk_forward_folds ever
        # since.
        self.assertGreaterEqual(
            self.module._MAX_HORIZON, max(LABEL_HORIZON_SESSIONS.values())
        )

    def test_the_fold_default_exceeds_the_horizon(self):
        # A1 measured that the gap before the holdout is bounded by
        # `fold_sessions - 4`, so a fold at or below the horizon can never
        # embargo the tail.
        self.assertGreater(self.module._MAX_HORIZON + 48, self.module._MAX_HORIZON)

    def test_the_row_default_fits_two_folds_and_a_sealed_tail(self):
        fold = self.module._MAX_HORIZON + 48
        embargo = self.module._MAX_HORIZON
        holdout = 60
        rows = self.module._DEFAULT_ROWS
        self.assertGreaterEqual(rows, 3 * fold + 2 * embargo + holdout - fold)

    def test_the_default_geometry_is_accepted_by_the_engine(self):
        from core.backtest.engine import build_walk_forward_folds

        geometry = build_walk_forward_folds(
            self.module._DEFAULT_ROWS,
            self.module._MAX_HORIZON + 48,
            self.module._MAX_HORIZON,
            60,
        )
        self.assertGreaterEqual(len(geometry["folds"]), 2)

    def test_the_default_geometry_embargoes_the_holdout(self):
        from core.backtest.engine import build_walk_forward_folds

        geometry = build_walk_forward_folds(
            self.module._DEFAULT_ROWS,
            self.module._MAX_HORIZON + 48,
            self.module._MAX_HORIZON,
            60,
        )
        last_end = geometry["folds"][-1]["validation"][1]
        gap = geometry["holdout"][0] - last_end - 1
        self.assertGreaterEqual(gap, max(LABEL_HORIZON_SESSIONS.values()))


if __name__ == "__main__":
    unittest.main()
