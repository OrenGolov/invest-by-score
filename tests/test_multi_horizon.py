"""A3 tests — an edge must be checkable across horizons.

X6 reported HORIZONS_MISSING because every run in the ledger trained the single
`20d` target, so there was no spread to compare. These tests pin the multi-horizon
training path and the distinction X6 rests on: a horizon that COULD NOT TRAIN is
not a horizon that trained and showed nothing.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import LABEL_HORIZON_SESSIONS, TEMPORAL_HORIZONS
from core.feature_registry import build_default_registry, feature_set_hash
from core.temporal_robustness import evaluate_temporal_robustness
from core.training import (
    TrainingError,
    horizon_metrics,
    train_across_horizons,
)
from core.training_dataset import TrainingDataset, TrainingRow, dataset_hash


def rows_for(horizon: str, count: int = 1500, *, edge: float = 0.0, seed: int = 7):
    """Rows whose forward return carries `edge` predictability, per horizon."""
    rng = np.random.default_rng(seed)
    built = []
    for index in range(count):
        signal = float(rng.normal(0, 1))
        outcome = float(edge * signal + rng.normal(0, 0.02))
        built.append(
            TrainingRow(
                ticker="AAA",
                prediction_time=f"t{index:05d}",
                features={"rsi": 50.0 + signal, "volatility": 0.01},
                feature_contracts={
                    "rsi": {"calculation_version": "v1"},
                    "volatility": {"calculation_version": "v1"},
                },
                target_horizon=horizon,
                forward_return=outcome,
                label_up=outcome > 0,
                realized_vol=0.1,
                adverse_excursion=-0.01,
                label_version="lv1",
                label_record_hash=f"{horizon}-{index}",
                regime="bullish",
            )
        )
    return built


def dataset_for(horizon: str, count: int = 1500, *, edge: float = 0.0):
    rows = rows_for(horizon, count, edge=edge)
    registry = build_default_registry()
    digest = feature_set_hash(["rsi", "volatility"], registry)
    return TrainingDataset(
        rows=rows,
        feature_names=["rsi", "volatility"],
        target_horizon=horizon,
        dataset_hash=dataset_hash(rows, digest, horizon),
        feature_set_hash=digest,
    )


GEOMETRY = {"fold_sessions": 300, "embargo_sessions": 252, "holdout_sessions": 60}


class TrainingAcrossHorizonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_default_registry()
        cls.result = train_across_horizons(
            {h: dataset_for(h, edge=0.02) for h in TEMPORAL_HORIZONS},
            estimator="ridge",
            registry=cls.registry,
            **GEOMETRY,
        )

    def test_every_required_horizon_trains(self):
        self.assertEqual(sorted(self.result["runs"]), sorted(TEMPORAL_HORIZONS))

    def test_nothing_failed(self):
        self.assertEqual(self.result["failed"], {})

    def test_each_run_carries_its_own_horizon(self):
        for horizon, run in self.result["runs"].items():
            with self.subTest(horizon=horizon):
                self.assertEqual(run.target_horizon, horizon)

    def test_the_runs_are_distinct_datasets(self):
        # One dataset PER horizon: the label, and therefore which rows have a
        # matured outcome, differs per horizon. Sharing rows would leak.
        hashes = {run.dataset_hash for run in self.result["runs"].values()}
        self.assertEqual(len(hashes), len(self.result["runs"]))


class HorizonMetricsShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_across_horizons(
            {h: dataset_for(h, edge=0.02) for h in TEMPORAL_HORIZONS},
            estimator="ridge",
            registry=build_default_registry(),
            **GEOMETRY,
        )
        cls.metrics = horizon_metrics(cls.result["runs"])

    def test_every_horizon_reports_accuracy_and_observations(self):
        for horizon in TEMPORAL_HORIZONS:
            with self.subTest(horizon=horizon):
                entry = self.metrics[horizon]
                self.assertIsInstance(entry["directional_accuracy"], float)
                self.assertGreater(entry["observations"], 0)

    def test_x6_can_now_evaluate(self):
        report = evaluate_temporal_robustness(self.metrics)
        self.assertNotEqual(report["verdict"], "NOT_EVALUATED")

    def test_directional_accuracy_needs_no_label_up(self):
        # It is computed from the SIGN of the forward return. `label_up` is
        # emitted only for 20d, so depending on it would make every other
        # horizon unmeasurable.
        for horizon in TEMPORAL_HORIZONS:
            if horizon == "20d":
                continue
            with self.subTest(horizon=horizon):
                self.assertIn("directional_accuracy", self.metrics[horizon])


class AFailedHorizonIsNotAWeakOneTests(unittest.TestCase):
    """The distinction X6 rests on."""

    def test_a_horizon_that_cannot_train_is_recorded_as_failed(self):
        # 400 rows cannot hold 2*300 + 252 + 60.
        result = train_across_horizons(
            {"20d": dataset_for("20d", count=400)},
            estimator="ridge",
            registry=build_default_registry(),
            **GEOMETRY,
        )
        self.assertEqual(result["runs"], {})
        self.assertIn("20d", result["failed"])

    def test_a_failed_horizon_is_omitted_from_the_metrics(self):
        # NOT recorded as 0.0 accuracy: X6 distinguishes MISSING from "no edge",
        # and a zero would claim the second where the truth is the first.
        result = train_across_horizons(
            {"20d": dataset_for("20d", count=400)},
            estimator="ridge",
            registry=build_default_registry(),
            **GEOMETRY,
        )
        self.assertEqual(horizon_metrics(result["runs"]), {})

    def test_a_missing_horizon_makes_x6_report_missing(self):
        partial = {
            h: {"directional_accuracy": 0.55, "observations": 300}
            for h in list(TEMPORAL_HORIZONS)[:3]
        }
        report = evaluate_temporal_robustness(partial)
        self.assertEqual(report["verdict"], "NOT_EVALUATED")
        self.assertEqual(report.get("reason_code"), "HORIZONS_MISSING")

    def test_a_none_dataset_is_recorded_not_raised(self):
        result = train_across_horizons(
            {"20d": None}, estimator="ridge", registry=build_default_registry()
        )
        self.assertIn("20d", result["failed"])

    def test_an_empty_request_is_empty_not_fatal(self):
        for empty in ({}, None):
            with self.subTest(request=empty):
                result = train_across_horizons(empty, estimator="ridge")
                self.assertEqual(result["runs"], {})
                self.assertEqual(result["failed"], {})


class HorizonMetricsRefusesToInventTests(unittest.TestCase):
    def test_a_run_without_metrics_is_skipped(self):
        class Bare:
            metrics: dict = {}

        self.assertEqual(horizon_metrics({"20d": Bare()}), {})

    def test_a_run_missing_accuracy_is_skipped(self):
        class Partial:
            metrics = {"observations": 300}

        self.assertEqual(horizon_metrics({"20d": Partial()}), {})

    def test_a_none_run_is_skipped(self):
        self.assertEqual(horizon_metrics({"20d": None}), {})

    def test_a_dict_run_is_accepted(self):
        # Persisted runs come back as dicts, not dataclasses.
        metrics = horizon_metrics(
            {"20d": {"metrics": {"directional_accuracy": 0.6, "observations": 300}}}
        )
        self.assertEqual(metrics["20d"]["observations"], 300)


class EveryRequiredHorizonIsALabelHorizonTests(unittest.TestCase):
    """X6 cannot ask for a horizon the label builder cannot produce."""

    def test_the_required_horizons_are_all_labelled(self):
        for horizon in TEMPORAL_HORIZONS:
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, LABEL_HORIZON_SESSIONS)

    def test_the_longest_label_horizon_is_not_required_by_x6(self):
        # 252d is labelled but NOT in X6's required set, which is why A2's
        # geometry (embargo 252) can coexist with X6's five horizons.
        self.assertNotIn("252d", TEMPORAL_HORIZONS)


if __name__ == "__main__":
    unittest.main()
