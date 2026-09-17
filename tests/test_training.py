"""Offline training pipeline tests (Sprint M3, board numbering).

The board's M3 acceptance criteria, pinned:

- two training runs with identical manifests produce identical metrics and
  artifact hashes;
- a feature added without a registry entry aborts training;
- time-safe splits come from the V2 harness, not sklearn defaults;
- seeds fixed and recorded.
"""

from __future__ import annotations

import ast
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.config import (
    TRAINING_DEFAULT_SEED,
    TRAINING_PIPELINE_VERSION,
)
from core.feature_registry import build_default_registry
from core.training import (
    BASELINE_ESTIMATORS,
    TrainingError,
    load_training_runs,
    persist_training_run,
    train_baseline,
    train_baseline_suite,
    training_request_problems,
)
from core.training_dataset import build_training_dataset
from tests._dataset_fixture import load_frame, shared_dataset

_FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "data" / "NVDA_5y_1d.parquet"
_FOLDS = {"fold_sessions": 60, "embargo_sessions": 60, "holdout_sessions": 60}


def _frame() -> pd.DataFrame:
    frame = pd.read_parquet(_FIXTURE)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


class TrainingTestCase(unittest.TestCase):
    """Shares one built dataset — construction is the slow part."""

    @classmethod
    def setUpClass(cls) -> None:
        # Shared across every class in the suite: building this replays the
        # live scoring path and costs ~45s, and it was previously rebuilt
        # identically by each class.
        cls.frame = load_frame()
        cls.dataset = shared_dataset(-700, -400)


class TestRegistryGate(TrainingTestCase):
    def test_unregistered_feature_aborts_training(self) -> None:
        """Board acceptance: a feature without a registry entry aborts training."""
        dataset = self.dataset
        original = list(dataset.feature_names)
        dataset.feature_names = original + ["totally_unregistered_feature"]
        try:
            with self.assertRaises(TrainingError) as ctx:
                train_baseline(dataset, "ridge", **_FOLDS)
            self.assertIn("not registered", str(ctx.exception))
        finally:
            dataset.feature_names = original

    def test_dataset_without_hash_is_refused(self) -> None:
        dataset = self.dataset
        original = dataset.dataset_hash
        dataset.dataset_hash = ""
        try:
            problems = training_request_problems(dataset, "ridge")
            self.assertTrue(any("dataset_hash" in p for p in problems))
        finally:
            dataset.dataset_hash = original

    def test_unknown_estimator_is_refused(self) -> None:
        problems = training_request_problems(self.dataset, "neural_oracle")
        self.assertTrue(any("not a baseline" in p for p in problems))

    def test_valid_request_has_no_problems(self) -> None:
        self.assertEqual(training_request_problems(self.dataset, "ridge"), [])

    def test_every_baseline_passes_the_registry_gate(self) -> None:
        for estimator in BASELINE_ESTIMATORS:
            with self.subTest(estimator=estimator):
                self.assertEqual(training_request_problems(self.dataset, estimator), [])


class TestTimeSafeSplits(TrainingTestCase):
    """Splits come from the V2 harness, never sklearn's shuffling defaults."""

    def test_module_imports_no_sklearn_splitter(self) -> None:
        """sklearn's splitters shuffle by default and leak the future."""
        source = (
            pathlib.Path(__file__).resolve().parent.parent / "core" / "training.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
        forbidden = (
            "train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit",
            "cross_val_score", "cross_validate", "TimeSeriesSplit",
        )
        for name in imported:
            for banned in forbidden:
                with self.subTest(imported=name):
                    self.assertNotIn(banned, name)

    def test_training_window_precedes_validation_in_every_fold(self) -> None:
        run = train_baseline(self.dataset, "historical_mean", **_FOLDS)
        for fold in run.folds:
            with self.subTest(fold=fold.fold_id):
                self.assertLess(fold.train_end_time, fold.validation_start_time)

    def test_embargo_separates_train_from_validation(self) -> None:
        """The gap must exceed the longest label horizon."""
        run = train_baseline(self.dataset, "historical_mean", **_FOLDS)
        rows = sorted(self.dataset.rows, key=lambda r: (r.prediction_time, r.ticker))
        stamps = [row.prediction_time for row in rows]
        for fold in run.folds:
            with self.subTest(fold=fold.fold_id):
                gap = stamps.index(fold.validation_start_time) - stamps.index(
                    fold.train_end_time
                )
                self.assertGreaterEqual(gap, _FOLDS["embargo_sessions"])

    def test_multiple_folds_are_produced(self) -> None:
        run = train_baseline(self.dataset, "historical_mean", **_FOLDS)
        self.assertGreater(len(run.folds), 1)

    def test_metrics_are_out_of_sample(self) -> None:
        """Pooled observations equal the sum of validation rows, not train rows."""
        run = train_baseline(self.dataset, "historical_mean", **_FOLDS)
        expected = sum(fold.validation_rows for fold in run.folds)
        self.assertEqual(run.metrics["observations"], expected)


class TestDeterminism(TrainingTestCase):
    """Board acceptance: identical manifests produce identical results."""

    def test_repeated_runs_are_identical(self) -> None:
        for estimator in BASELINE_ESTIMATORS:
            with self.subTest(estimator=estimator):
                first = train_baseline(self.dataset, estimator, **_FOLDS)
                second = train_baseline(self.dataset, estimator, **_FOLDS)
                self.assertEqual(first.artifact_hash, second.artifact_hash)
                self.assertEqual(first.metrics, second.metrics)
                self.assertEqual(first.run_hash(), second.run_hash())

    def test_different_seed_changes_a_stochastic_artifact(self) -> None:
        first = train_baseline(self.dataset, "random_forest", seed=1, **_FOLDS)
        second = train_baseline(self.dataset, "random_forest", seed=2, **_FOLDS)
        self.assertNotEqual(first.artifact_hash, second.artifact_hash)

    def test_seed_is_recorded(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        self.assertEqual(run.seed, TRAINING_DEFAULT_SEED)
        self.assertEqual(run.hyperparameters["random_state"], TRAINING_DEFAULT_SEED)

    def test_environment_is_recorded(self) -> None:
        """M8 needs to know WHY two runs diverged, not just that they did."""
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        for key in ("python", "numpy", "sklearn", "platform"):
            with self.subTest(key=key):
                self.assertTrue(run.environment[key])

    def test_dataset_hash_is_carried_into_the_run(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        self.assertEqual(run.dataset_hash, self.dataset.dataset_hash)

    def test_pipeline_version_is_stamped(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        self.assertEqual(run.pipeline_version, TRAINING_PIPELINE_VERSION)


class TestBaselineSuite(TrainingTestCase):
    def test_suite_trains_every_baseline(self) -> None:
        runs = train_baseline_suite(self.dataset, **_FOLDS)
        self.assertEqual(set(runs), set(BASELINE_ESTIMATORS))

    def test_pure_baselines_need_no_sklearn(self) -> None:
        """historical_mean and momentum are numpy only."""
        for estimator in ("historical_mean", "momentum"):
            with self.subTest(estimator=estimator):
                run = train_baseline(self.dataset, estimator, **_FOLDS)
                self.assertGreater(len(run.folds), 0)

    def test_every_run_reports_the_core_metrics(self) -> None:
        runs = train_baseline_suite(self.dataset, **_FOLDS)
        for name, run in runs.items():
            with self.subTest(estimator=name):
                for metric in ("mae", "rmse", "directional_accuracy", "observations"):
                    self.assertIn(metric, run.metrics)

    def test_model_family_matches_the_estimator(self) -> None:
        self.assertEqual(train_baseline(self.dataset, "ridge", **_FOLDS).model_family, "linear")
        self.assertEqual(
            train_baseline(self.dataset, "random_forest", **_FOLDS).model_family, "tree"
        )

    def test_directional_accuracy_is_a_probability(self) -> None:
        runs = train_baseline_suite(self.dataset, **_FOLDS)
        for name, run in runs.items():
            with self.subTest(estimator=name):
                self.assertGreaterEqual(run.metrics["directional_accuracy"], 0.0)
                self.assertLessEqual(run.metrics["directional_accuracy"], 1.0)


class TestPersistence(TrainingTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "training_runs.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def test_persist_is_idempotent_per_run_hash(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        persist_training_run(run, self.path)
        persist_training_run(run, self.path)
        self.assertEqual(len(load_training_runs(self.path)), 1)

    def test_record_carries_reproduction_fields(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        record = persist_training_run(run, self.path)
        for key in (
            "run_hash", "artifact_hash", "dataset_hash", "feature_set_hash",
            "seed", "hyperparameters", "environment", "metrics",
        ):
            with self.subTest(key=key):
                self.assertIn(key, record)

    def test_run_without_artifact_hash_is_refused(self) -> None:
        run = train_baseline(self.dataset, "ridge", **_FOLDS)
        run.artifact_hash = ""
        with self.assertRaises(TrainingError):
            persist_training_run(run, self.path)

    def test_missing_store_reads_empty(self) -> None:
        self.assertEqual(load_training_runs(self.path), [])

    def test_malformed_line_raises_loudly(self) -> None:
        self.path.write_text("{not json\n", encoding="utf-8")
        with self.assertRaises(TrainingError):
            load_training_runs(self.path)


if __name__ == "__main__":
    unittest.main()
