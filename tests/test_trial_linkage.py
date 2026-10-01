"""A5 tests — one trial per estimator, registered automatically, linked to its run.

MEASURED: `data/research_trials.jsonl` holds 2 rows that are 1 DISTINCT trial
against 8 trained estimators, so X7 must count the family from the runs and report
the registry as unreliable. Two causes, both covered here.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from core.feature_registry import build_default_registry, feature_set_hash
from core.multiple_testing import family_size
from core.training import (
    load_training_runs,
    persist_training_run,
    train_baseline,
    trial_for_run,
)
from core.training_dataset import TrainingDataset, TrainingRow, dataset_hash
from core.trial_registry import TRIAL_STATUS_REGISTERED, persist_trial

ESTIMATORS = ("ridge", "elastic_net", "historical_mean", "momentum")


def dataset(count=1500):
    rng = np.random.default_rng(4)
    rows = [
        TrainingRow(
            ticker="AAA",
            prediction_time=f"t{index:05d}",
            features={"rsi": float(rng.normal(50, 10)), "volatility": 0.01},
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
            label_record_hash=f"h{index}",
        )
        for index in range(count)
    ]
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


class OneTrialPerEstimatorTests(unittest.TestCase):
    """Cause 1: every estimator was passed as ONE hyperparameters entry."""

    @classmethod
    def setUpClass(cls):
        data, registry = dataset()
        cls.runs = {
            estimator: train_baseline(
                data,
                estimator=estimator,
                fold_sessions=300,
                embargo_sessions=252,
                holdout_sessions=60,
                registry=registry,
            )
            for estimator in ESTIMATORS
        }
        cls.trials = {
            estimator: trial_for_run(run) for estimator, run in cls.runs.items()
        }

    def test_each_estimator_gets_a_distinct_trial_id(self):
        # `trial_id` hashes the configuration, so the estimator must be part of
        # it. Before A5 four estimators produced ONE id.
        ids = {trial.trial_id for trial in self.trials.values()}
        self.assertEqual(len(ids), len(ESTIMATORS))

    def test_the_estimator_is_in_the_configuration(self):
        for estimator, trial in self.trials.items():
            with self.subTest(estimator=estimator):
                self.assertEqual(trial.hyperparameters["estimator"], estimator)

    def test_the_trial_agrees_with_the_run_it_describes(self):
        # Derived FROM the run, so the dataset hash and seed cannot drift from
        # what was actually fitted.
        for estimator, trial in self.trials.items():
            run = self.runs[estimator]
            with self.subTest(estimator=estimator):
                self.assertEqual(trial.dataset_hash, run.dataset_hash)
                self.assertEqual(trial.feature_set_version, run.feature_set_hash)
                self.assertEqual(trial.seed, run.seed)
                self.assertEqual(trial.horizons, [run.target_horizon])

    def test_the_same_run_yields_the_same_trial_id(self):
        # An identical configuration is the SAME experiment, so re-registering
        # cannot inflate the count.
        first = trial_for_run(self.runs["ridge"])
        second = trial_for_run(self.runs["ridge"])
        self.assertEqual(first.trial_id, second.trial_id)


class ARegisteredTrialCarriesNoMetricsTests(unittest.TestCase):
    """Cause 2 of my own first version: pre-registration must precede results."""

    def setUp(self):
        data, registry = dataset()
        self.run = train_baseline(
            data,
            estimator="ridge",
            fold_sessions=300,
            embargo_sessions=252,
            holdout_sessions=60,
            registry=registry,
        )

    def test_the_trial_carries_no_metrics(self):
        # M3 refuses a REGISTERED trial with metrics: the append-only ledger
        # showing registration BEFORE completion is the evidence of
        # pre-registration. A first version passed the run's metrics and the
        # registry rejected it.
        self.assertFalse(trial_for_run(self.run).metrics)

    def test_the_trial_is_accepted_by_the_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trials.jsonl"
            record = persist_trial(trial_for_run(self.run), path)
            self.assertEqual(record["status"], TRIAL_STATUS_REGISTERED)

    def test_a_generated_hypothesis_satisfies_the_registry(self):
        # M3 requires at least 20 characters. Defaulting inside `trial_for_run`
        # means an empty string cannot reach the registry from any caller.
        trial = trial_for_run(self.run, hypothesis="")
        self.assertGreaterEqual(len(trial.hypothesis), 20)
        self.assertIn("ridge", trial.hypothesis)

    def test_a_supplied_hypothesis_is_kept(self):
        stated = "Momentum features predict 20d returns in trending regimes only"
        self.assertEqual(
            trial_for_run(self.run, hypothesis=stated).hypothesis, stated
        )


class PersistingRegistersTests(unittest.TestCase):
    """Registration is no longer opt-in."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.registry = dataset()

    def train(self, estimator):
        return train_baseline(
            self.data,
            estimator=estimator,
            fold_sessions=300,
            embargo_sessions=252,
            holdout_sessions=60,
            registry=self.registry,
        )

    def test_a_persisted_run_records_its_trial_id(self):
        with tempfile.TemporaryDirectory() as directory:
            store = pathlib.Path(directory) / "runs.jsonl"
            record = persist_training_run(
                self.train("ridge"), store, register_trial=False
            )
            self.assertIn("trial_id", record)

    def test_persisting_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = pathlib.Path(directory) / "runs.jsonl"
            run = self.train("ridge")
            persist_training_run(run, store, register_trial=False)
            persist_training_run(run, store, register_trial=False)
            self.assertEqual(len(load_training_runs(store)), 1)

    def test_registration_is_on_by_default(self):
        import inspect

        signature = inspect.signature(persist_training_run)
        self.assertIs(signature.parameters["register_trial"].default, True)


class TheRegistryGapClosesTests(unittest.TestCase):
    """What X7 measures. The gap was 8 observed against 1 registered."""

    def test_four_estimators_registered_leave_no_gap(self):
        data, registry = dataset()
        runs, trials = [], []
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trials.jsonl"
            for estimator in ESTIMATORS:
                run = train_baseline(
                    data,
                    estimator=estimator,
                    fold_sessions=300,
                    embargo_sessions=252,
                    holdout_sessions=60,
                    registry=registry,
                )
                trial = trial_for_run(run)
                persist_trial(trial, path)
                runs.append({"estimator": estimator})
                trials.append(
                    {"trial_id": trial.trial_id, "model_family": trial.model_family}
                )
            written = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len({r["trial_id"] for r in written}), len(ESTIMATORS))
        counts = family_size(runs, trials)
        self.assertEqual(counts["observed_count"], len(ESTIMATORS))
        self.assertEqual(counts["registered_count"], len(ESTIMATORS))
        self.assertEqual(counts["registry_gap"], 0)

    def test_the_shipped_ledger_no_longer_shows_a_gap(self):
        """RESTATED after the ledger was regenerated under A5.

        The original 8 runs predated A5 and shared ONE trial. Regenerating the
        ledger wrote 8 more, each registering its own trial, so the gap is
        closed for the current cohort. The append-only registry keeps the old
        trial, which is why the count can exceed the run count.
        """
        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        trials_path = pathlib.Path("data/research_trials.jsonl")
        if not trials_path.exists():
            self.skipTest("no trial ledger on disk")
        trials = [
            json.loads(line)
            for line in trials_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        newest = runs[-1].get("dataset_hash")
        cohort = [run for run in runs if run.get("dataset_hash") == newest] or runs
        counts = family_size(cohort, trials)
        self.assertGreaterEqual(
            counts["registered_count"], counts["observed_count"]
        )


class TheCollapsingHelperIsGoneTests(unittest.TestCase):
    def test_train_py_no_longer_registers_one_trial_per_batch(self):
        source = pathlib.Path("scripts/train.py").read_text(encoding="utf-8")
        self.assertNotIn("def _register_trial(", source)
        self.assertIn("persist_training_run(run, hypothesis=", source)


if __name__ == "__main__":
    unittest.main()
