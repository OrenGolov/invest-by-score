"""Research trial registry tests (Sprint M3).

The M3 purpose, pinned: "protects against uncontrolled experimentation and
accidental cherry-picking."

Concretely that means four properties must hold:

- every M3 field is recorded and validated;
- the hypothesis and primary metric are pre-registered, before results;
- a recorded result is immutable, and an identical configuration is the SAME
  experiment rather than a new one;
- abandoned trials are recorded, never deleted (file-drawer bias).
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from core.config import (
    OUTCOME_LABEL_VERSION,
    TRIAL_REGISTRY_VERSION,
    TRIAL_STATUS_ABANDONED,
    TRIAL_STATUS_COMPLETED,
    TRIAL_STATUS_REGISTERED,
)
from core.trial_registry import (
    Trial,
    TrialRegistry,
    TrialRegistryError,
    load_trial_records,
    load_trial_registry,
    persist_trial,
    trial_problems,
)

_AT = "2026-09-16T00:00:00+00:00"
_HYPOTHESIS = (
    "20d momentum features carry out-of-sample directional edge beyond the "
    "technical baseline"
)


def _trial(**overrides) -> Trial:
    payload = dict(
        hypothesis=_HYPOTHESIS,
        feature_set_version="fs-abc123",
        model_family="boosting",
        label_version=OUTCOME_LABEL_VERSION,
        horizons=["20d"],
        training_window={"start": "2021-01-01", "end": "2025-01-01"},
        validation_scheme="walk_forward_embargo",
        costs={"cost_table_version": "backtest-cost-table-v2", "slippage_model": "v2"},
        seed=42,
        dataset_hash="a" * 64,
        primary_metric="oos_sharpe",
        hyperparameters={"n_estimators": 300, "max_depth": 4},
    )
    payload.update(overrides)
    return Trial(**payload)


class TestRequiredFields(unittest.TestCase):
    """Every M3 field is recorded and validated."""

    def test_valid_trial_has_no_problems(self) -> None:
        self.assertEqual(trial_problems(_trial()), [])

    def test_all_m3_fields_are_recorded(self) -> None:
        recorded = _trial().to_dict()
        for required in (
            "trial_id", "hypothesis", "feature_set_version", "model_family",
            "hyperparameters", "label_version", "horizons", "training_window",
            "validation_scheme", "costs", "seed", "dataset_hash", "metrics",
        ):
            with self.subTest(field=required):
                self.assertIn(required, recorded)

    def test_empty_hypothesis_is_refused(self) -> None:
        self.assertTrue(any("hypothesis" in p for p in trial_problems(_trial(hypothesis=""))))

    def test_trivial_hypothesis_is_refused(self) -> None:
        """'it works' is not a hypothesis."""
        problems = trial_problems(_trial(hypothesis="it works"))
        self.assertTrue(any("hypothesis" in p for p in problems))

    def test_unknown_model_family_is_refused(self) -> None:
        problems = trial_problems(_trial(model_family="astrology"))
        self.assertTrue(any("model family" in p for p in problems))

    def test_unknown_validation_scheme_is_refused(self) -> None:
        problems = trial_problems(_trial(validation_scheme="eyeballing"))
        self.assertTrue(any("validation_scheme" in p for p in problems))

    def test_missing_costs_are_refused(self) -> None:
        """A backtest without explicit cost assumptions is not evidence."""
        self.assertTrue(any("costs" in p for p in trial_problems(_trial(costs={}))))

    def test_costs_must_name_their_table_version(self) -> None:
        problems = trial_problems(_trial(costs={"slippage": 1.0}))
        self.assertTrue(any("cost_table_version" in p for p in problems))

    def test_missing_dataset_hash_is_refused(self) -> None:
        problems = trial_problems(_trial(dataset_hash=""))
        self.assertTrue(any("dataset_hash" in p for p in problems))

    def test_unknown_horizon_is_refused(self) -> None:
        problems = trial_problems(_trial(horizons=["999d"]))
        self.assertTrue(any("horizon" in p for p in problems))

    def test_empty_horizons_are_refused(self) -> None:
        self.assertTrue(trial_problems(_trial(horizons=[])))

    def test_stale_label_version_is_refused(self) -> None:
        problems = trial_problems(_trial(label_version="outcome-label-v0"))
        self.assertTrue(any("label_version" in p for p in problems))

    def test_non_integer_seed_is_refused(self) -> None:
        self.assertTrue(any("seed" in p for p in trial_problems(_trial(seed="42"))))

    def test_training_window_needs_both_bounds(self) -> None:
        problems = trial_problems(_trial(training_window={"start": "2021-01-01"}))
        self.assertTrue(any("training_window.end" in p for p in problems))


class TestPreRegistration(unittest.TestCase):
    """The hypothesis and metric are recorded before the result exists."""

    def test_registered_trial_may_not_carry_metrics(self) -> None:
        problems = trial_problems(_trial(metrics={"oos_sharpe": 1.4}))
        self.assertTrue(any("must not carry metrics" in p for p in problems))

    def test_registry_refuses_a_trial_that_is_not_registered_status(self) -> None:
        """A trial cannot be born completed — metrics are attached later."""
        registry = TrialRegistry()
        with self.assertRaises(TrialRegistryError):
            registry.register(_trial(status=TRIAL_STATUS_COMPLETED))

    def test_registry_refuses_a_prefilled_completed_trial(self) -> None:
        """Even fully-formed, a completed trial cannot enter as a registration."""
        registry = TrialRegistry()
        prefilled = _trial(
            status=TRIAL_STATUS_COMPLETED,
            metrics={"oos_sharpe": 1.4},
            completed_at=_AT,
        )
        with self.assertRaises(TrialRegistryError) as ctx:
            registry.register(prefilled)
        self.assertIn("must enter the registry", str(ctx.exception))

    def test_primary_metric_is_required(self) -> None:
        problems = trial_problems(_trial(primary_metric=""))
        self.assertTrue(any("primary_metric" in p for p in problems))

    def test_completion_requires_the_pre_registered_metric(self) -> None:
        """The core anti-cherry-picking rule."""
        registry = TrialRegistry()
        trial = registry.register(_trial())
        with self.assertRaises(TrialRegistryError) as ctx:
            registry.complete(trial.trial_id, {"accuracy": 0.91}, _AT)
        self.assertIn("pre-registered primary metric", str(ctx.exception))

    def test_completion_succeeds_when_the_metric_is_reported(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        completed = registry.complete(
            trial.trial_id, {"oos_sharpe": 1.31, "accuracy": 0.55}, _AT
        )
        self.assertEqual(completed.status, TRIAL_STATUS_COMPLETED)
        self.assertEqual(completed.metrics["oos_sharpe"], 1.31)

    def test_a_losing_result_is_recorded_not_rejected(self) -> None:
        """A negative result is a valid scientific outcome and must be kept."""
        registry = TrialRegistry()
        trial = registry.register(_trial())
        completed = registry.complete(trial.trial_id, {"oos_sharpe": -0.4}, _AT)
        self.assertEqual(completed.status, TRIAL_STATUS_COMPLETED)
        self.assertEqual(completed.metrics["oos_sharpe"], -0.4)

    def test_empty_metrics_cannot_complete_a_trial(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        with self.assertRaises(TrialRegistryError):
            registry.complete(trial.trial_id, {}, _AT)


class TestImmutabilityAndIdentity(unittest.TestCase):
    def test_trial_id_is_deterministic(self) -> None:
        self.assertEqual(_trial().trial_id, _trial().trial_id)

    def test_identical_configuration_is_the_same_experiment(self) -> None:
        registry = TrialRegistry()
        registry.register(_trial())
        with self.assertRaises(TrialRegistryError) as ctx:
            registry.register(_trial())
        self.assertIn("already registered", str(ctx.exception))

    def test_changed_configuration_is_a_distinct_trial(self) -> None:
        for field_name, value in (
            ("seed", 7),
            ("model_family", "linear"),
            ("dataset_hash", "b" * 64),
            ("primary_metric", "brier_score"),
            ("hyperparameters", {"n_estimators": 10}),
            ("validation_scheme", "holdout"),
        ):
            with self.subTest(changed=field_name):
                self.assertNotEqual(_trial().trial_id, _trial(**{field_name: value}).trial_id)

    def test_changed_hypothesis_is_a_distinct_trial(self) -> None:
        """You cannot rewrite the hypothesis and keep the identity."""
        other = _trial(hypothesis="Macro regime tilt improves 60d directional accuracy")
        self.assertNotEqual(_trial().trial_id, other.trial_id)

    def test_metrics_do_not_change_the_trial_id(self) -> None:
        """Same experiment hashes the same before and after it runs."""
        registry = TrialRegistry()
        trial = registry.register(_trial())
        before = trial.trial_id
        registry.complete(trial.trial_id, {"oos_sharpe": 1.4}, _AT)
        self.assertEqual(trial.trial_id, before)

    def test_completed_trial_cannot_be_recompleted(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        registry.complete(trial.trial_id, {"oos_sharpe": 0.2}, _AT)
        with self.assertRaises(TrialRegistryError) as ctx:
            registry.complete(trial.trial_id, {"oos_sharpe": 9.9}, _AT)
        self.assertIn("immutable", str(ctx.exception))

    def test_completed_trial_cannot_be_abandoned(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        registry.complete(trial.trial_id, {"oos_sharpe": 0.2}, _AT)
        with self.assertRaises(TrialRegistryError):
            registry.abandon(trial.trial_id, "changed my mind", _AT)

    def test_tampered_configuration_is_detected(self) -> None:
        """Editing a registered trial breaks its id, and validation says so."""
        trial = _trial()
        trial.seed = 999
        self.assertTrue(any("does not match" in p for p in trial_problems(trial)))

    def test_unregistered_trial_cannot_be_completed(self) -> None:
        with self.assertRaises(TrialRegistryError) as ctx:
            TrialRegistry().complete("ghost", {"oos_sharpe": 1.0}, _AT)
        self.assertIn("not registered", str(ctx.exception))


class TestAbandonedTrialsAreKept(unittest.TestCase):
    """File-drawer bias is exactly what this registry exists to prevent."""

    def test_abandonment_requires_a_reason(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        with self.assertRaises(TrialRegistryError) as ctx:
            registry.abandon(trial.trial_id, "", _AT)
        self.assertIn("must record why", str(ctx.exception))

    def test_abandoned_trial_is_retained(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        registry.abandon(trial.trial_id, "data coverage too thin", _AT)
        kept = registry.get(trial.trial_id)
        self.assertIsNotNone(kept, "the trial was deleted")
        self.assertEqual(kept.status, TRIAL_STATUS_ABANDONED)
        self.assertEqual(kept.abandoned_reason, "data coverage too thin")

    def test_abandoned_trials_count_toward_multiple_testing(self) -> None:
        """The number people forget is the one that matters for X7."""
        registry = TrialRegistry()
        kept = registry.register(_trial())
        dropped = registry.register(_trial(seed=7))
        registry.complete(kept.trial_id, {"oos_sharpe": 1.2}, _AT)
        registry.abandon(dropped.trial_id, "looked bad early", _AT)

        report = registry.multiple_testing_report()
        self.assertEqual(report["distinct_trials"], 2)
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["abandoned"], 1)
        self.assertEqual(report["completed_by_primary_metric"], {"oos_sharpe": 1})

    def test_report_counts_still_registered_trials(self) -> None:
        registry = TrialRegistry()
        registry.register(_trial())
        self.assertEqual(registry.multiple_testing_report()["still_registered"], 1)


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "research_trials.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def test_ledger_shows_registration_before_completion(self) -> None:
        """The append order IS the evidence of pre-registration."""
        registry = TrialRegistry()
        trial = registry.register(_trial())
        persist_trial(trial, self.path)
        registry.complete(trial.trial_id, {"oos_sharpe": 1.4}, _AT)
        persist_trial(trial, self.path)

        records = load_trial_records(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], TRIAL_STATUS_REGISTERED)
        self.assertEqual(records[0]["metrics"], {})
        self.assertEqual(records[1]["status"], TRIAL_STATUS_COMPLETED)

    def test_reload_takes_the_newest_state(self) -> None:
        registry = TrialRegistry()
        trial = registry.register(_trial())
        persist_trial(trial, self.path)
        registry.complete(trial.trial_id, {"oos_sharpe": 1.4}, _AT)
        persist_trial(trial, self.path)

        reloaded = load_trial_registry(self.path)
        self.assertEqual(len(reloaded.all_trials()), 1)
        self.assertEqual(reloaded.get(trial.trial_id).status, TRIAL_STATUS_COMPLETED)

    def test_invalid_trial_is_not_persisted(self) -> None:
        with self.assertRaises(TrialRegistryError):
            persist_trial(_trial(costs={}), self.path)
        self.assertFalse(self.path.exists())

    def test_missing_store_reads_empty(self) -> None:
        self.assertEqual(load_trial_records(self.path), [])

    def test_malformed_line_raises_loudly(self) -> None:
        self.path.write_text("{not json\n", encoding="utf-8")
        with self.assertRaises(TrialRegistryError):
            load_trial_records(self.path)

    def test_registry_version_is_stamped(self) -> None:
        self.assertEqual(_trial().registry_version, TRIAL_REGISTRY_VERSION)


if __name__ == "__main__":
    unittest.main()
