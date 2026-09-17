"""ML reproducibility gate tests (Sprint M8).

The M8 rule, pinned: "Same data, features, seed, code, configuration must
produce identical model artifacts/predictions WITHIN EXPLICITLY DEFINED
REPRODUCIBILITY GUARANTEES."

Two things are tested, and the second matters as much as the first:

- the guarantee HOLDS — retraining the same configuration produces identical
  artifacts, predictions and metrics, across separate processes;
- the guarantee is HONEST — what is not promised (cross-version, cross-
  platform bitwise identity) is stated, and a divergence there is reported
  as expected rather than as a defect.
"""

from __future__ import annotations

import pathlib
import unittest
from types import SimpleNamespace

import pandas as pd

from core.config import (
    REPRODUCIBILITY_ENVIRONMENT_KEYS,
    REPRODUCIBILITY_GUARANTEES,
    REPRODUCIBILITY_NON_GUARANTEES,
    REPRODUCIBILITY_TOLERANCE,
    REPRODUCIBILITY_VERSION,
)
from core.reproducibility import (
    ReproducibilityError,
    compare_predictions,
    compare_runs,
    environment_differences,
    predictions_of,
    reproducibility_manifest,
    verify_reproducible,
)
from core.training import BASELINE_ESTIMATORS, train_baseline
from core.training_dataset import build_training_dataset
from tests._dataset_fixture import shared_dataset

_FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "data" / "NVDA_5y_1d.parquet"
_FOLDS = {"fold_sessions": 60, "embargo_sessions": 60, "holdout_sessions": 60}

# Deterministic algorithms: the seed is inert, so the fitted parameters — and
# therefore the artifact — must be identical regardless of it.
_DETERMINISTIC = ("historical_mean", "momentum", "mean_reversion", "ridge", "elastic_net")
# Stochastic: the seed genuinely changes the fit.
_STOCHASTIC = ("random_forest", "gradient_boosting")


class ReproducibilityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = shared_dataset(-700, -400)


class TestGuaranteeHolds(ReproducibilityTestCase):
    def test_every_baseline_is_reproducible(self) -> None:
        """The core M8 assertion, re-trained rather than re-read."""
        for estimator in BASELINE_ESTIMATORS:
            with self.subTest(estimator=estimator):
                report = verify_reproducible(self.dataset, estimator, seed=42, **_FOLDS)
                self.assertTrue(report.reproducible, report.divergences)

    def test_identical_runs_share_every_identity(self) -> None:
        first = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        second = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        self.assertEqual(first.artifact_hash, second.artifact_hash)
        self.assertEqual(first.run_hash(), second.run_hash())
        self.assertEqual(first.metrics, second.metrics)

    def test_predictions_are_bit_identical(self) -> None:
        first = train_baseline(self.dataset, "gradient_boosting", seed=42, **_FOLDS)
        second = train_baseline(self.dataset, "gradient_boosting", seed=42, **_FOLDS)
        self.assertEqual(predictions_of(first), predictions_of(second))
        self.assertEqual(compare_predictions(predictions_of(first), predictions_of(second)), [])

    def test_per_fold_metrics_match(self) -> None:
        report = verify_reproducible(self.dataset, "random_forest", seed=7, **_FOLDS)
        self.assertTrue(report.reproducible)
        self.assertEqual(report.divergences, [])


class TestArtifactVersusRunIdentity(ReproducibilityTestCase):
    """Two identities, deliberately separate."""

    def test_seed_does_not_change_a_deterministic_artifact(self) -> None:
        """Regression: the seed used to be hashed into the artifact.

        historical_mean and ridge do not consult the seed, so a seed change
        produced an identical model but a different artifact hash — a false
        difference, and exactly the noise that trains people to ignore a gate.
        """
        for estimator in _DETERMINISTIC:
            with self.subTest(estimator=estimator):
                first = train_baseline(self.dataset, estimator, seed=1, **_FOLDS)
                second = train_baseline(self.dataset, estimator, seed=2, **_FOLDS)
                self.assertEqual(first.artifact_hash, second.artifact_hash)

    def test_seed_does_change_a_stochastic_artifact(self) -> None:
        for estimator in _STOCHASTIC:
            with self.subTest(estimator=estimator):
                first = train_baseline(self.dataset, estimator, seed=1, **_FOLDS)
                second = train_baseline(self.dataset, estimator, seed=2, **_FOLDS)
                self.assertNotEqual(first.artifact_hash, second.artifact_hash)

    def test_run_hash_always_tracks_the_seed(self) -> None:
        """Configuration identity includes the seed even when the fit ignores it."""
        for estimator in _DETERMINISTIC + _STOCHASTIC:
            with self.subTest(estimator=estimator):
                first = train_baseline(self.dataset, estimator, seed=1, **_FOLDS)
                second = train_baseline(self.dataset, estimator, seed=2, **_FOLDS)
                self.assertNotEqual(first.run_hash(), second.run_hash())

    def test_artifact_hash_basis_is_declared(self) -> None:
        run = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        self.assertEqual(run.artifact_hash_basis, "fitted_parameters")


class TestDivergenceDetection(ReproducibilityTestCase):
    def test_a_different_seed_is_reported_as_divergent(self) -> None:
        first = train_baseline(self.dataset, "random_forest", seed=1, **_FOLDS)
        second = train_baseline(self.dataset, "random_forest", seed=2, **_FOLDS)
        report = compare_runs(first, second)
        self.assertFalse(report.reproducible)
        self.assertTrue(any("seed" in d for d in report.divergences))

    def test_same_environment_divergence_is_called_a_defect(self) -> None:
        first = train_baseline(self.dataset, "random_forest", seed=1, **_FOLDS)
        second = train_baseline(self.dataset, "random_forest", seed=2, **_FOLDS)
        self.assertIn("defect", compare_runs(first, second).explanation())

    def test_cross_environment_divergence_is_called_expected(self) -> None:
        """What is not guaranteed must not be reported as a bug."""
        first = train_baseline(self.dataset, "random_forest", seed=1, **_FOLDS)
        second = train_baseline(self.dataset, "random_forest", seed=2, **_FOLDS)
        second.environment = {**second.environment, "sklearn": "1.5.0"}
        report = compare_runs(first, second)
        self.assertFalse(report.environment_matches)
        self.assertIn("not a defect", report.explanation())

    def test_prediction_mismatch_is_detected(self) -> None:
        self.assertTrue(compare_predictions([0.1, 0.2], [0.1, 0.3]))

    def test_prediction_count_mismatch_is_detected(self) -> None:
        self.assertTrue(compare_predictions([0.1], [0.1, 0.2]))

    def test_comparison_tolerance_is_exactly_zero(self) -> None:
        """A tolerance would hide the nondeterminism this gate exists to catch."""
        self.assertEqual(REPRODUCIBILITY_TOLERANCE, 0.0)
        self.assertTrue(compare_predictions([0.1], [0.1 + 1e-15]))

    def test_comparing_different_estimators_is_refused(self) -> None:
        first = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        second = train_baseline(self.dataset, "historical_mean", seed=42, **_FOLDS)
        with self.assertRaises(ReproducibilityError):
            compare_runs(first, second)

    def test_a_run_without_folds_is_refused(self) -> None:
        with self.assertRaises(ReproducibilityError):
            compare_runs(SimpleNamespace(folds=[]), SimpleNamespace(folds=[]))


class TestEnvironmentAttribution(ReproducibilityTestCase):
    def test_every_environment_key_is_recorded(self) -> None:
        run = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        for key in REPRODUCIBILITY_ENVIRONMENT_KEYS:
            with self.subTest(key=key):
                self.assertTrue(run.environment.get(key))

    def test_identical_environments_report_no_difference(self) -> None:
        run = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        self.assertEqual(environment_differences(run.environment, run.environment), [])

    def test_a_version_change_is_named(self) -> None:
        differences = environment_differences({"numpy": "2.5.2"}, {"numpy": "1.26.0"})
        self.assertTrue(any("numpy" in d for d in differences))


class TestManifest(ReproducibilityTestCase):
    def test_manifest_carries_everything_needed_to_reproduce(self) -> None:
        manifest = reproducibility_manifest(
            train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        )
        for key in (
            "artifact_hash", "run_hash", "dataset_hash", "feature_set_hash",
            "seed", "hyperparameters", "pipeline_version", "environment",
        ):
            with self.subTest(key=key):
                self.assertIn(key, manifest)

    def test_manifest_states_the_terms_of_the_promise(self) -> None:
        """A future reader must see what was guaranteed AT THE TIME."""
        manifest = reproducibility_manifest(
            train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        )
        self.assertEqual(manifest["guarantees"], list(REPRODUCIBILITY_GUARANTEES))
        self.assertEqual(manifest["non_guarantees"], list(REPRODUCIBILITY_NON_GUARANTEES))
        self.assertEqual(manifest["version"], REPRODUCIBILITY_VERSION)

    def test_a_run_missing_environment_is_refused(self) -> None:
        run = train_baseline(self.dataset, "ridge", seed=42, **_FOLDS)
        run.environment = {"python": "3.12.10"}
        with self.assertRaises(ReproducibilityError) as ctx:
            reproducibility_manifest(run)
        self.assertIn("could not be attributed", str(ctx.exception))


class TestGuaranteesAreCoherent(unittest.TestCase):
    def test_no_property_is_both_guaranteed_and_not(self) -> None:
        self.assertEqual(
            set(REPRODUCIBILITY_GUARANTEES) & set(REPRODUCIBILITY_NON_GUARANTEES), set()
        )

    def test_cross_version_identity_is_explicitly_not_promised(self) -> None:
        """Because it would be false, and a false guarantee is worse than none."""
        self.assertIn("cross_library_version_bitwise", REPRODUCIBILITY_NON_GUARANTEES)
        self.assertIn("cross_platform_bitwise", REPRODUCIBILITY_NON_GUARANTEES)


if __name__ == "__main__":
    unittest.main()
