"""Baseline model suite tests (Sprint M4).

The M4 rule, pinned: "No advanced model is promoted without beating the
incumbent out-of-sample."

Establishing baselines is the easy half. These tests cover the half that
matters — that the rule cannot be talked around:

- the incumbent is the best SIMPLE baseline, not the best model;
- comparisons must be like-for-like (same data, horizon, feature set);
- a win must be real: above the margin AND consistent across folds;
- a losing candidate cannot produce a promotion comparison;
- "nothing earned promotion" is a first-class result, not an error.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.baseline_suite import (
    SIMPLE_BASELINES,
    BaselineSuiteError,
    best_simple_baseline,
    compare_runs,
    comparison_problems,
    evaluate_suite,
    higher_is_better,
    promotion_comparison,
)
from core.config import (
    BASELINE_MIN_WINNING_FOLD_RATIO,
    BASELINE_PROMOTION_MARGIN,
    BASELINE_SUITE_VERSION,
)

_METRIC = "directional_accuracy"


def _run(
    estimator: str,
    value: float,
    fold_values: list[float] | None = None,
    metric: str = _METRIC,
    dataset_hash: str = "d" * 64,
    horizon: str = "20d",
    feature_hash: str = "f" * 64,
) -> SimpleNamespace:
    folds = fold_values if fold_values is not None else [value] * 5
    return SimpleNamespace(
        estimator=estimator,
        dataset_hash=dataset_hash,
        target_horizon=horizon,
        feature_set_hash=feature_hash,
        metrics={metric: value},
        folds=[
            SimpleNamespace(fold_id=index, metrics={metric: fold_value})
            for index, fold_value in enumerate(folds)
        ],
    )


class TestMetricDirection(unittest.TestCase):
    def test_accuracy_is_higher_better(self) -> None:
        self.assertTrue(higher_is_better("directional_accuracy"))

    def test_error_metrics_are_lower_better(self) -> None:
        for metric in ("rmse", "mae", "brier_score", "log_loss"):
            with self.subTest(metric=metric):
                self.assertFalse(higher_is_better(metric))

    def test_lower_is_better_metric_compares_correctly(self) -> None:
        """A smaller RMSE must count as a win, not a loss."""
        verdict = compare_runs(
            _run("ridge", 0.10, [0.10] * 5, metric="rmse"),
            _run("historical_mean", 0.20, [0.20] * 5, metric="rmse"),
            metric="rmse",
        )
        self.assertTrue(verdict.promoted)
        self.assertAlmostEqual(verdict.margin, 0.10, places=6)


class TestLikeForLike(unittest.TestCase):
    def test_different_dataset_is_refused(self) -> None:
        problems = comparison_problems(
            _run("ridge", 0.7), _run("historical_mean", 0.5, dataset_hash="e" * 64), _METRIC
        )
        self.assertTrue(any("dataset hashes differ" in p for p in problems))

    def test_different_horizon_is_refused(self) -> None:
        problems = comparison_problems(
            _run("ridge", 0.7), _run("historical_mean", 0.5, horizon="5d"), _METRIC
        )
        self.assertTrue(any("target horizons differ" in p for p in problems))

    def test_different_feature_set_is_refused(self) -> None:
        problems = comparison_problems(
            _run("ridge", 0.7), _run("historical_mean", 0.5, feature_hash="g" * 64), _METRIC
        )
        self.assertTrue(any("feature sets differ" in p for p in problems))

    def test_self_comparison_is_refused(self) -> None:
        problems = comparison_problems(_run("ridge", 0.7), _run("ridge", 0.5), _METRIC)
        self.assertTrue(any("against itself" in p for p in problems))

    def test_missing_metric_is_refused(self) -> None:
        problems = comparison_problems(_run("ridge", 0.7), _run("historical_mean", 0.5), "sharpe")
        self.assertTrue(any("does not report" in p for p in problems))

    def test_incomparable_runs_raise(self) -> None:
        with self.assertRaises(BaselineSuiteError):
            compare_runs(_run("ridge", 0.9), _run("historical_mean", 0.5, horizon="5d"))

    def test_comparable_runs_have_no_problems(self) -> None:
        self.assertEqual(
            comparison_problems(_run("ridge", 0.7), _run("historical_mean", 0.5), _METRIC), []
        )


class TestPromotionRule(unittest.TestCase):
    def test_genuine_consistent_winner_is_promoted(self) -> None:
        """The rule must be able to PASS — one that never passes proves nothing."""
        verdict = compare_runs(
            _run("gradient_boosting", 0.65, [0.66, 0.64, 0.67, 0.63, 0.65]),
            _run("historical_mean", 0.55, [0.54, 0.56, 0.55, 0.53, 0.57]),
        )
        self.assertTrue(verdict.promoted)
        self.assertEqual(verdict.reasons, [])
        self.assertEqual(verdict.winning_folds, 5)

    def test_losing_candidate_is_refused(self) -> None:
        verdict = compare_runs(_run("ridge", 0.45), _run("historical_mean", 0.60))
        self.assertFalse(verdict.promoted)
        self.assertTrue(any("does not beat incumbent" in r for r in verdict.reasons))

    def test_thin_margin_is_refused(self) -> None:
        """Beating the incumbent by a hair is noise."""
        verdict = compare_runs(
            _run("ridge", 0.5501, [0.56] * 5), _run("historical_mean", 0.55, [0.55] * 5)
        )
        self.assertFalse(verdict.promoted)
        self.assertTrue(any("below the required" in r for r in verdict.reasons))

    def test_exactly_at_the_margin_is_refused(self) -> None:
        """The bar is strictly above the margin, so it cannot be gamed by equality."""
        incumbent_value = 0.55
        candidate_value = incumbent_value + BASELINE_PROMOTION_MARGIN - 1e-9
        verdict = compare_runs(
            _run("ridge", candidate_value, [candidate_value] * 5),
            _run("historical_mean", incumbent_value, [incumbent_value] * 5),
        )
        self.assertFalse(verdict.promoted)

    def test_one_lucky_fold_is_refused(self) -> None:
        """A single outsized fold must not carry a promotion."""
        verdict = compare_runs(
            _run("ridge", 0.60, [0.95, 0.50, 0.50, 0.50, 0.55]),
            _run("historical_mean", 0.55, [0.55] * 5),
        )
        self.assertFalse(verdict.promoted)
        self.assertTrue(any("not consistent" in r for r in verdict.reasons))
        self.assertLess(verdict.winning_folds / verdict.total_folds, BASELINE_MIN_WINNING_FOLD_RATIO)

    def test_verdict_records_the_comparison(self) -> None:
        verdict = compare_runs(_run("ridge", 0.45), _run("historical_mean", 0.60))
        payload = verdict.to_dict()
        for key in (
            "candidate", "incumbent", "metric", "candidate_value",
            "incumbent_value", "margin", "winning_folds", "total_folds",
            "promoted", "reasons",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(payload["suite_version"], BASELINE_SUITE_VERSION)


class TestIncumbentSelection(unittest.TestCase):
    def test_incumbent_is_the_best_simple_baseline(self) -> None:
        """Not the best model — a trained model must clear the simple bar."""
        runs = {
            "historical_mean": _run("historical_mean", 0.55),
            "momentum": _run("momentum", 0.51),
            "mean_reversion": _run("mean_reversion", 0.49),
            "gradient_boosting": _run("gradient_boosting", 0.80),
        }
        self.assertEqual(best_simple_baseline(runs).estimator, "historical_mean")

    def test_strongest_simple_baseline_wins_selection(self) -> None:
        runs = {
            "historical_mean": _run("historical_mean", 0.50),
            "momentum": _run("momentum", 0.58),
            "mean_reversion": _run("mean_reversion", 0.49),
        }
        self.assertEqual(best_simple_baseline(runs).estimator, "momentum")

    def test_missing_simple_baseline_raises(self) -> None:
        with self.assertRaises(BaselineSuiteError) as ctx:
            best_simple_baseline({"ridge": _run("ridge", 0.9)})
        self.assertIn("no simple baseline", str(ctx.exception))

    def test_all_three_simple_baselines_are_recognised(self) -> None:
        self.assertEqual(
            set(SIMPLE_BASELINES), {"historical_mean", "momentum", "mean_reversion"}
        )


class TestSuiteEvaluation(unittest.TestCase):
    def _suite(self, boosting_value: float) -> dict:
        return {
            "historical_mean": _run("historical_mean", 0.55, [0.55] * 5),
            "momentum": _run("momentum", 0.50, [0.50] * 5),
            "mean_reversion": _run("mean_reversion", 0.49, [0.49] * 5),
            "gradient_boosting": _run(
                "gradient_boosting", boosting_value, [boosting_value] * 5
            ),
        }

    def test_nothing_promotable_is_a_first_class_result(self) -> None:
        """Not an exception, and not a silently chosen best-of."""
        report = evaluate_suite(self._suite(0.45))
        self.assertFalse(report["any_promotable"])
        self.assertEqual(report["promotable"], [])
        self.assertIn("no candidate beat", report["summary"])

    def test_a_real_winner_is_reported(self) -> None:
        report = evaluate_suite(self._suite(0.70))
        self.assertTrue(report["any_promotable"])
        self.assertEqual(report["promotable"], ["gradient_boosting"])

    def test_incumbent_is_excluded_from_its_own_verdicts(self) -> None:
        report = evaluate_suite(self._suite(0.45))
        self.assertNotIn(report["incumbent"], report["verdicts"])

    def test_every_candidate_is_judged(self) -> None:
        report = evaluate_suite(self._suite(0.45))
        self.assertEqual(
            set(report["verdicts"]), {"momentum", "mean_reversion", "gradient_boosting"}
        )

    def test_empty_suite_raises(self) -> None:
        with self.assertRaises(BaselineSuiteError):
            evaluate_suite({})


class TestPromotionComparison(unittest.TestCase):
    """The bridge into the M2 promotion gate."""

    def _winner(self):
        return compare_runs(
            _run("gradient_boosting", 0.65, [0.66, 0.64, 0.67, 0.63, 0.65]),
            _run("historical_mean", 0.55, [0.54, 0.56, 0.55, 0.53, 0.57]),
        )

    def test_losing_candidate_cannot_produce_a_comparison(self) -> None:
        """The gate and the measurement must not be able to disagree."""
        losing = compare_runs(_run("ridge", 0.45), _run("historical_mean", 0.60))
        with self.assertRaises(BaselineSuiteError) as ctx:
            promotion_comparison(losing, "ridge-v1", "hm-v1")
        self.assertIn("did not earn promotion", str(ctx.exception))

    def test_comparison_has_the_fields_the_m2_gate_requires(self) -> None:
        comparison = promotion_comparison(self._winner(), "gb-v1", "hm-v1")
        for key in ("primary_metric", "candidate_value", "incumbent_value", "sample"):
            with self.subTest(key=key):
                self.assertIsNotNone(comparison.get(key))

    def test_m2_gate_accepts_a_real_comparison(self) -> None:
        """End to end: a measured win actually promotes a model."""
        from core.model_registry import ModelEntry, build_default_model_registry

        registry = build_default_model_registry()
        registry.register(
            ModelEntry(
                model_version="gb-v1", family="boosting", feature_set_version="fs-1"
            )
        )
        incumbent = registry.incumbent("technical_analysis")
        comparison = promotion_comparison(
            self._winner(), "gb-v1", incumbent.model_version
        )
        entry = registry.promote(
            "gb-v1", "oren", "2026-09-17T00:00:00+00:00", comparison
        )
        self.assertEqual(entry.status, "approved")
        self.assertEqual(entry.oos_comparison["primary_metric"], _METRIC)


if __name__ == "__main__":
    unittest.main()
