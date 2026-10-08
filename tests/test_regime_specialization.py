"""L7 regime-specialization tests.

The behaviour under test is RESTRAINT. The task says to specialize *only if*
OOS evidence supports it, and MEASURED, specialization loses on thin data even
when the skill genuinely differs by regime.
"""

from __future__ import annotations

import random
import statistics
import unittest

from core.config import (
    REGIME_LABELS,
    REGIME_SPEC_INSUFFICIENT,
    REGIME_SPEC_NOT_EVALUATED,
    REGIME_SPEC_POOLED,
    REGIME_SPEC_SPECIALIZED,
    REGIME_SPECIALIZATION_DEFAULT_POOLED,
    REGIME_SPECIALIZATION_FALLBACK_POOLED,
    REGIME_SPECIALIZATION_FOLDS,
    REGIME_SPECIALIZATION_FOLDS_REQUIRED,
    REGIME_SPECIALIZATION_LABELS,
    REGIME_SPECIALIZATION_MARGIN,
    REGIME_SPECIALIZATION_MIN_CELL,
    REGIME_SPECIALIZATION_PER_REGIME,
    REGIME_SPECIALIZATION_VERDICTS,
)
from core.regime_specialization import (
    RegimeSpecializationError,
    brier,
    evaluate_regime,
    fold_comparison,
    relative_improvement,
    render_specialization,
    specialization_problems,
    specialization_report,
)

FOLDS = REGIME_SPECIALIZATION_FOLDS
SIZE = max(int(REGIME_SPECIALIZATION_MIN_CELL) * 3, 180)


def folds(count, size, pooled_p, specialized_p, true_p, seed):
    rng = random.Random(seed)
    return [
        {
            "pooled": [pooled_p] * size,
            "specialized": [specialized_p] * size,
            "outcomes": [1 if rng.random() < true_p else 0 for _ in range(size)],
        }
        for _ in range(count)
    ]


def noisy_folds(count, size, seed):
    """A specialized model fitted on its own finite sample — real noise."""
    rng = random.Random(seed)
    built = []
    for _ in range(count):
        outcomes = [1 if rng.random() < 0.55 else 0 for _ in range(size)]
        fitted = statistics.mean(
            [1 if rng.random() < 0.55 else 0 for _ in range(size)]
        )
        built.append(
            {"pooled": [0.55] * size, "specialized": [fitted] * size,
             "outcomes": outcomes}
        )
    return built


class DefaultIsPooledTests(unittest.TestCase):
    def test_pooled_is_the_declared_default(self):
        self.assertTrue(REGIME_SPECIALIZATION_DEFAULT_POOLED)

    def test_a_regime_with_no_difference_stays_pooled(self):
        result = evaluate_regime("range", folds(FOLDS, SIZE, 0.55, 0.55, 0.55, 1))
        self.assertEqual(result["verdict"], REGIME_SPEC_POOLED)

    def test_noise_is_rarely_adopted(self):
        adopted = sum(
            1
            for seed in range(60)
            if evaluate_regime("bearish", noisy_folds(FOLDS, 150, 700 + seed))[
                "verdict"
            ]
            == REGIME_SPEC_SPECIALIZED
        )
        self.assertLessEqual(adopted, 2, f"{adopted}/60 noise regimes adopted")


class RealSpecializationTests(unittest.TestCase):
    """Strictness must be a filter, not a gag."""

    def test_a_genuinely_better_model_is_adopted(self):
        result = evaluate_regime("bullish", folds(FOLDS, SIZE, 0.55, 0.70, 0.70, 2))
        self.assertEqual(result["verdict"], REGIME_SPEC_SPECIALIZED)

    def test_an_adopted_verdict_carries_its_gain(self):
        result = evaluate_regime("bullish", folds(FOLDS, SIZE, 0.55, 0.70, 0.70, 3))
        self.assertGreater(result["median_gain"], 0)
        self.assertGreaterEqual(result["folds_won"], REGIME_SPECIALIZATION_FOLDS_REQUIRED)

    def test_a_worse_specialized_model_is_rejected(self):
        result = evaluate_regime("bullish", folds(FOLDS, SIZE, 0.70, 0.30, 0.70, 4))
        self.assertEqual(result["verdict"], REGIME_SPEC_POOLED)


class RareRegimeTests(unittest.TestCase):
    """Stress is ~5% of observations; its estimate is the noise."""

    def test_a_thin_regime_is_refused(self):
        result = evaluate_regime("stress", folds(FOLDS, 20, 0.55, 0.70, 0.70, 5))
        self.assertEqual(result["verdict"], REGIME_SPEC_INSUFFICIENT)

    def test_insufficient_is_not_pooled(self):
        self.assertNotEqual(REGIME_SPEC_INSUFFICIENT, REGIME_SPEC_POOLED)

    def test_an_untested_regime_is_not_evaluated(self):
        self.assertEqual(
            evaluate_regime("stress", [])["verdict"], REGIME_SPEC_NOT_EVALUATED
        )

    def test_too_few_testable_folds_is_insufficient(self):
        mixed = folds(2, SIZE, 0.55, 0.70, 0.70, 6) + folds(3, 10, 0.55, 0.70, 0.70, 7)
        self.assertEqual(
            evaluate_regime("stress", mixed)["verdict"], REGIME_SPEC_INSUFFICIENT
        )

    def test_the_cell_floor_supports_a_rate(self):
        self.assertGreaterEqual(REGIME_SPECIALIZATION_MIN_CELL, 60)


class EvidenceRuleTests(unittest.TestCase):
    def test_the_margin_is_positive(self):
        self.assertGreater(REGIME_SPECIALIZATION_MARGIN, 0)

    def test_the_required_folds_are_a_majority(self):
        self.assertGreater(
            REGIME_SPECIALIZATION_FOLDS_REQUIRED * 2, REGIME_SPECIALIZATION_FOLDS
        )

    def test_a_fold_win_needs_the_margin(self):
        outcomes = [1] * 100
        tiny = fold_comparison([0.80] * 100, [0.8001] * 100, outcomes)
        self.assertFalse(tiny["won"])

    def test_a_clear_fold_win_counts(self):
        outcomes = [1] * 100
        clear = fold_comparison([0.50] * 100, [0.90] * 100, outcomes)
        self.assertTrue(clear["won"])

    def test_mismatched_pairs_are_refused(self):
        with self.assertRaises(RegimeSpecializationError):
            fold_comparison([0.5, 0.5], [0.5], [1, 0])

    def test_relative_improvement_is_relative(self):
        self.assertAlmostEqual(relative_improvement(0.25, 0.20), 0.2)
        self.assertIsNone(relative_improvement(0.0, 0.0))

    def test_brier_is_mean_squared_error(self):
        self.assertAlmostEqual(brier([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(brier([0.5, 0.5], [1.0, 0.0]), 0.25)


class GovernedVocabularyTests(unittest.TestCase):
    """W5: L7 uses the governed regime labels, never its own."""

    def test_the_labels_are_the_governed_ones(self):
        self.assertEqual(
            set(REGIME_SPECIALIZATION_LABELS), set(REGIME_LABELS)
        )

    def test_an_invented_regime_is_refused(self):
        with self.assertRaises(RegimeSpecializationError):
            specialization_report({"euphoria": []})

    def test_an_invented_regime_is_refused_by_evaluate(self):
        with self.assertRaises(RegimeSpecializationError):
            evaluate_regime("euphoria", [])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = specialization_report(
            {
                "bullish": folds(FOLDS, SIZE, 0.55, 0.70, 0.70, 8),
                "stress": folds(FOLDS, 20, 0.55, 0.70, 0.70, 9),
            }
        )

    def test_every_regime_is_covered(self):
        self.assertEqual(
            set(self.report["verdicts"]), set(REGIME_SPECIALIZATION_LABELS)
        )

    def test_specialization_is_decided_per_regime(self):
        self.assertTrue(REGIME_SPECIALIZATION_PER_REGIME)
        self.assertEqual(self.report["verdicts"]["bullish"], REGIME_SPEC_SPECIALIZED)
        self.assertEqual(self.report["verdicts"]["stress"], REGIME_SPEC_INSUFFICIENT)

    def test_every_regime_is_served(self):
        self.assertTrue(REGIME_SPECIALIZATION_FALLBACK_POOLED)
        self.assertEqual(
            set(self.report["serving"]), set(REGIME_SPECIALIZATION_LABELS)
        )

    def test_an_unearned_regime_falls_back_to_pooled(self):
        self.assertEqual(self.report["serving"]["stress"], "pooled")

    def test_an_earned_regime_is_served_by_its_own_model(self):
        self.assertEqual(self.report["serving"]["bullish"], "specialized")

    def test_the_report_is_contract_clean(self):
        self.assertEqual(specialization_problems(self.report), [])

    def test_an_empty_report_is_contract_clean(self):
        empty = specialization_report()
        self.assertEqual(specialization_problems(empty), [])
        self.assertEqual(empty["specialized"], [])

    def test_render_returns_one_line_per_regime(self):
        self.assertEqual(
            len(render_specialization(self.report)),
            len(REGIME_SPECIALIZATION_LABELS),
        )

    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(REGIME_SPECIALIZATION_VERDICTS[0], REGIME_SPEC_NOT_EVALUATED)
        self.assertEqual(REGIME_SPECIALIZATION_VERDICTS[-1], REGIME_SPEC_SPECIALIZED)

    def test_a_forged_serving_map_is_a_problem(self):
        forged = dict(self.report)
        forged["serving"] = dict(self.report["serving"])
        forged["serving"]["stress"] = "specialized"
        self.assertTrue(specialization_problems(forged))


if __name__ == "__main__":
    unittest.main()
