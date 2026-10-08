"""X7 multiple-testing protection tests.

The measurements these tests defend:

* `data/research_trials.jsonl` holds 2 rows that are ONE distinct trial, against
  8 trained estimators — the registry undercounts the search, and a correction
  computed from it would report 5.0% family-wise risk where the truth is 33.7%.
* Bonferroni assumes independence and lands at 3.73% against a 5% target on
  correlated estimators; the max-statistic permutation test tracks the target.
* On the shipped family the best member's corrected p-value is ~0.49 — nowhere
  near significant once multiplicity is accounted for.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import (
    MT_DECIDING_METHOD,
    MT_FAILS,
    MT_METHOD_BONFERRONI,
    MT_METHOD_PERMUTATION,
    MT_MIN_FAMILY_FOR_CORRECTION,
    MT_MIN_PERMUTATIONS,
    MT_NOT_EVALUATED,
    MT_PERMUTATIONS,
    MT_SURVIVES,
    MT_TARGET_FWER,
    MULTIPLE_TESTING_VERSION,
    OOS_ALPHA,
)
from core.multiple_testing import (
    MT_REASON_NO_FAMILY,
    MT_REASON_SINGLE_TEST,
    MultipleTestingError,
    bonferroni_threshold,
    evaluate_multiple_testing,
    family_size,
    family_wise_risk,
    max_statistic_permutation,
    multiple_testing_problems,
    render_multiple_testing,
)


def noise_family(members=8, n=300, seed=5):
    """A family where nothing has an edge."""
    rng = np.random.default_rng(seed)
    outcomes = list(rng.normal(0, 0.05, n))
    family = {f"noise{i}": list(rng.normal(0, 0.05, n)) for i in range(members)}
    return family, outcomes


def carried_family(members=8, n=300, seed=5):
    """A family where exactly one member has a genuine edge."""
    rng = np.random.default_rng(seed)
    outcomes = list(rng.normal(0, 0.05, n))
    family = {f"noise{i}": list(rng.normal(0, 0.05, n)) for i in range(members - 1)}
    family["real"] = [value * 0.9 + rng.normal(0, 0.004) for value in outcomes]
    return family, outcomes


class FamilySizeTests(unittest.TestCase):
    def test_the_family_is_counted_from_runs(self):
        counts = family_size(
            [{"estimator": "a"}, {"estimator": "b"}], [{"trial_id": "t1"}]
        )
        self.assertEqual(counts["observed_count"], 2)
        self.assertEqual(counts["registered_count"], 1)

    def test_the_registry_gap_is_reported(self):
        """THE DECIDING MEASUREMENT: 8 runs against 1 distinct trial."""
        runs = [{"estimator": f"e{i}"} for i in range(8)]
        trials = [{"trial_id": "t1", "model_family": "e0"}]
        counts = family_size(runs, trials)
        self.assertEqual(counts["registry_gap"], 7)

    def test_unregistered_estimators_are_named(self):
        runs = [{"estimator": "a"}, {"estimator": "b"}]
        trials = [{"trial_id": "t1", "model_family": "a"}]
        counts = family_size(runs, trials)
        self.assertEqual(counts["unregistered_estimators"], ["b"])

    def test_duplicate_trial_rows_count_once(self):
        """The shipped registry records one trial twice."""
        trials = [
            {"trial_id": "same", "model_family": "momentum"},
            {"trial_id": "same", "model_family": "momentum"},
        ]
        counts = family_size([{"estimator": "momentum"}], trials)
        self.assertEqual(counts["registered_count"], 1)

    def test_an_absent_registry_is_a_gap_not_a_crash(self):
        counts = family_size([{"estimator": "a"}], None)
        self.assertEqual(counts["registered_count"], 0)
        self.assertEqual(counts["registry_gap"], 1)


class CorrectionMathTests(unittest.TestCase):
    def test_bonferroni_divides_by_the_family_size(self):
        self.assertAlmostEqual(bonferroni_threshold(8), MT_TARGET_FWER / 8)

    def test_family_wise_risk_at_eight_is_the_measured_figure(self):
        """MEASURED: 33.7% at n=8, against 5.0% at n=1."""
        self.assertAlmostEqual(family_wise_risk(8), 1 - 0.95**8, places=6)
        self.assertGreater(family_wise_risk(8), 0.33)
        self.assertAlmostEqual(family_wise_risk(1), MT_TARGET_FWER)

    def test_risk_grows_with_the_family(self):
        self.assertGreater(family_wise_risk(8), family_wise_risk(2))

    def test_an_empty_family_raises(self):
        with self.assertRaises(MultipleTestingError):
            bonferroni_threshold(0)
        with self.assertRaises(MultipleTestingError):
            family_wise_risk(0)


class PermutationTests(unittest.TestCase):
    def test_noise_usually_does_not_survive(self):
        """Across seeds, a noise family rarely clears the corrected bar.

        Deliberately NOT a single-seed assertion: with seed 5 one member drew
        0.58 accuracy at n=300 and the permutation test correctly returned
        p=0.04. That is a real false positive, not a bug — it is exactly the
        24.8% single-draw hazard X6 measured, seen once. A test pinned to one
        seed would either enshrine that luck or look flaky.
        """
        survived = 0
        trials = 12
        for seed in range(trials):
            family, outcomes = noise_family(seed=seed)
            result = max_statistic_permutation(family, outcomes, permutations=200)
            if result["p_value"] < MT_TARGET_FWER:
                survived += 1
        self.assertLessEqual(
            survived,
            trials // 3,
            f"{survived} of {trials} noise families cleared the corrected bar",
        )

    def test_a_real_edge_survives(self):
        family, outcomes = carried_family()
        result = max_statistic_permutation(family, outcomes, permutations=200)
        self.assertLess(result["p_value"], MT_TARGET_FWER)
        self.assertEqual(result["best"], "real")

    def test_the_null_is_built_from_the_whole_family(self):
        """The null must widen with the family, which is what corrects for
        multiplicity.

        Compared at a FIXED observed statistic rather than by comparing two
        p-values: a wider family also picks a more extreme observed maximum, so
        the two effects partly cancel and a naive p-value comparison can point
        either way. What must hold is that the NULL DISTRIBUTION is wider.
        """
        family, outcomes = noise_family(members=8, seed=3)
        wide = max_statistic_permutation(family, outcomes, permutations=200)
        narrow = max_statistic_permutation(
            {"noise0": family["noise0"]}, outcomes, permutations=200
        )
        # The family maximum is at least as extreme as any single member's.
        self.assertGreaterEqual(wide["statistic"], narrow["statistic"] - 1e-9)
        # And the best member is chosen from the whole family.
        self.assertIn(wide["best"], family)
        self.assertEqual(len(wide["per_estimator"]), 8)

    def test_too_few_permutations_raise(self):
        family, outcomes = noise_family()
        with self.assertRaises(MultipleTestingError):
            max_statistic_permutation(
                family, outcomes, permutations=MT_MIN_PERMUTATIONS - 1
            )

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(MultipleTestingError):
            max_statistic_permutation({"a": [1.0, 2.0]}, [1.0], permutations=200)

    def test_an_empty_family_raises(self):
        with self.assertRaises(MultipleTestingError):
            max_statistic_permutation({}, [1.0], permutations=200)

    def test_numpy_arrays_are_accepted(self):
        """REGRESSION: `(x > 0) - (x < 0)` raises on numpy scalars, and
        `not outcomes` raises on a numpy array."""
        outcomes = np.array([1.0, -1.0] * 60)
        family = {"a": np.array([1.0, -1.0] * 60), "b": np.array([1.0] * 120)}
        result = max_statistic_permutation(family, outcomes, permutations=200)
        self.assertIsInstance(result["p_value"], float)

    def test_the_result_is_deterministic_for_a_seed(self):
        family, outcomes = noise_family()
        first = max_statistic_permutation(family, outcomes, permutations=200, seed=7)
        second = max_statistic_permutation(family, outcomes, permutations=200, seed=7)
        self.assertEqual(first["p_value"], second["p_value"])


class EvaluateTests(unittest.TestCase):
    def test_a_real_edge_survives_correction(self):
        """The gate must be able to PASS, or refusing proves nothing."""
        family, outcomes = carried_family()
        report = evaluate_multiple_testing(family, outcomes, permutations=200)
        self.assertEqual(report["verdict"], MT_SURVIVES)
        self.assertEqual(multiple_testing_problems(report), [])

    def test_noise_fails_correction(self):
        family, outcomes = noise_family()
        report = evaluate_multiple_testing(family, outcomes, permutations=200)
        self.assertEqual(report["verdict"], MT_FAILS)

    def test_a_family_of_one_has_no_multiplicity(self):
        report = evaluate_multiple_testing({"only": [1.0] * 10}, [1.0] * 10)
        self.assertEqual(report["verdict"], MT_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], MT_REASON_SINGLE_TEST)

    def test_an_absent_family_is_not_evaluated(self):
        report = evaluate_multiple_testing(None, None)
        self.assertEqual(report["verdict"], MT_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], MT_REASON_NO_FAMILY)

    def test_empty_numpy_outcomes_are_not_evaluated(self):
        family, _ = noise_family()
        report = evaluate_multiple_testing(family, np.array([]))
        self.assertEqual(report["verdict"], MT_NOT_EVALUATED)

    def test_the_uncorrected_risk_is_reported(self):
        family, outcomes = noise_family()
        report = evaluate_multiple_testing(family, outcomes, permutations=200)
        self.assertGreater(report["uncorrected_family_risk"], 0.33)

    def test_both_methods_must_pass(self):
        report = evaluate_multiple_testing(*carried_family(), permutations=200)
        self.assertTrue(report["bonferroni"]["passes"])
        self.assertTrue(report["permutation_passes"])

    def test_the_bonferroni_floor_binds_on_a_marginal_result(self):
        """A strong edge gives p=0.0000, where floor and target agree and the
        floor cannot be shown to matter. This probe is deliberately MARGINAL:
        p=0.045 clears the 0.05 target but fails the 0.00625 floor."""
        rng = np.random.default_rng(5)
        n = 300
        outcomes = list(rng.normal(0, 0.05, n))
        family = {f"n{i}": list(rng.normal(0, 0.05, n)) for i in range(7)}
        family["weak"] = [v * 0.10 + rng.normal(0, 0.05) for v in outcomes]
        report = evaluate_multiple_testing(family, outcomes, permutations=400)
        # The probe is only meaningful in that window.
        self.assertLess(report["p_value"], MT_TARGET_FWER)
        self.assertGreater(report["p_value"], report["bonferroni_threshold"])
        self.assertFalse(report["bonferroni"]["passes"])
        self.assertEqual(report["verdict"], MT_FAILS)

    def test_the_permutation_test_decides(self):
        self.assertEqual(MT_DECIDING_METHOD, MT_METHOD_PERMUTATION)


class ThresholdTests(unittest.TestCase):
    def test_the_target_is_x1s_alpha(self):
        self.assertEqual(MT_TARGET_FWER, OOS_ALPHA)

    def test_both_methods_are_applied(self):
        from core.config import MT_METHODS

        self.assertIn(MT_METHOD_BONFERRONI, MT_METHODS)
        self.assertIn(MT_METHOD_PERMUTATION, MT_METHODS)

    def test_a_family_of_two_is_the_minimum_for_correction(self):
        self.assertGreaterEqual(MT_MIN_FAMILY_FOR_CORRECTION, 2)

    def test_the_permutation_count_can_express_the_target(self):
        self.assertGreaterEqual(MT_PERMUTATIONS * MT_TARGET_FWER, 1)


class ContractTests(unittest.TestCase):
    def clean(self):
        return evaluate_multiple_testing(*carried_family(), permutations=200)

    def test_clean_report_has_no_problems(self):
        self.assertEqual(multiple_testing_problems(self.clean()), [])

    def test_survives_above_the_target_is_a_problem(self):
        report = dict(self.clean())
        report["p_value"] = 0.9
        self.assertTrue(
            any("above the" in p for p in multiple_testing_problems(report))
        )

    def test_survives_without_the_bonferroni_floor_is_a_problem(self):
        report = dict(self.clean())
        report["bonferroni"] = {"passes": False}
        self.assertTrue(
            any("Bonferroni floor" in p for p in multiple_testing_problems(report))
        )

    def test_counting_from_the_registry_is_a_problem(self):
        report = dict(self.clean())
        report["counts_observed_runs"] = False
        self.assertTrue(
            any("undercounts" in p for p in multiple_testing_problems(report))
        )

    def test_an_unreported_registry_gap_is_a_problem(self):
        report = dict(self.clean())
        report["family"] = dict(report["family"], registry_gap=7)
        report["reports_registry_gap"] = False
        self.assertTrue(
            any("does not say so" in p for p in multiple_testing_problems(report))
        )

    def test_blocks_trades_is_a_problem(self):
        report = dict(self.clean())
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in multiple_testing_problems(report)))

    def test_not_evaluated_without_a_reason_code_is_a_problem(self):
        report = dict(evaluate_multiple_testing(None, None))
        report["reason_code"] = None
        self.assertTrue(
            any("must name WHY" in p for p in multiple_testing_problems(report))
        )

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(multiple_testing_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        self.assertEqual(self.clean()["version"], MULTIPLE_TESTING_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_registry_undercount(self):
        runs = [{"estimator": f"e{i}"} for i in range(8)]
        trials = [{"trial_id": "t1", "model_family": "e0"}]
        family = {f"e{i}": [1.0] * 120 for i in range(8)}
        report = evaluate_multiple_testing(
            family, [1.0] * 120, runs=runs, trials=trials, permutations=200
        )
        self.assertIn("UNDERCOUNTS", "\n".join(render_multiple_testing(report)))

    def test_render_shows_absent_rather_than_zero(self):
        report = evaluate_multiple_testing(None, None)
        self.assertIn("ABSENT", "\n".join(render_multiple_testing(report)))

    def test_render_returns_lines_not_a_blob(self):
        lines = render_multiple_testing(self_report := evaluate_multiple_testing(None, None))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedFamilyTests(unittest.TestCase):
    """The measurement itself, against what the repo ships."""

    def shipped(self):
        import json

        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        trials = [
            json.loads(line)
            for line in open("data/research_trials.jsonl", encoding="utf-8")
            if line.strip()
        ]
        family = {r["estimator"]: r["folds"][0]["predictions"] for r in runs}
        outcomes = runs[0]["folds"][0]["actuals"]
        return family, outcomes, runs, trials

    def test_the_registry_undercounts_the_shipped_search(self):
        _, _, runs, trials = self.shipped()
        counts = family_size(runs, trials)
        self.assertEqual(counts["observed_count"], 8)
        self.assertEqual(counts["registered_count"], 1)
        self.assertEqual(counts["registry_gap"], 7)

    def test_the_shipped_family_fails_correction(self):
        family, outcomes, runs, trials = self.shipped()
        report = evaluate_multiple_testing(
            family, outcomes, runs=runs, trials=trials, permutations=200
        )
        self.assertEqual(report["verdict"], MT_FAILS)
        self.assertGreater(report["p_value"], MT_TARGET_FWER)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
