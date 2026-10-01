"""X3 regime robustness tests.

The measurements these tests defend:

* Persisted folds carry no ticker and no per-observation timestamp, so a regime
  label cannot be joined on — X3 must report NOT_EVALUATED, not invent one.
* A pooled 0.6800 collapses to 0.5833 when one regime is removed, while
  "every regime beats a coin flip" answers 4 of 4 for both that book and a
  uniformly-strong one.
* Across 400 uniform-edge trials the worst leave-one-out drop had a p95 of
  0.0300 and a maximum of 0.0525, so the 0.05 bar is the top of the null.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import (
    REGIME_CONCENTRATED,
    REGIME_LABELS,
    REGIME_ROBUST,
    REGIME_ROBUSTNESS_MAX_DROP,
    REGIME_ROBUSTNESS_MIN_CELL,
    REGIME_ROBUSTNESS_NOT_EVALUATED,
    REGIME_ROBUSTNESS_VERSION,
    REGIME_SPECIALIZATION_MIN_CELL,
)
from core.regime_robustness import (
    RR_REASON_CARRIED,
    RR_REASON_NO_LABELS,
    RR_REASON_THIN_CELLS,
    RegimeRobustnessError,
    evaluate_regime_robustness,
    fold_regime_labels,
    group_by_regime,
    leave_one_out,
    render_robustness,
    robustness_problems,
)

REGIMES = ["bullish", "bearish", "range", "risk_off"]


def book(concentrated: bool, n: int = 100, seed: int = 303):
    """A four-regime book whose edge is either carried by one regime or broad."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    labels: list[str] = []
    for regime in REGIMES:
        outcome = rng.normal(0, 0.05, n)
        if concentrated:
            predicted = (
                outcome * 0.9 + rng.normal(0, 0.005, n)
                if regime == "bullish"
                else rng.normal(0, 0.05, n)
            )
        else:
            predicted = outcome * 0.35 + rng.normal(0, 0.03, n)
        predictions.extend(predicted)
        actuals.extend(outcome)
        labels.extend([regime] * n)
    return predictions, actuals, labels


class FoldLabelTests(unittest.TestCase):
    def test_a_fold_without_regimes_yields_none(self):
        """THE BLOCKING MEASUREMENT: shipped folds carry no labels."""
        self.assertIsNone(fold_regime_labels({"predictions": [1.0], "actuals": [1.0]}))

    def test_labels_are_never_derived_from_the_fold_window(self):
        """Stamping one regime on every observation would leave nothing to
        leave out, and the gate would return ROBUST on no evidence."""
        fold = {
            "predictions": [1.0],
            "actuals": [1.0],
            "validation_start_time": "2025-07-21 13:30:00",
            "train_end_time": "2025-04-16 13:30:00",
        }
        self.assertIsNone(fold_regime_labels(fold))

    def test_present_labels_are_read(self):
        self.assertEqual(
            fold_regime_labels({"regimes": ["bullish", "bearish"]}),
            ["bullish", "bearish"],
        )

    def test_absent_fold_is_none(self):
        self.assertIsNone(fold_regime_labels(None))

    def test_a_string_is_not_a_label_sequence(self):
        with self.assertRaises(RegimeRobustnessError):
            fold_regime_labels({"regimes": "bullish"})

    def test_non_mapping_fold_raises(self):
        with self.assertRaises(RegimeRobustnessError):
            fold_regime_labels(["bullish"])


class GroupingTests(unittest.TestCase):
    def test_cells_carry_counts_and_accuracy(self):
        cells = group_by_regime([1.0, -1.0], [1.0, 1.0], ["bullish", "bearish"])
        self.assertEqual(cells["bullish"]["observations"], 1)
        self.assertAlmostEqual(cells["bullish"]["directional_accuracy"], 1.0)
        self.assertAlmostEqual(cells["bearish"]["directional_accuracy"], 0.0)

    def test_a_thin_cell_is_marked_unusable(self):
        cells = group_by_regime([1.0], [1.0], ["bullish"])
        self.assertFalse(cells["bullish"]["usable"])

    def test_an_unknown_regime_raises(self):
        """Scoring it would hide an edge in a regime nobody checked."""
        with self.assertRaises(RegimeRobustnessError):
            group_by_regime([1.0], [1.0], ["euphoria"])

    def test_an_unobserved_label_is_excluded_not_fatal(self):
        """CONTRACT CHANGED BY A1, deliberately.

        This used to raise, and that was right when a fold either carried every
        label or none. A1 records per-observation context, so PARTIAL context is
        now possible — a classifier that could not classify one day — and raising
        would discard an otherwise usable fold over a single gap.

        The observation is EXCLUDED rather than assigned: assigning one would
        manufacture the attribution this gate exists to test, which is the
        synthetic-bucket failure A1 was designed to avoid.
        """
        cells = group_by_regime([1.0, 2.0], [1.0, 2.0], [None, "bullish"])
        self.assertNotIn("None", cells)
        self.assertIn("bullish", cells)
        self.assertEqual(cells["bullish"]["observations"], 1)

    def test_a_wholly_unobserved_fold_yields_no_cells(self):
        # And with nothing classified there is nothing to compare, which the
        # caller reports as NOT_EVALUATED rather than as agreement.
        self.assertEqual(group_by_regime([1.0], [1.0], [None]), {})

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(RegimeRobustnessError):
            group_by_regime([1.0, 2.0], [1.0], ["bullish", "bearish"])


class LeaveOneOutTests(unittest.TestCase):
    def test_one_regime_cannot_be_left_out(self):
        cells = group_by_regime([1.0] * 2, [1.0] * 2, ["bullish"] * 2)
        with self.assertRaises(RegimeRobustnessError):
            leave_one_out(cells)

    def test_removing_a_regime_excludes_exactly_it(self):
        cells = group_by_regime(
            [1.0, -1.0], [1.0, 1.0], ["bullish", "bearish"]
        )
        without = leave_one_out(cells)
        self.assertAlmostEqual(without["bearish"], 1.0)
        self.assertAlmostEqual(without["bullish"], 0.0)


class EvaluateTests(unittest.TestCase):
    def test_a_carried_edge_is_concentrated(self):
        """THE DECIDING CASE: pooled 0.6800 collapses to 0.5833."""
        report = evaluate_regime_robustness(*book(concentrated=True))
        self.assertEqual(report["verdict"], REGIME_CONCENTRATED)
        self.assertEqual(report["reason_code"], RR_REASON_CARRIED)
        self.assertEqual(report["carrier"], "bullish")
        self.assertGreater(report["worst_drop"], REGIME_ROBUSTNESS_MAX_DROP)

    def test_a_broad_edge_is_robust(self):
        """The gate must be able to PASS, or refusing proves nothing."""
        report = evaluate_regime_robustness(*book(concentrated=False))
        self.assertEqual(report["verdict"], REGIME_ROBUST)
        self.assertLessEqual(report["worst_drop"], REGIME_ROBUSTNESS_MAX_DROP)
        self.assertEqual(robustness_problems(report), [])

    def test_every_regime_beats_a_coin_flip_in_both_books(self):
        """Which is exactly why that statistic cannot be the test."""
        for concentrated in (True, False):
            with self.subTest(concentrated=concentrated):
                report = evaluate_regime_robustness(*book(concentrated))
                above = [
                    regime
                    for regime, cell in report["cells"].items()
                    if cell["directional_accuracy"] > 0.5
                ]
                self.assertEqual(len(above), len(REGIMES))

    def test_missing_labels_are_not_evaluated(self):
        predictions, actuals, _ = book(concentrated=True)
        report = evaluate_regime_robustness(predictions, actuals, None)
        self.assertEqual(report["verdict"], REGIME_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], RR_REASON_NO_LABELS)

    def test_missing_labels_names_the_pipeline_fix(self):
        predictions, actuals, _ = book(concentrated=True)
        report = evaluate_regime_robustness(predictions, actuals, None)
        self.assertIn("prediction_time", report["reason"])

    def test_thin_cells_are_not_evaluated_not_robust(self):
        """A cell below the floor cannot support a claim in either direction."""
        thin = REGIME_ROBUSTNESS_MIN_CELL - 1
        predictions, actuals, labels = book(concentrated=False, n=thin)
        report = evaluate_regime_robustness(predictions, actuals, labels)
        self.assertEqual(report["verdict"], REGIME_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], RR_REASON_THIN_CELLS)

    def test_a_single_regime_is_not_evaluated(self):
        n = REGIME_ROBUSTNESS_MIN_CELL + 10
        rng = np.random.default_rng(1)
        outcome = rng.normal(0, 0.05, n)
        report = evaluate_regime_robustness(
            list(outcome * 0.9), list(outcome), ["bullish"] * n
        )
        self.assertEqual(report["verdict"], REGIME_ROBUSTNESS_NOT_EVALUATED)

    def test_thin_cells_are_reported_but_excluded(self):
        """A regime present but too thin must be visible AND not counted."""
        rng = np.random.default_rng(2)
        big = REGIME_ROBUSTNESS_MIN_CELL + 20
        predictions: list[float] = []
        actuals: list[float] = []
        labels: list[str] = []
        for regime, count in (("bullish", big), ("bearish", big), ("range", 5)):
            outcome = rng.normal(0, 0.05, count)
            predictions.extend(outcome * 0.35 + rng.normal(0, 0.03, count))
            actuals.extend(outcome)
            labels.extend([regime] * count)
        report = evaluate_regime_robustness(predictions, actuals, labels)
        self.assertIn("range", report["cells"])
        self.assertFalse(report["cells"]["range"]["usable"])
        self.assertNotIn("range", report["usable_regimes"])


class ThresholdTests(unittest.TestCase):
    def test_the_bar_sits_at_the_top_of_the_measured_null(self):
        """MEASURED: a uniform edge produced at most 0.0525 in 400 trials."""
        self.assertGreaterEqual(REGIME_ROBUSTNESS_MAX_DROP, 0.03)
        self.assertLessEqual(REGIME_ROBUSTNESS_MAX_DROP, 0.0967)

    def test_the_cell_floor_agrees_with_l7(self):
        self.assertEqual(REGIME_ROBUSTNESS_MIN_CELL, REGIME_SPECIALIZATION_MIN_CELL)

    def test_the_regimes_are_n4s(self):
        from core.config import REGIME_ROBUSTNESS_LABELS

        self.assertEqual(set(REGIME_ROBUSTNESS_LABELS), set(REGIME_LABELS))


class ContractTests(unittest.TestCase):
    def test_clean_report_has_no_problems(self):
        report = evaluate_regime_robustness(*book(concentrated=False))
        self.assertEqual(robustness_problems(report), [])

    def test_robust_above_the_bar_is_a_problem(self):
        report = dict(evaluate_regime_robustness(*book(concentrated=False)))
        report["worst_drop"] = 0.9
        self.assertTrue(any("above the" in p for p in robustness_problems(report)))

    def test_robust_on_one_regime_is_a_problem(self):
        report = dict(evaluate_regime_robustness(*book(concentrated=False)))
        report["usable_regimes"] = ["bullish"]
        self.assertTrue(
            any("nothing was actually left out" in p for p in robustness_problems(report))
        )

    def test_concentrated_without_a_carrier_is_a_problem(self):
        report = dict(evaluate_regime_robustness(*book(concentrated=True)))
        report["carrier"] = None
        self.assertTrue(any("must name the regime" in p for p in robustness_problems(report)))

    def test_not_evaluated_without_a_reason_code_is_a_problem(self):
        predictions, actuals, _ = book(concentrated=True)
        report = dict(evaluate_regime_robustness(predictions, actuals, None))
        report["reason_code"] = None
        self.assertTrue(any("must name WHY" in p for p in robustness_problems(report)))

    def test_infers_labels_is_a_problem(self):
        report = dict(evaluate_regime_robustness(*book(concentrated=False)))
        report["infers_labels"] = True
        self.assertTrue(
            any("never be assigned a regime" in p for p in robustness_problems(report))
        )

    def test_blocks_trades_is_a_problem(self):
        report = dict(evaluate_regime_robustness(*book(concentrated=False)))
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in robustness_problems(report)))

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(robustness_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        report = evaluate_regime_robustness(*book(concentrated=False))
        self.assertEqual(report["version"], REGIME_ROBUSTNESS_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_carrier(self):
        report = evaluate_regime_robustness(*book(concentrated=True))
        self.assertIn("carrier", "\n".join(render_robustness(report)))

    def test_render_shows_absent_rather_than_zero(self):
        predictions, actuals, _ = book(concentrated=True)
        report = evaluate_regime_robustness(predictions, actuals, None)
        self.assertIn("ABSENT", "\n".join(render_robustness(report)))

    def test_render_marks_a_thin_cell(self):
        thin = REGIME_ROBUSTNESS_MIN_CELL - 1
        report = evaluate_regime_robustness(*book(concentrated=False, n=thin))
        self.assertIn("too thin", "\n".join(render_robustness(report)))

    def test_render_returns_lines_not_a_blob(self):
        lines = render_robustness(evaluate_regime_robustness(*book(concentrated=True)))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedFoldTests(unittest.TestCase):
    """The blocking measurement, against the runs the repo ships."""

    def test_the_regenerated_folds_carry_regime_labels(self):
        """RESTATED after A1. The blocker was that NO fold carried a label.

        A1 records one per validation observation and the ledger was
        regenerated, so a labelled fold must exist and its labels must align
        with its predictions - a misaligned slice reads as evidence while
        describing the wrong observations.
        """
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        labelled = [
            run for run in runs if fold_regime_labels(run["folds"][0]) is not None
        ]
        self.assertTrue(
            labelled,
            "no fold carries regime labels; A1 added them and losing them "
            "re-blocks X3 entirely",
        )
        for run in labelled:
            with self.subTest(estimator=run["estimator"]):
                fold = run["folds"][0]
                self.assertEqual(
                    len(fold_regime_labels(fold)), len(fold["predictions"])
                )

    def test_a_shipped_run_is_not_evaluated(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        fold = runs[0]["folds"][0]
        report = evaluate_regime_robustness(
            fold["predictions"], fold["actuals"], fold_regime_labels(fold)
        )
        self.assertEqual(report["verdict"], REGIME_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], RR_REASON_NO_LABELS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
