"""X6 temporal robustness tests.

The measurements these tests defend:

* Every trained run is 20d, so four of the five required horizons have no model
  — a DATA gap, not a structural one.
* A model judged at ONE horizon clears the band 24.80% of the time on pure
  noise; at three of five that falls to 0.13% while detection stays at 99.93%.
* A horizon with no model is MISSING, never "no edge there".
"""

from __future__ import annotations

import unittest

from core.config import (
    LABEL_HORIZON_SESSIONS,
    OOS_ALPHA,
    OOS_MIN_OBSERVATIONS,
    TEMPORAL_ALPHA,
    TEMPORAL_FRAGILE,
    TEMPORAL_HORIZONS,
    TEMPORAL_MIN_AGREEING,
    TEMPORAL_MIN_OBSERVATIONS,
    TEMPORAL_NOT_EVALUATED,
    TEMPORAL_REQUIRES_SPREAD,
    TEMPORAL_ROBUST,
    TEMPORAL_ROBUSTNESS_VERSION,
)
from core.temporal_robustness import (
    TR_HORIZON_EDGE,
    TR_HORIZON_MISSING,
    TR_HORIZON_NO_EDGE,
    TR_HORIZON_THIN,
    TR_REASON_MISSING_HORIZONS,
    TR_REASON_TOO_FEW_AGREE,
    TemporalRobustnessError,
    evaluate_temporal_robustness,
    horizon_edge,
    render_temporal,
    temporal_problems,
)


def sweep(accuracies, observations=120):
    """A result for every required horizon."""
    return {
        horizon: {"directional_accuracy": accuracy, "observations": observations}
        for horizon, accuracy in zip(TEMPORAL_HORIZONS, accuracies)
    }


class HorizonEdgeTests(unittest.TestCase):
    def test_a_clear_edge_is_an_edge(self):
        outcome, excess = horizon_edge(0.70, 120)
        self.assertEqual(outcome, TR_HORIZON_EDGE)
        self.assertAlmostEqual(excess, 0.20)

    def test_inside_the_band_is_no_edge(self):
        """The band at n=120 is ±0.0895, so 0.55 does not clear it."""
        outcome, _ = horizon_edge(0.55, 120)
        self.assertEqual(outcome, TR_HORIZON_NO_EDGE)

    def test_a_thin_sample_cannot_decide(self):
        outcome, excess = horizon_edge(0.70, OOS_MIN_OBSERVATIONS - 1)
        self.assertEqual(outcome, TR_HORIZON_THIN)
        self.assertIsNone(excess)

    def test_an_absent_horizon_is_missing_not_no_edge(self):
        """THE CENTRAL DISTINCTION."""
        outcome, excess = horizon_edge(None, None)
        self.assertEqual(outcome, TR_HORIZON_MISSING)
        self.assertIsNone(excess)
        self.assertNotEqual(outcome, TR_HORIZON_NO_EDGE)

    def test_a_missing_accuracy_is_missing_even_with_observations(self):
        outcome, _ = horizon_edge(None, 120)
        self.assertEqual(outcome, TR_HORIZON_MISSING)


class EvaluateTests(unittest.TestCase):
    def test_three_agreeing_horizons_are_robust(self):
        """The gate must be able to PASS, or refusing proves nothing."""
        report = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.52, 0.51]))
        self.assertEqual(report["verdict"], TEMPORAL_ROBUST)
        self.assertEqual(len(report["agreeing_horizons"]), 3)
        self.assertEqual(temporal_problems(report), [])

    def test_two_agreeing_horizons_are_fragile(self):
        report = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.52, 0.51, 0.50]))
        self.assertEqual(report["verdict"], TEMPORAL_FRAGILE)
        self.assertEqual(report["reason_code"], TR_REASON_TOO_FEW_AGREE)

    def test_missing_horizons_block_the_verdict(self):
        """Four of five missing is the shipped state."""
        report = evaluate_temporal_robustness(
            {"20d": {"directional_accuracy": 0.70, "observations": 120}}
        )
        self.assertEqual(report["verdict"], TEMPORAL_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], TR_REASON_MISSING_HORIZONS)
        self.assertEqual(len(report["missing_horizons"]), 4)

    def test_a_strong_single_horizon_still_cannot_be_robust(self):
        """The 24.8% false-positive rate, enforced: one horizon is not a sweep."""
        report = evaluate_temporal_robustness(
            {"20d": {"directional_accuracy": 0.95, "observations": 120}}
        )
        self.assertNotEqual(report["verdict"], TEMPORAL_ROBUST)

    def test_thin_horizons_block_the_verdict(self):
        report = evaluate_temporal_robustness(sweep([0.70] * 5, observations=50))
        self.assertEqual(report["verdict"], TEMPORAL_NOT_EVALUATED)

    def test_every_required_horizon_is_reported(self):
        report = evaluate_temporal_robustness({})
        self.assertEqual(sorted(report["horizons"]), sorted(TEMPORAL_HORIZONS))

    def test_an_unevaluated_horizon_carries_no_measurement(self):
        report = evaluate_temporal_robustness({})
        for entry in report["horizons"].values():
            self.assertIsNone(entry["excess_over_coin_flip"])

    def test_an_unknown_horizon_raises(self):
        with self.assertRaises(TemporalRobustnessError):
            evaluate_temporal_robustness(
                {"252d": {"directional_accuracy": 0.7, "observations": 120}}
            )

    def test_absent_input_is_not_evaluated(self):
        report = evaluate_temporal_robustness(None)
        self.assertEqual(report["verdict"], TEMPORAL_NOT_EVALUATED)

    def test_the_short_long_shape_is_reported(self):
        report = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.52, 0.51]))
        self.assertEqual(sorted(report["agreeing_short"]), ["1d", "5d"])
        self.assertEqual(report["agreeing_long"], [])

    def test_adjacent_agreement_still_counts(self):
        """The spread rule was TESTED AND DROPPED: adjacent and spread
        agreement are indistinguishable under the null (0.50% vs 0.45%)."""
        adjacent = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.51, 0.50]))
        self.assertEqual(adjacent["verdict"], TEMPORAL_ROBUST)
        self.assertFalse(TEMPORAL_REQUIRES_SPREAD)


class ThresholdTests(unittest.TestCase):
    def test_three_of_five_is_required(self):
        """MEASURED: 2 of 5 false-positives at 2.53%, 3 of 5 at 0.13%."""
        self.assertEqual(TEMPORAL_MIN_AGREEING, 3)

    def test_all_five_roadmap_horizons_are_evaluated(self):
        for horizon in ("1d", "5d", "20d", "60d", "120d"):
            self.assertIn(horizon, TEMPORAL_HORIZONS)

    def test_every_horizon_is_labelable(self):
        for horizon in TEMPORAL_HORIZONS:
            self.assertIn(horizon, LABEL_HORIZON_SESSIONS)

    def test_the_floors_agree_with_x1(self):
        self.assertEqual(TEMPORAL_MIN_OBSERVATIONS, OOS_MIN_OBSERVATIONS)
        self.assertEqual(TEMPORAL_ALPHA, OOS_ALPHA)


class ContractTests(unittest.TestCase):
    def clean(self):
        return evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.52, 0.51]))

    def test_clean_report_has_no_problems(self):
        self.assertEqual(temporal_problems(self.clean()), [])

    def test_robust_with_missing_horizons_is_a_problem(self):
        report = dict(self.clean())
        report["missing_horizons"] = ["1d"]
        self.assertTrue(
            any("never evaluated" in p for p in temporal_problems(report))
        )

    def test_robust_below_the_bar_is_a_problem(self):
        report = dict(self.clean())
        report["agreeing_horizons"] = ["20d"]
        self.assertTrue(any("below the" in p for p in temporal_problems(report)))

    def test_fragile_with_missing_horizons_is_a_problem(self):
        report = dict(evaluate_temporal_robustness(sweep([0.70, 0.68, 0.52, 0.51, 0.50])))
        report["missing_horizons"] = ["1d"]
        self.assertTrue(
            any("different answers" in p for p in temporal_problems(report))
        )

    def test_a_missing_horizon_carrying_a_measurement_is_a_problem(self):
        report = dict(evaluate_temporal_robustness({}))
        horizons = dict(report["horizons"])
        entry = dict(horizons["1d"])
        entry["excess_over_coin_flip"] = 0.2
        horizons["1d"] = entry
        report["horizons"] = horizons
        self.assertTrue(
            any("nothing to measure" in p for p in temporal_problems(report))
        )

    def test_a_dropped_horizon_is_a_problem(self):
        report = dict(self.clean())
        horizons = dict(report["horizons"])
        horizons.pop("120d")
        report["horizons"] = horizons
        self.assertTrue(any("roadmap names" in p for p in temporal_problems(report)))

    def test_absent_means_no_edge_is_a_problem(self):
        report = dict(self.clean())
        report["absent_means_no_edge"] = True
        self.assertTrue(
            any("untested horizon" in p for p in temporal_problems(report))
        )

    def test_blocks_trades_is_a_problem(self):
        report = dict(self.clean())
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in temporal_problems(report)))

    def test_not_evaluated_without_a_reason_code_is_a_problem(self):
        report = dict(evaluate_temporal_robustness({}))
        report["reason_code"] = None
        self.assertTrue(any("must name WHY" in p for p in temporal_problems(report)))

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(temporal_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        self.assertEqual(self.clean()["version"], TEMPORAL_ROBUSTNESS_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_missing_horizons(self):
        text = "\n".join(render_temporal(evaluate_temporal_robustness({})))
        self.assertIn("NO MODEL", text)

    def test_render_shows_absent_rather_than_zero(self):
        text = "\n".join(render_temporal(evaluate_temporal_robustness({})))
        self.assertIn("ABSENT", text)

    def test_render_lists_every_horizon(self):
        text = "\n".join(
            render_temporal(evaluate_temporal_robustness(sweep([0.70] * 5)))
        )
        for horizon in TEMPORAL_HORIZONS:
            self.assertIn(horizon, text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_temporal(evaluate_temporal_robustness({}))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedRunTests(unittest.TestCase):
    """The blocking measurement, against the runs the repo ships."""

    def test_every_trained_run_is_20d(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        self.assertEqual({r["target_horizon"] for r in runs}, {"20d"})

    def test_the_shipped_state_is_not_evaluated(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        report = evaluate_temporal_robustness(
            {"20d": runs[0]["metrics"]}, estimator=runs[0]["estimator"]
        )
        self.assertEqual(report["verdict"], TEMPORAL_NOT_EVALUATED)
        self.assertEqual(len(report["missing_horizons"]), 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
