"""R6 stress scenario tests.

The central claim under test: correlation rises under stress, and an estimator
that demeans within the stress subset reports the OPPOSITE. These tests build
a market with a deliberate crash regime from a seeded generator rather than
reading any data file, so they mean the same thing on a clean clone.

The real-data measurements quoted here came from 76 tickers over 1,170 common
sessions in the local ingest, which is gitignored. The synthetic market below
reproduces the same structural effect, which is what the tests pin.
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from core.config import (
    STRESS_BLOCKS_TRADES,
    STRESS_CORRELATION_LEVEL,
    STRESS_MATERIAL_DEGRADATION,
    STRESS_MIN_SESSIONS,
    STRESS_SCENARIO_CORRELATION_SHOCK,
    STRESS_SCENARIO_HISTORICAL,
    STRESS_SCENARIO_VERSION,
    STRESS_SCENARIO_VOLATILITY_SHOCK,
    STRESS_SCENARIOS,
    STRESS_VOLATILITY_MULTIPLIER,
    STRESS_WORST_SHARE,
)
from core.stress_scenarios import (
    StressScenarioError,
    average_correlation,
    covariance_over,
    diversification_ratio,
    render_stress,
    shock_correlation,
    shock_volatility,
    stress_problems,
    stress_report,
    worst_sessions,
)


def crashing_market(seed=13, sessions=800, names=12, crash_share=0.10):
    """Mostly idiosyncratic, but crashes together — the real data's structure.

    The crash magnitude is DISPERSED (sd 0.005) and per-name noise stays high
    on crash days (sd 0.025), because that is what real crashes look like and
    it is what makes the estimator trap visible. MEASURED, an earlier version
    used a tight crash with almost no idiosyncratic noise; subset demeaning
    still reported 0.88 correlation there, so the trap never appeared and a
    gate built on it passed the very sabotage it existed to catch.
    """
    rng = random.Random(seed)
    tickers = [f"T{i:02d}" for i in range(names)]
    matrix = [[0.0] * sessions for _ in range(names)]
    crash_days = set(
        rng.sample(
            range(sessions), max(STRESS_MIN_SESSIONS, int(sessions * crash_share))
        )
    )
    for day in range(sessions):
        if day in crash_days:
            common = rng.gauss(-0.040, 0.005)
            for i in range(names):
                matrix[i][day] = common + rng.gauss(0, 0.025)
        else:
            common = rng.gauss(0.0012, 0.003)
            for i in range(names):
                matrix[i][day] = common + rng.gauss(0, 0.012)
    return tickers, matrix, sorted(crash_days)


EQUAL = None


def equal_weights(tickers):
    return {t: 1.0 / len(tickers) for t in tickers}


class TheEstimatorTrapTest(unittest.TestCase):
    """Demeaning within the stress subset reports the opposite of the truth."""

    def test_subset_demeaning_understates_stress_correlation(self):
        tickers, matrix, crash_days = crashing_market()
        full_means = [statistics.mean(row) for row in matrix]

        correct = average_correlation(
            covariance_over(matrix, crash_days, means=full_means)
        )
        subset_means = [
            statistics.mean(row[d] for d in crash_days) for row in matrix
        ]
        trapped = average_correlation(
            covariance_over(matrix, crash_days, means=subset_means)
        )
        self.assertGreater(
            correct,
            trapped,
            "demeaning within the stress subset did not understate "
            "correlation — if this ever holds, the full-period rule loses "
            "its evidence and must be re-derived",
        )

    def test_the_correct_estimator_shows_correlation_rising(self):
        tickers, matrix, crash_days = crashing_market()
        full_means = [statistics.mean(row) for row in matrix]
        calm_days = [d for d in range(len(matrix[0])) if d not in set(crash_days)]
        calm = average_correlation(
            covariance_over(matrix, calm_days, means=full_means)
        )
        stressed = average_correlation(
            covariance_over(matrix, crash_days, means=full_means)
        )
        self.assertGreater(
            stressed,
            calm,
            "correlation did not rise under stress in a market built to "
            "crash together",
        )

    def test_the_report_uses_the_full_period_mean(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertEqual(report["demean"], "full_period")
        self.assertTrue(report["demean_reason"].strip())

    def test_a_report_demeaned_on_the_subset_is_caught(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        report["demean"] = "subset"
        self.assertTrue(
            stress_problems(report),
            "a subset-demeaned report passed the contract check",
        )

    def test_stress_looking_calmer_than_calm_is_caught(self):
        """The estimator trap, caught at the contract boundary too."""
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        scenario = report["scenarios"][STRESS_SCENARIO_HISTORICAL]
        scenario["stressed_correlation"] = scenario["baseline_correlation"] - 0.1
        self.assertTrue(stress_problems(report))


class TheDecidingMeasurementTest(unittest.TestCase):
    """Correlation rises and diversification falls under stress."""

    def test_stress_raises_portfolio_volatility(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        historical = report["scenarios"][STRESS_SCENARIO_HISTORICAL]
        self.assertGreater(
            historical["volatility_multiple"],
            1.0,
            "stress did not raise portfolio volatility",
        )

    def test_stress_costs_diversification(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        historical = report["scenarios"][STRESS_SCENARIO_HISTORICAL]
        self.assertGreater(
            historical["diversification_lost"],
            0.0,
            "diversification survived a market built to crash together",
        )
        self.assertLess(
            historical["stressed_diversification"],
            historical["baseline_diversification"],
        )

    def test_the_historical_scenario_is_flagged_material(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertIn(STRESS_SCENARIO_HISTORICAL, report["material"])


class WorstSessionsTest(unittest.TestCase):
    def test_it_selects_the_worst_market_wide_days(self):
        tickers, matrix, crash_days = crashing_market()
        selected = set(worst_sessions(matrix, share=0.10))
        overlap = len(selected & set(crash_days)) / len(selected)
        self.assertGreater(
            overlap,
            0.8,
            f"only {overlap:.0%} of the selected sessions were real crash "
            f"days — stress is not being located",
        )

    def test_the_count_follows_the_share(self):
        tickers, matrix, _ = crashing_market()
        sessions = len(matrix[0])
        for share in (0.05, 0.10, 0.20):
            self.assertEqual(
                len(worst_sessions(matrix, share=share)), int(sessions * share)
            )

    def test_at_least_one_session_is_always_selected(self):
        matrix = [[0.01, -0.02, 0.03] for _ in range(3)]
        self.assertGreaterEqual(len(worst_sessions(matrix, share=0.01)), 1)

    def test_an_impossible_share_is_refused(self):
        tickers, matrix, _ = crashing_market()
        for bad in (0.0, 1.0, -0.5):
            with self.assertRaises(StressScenarioError):
                worst_sessions(matrix, share=bad)

    def test_an_empty_matrix_is_refused(self):
        with self.assertRaises(StressScenarioError):
            worst_sessions([])


class CovarianceOverTest(unittest.TestCase):
    def test_too_few_observations_are_refused(self):
        matrix = [[0.01, 0.02] for _ in range(3)]
        with self.assertRaises(StressScenarioError):
            covariance_over(matrix, [0])

    def test_a_mean_is_required_per_series(self):
        matrix = [[0.01, 0.02, 0.03] for _ in range(3)]
        with self.assertRaises(StressScenarioError):
            covariance_over(matrix, [0, 1, 2], means=[0.0])

    def test_it_defaults_to_the_full_period_mean(self):
        tickers, matrix, crash_days = crashing_market()
        explicit = covariance_over(
            matrix, crash_days, means=[statistics.mean(r) for r in matrix]
        )
        default = covariance_over(matrix, crash_days)
        self.assertAlmostEqual(explicit[0][0], default[0][0], places=12)


class ShockTest(unittest.TestCase):
    def test_a_correlation_shock_raises_correlation(self):
        tickers, matrix, _ = crashing_market()
        calm = covariance_over(matrix, list(range(len(matrix[0]))))
        shocked = shock_correlation(calm)
        self.assertGreater(
            average_correlation(shocked), average_correlation(calm)
        )

    def test_a_correlation_shock_keeps_each_volatility(self):
        tickers, matrix, _ = crashing_market()
        calm = covariance_over(matrix, list(range(len(matrix[0]))))
        shocked = shock_correlation(calm)
        for i in range(len(calm)):
            self.assertAlmostEqual(calm[i][i], shocked[i][i], places=12)

    def test_a_degenerate_correlation_is_refused(self):
        tickers, matrix, _ = crashing_market()
        calm = covariance_over(matrix, list(range(len(matrix[0]))))
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(StressScenarioError):
                shock_correlation(calm, level=bad)

    def test_a_volatility_shock_scales_volatility_not_correlation(self):
        tickers, matrix, _ = crashing_market()
        calm = covariance_over(matrix, list(range(len(matrix[0]))))
        shocked = shock_volatility(calm, 2.0)
        self.assertAlmostEqual(
            average_correlation(calm), average_correlation(shocked), places=8
        )
        self.assertAlmostEqual(shocked[0][0], calm[0][0] * 4.0, places=12)

    def test_a_non_positive_multiplier_is_refused(self):
        tickers, matrix, _ = crashing_market()
        calm = covariance_over(matrix, list(range(len(matrix[0]))))
        with self.assertRaises(StressScenarioError):
            shock_volatility(calm, 0.0)


class DiversificationTest(unittest.TestCase):
    def test_a_single_holding_has_no_diversification(self):
        tickers, matrix, _ = crashing_market()
        cov = covariance_over(matrix, list(range(len(matrix[0]))))
        ratio = diversification_ratio({tickers[0]: 1.0}, tickers, cov)
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_many_holdings_diversify(self):
        tickers, matrix, _ = crashing_market()
        cov = covariance_over(matrix, list(range(len(matrix[0]))))
        self.assertGreater(
            diversification_ratio(equal_weights(tickers), tickers, cov), 1.0
        )

    def test_an_empty_portfolio_is_none_not_zero(self):
        tickers, matrix, _ = crashing_market()
        cov = covariance_over(matrix, list(range(len(matrix[0]))))
        self.assertIsNone(
            diversification_ratio({}, tickers, cov),
            "a ratio of 0.0 reads as perfect diversification rather than "
            "unmeasured",
        )


class ReportTest(unittest.TestCase):
    def test_a_clean_report_has_no_problems(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertEqual(stress_problems(report), [])
        self.assertEqual(report["version"], STRESS_SCENARIO_VERSION)

    def test_every_declared_scenario_is_reported(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        for name in STRESS_SCENARIOS:
            self.assertIn(name, report["scenarios"])
            self.assertTrue(report["scenarios"][name]["reason"].strip())

    def test_too_few_stressed_sessions_is_not_evaluated(self):
        """Not enough stress to measure is not an absence of stress."""
        tickers = ["A", "B", "C"]
        matrix = [[0.01, -0.02, 0.005, 0.001] for _ in tickers]
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertEqual(report["status"], "NOT_EVALUATED")
        self.assertEqual(report["scenarios"], {})
        self.assertTrue(report["reason"].strip())
        self.assertEqual(stress_problems(report), [])

    def test_a_mismatched_matrix_is_refused(self):
        tickers, matrix, _ = crashing_market()
        with self.assertRaises(StressScenarioError):
            stress_report(equal_weights(tickers), tickers[:-1], matrix)

    def test_a_non_mapping_weight_set_is_refused(self):
        tickers, matrix, _ = crashing_market()
        with self.assertRaises(StressScenarioError):
            stress_report(["A"], tickers, matrix)

    def test_an_empty_matrix_is_refused(self):
        with self.assertRaises(StressScenarioError):
            stress_report({}, [], [])

    def test_r6_does_not_block_trades(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertFalse(report["blocks_trades"])
        self.assertFalse(STRESS_BLOCKS_TRADES)


class ProblemDetectionTest(unittest.TestCase):
    """stress_problems must be able to fail."""

    def _report(self):
        tickers, matrix, _ = crashing_market()
        return stress_report(equal_weights(tickers), tickers, matrix)

    def test_a_missing_historical_scenario_is_caught(self):
        report = self._report()
        del report["scenarios"][STRESS_SCENARIO_HISTORICAL]
        self.assertTrue(
            stress_problems(report),
            "a report with only synthetic shocks passed — an unchecked "
            "hypothetical is an assumption",
        )

    def test_stress_lowering_volatility_is_caught(self):
        report = self._report()
        report["scenarios"][STRESS_SCENARIO_HISTORICAL]["volatility_multiple"] = 0.8
        self.assertTrue(stress_problems(report))

    def test_a_scenario_with_no_reason_is_caught(self):
        report = self._report()
        report["scenarios"][STRESS_SCENARIO_HISTORICAL]["reason"] = "  "
        self.assertTrue(stress_problems(report))

    def test_a_materiality_disagreeing_with_its_threshold_is_caught(self):
        report = self._report()
        scenario = report["scenarios"][STRESS_SCENARIO_HISTORICAL]
        scenario["diversification_lost"] = 0.9
        scenario["material"] = False
        self.assertTrue(stress_problems(report))

    def test_a_report_that_blocks_trades_is_caught(self):
        report = self._report()
        report["blocks_trades"] = True
        self.assertTrue(stress_problems(report))

    def test_a_missing_demean_reason_is_caught(self):
        report = self._report()
        report["demean_reason"] = ""
        self.assertTrue(stress_problems(report))

    def test_not_evaluated_carrying_scenarios_is_caught(self):
        report = self._report()
        report["status"] = "NOT_EVALUATED"
        report["reason"] = "too few sessions"
        self.assertTrue(stress_problems(report))

    def test_a_zero_diversification_ratio_is_caught(self):
        report = self._report()
        report["scenarios"][STRESS_SCENARIO_HISTORICAL][
            "stressed_diversification"
        ] = 0.0
        self.assertTrue(stress_problems(report))


class RenderTest(unittest.TestCase):
    def test_every_scenario_is_rendered(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        text = "\n".join(render_stress(report))
        for name in STRESS_SCENARIOS:
            self.assertIn(name, text)

    def test_a_material_scenario_is_marked(self):
        tickers, matrix, _ = crashing_market()
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertIn("!!", "\n".join(render_stress(report)))

    def test_not_evaluated_renders_its_reason(self):
        tickers = ["A", "B", "C"]
        matrix = [[0.01, -0.02, 0.005, 0.001] for _ in tickers]
        report = stress_report(equal_weights(tickers), tickers, matrix)
        self.assertIn("NOT_EVALUATED", render_stress(report)[0])


if __name__ == "__main__":
    unittest.main()
