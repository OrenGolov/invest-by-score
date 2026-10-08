"""R5 expected portfolio impact tests.

The central claim under test: volatility and concentration disagree often
enough that reporting either alone is misleading. These tests build their own
covariance from seeded generators rather than reading any data file, so they
mean the same thing on a clean clone.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    EXPECTED_IMPACT_VERSION,
    IMPACT_BLOCKS_TRADES,
    IMPACT_DIMENSION_CONCENTRATION,
    IMPACT_DIMENSION_LARGEST,
    IMPACT_DIMENSION_VOLATILITY,
    IMPACT_DIMENSIONS,
    IMPACT_IMPROVES,
    IMPACT_MATERIAL_CONCENTRATION,
    IMPACT_MATERIAL_VOLATILITY,
    IMPACT_NEUTRAL,
    IMPACT_NOT_EVALUATED,
    IMPACT_PROJECTS_RETURN,
    IMPACT_WORSENS,
)
from core.correlation_sizing import size_position
from core.expected_impact import (
    ExpectedImpactError,
    blend,
    concentration_of,
    impact_problems,
    impact_report,
    project_impact,
    render_impact,
)
from core.position_exposure import (
    aligned_returns,
    covariance,
    portfolio_variance,
    risk_contributions,
)


def reference(seed=11, sessions=500):
    """Two correlated holdings, a levered twin, and a true diversifier."""
    rng = random.Random(seed)
    series = {"HELD_A": {}, "HELD_B": {}, "TWIN": {}, "DIVR": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        base = rng.gauss(0, 0.008)
        shock = rng.gauss(0, 0.018)
        series["HELD_A"][date] = base + shock + rng.gauss(0, 0.003)
        series["HELD_B"][date] = base + shock + rng.gauss(0, 0.003)
        series["TWIN"][date] = 1.4 * (base + shock)
        series["DIVR"][date] = rng.gauss(0, 0.009)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


def opposing(seed=5, sessions=600):
    """A market where volatility and concentration point opposite ways.

    BIG dominates risk. CAND is correlated with BIG and more volatile, so
    adding it RAISES volatility while splitting the risk share away from BIG.
    """
    rng = random.Random(seed)
    series = {"BIG": {}, "SMALL": {}, "CAND": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        big = rng.gauss(0, 0.030)
        series["BIG"][date] = big
        series["SMALL"][date] = rng.gauss(0, 0.004)
        series["CAND"][date] = 1.3 * big + rng.gauss(0, 0.004)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


def doubling_down(seed=9, sessions=600):
    """Three independent names, one already the largest risk driver."""
    rng = random.Random(seed)
    series = {"BIG": {}, "A": {}, "B": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        series["BIG"][date] = rng.gauss(0, 0.025)
        series["A"][date] = rng.gauss(0, 0.010)
        series["B"][date] = rng.gauss(0, 0.010)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


HELD = {"HELD_A": 0.5, "HELD_B": 0.5}
SKEWED = {"BIG": 0.9, "SMALL": 0.1}
EVENLY = {"BIG": 0.34, "A": 0.33, "B": 0.33}


class TheDecidingMeasurementTest(unittest.TestCase):
    """Volatility and concentration disagree about half the time."""

    def test_the_dimensions_disagree_across_many_markets(self):
        def hhi(shares):
            return sum(v * v for v in shares.values())

        disagreements = 0
        total = 0
        for seed in range(1, 21):
            tickers, cov = reference(seed=seed)
            base_vol = math.sqrt(portfolio_variance(HELD, tickers, cov))
            base_hhi = hhi(risk_contributions(HELD, tickers, cov))
            for ticker, size in (("TWIN", 0.1278), ("DIVR", 0.20)):
                after = blend(HELD, ticker, size)
                vol = math.sqrt(portfolio_variance(after, tickers, cov))
                concentration = hhi(risk_contributions(after, tickers, cov))
                total += 1
                if (vol > base_vol) != (concentration > base_hhi):
                    disagreements += 1
        self.assertGreater(
            disagreements,
            total * 0.25,
            f"only {disagreements} of {total} trades moved volatility and "
            f"concentration in opposite directions. If this ever falls near "
            f"zero, reporting both dimensions loses its evidence and the "
            f"design must be re-derived rather than kept",
        )

    def test_a_trade_can_raise_volatility_while_improving_concentration(self):
        """The case a single risk number would hide."""
        tickers, cov = opposing()
        projection = project_impact(SKEWED, "CAND", 0.25, tickers, cov)
        volatility = projection["dimensions"][IMPACT_DIMENSION_VOLATILITY]
        concentration = projection["dimensions"][IMPACT_DIMENSION_CONCENTRATION]
        self.assertEqual(volatility["direction"], IMPACT_WORSENS)
        self.assertEqual(concentration["direction"], IMPACT_IMPROVES)
        self.assertTrue(
            projection["disagreement"]["disagree"],
            "the dimensions pointed opposite ways and the report did not say "
            "so — a reader seeing one direction would act on half the evidence",
        )

    def test_the_disagreement_names_both_sides(self):
        tickers, cov = opposing()
        projection = project_impact(SKEWED, "CAND", 0.25, tickers, cov)
        disagreement = projection["disagreement"]
        self.assertIn(IMPACT_DIMENSION_VOLATILITY, disagreement["worsening"])
        self.assertIn(IMPACT_DIMENSION_CONCENTRATION, disagreement["improving"])
        self.assertTrue(disagreement["reason"].strip())


class RefusesToProjectReturnTest(unittest.TestCase):
    """No trained model exists, so an expected return would be fabricated."""

    def test_the_report_does_not_project_a_return(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        self.assertFalse(projection["projects_return"])
        self.assertFalse(IMPACT_PROJECTS_RETURN)

    def test_no_return_field_is_emitted_anywhere(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        for key in projection:
            self.assertNotIn(
                "expected_return",
                str(key),
                "an expected_return field appeared; a 0.0 there renders as "
                "'flat' to any consumer that coalesces nulls",
            )
        self.assertNotIn("expected_return", projection["dimensions"])

    def test_the_refusal_carries_its_reason(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        self.assertIn("model", projection["return_reason"].lower())

    def test_a_report_claiming_to_project_return_is_caught(self):
        tickers, cov = reference()
        report = impact_report(HELD, {"TWIN": {"size": 0.10}}, tickers, cov)
        report["projects_return"] = True
        self.assertTrue(impact_problems(report))


class BlendTest(unittest.TestCase):
    def test_holdings_are_funded_pro_rata(self):
        blended = blend({"A": 0.6, "B": 0.4}, "C", 0.20)
        self.assertAlmostEqual(blended["C"], 0.20, places=9)
        self.assertAlmostEqual(blended["A"], 0.48, places=9)
        self.assertAlmostEqual(blended["B"], 0.32, places=9)

    def test_the_blended_portfolio_still_sums_to_one(self):
        for size in (0.0, 0.05, 0.25, 0.9):
            blended = blend({"A": 0.6, "B": 0.4}, "C", size)
            self.assertAlmostEqual(sum(blended.values()), 1.0, places=9)

    def test_adding_to_an_existing_holding_accumulates(self):
        blended = blend({"A": 0.5, "B": 0.5}, "A", 0.20)
        self.assertAlmostEqual(blended["A"], 0.5 * 0.8 + 0.20, places=9)
        self.assertAlmostEqual(sum(blended.values()), 1.0, places=9)

    def test_an_impossible_size_is_refused(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ExpectedImpactError):
                blend({"A": 1.0}, "B", bad)


class ConcentrationTest(unittest.TestCase):
    def test_an_empty_set_is_none_not_zero(self):
        self.assertIsNone(
            concentration_of({}),
            "an HHI of 0.0 over an empty set reads as perfect "
            "diversification, the opposite of unknown",
        )

    def test_equal_shares_give_the_one_over_n_floor(self):
        for n in (2, 4, 10):
            shares = {f"T{i}": 1.0 / n for i in range(n)}
            self.assertAlmostEqual(concentration_of(shares), 1.0 / n, places=6)


class SameCovarianceTest(unittest.TestCase):
    """Before and after must be measured on one matrix."""

    def test_the_before_state_matches_a_direct_measurement(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        direct = math.sqrt(portfolio_variance(HELD, tickers, cov))
        self.assertAlmostEqual(
            projection["before"]["volatility"], direct, places=8
        )

    def test_the_after_state_matches_the_blended_portfolio(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        direct = math.sqrt(
            portfolio_variance(blend(HELD, "TWIN", 0.10), tickers, cov)
        )
        self.assertAlmostEqual(projection["after"]["volatility"], direct, places=8)

    def test_a_zero_size_trade_changes_nothing(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.0, tickers, cov)
        self.assertAlmostEqual(
            projection["before"]["volatility"],
            projection["after"]["volatility"],
            places=8,
        )
        self.assertEqual(
            projection["dimensions"][IMPACT_DIMENSION_VOLATILITY]["direction"],
            IMPACT_NEUTRAL,
        )


class DirectionTest(unittest.TestCase):
    def test_lower_volatility_improves(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "DIVR", 0.20, tickers, cov)
        volatility = projection["dimensions"][IMPACT_DIMENSION_VOLATILITY]
        self.assertEqual(volatility["direction"], IMPACT_IMPROVES)
        self.assertLess(volatility["change"], 0)

    def test_a_small_move_is_neutral_not_a_direction(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.001, tickers, cov)
        volatility = projection["dimensions"][IMPACT_DIMENSION_VOLATILITY]
        self.assertEqual(volatility["direction"], IMPACT_NEUTRAL)
        self.assertLess(abs(volatility["relative_change"]), IMPACT_MATERIAL_VOLATILITY)

    def test_doubling_down_on_the_largest_driver_worsens_concentration(self):
        """Every direction must be reachable in BOTH senses.

        A direction function that always returned IMPROVES passed the whole
        gate until this case existed, because every other scenario happens to
        move DOWN and never contradicts the sign.
        """
        tickers, cov = doubling_down()
        projection = project_impact(EVENLY, "BIG", 0.30, tickers, cov)
        concentration = projection["dimensions"][IMPACT_DIMENSION_CONCENTRATION]
        largest = projection["dimensions"][IMPACT_DIMENSION_LARGEST]
        self.assertEqual(concentration["direction"], IMPACT_WORSENS)
        self.assertGreater(concentration["change"], 0)
        self.assertEqual(largest["direction"], IMPACT_WORSENS)
        self.assertGreater(largest["change"], 0)
        self.assertEqual(impact_problems(projection), [])

    def test_every_dimension_is_reported(self):
        tickers, cov = reference()
        projection = project_impact(HELD, "TWIN", 0.10, tickers, cov)
        for name in IMPACT_DIMENSIONS:
            self.assertIn(name, projection["dimensions"])
            self.assertTrue(projection["dimensions"][name]["reason"].strip())

    def test_a_ticker_without_history_is_refused(self):
        tickers, cov = reference()
        with self.assertRaises(ExpectedImpactError):
            project_impact(HELD, "UNKNOWN", 0.10, tickers, cov)

    def test_a_non_mapping_portfolio_is_refused(self):
        tickers, cov = reference()
        with self.assertRaises(ExpectedImpactError):
            project_impact(["HELD_A"], "TWIN", 0.10, tickers, cov)


class ReportTest(unittest.TestCase):
    def test_a_clean_report_has_no_problems(self):
        tickers, cov = reference()
        proposals = {
            t: size_position(HELD, t, tickers, cov) for t in ("TWIN", "DIVR")
        }
        report = impact_report(HELD, proposals, tickers, cov)
        self.assertEqual(impact_problems(report), [])
        self.assertEqual(report["version"], EXPECTED_IMPACT_VERSION)

    def test_it_reads_r4s_adjusted_size_in_preference_to_r2s(self):
        tickers, cov = reference()
        report = impact_report(
            HELD, {"TWIN": {"size": 0.20, "adjusted_size": 0.05}}, tickers, cov
        )
        self.assertAlmostEqual(report["projections"]["TWIN"]["size"], 0.05, places=6)

    def test_a_proposal_with_no_size_is_not_projected_as_no_impact(self):
        tickers, cov = reference()
        report = impact_report(
            HELD, {"TWIN": {"verdict": "NO_TRADE", "adjusted_size": None}}, tickers, cov
        )
        projection = report["projections"]["TWIN"]
        self.assertIsNone(projection["size"])
        for name in IMPACT_DIMENSIONS:
            self.assertEqual(
                projection["dimensions"][name]["direction"],
                IMPACT_NOT_EVALUATED,
                "a trade with no size was projected as having no impact, "
                "which is a different claim",
            )
        self.assertEqual(impact_problems(report), [])

    def test_disagreeing_candidates_are_listed(self):
        tickers, cov = opposing()
        report = impact_report(SKEWED, {"CAND": {"size": 0.25}}, tickers, cov)
        self.assertIn("CAND", report["disagreeing"])

    def test_r5_does_not_block_trades(self):
        tickers, cov = reference()
        report = impact_report(HELD, {"TWIN": {"size": 0.10}}, tickers, cov)
        self.assertFalse(report["blocks_trades"])
        self.assertFalse(IMPACT_BLOCKS_TRADES)

    def test_an_empty_report_is_clean(self):
        tickers, cov = reference()
        report = impact_report(HELD, {}, tickers, cov)
        self.assertEqual(impact_problems(report), [])
        self.assertEqual(report["candidates"], 0)

    def test_a_non_mapping_proposal_set_is_refused(self):
        tickers, cov = reference()
        with self.assertRaises(ExpectedImpactError):
            impact_report(HELD, [{"size": 0.1}], tickers, cov)


class ProblemDetectionTest(unittest.TestCase):
    """impact_problems must be able to fail."""

    def _report(self):
        tickers, cov = reference()
        return impact_report(HELD, {"TWIN": {"size": 0.10}}, tickers, cov)

    def test_a_missing_dimension_is_caught(self):
        report = self._report()
        del report["projections"]["TWIN"]["dimensions"][IMPACT_DIMENSION_CONCENTRATION]
        self.assertTrue(
            impact_problems(report),
            "a report omitting concentration passed — MEASURED, volatility "
            "alone answers only half the question",
        )

    def test_an_unknown_direction_is_caught(self):
        report = self._report()
        report["projections"]["TWIN"]["dimensions"][IMPACT_DIMENSION_VOLATILITY][
            "direction"
        ] = "BETTER_ISH"
        self.assertTrue(impact_problems(report))

    def test_a_direction_contradicting_its_sign_is_caught(self):
        report = self._report()
        dimension = report["projections"]["TWIN"]["dimensions"][
            IMPACT_DIMENSION_CONCENTRATION
        ]
        dimension["direction"] = IMPACT_IMPROVES
        dimension["change"] = 0.5
        self.assertTrue(
            impact_problems(report),
            "IMPROVES with a positive change passed — lower is better on "
            "every dimension R5 projects",
        )

    def test_a_not_evaluated_carrying_a_change_is_caught(self):
        report = self._report()
        dimension = report["projections"]["TWIN"]["dimensions"][
            IMPACT_DIMENSION_LARGEST
        ]
        dimension["direction"] = IMPACT_NOT_EVALUATED
        dimension["change"] = 0.1
        self.assertTrue(impact_problems(report))

    def test_a_direction_with_no_change_is_caught(self):
        report = self._report()
        report["projections"]["TWIN"]["dimensions"][IMPACT_DIMENSION_VOLATILITY][
            "change"
        ] = None
        self.assertTrue(impact_problems(report))

    def test_an_unstated_disagreement_is_caught(self):
        tickers, cov = opposing()
        report = impact_report(SKEWED, {"CAND": {"size": 0.25}}, tickers, cov)
        self.assertTrue(report["projections"]["CAND"]["disagreement"]["disagree"])
        report["projections"]["CAND"]["disagreement"]["disagree"] = False
        self.assertTrue(
            impact_problems(report),
            "a silenced disagreement passed the contract check",
        )

    def test_a_dimension_with_no_reason_is_caught(self):
        report = self._report()
        report["projections"]["TWIN"]["dimensions"][IMPACT_DIMENSION_VOLATILITY][
            "reason"
        ] = "  "
        self.assertTrue(impact_problems(report))

    def test_a_report_that_blocks_trades_is_caught(self):
        report = self._report()
        report["blocks_trades"] = True
        self.assertTrue(impact_problems(report))

    def test_a_missing_return_reason_is_caught(self):
        report = self._report()
        report["return_reason"] = ""
        self.assertTrue(impact_problems(report))


class RenderTest(unittest.TestCase):
    def test_every_candidate_and_dimension_is_rendered(self):
        tickers, cov = reference()
        report = impact_report(
            HELD, {"TWIN": {"size": 0.10}, "DIVR": {"size": 0.10}}, tickers, cov
        )
        lines = render_impact(report)
        text = "\n".join(lines)
        for ticker in ("TWIN", "DIVR"):
            self.assertIn(ticker, text)
        for name in IMPACT_DIMENSIONS:
            self.assertIn(name, text)

    def test_a_disagreement_is_called_out_in_the_render(self):
        tickers, cov = opposing()
        report = impact_report(SKEWED, {"CAND": {"size": 0.25}}, tickers, cov)
        text = "\n".join(render_impact(report))
        self.assertIn("worsens", text.lower())

    def test_an_unsized_candidate_renders_as_absent(self):
        tickers, cov = reference()
        report = impact_report(HELD, {"TWIN": {"adjusted_size": None}}, tickers, cov)
        line = render_impact(report)[0]
        self.assertNotIn("0.00%", line)

    def test_an_empty_report_renders_without_error(self):
        tickers, cov = reference()
        self.assertEqual(render_impact(impact_report(HELD, {}, tickers, cov)), [])


if __name__ == "__main__":
    unittest.main()
