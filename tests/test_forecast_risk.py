"""R4 forecast-adjusted risk tests.

The central claim under test: confidence must shrink a position that raises
portfolio risk, and must NOT shrink one that lowers it. These tests build
their own covariance from a seeded generator rather than reading any data
file, so they mean the same thing on a clean clone.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    FORECAST_RISK_BLOCKS_TRADES,
    FORECAST_RISK_MAX_MULTIPLIER,
    FORECAST_RISK_MIN_CONFIDENCE,
    FORECAST_RISK_VERSION,
    FRISK_NO_TRADE,
    FRISK_NOT_EVALUATED,
    FRISK_REDUCED,
    FRISK_UNCHANGED,
    SIZING_MIN_WEIGHT,
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
    SIZING_SIZED,
)
from core.correlation_sizing import size_position
from core.forecast_risk import (
    ForecastRiskError,
    adjust_size,
    confidence_of,
    forecast_risk_problems,
    forecast_risk_report,
    multiplier_for,
    render_forecast_risk,
)
from core.position_exposure import aligned_returns, covariance, portfolio_variance


def market(seed=7, sessions=500):
    """Two correlated holdings, a levered twin, and a true diversifier."""
    rng = random.Random(seed)
    dates = [f"d{i:04d}" for i in range(sessions)]
    series = {"HELD_A": {}, "HELD_B": {}, "TWIN": {}, "DIVR": {}}
    for date in dates:
        base = rng.gauss(0, 0.008)
        shock = rng.gauss(0, 0.018)
        series["HELD_A"][date] = base + shock + rng.gauss(0, 0.003)
        series["HELD_B"][date] = base + shock + rng.gauss(0, 0.003)
        # Correlated with the holdings AND more volatile: a concentrator.
        series["TWIN"][date] = 1.4 * (base + shock)
        series["DIVR"][date] = rng.gauss(0, 0.009)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


HELD = {"HELD_A": 0.5, "HELD_B": 0.5}


def assessment(confidence):
    """A minimal F7-shaped assessment."""
    return {"assessed_object": "forecast", "confidence": confidence}


def proposal(size=0.10, *, diversifying=False, increase=0.05, verdict=SIZING_SIZED):
    """A minimal R2-shaped sizing proposal."""
    return {
        "ticker": "CAND",
        "verdict": verdict,
        "size": size,
        "diversifying": diversifying,
        "volatility_increase": increase,
    }


class TheDecidingMeasurementTest(unittest.TestCase):
    """Confidence-scaling helps one direction and hurts the other."""

    def test_scaling_a_risk_increaser_reduces_portfolio_volatility(self):
        tickers, cov = market()
        base = math.sqrt(portfolio_variance(HELD, tickers, cov))
        previous = None
        for confidence in (1.0, 0.75, 0.5, 0.25, 0.05):
            size = 0.20 * confidence
            blended = {
                "HELD_A": 0.5 * (1 - size),
                "HELD_B": 0.5 * (1 - size),
                "TWIN": size,
            }
            vol = math.sqrt(portfolio_variance(blended, tickers, cov))
            self.assertGreater(
                vol, base, "the TWIN candidate must RAISE portfolio volatility"
            )
            if previous is not None:
                self.assertLess(
                    vol,
                    previous,
                    "shrinking a risk increaser by confidence did not reduce "
                    "portfolio volatility",
                )
            previous = vol

    def test_scaling_a_risk_reducer_removes_the_benefit(self):
        """The negative result that shapes the module."""
        tickers, cov = market()
        base = math.sqrt(portfolio_variance(HELD, tickers, cov))
        previous = None
        for confidence in (1.0, 0.75, 0.5, 0.25, 0.05):
            size = 0.20 * confidence
            blended = {
                "HELD_A": 0.5 * (1 - size),
                "HELD_B": 0.5 * (1 - size),
                "DIVR": size,
            }
            vol = math.sqrt(portfolio_variance(blended, tickers, cov))
            self.assertLess(
                vol, base, "the DIVR candidate must LOWER portfolio volatility"
            )
            if previous is not None:
                self.assertGreater(
                    vol,
                    previous,
                    "shrinking a risk REDUCER by confidence lowered volatility "
                    "— if this ever holds, the increasers-only rule loses its "
                    "evidence and must be re-derived",
                )
            previous = vol

    def test_the_candidate_dominates_the_forecast(self):
        """A high-confidence concentrator is worse than a low-confidence diversifier."""
        tickers, cov = market()
        base = math.sqrt(portfolio_variance(HELD, tickers, cov))
        confident_concentrator = math.sqrt(
            portfolio_variance(
                {"HELD_A": 0.4, "HELD_B": 0.4, "TWIN": 0.20}, tickers, cov
            )
        )
        doubtful_diversifier = math.sqrt(
            portfolio_variance(
                {"HELD_A": 0.475, "HELD_B": 0.475, "DIVR": 0.05}, tickers, cov
            )
        )
        self.assertGreater(confident_concentrator, base)
        self.assertLess(doubtful_diversifier, base)
        self.assertGreater(
            confident_concentrator,
            doubtful_diversifier,
            "a high-confidence concentrator scored better than a "
            "low-confidence diversifier — confidence would be ranking "
            "candidates backwards",
        )


class HaircutDirectionTest(unittest.TestCase):
    def test_a_risk_increaser_is_cut_by_confidence(self):
        result = adjust_size(proposal(0.10, increase=0.05), assessment(0.5))
        self.assertEqual(result["verdict"], FRISK_REDUCED)
        self.assertAlmostEqual(result["adjusted_size"], 0.05, places=6)

    def test_a_risk_reducer_keeps_its_size(self):
        result = adjust_size(
            proposal(0.20, diversifying=True, increase=-0.19), assessment(0.30)
        )
        self.assertEqual(result["verdict"], FRISK_UNCHANGED)
        self.assertAlmostEqual(result["adjusted_size"], 0.20, places=6)
        self.assertEqual(result["multiplier"], 1.0)

    def test_a_risk_reducer_is_not_refused_on_low_confidence(self):
        """Declining a free variance reduction is the failure mode here."""
        result = adjust_size(
            proposal(0.20, diversifying=True, increase=-0.19), assessment(0.01)
        )
        self.assertNotEqual(result["verdict"], FRISK_NO_TRADE)
        self.assertEqual(result["adjusted_size"], 0.20)

    def test_direction_is_inferred_when_the_flag_is_absent(self):
        bare = {"ticker": "X", "verdict": SIZING_SIZED, "size": 0.10,
                "volatility_increase": -0.05}
        result = adjust_size(bare, assessment(0.30))
        self.assertTrue(result["diversifying"])
        self.assertEqual(result["verdict"], FRISK_UNCHANGED)


class HaircutNotMultiplierTest(unittest.TestCase):
    def test_full_confidence_earns_exactly_r2s_size(self):
        result = adjust_size(proposal(0.10), assessment(1.0))
        self.assertEqual(result["verdict"], FRISK_UNCHANGED)
        self.assertAlmostEqual(result["adjusted_size"], 0.10, places=6)

    def test_the_multiplier_never_exceeds_one(self):
        for confidence in (1.0, 0.99, 0.5):
            self.assertLessEqual(
                multiplier_for(confidence), FORECAST_RISK_MAX_MULTIPLIER
            )

    def test_the_size_never_grows(self):
        for confidence in (0.25, 0.5, 0.75, 1.0):
            result = adjust_size(proposal(0.10), assessment(confidence))
            if result["adjusted_size"] is not None:
                self.assertLessEqual(result["adjusted_size"], 0.10 + 1e-9)


class FloorTest(unittest.TestCase):
    def test_below_the_floor_is_no_trade_not_a_token_position(self):
        result = adjust_size(
            proposal(0.10), assessment(FORECAST_RISK_MIN_CONFIDENCE - 0.01)
        )
        self.assertEqual(result["verdict"], FRISK_NO_TRADE)
        self.assertIsNone(
            result["adjusted_size"],
            "a refusal carried a size — 0.0 and 'do not trade' must not share "
            "a representation",
        )

    def test_at_the_floor_a_position_is_still_proposed(self):
        result = adjust_size(proposal(0.10), assessment(FORECAST_RISK_MIN_CONFIDENCE))
        self.assertEqual(result["verdict"], FRISK_REDUCED)
        self.assertIsNotNone(result["adjusted_size"])

    def test_a_haircut_cannot_sneak_below_the_minimum_tradeable_weight(self):
        """R2 refuses below min_weight; R4 must not reach it another way."""
        tiny = SIZING_MIN_WEIGHT * 1.5
        result = adjust_size(proposal(tiny), assessment(0.30))
        self.assertEqual(result["verdict"], FRISK_NO_TRADE)
        self.assertIsNone(result["adjusted_size"])
        self.assertIn("minimum tradeable", result["reason"])


class R2IsNotOverruledTest(unittest.TestCase):
    def test_a_refused_position_stays_refused_however_confident(self):
        refused = {"ticker": "X", "verdict": SIZING_REFUSED, "size": None}
        result = adjust_size(refused, assessment(1.0))
        self.assertEqual(result["verdict"], FRISK_NOT_EVALUATED)
        self.assertIsNone(result.get("adjusted_size"))

    def test_an_unevaluated_position_is_not_invented(self):
        unevaluated = {"ticker": "X", "verdict": SIZING_NOT_EVALUATED, "size": None}
        result = adjust_size(unevaluated, assessment(1.0))
        self.assertEqual(result["verdict"], FRISK_NOT_EVALUATED)
        self.assertIsNone(result.get("adjusted_size"))


class UnmeasuredConfidenceTest(unittest.TestCase):
    """Missing is not zero — the Number(x ?? 0) failure the system refuses."""

    def test_no_assessment_is_not_evaluated_not_refused(self):
        result = adjust_size(proposal(0.10), None)
        self.assertEqual(result["verdict"], FRISK_NOT_EVALUATED)
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result.get("adjusted_size"))

    def test_an_assessment_without_a_confidence_is_not_zero(self):
        self.assertIsNone(confidence_of({"assessed_object": "forecast"}))
        result = adjust_size(proposal(0.10), {"assessed_object": "forecast"})
        self.assertEqual(result["verdict"], FRISK_NOT_EVALUATED)

    def test_a_missing_confidence_does_not_silently_refuse_every_trade(self):
        result = adjust_size(proposal(0.10), None)
        self.assertNotEqual(
            result["verdict"],
            FRISK_NO_TRADE,
            "an unassessed forecast was refused as though measured weak",
        )

    def test_a_confidence_outside_the_unit_interval_is_refused(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ForecastRiskError):
                confidence_of({"confidence": bad})

    def test_a_non_numeric_confidence_is_refused(self):
        with self.assertRaises(ForecastRiskError):
            confidence_of({"confidence": "high"})

    def test_a_non_mapping_assessment_is_refused(self):
        with self.assertRaises(ForecastRiskError):
            confidence_of(["confident"])


class ReportTest(unittest.TestCase):
    def test_a_clean_report_has_no_problems(self):
        tickers, cov = market()
        proposals = {
            t: size_position(HELD, t, tickers, cov) for t in ("TWIN", "DIVR")
        }
        report = forecast_risk_report(
            proposals, {"TWIN": assessment(0.5), "DIVR": assessment(0.5)}
        )
        self.assertEqual(forecast_risk_problems(report), [])
        self.assertEqual(report["version"], FORECAST_RISK_VERSION)

    def test_the_real_sizer_routes_both_directions(self):
        tickers, cov = market()
        proposals = {
            t: size_position(HELD, t, tickers, cov) for t in ("TWIN", "DIVR")
        }
        self.assertFalse(proposals["TWIN"]["diversifying"])
        self.assertTrue(proposals["DIVR"]["diversifying"])
        report = forecast_risk_report(
            proposals, {"TWIN": assessment(0.5), "DIVR": assessment(0.5)}
        )
        self.assertEqual(report["adjusted"]["TWIN"]["verdict"], FRISK_REDUCED)
        self.assertEqual(report["adjusted"]["DIVR"]["verdict"], FRISK_UNCHANGED)

    def test_counts_cover_every_candidate(self):
        report = forecast_risk_report(
            {"A": proposal(0.10), "B": proposal(0.10, diversifying=True, increase=-0.1)},
            {"A": assessment(0.5), "B": assessment(0.5)},
        )
        self.assertEqual(sum(report["counts"].values()), report["candidates"])

    def test_r4_does_not_block_trades(self):
        report = forecast_risk_report({"A": proposal()}, {"A": assessment(0.5)})
        self.assertFalse(report["blocks_trades"])
        self.assertFalse(FORECAST_RISK_BLOCKS_TRADES)

    def test_a_non_mapping_proposal_set_is_refused(self):
        with self.assertRaises(ForecastRiskError):
            forecast_risk_report([proposal()])

    def test_a_non_mapping_proposal_is_refused(self):
        with self.assertRaises(ForecastRiskError):
            adjust_size("size it", assessment(0.5))

    def test_an_empty_report_is_clean(self):
        report = forecast_risk_report({})
        self.assertEqual(forecast_risk_problems(report), [])
        self.assertEqual(report["candidates"], 0)


class ProblemDetectionTest(unittest.TestCase):
    """forecast_risk_problems must be able to fail."""

    def test_a_grown_size_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["adjusted"]["A"]["adjusted_size"] = 0.50
        self.assertTrue(forecast_risk_problems(report))

    def test_a_multiplier_above_one_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["adjusted"]["A"]["multiplier"] = 1.5
        self.assertTrue(forecast_risk_problems(report))

    def test_a_refusal_carrying_a_size_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.01)})
        self.assertEqual(report["adjusted"]["A"]["verdict"], FRISK_NO_TRADE)
        report["adjusted"]["A"]["adjusted_size"] = 0.0
        self.assertTrue(
            forecast_risk_problems(report),
            "a NO_TRADE carrying size 0.0 passed — that is exactly the "
            "collapse the shape rule exists to prevent",
        )

    def test_a_haircut_on_a_risk_reducer_is_caught(self):
        report = forecast_risk_report(
            {"A": proposal(0.10, diversifying=True, increase=-0.1)},
            {"A": assessment(0.5)},
        )
        report["adjusted"]["A"]["verdict"] = FRISK_REDUCED
        self.assertTrue(forecast_risk_problems(report))

    def test_a_refused_risk_reducer_is_caught(self):
        report = forecast_risk_report(
            {"A": proposal(0.10, diversifying=True, increase=-0.1)},
            {"A": assessment(0.5)},
        )
        report["adjusted"]["A"]["verdict"] = FRISK_NO_TRADE
        report["adjusted"]["A"]["adjusted_size"] = None
        self.assertTrue(forecast_risk_problems(report))

    def test_an_unknown_verdict_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["adjusted"]["A"]["verdict"] = "PROBABLY"
        self.assertTrue(forecast_risk_problems(report))

    def test_a_verdict_without_a_reason_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["adjusted"]["A"]["reason"] = "  "
        self.assertTrue(forecast_risk_problems(report))

    def test_a_report_that_blocks_trades_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["blocks_trades"] = True
        self.assertTrue(forecast_risk_problems(report))

    def test_an_inconsistent_size_is_caught(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        report["adjusted"]["A"]["adjusted_size"] = 0.049
        self.assertTrue(
            forecast_risk_problems(report),
            "an adjusted size that is not proposed x multiplier passed",
        )

    def test_an_unknown_verdict_cannot_be_constructed(self):
        with self.assertRaises(ForecastRiskError):
            from core.forecast_risk import _result

            _result("MAYBE", "because")


class RenderTest(unittest.TestCase):
    def test_every_candidate_is_rendered(self):
        report = forecast_risk_report(
            {"A": proposal(0.10), "B": proposal(0.10, diversifying=True, increase=-0.1)},
            {"A": assessment(0.5), "B": assessment(0.5)},
        )
        lines = render_forecast_risk(report)
        self.assertEqual(len(lines), 2)
        for ticker in ("A", "B"):
            self.assertTrue(any(ticker in line for line in lines))

    def test_a_refusal_renders_as_absent_not_zero(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.01)})
        for line in render_forecast_risk(report):
            # Only the R4 column is under test; "R2 10.00%" legitimately
            # contains the substring "0.00%".
            column = line.split("R4", 1)[1]
            self.assertNotIn(
                "0.00%",
                column,
                "a refused position rendered as 0.00%, which reads as a "
                "position of zero size rather than no position",
            )

    def test_an_unmeasured_confidence_renders_as_absent(self):
        report = forecast_risk_report({"A": proposal(0.10)}, {})
        line = render_forecast_risk(report)[0]
        self.assertNotIn("0.000", line)

    def test_an_empty_report_renders_without_error(self):
        self.assertEqual(render_forecast_risk(forecast_risk_report({})), [])


if __name__ == "__main__":
    unittest.main()
