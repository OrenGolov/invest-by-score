"""R2 correlation-sizing tests.

The behaviour under test is that the SAME risk budget buys different amounts
of different tickers, bounded twice, and that sizing never claims to fix
selection.
"""

from __future__ import annotations

import random
import unittest

from core.config import (
    SIZING_CAPPED,
    SIZING_IS_ADVISORY,
    SIZING_MAX_WEIGHT,
    SIZING_MIN_SESSIONS,
    SIZING_MIN_WEIGHT,
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
    SIZING_REPORT_BINDING_CONSTRAINT,
    SIZING_RISK_BUDGET,
    SIZING_SIZED,
    SIZING_VERDICTS,
)
from core.correlation_sizing import (
    BINDING_RISK_BUDGET,
    BINDING_WEIGHT_CAP,
    CorrelationSizingError,
    blended,
    compare_candidates,
    size_position,
    sizing_problems,
    sizing_report,
    volatility,
)
from core.position_exposure import aligned_returns, covariance


def synthetic(sessions=400, seed=11):
    rng = random.Random(seed)
    dates = [f"d{i:04d}" for i in range(sessions)]
    market = [rng.gauss(0, 0.008) for _ in range(sessions)]
    series = {
        "SEMI_A": {}, "SEMI_B": {}, "INDY_A": {}, "INDY_B": {},
        "LEVERED": {}, "AAA_SEMI": {}, "ZZZ_DIVERSIFIER": {},
    }
    for index, date in enumerate(dates):
        shock = rng.gauss(0, 0.018)
        series["SEMI_A"][date] = market[index] + shock + rng.gauss(0, 0.003)
        series["SEMI_B"][date] = market[index] + shock + rng.gauss(0, 0.003)
        series["INDY_A"][date] = rng.gauss(0, 0.009)
        series["INDY_B"][date] = rng.gauss(0, 0.009)
        series["LEVERED"][date] = 3.0 * (market[index] + shock)
        series["AAA_SEMI"][date] = series["SEMI_A"][date]
        series["ZZZ_DIVERSIFIER"][date] = series["INDY_B"][date]
    return series


SERIES = synthetic()
TICKERS, MATRIX = aligned_returns(SERIES)
COV = covariance(MATRIX)
HELD = {"SEMI_A": 0.60, "INDY_A": 0.40}


class CorrelationChangesTheSizeTests(unittest.TestCase):
    def setUp(self):
        self.correlated = size_position(HELD, "SEMI_B", TICKERS, COV)
        self.diversifier = size_position(HELD, "INDY_B", TICKERS, COV)

    def test_a_correlated_name_is_sized_smaller(self):
        self.assertLess(self.correlated["size"], self.diversifier["size"])

    def test_a_diversifier_raises_risk_less(self):
        self.assertLess(
            self.diversifier["volatility_increase"],
            self.correlated["volatility_increase"],
        )

    def test_the_risk_budget_is_respected(self):
        self.assertLessEqual(
            self.correlated["volatility_increase"], SIZING_RISK_BUDGET + 1e-6
        )

    def test_a_diversifying_position_is_named(self):
        self.assertTrue(self.diversifier["diversifying"])
        self.assertFalse(self.correlated["diversifying"])


class TwoBoundsTests(unittest.TestCase):
    """The risk budget alone does not bound concentration."""

    def test_a_diversifier_is_bound_by_the_weight_cap(self):
        result = size_position(HELD, "INDY_B", TICKERS, COV)
        self.assertEqual(result["binding"], BINDING_WEIGHT_CAP)
        self.assertEqual(result["verdict"], SIZING_CAPPED)

    def test_a_correlated_name_is_bound_by_the_risk_budget(self):
        result = size_position(HELD, "SEMI_B", TICKERS, COV)
        self.assertEqual(result["binding"], BINDING_RISK_BUDGET)
        self.assertEqual(result["verdict"], SIZING_SIZED)

    def test_no_size_exceeds_the_cap(self):
        for ticker in ("SEMI_B", "INDY_B", "AAA_SEMI", "ZZZ_DIVERSIFIER"):
            result = size_position(HELD, ticker, TICKERS, COV)
            if result.get("size") is not None:
                self.assertLessEqual(result["size"], SIZING_MAX_WEIGHT + 1e-9)

    def test_the_binding_constraint_is_always_reported(self):
        self.assertTrue(SIZING_REPORT_BINDING_CONSTRAINT)
        self.assertIn("binding", size_position(HELD, "SEMI_B", TICKERS, COV))


class RefusalTests(unittest.TestCase):
    """A position that cannot fit is refused, never shrunk silently."""

    def test_a_candidate_that_cannot_fit_is_refused(self):
        result = size_position(
            {"SEMI_A": 1.0}, "LEVERED", TICKERS, COV,
            risk_budget=0.001, min_weight=0.05,
        )
        self.assertEqual(result["verdict"], SIZING_REFUSED)

    def test_a_refused_proposal_carries_no_size(self):
        # The shape rule: 0.0 reads as "propose nothing", a different answer.
        result = size_position(
            {"SEMI_A": 1.0}, "LEVERED", TICKERS, COV,
            risk_budget=0.001, min_weight=0.05,
        )
        self.assertIsNone(result.get("size"))

    def test_refused_is_not_not_evaluated(self):
        self.assertNotEqual(SIZING_REFUSED, SIZING_NOT_EVALUATED)

    def test_an_unmeasurable_candidate_is_refused(self):
        with self.assertRaises(CorrelationSizingError):
            size_position(HELD, "NEVER_SEEN", TICKERS, COV)

    def test_a_zero_volatility_portfolio_is_not_evaluated(self):
        flat = {t: 0.0 for t in HELD}
        result = size_position(flat, "SEMI_B", TICKERS, COV)
        self.assertEqual(result["verdict"], SIZING_NOT_EVALUATED)

    def test_an_invalid_budget_is_refused(self):
        for budget in (0.0, 1.0, -0.1):
            with self.assertRaises(CorrelationSizingError):
                size_position(HELD, "SEMI_B", TICKERS, COV, risk_budget=budget)


class SizingCannotFixSelectionTests(unittest.TestCase):
    """MEASURED: reweighting buys ~1%, changing composition buys 10-34%."""

    def test_composition_beats_reweighting(self):
        equal = {"SEMI_A": 0.20, "SEMI_B": 0.20, "INDY_A": 0.60}
        tilted = {"SEMI_A": 0.28, "SEMI_B": 0.12, "INDY_A": 0.60}
        swapped = {"SEMI_A": 0.20, "INDY_B": 0.20, "INDY_A": 0.60}
        base = volatility(equal, TICKERS, COV)
        reweight = 1.0 - volatility(tilted, TICKERS, COV) / base
        swap = 1.0 - volatility(swapped, TICKERS, COV) / base
        self.assertGreater(swap, reweight * 3)


class FundingTests(unittest.TestCase):
    def test_a_trade_is_funded_pro_rata(self):
        mixed = blended(HELD, "INDY_B", 0.10)
        self.assertAlmostEqual(sum(mixed.values()), 1.0, places=9)

    def test_adding_to_an_existing_holding_accumulates(self):
        mixed = blended(HELD, "SEMI_A", 0.10)
        self.assertGreater(mixed["SEMI_A"], HELD["SEMI_A"] * 0.9)
        self.assertAlmostEqual(sum(mixed.values()), 1.0, places=9)


class OrderingTests(unittest.TestCase):
    def test_candidates_are_ordered_by_portfolio_effect(self):
        # The names deliberately disagree with the answer, so an alphabetical
        # sort cannot masquerade as a risk-ordered one.
        ordered = compare_candidates(
            HELD, ["AAA_SEMI", "ZZZ_DIVERSIFIER"], TICKERS, COV
        )
        self.assertEqual(ordered[0]["ticker"], "ZZZ_DIVERSIFIER")

    def test_an_unmeasurable_candidate_still_appears(self):
        ordered = compare_candidates(HELD, ["INDY_B", "NEVER_SEEN"], TICKERS, COV)
        self.assertEqual(len(ordered), 2)
        self.assertEqual(ordered[-1]["verdict"], SIZING_NOT_EVALUATED)


class ReportTests(unittest.TestCase):
    def test_a_report_is_contract_clean(self):
        report = sizing_report(HELD, ["SEMI_B", "INDY_B"], SERIES)
        self.assertEqual(sizing_problems(report), [])

    def test_a_thin_history_is_not_evaluated(self):
        thin = {t: dict(list(s.items())[:20]) for t, s in SERIES.items()}
        report = sizing_report(HELD, ["INDY_B"], thin)
        self.assertEqual(report["verdict"], SIZING_NOT_EVALUATED)
        self.assertEqual(sizing_problems(report), [])

    def test_a_missing_history_is_not_evaluated(self):
        report = sizing_report(HELD, ["NOT_IN_SERIES"], SERIES)
        self.assertEqual(report["verdict"], SIZING_NOT_EVALUATED)
        self.assertEqual(report["proposals"], [])

    def test_sizing_is_advisory(self):
        self.assertTrue(SIZING_IS_ADVISORY)
        self.assertTrue(sizing_report(HELD, ["INDY_B"], SERIES)["advisory"])

    def test_a_forged_oversized_proposal_is_caught(self):
        forged = {
            "advisory": True,
            "proposals": [
                {
                    "ticker": "X",
                    "verdict": SIZING_SIZED,
                    "reason": "forged",
                    "size": 0.95,
                    "binding": BINDING_RISK_BUDGET,
                }
            ],
        }
        self.assertTrue(sizing_problems(forged))

    def test_a_refused_proposal_with_a_size_is_caught(self):
        forged = {
            "advisory": True,
            "proposals": [
                {"ticker": "X", "verdict": SIZING_REFUSED, "reason": "no", "size": 0.1}
            ],
        }
        self.assertTrue(sizing_problems(forged))

    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(SIZING_VERDICTS[0], SIZING_NOT_EVALUATED)
        self.assertEqual(SIZING_VERDICTS[-1], SIZING_SIZED)

    def test_the_bounds_are_coherent(self):
        self.assertLess(SIZING_MIN_WEIGHT, SIZING_MAX_WEIGHT)
        self.assertGreaterEqual(SIZING_MIN_SESSIONS, 120)


if __name__ == "__main__":
    unittest.main()
