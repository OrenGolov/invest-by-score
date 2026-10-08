"""R1 position-exposure tests.

The behaviour under test is that WEIGHT IS NOT EXPOSURE. MEASURED on real
returns, a portfolio of 10% each in four correlated semiconductors looks four
times more diversified than one holding 40% NVDA outright, and is slightly
more volatile.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    EXPOSURE_ALIGN_BY_DATE,
    EXPOSURE_BASES,
    EXPOSURE_BASIS_RISK,
    EXPOSURE_BASIS_WEIGHT,
    EXPOSURE_BLOCKS_TRADES,
    EXPOSURE_MIN_SESSIONS,
    EXPOSURE_RISK_REVIEW,
    EXPOSURE_WEIGHT_REVIEW,
    POSITION_REQUIRED_FIELDS,
)
from core.position_exposure import (
    PositionExposureError,
    aligned_returns,
    covariance,
    exposure_problems,
    exposure_report,
    load_positions,
    marginal_exposure,
    market_values,
    portfolio_variance,
    position_problems,
    render_exposure,
    risk_contributions,
    weights,
)


def synthetic(sessions=400, seed=7):
    """Two correlated 'semis' and two independent names, by construction."""
    rng = random.Random(seed)
    dates = [f"d{index:04d}" for index in range(sessions)]
    market = [rng.gauss(0, 0.010) for _ in range(sessions)]
    series = {"SEMI_A": {}, "SEMI_B": {}, "INDY_A": {}, "INDY_B": {}}
    for index, date in enumerate(dates):
        shock = rng.gauss(0, 0.015)
        series["SEMI_A"][date] = market[index] + shock + rng.gauss(0, 0.004)
        series["SEMI_B"][date] = market[index] + shock + rng.gauss(0, 0.004)
        series["INDY_A"][date] = market[index] + rng.gauss(0, 0.008)
        series["INDY_B"][date] = market[index] + rng.gauss(0, 0.008)
    return series


SERIES = synthetic()
PRICES = {t: 100.0 for t in SERIES}


def positions(weight_map):
    return [
        {"ticker": t, "quantity": (w * 1_000_000) / PRICES[t], "as_of": "2026-09-21"}
        for t, w in weight_map.items()
    ]


class HoldingTests(unittest.TestCase):
    """A watchlist entry is not a holding."""

    def test_a_position_needs_a_quantity(self):
        self.assertIn("quantity", POSITION_REQUIRED_FIELDS)
        self.assertTrue(position_problems({"ticker": "AAPL", "as_of": "2026-09-21"}))

    def test_a_position_needs_an_as_of(self):
        self.assertTrue(position_problems({"ticker": "AAPL", "quantity": 10}))

    def test_a_complete_position_is_accepted(self):
        self.assertEqual(
            position_problems(
                {"ticker": "AAPL", "quantity": 10, "as_of": "2026-09-21"}
            ),
            [],
        )

    def test_a_watchlist_entry_is_refused(self):
        with self.assertRaises(PositionExposureError):
            load_positions([{"ticker": "AAPL"}])

    def test_a_non_numeric_quantity_is_refused(self):
        self.assertTrue(
            position_problems({"ticker": "A", "quantity": "ten", "as_of": "d"})
        )

    def test_an_infinite_quantity_is_refused(self):
        self.assertTrue(
            position_problems({"ticker": "A", "quantity": math.inf, "as_of": "d"})
        )

    def test_a_negative_quantity_is_allowed(self):
        # A short is a real position and must be representable.
        self.assertEqual(
            position_problems({"ticker": "A", "quantity": -10, "as_of": "d"}), []
        )

    def test_a_missing_price_is_refused(self):
        with self.assertRaises(PositionExposureError):
            market_values(
                load_positions([{"ticker": "A", "quantity": 1, "as_of": "d"}])
            )


class WeightIsNotExposureTests(unittest.TestCase):
    """The deciding measurement."""

    def setUp(self):
        self.concentrated = exposure_report(
            positions({"SEMI_A": 0.40, "INDY_A": 0.30, "INDY_B": 0.30}), SERIES, PRICES
        )
        self.spread = exposure_report(
            positions(
                {"SEMI_A": 0.20, "SEMI_B": 0.20, "INDY_A": 0.30, "INDY_B": 0.30}
            ),
            SERIES,
            PRICES,
        )

    def test_halving_the_weight_does_not_halve_the_risk(self):
        before = self.concentrated["risk_contributions"]["SEMI_A"]
        after = (
            self.spread["risk_contributions"]["SEMI_A"]
            + self.spread["risk_contributions"]["SEMI_B"]
        )
        self.assertGreater(after, before * 0.75)

    def test_the_spread_portfolio_has_a_lower_maximum_weight(self):
        self.assertLess(
            max(self.spread["weights"].values()),
            max(self.concentrated["weights"].values()),
        )

    def test_risk_contributions_sum_to_the_whole(self):
        total = sum(self.concentrated["risk_contributions"].values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_both_scales_are_reported(self):
        self.assertIn(EXPOSURE_BASIS_WEIGHT, EXPOSURE_BASES)
        self.assertIn(EXPOSURE_BASIS_RISK, EXPOSURE_BASES)
        self.assertTrue(self.concentrated["weights"])
        self.assertTrue(self.concentrated["risk_contributions"])

    def test_a_risk_flag_names_its_basis(self):
        risk_flags = [
            f
            for f in self.concentrated["flags"]
            if f["basis"] == EXPOSURE_BASIS_RISK
        ]
        self.assertTrue(risk_flags)
        self.assertTrue(all(f["reason"] for f in risk_flags))

    def test_the_report_is_contract_clean(self):
        self.assertEqual(exposure_problems(self.concentrated), [])


class AlignmentTests(unittest.TestCase):
    """Aligned by DATE, never by position."""

    def test_alignment_is_by_date(self):
        self.assertTrue(EXPOSURE_ALIGN_BY_DATE)

    def test_an_offset_series_still_correlates(self):
        # MEASURED in production data: a position-offset join reported MSFT's
        # correlation with everything as ~0.00, including against VOO (0.53).
        offset = {
            "SEMI_A": SERIES["SEMI_A"],
            "SEMI_B": {k: v for k, v in list(SERIES["SEMI_B"].items())[:-3]},
        }
        tickers, matrix = aligned_returns(offset)
        cov = covariance(matrix)
        correlation = cov[0][1] / math.sqrt(cov[0][0] * cov[1][1])
        self.assertGreater(correlation, 0.5)

    def test_only_shared_dates_are_used(self):
        offset = {
            "SEMI_A": SERIES["SEMI_A"],
            "SEMI_B": {k: v for k, v in list(SERIES["SEMI_B"].items())[:-3]},
        }
        _, matrix = aligned_returns(offset)
        self.assertEqual(len(matrix[0]), len(SERIES["SEMI_B"]) - 3)

    def test_a_thin_history_is_refused(self):
        thin = {t: dict(list(s.items())[:20]) for t, s in SERIES.items()}
        with self.assertRaises(PositionExposureError):
            aligned_returns(thin)

    def test_the_session_floor_supports_a_covariance(self):
        self.assertGreaterEqual(EXPOSURE_MIN_SESSIONS, 120)

    def test_a_single_observation_cannot_make_a_covariance(self):
        with self.assertRaises(PositionExposureError):
            covariance([[0.1]])


class MissingHistoryTests(unittest.TestCase):
    """A missing history is not zero risk."""

    def setUp(self):
        prices = dict(PRICES, UNKNOWN=100.0)
        self.report = exposure_report(
            [
                {"ticker": "SEMI_A", "quantity": 5000, "as_of": "d"},
                {"ticker": "UNKNOWN", "quantity": 5000, "as_of": "d"},
            ],
            SERIES,
            prices,
        )

    def test_risk_is_not_evaluated(self):
        self.assertEqual(self.report["risk_basis"], "NOT_EVALUATED")

    def test_no_risk_figures_are_invented(self):
        # Number(x ?? 0) would render the least-understood holding as safest.
        self.assertEqual(self.report["risk_contributions"], {})

    def test_the_reason_is_given(self):
        self.assertIn("UNKNOWN", self.report["risk_reason"])

    def test_weights_are_still_reported(self):
        self.assertTrue(self.report["weights"])

    def test_the_report_is_contract_clean(self):
        self.assertEqual(exposure_problems(self.report), [])


class MarginalExposureTests(unittest.TestCase):
    """The question Sprint R asks."""

    def setUp(self):
        self.tickers, matrix = aligned_returns(SERIES)
        self.cov = covariance(matrix)
        self.base = {"SEMI_A": 0.5, "INDY_A": 0.25, "INDY_B": 0.25}

    def test_a_correlated_addition_adds_more_risk(self):
        correlated = marginal_exposure(
            self.base, self.tickers, self.cov, "SEMI_B", 0.05
        )
        diversifier = marginal_exposure(
            self.base, self.tickers, self.cov, "INDY_B", 0.05
        )
        self.assertGreater(correlated["change"], diversifier["change"])

    def test_the_direction_is_named(self):
        result = marginal_exposure(self.base, self.tickers, self.cov, "SEMI_B", 0.05)
        self.assertIn(result["direction"], ("ADDS_RISK", "REDUCES_RISK", "NEUTRAL"))

    def test_an_unknown_ticker_is_refused(self):
        with self.assertRaises(PositionExposureError):
            marginal_exposure(self.base, self.tickers, self.cov, "NOPE", 0.05)

    def test_an_invalid_size_is_refused(self):
        for size in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(PositionExposureError):
                marginal_exposure(self.base, self.tickers, self.cov, "SEMI_B", size)


class ContractTests(unittest.TestCase):
    def test_exposure_does_not_block_trades(self):
        self.assertFalse(EXPOSURE_BLOCKS_TRADES)
        report = exposure_report(positions({"SEMI_A": 1.0}), SERIES, PRICES)
        self.assertFalse(report["blocks_trades"])

    def test_an_empty_portfolio_is_contract_clean(self):
        report = exposure_report([], SERIES, PRICES)
        self.assertEqual(exposure_problems(report), [])
        self.assertEqual(report["weights"], {})

    def test_weights_sum_to_one(self):
        computed = weights({"A": 60.0, "B": 40.0})
        self.assertAlmostEqual(sum(computed.values()), 1.0)

    def test_render_returns_one_line_per_holding(self):
        report = exposure_report(
            positions({"SEMI_A": 0.5, "INDY_A": 0.5}), SERIES, PRICES
        )
        self.assertEqual(len(render_exposure(report)), 2)

    def test_a_forged_weight_only_report_is_caught(self):
        forged = {"bases": [EXPOSURE_BASIS_WEIGHT], "weights": {"A": 1.0}}
        self.assertTrue(exposure_problems(forged))

    def test_the_review_thresholds_are_fractions(self):
        self.assertTrue(0.0 < EXPOSURE_WEIGHT_REVIEW < 1.0)
        self.assertTrue(0.0 < EXPOSURE_RISK_REVIEW < 1.0)


if __name__ == "__main__":
    unittest.main()
