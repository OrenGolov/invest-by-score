"""R3 sector concentration tests.

The central claim under test: a portfolio can satisfy every per-name weight
cap and still be a single-sector bet. These tests build that portfolio
explicitly rather than depending on any data file, so they mean the same
thing on a clean clone as they do on the machine that wrote them.
"""

from __future__ import annotations

import unittest

from core.config import (
    SECTOR_BASIS_RISK,
    SECTOR_BASIS_WEIGHT,
    SECTOR_BLOCKS_TRADES,
    SECTOR_CONCENTRATION_VERSION,
    SECTOR_HHI_REVIEW,
    SECTOR_RISK_REVIEW,
    SECTOR_UNCLASSIFIED,
    SECTOR_UNCLASSIFIED_REVIEW,
    SECTOR_WEIGHT_REVIEW,
)
from core.market_context import sector_for
from core.sector_concentration import (
    SectorConcentrationError,
    classify,
    concentration_problems,
    concentration_report,
    effective_sectors,
    group_shares,
    herfindahl,
    render_concentration,
)

# Names whose GICS sector the shared map already knows. Chosen so the tests
# read against the real map rather than a fixture that could drift from it.
IT = ("NVDA", "AMD", "AVGO", "QCOM", "MSFT", "AAPL", "ORCL", "CSCO")
HEALTH = ("LLY", "VRTX")
ENERGY = ("BKR",)
FUND = "VOO"


def _exposure(weights, *, risk=None, reason=""):
    """A minimal R1-shaped exposure report."""
    report = {
        "version": "position-exposure-v1",
        "positions": len(weights),
        "weights": dict(weights),
        "bases": [SECTOR_BASIS_WEIGHT, SECTOR_BASIS_RISK],
        "blocks_trades": False,
    }
    if risk is None:
        report["risk_contributions"] = {}
        report["risk_basis"] = "NOT_EVALUATED"
        report["risk_reason"] = reason or "no return history"
    else:
        report["risk_contributions"] = dict(risk)
        report["risk_basis"] = SECTOR_BASIS_RISK
        report["risk_reason"] = ""
    return report


def _equal(tickers):
    share = 1.0 / len(tickers)
    return {t: share for t in tickers}


class SectorMapReuseTest(unittest.TestCase):
    """R3 must not carry a second sector table."""

    def test_classification_comes_from_the_shared_map(self):
        for ticker in IT:
            self.assertEqual(
                classify([ticker])[ticker],
                sector_for(ticker),
                f"{ticker} classified differently from market_context",
            )

    def test_every_holding_appears_in_the_classification(self):
        tickers = list(IT) + [FUND]
        mapping = classify(tickers)
        self.assertEqual(sorted(mapping), sorted(tickers))

    def test_a_fund_is_unclassified_not_guessed(self):
        self.assertEqual(classify([FUND])[FUND], SECTOR_UNCLASSIFIED)

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(SectorConcentrationError):
            classify(["NVDA", ""])


class TheDecidingMeasurementTest(unittest.TestCase):
    """Every per-name cap passes while the book is one sector."""

    def test_per_name_caps_pass_while_the_sector_is_flagged(self):
        weights = _equal(IT)
        # Every holding is far below R1's per-name review threshold...
        for ticker, weight in weights.items():
            self.assertLess(
                weight,
                SECTOR_WEIGHT_REVIEW,
                f"{ticker} breaches the per-name threshold; this test needs a "
                f"portfolio that does NOT",
            )
        # ...and yet the sector is 100% of the book.
        report = concentration_report(_exposure(weights))
        self.assertEqual(concentration_problems(report), [])
        flagged = {f["sector"] for f in report["flags"]}
        self.assertIn(
            sector_for("NVDA"),
            flagged,
            "a single-sector book cleared every per-name cap AND went "
            "unflagged at the sector level — this is exactly the blind spot "
            "R3 exists to close",
        )

    def test_splitting_a_position_across_the_sector_does_not_reduce_it(self):
        """The R1 gaming move, one level up: it must not help."""
        concentrated = concentration_report(_exposure({"NVDA": 1.0}))
        split = concentration_report(_exposure(_equal(IT)))
        sector = sector_for("NVDA")
        self.assertAlmostEqual(
            concentrated["weight_by_sector"][sector],
            split["weight_by_sector"][sector],
            places=6,
            msg="splitting one name across eight in the same sector changed "
            "the sector share — the cap would be gameable again",
        )


class HerfindahlTest(unittest.TestCase):
    def test_equal_sectors_give_the_one_over_n_floor(self):
        for n in (2, 4, 10):
            shares = {f"S{i}": 1.0 / n for i in range(n)}
            self.assertAlmostEqual(herfindahl(shares), 1.0 / n, places=6)
            self.assertAlmostEqual(effective_sectors(herfindahl(shares)), n, places=4)

    def test_a_single_sector_scores_one(self):
        self.assertAlmostEqual(herfindahl({"S0": 1.0}), 1.0, places=6)
        self.assertAlmostEqual(effective_sectors(1.0), 1.0, places=6)

    def test_unclassified_is_excluded_and_the_rest_renormalised(self):
        """A fund bucket is not a concentrated position in anything."""
        with_fund = {"S0": 0.5, "S1": 0.25, SECTOR_UNCLASSIFIED: 0.25}
        without = {"S0": 2 / 3, "S1": 1 / 3}
        self.assertAlmostEqual(
            herfindahl(with_fund), herfindahl(without), places=6
        )

    def test_nothing_classified_is_none_not_zero(self):
        self.assertIsNone(
            herfindahl({SECTOR_UNCLASSIFIED: 1.0}),
            "an HHI of 0.0 over an empty set reads as perfect "
            "diversification, which is the opposite of unknown",
        )
        self.assertIsNone(effective_sectors(None))

    def test_effective_count_never_exceeds_sectors_held(self):
        for shares in (
            {"S0": 0.9, "S1": 0.1},
            {"S0": 0.4, "S1": 0.35, "S2": 0.25},
            {"S0": 0.52, "S1": 0.15, "S2": 0.08, "S3": 0.25},
        ):
            hhi = herfindahl(shares)
            self.assertLessEqual(effective_sectors(hhi), len(shares) + 1e-6)


class GroupingTest(unittest.TestCase):
    def test_shares_are_preserved_not_dropped(self):
        weights = _equal(list(IT) + list(HEALTH) + [FUND])
        grouped = group_shares(weights, classify(weights))
        self.assertAlmostEqual(sum(grouped.values()), 1.0, places=6)

    def test_unclassified_holdings_accumulate_in_their_own_bucket(self):
        weights = {"NVDA": 0.5, FUND: 0.5}
        grouped = group_shares(weights, classify(weights))
        self.assertAlmostEqual(grouped[SECTOR_UNCLASSIFIED], 0.5, places=6)


class ReportContractTest(unittest.TestCase):
    def test_a_clean_report_has_no_problems(self):
        weights = _equal(list(IT) + list(HEALTH) + list(ENERGY))
        report = concentration_report(_exposure(weights, risk=weights))
        self.assertEqual(concentration_problems(report), [])
        self.assertEqual(report["version"], SECTOR_CONCENTRATION_VERSION)

    def test_both_bases_are_carried(self):
        report = concentration_report(_exposure(_equal(IT)))
        self.assertIn(SECTOR_BASIS_WEIGHT, report["bases"])
        self.assertIn(SECTOR_BASIS_RISK, report["bases"])

    def test_r3_does_not_block_trades(self):
        report = concentration_report(_exposure(_equal(IT)))
        self.assertFalse(report["blocks_trades"])
        self.assertFalse(SECTOR_BLOCKS_TRADES)

    def test_unclassified_is_not_counted_as_a_sector(self):
        weights = {"NVDA": 0.5, FUND: 0.5}
        report = concentration_report(_exposure(weights))
        self.assertEqual(
            report["sectors_present"],
            1,
            "UNCLASSIFIED was counted among the sectors held",
        )
        self.assertNotIn(
            SECTOR_UNCLASSIFIED,
            {f["sector"] for f in report["flags"] if f["basis"] == SECTOR_BASIS_RISK},
        )

    def test_a_non_mapping_exposure_is_refused(self):
        for bad in (None, [], "exposure", 3):
            with self.assertRaises(SectorConcentrationError):
                concentration_report(bad)

    def test_an_empty_portfolio_reports_no_false_diversification(self):
        report = concentration_report(_exposure({}))
        self.assertEqual(concentration_problems(report), [])
        self.assertIsNone(report["hhi"])
        self.assertIsNone(report["effective_sectors"])
        self.assertEqual(report["sectors_present"], 0)

    def test_benchmarks_name_a_real_etf_per_sector(self):
        weights = _equal(list(IT) + list(HEALTH) + [FUND])
        report = concentration_report(_exposure(weights))
        self.assertNotIn(SECTOR_UNCLASSIFIED, report["benchmarks"])
        for sector, etf in report["benchmarks"].items():
            self.assertTrue(etf, f"{sector} mapped to an empty benchmark")
            self.assertIn(sector, report["weight_by_sector"])


class ShapeRuleTest(unittest.TestCase):
    """A sector risk figure exists IFF it was measured."""

    def test_unmeasured_risk_is_not_summed_into_sectors(self):
        weights = _equal(IT)
        report = concentration_report(
            _exposure(weights, risk=None, reason="no return history for AVGO")
        )
        self.assertEqual(report["risk_basis"], "NOT_EVALUATED")
        self.assertEqual(
            report["risk_by_sector"],
            {},
            "absent per-name risk was summed into a sector total, which "
            "would render the least-understood sector as the safest",
        )
        self.assertTrue(report["risk_reason"].strip())
        self.assertEqual(concentration_problems(report), [])

    def test_the_underlying_reason_is_carried_through_not_replaced(self):
        report = concentration_report(
            _exposure({}, risk=None, reason="no return history for AVGO")
        )
        self.assertIn("AVGO", report["risk_reason"])

    def test_measured_risk_is_grouped_by_sector(self):
        weights = _equal(list(IT) + list(HEALTH))
        report = concentration_report(_exposure(weights, risk=weights))
        self.assertEqual(report["risk_basis"], SECTOR_BASIS_RISK)
        self.assertAlmostEqual(
            sum(report["risk_by_sector"].values()), 1.0, places=6
        )
        self.assertEqual(report["risk_reason"], "")

    def test_a_report_claiming_unevaluated_risk_with_shares_is_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["risk_by_sector"] = {"Information Technology": 1.0}
        self.assertTrue(
            concentration_problems(report),
            "NOT_EVALUATED alongside reported shares went undetected",
        )

    def test_a_report_claiming_unevaluated_risk_with_no_reason_is_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["risk_reason"] = "  "
        self.assertTrue(concentration_problems(report))


class FlagTest(unittest.TestCase):
    def test_a_heavy_sector_is_flagged_on_weight(self):
        weights = _equal(IT)
        report = concentration_report(_exposure(weights))
        weight_flags = [
            f for f in report["flags"] if f["basis"] == SECTOR_BASIS_WEIGHT
        ]
        self.assertTrue(weight_flags)
        for flag in weight_flags:
            self.assertTrue(flag["reason"].strip())

    def test_a_heavy_sector_is_flagged_on_risk(self):
        weights = _equal(list(IT) + list(HEALTH) + list(ENERGY))
        risk = dict(weights)
        # Concentrate variance in one sector without changing any weight.
        for ticker in IT:
            risk[ticker] = SECTOR_RISK_REVIEW / len(IT) + 0.02
        total = sum(risk.values())
        risk = {t: v / total for t, v in risk.items()}
        report = concentration_report(_exposure(weights, risk=risk))
        self.assertEqual(concentration_problems(report), [])
        risk_flags = [f for f in report["flags"] if f["basis"] == SECTOR_BASIS_RISK]
        self.assertTrue(
            risk_flags,
            "a sector above the variance threshold went unflagged — the "
            "scale a weight cap cannot see",
        )

    def test_hhi_is_flagged_when_diversification_is_thin(self):
        weights = _equal(IT)
        report = concentration_report(_exposure(weights))
        hhi_flags = [f for f in report["flags"] if f["sector"] == "*"]
        self.assertTrue(hhi_flags)
        self.assertGreaterEqual(hhi_flags[0]["value"], SECTOR_HHI_REVIEW)

    def test_a_spread_portfolio_is_not_flagged(self):
        """The gate must be capable of staying quiet."""
        # Twelve synthetic sectors at equal weight: HHI 1/12 = 0.083.
        weights = {f"T{i}": 1 / 12 for i in range(12)}
        sectors = {f"T{i}": f"Sector {i}" for i in range(12)}
        grouped = group_shares(weights, sectors)
        hhi = herfindahl(grouped)
        self.assertLess(
            hhi,
            SECTOR_HHI_REVIEW,
            "an evenly spread book scored above the HHI review threshold, so "
            "the threshold flags everything and means nothing",
        )

    def test_a_mostly_unclassified_book_is_flagged(self):
        weights = {"NVDA": 0.3, FUND: 0.7}
        report = concentration_report(_exposure(weights))
        flags = [f for f in report["flags"] if f["sector"] == SECTOR_UNCLASSIFIED]
        self.assertTrue(
            flags,
            "70% of the book had no sector and the report presented its "
            "sector shares without qualification",
        )
        self.assertGreaterEqual(flags[0]["value"], SECTOR_UNCLASSIFIED_REVIEW)

    def test_every_flag_names_its_basis_and_reason(self):
        weights = _equal(list(IT) + [FUND])
        report = concentration_report(_exposure(weights, risk=weights))
        for flag in report["flags"]:
            self.assertIn(flag["basis"], (SECTOR_BASIS_WEIGHT, SECTOR_BASIS_RISK))
            self.assertTrue(flag["reason"].strip())
            self.assertIn("threshold", flag)


class ProblemDetectionTest(unittest.TestCase):
    """concentration_problems must be able to fail."""

    def test_weights_that_do_not_sum_are_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["weight_by_sector"] = {"Information Technology": 0.5}
        self.assertTrue(concentration_problems(report))

    def test_a_missing_risk_basis_is_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["bases"] = [SECTOR_BASIS_WEIGHT]
        self.assertTrue(
            concentration_problems(report),
            "a weight-only report passed the contract check",
        )

    def test_a_report_that_blocks_trades_is_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["blocks_trades"] = True
        self.assertTrue(concentration_problems(report))

    def test_an_impossible_hhi_is_caught(self):
        report = concentration_report(_exposure(_equal(list(IT) + list(HEALTH))))
        report["hhi"] = 0.01  # below the 1/n floor for 2 sectors
        self.assertTrue(concentration_problems(report))

    def test_effective_sectors_without_an_hhi_is_caught(self):
        report = concentration_report(_exposure({}))
        report["effective_sectors"] = 4.0
        self.assertTrue(concentration_problems(report))

    def test_a_flag_on_an_unknown_basis_is_caught(self):
        report = concentration_report(_exposure(_equal(IT)))
        report["flags"] = [
            {"sector": "X", "basis": "vibes", "value": 1.0, "reason": "because"}
        ]
        self.assertTrue(concentration_problems(report))


class RenderTest(unittest.TestCase):
    def test_every_sector_appears_heaviest_first(self):
        weights = _equal(list(IT) + list(HEALTH) + [FUND])
        report = concentration_report(_exposure(weights))
        lines = render_concentration(report)
        self.assertTrue(lines)
        for sector in report["weight_by_sector"]:
            self.assertTrue(
                any(sector in line for line in lines), f"{sector} not rendered"
            )
        shown = [
            s
            for s in sorted(
                report["weight_by_sector"],
                key=lambda k: (-report["weight_by_sector"][k], k),
            )
        ]
        positions = [
            next(i for i, line in enumerate(lines) if s in line) for s in shown
        ]
        self.assertEqual(positions, sorted(positions))

    def test_unmeasured_risk_renders_as_absent_not_zero(self):
        report = concentration_report(_exposure(_equal(IT)))
        for line in render_concentration(report):
            if "risk" not in line:
                continue
            # Only the risk column is under test; "weight 100.0%" legitimately
            # contains the substring "0.0%".
            column = line.split("risk", 1)[1]
            self.assertNotIn(
                "0.0%",
                column,
                "unmeasured sector risk rendered as 0.0%, which reads as "
                "'carries no risk'",
            )

    def test_an_empty_portfolio_renders_without_error(self):
        self.assertEqual(render_concentration(concentration_report(_exposure({}))), [])


if __name__ == "__main__":
    unittest.main()
