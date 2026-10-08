"""R7 portfolio-level NO_TRADE tests.

The central claim under test: a set of individually-correct trades can be
collectively impossible, and only a portfolio-level check can see it. These
tests build their own covariance from a seeded generator rather than reading
any data file, so they mean the same thing on a clean clone.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    PORTFOLIO_BLOCKS_TRADES,
    PORTFOLIO_DECISION_VERSION,
    PORTFOLIO_FAIL_CLOSED,
    PORTFOLIO_MAX_TOTAL_SIZE,
    PORTFOLIO_MAX_VOLATILITY_INCREASE,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    PORTFOLIO_REDUCED,
    PORTFOLIO_RULES,
    PORTFOLIO_SEVERITY_VETO,
    PORTFOLIO_SEVERITY_WARN,
    SIZING_RISK_BUDGET,
)
from core.correlation_sizing import size_position
from core.portfolio_decision import (
    PortfolioDecisionError,
    accepted_sizes,
    decision_problems,
    evaluate_portfolio,
    render_decision,
    set_volatility_increase,
)
from core.position_exposure import covariance, portfolio_variance
from core.risk_policy import evaluate_risk_policy

BASE = ["D1", "D2", "D3", "D4"]
HELD = {t: 0.25 for t in BASE}


def clustered_market(seed=41, sessions=800, candidates=6):
    """A diversified book plus candidates that all share one factor.

    Each candidate diversifies the ORIGINAL book, which is why R2 approves
    every one of them. They do not diversify each other.
    """
    rng = random.Random(seed)
    cands = [f"C{i}" for i in range(candidates)]
    names = BASE + cands
    matrix = [[0.0] * sessions for _ in names]
    crash = set(rng.sample(range(sessions), 80))
    for day in range(sessions):
        if day in crash:
            common = rng.gauss(-0.040, 0.005)
            for i in range(len(names)):
                matrix[i][day] = common + rng.gauss(0, 0.025)
        else:
            market = rng.gauss(0.0012, 0.004)
            cluster = rng.gauss(0, 0.020)
            for i in range(len(BASE)):
                matrix[i][day] = market + rng.gauss(0, 0.014)
            for j in range(candidates):
                matrix[len(BASE) + j][day] = market + cluster + rng.gauss(0, 0.005)
    return names, cands, covariance(matrix)


def clean_concentration():
    return {"flags": []}


def clean_stress():
    return {"status": "MEASURED", "material": [], "scenarios": {}}


def proposal(ticker, size, **extra):
    payload = {"ticker": ticker, "verdict": "SIZED", "size": size}
    payload.update(extra)
    return payload


class TheDecidingMeasurementTest(unittest.TestCase):
    """Individually correct, collectively impossible."""

    def test_r2_approves_every_candidate_in_isolation(self):
        names, cands, cov = clustered_market()
        for ticker in cands:
            sized = size_position(HELD, ticker, names, cov)
            self.assertIsNotNone(sized.get("size"))
            self.assertTrue(
                sized.get("diversifying"),
                f"{ticker} was not reported as diversifying; this test needs "
                f"candidates R2 approves individually",
            )

    def test_the_approved_set_exceeds_the_whole_book(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        total = sum(accepted_sizes(proposals).values())
        self.assertGreater(
            total,
            1.0,
            f"the individually-approved set totals {total:.1%}; this test "
            f"needs a set that cannot be executed",
        )

    def test_the_set_raises_volatility_although_each_member_lowers_it(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        for sized in proposals.values():
            self.assertLess(
                sized["volatility_increase"],
                0,
                "a candidate raised volatility on its own; the finding is "
                "about candidates that each LOWER it",
            )
        joint = set_volatility_increase(HELD, accepted_sizes(proposals), names, cov)
        self.assertGreater(
            joint,
            0,
            "the set of individually volatility-reducing trades did not raise "
            "volatility together — if this ever holds, R7's central "
            "measurement must be re-derived rather than kept",
        )

    def test_r7_refuses_the_set_r2_approved(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        decision = evaluate_portfolio(HELD, proposals, names, cov)
        self.assertEqual(decision["verdict"], PORTFOLIO_NO_TRADE)
        self.assertIn("total_size_exceeds_budget", decision["veto_rule_ids"])
        self.assertEqual(decision_problems(decision), [])


class DoesNotDuplicateW2Test(unittest.TestCase):
    """W2 vetoes on evidence; R7 vetoes on the portfolio."""

    def test_w2_approves_a_trade_r7_refuses(self):
        context = {
            "market_data_quality": 95.0,
            "market_source_confidence": 0.95,
            "market_timestamp_valid": True,
            "fundamental_point_in_time_valid": True,
            "fundamental_source_confidence": 0.9,
            "fundamental_source_status": "ok",
            "score": 80.0,
            "action": "BUY",
            "confidence": 0.85,
            "confidence_breakdown": {"factors": []},
            "market_regime": "bullish",
        }
        self.assertFalse(
            evaluate_risk_policy(context)["veto"],
            "this test needs per-ticker evidence W2 accepts",
        )
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        decision = evaluate_portfolio(HELD, proposals, names, cov)
        self.assertEqual(
            decision["verdict"],
            PORTFOLIO_NO_TRADE,
            "perfect per-ticker evidence passed W2 and R7 did not refuse the "
            "portfolio-level problem — R7 would add nothing",
        )

    def test_r7_uses_w2s_veto_vocabulary(self):
        self.assertEqual(PORTFOLIO_SEVERITY_VETO, "veto")


class FailClosedTest(unittest.TestCase):
    """A missing input triggers the rule that reads it."""

    def test_missing_concentration_triggers_its_rule(self):
        names, cands, cov = clustered_market(candidates=1)
        decision = evaluate_portfolio(
            HELD, {"C0": proposal("C0", 0.05)}, names, cov, stress=clean_stress()
        )
        self.assertIn("sector_concentration_breach", decision["warning_rule_ids"])

    def test_missing_stress_triggers_its_rule(self):
        names, cands, cov = clustered_market(candidates=1)
        decision = evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05)},
            names,
            cov,
            concentration=clean_concentration(),
        )
        self.assertIn("stress_diversification_collapse", decision["warning_rule_ids"])

    def test_missing_covariance_triggers_the_volatility_veto(self):
        decision = evaluate_portfolio(HELD, {"C0": proposal("C0", 0.05)})
        self.assertIn("set_volatility_exceeds_budget", decision["veto_rule_ids"])
        self.assertEqual(
            decision["verdict"],
            PORTFOLIO_NO_TRADE,
            "an unmeasurable joint volatility effect was treated as a safe "
            "one",
        )

    def test_unevaluated_stress_triggers_its_rule(self):
        names, cands, cov = clustered_market(candidates=1)
        decision = evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05)},
            names,
            cov,
            concentration=clean_concentration(),
            stress={"status": "NOT_EVALUATED", "reason": "too few sessions"},
        )
        self.assertIn("stress_diversification_collapse", decision["warning_rule_ids"])

    def test_the_config_declares_fail_closed(self):
        self.assertTrue(PORTFOLIO_FAIL_CLOSED)


class VerdictTest(unittest.TestCase):
    def test_a_clean_set_proceeds(self):
        names, cands, cov = clustered_market(candidates=2)
        decision = evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05), "C1": proposal("C1", 0.05)},
            names,
            cov,
            concentration=clean_concentration(),
            stress=clean_stress(),
        )
        self.assertEqual(decision["verdict"], PORTFOLIO_PROCEED)
        self.assertEqual(decision_problems(decision), [])
        self.assertFalse(decision["veto"])

    def test_warnings_alone_do_not_refuse(self):
        """Counting warnings into a veto would invent a threshold."""
        names, cands, cov = clustered_market(candidates=2)
        decision = evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05), "C1": proposal("C1", 0.05)},
            names,
            cov,
        )
        self.assertTrue(decision["warning_rule_ids"])
        self.assertFalse(decision["veto_rule_ids"])
        self.assertEqual(decision["verdict"], PORTFOLIO_REDUCED)

    def test_no_surviving_candidate_is_a_veto(self):
        names, cands, cov = clustered_market(candidates=2)
        decision = evaluate_portfolio(
            HELD,
            {"C0": {"ticker": "C0", "verdict": "NO_TRADE", "adjusted_size": None}},
            names,
            cov,
            concentration=clean_concentration(),
            stress=clean_stress(),
        )
        self.assertIn("no_candidate_survived", decision["veto_rule_ids"])
        self.assertEqual(decision["verdict"], PORTFOLIO_NO_TRADE)

    def test_a_refusal_always_names_its_rule(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        decision = evaluate_portfolio(HELD, proposals, names, cov)
        self.assertTrue(decision["veto_rule_ids"])
        self.assertTrue(decision["reason"].strip())

    def test_every_declared_rule_is_evaluated(self):
        names, cands, cov = clustered_market(candidates=2)
        decision = evaluate_portfolio(
            HELD, {"C0": proposal("C0", 0.05)}, names, cov
        )
        evaluated = {r["rule_id"] for r in decision["rules"]}
        self.assertEqual(evaluated, set(PORTFOLIO_RULES))

    def test_every_rule_carries_its_measurement(self):
        names, cands, cov = clustered_market(candidates=2)
        decision = evaluate_portfolio(
            HELD, {"C0": proposal("C0", 0.05)}, names, cov
        )
        for rule in decision["rules"]:
            self.assertTrue(
                rule["measurement"].strip(),
                f"{rule['rule_id']} carries no measurement — a rule that "
                f"cannot say what it enforces is an assertion",
            )
            self.assertTrue(rule["source"].strip())


class AcceptedSizesTest(unittest.TestCase):
    def test_r4s_adjusted_size_wins_over_r2s(self):
        sizes = accepted_sizes({"C0": {"size": 0.20, "adjusted_size": 0.05}})
        self.assertAlmostEqual(sizes["C0"], 0.05, places=9)

    def test_a_candidate_with_no_size_is_absent_not_zero(self):
        sizes = accepted_sizes({"C0": {"verdict": "NO_TRADE", "adjusted_size": None}})
        self.assertNotIn(
            "C0",
            sizes,
            "a refused candidate appeared at size 0.0, which is a position "
            "rather than an absence",
        )

    def test_a_zero_size_is_not_accepted(self):
        self.assertEqual(accepted_sizes({"C0": {"size": 0.0}}), {})

    def test_a_non_mapping_proposal_is_refused(self):
        with self.assertRaises(PortfolioDecisionError):
            accepted_sizes({"C0": "buy it"})


class SetVolatilityTest(unittest.TestCase):
    def test_an_empty_set_is_none_not_zero(self):
        names, cands, cov = clustered_market(candidates=1)
        self.assertIsNone(
            set_volatility_increase(HELD, {}, names, cov),
            "an empty set reported 0.0, which reads as 'no change' rather "
            "than 'nothing to measure'",
        )

    def test_it_measures_the_whole_set_not_one_member(self):
        names, cands, cov = clustered_market()
        one = set_volatility_increase(HELD, {"C0": 0.20}, names, cov)
        many = set_volatility_increase(
            HELD, {c: 0.20 for c in cands}, names, cov
        )
        self.assertGreater(
            many,
            one,
            "the set effect equalled a single member's effect — the joint "
            "measurement is not happening",
        )


class ProblemDetectionTest(unittest.TestCase):
    """decision_problems must be able to fail."""

    def _decision(self):
        names, cands, cov = clustered_market(candidates=2)
        return evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05)},
            names,
            cov,
            concentration=clean_concentration(),
            stress=clean_stress(),
        )

    def test_a_veto_that_does_not_refuse_is_caught(self):
        decision = self._decision()
        decision["veto"] = True
        decision["veto_rule_ids"] = ["total_size_exceeds_budget"]
        self.assertTrue(
            decision_problems(decision),
            "a fired veto alongside a PROCEED verdict passed the contract "
            "check",
        )

    def test_a_refusal_with_no_named_rule_is_caught(self):
        decision = self._decision()
        decision["verdict"] = PORTFOLIO_NO_TRADE
        self.assertTrue(decision_problems(decision))

    def test_a_report_that_cannot_block_is_caught(self):
        decision = self._decision()
        decision["blocks_trades"] = False
        self.assertTrue(
            decision_problems(decision),
            "R7 claiming it cannot block trades passed — every other module "
            "deferred its refusal here",
        )

    def test_a_report_that_is_not_fail_closed_is_caught(self):
        decision = self._decision()
        decision["fail_closed"] = False
        self.assertTrue(decision_problems(decision))

    def test_a_missing_rule_is_caught(self):
        decision = self._decision()
        decision["rules"] = decision["rules"][:-1]
        self.assertTrue(decision_problems(decision))

    def test_a_severity_disagreeing_with_policy_is_caught(self):
        decision = self._decision()
        decision["rules"][0]["severity"] = PORTFOLIO_SEVERITY_WARN
        self.assertTrue(decision_problems(decision))

    def test_a_rule_with_no_measurement_is_caught(self):
        decision = self._decision()
        decision["rules"][0]["measurement"] = "  "
        self.assertTrue(decision_problems(decision))

    def test_an_unknown_verdict_is_caught(self):
        decision = self._decision()
        decision["verdict"] = "PROBABLY"
        self.assertTrue(decision_problems(decision))

    def test_a_mismatched_total_is_caught(self):
        decision = self._decision()
        decision["total_size"] = 0.99
        self.assertTrue(decision_problems(decision))

    def test_a_proceed_with_nothing_accepted_is_caught(self):
        decision = self._decision()
        decision["accepted"] = {}
        self.assertTrue(decision_problems(decision))

    def test_a_non_mapping_portfolio_is_refused(self):
        with self.assertRaises(PortfolioDecisionError):
            evaluate_portfolio(["D1"], {})

    def test_a_non_mapping_proposal_set_is_refused(self):
        with self.assertRaises(PortfolioDecisionError):
            evaluate_portfolio(HELD, [proposal("C0", 0.05)])


class RenderTest(unittest.TestCase):
    def test_the_verdict_and_every_rule_are_rendered(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        decision = evaluate_portfolio(HELD, proposals, names, cov)
        text = "\n".join(render_decision(decision))
        self.assertIn(PORTFOLIO_NO_TRADE, text)
        for rule_id in PORTFOLIO_RULES:
            self.assertIn(rule_id, text)

    def test_triggered_rules_are_marked(self):
        names, cands, cov = clustered_market()
        proposals = {t: size_position(HELD, t, names, cov) for t in cands}
        decision = evaluate_portfolio(HELD, proposals, names, cov)
        self.assertIn("TRIGGERED", "\n".join(render_decision(decision)))


if __name__ == "__main__":
    unittest.main()
