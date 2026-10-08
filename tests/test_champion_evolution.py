"""L8 champion-evolution tests.

The behaviour under test is that ALL SEVEN conditions are required and that
historical forecasts cannot be rewritten. Three of the seven are L8's own
because nothing else measured them; the rest are relayed from their owners.
"""

from __future__ import annotations

import random
import unittest

from core.config import (
    CHAMPION_EVOLUTION_CONDITIONS,
    CHAMPION_EVOLUTION_CREDIBILITY_SIGMA,
    CHAMPION_EVOLUTION_HISTORY_IMMUTABLE,
    CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO,
    CHAMPION_EVOLUTION_MAX_FP_INCREASE,
    CHAMPION_EVOLUTION_MIN_ACTED_CALLS,
    CHAMPION_EVOLUTION_MIN_OOS_SAMPLE,
    CHAMPION_EVOLUTION_OUTCOMES,
    CHAMPION_EVOLUTION_OWNED,
    CHAMPION_EVOLUTION_REQUIRED,
    EVO_CALIBRATION,
    EVO_FAIL,
    EVO_FALSE_POSITIVE,
    EVO_GOVERNANCE,
    EVO_NOT_EVALUATED,
    EVO_OOS,
    EVO_PASS,
    EVO_REGIME,
    EVO_REPRODUCIBILITY,
    EVO_RISK,
)
from core.champion_evolution import (
    ChampionEvolutionError,
    credibility,
    evaluate_replacement,
    evolution_problems,
    false_positive_condition,
    history_immutability_problems,
    left_tail,
    max_drawdown,
    oos_condition,
    render_evolution,
    risk_condition,
)

OK = {"outcome": EVO_PASS, "reason": "ok"}


def series(count, hit, loss_size, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        if rng.random() < hit:
            out.append(abs(rng.gauss(0.010, 0.004)))
        else:
            out.append(-abs(rng.gauss(0.010 * loss_size, 0.004 * loss_size)))
    return out


def calls(true_positives, false_positives):
    return {"true_positives": true_positives, "false_positives": false_positives}


def replacement(**overrides):
    kwargs = {
        "oos_comparison": {
            "candidate_value": 0.62,
            "incumbent_value": 0.55,
            "sample": 2000,
        },
        "incumbent_returns": series(500, 0.58, 1.0, 1),
        "candidate_returns": series(500, 0.62, 1.0, 2),
        "incumbent_calls": calls(130, 70),
        "candidate_calls": calls(140, 60),
        "calibration": OK,
        "regime": OK,
        "reproducibility": OK,
        "governance": OK,
    }
    kwargs.update(overrides)
    return evaluate_replacement(**kwargs)


class AllSevenRequiredTests(unittest.TestCase):
    def test_seven_conditions_are_declared(self):
        self.assertEqual(len(CHAMPION_EVOLUTION_CONDITIONS), 7)

    def test_every_condition_is_required(self):
        self.assertEqual(
            set(CHAMPION_EVOLUTION_REQUIRED), set(CHAMPION_EVOLUTION_CONDITIONS)
        )

    def test_a_clean_replacement_is_allowed(self):
        self.assertTrue(replacement()["may_replace"])

    def test_the_report_covers_every_condition(self):
        self.assertEqual(
            set(replacement()["outcomes"]), set(CHAMPION_EVOLUTION_CONDITIONS)
        )

    def test_any_single_failure_blocks(self):
        for field in ("calibration", "regime", "reproducibility", "governance"):
            report = replacement(**{field: {"outcome": EVO_FAIL, "reason": "no"}})
            self.assertFalse(report["may_replace"], field)


class CredibilityTests(unittest.TestCase):
    """MEASURED: identical models differ by 5.6 points on 10% of runs at n=250."""

    def test_a_small_sample_improvement_is_not_credible(self):
        report = replacement(
            oos_comparison={
                "candidate_value": 0.60,
                "incumbent_value": 0.55,
                "sample": 250,
            }
        )
        self.assertFalse(report["may_replace"])
        self.assertIn(EVO_OOS, report["blocked_by"])

    def test_a_marginal_improvement_is_not_credible(self):
        self.assertFalse(credibility(0.56, 0.55, 2000)["credible"])

    def test_a_clear_improvement_is_credible(self):
        self.assertTrue(credibility(0.62, 0.55, 2000)["credible"])

    def test_the_bar_scales_with_the_evidence(self):
        # The SAME gap, credible on more data and not on less.
        self.assertTrue(credibility(0.59, 0.55, 5000)["credible"])
        self.assertFalse(credibility(0.59, 0.55, 600)["credible"])

    def test_below_the_sample_floor_nothing_is_credible(self):
        result = credibility(0.99, 0.55, CHAMPION_EVOLUTION_MIN_OOS_SAMPLE - 1)
        self.assertFalse(result["credible"])

    def test_a_missing_comparison_is_not_evaluated(self):
        self.assertEqual(oos_condition(None)["outcome"], EVO_NOT_EVALUATED)

    def test_an_incomplete_comparison_is_not_evaluated(self):
        self.assertEqual(
            oos_condition({"candidate_value": 0.6})["outcome"], EVO_NOT_EVALUATED
        )

    def test_the_bar_is_at_least_the_95_percent_level(self):
        self.assertGreaterEqual(CHAMPION_EVOLUTION_CREDIBILITY_SIGMA, 1.96)


class RiskTests(unittest.TestCase):
    """Accuracy cannot see risk. MEASURED: same hit rate, 28x worse drawdown."""

    def test_risk_is_owned_by_l8(self):
        self.assertIn(EVO_RISK, CHAMPION_EVOLUTION_OWNED)

    def test_the_same_hit_rate_with_bigger_losses_fails(self):
        result = risk_condition(series(500, 0.58, 1.0, 3), series(500, 0.58, 3.0, 4))
        self.assertEqual(result["outcome"], EVO_FAIL)

    def test_better_accuracy_does_not_excuse_worse_risk(self):
        report = replacement(candidate_returns=series(500, 0.62, 3.0, 5))
        self.assertFalse(report["may_replace"])
        self.assertIn(EVO_RISK, report["blocked_by"])

    def test_comparable_risk_passes(self):
        result = risk_condition(series(500, 0.58, 1.0, 6), series(500, 0.60, 1.0, 7))
        self.assertEqual(result["outcome"], EVO_PASS)

    def test_a_missing_series_is_not_evaluated(self):
        self.assertEqual(risk_condition(None, None)["outcome"], EVO_NOT_EVALUATED)
        self.assertEqual(
            risk_condition(series(100, 0.6, 1.0, 8), None)["outcome"],
            EVO_NOT_EVALUATED,
        )

    def test_max_drawdown_is_peak_to_trough(self):
        self.assertAlmostEqual(max_drawdown([0.1, -0.05, -0.05, 0.02]), -0.10)
        self.assertAlmostEqual(max_drawdown([0.1, 0.1]), 0.0)

    def test_left_tail_reports_a_bad_day(self):
        # The 5th percentile of 100 values is the 6th worst, so ONE bad day in
        # 100 is correctly not the tail — six of them are.
        self.assertLess(left_tail([-0.5] * 6 + [0.01] * 94), 0.0)
        self.assertAlmostEqual(left_tail([-0.5] + [0.01] * 99), 0.01)
        self.assertIsNone(left_tail([]))

    def test_a_worse_tail_is_detected_at_equal_drawdown(self):
        # A model can hold its drawdown while making its bad days worse.
        # The worsened days must reach the 5th percentile to count as tail:
        # 10 of 500 is 2%, which is inside it and correctly invisible here.
        mild = [0.02, -0.01] * 250
        harsh = [0.02, -0.01] * 210 + [0.02, -0.05] * 40
        self.assertLess(left_tail(harsh), left_tail(mild))

    def test_the_drawdown_bound_is_a_bound(self):
        self.assertLessEqual(CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO, 2.0)
        self.assertGreater(CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO, 1.0)


class FalsePositiveTests(unittest.TestCase):
    """The acted-on rate, not the veto rate."""

    def test_a_degraded_acted_on_rate_fails(self):
        report = replacement(candidate_calls=calls(90, 110))
        self.assertFalse(report["may_replace"])
        self.assertIn(EVO_FALSE_POSITIVE, report["blocked_by"])

    def test_an_improved_acted_on_rate_passes(self):
        result = false_positive_condition(calls(130, 70), calls(160, 40))
        self.assertEqual(result["outcome"], EVO_PASS)

    def test_too_few_calls_is_not_evaluated(self):
        result = false_positive_condition(calls(5, 2), calls(6, 1))
        self.assertEqual(result["outcome"], EVO_NOT_EVALUATED)

    def test_no_calls_at_all_is_not_evaluated(self):
        self.assertEqual(
            false_positive_condition(calls(0, 0), calls(150, 50))["outcome"],
            EVO_NOT_EVALUATED,
        )

    def test_a_missing_record_is_not_evaluated(self):
        self.assertEqual(
            false_positive_condition(None, calls(150, 50))["outcome"],
            EVO_NOT_EVALUATED,
        )

    def test_the_floor_and_bound_are_real(self):
        self.assertGreaterEqual(CHAMPION_EVOLUTION_MIN_ACTED_CALLS, 100)
        self.assertLessEqual(CHAMPION_EVOLUTION_MAX_FP_INCREASE, 0.10)


class DelegationTests(unittest.TestCase):
    """W5: L8 sequences the seven, it does not re-decide four of them."""

    def test_governance_is_not_owned(self):
        self.assertNotIn(EVO_GOVERNANCE, CHAMPION_EVOLUTION_OWNED)

    def test_a_delegated_reason_is_relayed_verbatim(self):
        report = replacement(
            calibration={"outcome": EVO_FAIL, "reason": "calibration gap 0.31"}
        )
        reasons = [c["reason"] for c in report["conditions"]]
        self.assertIn("calibration gap 0.31", reasons)

    def test_a_missing_delegated_verdict_blocks(self):
        for field in ("calibration", "regime", "reproducibility", "governance"):
            self.assertFalse(replacement(**{field: None})["may_replace"], field)

    def test_an_unknown_delegated_outcome_becomes_not_evaluated(self):
        report = replacement(calibration={"outcome": "MAYBE", "reason": "?"})
        self.assertEqual(report["outcomes"][EVO_CALIBRATION], EVO_NOT_EVALUATED)
        self.assertFalse(report["may_replace"])


class ImmutabilityTests(unittest.TestCase):
    """The one clause that is an invariant, not a threshold."""

    def setUp(self):
        self.before = [
            {"forecast_id": "f1", "value": 0.6, "claim": "POINT", "model_version": "m-1"},
            {"forecast_id": "f2", "value": 0.4, "claim": "POINT", "model_version": "m-1"},
        ]

    def test_an_unchanged_history_is_clean(self):
        self.assertEqual(history_immutability_problems(self.before, self.before), [])

    def test_a_rewritten_value_is_detected(self):
        after = [dict(self.before[0], value=0.9), self.before[1]]
        self.assertTrue(history_immutability_problems(self.before, after))

    def test_a_reattributed_model_is_detected(self):
        after = [dict(self.before[0], model_version="m-2"), self.before[1]]
        self.assertTrue(history_immutability_problems(self.before, after))

    def test_a_deleted_forecast_is_detected(self):
        self.assertTrue(history_immutability_problems(self.before, [self.before[0]]))

    def test_appending_new_forecasts_is_allowed(self):
        after = self.before + [
            {"forecast_id": "f3", "value": 0.7, "claim": "POINT", "model_version": "m-2"}
        ]
        self.assertEqual(history_immutability_problems(self.before, after), [])

    def test_the_invariant_is_declared(self):
        self.assertTrue(CHAMPION_EVOLUTION_HISTORY_IMMUTABLE)


class ReportContractTests(unittest.TestCase):
    def test_a_clean_report_is_contract_clean(self):
        self.assertEqual(evolution_problems(replacement()), [])

    def test_a_forged_report_is_caught(self):
        report = replacement()
        forged = dict(report)
        forged["outcomes"] = dict(report["outcomes"])
        forged["outcomes"][EVO_RISK] = EVO_FAIL
        self.assertTrue(evolution_problems(forged))

    def test_render_returns_one_line_per_condition(self):
        self.assertEqual(
            len(render_evolution(replacement())), len(CHAMPION_EVOLUTION_CONDITIONS)
        )

    def test_not_evaluated_is_not_pass(self):
        self.assertNotEqual(EVO_NOT_EVALUATED, EVO_PASS)
        self.assertIn(EVO_NOT_EVALUATED, CHAMPION_EVOLUTION_OUTCOMES)

    def test_the_report_states_replacement_is_not_automatic(self):
        self.assertIn("human act", replacement()["note"])


if __name__ == "__main__":
    unittest.main()
