"""L5 controlled-learning tests.

The behaviour under test is that the chain is SEQUENCED and the champion is
BOUNDED. Most stages belong to M5/M7 and are delegated, not re-decided here.
"""

from __future__ import annotations

import unittest

from core.config import (
    CHAMPION_MIN_TENURE_DAYS,
    CHAMPION_TENURE_ALLOWS_ROLLBACK,
    CHAMPION_TENURE_MEASURED_FROM,
    L5_AUTO_PROMOTE,
    L5_OWNED_STAGES,
    L5_STAGE_CHAMPION,
    L5_STAGE_GATE,
    L5_STAGE_SHADOW,
    L5_STAGE_TENURE,
    L5_STAGES,
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
    SHADOW_MIN_OBSERVATIONS,
)
from core.controlled_learning import (
    ControlledLearningError,
    chain_problems,
    candidate_check,
    data_check,
    delegated_stages,
    evaluate_chain,
    owned_stages,
    render_chain,
    shadow_check,
    tenure_check,
    tenure_days,
)


class Candidate:
    model_version = "cand-1"


CLEAN_VERDICT = {
    "approved": True,
    "blocking": [],
    "checks": {
        "drift_clean": {"check": "drift_clean", "outcome": PROMO_PASS, "reason": "clean"},
        "oos_beats_incumbent": {
            "check": "oos_beats_incumbent", "outcome": PROMO_PASS, "reason": "wins"
        },
        "human_approval": {
            "check": "human_approval", "outcome": PROMO_PASS, "reason": "approved"
        },
    },
}


def chain(**overrides):
    kwargs = {
        "candidate": Candidate(),
        "new_observations": 500,
        "shadow_observations": 150,
        "promotion_verdict": CLEAN_VERDICT,
        "incumbent_crowned_at": "2026-01-01T00:00:00Z",
        "now": "2026-09-21T00:00:00Z",
    }
    kwargs.update(overrides)
    return evaluate_chain(**kwargs)


class TenureTests(unittest.TestCase):
    """'Never auto-replace the production champion daily.'"""

    def test_a_champion_crowned_minutes_ago_cannot_be_replaced(self):
        result = tenure_check("2026-09-21T09:00:00Z", "2026-09-21T09:15:00Z")
        self.assertEqual(result["outcome"], PROMO_FAIL)

    def test_a_day_old_champion_cannot_be_replaced(self):
        result = tenure_check("2026-09-20T00:00:00Z", "2026-09-21T00:00:00Z")
        self.assertEqual(result["outcome"], PROMO_FAIL)

    def test_a_seasoned_champion_can_be_replaced(self):
        result = tenure_check("2026-01-01T00:00:00Z", "2026-09-21T00:00:00Z")
        self.assertEqual(result["outcome"], PROMO_PASS)

    def test_the_boundary_is_inclusive(self):
        result = tenure_check("2026-08-22T00:00:00Z", "2026-09-21T00:00:00Z")
        self.assertEqual(result["outcome"], PROMO_PASS)

    def test_one_day_short_of_the_bound_fails(self):
        result = tenure_check("2026-08-23T00:00:00Z", "2026-09-21T00:00:00Z")
        self.assertEqual(result["outcome"], PROMO_FAIL)
        self.assertIn("remaining_days", result)

    def test_no_incumbent_is_not_a_violation(self):
        # The first champion has nothing to displace; refusing would make the
        # system unable to ever start.
        self.assertEqual(
            tenure_check(None, "2026-09-21T00:00:00Z")["outcome"], PROMO_PASS
        )

    def test_a_rollback_is_never_blocked(self):
        result = tenure_check(
            "2026-09-21T09:00:00Z", "2026-09-21T09:15:00Z", is_rollback=True
        )
        self.assertEqual(result["outcome"], PROMO_PASS)
        self.assertTrue(result["rollback"])

    def test_a_future_crowning_cannot_buy_tenure(self):
        result = tenure_check("2026-12-01T00:00:00Z", "2026-09-21T00:00:00Z")
        self.assertNotEqual(result["outcome"], PROMO_PASS)

    def test_an_unreadable_timestamp_is_not_evaluated(self):
        result = tenure_check("not-a-timestamp", "2026-09-21T00:00:00Z")
        self.assertEqual(result["outcome"], PROMO_NOT_EVALUATED)

    def test_naive_and_aware_timestamps_agree(self):
        aware = tenure_days("2026-01-01T00:00:00Z", "2026-01-31T00:00:00Z")
        naive = tenure_days("2026-01-01T00:00:00", "2026-01-31T00:00:00")
        self.assertEqual(aware, naive)

    def test_the_bound_is_a_real_bound(self):
        self.assertGreaterEqual(CHAMPION_MIN_TENURE_DAYS, 2)
        self.assertLessEqual(CHAMPION_MIN_TENURE_DAYS, 180)

    def test_tenure_is_measured_from_the_incumbent(self):
        self.assertEqual(CHAMPION_TENURE_MEASURED_FROM, "incumbent_crowned_at")

    def test_rollback_is_permitted_by_contract(self):
        self.assertTrue(CHAMPION_TENURE_ALLOWS_ROLLBACK)


class ChainTests(unittest.TestCase):
    def test_every_stage_reaches_a_verdict_in_order(self):
        report = chain()
        self.assertEqual(
            [item["stage"] for item in report["stages"]], list(L5_STAGES)
        )

    def test_a_clean_chain_may_promote(self):
        self.assertTrue(chain()["may_promote"])

    def test_identical_evidence_with_a_fresh_incumbent_is_blocked(self):
        report = chain(incumbent_crowned_at="2026-09-20T00:00:00Z")
        self.assertFalse(report["may_promote"])
        self.assertEqual(report["blocked_at"], L5_STAGE_TENURE)

    def test_a_missing_checklist_blocks_rather_than_passes(self):
        report = chain(promotion_verdict=None)
        self.assertFalse(report["may_promote"])

    def test_no_new_data_blocks(self):
        self.assertFalse(chain(new_observations=0)["may_promote"])

    def test_a_missing_candidate_blocks(self):
        self.assertFalse(chain(candidate=None)["may_promote"])

    def test_a_refused_gate_blocks(self):
        refused = {
            "approved": False,
            "blocking": ["drift_clean"],
            "checks": {
                "drift_clean": {
                    "check": "drift_clean", "outcome": PROMO_FAIL, "reason": "psi"
                },
                "oos_beats_incumbent": {
                    "check": "oos_beats_incumbent",
                    "outcome": PROMO_PASS,
                    "reason": "wins",
                },
                "human_approval": {
                    "check": "human_approval",
                    "outcome": PROMO_PASS,
                    "reason": "ok",
                },
            },
        }
        report = chain(promotion_verdict=refused)
        self.assertFalse(report["may_promote"])

    def test_the_report_is_contract_clean(self):
        self.assertEqual(chain_problems(chain()), [])

    def test_render_returns_one_line_per_stage(self):
        self.assertEqual(len(render_chain(chain())), len(L5_STAGES))

    def test_a_blocked_chain_names_its_first_blocker(self):
        report = chain(new_observations=0)
        self.assertEqual(report["blocked_at"], "new_data")
        self.assertTrue(report["blocking_reasons"])


class DelegationTests(unittest.TestCase):
    """W5: L5 sequences the chain, it does not re-decide it."""

    def test_l5_owns_only_the_tenure_stage(self):
        self.assertEqual(tuple(owned_stages()), tuple(L5_OWNED_STAGES))
        self.assertIn(L5_STAGE_TENURE, owned_stages())

    def test_l5_does_not_own_the_promotion_gate(self):
        self.assertNotIn(L5_STAGE_GATE, owned_stages())

    def test_the_gate_is_delegated_to_core_promotion(self):
        self.assertIn("core.promotion", delegated_stages()[L5_STAGE_GATE])

    def test_the_shadow_floor_comes_from_m7(self):
        self.assertEqual(
            shadow_check(SHADOW_MIN_OBSERVATIONS - 1)["outcome"], PROMO_FAIL
        )
        self.assertEqual(
            shadow_check(SHADOW_MIN_OBSERVATIONS)["outcome"], PROMO_PASS
        )

    def test_a_delegated_outcome_is_relayed_verbatim(self):
        report = chain()
        drift = next(i for i in report["stages"] if i["stage"] == "drift_testing")
        self.assertEqual(drift["outcome"], PROMO_PASS)
        self.assertEqual(drift["delegated_to"], "drift_clean")


class NotEvaluatedBlocksTests(unittest.TestCase):
    """Absence of evidence is not evidence of safety."""

    def test_an_unreported_shadow_count_blocks(self):
        self.assertEqual(shadow_check(None)["outcome"], PROMO_NOT_EVALUATED)
        self.assertFalse(chain(shadow_observations=None)["may_promote"])

    def test_an_unreported_data_count_blocks(self):
        self.assertEqual(data_check(None)["outcome"], PROMO_NOT_EVALUATED)

    def test_a_missing_delegated_check_is_not_evaluated(self):
        partial = {"approved": True, "blocking": [], "checks": {}}
        report = chain(promotion_verdict=partial)
        drift = next(i for i in report["stages"] if i["stage"] == "drift_testing")
        self.assertEqual(drift["outcome"], PROMO_NOT_EVALUATED)
        self.assertFalse(report["may_promote"])


class AutoPromotionTests(unittest.TestCase):
    def test_auto_promotion_is_off(self):
        self.assertFalse(L5_AUTO_PROMOTE)
        self.assertFalse(chain()["auto_promote"])

    def test_may_promote_is_permission_not_an_act(self):
        self.assertIn("human act", chain()["note"])

    def test_a_valid_stage_call_does_not_raise(self):
        self.assertEqual(candidate_check(Candidate())["outcome"], PROMO_PASS)

    def test_an_unknown_stage_is_refused(self):
        from core.controlled_learning import _stage

        with self.assertRaises(ControlledLearningError):
            _stage("not_a_stage", PROMO_PASS, "x")

    def test_an_unknown_outcome_is_refused(self):
        from core.controlled_learning import _stage

        with self.assertRaises(ControlledLearningError):
            _stage(L5_STAGE_TENURE, "MAYBE", "x")


if __name__ == "__main__":
    unittest.main()
