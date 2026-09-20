"""M5 promotion-gate tests.

A promotion gate is only a gate if it refuses. The behaviour under test is
that every check can block alone, and that NOT_EVALUATED blocks exactly as a
failure does — absent evidence is not evidence of safety.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import (
    PROMO_CHECK_APPROVAL,
    PROMO_CHECK_DRIFT,
    PROMO_CHECK_MANIFEST,
    PROMO_CHECK_OOS,
    PROMO_CHECK_REGRESSION,
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
    PROMOTION_CHECKS,
    PROMOTION_HISTORY_IMMUTABLE,
    PROMOTION_MAX_VETO_RATE_DECREASE,
    PROMOTION_MAX_VETO_RATE_INCREASE,
    PROMOTION_MIN_DECISIONS_FOR_REGRESSION,
    PROMOTION_OUTCOMES,
    PROMOTION_PSI_FAIL,
    PROMOTION_PSI_WARN,
    PROMOTION_REQUIRED_CHECKS,
    PROMOTION_REQUIRED_MANIFEST_FIELDS,
)
from core.promotion import (
    PromotionError,
    _check,
    approval_check,
    drift_check,
    evaluate_promotion,
    history_immutable_problems,
    manifest_check,
    oos_check,
    promotion_problems,
    regression_check,
    render_checklist,
)

_RNG = np.random.default_rng(7)
REFERENCE = list(_RNG.normal(5.5, 1.2, 400).clip(0, 10))
SAME = list(_RNG.normal(5.5, 1.2, 400).clip(0, 10))
DRIFTED = list(_RNG.normal(7.4, 1.2, 400).clip(0, 10))

CANDIDATE = {
    "model_version": "cand-1", "family": "linear", "feature_set_version": "fs-1",
    "training_data_cutoff": "2026-01-01", "dataset_hash": "d" * 16,
    "code_commit": "abc123", "seed": 42, "artifact_hash": "a" * 16,
    "hyperparameters": {"alpha": 1.0}, "forecast_target": "probability_up",
    "horizon": "20d",
}
OOS = {
    "primary_metric": "directional_accuracy", "candidate_version": "cand-1",
    "candidate_value": 0.61, "incumbent_value": 0.575, "sample": 406,
}


def decisions(count: int, every: int) -> list[dict]:
    return [
        {
            "decision_id": f"d{index}",
            "veto": {"rule_ids": ["position_cap"] if index % every == 0 else []},
        }
        for index in range(count)
    ]


def verdict(**overrides):
    payload = dict(
        incumbent_version="inc-1", oos_comparison=OOS,
        approved_by="reviewer@example.com", approved_at="2026-09-20T12:00:00Z",
        incumbent_decisions=decisions(100, 5),
        candidate_decisions=decisions(100, 5),
        reference_scores=REFERENCE, candidate_scores=SAME,
    )
    candidate = overrides.pop("candidate", CANDIDATE)
    payload.update(overrides)
    return evaluate_promotion(candidate, **payload)


class CleanCandidateTests(unittest.TestCase):
    def test_a_clean_candidate_is_approved(self):
        # A gate that never approves is not a gate, it is a wall.
        result = verdict()
        self.assertTrue(result["approved"], result["blocking"])
        self.assertEqual(result["blocking"], [])

    def test_a_clean_verdict_is_contract_clean(self):
        self.assertEqual(promotion_problems(verdict()), [])

    def test_every_check_is_reported_in_order(self):
        self.assertEqual(list(verdict()["checks"]), list(PROMOTION_CHECKS))

    def test_every_check_is_required(self):
        self.assertEqual(set(PROMOTION_REQUIRED_CHECKS), set(PROMOTION_CHECKS))


class EachCheckBlocksAloneTests(unittest.TestCase):
    """A check that cannot block on its own is decoration."""

    def _blocked_by(self, check, **override):
        result = verdict(**override)
        self.assertFalse(result["approved"], f"{check} did not block")
        self.assertIn(check, result["blocking"])
        self.assertEqual(promotion_problems(result), [])

    def test_an_incomplete_manifest_blocks(self):
        self._blocked_by(
            PROMO_CHECK_MANIFEST, candidate={**CANDIDATE, "dataset_hash": None}
        )

    def test_losing_out_of_sample_blocks(self):
        self._blocked_by(
            PROMO_CHECK_OOS, oos_comparison={**OOS, "candidate_value": 0.40}
        )

    def test_a_quality_regression_blocks(self):
        self._blocked_by(
            PROMO_CHECK_REGRESSION, candidate_decisions=decisions(100, 100)
        )

    def test_score_drift_blocks(self):
        self._blocked_by(PROMO_CHECK_DRIFT, candidate_scores=DRIFTED)

    def test_a_missing_approver_blocks(self):
        self._blocked_by(PROMO_CHECK_APPROVAL, approved_by=None)

    def test_a_missing_approval_time_blocks(self):
        self._blocked_by(PROMO_CHECK_APPROVAL, approved_at=None)


class NotEvaluatedBlocksTests(unittest.TestCase):
    """Absent evidence is not evidence of safety."""

    def test_missing_drift_data_blocks(self):
        result = verdict(reference_scores=None, candidate_scores=None)
        self.assertFalse(result["approved"])
        self.assertEqual(
            result["checks"][PROMO_CHECK_DRIFT]["outcome"], PROMO_NOT_EVALUATED
        )

    def test_too_few_decisions_blocks(self):
        result = verdict(
            incumbent_decisions=decisions(5, 5), candidate_decisions=decisions(5, 5)
        )
        self.assertFalse(result["approved"])
        self.assertEqual(
            result["checks"][PROMO_CHECK_REGRESSION]["outcome"], PROMO_NOT_EVALUATED
        )

    def test_missing_veto_metadata_blocks(self):
        bare = [{"decision_id": f"d{i}"} for i in range(100)]
        result = verdict(incumbent_decisions=bare, candidate_decisions=bare)
        self.assertFalse(result["approved"])
        self.assertEqual(
            result["checks"][PROMO_CHECK_REGRESSION]["outcome"], PROMO_NOT_EVALUATED
        )

    def test_not_evaluated_is_a_declared_outcome(self):
        self.assertIn(PROMO_NOT_EVALUATED, PROMOTION_OUTCOMES)

    def test_a_verdict_approving_a_non_pass_check_is_reported(self):
        result = verdict()
        result["checks"][PROMO_CHECK_DRIFT]["outcome"] = PROMO_NOT_EVALUATED
        self.assertTrue(any("found nothing" in p for p in promotion_problems(result)))


class RegressionDirectionTests(unittest.TestCase):
    """A veto rate that FALLS is a regression too."""

    def test_a_collapsed_veto_rate_fails(self):
        result = regression_check(decisions(100, 5), decisions(100, 100))
        self.assertEqual(result["outcome"], PROMO_FAIL)
        self.assertIn("FELL", result["reason"])

    def test_a_spiked_veto_rate_fails(self):
        self.assertEqual(
            regression_check(decisions(100, 20), decisions(100, 2))["outcome"],
            PROMO_FAIL,
        )

    def test_a_steady_veto_rate_passes(self):
        self.assertEqual(
            regression_check(decisions(100, 5), decisions(100, 5))["outcome"],
            PROMO_PASS,
        )

    def test_both_tolerances_exist(self):
        self.assertGreater(PROMOTION_MAX_VETO_RATE_DECREASE, 0)
        self.assertGreater(PROMOTION_MAX_VETO_RATE_INCREASE, 0)

    def test_the_deltas_are_reported(self):
        result = regression_check(decisions(100, 5), decisions(100, 5))
        self.assertIn("deltas", result)


class DriftTests(unittest.TestCase):
    def test_an_unshifted_distribution_passes(self):
        result = drift_check(REFERENCE, SAME)
        self.assertEqual(result["outcome"], PROMO_PASS)
        self.assertLess(result["psi"], PROMOTION_PSI_FAIL)

    def test_a_shifted_distribution_fails(self):
        result = drift_check(REFERENCE, DRIFTED)
        self.assertEqual(result["outcome"], PROMO_FAIL)
        self.assertGreaterEqual(result["psi"], PROMOTION_PSI_FAIL)

    def test_no_data_is_not_evaluated(self):
        self.assertEqual(drift_check(None, None)["outcome"], PROMO_NOT_EVALUATED)
        self.assertEqual(drift_check([], [])["outcome"], PROMO_NOT_EVALUATED)

    def test_the_thresholds_ascend(self):
        self.assertLess(0.0, PROMOTION_PSI_WARN)
        self.assertLess(PROMOTION_PSI_WARN, PROMOTION_PSI_FAIL)


class ManifestTests(unittest.TestCase):
    def test_a_complete_manifest_passes(self):
        self.assertEqual(manifest_check(CANDIDATE)["outcome"], PROMO_PASS)

    def test_every_required_field_is_enforced(self):
        for field in PROMOTION_REQUIRED_MANIFEST_FIELDS:
            result = manifest_check({**CANDIDATE, field: None})
            self.assertEqual(result["outcome"], PROMO_FAIL, field)
            self.assertIn(field, result["missing"], field)

    def test_no_candidate_fails(self):
        self.assertEqual(manifest_check(None)["outcome"], PROMO_FAIL)

    def test_it_reads_objects_as_well_as_dicts(self):
        from types import SimpleNamespace

        self.assertEqual(
            manifest_check(SimpleNamespace(**CANDIDATE))["outcome"], PROMO_PASS
        )

    def test_the_manifest_check_runs_first(self):
        # A candidate that cannot be reproduced should be rejected before
        # anyone evaluates its metrics.
        self.assertEqual(PROMOTION_CHECKS[0], PROMO_CHECK_MANIFEST)


class HistoryImmutabilityTests(unittest.TestCase):
    def setUp(self):
        self.before = [
            {"decision_id": "d1", "model_version": "inc-1", "score": 7.2},
            {"decision_id": "d2", "model_version": "inc-1", "score": 5.1},
        ]

    def test_an_unchanged_history_is_clean(self):
        self.assertEqual(
            history_immutable_problems(self.before, list(self.before)), []
        )

    def test_a_rewritten_model_version_is_caught(self):
        rewritten = [{**self.before[0], "model_version": "cand-1"}, self.before[1]]
        problems = history_immutable_problems(self.before, rewritten)
        self.assertTrue(any("model_version changed" in p for p in problems))

    def test_a_rewritten_score_is_caught(self):
        rewritten = [{**self.before[0], "score": 9.9}, self.before[1]]
        self.assertTrue(history_immutable_problems(self.before, rewritten))

    def test_a_vanished_decision_is_caught(self):
        problems = history_immutable_problems(self.before, self.before[:1])
        self.assertTrue(any("disappeared" in p for p in problems))

    def test_new_decisions_are_allowed(self):
        # Promotion is append-only, not frozen.
        extended = self.before + [
            {"decision_id": "d3", "model_version": "cand-1", "score": 6.0}
        ]
        self.assertEqual(history_immutable_problems(self.before, extended), [])

    def test_immutability_is_declared(self):
        self.assertTrue(PROMOTION_HISTORY_IMMUTABLE)


class CheckRowTests(unittest.TestCase):
    def test_a_non_pass_check_must_state_a_reason(self):
        with self.assertRaises(PromotionError):
            _check("x", PROMO_FAIL, "")

    def test_a_pass_check_needs_no_reason(self):
        self.assertEqual(_check("x", PROMO_PASS, "")["outcome"], PROMO_PASS)

    def test_composed_checks_match_the_registry(self):
        # The checklist calls the registry's own functions, so the script and
        # ModelRegistry.promote cannot judge a candidate differently.
        from core.model_registry import approver_problems, oos_comparison_problems

        self.assertEqual(
            approval_check(None, "t")["outcome"],
            PROMO_FAIL if approver_problems(None) else PROMO_PASS,
        )
        self.assertEqual(
            oos_check(None, "cand-1", "inc-1")["outcome"],
            PROMO_FAIL if oos_comparison_problems(None, "cand-1", "inc-1") else PROMO_PASS,
        )


class RenderTests(unittest.TestCase):
    def test_render_covers_every_check_in_order(self):
        rows = render_checklist(verdict())
        self.assertEqual([r["check"] for r in rows], list(PROMOTION_CHECKS))

    def test_render_marks_blocking_checks(self):
        rows = render_checklist(verdict(approved_by=None))
        blocking = [r for r in rows if r["blocking"]]
        self.assertEqual([r["check"] for r in blocking], [PROMO_CHECK_APPROVAL])

    def test_render_never_leaves_a_failure_blank(self):
        for row in render_checklist(verdict(approved_by=None)):
            if row["outcome"] != PROMO_PASS:
                self.assertTrue(row["reason"], row["check"])


if __name__ == "__main__":
    unittest.main()
