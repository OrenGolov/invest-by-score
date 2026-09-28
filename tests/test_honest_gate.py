"""X10 honest-gate-report tests.

The behaviour under test is that NOT_EVALUATED BLOCKS A RELEASE. MEASURED, only
2 of 9 gates carry an explicit failure verdict, so a rule of "no FAIL means ship
it" would approve 7 of 9 on a system where 6 gates could not run at all.
"""

from __future__ import annotations

import unittest

from core.config import (
    GATE_BLOCKED,
    GATE_NOT_MET_IS_A_VALID_OUTCOME,
    GATE_PASSED,
    GATE_REPORTS_COMMON_BLOCKER,
    GATE_REQUIRES_ALL,
    GATE_REQUIRES_NEXT_ACTION,
    GATE_STATES,
    GATE_UNEVALUATED,
    GATE_UNEVALUATED_BLOCKS_RELEASE,
    HONEST_GATE_BLOCKS_TRADES,
    HONEST_GATE_VERDICTS,
    RELEASE_APPROVED,
    RELEASE_GATES,
    RELEASE_NOT_APPROVED,
)
from core.honest_gate import (
    COMMON_BLOCKER,
    COMMON_BLOCKER_GATES,
    NEXT_ACTIONS,
    PASSING_VERDICTS,
    UNEVALUATED_VERDICTS,
    HonestGateError,
    classify,
    evaluate_release,
    release_problems,
    render_release,
)

ALL_PASSING = {
    "X1_oos_validation": {"verdict": "APPROVED"},
    "X2_calibration": {"verdict": "CALIBRATED"},
    "X3_regime_robustness": {"verdict": "ROBUST"},
    "X4_event_robustness": {"verdict": "ROBUST"},
    "X5_feature_ablation": {"verdict": "INCREMENTAL"},
    "X6_temporal_robustness": {"verdict": "ROBUST"},
    "X7_multiple_testing": {"verdict": "SURVIVES_CORRECTION"},
    "X8_sealed_holdout": {"verdict": "SEALED"},
    "X9_release_snapshot": {"verdict": "FROZEN"},
}

# The verdicts the nine gates actually report on the shipped data.
SHIPPED = {
    "X1_oos_validation": {"verdict": "NOT_APPROVED"},
    "X2_calibration": {"verdict": "NOT_EVALUATED", "reason_code": "IN_SAMPLE_ONLY"},
    "X3_regime_robustness": {"verdict": "NOT_EVALUATED", "reason_code": "NO_REGIME_LABELS"},
    "X4_event_robustness": {
        "verdict": "NOT_EVALUATED",
        "reason_code": "NO_EVENT_ATTRIBUTION",
    },
    "X5_feature_ablation": {"verdict": "NOT_EVALUATED"},
    "X6_temporal_robustness": {"verdict": "NOT_EVALUATED", "reason_code": "HORIZONS_MISSING"},
    "X7_multiple_testing": {"verdict": "FAILS_CORRECTION"},
    "X8_sealed_holdout": {"verdict": "NOT_EVALUATED", "reason_code": "NO_RECORDED_BOUNDS"},
    "X9_release_snapshot": {"verdict": "DIRTY"},
}


class UnevaluatedBlocksTheReleaseTests(unittest.TestCase):
    """The deciding measurement, and the single most important rule in X10."""

    def test_the_contract_makes_unevaluated_blocking(self):
        self.assertTrue(GATE_UNEVALUATED_BLOCKS_RELEASE)

    def test_one_unevaluated_gate_sinks_a_release(self):
        results = dict(ALL_PASSING)
        results["X3_regime_robustness"] = {"verdict": "NOT_EVALUATED"}
        report = evaluate_release(results)
        self.assertEqual(report["verdict"], RELEASE_NOT_APPROVED)
        self.assertEqual(report["blocking"], ["X3_regime_robustness"])

    def test_the_shipped_data_is_not_approved(self):
        report = evaluate_release(SHIPPED)
        self.assertEqual(report["verdict"], RELEASE_NOT_APPROVED)

    def test_no_gate_passes_on_the_shipped_data(self):
        report = evaluate_release(SHIPPED)
        self.assertEqual(report["counts"][GATE_PASSED], 0)
        self.assertEqual(report["counts"]["total"], len(RELEASE_GATES))

    def test_six_shipped_gates_could_not_run(self):
        report = evaluate_release(SHIPPED)
        self.assertEqual(report["counts"][GATE_UNEVALUATED], 6)

    def test_only_two_shipped_gates_carry_an_explicit_failure(self):
        # THE TRAP: a 'no FAIL means pass' rule would approve the other seven.
        explicit = [
            gate
            for gate, result in SHIPPED.items()
            if str(result["verdict"]) in ("NOT_APPROVED", "FAILS_CORRECTION")
        ]
        self.assertEqual(len(explicit), 2)
        would_approve = [gate for gate in SHIPPED if gate not in explicit]
        self.assertEqual(len(would_approve), 7)

    def test_the_reason_says_absence_is_not_evidence(self):
        report = evaluate_release(SHIPPED)
        self.assertIn("Absence of evidence", report["reason"])


class ApprovalIsReachableTests(unittest.TestCase):
    """Otherwise NOT_APPROVED proves nothing."""

    def test_all_nine_passing_is_approved(self):
        report = evaluate_release(ALL_PASSING)
        self.assertEqual(report["verdict"], RELEASE_APPROVED)
        self.assertEqual(release_problems(report), [])

    def test_an_approved_report_has_nothing_blocking(self):
        report = evaluate_release(ALL_PASSING)
        self.assertEqual(report["blocking"], [])
        self.assertEqual(report["counts"][GATE_PASSED], len(RELEASE_GATES))

    def test_every_gate_must_pass(self):
        self.assertTrue(GATE_REQUIRES_ALL)
        for gate in RELEASE_GATES:
            with self.subTest(gate=gate):
                results = dict(ALL_PASSING)
                results[gate] = {"verdict": "NOT_EVALUATED"}
                self.assertEqual(
                    evaluate_release(results)["verdict"], RELEASE_NOT_APPROVED
                )


class AMissingGateIsNotASkippedGateTests(unittest.TestCase):
    """A gate absent from the results is indistinguishable from a pass."""

    def test_a_gate_absent_from_the_results_blocks(self):
        results = {k: v for k, v in ALL_PASSING.items() if k != "X8_sealed_holdout"}
        report = evaluate_release(results)
        self.assertEqual(report["verdict"], RELEASE_NOT_APPROVED)
        self.assertIn("X8_sealed_holdout", report["blocking"])

    def test_every_declared_gate_appears_in_the_report(self):
        report = evaluate_release({})
        named = {record["gate"] for record in report["gates"]}
        self.assertEqual(named, set(RELEASE_GATES))

    def test_no_results_at_all_is_not_approved(self):
        for empty in ({}, None):
            report = evaluate_release(empty)
            self.assertEqual(report["verdict"], RELEASE_NOT_APPROVED)
            self.assertEqual(report["counts"][GATE_UNEVALUATED], len(RELEASE_GATES))

    def test_a_non_mapping_result_set_is_refused(self):
        with self.assertRaises(HonestGateError):
            evaluate_release(["not", "a", "mapping"])


class ClassificationTests(unittest.TestCase):
    """Passes are enumerated POSITIVELY."""

    def test_each_declared_passing_verdict_classifies_as_passed(self):
        for verdict in PASSING_VERDICTS:
            with self.subTest(verdict=verdict):
                self.assertEqual(classify(verdict), GATE_PASSED)

    def test_each_unevaluated_verdict_classifies_as_unevaluated(self):
        for verdict in UNEVALUATED_VERDICTS:
            with self.subTest(verdict=verdict):
                self.assertEqual(classify(verdict), GATE_UNEVALUATED)

    def test_an_unknown_verdict_is_blocked_not_passed(self):
        # Fail-closed: a verdict this module has never seen must not be read as
        # a pass just because it is not a recognised failure.
        for verdict in ("SOMETHING_NEW", "PROVISIONAL", "PARTIAL"):
            with self.subTest(verdict=verdict):
                self.assertEqual(classify(verdict), GATE_BLOCKED)

    def test_a_missing_verdict_is_unevaluated(self):
        for verdict in (None, "", "   "):
            with self.subTest(verdict=verdict):
                self.assertEqual(classify(verdict), GATE_UNEVALUATED)

    def test_classification_ignores_case(self):
        self.assertEqual(classify("robust"), GATE_PASSED)

    def test_no_passing_verdict_is_also_an_unevaluated_one(self):
        self.assertEqual(PASSING_VERDICTS & UNEVALUATED_VERDICTS, frozenset())


class NextActionTests(unittest.TestCase):
    """A negative verdict must be actionable."""

    def test_the_contract_requires_a_next_action(self):
        self.assertTrue(GATE_REQUIRES_NEXT_ACTION)

    def test_every_blocking_gate_names_a_next_action(self):
        report = evaluate_release(SHIPPED)
        for record in report["gates"]:
            if record["state"] != GATE_PASSED:
                with self.subTest(gate=record["gate"]):
                    self.assertTrue(str(record["next_action"]).strip())

    def test_every_declared_gate_has_an_action_on_file(self):
        for gate in RELEASE_GATES:
            with self.subTest(gate=gate):
                self.assertIn(gate, NEXT_ACTIONS)
                self.assertTrue(NEXT_ACTIONS[gate].strip())

    def test_a_passing_gate_needs_no_next_action(self):
        report = evaluate_release(ALL_PASSING)
        for record in report["gates"]:
            self.assertIsNone(record.get("next_action"))


class CommonBlockerTests(unittest.TestCase):
    """MEASURED: five of six blockers are one thin-fold defect."""

    def test_the_contract_requires_the_common_blocker(self):
        self.assertTrue(GATE_REPORTS_COMMON_BLOCKER)

    def test_the_shipped_report_names_the_common_prerequisite(self):
        report = evaluate_release(SHIPPED)
        self.assertEqual(report["common_blocker"]["code"], COMMON_BLOCKER)
        self.assertEqual(
            sorted(report["common_blocker"]["gates"]), sorted(COMMON_BLOCKER_GATES)
        )

    def test_one_shared_gate_is_not_a_common_blocker(self):
        # Naming a "common" prerequisite behind a single gate would inflate one
        # fix into a theme.
        results = dict(ALL_PASSING)
        results["X3_regime_robustness"] = {"verdict": "NOT_EVALUATED"}
        self.assertIsNone(evaluate_release(results).get("common_blocker"))

    def test_two_shared_gates_are_a_common_blocker(self):
        results = dict(ALL_PASSING)
        results["X3_regime_robustness"] = {"verdict": "NOT_EVALUATED"}
        results["X4_event_robustness"] = {"verdict": "NOT_EVALUATED"}
        report = evaluate_release(results)
        self.assertEqual(len(report["common_blocker"]["gates"]), 2)

    def test_an_approved_release_names_no_common_blocker(self):
        self.assertIsNone(evaluate_release(ALL_PASSING).get("common_blocker"))


class GateNotMetIsAValidOutcomeTests(unittest.TestCase):
    def test_the_contract_says_so(self):
        self.assertTrue(GATE_NOT_MET_IS_A_VALID_OUTCOME)

    def test_a_not_approved_report_is_contract_clean(self):
        # A correct negative is this module WORKING. It must not look like a bug.
        report = evaluate_release(SHIPPED)
        self.assertEqual(release_problems(report), [])

    def test_an_empty_report_is_contract_clean(self):
        self.assertEqual(release_problems(evaluate_release({})), [])


class ContractTests(unittest.TestCase):
    def test_not_approved_is_the_weakest_verdict(self):
        self.assertEqual(HONEST_GATE_VERDICTS[0], RELEASE_NOT_APPROVED)

    def test_there_are_only_two_verdicts(self):
        # Every gradation is a place for a negative to be read as a positive.
        self.assertEqual(len(HONEST_GATE_VERDICTS), 2)

    def test_unevaluated_is_the_weakest_gate_state(self):
        self.assertEqual(GATE_STATES[0], GATE_UNEVALUATED)

    def test_blocked_and_unevaluated_stay_distinct(self):
        self.assertNotEqual(GATE_BLOCKED, GATE_UNEVALUATED)

    def test_nine_gates_are_declared(self):
        self.assertEqual(len(RELEASE_GATES), 9)
        self.assertEqual(len(set(RELEASE_GATES)), 9)

    def test_x10_reports_and_does_not_block(self):
        self.assertFalse(HONEST_GATE_BLOCKS_TRADES)
        self.assertFalse(evaluate_release(SHIPPED)["blocks_trades"])

    def test_a_forged_approval_with_blockers_is_caught(self):
        forged = {
            "verdict": RELEASE_APPROVED,
            "reason": "trust me",
            "blocking": ["X1_oos_validation"],
            "gates": [
                {"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES
            ],
        }
        self.assertTrue(release_problems(forged))

    def test_a_forged_approval_with_unevaluated_gates_is_caught(self):
        forged = {
            "verdict": RELEASE_APPROVED,
            "reason": "r",
            "blocking": [],
            "unevaluated": ["X3_regime_robustness"],
            "counts": {GATE_PASSED: 9, "total": 9},
            "gates": [
                {"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES
            ],
        }
        self.assertTrue(release_problems(forged))

    def test_a_forged_approval_with_a_short_pass_count_is_caught(self):
        forged = {
            "verdict": RELEASE_APPROVED,
            "reason": "r",
            "blocking": [],
            "counts": {GATE_PASSED: 8, "total": 9},
            "gates": [
                {"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES
            ],
        }
        self.assertTrue(release_problems(forged))

    def test_a_report_omitting_a_gate_is_caught(self):
        forged = {
            "verdict": RELEASE_NOT_APPROVED,
            "reason": "r",
            "blocking": ["X1_oos_validation"],
            "gates": [{"gate": "X1_oos_validation", "state": GATE_BLOCKED,
                       "next_action": "do something"}],
        }
        self.assertTrue(release_problems(forged))

    def test_a_blocking_gate_without_a_next_action_is_caught(self):
        forged = {
            "verdict": RELEASE_NOT_APPROVED,
            "reason": "r",
            "blocking": ["X1_oos_validation"],
            "gates": [
                {"gate": gate, "state": GATE_BLOCKED} for gate in RELEASE_GATES
            ],
        }
        self.assertTrue(release_problems(forged))

    def test_a_not_approved_report_with_no_blockers_is_caught(self):
        forged = {
            "verdict": RELEASE_NOT_APPROVED,
            "reason": "r",
            "blocking": [],
            "gates": [
                {"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES
            ],
        }
        self.assertTrue(release_problems(forged))

    def test_a_blocking_report_is_caught(self):
        self.assertTrue(
            release_problems(
                {
                    "verdict": RELEASE_NOT_APPROVED,
                    "reason": "r",
                    "blocks_trades": True,
                    "blocking": ["X1_oos_validation"],
                    "gates": [],
                }
            )
        )

    def test_a_non_mapping_report_is_caught(self):
        self.assertTrue(release_problems("not a mapping"))

    def test_a_report_without_gate_records_is_caught(self):
        self.assertTrue(
            release_problems({"verdict": RELEASE_NOT_APPROVED, "reason": "r"})
        )


class RenderTests(unittest.TestCase):
    def test_the_headline_is_the_roadmap_phrasing(self):
        text = "\n".join(render_release(evaluate_release(SHIPPED)))
        self.assertIn("FORECASTING RELEASE: NOT APPROVED", text)

    def test_an_approved_release_says_approved(self):
        text = "\n".join(render_release(evaluate_release(ALL_PASSING)))
        self.assertIn("FORECASTING RELEASE: APPROVED", text)
        self.assertNotIn("NOT APPROVED", text)

    def test_the_unevaluated_count_says_it_blocks(self):
        text = "\n".join(render_release(evaluate_release(SHIPPED)))
        self.assertIn("could not run", text)
        self.assertIn("blocks the release", text)

    def test_every_gate_gets_a_line(self):
        lines = render_release(evaluate_release(SHIPPED))
        for gate in RELEASE_GATES:
            self.assertTrue(any(gate in line for line in lines))

    def test_the_next_actions_are_rendered(self):
        text = "\n".join(render_release(evaluate_release(SHIPPED)))
        self.assertIn("NEXT ACTIONS", text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_release(evaluate_release(SHIPPED))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedGateTests(unittest.TestCase):
    """Against the real X1-X9 modules, so the report cannot drift from them."""

    def setUp(self):
        import json
        import pathlib

        from core.training import load_training_runs

        self.runs = load_training_runs()
        if not self.runs:
            self.skipTest("no training runs on disk")
        datasets = pathlib.Path("data/training_datasets.jsonl")
        if not datasets.exists():
            self.skipTest("no dataset ledger on disk")
        self.rows = json.loads(
            datasets.read_text(encoding="utf-8").splitlines()[0]
        )["row_count"]

    def test_the_real_holdout_gate_still_blocks(self):
        from core.sealed_holdout import evaluate_sealed_holdout

        verdict = evaluate_sealed_holdout(self.runs, dataset_rows=self.rows)["verdict"]
        self.assertEqual(classify(verdict), GATE_UNEVALUATED)

    def test_the_real_event_gate_still_blocks(self):
        from core.event_robustness import (
            EVENT_ROBUSTNESS_AXIS_EVENT,
            EVENT_ROBUSTNESS_AXIS_SOURCE,
            attribution_of,
            evaluate_event_robustness,
        )

        fold = self.runs[0]["folds"][0]
        verdict = evaluate_event_robustness(
            fold["predictions"],
            fold["actuals"],
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT),
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE),
        )["verdict"]
        self.assertEqual(classify(verdict), GATE_UNEVALUATED)

    def test_the_folds_carry_none_of_what_the_gates_need(self):
        # The thin-fold defect, measured rather than asserted.
        fold = self.runs[0]["folds"][0]
        for key in ("regimes", "events", "sources", "validation", "ablation"):
            with self.subTest(key=key):
                self.assertNotIn(key, fold)

    def test_all_runs_train_a_single_horizon(self):
        # Which is why X6 has no horizon spread to compare.
        self.assertEqual(len({run.get("target_horizon") for run in self.runs}), 1)


if __name__ == "__main__":
    unittest.main()
