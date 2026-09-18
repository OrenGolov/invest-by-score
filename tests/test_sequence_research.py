"""Sequence model research tests (Sprint C6).

C6's precondition is the whole sprint: *evaluate only after baseline
validation*, and *complexity must earn its place through out-of-sample
evidence*.

What these tests pin is that the gate cannot be talked around:

**No measured incumbent, no evaluation.** Before `scripts/establish_baseline.py`
ran, the repository had a fully tested baseline suite and ZERO recorded trials —
the machinery to measure, and nothing measured. `research_readiness` must tell
that truth rather than reporting readiness because the code exists.

**One bar, not two.** A sequence candidate is judged by the same
`compare_runs` rules every other candidate faces. A separate, friendlier
threshold for the fashionable architecture is how complexity escapes its
evidence.

**Declaring an architecture is not implementing it.** All five are `proposed`;
saying so is the honest state, and a readiness report that claimed otherwise
would be a defect.
"""

from __future__ import annotations

import unittest

from core.baseline_suite import BaselineSuiteError
from core.training import FoldResult
from core.config import (
    BASELINE_PROMOTION_MARGIN,
    SEQUENCE_ARCH_REGISTRY,
    SEQUENCE_ARCH_STATUS_PROPOSED,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_PRIMARY_METRIC,
)
from core.sequence_research import (
    SequenceResearchError,
    baseline_trials,
    declared_architectures,
    evaluate_candidate,
    implemented_architectures,
    incumbent_metric,
    readiness_problems,
    research_readiness,
)


def _folds(values):
    """Real FoldResult objects — the comparison pairs folds by fold_id, so a
    dict stand-in would not exercise the actual contract."""
    return [
        FoldResult(
            fold_id=index,
            train_rows=200,
            validation_rows=40,
            train_end_time=f"2026-0{index + 1}-01",
            validation_start_time=f"2026-0{index + 1}-02",
            metrics={SEQUENCE_PRIMARY_METRIC: value},
        )
        for index, value in enumerate(values)
    ]


class _Run:
    """A minimal TrainingRun stand-in for comparison tests."""

    def __init__(self, estimator, accuracy, fold_values=None, **extra):
        self.estimator = estimator
        self.metrics = {SEQUENCE_PRIMARY_METRIC: accuracy, "observations": 120}
        self.metrics.update(extra)
        self.folds = _folds(fold_values if fold_values is not None else [accuracy] * 5)
        self.dataset_hash = "same-dataset"
        self.label_version = "outcome-label-v1"
        self.feature_set_hash = "same-features"
        self.target_horizon = "20d"
        self.seed = 7


def _trial(trial_id, status="completed", accuracy=0.575):
    metrics = {SEQUENCE_PRIMARY_METRIC: accuracy} if accuracy is not None else {}
    return {"trial_id": trial_id, "status": status, "metrics": metrics}


class BaselinePreconditionTests(unittest.TestCase):
    """The state C6 inherited: machinery everywhere, measurement nowhere."""

    def test_no_trials_means_no_incumbent(self):
        self.assertIsNone(incumbent_metric([]))

    def test_no_trials_blocks_evaluation(self):
        readiness = research_readiness([])
        self.assertFalse(readiness["baseline_established"])
        self.assertFalse(readiness["ready_to_evaluate"])
        self.assertTrue(
            any("unmeasured incumbent" in blocker for blocker in readiness["blockers"])
        )

    def test_a_registered_but_unfinished_trial_is_not_evidence(self):
        """Pre-registration records the hypothesis; it proves nothing yet."""
        records = [_trial("t1", status="registered", accuracy=None)]
        self.assertEqual(baseline_trials(records), [])
        self.assertIsNone(incumbent_metric(records))

    def test_a_completed_trial_without_the_primary_metric_is_not_evidence(self):
        """A trial is judged on the metric it declared, not one that looks good."""
        records = [{"trial_id": "t1", "status": "completed", "metrics": {"rmse": 0.17}}]
        self.assertEqual(baseline_trials(records), [])

    def test_a_completed_trial_establishes_the_incumbent(self):
        readiness = research_readiness([_trial("t1", accuracy=0.575)])
        self.assertTrue(readiness["baseline_established"])
        self.assertAlmostEqual(readiness["incumbent_metric"], 0.575)

    def test_the_best_completed_trial_sets_the_bar(self):
        records = [_trial("t1", accuracy=0.55), _trial("t2", accuracy=0.61)]
        self.assertAlmostEqual(incumbent_metric(records), 0.61)

    def test_the_latest_state_of_a_trial_wins(self):
        """The ledger is append-only: registered then completed is ONE trial."""
        records = [
            _trial("t1", status="registered", accuracy=None),
            _trial("t1", status="completed", accuracy=0.575),
        ]
        self.assertEqual(len(baseline_trials(records)), 1)


class ArchitectureDeclarationTests(unittest.TestCase):
    def test_every_c6_architecture_is_declared(self):
        declared = declared_architectures()
        for name in ("temporal_convolution", "lstm", "gru",
                     "time_series_transformer", "temporal_fusion"):
            with self.subTest(architecture=name):
                self.assertIn(name, declared)

    def test_every_architecture_justifies_its_presence(self):
        """An architecture listed without a rationale is cargo cult."""
        for name, entry in declared_architectures().items():
            with self.subTest(architecture=name):
                self.assertTrue(entry.get("rationale"))

    def test_nothing_is_implemented_yet_and_that_is_reported(self):
        """Declaring is not implementing. Saying so is the honest state."""
        self.assertEqual(implemented_architectures(), [])
        for name in SEQUENCE_ARCHITECTURES:
            with self.subTest(architecture=name):
                self.assertEqual(
                    SEQUENCE_ARCH_REGISTRY[name]["status"], SEQUENCE_ARCH_STATUS_PROPOSED
                )

    def test_no_deep_learning_framework_is_imported(self):
        """A framework is a dependency decision, not an implementation detail."""
        import inspect

        import core.sequence_research as module

        source = inspect.getsource(module)
        for framework in ("import torch", "import tensorflow", "from torch", "from tensorflow"):
            with self.subTest(framework=framework):
                self.assertNotIn(framework, source)

    def test_an_unimplemented_architecture_blocks_readiness(self):
        readiness = research_readiness([_trial("t1", accuracy=0.575)])
        self.assertFalse(readiness["ready_to_evaluate"])
        self.assertTrue(any("not implemented" in b or "proposed" in b for b in readiness["blockers"]))


class OneBarTests(unittest.TestCase):
    """The sequence model faces the same gate as everything else."""

    def test_a_candidate_that_clears_the_margin_is_promoted(self):
        incumbent = _Run("momentum", 0.575)
        candidate = _Run("lstm", 0.575 + BASELINE_PROMOTION_MARGIN + 0.01)
        verdict = evaluate_candidate(candidate, incumbent)
        self.assertTrue(verdict.promoted)

    def test_a_candidate_that_merely_ties_is_refused(self):
        incumbent = _Run("momentum", 0.575)
        verdict = evaluate_candidate(_Run("lstm", 0.575), incumbent)
        self.assertFalse(verdict.promoted)

    def test_a_thin_win_is_refused_as_noise(self):
        """Beating by 0.001 is not evidence of an edge."""
        incumbent = _Run("momentum", 0.575)
        verdict = evaluate_candidate(_Run("lstm", 0.576), incumbent)
        self.assertFalse(verdict.promoted)
        self.assertTrue(any("noise" in reason for reason in verdict.reasons))

    def test_a_win_driven_by_one_lucky_fold_is_refused(self):
        incumbent = _Run("momentum", 0.575, fold_values=[0.575] * 5)
        candidate = _Run("lstm", 0.60, fold_values=[0.95, 0.50, 0.50, 0.50, 0.55])
        verdict = evaluate_candidate(candidate, incumbent)
        self.assertFalse(verdict.promoted)
        self.assertTrue(any("consistent" in reason for reason in verdict.reasons))

    def test_comparing_against_nothing_is_refused(self):
        """A placeholder incumbent would manufacture a result."""
        with self.assertRaises(SequenceResearchError):
            evaluate_candidate(_Run("lstm", 0.9), None)

    def test_a_mismatched_dataset_cannot_be_compared(self):
        """Like-for-like: a different dataset is a different question."""
        incumbent = _Run("momentum", 0.575)
        candidate = _Run("lstm", 0.9)
        candidate.dataset_hash = "a-different-dataset"
        with self.assertRaises(BaselineSuiteError):
            evaluate_candidate(candidate, incumbent)


class ReadinessContractTests(unittest.TestCase):
    def test_a_healthy_readiness_report_is_contract_clean(self):
        self.assertEqual(readiness_problems(research_readiness([_trial("t1")])), [])

    def test_claiming_ready_while_blocked_is_a_problem(self):
        readiness = research_readiness([_trial("t1")])
        readiness["ready_to_evaluate"] = True
        self.assertTrue(any("carrying blockers" in p for p in readiness_problems(readiness)))

    def test_claiming_a_baseline_with_no_metric_is_a_problem(self):
        readiness = research_readiness([])
        readiness["baseline_established"] = True
        self.assertTrue(any("no incumbent metric" in p for p in readiness_problems(readiness)))

    def test_architecture_drift_is_caught(self):
        readiness = research_readiness([_trial("t1")])
        readiness["declared_architectures"] = ["lstm"]
        self.assertTrue(any("drifted" in p for p in readiness_problems(readiness)))

    def test_an_undeclared_implementation_is_caught(self):
        readiness = research_readiness([_trial("t1")])
        readiness["implemented_architectures"] = ["mystery_net"]
        self.assertTrue(any("not declared" in p for p in readiness_problems(readiness)))


class LiveRegistryTests(unittest.TestCase):
    """Against the real ledger, which now carries a measured baseline."""

    def test_the_repository_has_a_measured_incumbent(self):
        """establish_baseline.py ran; C6's precondition is satisfied."""
        readiness = research_readiness()
        self.assertTrue(
            readiness["baseline_established"],
            "no measured incumbent — run scripts/establish_baseline.py",
        )
        self.assertIsNotNone(readiness["incumbent_metric"])

    def test_the_live_readiness_report_is_contract_clean(self):
        self.assertEqual(readiness_problems(research_readiness()), [])
