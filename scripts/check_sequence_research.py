"""CI drift gate for the C6 sequence-model research contract.

C6 says "evaluate only after baseline validation" and "complexity must earn its
place through out-of-sample evidence". This gate proves, on every push, that the
bar cannot be talked around:

1. a measured incumbent actually EXISTS in the trial ledger — not just the
   machinery to produce one (the state C6 inherited: a fully tested baseline
   suite and zero recorded trials);
2. a registered-but-unfinished trial is NOT evidence, and neither is a
   completed trial that never reported its pre-registered metric;
3. every architecture C6 names is declared, with a rationale;
4. declaring is not implementing — an unimplemented architecture blocks
   readiness rather than being quietly counted as done;
5. no deep-learning framework is imported: that is a dependency decision;
6. ONE bar, not two — a sequence candidate is judged by the same
   `compare_runs` rules as everything else, so a tie, a thin win, and a
   one-lucky-fold win are all refused;
7. comparing against a missing incumbent raises rather than manufacturing a
   verdict.

Synthetic apart from reading the real trial ledger, so it needs no network.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import core.sequence_research as research_module  # noqa: E402
from core.baseline_suite import BaselineSuiteError  # noqa: E402
from core.config import (  # noqa: E402
    BASELINE_PROMOTION_MARGIN,
    SEQUENCE_ARCH_REGISTRY,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_PRIMARY_METRIC,
)
from core.sequence_research import (  # noqa: E402
    SequenceResearchError,
    baseline_trials,
    evaluate_candidate,
    implemented_architectures,
    readiness_problems,
    research_readiness,
)
from core.training import FoldResult  # noqa: E402

_FRAMEWORKS = ("import torch", "import tensorflow", "from torch", "from tensorflow")


def _folds(values):
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
    def __init__(self, estimator, accuracy, fold_values=None):
        self.estimator = estimator
        self.metrics = {SEQUENCE_PRIMARY_METRIC: accuracy, "observations": 120}
        self.folds = _folds(fold_values if fold_values is not None else [accuracy] * 5)
        self.dataset_hash = "same-dataset"
        self.label_version = "outcome-label-v1"
        self.feature_set_hash = "same-features"
        self.target_horizon = "20d"
        self.seed = 7


def main() -> int:
    failures: list[str] = []

    # 1. a measured incumbent exists in the REAL ledger
    live = research_readiness()
    if not live["baseline_established"] or live["incumbent_metric"] is None:
        failures.append(
            "no measured incumbent in the trial ledger — C6 cannot evaluate a "
            "sequence model against machinery that has never been run. "
            "Run scripts/establish_baseline.py"
        )
    for problem in readiness_problems(live):
        failures.append(f"live readiness problem: {problem}")

    # 2. what does and does not count as evidence
    registered_only = [{"trial_id": "t1", "status": "registered", "metrics": {}}]
    if baseline_trials(registered_only):
        failures.append("a registered-but-unfinished trial was counted as evidence")
    wrong_metric = [{"trial_id": "t1", "status": "completed", "metrics": {"rmse": 0.2}}]
    if baseline_trials(wrong_metric):
        failures.append(
            "a completed trial without the pre-registered primary metric was counted "
            "as evidence — a trial is judged on the metric it declared"
        )
    if research_readiness([])["baseline_established"]:
        failures.append("an empty ledger reported an established baseline")

    # 3 + 4. architectures declared, with rationale, and honestly statused
    for name in ("temporal_convolution", "lstm", "gru",
                 "time_series_transformer", "temporal_fusion"):
        if name not in SEQUENCE_ARCH_REGISTRY:
            failures.append(f"C6 architecture {name!r} is not declared")
    for name, entry in SEQUENCE_ARCH_REGISTRY.items():
        if not entry.get("rationale"):
            failures.append(f"architecture {name!r} is declared with no rationale")
    if set(SEQUENCE_ARCH_REGISTRY) != set(SEQUENCE_ARCHITECTURES):
        failures.append("the architecture registry drifted from the declared tuple")
    if implemented_architectures() and live["ready_to_evaluate"] is False and not live["blockers"]:
        failures.append("readiness is inconsistent about implemented architectures")

    # 5. no framework smuggled in
    source = inspect.getsource(research_module)
    for framework in _FRAMEWORKS:
        if framework in source:
            failures.append(
                f"core.sequence_research contains {framework!r} — adding a "
                f"deep-learning framework is a dependency decision, not an "
                f"implementation detail"
            )

    # 6. ONE bar: the same gate everything else faces
    incumbent = _Run("momentum", 0.575)
    clear_win = _Run("lstm", 0.575 + BASELINE_PROMOTION_MARGIN + 0.01)
    if not evaluate_candidate(clear_win, incumbent).promoted:
        failures.append("a candidate clearing the margin on every fold was refused")
    if evaluate_candidate(_Run("lstm", 0.575), incumbent).promoted:
        failures.append("a candidate that merely tied the incumbent was promoted")
    if evaluate_candidate(_Run("lstm", 0.576), incumbent).promoted:
        failures.append("a win of 0.001 was promoted — that margin is noise")
    lucky = _Run("lstm", 0.60, fold_values=[0.95, 0.50, 0.50, 0.50, 0.55])
    if evaluate_candidate(lucky, incumbent).promoted:
        failures.append("a win driven by one lucky fold was promoted")

    # 7. no incumbent, no verdict
    try:
        evaluate_candidate(_Run("lstm", 0.99), None)
        failures.append("comparing against a missing incumbent produced a verdict")
    except SequenceResearchError:
        pass

    mismatched = _Run("lstm", 0.99)
    mismatched.dataset_hash = "a-different-dataset"
    try:
        evaluate_candidate(mismatched, incumbent)
        failures.append("a candidate trained on a different dataset was compared anyway")
    except BaselineSuiteError:
        pass

    if failures:
        print("C6 sequence-research gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C6 sequence-research gate OK:")
    print(f"  measured incumbent present: {SEQUENCE_PRIMARY_METRIC}={live['incumbent_metric']} "
          f"from {live['completed_baseline_trials']} completed trial(s).")
    print(f"  {len(SEQUENCE_ARCHITECTURES)} architectures declared with rationale; "
          f"{len(implemented_architectures())} implemented (declaring is not implementing).")
    print("  registered-only and wrong-metric trials are not counted as evidence.")
    print("  one bar: tie, thin win and one-lucky-fold win are all refused.")
    print("  no deep-learning framework imported; a missing incumbent raises.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
