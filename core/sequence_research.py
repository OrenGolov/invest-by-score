"""Sequence model research (Sprint C6) — the gate complexity must pass.

C6's own words: *evaluate only after baseline validation*, and *complexity must
earn its place through out-of-sample evidence*. This module is that gate.

**What it deliberately does not do.** It trains no neural network and imports no
deep-learning framework. When this was written the repository had a fully tested
baseline suite and ZERO recorded trials — the machinery to measure, and nothing
measured. An LSTM built in that state could not have been evaluated against
anything, which is precisely the unfalsifiable result the master context forbids
("do not build a complex neural network before proving simple baselines").

`scripts/establish_baseline.py` fixed the precondition: a real M2 dataset, the
M4 suite trained on it, and a completed M3 trial. So the bar now exists and has
a number. This module enforces it.

**One bar, not two.** A sequence candidate is judged by
`core.baseline_suite.compare_runs` — the same margin and fold-consistency rules
every other candidate faces. Defining a friendlier threshold for the
fashionable architecture is exactly how complexity slips past its evidence, so
there is no second rule here.

**Readiness is measured, not assumed.** `research_readiness()` reads the trial
registry and reports whether a measured incumbent actually exists. If it does
not, `evaluate_candidate` REFUSES rather than comparing against a placeholder.

The architectures C6 names (temporal convolution, LSTM/GRU, transformer,
temporal fusion) are declared in `SEQUENCE_ARCH_REGISTRY` with a rationale and a
status. All are `proposed`: declaring them records the plan without pretending
the work is done.
"""

from __future__ import annotations

import logging

from core.baseline_suite import compare_runs
from core.config import (
    SEQUENCE_ARCH_REGISTRY,
    SEQUENCE_ARCH_STATUS_IMPLEMENTED,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_MIN_BASELINE_TRIALS,
    SEQUENCE_PRIMARY_METRIC,
    SEQUENCE_RESEARCH_VERSION,
    TRIAL_STATUS_COMPLETED,
)
from core.trial_registry import load_trial_records

LOGGER = logging.getLogger("core.sequence_research")


class SequenceResearchError(ValueError):
    """Raised when a sequence-research request violates the C6 contract."""


def declared_architectures() -> dict[str, dict[str, str]]:
    """The architectures C6 may evaluate, with status and rationale."""
    return {name: dict(SEQUENCE_ARCH_REGISTRY[name]) for name in SEQUENCE_ARCHITECTURES}


def implemented_architectures() -> list[str]:
    """Architectures that actually exist. Empty is the honest answer today."""
    return sorted(
        name for name, entry in SEQUENCE_ARCH_REGISTRY.items()
        if entry.get("status") == SEQUENCE_ARCH_STATUS_IMPLEMENTED
    )


def baseline_trials(records: list[dict] | None = None) -> list[dict]:
    """Completed trials carrying the primary metric.

    A registered-but-unfinished trial is not evidence, and a completed trial
    that never reported the pre-registered metric cannot be compared against.
    """
    rows = records if records is not None else load_trial_records()
    latest: dict[str, dict] = {}
    for row in rows:
        trial_id = row.get("trial_id")
        if trial_id:
            latest[trial_id] = row
    return [
        row for row in latest.values()
        if row.get("status") == TRIAL_STATUS_COMPLETED
        and isinstance((row.get("metrics") or {}).get(SEQUENCE_PRIMARY_METRIC), (int, float))
    ]


def incumbent_metric(records: list[dict] | None = None) -> float | None:
    """The best measured incumbent score, or None when nothing is measured."""
    trials = baseline_trials(records)
    if not trials:
        return None
    return max(float(trial["metrics"][SEQUENCE_PRIMARY_METRIC]) for trial in trials)


def research_readiness(records: list[dict] | None = None) -> dict:
    """Whether C6 may proceed, and exactly what is missing if not.

    The point of reporting this rather than assuming it: "we have a baseline"
    was true of the MACHINERY long before it was true of any NUMBER.
    """
    trials = baseline_trials(records)
    incumbent = incumbent_metric(records)
    blockers: list[str] = []
    if len(trials) < SEQUENCE_MIN_BASELINE_TRIALS:
        blockers.append(
            f"{len(trials)} completed baseline trial(s) carrying "
            f"{SEQUENCE_PRIMARY_METRIC!r}; {SEQUENCE_MIN_BASELINE_TRIALS} required — "
            f"a sequence model cannot be judged against an unmeasured incumbent"
        )
    if not implemented_architectures():
        blockers.append(
            "no sequence architecture is implemented; all are 'proposed'. "
            "Implementing one requires a deep-learning framework, which is a "
            "dependency decision, not an implementation detail"
        )
    return {
        "research_version": SEQUENCE_RESEARCH_VERSION,
        "primary_metric": SEQUENCE_PRIMARY_METRIC,
        "completed_baseline_trials": len(trials),
        "incumbent_metric": incumbent,
        "declared_architectures": sorted(SEQUENCE_ARCHITECTURES),
        "implemented_architectures": implemented_architectures(),
        "baseline_established": incumbent is not None,
        "ready_to_evaluate": not blockers,
        "blockers": blockers,
    }


def evaluate_candidate(candidate_run, incumbent_run, metric: str = SEQUENCE_PRIMARY_METRIC):
    """Judge a sequence candidate against the measured incumbent.

    Delegates to `compare_runs` deliberately: the sequence model faces the same
    margin and fold-consistency bar as every other candidate. A separate,
    friendlier rule for the fashionable architecture is how complexity escapes
    its evidence.
    """
    if candidate_run is None or incumbent_run is None:
        raise SequenceResearchError(
            "both a candidate and a measured incumbent are required — comparing "
            "against a placeholder would manufacture a result"
        )
    return compare_runs(candidate_run, incumbent_run, metric=metric)


def readiness_problems(readiness: dict) -> list[str]:
    """Validate a readiness report against the C6 contract."""
    problems: list[str] = []
    if not isinstance(readiness, dict):
        return ["readiness must be a dict"]
    for field in ("research_version", "primary_metric", "declared_architectures"):
        if not readiness.get(field):
            problems.append(f"readiness field {field!r} missing/empty")
    if readiness.get("ready_to_evaluate") and readiness.get("blockers"):
        problems.append("readiness claims ready while carrying blockers")
    if readiness.get("baseline_established") and readiness.get("incumbent_metric") is None:
        problems.append("a baseline is claimed established with no incumbent metric")
    if not readiness.get("baseline_established") and readiness.get("incumbent_metric") is not None:
        problems.append("an incumbent metric exists but the baseline is reported unestablished")
    declared = set(readiness.get("declared_architectures") or [])
    if declared != set(SEQUENCE_ARCHITECTURES):
        problems.append("declared architectures drifted from the registry")
    implemented = set(readiness.get("implemented_architectures") or [])
    if not implemented <= declared:
        problems.append("an implemented architecture is not declared")
    return problems
