"""X10 honest gate report — what the evidence actually supports.

"If evidence is insufficient, report FORECASTING RELEASE: NOT APPROVED with the
reason and the next action. 'Gate not met' is a valid, successful engineering
outcome."

**MEASURED across X1–X9 on the shipped data: six of nine gates could not run at
all.** One pass-like verdict exists (X9 FROZEN), and it says only that the tree
is frozen — not that anything in it works.

**The trap this closes.** Only TWO gates carry an explicit failure verdict, so a
reading of "no FAIL means ship it" would approve **seven of nine (78%)** on a
system where two thirds of the evidence does not exist. NOT_EVALUATED is counted
as a BLOCKER here, never as a pass.

**Five of the six blockers are one defect: the training fold is a thin record.**
It carries predictions, actuals, metrics and row counts, and none of the regime
labels, event ids, sources, absolute fold indices or per-feature-set scores the
gates need. X6 is the same defect at run level — all 8 runs train the single
`20d` horizon, so there is no horizon spread to compare. Only X2's blocker
differs in kind: it needs a holdout split, which X8 measured is not currently
constructible on 406 rows.

So the report names the common prerequisite AND each gate's own blocker. Naming
one fix would understate the work; naming six independent fixes would overstate
it by 5x.

**A negative report is this module working, not failing.** `evaluate_release`
returns NOT_APPROVED as an ordinary, correct result.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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
    HONEST_GATE_VERSION,
    RELEASE_APPROVED,
    RELEASE_GATES,
    RELEASE_NOT_APPROVED,
)


class HonestGateError(ValueError):
    """Raised when a release report request is structurally invalid."""


# The verdicts each gate reports that constitute a GENUINE PASS. Enumerated
# POSITIVELY: a negative list ("anything that is not a failure") is exactly the
# rule that would approve 7 of 9 gates, because most non-passes are not explicit
# failures.
PASSING_VERDICTS: frozenset[str] = frozenset(
    {
        "APPROVED",          # X1 out-of-sample validation
        "CALIBRATED",        # X2 calibration
        "ROBUST",            # X3 regime, X4 event, X6 temporal robustness
        "INCREMENTAL",       # X5 feature ablation
        "SURVIVES_CORRECTION",  # X7 multiple testing
        "SEALED",            # X8 sealed holdout
        "FROZEN",            # X9 release snapshot
    }
)

# The verdicts that mean "this could not be evaluated". Mapped explicitly because
# each module names its own, and a substring match on "NOT_EVALUATED" would miss
# the ones that phrase it differently.
UNEVALUATED_VERDICTS: frozenset[str] = frozenset(
    {"NOT_EVALUATED", "UNEVALUATED", "ABSENT", "NOT_MEASURED"}
)

# The common prerequisite behind five of the six shipped blockers, and the gates
# it unblocks. MEASURED against the fold keys on disk.
COMMON_BLOCKER = "THIN_FOLD_RECORD"
COMMON_BLOCKER_DETAIL = (
    "The training fold records predictions, actuals, metrics and row counts, and "
    "none of the per-observation context the robustness gates need: no regime "
    "label, no event id, no source, no absolute fold indices, no per-feature-set "
    "variant. All 8 runs also train the single '20d' horizon, so there is no "
    "horizon spread to compare."
)
COMMON_BLOCKER_GATES: tuple[str, ...] = (
    "X3_regime_robustness",
    "X4_event_robustness",
    "X5_feature_ablation",
    "X6_temporal_robustness",
    "X8_sealed_holdout",
)

# What would settle each gate. Prose, because the fix is an engineering decision
# rather than a value: X10's contract is that a blocker names a next action, not
# that the action is machine-executable.
NEXT_ACTIONS: dict[str, str] = {
    "X1_oos_validation": (
        "Establish a directional edge that survives the sampling band on unseen "
        "data. The current best is inside the band, so the honest reading is "
        "'no edge demonstrated', not 'a small edge'."
    ),
    "X2_calibration": (
        "Score calibration on a HOLDOUT split, not in-sample. Blocked behind the "
        "same geometry shortfall X8 measured: the cheapest legal walk-forward "
        "layout needs 432 rows against the 406 available."
    ),
    "X3_regime_robustness": (
        "Record the regime label for each observation in the training fold, so "
        "per-regime performance can be compared at all."
    ),
    "X4_event_robustness": (
        "Record per-observation event ids and sources in the training fold, so a "
        "leave-one-out test has something to leave out."
    ),
    "X5_feature_ablation": (
        "Train the same geometry with each feature family removed and record the "
        "per-variant scores, so incremental value can be measured rather than "
        "assumed."
    ),
    "X6_temporal_robustness": (
        "Train across the declared horizons (1d/5d/20d/60d/120d/252d) rather "
        "than the single 20d target, so the edge can be checked for horizon "
        "dependence."
    ),
    "X7_multiple_testing": (
        "Either demonstrate an edge strong enough to survive correction over the "
        "observed family of 8, or reduce the family. Also register every trial: "
        "the registry holds 1 distinct trial against 8 trained estimators."
    ),
    "X8_sealed_holdout": (
        "Record the holdout bounds, dataset row count, geometry and absolute "
        "fold indices on each run, and extend the dataset past 432 rows so a "
        "legal geometry exists at all."
    ),
    "X9_release_snapshot": (
        "Take the snapshot from a clean tree: a digest that matches no "
        "checkout-able commit cannot be reproduced by anyone else."
    ),
}


def classify(verdict: Any) -> str:
    """One gate verdict -> PASSED / BLOCKED / UNEVALUATED.

    PASSES ARE ENUMERATED POSITIVELY. A rule of "not an explicit failure" is the
    one that would approve 7 of 9 gates, because most non-passes do not announce
    themselves as failures.
    """
    text = str(verdict or "").strip().upper()
    if not text:
        return GATE_UNEVALUATED
    if text in PASSING_VERDICTS:
        return GATE_PASSED
    if text in UNEVALUATED_VERDICTS:
        return GATE_UNEVALUATED
    return GATE_BLOCKED


def _gate_record(name: str, result: Mapping[str, Any] | None) -> dict:
    verdict = (result or {}).get("verdict")
    state = classify(verdict)
    record = {
        "gate": name,
        "state": state,
        "verdict": verdict if verdict is not None else None,
        "reason_code": str((result or {}).get("reason_code") or "") or None,
        "reason": str((result or {}).get("reason") or "") or None,
    }
    if state != GATE_PASSED and GATE_REQUIRES_NEXT_ACTION:
        record["next_action"] = NEXT_ACTIONS.get(
            name, "No next action is recorded for this gate."
        )
    return record


def evaluate_release(
    results: Mapping[str, Mapping[str, Any]] | None,
    *,
    gates: Sequence[str] = RELEASE_GATES,
) -> dict:
    """The release verdict, and why.

    A MISSING GATE IS UNEVALUATED, never skipped. Every declared gate appears in
    the report: a gate absent from the results mapping is indistinguishable from
    one that passed unless the report says otherwise, and that is the whole
    failure mode X10 addresses.
    """
    if results is not None and not isinstance(results, Mapping):
        raise HonestGateError("release results must be a mapping of gate -> report")
    supplied = dict(results or {})

    records = [_gate_record(name, supplied.get(name)) for name in gates]
    by_state: dict[str, list[str]] = {state: [] for state in GATE_STATES}
    for record in records:
        by_state[record["state"]].append(record["gate"])

    passed = by_state[GATE_PASSED]
    blocked = by_state[GATE_BLOCKED]
    unevaluated = by_state[GATE_UNEVALUATED]

    # NOT_EVALUATED BLOCKS. The one line the measurement demanded.
    blocking = list(blocked)
    if GATE_UNEVALUATED_BLOCKS_RELEASE:
        blocking = sorted(blocked + unevaluated)

    approved = GATE_REQUIRES_ALL and not blocking and len(passed) == len(gates)
    verdict = RELEASE_APPROVED if approved else RELEASE_NOT_APPROVED

    report = {
        "version": HONEST_GATE_VERSION,
        "verdict": verdict,
        "gates": records,
        "passed": sorted(passed),
        "blocked": sorted(blocked),
        "unevaluated": sorted(unevaluated),
        "blocking": blocking,
        "counts": {
            GATE_PASSED: len(passed),
            GATE_BLOCKED: len(blocked),
            GATE_UNEVALUATED: len(unevaluated),
            "total": len(gates),
        },
        "unevaluated_blocks_release": GATE_UNEVALUATED_BLOCKS_RELEASE,
        "blocks_trades": HONEST_GATE_BLOCKS_TRADES,
        "gate_not_met_is_valid": GATE_NOT_MET_IS_A_VALID_OUTCOME,
    }

    if approved:
        report["reason"] = (
            f"all {len(gates)} release gates reached a genuine pass"
        )
    else:
        parts = []
        if unevaluated:
            parts.append(
                f"{len(unevaluated)} gate(s) could not be evaluated at all "
                f"({', '.join(sorted(unevaluated))})"
            )
        if blocked:
            parts.append(
                f"{len(blocked)} gate(s) ran and did not pass "
                f"({', '.join(sorted(blocked))})"
            )
        if not parts:
            parts.append("no gate reached a genuine pass")
        report["reason"] = (
            "; ".join(parts)
            + ". Absence of evidence is not evidence of safety: an unevaluated "
            "gate blocks the release exactly as a failed one does."
        )

    # THE COMMON PREREQUISITE, where the blockers share one.
    if GATE_REPORTS_COMMON_BLOCKER:
        shared = sorted(set(blocking) & set(COMMON_BLOCKER_GATES))
        if len(shared) >= 2:
            report["common_blocker"] = {
                "code": COMMON_BLOCKER,
                "detail": COMMON_BLOCKER_DETAIL,
                "gates": shared,
                "note": (
                    f"{len(shared)} of {len(blocking)} blocking gates share this "
                    f"prerequisite, so they are not {len(shared)} independent "
                    f"pieces of work"
                ),
            }
    return report


def release_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a release report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    verdict = report.get("verdict")
    if verdict not in HONEST_GATE_VERDICTS:
        problems.append(f"unknown release verdict {verdict!r}")
    if not str(report.get("reason") or "").strip():
        problems.append("no reason given")
    if report.get("blocks_trades"):
        problems.append("X10 reports; the registry promotes")

    records = report.get("gates")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        problems.append("the report records no per-gate states")
        return problems

    named = {str((record or {}).get("gate")) for record in records}
    for gate in RELEASE_GATES:
        if gate not in named:
            problems.append(
                f"{gate}: absent from the report — a gate that is not listed is "
                f"indistinguishable from one that passed"
            )
    for record in records:
        if not isinstance(record, Mapping):
            problems.append("a gate record is not a mapping")
            continue
        gate = str(record.get("gate") or "")
        if record.get("state") not in GATE_STATES:
            problems.append(f"{gate}: state {record.get('state')!r} is not a gate state")
        # THE CONTRACT THAT MAKES A NEGATIVE USEFUL: a non-pass names a next
        # action. "Not approved" with nowhere to go is a verdict nobody can act
        # on.
        if GATE_REQUIRES_NEXT_ACTION and record.get("state") != GATE_PASSED:
            if not str(record.get("next_action") or "").strip():
                problems.append(f"{gate}: blocks the release but names no next action")

    if verdict == RELEASE_APPROVED:
        if report.get("blocking"):
            problems.append(
                f"APPROVED while {report['blocking']} block the release"
            )
        if GATE_UNEVALUATED_BLOCKS_RELEASE and report.get("unevaluated"):
            problems.append(
                f"APPROVED with {report['unevaluated']} unevaluated; absence of "
                f"evidence is not evidence of safety"
            )
        counts = report.get("counts") or {}
        if counts.get(GATE_PASSED) != counts.get("total"):
            problems.append(
                f"APPROVED with {counts.get(GATE_PASSED)} of "
                f"{counts.get('total')} gates passing; every gate must pass"
            )
    if verdict == RELEASE_NOT_APPROVED and not report.get("blocking"):
        problems.append("NOT_APPROVED but nothing is recorded as blocking")
    return problems


def render_release(report: Mapping[str, Any]) -> list[str]:
    """The honest gate report, as the roadmap phrases it."""
    verdict = report.get("verdict")
    lines = [
        f"FORECASTING RELEASE: {'APPROVED' if verdict == RELEASE_APPROVED else 'NOT APPROVED'}",
        "",
    ]
    counts = report.get("counts") or {}
    lines.append(
        f"  gates passed       {counts.get(GATE_PASSED, 0)} of {counts.get('total', 0)}"
    )
    lines.append(f"  blocked           {counts.get(GATE_BLOCKED, 0)}")
    lines.append(
        f"  could not run     {counts.get(GATE_UNEVALUATED, 0)}"
        f"  (blocks the release)"
    )
    lines.append("")
    for record in report.get("gates") or []:
        state = (record or {}).get("state", GATE_UNEVALUATED)
        verdict_text = (record or {}).get("verdict") or "ABSENT"
        lines.append(
            f"  {str((record or {}).get('gate')):24s} {state:12s} {verdict_text}"
        )
    common = report.get("common_blocker")
    if common:
        lines.append("")
        lines.append(f"  COMMON PREREQUISITE: {common.get('code')}")
        lines.append(f"    {common.get('note')}")
    lines.append("")
    lines.append("  NEXT ACTIONS")
    for record in report.get("gates") or []:
        if (record or {}).get("state") != GATE_PASSED:
            lines.append(f"    {(record or {}).get('gate')}")
            lines.append(f"      {(record or {}).get('next_action')}")
    return lines
