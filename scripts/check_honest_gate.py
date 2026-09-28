"""X10 governance gate — the release report must be honest.

Exits 1 when the REPORT is malformed. It does NOT exit 1 because the release is
not approved: "gate not met" is a valid, successful engineering outcome, and
conflating an honest negative with a crash would make the two indistinguishable.

THE DECIDING MEASUREMENT, RECOMPUTED HERE by running X1-X9 against the tracked
ledgers: six of nine gates cannot be evaluated at all, and only two carry an
explicit failure verdict. A rule of "no FAIL means ship it" would approve SEVEN
OF NINE on a system where two thirds of the evidence does not exist.

`data/training_runs.jsonl`, `data/training_datasets.jsonl` and
`data/research_trials.jsonl` are all TRACKED, so this recomputes identically on a
fresh clone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
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
from core.honest_gate import (  # noqa: E402
    COMMON_BLOCKER,
    COMMON_BLOCKER_GATES,
    NEXT_ACTIONS,
    PASSING_VERDICTS,
    UNEVALUATED_VERDICTS,
    classify,
    evaluate_release,
    release_problems,
    render_release,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


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


# --- 1. RUN X1-X9 FOR REAL --------------------------------------------------------

runs = load_training_runs()
check(bool(runs), "no training runs are available; X10 cannot report on anything")

dataset_rows = None
datasets_path = REPO_ROOT / "data" / "training_datasets.jsonl"
if datasets_path.exists():
    lines = [
        line
        for line in datasets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if lines:
        dataset_rows = json.loads(lines[0]).get("row_count")

trials = []
trials_path = REPO_ROOT / "data" / "research_trials.jsonl"
if trials_path.exists():
    trials = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

shipped: dict[str, dict] = {}
if runs:
    fold = runs[0]["folds"][0]

    from core.calibration_gate import evaluate_calibration
    from core.event_robustness import (
        EVENT_ROBUSTNESS_AXIS_EVENT,
        EVENT_ROBUSTNESS_AXIS_SOURCE,
        attribution_of,
        evaluate_event_robustness,
    )
    from core.feature_ablation import evaluate_ablation
    from core.multiple_testing import evaluate_multiple_testing
    from core.oos_validation import validate_suite
    from core.regime_robustness import evaluate_regime_robustness
    from core.release_snapshot import build_release_snapshot
    from core.sealed_holdout import evaluate_sealed_holdout
    from core.temporal_robustness import evaluate_temporal_robustness

    family = {
        run["estimator"]: run["folds"][0]["predictions"]
        for run in runs
        if run["folds"][0].get("predictions")
    }

    shipped = {
        "X1_oos_validation": validate_suite(runs),
        "X2_calibration": evaluate_calibration(fold["predictions"], fold["actuals"]),
        "X3_regime_robustness": evaluate_regime_robustness(
            fold["predictions"], fold["actuals"], None
        ),
        "X4_event_robustness": evaluate_event_robustness(
            fold["predictions"],
            fold["actuals"],
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT),
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE),
        ),
        "X5_feature_ablation": evaluate_ablation(sorted(runs[0]["feature_names"])),
        "X6_temporal_robustness": evaluate_temporal_robustness(
            {runs[0].get("target_horizon", "?"): runs[0].get("metrics", {})}
        ),
        "X7_multiple_testing": evaluate_multiple_testing(
            family, fold["actuals"], runs=runs, trials=trials, permutations=100
        ),
        "X8_sealed_holdout": evaluate_sealed_holdout(runs, dataset_rows=dataset_rows),
        "X9_release_snapshot": build_release_snapshot(
            validation={"X8": {"verdict": "NOT_EVALUATED"}}
        ),
    }

    report = evaluate_release(shipped)

    check(
        report["verdict"] == RELEASE_NOT_APPROVED,
        f"the shipped release reported {report['verdict']}; with 6 of 9 gates "
        f"unevaluable the only honest answer is NOT_APPROVED",
    )
    # X9 IS THE ONE GATE WHOSE VERDICT DEPENDS ON THE CHECKOUT: FROZEN in a
    # clean clone, DIRTY in any working tree someone is editing. A FIRST VERSION
    # OF THIS GATE HARDCODED THE COUNTS MEASURED IN A DIRTY TREE (0 passing, 3
    # explicit non-passes) and failed on a fresh clone, where X9 correctly passes
    # and the numbers are 1 and 2.
    #
    # That is the defect in open item 6 for the third time, so the counts are now
    # derived: the EVIDENCE gates (X1-X8) are asserted, and X9 is excluded from
    # the arithmetic because it measures the tree rather than the model.
    evidence_gates = [gate for gate in RELEASE_GATES if gate != "X9_release_snapshot"]
    evidence = {gate: shipped[gate] for gate in evidence_gates}
    evidence_report = evaluate_release(evidence, gates=evidence_gates)

    check(
        evidence_report["counts"][GATE_PASSED] == 0,
        f"{evidence_report['counts'][GATE_PASSED]} evidence gate(s) now pass "
        f"({evidence_report['passed']}). That is progress, and this gate's "
        f"'nothing passes' claim is stale - the numbers in the X10 config block "
        f"must be revisited",
    )
    check(
        evidence_report["counts"][GATE_UNEVALUATED] == 6,
        f"{evidence_report['counts'][GATE_UNEVALUATED]} evidence gates are "
        f"unevaluable, expected 6; X10's measurement is stale either way",
    )
    check(
        evidence_report["counts"][GATE_BLOCKED] == 2,
        f"{evidence_report['counts'][GATE_BLOCKED]} evidence gates carry an "
        f"explicit non-pass verdict, expected 2 (X1 NOT_APPROVED, X7 "
        f"FAILS_CORRECTION)",
    )
    # And X9 must report one of exactly two states, according to the tree.
    check(
        str(shipped["X9_release_snapshot"].get("verdict")) in ("FROZEN", "DIRTY"),
        f"X9 reported {shipped['X9_release_snapshot'].get('verdict')!r}; from a "
        f"complete checkout it is FROZEN (clean) or DIRTY (edited), and anything "
        f"else means a component went missing",
    )
    check(
        release_problems(report) == [],
        f"the shipped report is not contract-clean: {release_problems(report)}",
    )

    # THE TRAP, recomputed: only two gates announce a failure.
    explicit = [
        gate
        for gate, result in evidence.items()
        if classify(result.get("verdict")) == GATE_BLOCKED
        and str(result.get("verdict")).upper() not in UNEVALUATED_VERDICTS
    ]
    check(
        sorted(explicit) == ["X1_oos_validation", "X7_multiple_testing"],
        f"the evidence gates carrying an explicit non-pass verdict are "
        f"{sorted(explicit)}; X10 measured exactly X1 (NOT_APPROVED) and X7 "
        f"(FAILS_CORRECTION)",
    )
    # THE TRAP, recomputed on the evidence gates: six of eight announce nothing,
    # so a 'no FAIL means pass' rule would approve them.
    would_approve = [gate for gate in evidence if gate not in explicit]
    check(
        len(would_approve) == 6,
        f"a 'no FAIL means pass' rule would approve {len(would_approve)} of "
        f"{len(evidence)} evidence gates, expected 6; X10's central measurement "
        f"is stale",
    )

    # THE COMMON PREREQUISITE.
    if GATE_REPORTS_COMMON_BLOCKER:
        common = report.get("common_blocker")
        check(
            bool(common) and common.get("code") == COMMON_BLOCKER,
            f"the shipped report names no common prerequisite; X10 measured that "
            f"five blockers share the thin-fold defect",
        )
        if common:
            check(
                sorted(common["gates"]) == sorted(COMMON_BLOCKER_GATES),
                f"the common-blocker gates are {sorted(common['gates'])}, "
                f"expected {sorted(COMMON_BLOCKER_GATES)}",
            )


# --- 2. THE THIN-FOLD DEFECT, MEASURED NOT ASSERTED ------------------------------

if runs:
    fold = runs[0]["folds"][0]
    for key in ("regimes", "events", "sources", "validation", "ablation"):
        check(
            key not in fold,
            f"folds now carry {key!r}. X10's thin-fold finding is stale and the "
            f"gate it unblocks must be re-run for real",
        )
    check(
        len({run.get("target_horizon") for run in runs}) == 1,
        "the runs now span more than one target horizon; X6's "
        "HORIZONS_MISSING blocker is stale",
    )


# --- 3. UNEVALUATED BLOCKS, AND APPROVAL IS REACHABLE ----------------------------

check(GATE_UNEVALUATED_BLOCKS_RELEASE, "NOT_EVALUATED no longer blocks a release")
check(GATE_REQUIRES_ALL, "the release no longer requires every gate to pass")

approved = evaluate_release(ALL_PASSING)
check(
    approved["verdict"] == RELEASE_APPROVED,
    f"all nine gates passing reported {approved['verdict']}; if APPROVED is "
    f"unreachable then NOT_APPROVED proves nothing",
)
check(
    release_problems(approved) == [],
    f"the approved report is not contract-clean: {release_problems(approved)}",
)

for gate in RELEASE_GATES:
    degraded = dict(ALL_PASSING)
    degraded[gate] = {"verdict": "NOT_EVALUATED"}
    result = evaluate_release(degraded)
    check(
        result["verdict"] == RELEASE_NOT_APPROVED and gate in result["blocking"],
        f"eight gates passing with {gate} unevaluated reported "
        f"{result['verdict']}; one missing piece of evidence must sink a release",
    )

# A gate ABSENT from the results must block, not be skipped.
for gate in RELEASE_GATES:
    partial = {k: v for k, v in ALL_PASSING.items() if k != gate}
    result = evaluate_release(partial)
    check(
        result["verdict"] == RELEASE_NOT_APPROVED and gate in result["blocking"],
        f"{gate} absent from the results did not block; a gate that is not "
        f"listed is indistinguishable from one that passed",
    )

check(
    evaluate_release({})["counts"][GATE_UNEVALUATED] == len(RELEASE_GATES),
    "an empty result set did not mark every gate unevaluated",
)


# --- 4. CLASSIFICATION IS POSITIVE, NOT NEGATIVE ---------------------------------
#
# A rule of "anything that is not a known failure passes" is precisely what would
# approve 7 of 9. An unrecognised verdict must NOT read as a pass.

for verdict in ("SOMETHING_NEW", "PROVISIONAL", "PARTIAL", "MOSTLY_FINE"):
    check(
        classify(verdict) == GATE_BLOCKED,
        f"{verdict!r} classified as {classify(verdict)}; an unrecognised verdict "
        f"must never read as a pass",
    )
for verdict in PASSING_VERDICTS:
    check(classify(verdict) == GATE_PASSED, f"{verdict!r} is not treated as a pass")
for verdict in UNEVALUATED_VERDICTS:
    check(
        classify(verdict) == GATE_UNEVALUATED,
        f"{verdict!r} is not treated as unevaluated",
    )
for verdict in (None, "", "   "):
    check(
        classify(verdict) == GATE_UNEVALUATED,
        f"a missing verdict ({verdict!r}) did not read as unevaluated",
    )
check(
    PASSING_VERDICTS & UNEVALUATED_VERDICTS == frozenset(),
    "a verdict is listed as both passing and unevaluated",
)


# --- 5. A NEGATIVE MUST BE ACTIONABLE --------------------------------------------

check(GATE_REQUIRES_NEXT_ACTION, "a blocking gate no longer needs a next action")
for gate in RELEASE_GATES:
    check(
        gate in NEXT_ACTIONS and bool(NEXT_ACTIONS[gate].strip()),
        f"{gate} has no next action on file; 'not approved' with nowhere to go "
        f"is a verdict nobody can act on",
    )
if shipped:
    for record in evaluate_release(shipped)["gates"]:
        if record["state"] != GATE_PASSED:
            check(
                bool(str(record.get("next_action") or "").strip()),
                f"{record['gate']} blocks the release but names no next action",
            )


# --- 6. CONTRACT ------------------------------------------------------------------

check(
    HONEST_GATE_VERDICTS[0] == RELEASE_NOT_APPROVED,
    "NOT_APPROVED is not the weakest verdict; a release gate that defaults to "
    "approved is not fail-closed",
)
check(
    len(HONEST_GATE_VERDICTS) == 2,
    "the release verdict set has grown; every gradation is a place for a "
    "negative to be read as a positive",
)
check(GATE_STATES[0] == GATE_UNEVALUATED, "UNEVALUATED is not the weakest state")
check(
    GATE_BLOCKED != GATE_UNEVALUATED,
    "'the gate ran and said no' and 'the gate could not run' collapsed into one "
    "state; they need different work",
)
check(len(RELEASE_GATES) == 9, f"{len(RELEASE_GATES)} release gates declared, expected 9")
check(not HONEST_GATE_BLOCKS_TRADES, "X10 reports; the registry promotes")
check(
    GATE_NOT_MET_IS_A_VALID_OUTCOME,
    "'gate not met' is no longer a valid outcome, so an honest negative is "
    "indistinguishable from a crash",
)

forged_approval = {
    "verdict": RELEASE_APPROVED,
    "reason": "trust me",
    "blocking": ["X1_oos_validation"],
    "gates": [{"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES],
}
check(
    bool(release_problems(forged_approval)),
    "an APPROVED report with a blocking gate was accepted",
)
forged_unevaluated = {
    "verdict": RELEASE_APPROVED,
    "reason": "r",
    "blocking": [],
    "unevaluated": ["X3_regime_robustness"],
    "counts": {GATE_PASSED: 9, "total": 9},
    "gates": [{"gate": gate, "state": GATE_PASSED} for gate in RELEASE_GATES],
}
check(
    bool(release_problems(forged_unevaluated)),
    "an APPROVED report with an unevaluated gate was accepted; absence of "
    "evidence is not evidence of safety",
)
forged_omission = {
    "verdict": RELEASE_NOT_APPROVED,
    "reason": "r",
    "blocking": ["X1_oos_validation"],
    "gates": [
        {"gate": "X1_oos_validation", "state": GATE_BLOCKED, "next_action": "x"}
    ],
}
check(
    bool(release_problems(forged_omission)),
    "a report omitting eight gates was accepted",
)

if shipped:
    lines = render_release(evaluate_release(shipped))
    text = "\n".join(lines)
    check(
        "FORECASTING RELEASE: NOT APPROVED" in text,
        "the rendered report does not carry the roadmap's headline",
    )
    check(
        "NEXT ACTIONS" in text,
        "the rendered report does not list next actions",
    )
    check(
        "blocks the release" in text,
        "the rendered report does not say that unevaluated gates block",
    )
    check(
        bool(lines) and all(isinstance(line, str) for line in lines),
        "render_release returned no lines",
    )


if failures:
    print("X10 HONEST GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

# THE REPORT ITSELF IS WELL-FORMED, so this gate passes — while the RELEASE it
# describes is not approved. Those are different questions, and X10 exists to
# keep them apart.
print("X10 honest gate: OK")
print()
if shipped:
    print("\n".join(render_release(evaluate_release(shipped))))
    print()
print("  the report is well-formed; the RELEASE is NOT APPROVED, which is a")
print("  valid engineering outcome and not a failure of this gate")
sys.exit(0)
