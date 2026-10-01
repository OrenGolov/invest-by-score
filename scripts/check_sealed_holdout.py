"""X8 governance gate — the final period must be sealed, and provably so.

Exits 1 when any of these fails.

THE DECIDING MEASUREMENT, RECOMPUTED HERE: `data/training_runs.jsonl` (8 runs)
and `data/training_datasets.jsonl` (406 rows) are both TRACKED, so this
recomputes identically on a fresh clone. Neither is gitignored, which open item
6 in `docs/open-decisions.md` records as the rule these gates broke twice.

The shipped answer is NOT_EVALUATED: the tail is untouched by arithmetic
accident, the seal is recorded nowhere, and no legal geometry fits the data. The
seal MACHINERY is proved on constructed runs, so SEALED and BURNED keep being
exercised while the shipped verdict stays honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    HOLDOUT_BURNED,
    HOLDOUT_MAX_OPENINGS,
    HOLDOUT_NOT_EVALUATED,
    HOLDOUT_OPENED,
    HOLDOUT_RECORDS_BOUNDS,
    HOLDOUT_RECORDS_GEOMETRY,
    HOLDOUT_REQUIRES_EMBARGO,
    HOLDOUT_SEALED,
    HOLDOUT_VERDICTS,
    LABEL_HORIZON_SESSIONS,
)
from core.sealed_holdout import (  # noqa: E402
    HOLDOUT_REASON_EMBARGO_SHORT,
    HOLDOUT_REASON_NO_RUNS,
    HOLDOUT_REASON_OVERLAP,
    SEAL_REQUIRED_FACTS,
    describe_seal,
    evaluate_sealed_holdout,
    geometry_problems,
    holdout_problems,
    max_label_horizon,
    minimum_rows,
    opening_verdict,
    recorded_facts,
    render_sealed_holdout,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def constructed(rows=700, fold=60, embargo=252, holdout=60, validation_end=371):
    """A run that records all four seal facts. 700 rows clears the 432 minimum."""
    return {
        "estimator": "constructed",
        "holdout": [rows - holdout, rows - 1],
        "dataset_rows": rows,
        "geometry": {
            "fold_sessions": fold,
            "embargo_sessions": embargo,
            "holdout_sessions": holdout,
        },
        "folds": [
            {
                "fold_id": 0,
                "train": [0, fold - 1],
                "validation": [validation_end - fold + 1, validation_end],
            }
        ],
    }


# --- 1. THE DECIDING MEASUREMENT: the seal is recorded nowhere --------------------

runs = load_training_runs()
check(bool(runs), "no training runs are available; X8 cannot state its blocker")

datasets_path = REPO_ROOT / "data" / "training_datasets.jsonl"
dataset_rows = None
if datasets_path.exists():
    lines = [
        line
        for line in datasets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if lines:
        dataset_rows = json.loads(lines[0]).get("row_count")

check(
    dataset_rows == 406,
    f"the dataset ledger reports {dataset_rows} rows, expected 406; X8's "
    f"geometry shortfall is measured against that count",
)

if runs:
    recorded = [
        estimator
        for estimator, facts in (
            (run["estimator"], recorded_facts(run)) for run in runs
        )
        if any(facts.values())
    ]
    # RESTATED 2026-10-01. A1 added the four seal facts to TrainingRun and the
    # ledger was regenerated, so the blocking claim this gate carried - "of the
    # four facts needed to verify a seal, the run ledger carries none" - is
    # history. The gate now asserts the opposite: the runs that record a seal
    # must produce a real verdict rather than NOT_EVALUATED.
    sealed_cohort = [run for run in runs if run.get("dataset_rows")]
    check(
        bool(sealed_cohort),
        "no run records its dataset row count. A1 added the seal facts and the "
        "ledger was regenerated with them; if none survives, that regression "
        "re-blocks X8",
    )

    # NO dataset_rows OVERRIDE: the run records its own count, which is what A1
    # added it for. The dataset ledger's first line describes the OLD 406-row
    # dataset, and passing it made a 1,464-row run's folds look like they
    # reached into a 406-row holdout.
    shipped = evaluate_sealed_holdout(sealed_cohort or runs)
    check(
        shipped["verdict"] == HOLDOUT_SEALED,
        f"the shipped runs reported {shipped['verdict']}, not SEALED. A1 records "
        f"the bounds, the geometry and the absolute fold indices, and A2 set a "
        f"geometry whose last fold stops a full horizon short of the tail",
    )
    check(
        shipped.get("holdout") is not None,
        "a SEALED report must name its bounds; an unlocated seal is not a seal",
    )
    facts = recorded_facts((sealed_cohort or runs)[0])
    check(
        all(facts.values()),
        f"the runs record only {[k for k, v in facts.items() if v]} of the four "
        f"seal facts; X8 cannot be audited on a partial record",
    )
    check(
        holdout_problems(shipped) == [],
        f"the shipped report is not contract-clean: {holdout_problems(shipped)}",
    )


# --- 2. NO LEGAL GEOMETRY FITS THE DATA ------------------------------------------
#
# F2 (34bd464) added the 252-session horizon the day AFTER the ledger was
# written, so the geometry that produced it is illegal today and nothing smaller
# fits either.

horizon = max_label_horizon()
check(
    horizon == max(LABEL_HORIZON_SESSIONS.values()) == 252,
    f"the longest label horizon is {horizon}; X8's shortfall arithmetic assumes "
    f"252",
)

check(
    bool(geometry_problems(406, 120, 60, 60)),
    "the geometry that produced the shipped ledger (fold=120, embargo=60, "
    "holdout=60) is reported legal; an embargo of 60 cannot cover a "
    "252-session horizon",
)

for fold in (60, 120, 252):
    for holdout in (60, 126):
        problems = geometry_problems(406, fold, 252, holdout)
        check(
            bool(problems),
            f"fold={fold}, embargo=252, holdout={holdout} is reported to fit "
            f"406 rows; it needs {minimum_rows(fold, 252, holdout)}",
        )

check(
    minimum_rows(60, 252, 60) == 432,
    f"the cheapest legal geometry needs {minimum_rows(60, 252, 60)} rows, "
    f"expected 432",
)
check(
    "short by 26" in " ".join(geometry_problems(406, 60, 252, 60)),
    "the shortfall against the cheapest legal geometry is not named in rows",
)
check(
    geometry_problems(432, 60, 252, 60) == [],
    "432 rows is reported not to fit the cheapest legal geometry, so the "
    "shortfall arithmetic is wrong in the other direction",
)


# --- 3. THE SEAL MACHINERY STILL WORKS -------------------------------------------
#
# Proved on CONSTRUCTED runs. A gate whose only verdict is NOT_EVALUATED would
# keep passing if `evaluate_sealed_holdout` could never reach any other state.

sealed = evaluate_sealed_holdout([constructed()])
check(
    sealed["verdict"] == HOLDOUT_SEALED,
    f"a recorded, legal, unread holdout reported {sealed['verdict']}, not SEALED",
)
check(
    sealed.get("holdout") == [640, 699],
    f"the sealed bounds are {sealed.get('holdout')}, expected [640, 699]",
)
check(holdout_problems(sealed) == [], f"sealed report unclean: {holdout_problems(sealed)}")

overlap = evaluate_sealed_holdout([constructed(validation_end=660)])
check(
    overlap["verdict"] == HOLDOUT_BURNED
    and overlap.get("reason_code") == HOLDOUT_REASON_OVERLAP,
    f"a fold validating inside the holdout reported {overlap['verdict']}, "
    f"not BURNED",
)

# THE SEAL IS A PROPERTY OF THE WHOLE SEARCH: one estimator reaching into the
# tail spends it for every other one.
shared = evaluate_sealed_holdout([constructed(), constructed(validation_end=660)])
check(
    shared["verdict"] == HOLDOUT_BURNED,
    "one run reaching into the tail did not burn the seal for the others; the "
    "holdout is shared across the whole search",
)


# --- 4. A LEGAL EMBARGO SETTING IS NOT AN EMBARGOED SEAL -------------------------
#
# MEASURED: the setting bounds fold SPACING, the gap is where the last fold
# actually stopped, and only the gap can leak a holdout label. A gate checking
# the setting alone passes a run whose final fold ran up against the tail.

described = describe_seal(700, 60, 252, 60, last_validation_row=600)
check(
    described["legal"] and not described["gap_covers_horizon"],
    "a legal embargo setting with a 39-row gap to the tail was not "
    "distinguished from a properly embargoed seal",
)
check(
    described["gap_to_holdout"] == 39,
    f"the gap is reported as {described['gap_to_holdout']} rows, expected 39",
)

short_gap = evaluate_sealed_holdout([constructed(validation_end=600)])
check(
    short_gap["verdict"] == HOLDOUT_NOT_EVALUATED
    and short_gap.get("reason_code") == HOLDOUT_REASON_EMBARGO_SHORT,
    f"a 39-row gap against a 252-session horizon reported "
    f"{short_gap['verdict']}; a holdout label computed from prices the last "
    f"fold trained on is not sealed",
)
check(HOLDOUT_REQUIRES_EMBARGO, "the embargo requirement is switched off")


# --- 5. OPENING IS A ONE-WAY EVENT ------------------------------------------------

check(
    opening_verdict([]) is None and opening_verdict(None) is None,
    "'never opened' was treated as a verdict; it is not the same as sealed, "
    "because the geometry still has to hold",
)

once = evaluate_sealed_holdout(
    [constructed()], openings=[{"opened_by": "oren", "reason": "X9 release snapshot"}]
)
check(
    once["verdict"] == HOLDOUT_OPENED,
    f"one named opening reported {once['verdict']}, not OPENED",
)
check(holdout_problems(once) == [], f"opened report unclean: {holdout_problems(once)}")

twice = evaluate_sealed_holdout(
    [constructed()],
    openings=[
        {"opened_by": "a", "reason": "one"},
        {"opened_by": "b", "reason": "two"},
    ],
)
check(
    twice["verdict"] == HOLDOUT_BURNED,
    f"{HOLDOUT_MAX_OPENINGS + 1} openings reported {twice['verdict']}, not BURNED",
)

for anonymous in ({"opened_by": "", "reason": "r"}, {"opened_by": "a", "reason": ""}, {}):
    verdict = evaluate_sealed_holdout([constructed()], openings=[anonymous])["verdict"]
    check(
        verdict == HOLDOUT_BURNED,
        f"an anonymous opening ({anonymous}) reported {verdict}; 'do not use it "
        f"for further model selection' cannot be enforced against a read that "
        f"names nobody",
    )

selected = evaluate_sealed_holdout(
    [constructed()],
    openings=[{"opened_by": "a", "reason": "tuning", "used_for_selection": True}],
)
check(
    selected["verdict"] == HOLDOUT_BURNED,
    "using the holdout for model selection did not burn it, which is the one "
    "thing a sealed holdout forbids",
)

# An opening outranks the geometry: the read already happened.
regardless = evaluate_sealed_holdout(
    runs or [constructed()],
    dataset_rows=dataset_rows,
    openings=[{"opened_by": "a", "reason": "r"}, {"opened_by": "b", "reason": "r"}],
)
check(
    regardless["verdict"] == HOLDOUT_BURNED,
    "an unrecorded seal that was read twice reported something other than "
    "BURNED; the read outranks the missing geometry",
)


# --- 6. CONTRACT ------------------------------------------------------------------

check(HOLDOUT_RECORDS_BOUNDS, "the contract no longer requires recorded bounds")
check(HOLDOUT_RECORDS_GEOMETRY, "the contract no longer requires recorded geometry")
check(
    HOLDOUT_VERDICTS[0] == HOLDOUT_NOT_EVALUATED
    and HOLDOUT_VERDICTS[-1] == HOLDOUT_SEALED,
    "the holdout verdicts no longer run weakest to strongest",
)
check(
    HOLDOUT_BURNED != HOLDOUT_NOT_EVALUATED,
    "'the seal was broken' and 'the seal could not be established' collapsed "
    "into one verdict; they have different fixes",
)
check(
    HOLDOUT_SEALED != HOLDOUT_OPENED,
    "a holdout read once is spent, not sealed",
)
check(
    evaluate_sealed_holdout([])["reason_code"] == HOLDOUT_REASON_NO_RUNS,
    "an empty run set did not report NO_RUNS",
)
check(
    bool(holdout_problems({"verdict": HOLDOUT_SEALED, "reason": "trust me"})),
    "a SEALED report with no bounds was accepted; an unrecorded seal is the "
    "defect X8 measured",
)
check(
    bool(
        holdout_problems(
            {
                "verdict": HOLDOUT_SEALED,
                "reason": "r",
                "holdout": [1, 2],
                "geometry": {},
                "dataset_rows": 700,
                "openings": 1,
            }
        )
    ),
    "a SEALED report that records an opening was accepted",
)

for label, report in (
    ("shipped", evaluate_sealed_holdout(runs or [], dataset_rows=dataset_rows)),
    ("sealed", sealed),
    ("burned", overlap),
    ("opened", once),
):
    lines = render_sealed_holdout(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_sealed_holdout returned no lines for {label}")
        break

absent = "\n".join(
    render_sealed_holdout(evaluate_sealed_holdout(runs or [], dataset_rows=dataset_rows))
)
check(
    "ABSENT" in absent and "0.0%" not in absent,
    "an unlocated seal did not render as ABSENT; rendering it as 0 would read "
    "as an empty holdout rather than a missing one",
)


if failures:
    print("X8 SEALED HOLDOUT GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X8 sealed holdout gate: OK")
print(f"  dataset rows                      {dataset_rows}")
print(f"  seal facts recorded               {sum(facts.values())} of "
      f"{len(SEAL_REQUIRED_FACTS)}  -> {shipped['verdict']}")
print(f"  shipped geometry (emb 60)         ILLEGAL against a {horizon}-session horizon")
print(f"  cheapest legal geometry           {minimum_rows(60, 252, 60)} rows"
      f"  (short by {minimum_rows(60, 252, 60) - 406})")
print(f"  default geometry                  {minimum_rows(252, 252, 126)} rows"
      f"  (short by {minimum_rows(252, 252, 126) - 406})")
print(f"  constructed seal                  {sealed['verdict']}"
      f" rows {sealed['holdout'][0]}..{sealed['holdout'][1]}")
print(f"  fold inside the tail              {overlap['verdict']}")
print(f"  legal embargo, 39-row gap         {short_gap['verdict']}"
      f" ({short_gap['reason_code']})")
print(f"  opened once / twice               {once['verdict']} / {twice['verdict']}")
sys.exit(0)
