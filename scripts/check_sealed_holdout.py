"""X8 governance gate — the seal held, but nothing was holding it.

Exits 1 when any of these fails.

THE MEASUREMENT, RECOMPUTED HERE from the tracked training runs and dataset
manifest rather than quoted: the 406-row dataset, the single 120-train/120-
validation fold, the untouched 106-row tail, and the 60-row declared holdout that
sits below X1's 120-observation floor.

THE BOUNDARY IS RECOMPUTED FROM THE RECORDED GEOMETRY, never from live config.
That is finding 2 and this gate proves it: build_walk_forward_folds now REFUSES
the embargo=60 geometry that produced the shipped runs, because F2 raised the
label-horizon floor to 252 after those runs were written.

STALENESS DETECTORS. If the dataset grows past the floor, or a holdout opening is
ever recorded, or the embargo floor changes again, this gate fails and says the
finding must be revisited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.backtest.engine import build_walk_forward_folds  # noqa: E402
from core.config import (  # noqa: E402
    CALIBRATION_GATE_MIN_HOLDOUT,
    HOLDOUT_ABSENT,
    HOLDOUT_BOUNDARY_IS_RECORDED,
    HOLDOUT_FORBIDS_SELECTION_AFTER_OPENING,
    HOLDOUT_INSUFFICIENT,
    HOLDOUT_INSUFFICIENT_STAYS_SEALED,
    HOLDOUT_MAX_OPENINGS,
    HOLDOUT_MIN_OBSERVATIONS,
    HOLDOUT_OPENED,
    HOLDOUT_UNOPENED,
    LABEL_HORIZON_SESSIONS,
    OOS_MIN_OBSERVATIONS,
    SEALED_HOLDOUT_VERSION,
)
from core.sealed_holdout import (  # noqa: E402
    SealedHoldoutError,
    evaluate_sealed_holdout,
    holdout_state,
    record_opening,
    sampling_band,
    seal_problems,
    sealed_holdout_problems,
    render_sealed_holdout,
    verify_boundary,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# The geometry that produced the shipped runs, recorded because it can no longer
# be derived (finding 2). Sprint C, 2026-09-18, before F2 added the 252d horizon.
SHIPPED_FOLD_SESSIONS = 120
SHIPPED_EMBARGO_SESSIONS = 60
SHIPPED_HOLDOUT_SESSIONS = 60


def folds_from(rows: int, fold: int, embargo: int, holdout: int) -> dict:
    """The anchored walk-forward geometry, computed without the embargo floor.

    Deliberately a local reimplementation of the arithmetic in
    build_walk_forward_folds: that function enforces the CURRENT embargo floor
    and therefore cannot reproduce a seal recorded under an older one. This is
    the whole of finding 2, so the gate needs both — the refusal below, and this
    to recompute what actually ran.
    """
    holdout_start = rows - holdout
    windows = []
    fold_id = 0
    while True:
        train_end = (fold_id + 1) * fold - 1
        validation_start = train_end + embargo + 1
        validation_end = validation_start + fold - 1
        if validation_end >= holdout_start:
            break
        windows.append({
            "fold_id": fold_id,
            "train": [0, train_end],
            "validation": [validation_start, validation_end],
        })
        fold_id += 1
    return {"folds": windows, "holdout": [holdout_start, rows - 1]}


# --- 1. THE TRACKED EVIDENCE -----------------------------------------------------

runs = load_training_runs()
check(bool(runs), "no training runs are available; X8 cannot state its finding")

manifest_path = REPO_ROOT / "data" / "training_datasets.jsonl"
check(manifest_path.exists(), "data/training_datasets.jsonl is missing")

manifests = []
if manifest_path.exists():
    manifests = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
check(bool(manifests), "no dataset manifest is recorded")

row_count = None
if manifests:
    row_count = int(manifests[0]["row_count"])
    check(
        row_count == 406,
        f"the dataset holds {row_count} rows, expected 406. If the dataset grew, "
        f"X8's seal measurement is STALE and the seal geometry must be recomputed",
    )

# Every run must agree on the dataset, or "the seal" names more than one window.
if runs:
    dataset_hashes = {r["dataset_hash"] for r in runs}
    check(
        len(dataset_hashes) == 1,
        f"{len(dataset_hashes)} distinct dataset hashes across the runs; a seal "
        f"boundary is only meaningful against one dataset",
    )
    if manifests:
        check(
            dataset_hashes == {manifests[0]["dataset_hash"]},
            "the runs and the dataset manifest disagree about the dataset hash, "
            "so the recorded seal cannot be tied to the data it covers",
        )


# --- 2. FINDING 2: THE GEOMETRY CAN NO LONGER BE DERIVED -------------------------

max_horizon = max(LABEL_HORIZON_SESSIONS.values())
check(
    max_horizon > SHIPPED_EMBARGO_SESSIONS,
    f"the max label horizon is {max_horizon}, not above the shipped embargo of "
    f"{SHIPPED_EMBARGO_SESSIONS}. Finding 2 said the floor ROSE after the runs "
    f"were written; if it no longer has, the finding is STALE",
)

refused = False
try:
    build_walk_forward_folds(
        row_count or 406,
        fold_sessions=SHIPPED_FOLD_SESSIONS,
        embargo_sessions=SHIPPED_EMBARGO_SESSIONS,
        holdout_sessions=SHIPPED_HOLDOUT_SESSIONS,
    )
except ValueError:
    refused = True
check(
    refused,
    "build_walk_forward_folds ACCEPTED the shipped geometry. Finding 2 rests on "
    "it refusing that geometry under the current embargo floor; if it now "
    "accepts it, the boundary can be derived after all and X8's 'record, never "
    "derive' justification must be rewritten",
)

# And no current geometry reproduces the boundary: fold 120 forces
# embargo + holdout <= 166, below the 252 floor.
check(
    2 * SHIPPED_FOLD_SESSIONS + max_horizon + 1 > (row_count or 406),
    f"a geometry with the current {max_horizon}-session embargo now fits in "
    f"{row_count} rows, so the shipped boundary may be derivable again - "
    f"X8's finding 2 must be re-measured",
)


# --- 3. THE SEAL, RECOMPUTED FROM THE RECORDED GEOMETRY --------------------------

geometry = folds_from(
    row_count or 406,
    SHIPPED_FOLD_SESSIONS,
    SHIPPED_EMBARGO_SESSIONS,
    SHIPPED_HOLDOUT_SESSIONS,
)
check(
    len(geometry["folds"]) == 1,
    f"the recorded geometry yields {len(geometry['folds'])} folds, expected 1",
)

shipped_fold = geometry["folds"][0] if geometry["folds"] else None
if shipped_fold and runs:
    train_rows = shipped_fold["train"][1] - shipped_fold["train"][0] + 1
    validation_rows = shipped_fold["validation"][1] - shipped_fold["validation"][0] + 1
    recorded = runs[0]["folds"][0]
    check(
        train_rows == int(recorded["train_rows"]),
        f"the recomputed geometry gives {train_rows} train rows but the run "
        f"records {recorded['train_rows']}; the recorded geometry is wrong, so "
        f"the seal boundary derived from it cannot be trusted",
    )
    check(
        validation_rows == int(recorded["validation_rows"]),
        f"the recomputed geometry gives {validation_rows} validation rows but "
        f"the run records {recorded['validation_rows']}",
    )

touched_through = shipped_fold["validation"][1] if shipped_fold else None
check(
    touched_through == 299,
    f"validation last touched row {touched_through}, expected 299",
)

holdout_window = geometry["holdout"]
untouched_tail = (row_count or 406) - 1 - (touched_through or 0)
check(
    untouched_tail == 106,
    f"the untouched tail is {untouched_tail} rows, expected 106",
)

seal = {
    "dataset_hash": (runs[0]["dataset_hash"] if runs else "unknown"),
    "sealed_rows": list(holdout_window),
    "row_count": holdout_window[1] - holdout_window[0] + 1,
    "sealed_at": "2026-09-18T00:00:00Z",
    "openings": [],
}
check(
    seal_problems(seal) == [],
    f"the recomputed seal fails its own contract: {seal_problems(seal)}",
)


# --- 4. THE SEAL WAS NEVER OPENED ------------------------------------------------

# STALENESS: if a run ever records a holdout evaluation, the seal is spent and
# X8's "unopened" finding is false.
holdout_keys = []
for run in runs:
    for key in run:
        if "hold" in key.lower() or "seal" in key.lower():
            holdout_keys.append(key)
    for fold in run.get("folds", []):
        for key in fold:
            if "hold" in key.lower() or "seal" in key.lower():
                holdout_keys.append(key)
check(
    not holdout_keys,
    f"training runs now carry holdout/seal keys {sorted(set(holdout_keys))}. "
    f"X8 reported the seal as UNOPENED because no run recorded one; if a "
    f"holdout has been evaluated, that finding is STALE and the seal is spent",
)


# --- 5. FINDING 3: THE SEAL IS TOO SMALL TO DECIDE -------------------------------

report = evaluate_sealed_holdout(seal, touched_through=touched_through)
check(
    sealed_holdout_problems(report) == [],
    f"the shipped X8 report fails its own contract: {sealed_holdout_problems(report)}",
)
check(
    report["state"] == HOLDOUT_INSUFFICIENT,
    f"the shipped seal reported {report['state']}; at {seal['row_count']} rows "
    f"against a {HOLDOUT_MIN_OBSERVATIONS} floor the honest answer is "
    f"{HOLDOUT_INSUFFICIENT}. If the seal grew past the floor, X8's finding 3 "
    f"is STALE and the seal may now be openable",
)
check(
    report["may_open"] is False,
    "the shipped seal reported itself openable; an insufficient seal stays sealed",
)
check(
    seal["row_count"] < HOLDOUT_MIN_OBSERVATIONS,
    f"the seal holds {seal['row_count']} rows, at or above the "
    f"{HOLDOUT_MIN_OBSERVATIONS} floor - finding 3 is STALE",
)
check(
    untouched_tail < CALIBRATION_GATE_MIN_HOLDOUT,
    f"the {untouched_tail}-row tail now meets the {CALIBRATION_GATE_MIN_HOLDOUT} "
    f"calibration holdout bar; X8's finding 3 must be re-measured",
)

# The band is RECOMPUTED, so the quoted 62.6% keeps being justified.
band = sampling_band(seal["row_count"])
check(
    abs(band - 0.1265) < 0.0005,
    f"the sampling band at {seal['row_count']} rows is {band:.4f}, not the "
    f"0.1265 X8 quotes",
)
check(
    0.5 + band > 0.62,
    f"a coin reaches only {0.5 + band:.1%} at {seal['row_count']} rows; X8's "
    f"'a fair coin posts up to 62.6%' claim is wrong",
)

# The floor is reused, not reinvented.
check(
    HOLDOUT_MIN_OBSERVATIONS == OOS_MIN_OBSERVATIONS,
    "the holdout floor has drifted from X1's observation floor; 'enough "
    "evidence' would then mean two different things in one release gate",
)


# --- 6. THE BOUNDARY WAS NEVER BREACHED ------------------------------------------

boundary = verify_boundary(seal, touched_through)
check(
    boundary["verified"] is True,
    f"the seal boundary was breached: {boundary['reason']}",
)
check(
    boundary["gap_rows"] == 46,
    f"the gap between validation and the seal is {boundary['gap_rows']} rows, "
    f"expected 46",
)

# An unrecorded reach is NOT_VERIFIED, never a pass.
unrecorded = verify_boundary(seal, None)
check(
    unrecorded["verified"] is False,
    "a boundary with no recorded reach was reported verified; 'nobody recorded "
    "what was touched' is not 'nothing was touched'",
)

# Reaching the first sealed row is a breach, not a near miss.
check(
    verify_boundary(seal, seal["sealed_rows"][0])["verified"] is False,
    "touching the first sealed row was accepted; the seal is inclusive of its "
    "start",
)
check(
    verify_boundary(seal, seal["sealed_rows"][0] - 1)["verified"] is True,
    "stopping one row short of the seal was rejected",
)


# --- 7. ABSENT IS NOT UNOPENED ---------------------------------------------------

absent = evaluate_sealed_holdout(None)
check(
    absent["state"] == HOLDOUT_ABSENT,
    f"a missing seal reported {absent['state']!r}",
)
check(
    HOLDOUT_ABSENT != HOLDOUT_UNOPENED,
    "'no holdout exists' and 'a holdout exists and is untouched' share a value; "
    "they are opposite facts about the evidence",
)
check(
    absent["row_count"] is None,
    "an absent seal reported a row count it does not have",
)
check(
    absent["may_open"] is False,
    "an absent seal reported itself openable",
)
check(
    sealed_holdout_problems(absent) == [],
    f"the ABSENT report fails its own contract: {sealed_holdout_problems(absent)}",
)


# --- 8. OPENING IS PERMITTED EXACTLY ONCE ----------------------------------------

# Built on a seal at the floor, so the opening path is actually REACHABLE. A
# probe against the 60-row seal would be refused for size and prove nothing
# about the once-only rule.
openable = {
    **seal,
    "sealed_rows": [300, 300 + HOLDOUT_MIN_OBSERVATIONS - 1],
    "row_count": HOLDOUT_MIN_OBSERVATIONS,
}
openable_report = evaluate_sealed_holdout(openable, touched_through=299)
check(
    openable_report["state"] == HOLDOUT_UNOPENED,
    f"a seal at the floor reported {openable_report['state']}",
)
check(
    openable_report["may_open"] is True,
    f"a verified seal at the floor refused to open: {openable_report['reason']}. "
    f"The once-only rule below is only meaningful if opening is reachable",
)

opened = record_opening(
    openable,
    opened_by="orengolov02@gmail.com",
    reason="X8 gate probe: the once-only rule must be reachable to be tested",
    opened_at="2026-09-27T00:00:00Z",
    touched_through=299,
)
check(
    holdout_state(opened) == HOLDOUT_OPENED,
    f"an opened seal reported {holdout_state(opened)}",
)
check(
    len(opened["openings"]) == HOLDOUT_MAX_OPENINGS,
    f"{len(opened['openings'])} openings recorded after one opening",
)
check(
    openable["openings"] == [],
    "opening mutated the original seal; an append-only record must not be "
    "rewritten in place",
)

second_refused = False
try:
    record_opening(
        opened,
        opened_by="orengolov02@gmail.com",
        reason="a second look",
        opened_at="2026-09-28T00:00:00Z",
        touched_through=299,
    )
except SealedHoldoutError:
    second_refused = True
check(
    second_refused,
    "a SECOND opening was permitted. The second look is where selection bias "
    "enters, so it must be refused rather than logged as another data point",
)

# An opened seal may never inform selection again.
check(
    evaluate_sealed_holdout(opened, touched_through=299)["may_open"] is False,
    "an already-opened seal reported itself openable",
)

# An unattributed opening cannot be audited.
for field in ("opened_by", "reason", "opened_at"):
    kwargs = {
        "opened_by": "orengolov02@gmail.com",
        "reason": "final evaluation",
        "opened_at": "2026-09-27T00:00:00Z",
    }
    kwargs[field] = "  "
    unattributed_refused = False
    try:
        record_opening(openable, touched_through=299, **kwargs)
    except SealedHoldoutError:
        unattributed_refused = True
    check(
        unattributed_refused,
        f"an opening with a blank {field} was accepted; an unattributed "
        f"opening cannot be audited",
    )

# A breach must block opening even when the seal is big enough.
breach_refused = False
try:
    record_opening(
        openable,
        opened_by="orengolov02@gmail.com",
        reason="final evaluation",
        opened_at="2026-09-27T00:00:00Z",
        touched_through=openable["sealed_rows"][0] + 5,
    )
except SealedHoldoutError:
    breach_refused = True
check(
    breach_refused,
    "a seal whose boundary was breached was still opened; the size check must "
    "not be the only gate",
)


# --- 9. THE CONTRACT CHECK REFUSES A FORGED REPORT -------------------------------

forged_may_open = dict(report)
forged_may_open["may_open"] = True
check(
    sealed_holdout_problems(forged_may_open) != [],
    "the contract check accepted may_open on a seal below the floor",
)

forged_state = dict(report)
forged_state["state"] = HOLDOUT_UNOPENED
check(
    sealed_holdout_problems(forged_state) != [],
    "the contract check accepted UNOPENED on a seal below the floor; an intact "
    "seal too small to decide is INSUFFICIENT",
)

forged_band = dict(report)
forged_band["sampling_band"] = 0.01
check(
    sealed_holdout_problems(forged_band) != [],
    "the contract check accepted a sampling band that does not match the row "
    "count; a forged band would make an insufficient seal look decisive",
)

forged_boundary = dict(evaluate_sealed_holdout(openable, touched_through=400))
forged_boundary["boundary"] = dict(forged_boundary["boundary"], verified=True)
check(
    sealed_holdout_problems(forged_boundary) != [],
    "the contract check accepted a boundary claiming verified while validation "
    "reached inside the seal",
)

forged_openings = dict(evaluate_sealed_holdout(openable, touched_through=299))
forged_openings["openings"] = HOLDOUT_MAX_OPENINGS + 1
check(
    sealed_holdout_problems(forged_openings) != [],
    f"the contract check accepted {HOLDOUT_MAX_OPENINGS + 1} openings",
)

# A SEAL WHOSE COUNT AND RANGE DISAGREE. This probe exists because without it
# the row_count/range check is INERT in this gate: every seal built above has a
# count derived from its own range, so disabling the check changes nothing. It is
# reachable and the wrong answer is severe - MEASURED, a 60-row seal declaring
# 500 rows evaluates to UNOPENED with may_open TRUE, which is the exact failure
# X8 exists to prevent: an insufficient seal presenting itself as decisive.
inflated = {**seal, "row_count": 500}
check(
    seal_problems(inflated) != [],
    "a seal declaring 500 rows over a 60-row range was accepted; the count and "
    "the range must agree or the seal describes two different windows and "
    "neither can be verified against the data",
)
inflation_refused = False
try:
    evaluate_sealed_holdout(inflated, touched_through=touched_through)
except SealedHoldoutError:
    inflation_refused = True
check(
    inflation_refused,
    "a seal whose declared row count exceeds its range was evaluated rather "
    "than refused; MEASURED, it reports UNOPENED and may_open TRUE, turning a "
    "60-row seal into an apparently decisive one",
)

# And the same disagreement in the other direction: a count SMALLER than the
# range would understate the evidence, which is a different error with the same
# cause, so both must be refused.
check(
    seal_problems({**seal, "row_count": 10}) != [],
    "a seal declaring fewer rows than its range was accepted",
)


# --- 10. THE CONFIG CONTRACT HOLDS ------------------------------------------------

check(HOLDOUT_BOUNDARY_IS_RECORDED, "the seal boundary must be recorded, not derived")
check(
    HOLDOUT_FORBIDS_SELECTION_AFTER_OPENING,
    "an opened holdout must never inform model selection",
)
check(
    HOLDOUT_INSUFFICIENT_STAYS_SEALED,
    "a seal below the floor must stay sealed rather than be spent on a number "
    "inside the noise band",
)
check(HOLDOUT_MAX_OPENINGS == 1, "a sealed holdout is opened exactly once")
check(
    report["blocks_trades"] is False,
    "X8 reports; the registry promotes",
)

# Every report must render.
for label, candidate in (("shipped", report), ("absent", absent), ("openable", openable_report)):
    try:
        text = render_sealed_holdout(candidate)
    except SealedHoldoutError as exc:
        failures.append(f"the {label} report would not render: {exc}")
        continue
    if not text.strip():
        failures.append(f"render_sealed_holdout returned nothing for {label}")


if failures:
    print("X8 SEALED HOLDOUT GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X8 sealed holdout gate: OK")
print(f"  version                      {SEALED_HOLDOUT_VERSION}")
print(f"  dataset rows                 {row_count}")
print(f"  validation touched through   row {touched_through}")
print(f"  sealed rows                  [{holdout_window[0]}, {holdout_window[1]}]"
      f"  ({seal['row_count']} rows)")
print(f"  untouched tail               {untouched_tail} rows")
print(f"  boundary gap                 {boundary['gap_rows']} rows  (never breached)")
print(f"  observation floor            {HOLDOUT_MIN_OBSERVATIONS}"
      f"  -> state {report['state']}")
print(f"  sampling band at {seal['row_count']} rows      +/-{band:.4f}"
      f"  (a coin reaches {0.5 + band:.1%})")
print(f"  openings                     {report['openings']} of {HOLDOUT_MAX_OPENINGS}"
      f"  -> may_open {report['may_open']}")
print(f"  embargo floor then/now       {SHIPPED_EMBARGO_SESSIONS} / {max_horizon}"
      f"  (boundary recorded, not derived)")
sys.exit(0)
