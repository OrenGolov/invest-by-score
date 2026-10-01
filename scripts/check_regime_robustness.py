"""X3 governance gate — no single regime may explain the entire edge.

Exits 1 when any of these fails.

THE BLOCKING MEASUREMENT, RECOMPUTED HERE from the tracked training runs rather
than quoted: not one persisted fold carries a regime label, and none can be
derived, because a fold holds only a fold-level window with no ticker and no
per-observation timestamp. X3 therefore reports NOT_EVALUATED on the shipped
data, and that is the honest answer.

THE DETECTION ITSELF is proved on SEEDED books, so the gate keeps working once
labels exist — and would catch a regression in the statistic today.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    REGIME_CONCENTRATED,
    REGIME_LABELS,
    REGIME_ROBUST,
    REGIME_ROBUSTNESS_LABELS,
    REGIME_ROBUSTNESS_MAX_DROP,
    REGIME_ROBUSTNESS_MIN_CELL,
    REGIME_ROBUSTNESS_NOT_EVALUATED,
    REGIME_SPECIALIZATION_MIN_CELL,
)
from core.regime_robustness import (  # noqa: E402
    RR_REASON_CARRIED,
    RR_REASON_THIN_CELLS,
    RegimeRobustnessError,
    evaluate_regime_robustness,
    fold_regime_labels,
    group_by_regime,
    leave_one_out,
    render_robustness,
    robustness_problems,
)
from core.training import load_training_runs  # noqa: E402

REGIMES = ["bullish", "bearish", "range", "risk_off"]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def book(concentrated: bool, n: int = 100, seed: int = 303):
    """A four-regime book whose edge is carried by one regime, or broad."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    labels: list[str] = []
    for regime in REGIMES:
        outcome = rng.normal(0, 0.05, n)
        if concentrated:
            predicted = (
                outcome * 0.9 + rng.normal(0, 0.005, n)
                if regime == "bullish"
                else rng.normal(0, 0.05, n)
            )
        else:
            predicted = outcome * 0.35 + rng.normal(0, 0.03, n)
        predictions.extend(predicted)
        actuals.extend(outcome)
        labels.extend([regime] * n)
    return predictions, actuals, labels


# --- 1. THE BLOCKING MEASUREMENT: no shipped fold carries a regime ---------------

runs = load_training_runs()
check(bool(runs), "no training runs are available; X3 cannot state its blocker")

# RESTATED 2026-10-01. A1 records a regime label per validation observation and
# the ledger was regenerated, so the blocking claim - "not one persisted fold
# carries a regime label" - is history. The gate now requires them to be there.
labelled = [r["estimator"] for r in runs if fold_regime_labels(r["folds"][0]) is not None]
check(
    bool(labelled),
    "no run carries per-observation regime labels. A1 added them and the ledger "
    "was regenerated; losing them re-blocks X3 entirely",
)

if runs:
    # The ledger is append-only and holds runs of two vintages: the originals
    # (no context) and those written after A1. Read a LABELLED one, or this
    # measures the pre-A1 world forever.
    labelled_runs = [r for r in runs if fold_regime_labels(r["folds"][0]) is not None]
    fold = (labelled_runs or runs)[0]["folds"][0]
    shipped = evaluate_regime_robustness(
        fold["predictions"], fold["actuals"], fold_regime_labels(fold)
    )
    check(
        shipped["verdict"] != REGIME_ROBUSTNESS_NOT_EVALUATED,
        f"a shipped run still reported NOT_EVALUATED with labels present; the "
        f"labels exist, so the comparison must produce a verdict",
    )
    check(
        shipped.get("pooled_accuracy") is not None,
        "an evaluated report must carry a pooled accuracy",
    )
    check(
        shipped.get("worst_drop") is not None,
        "an evaluated report must carry a worst leave-one-out drop",
    )


# --- 2. A LABEL IS NEVER DERIVED FROM THE FOLD WINDOW -----------------------------
# The rule that makes the blocker honest. Stamping one regime on every
# observation would leave nothing to leave out, and the gate would return ROBUST
# on no evidence whatsoever.

windowed = {
    "predictions": [1.0] * 10,
    "actuals": [1.0] * 10,
    "train_end_time": "2025-04-16 13:30:00",
    "validation_start_time": "2025-07-21 13:30:00",
}
check(
    fold_regime_labels(windowed) is None,
    "a regime was derived from the fold window; every observation would share "
    "one label, leave-one-out would have nothing to leave, and the gate would "
    "return ROBUST on no evidence",
)


# --- 3. THE DETECTION WORKS: a carried edge is caught -----------------------------

concentrated = evaluate_regime_robustness(*book(concentrated=True))
check(
    concentrated["verdict"] == REGIME_CONCENTRATED,
    f"a book where one regime carries the edge was "
    f"{concentrated['verdict']}, expected CONCENTRATED",
)
check(
    concentrated["reason_code"] == RR_REASON_CARRIED,
    f"a carried edge gave reason {concentrated['reason_code']!r}",
)
check(
    concentrated.get("carrier") == "bullish",
    f"the carrier was named {concentrated.get('carrier')!r}, expected 'bullish'",
)
check(
    concentrated["worst_drop"] > REGIME_ROBUSTNESS_MAX_DROP,
    f"the carried book's worst drop is {concentrated['worst_drop']:.4f}, not "
    f"above the {REGIME_ROBUSTNESS_MAX_DROP} bar; the probe is inert",
)


# --- 4. AND IT CAN PASS: a broad edge is ROBUST -----------------------------------
# A gate that only ever refuses is a constant, not a test.

broad = evaluate_regime_robustness(*book(concentrated=False))
check(
    broad["verdict"] == REGIME_ROBUST,
    f"a book with a uniform edge was {broad['verdict']}, expected ROBUST; a "
    f"gate that cannot pass is a constant",
)
check(
    broad["worst_drop"] <= REGIME_ROBUSTNESS_MAX_DROP,
    f"the broad book's worst drop is {broad['worst_drop']:.4f}, above the bar",
)


# --- 5. WHY "EVERY REGIME BEATS A COIN FLIP" IS THE WRONG TEST --------------------
# MEASURED: it answers 4 of 4 for BOTH books, so it separates nothing. This is
# the check that stops someone "simplifying" the statistic later.

for label, report in (("concentrated", concentrated), ("broad", broad)):
    above = [
        regime
        for regime, cell in report["cells"].items()
        if cell["directional_accuracy"] > 0.5
    ]
    check(
        len(above) == len(REGIMES),
        f"{label}: only {len(above)} of {len(REGIMES)} regimes beat a coin "
        f"flip, so the two books are no longer indistinguishable on that "
        f"statistic and this demonstration is stale",
    )


# --- 6. THE THRESHOLD IS THE MEASURED NULL, recomputed ----------------------------
# 120 uniform-edge trials here rather than the 400 used to derive the bar - the
# point is that the observed maximum stays under it, not to re-derive it.

worst_uniform = 0.0
for seed in range(120):
    report = evaluate_regime_robustness(*book(concentrated=False, seed=seed))
    if report["verdict"] == REGIME_ROBUSTNESS_NOT_EVALUATED:
        failures.append(f"uniform trial seed={seed} could not be evaluated")
        break
    worst_uniform = max(worst_uniform, report["worst_drop"])

check(
    worst_uniform <= REGIME_ROBUSTNESS_MAX_DROP,
    f"a GENUINELY UNIFORM edge produced a worst leave-one-out drop of "
    f"{worst_uniform:.4f}, above the {REGIME_ROBUSTNESS_MAX_DROP} bar; robust "
    f"models would be flagged as concentrated",
)


# --- 7. A thin cell cannot support the claim IN EITHER DIRECTION ------------------

thin = evaluate_regime_robustness(*book(concentrated=False, n=REGIME_ROBUSTNESS_MIN_CELL - 1))
check(
    thin["verdict"] == REGIME_ROBUSTNESS_NOT_EVALUATED,
    f"a book whose cells are all below the {REGIME_ROBUSTNESS_MIN_CELL} floor "
    f"was {thin['verdict']}; a thin cell cannot support a leave-one-out claim",
)
check(
    thin["reason_code"] == RR_REASON_THIN_CELLS,
    f"a thin book gave reason {thin['reason_code']!r}",
)

# A thin cell must still be VISIBLE, and must not be counted as usable.
rng = np.random.default_rng(2)
big = REGIME_ROBUSTNESS_MIN_CELL + 20
mixed_predictions: list[float] = []
mixed_actuals: list[float] = []
mixed_labels: list[str] = []
for regime, count in (("bullish", big), ("bearish", big), ("range", 5)):
    outcome = rng.normal(0, 0.05, count)
    mixed_predictions.extend(outcome * 0.35 + rng.normal(0, 0.03, count))
    mixed_actuals.extend(outcome)
    mixed_labels.extend([regime] * count)
mixed = evaluate_regime_robustness(mixed_predictions, mixed_actuals, mixed_labels)
check(
    "range" in (mixed.get("cells") or {}),
    "a thin regime vanished from the report; it must be visible",
)
check(
    "range" not in (mixed.get("usable_regimes") or []),
    "a 5-observation regime was counted as usable",
)


# --- 8. An unknown regime raises rather than being scored -------------------------

try:
    group_by_regime([1.0], [1.0], ["euphoria"])
except RegimeRobustnessError:
    pass
else:
    failures.append(
        "an unrecognised regime was scored; an edge could hide in a regime "
        "nobody checked"
    )

# AN UNLABELLED OBSERVATION IS EXCLUDED, NEVER SCORED.
#
# CONTRACT CHANGED BY A1: this previously required a raise. A1 records
# per-observation regime labels and a classifier can fail on a single day, so
# raising would throw away an otherwise usable fold. What must still hold is that
# the observation is not scored under any label — checked here directly rather
# than via the exception it used to produce.
partial = group_by_regime([1.0, 2.0], [1.0, 2.0], [None, "bullish"])
if "None" in partial or "none" in partial:
    failures.append(
        "an unlabelled observation became its own regime cell; assigning one "
        "manufactures the attribution this gate tests"
    )
if partial.get("bullish", {}).get("observations") != 1:
    failures.append(
        "excluding the unlabelled observation also dropped the labelled one; "
        "partial context must keep the evidence it has"
    )
if group_by_regime([1.0], [1.0], [None]) != {}:
    failures.append(
        "a wholly unlabelled fold produced cells; with nothing classified there "
        "is nothing to compare and the caller must report NOT_EVALUATED"
    )

# ...and the reader reports absence as absence, so the caller can tell the two
# apart without inspecting every entry.
if fold_regime_labels({"regimes": [None, None]}) is not None:
    failures.append(
        "a wholly unobserved regime list did not read as absent; str(None) would "
        "make 'None' a regime and hide an edge in a bucket nobody checked"
    )
if fold_regime_labels({"regimes": [None, "bullish"]}) != [None, "bullish"]:
    failures.append(
        "partial regime labels were not preserved; the observed half is still "
        "evidence"
    )

try:
    leave_one_out(group_by_regime([1.0] * 2, [1.0] * 2, ["bullish"] * 2))
except RegimeRobustnessError:
    pass
else:
    failures.append(
        "leave-one-out ran on a single regime; there is nothing left to "
        "compare against"
    )


# --- 9. Vocabulary and floors are REUSED, not invented ----------------------------

check(
    set(REGIME_ROBUSTNESS_LABELS) == set(REGIME_LABELS),
    "X3 does not test the same regimes N4 classifies, so an edge could hide "
    "in a regime nobody checked",
)
check(
    REGIME_ROBUSTNESS_MIN_CELL == REGIME_SPECIALIZATION_MIN_CELL,
    "X3 and L7 disagree on what a usable regime cell is; one would trust a "
    "cell the other rejects",
)


# --- 10. Every report renders and passes its own contract ------------------------

for label, report in (
    ("concentrated", concentrated),
    ("broad", broad),
    ("thin", thin),
    ("mixed", mixed),
):
    problems = robustness_problems(report)
    if problems:
        failures.append(f"the {label} report failed its own contract: {problems}")
        break
    lines = render_robustness(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_robustness returned no lines for {label}")
        break

if runs:
    problems = robustness_problems(shipped)
    check(
        problems == [],
        f"the shipped NOT_EVALUATED report failed its own contract: {problems}",
    )


if failures:
    print("X3 REGIME ROBUSTNESS GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X3 regime robustness gate: OK")
print(f"  shipped runs with regime labels   {len(labelled)} of {len(runs)}  -> {shipped['verdict']}")
print(f"  carried book                      {concentrated['verdict']}"
      f" (carrier {concentrated['carrier']}, drop {concentrated['worst_drop']:.4f})")
print(f"  uniform book                      {broad['verdict']}"
      f" (drop {broad['worst_drop']:.4f})")
print(f"  worst drop over 120 uniform trials {worst_uniform:.4f}"
      f"  (bar {REGIME_ROBUSTNESS_MAX_DROP})")
print("  'every regime beats a coin flip'  4 of 4 in BOTH books - separates nothing")
sys.exit(0)
