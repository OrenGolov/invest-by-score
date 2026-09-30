"""X4 governance gate — no single viral event or source may explain the edge.

Exits 1 when any of these fails.

THE BLOCKING MEASUREMENT, RECOMPUTED HERE: there are 0 event memories on disk,
no fold carries an event id or a source, and the raw store holds only price and
fundamentals. X4's blocker is MISSING DATA, which is a different problem from
X3's dropped join key, and it needs TWO fixes rather than one.

THE DETECTION ITSELF is proved on SEEDED books, so the statistic keeps being
checked today and the gate keeps working once events exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    EVENT_CARRIED,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_ROBUST,
    EVENT_ROBUSTNESS_AXES,
    EVENT_ROBUSTNESS_AXIS_EVENT,
    EVENT_ROBUSTNESS_AXIS_SOURCE,
    EVENT_ROBUSTNESS_MAX_RATIO,
    EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS,
    EVENT_ROBUSTNESS_MIN_ITEMS,
    EVENT_ROBUSTNESS_NOT_EVALUATED,
)
from core.event_memory import load_memories  # noqa: E402
from core.event_robustness import (  # noqa: E402
    ER_REASON_NO_ATTRIBUTION,
    ER_REASON_TOO_FEW_ITEMS,
    EventRobustnessError,
    attribution_of,
    carrier_ratio,
    evaluate_event_robustness,
    event_robustness_problems,
    group_by_item,
    render_event_robustness,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def uniform_book(events=8, n=30, seed=11):
    """An edge spread evenly over many events and sources."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    ids: list[str] = []
    sources: list[str] = []
    for index in range(events):
        outcome = rng.normal(0, 0.05, n)
        predictions.extend(outcome * 0.35 + rng.normal(0, 0.03, n))
        actuals.extend(outcome)
        ids.extend([f"ev{index}"] * n)
        sources.extend([f"src{index % 3}"] * n)
    return predictions, actuals, ids, sources


def source_carried_book(seed=77):
    """One SOURCE carries the edge across several distinct events."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    ids: list[str] = []
    sources: list[str] = []
    for index in range(6):
        n = 40
        outcome = rng.normal(0, 0.05, n)
        carried = index % 3 == 0
        predicted = (
            outcome * 0.95 + rng.normal(0, 0.004, n)
            if carried
            else rng.normal(0, 0.05, n)
        )
        predictions.extend(predicted)
        actuals.extend(outcome)
        ids.extend([f"ev{index}"] * n)
        sources.extend(["src_hot" if carried else f"src{index}"] * n)
    return predictions, actuals, ids, sources


def viral_book(share, seed=0):
    """One EVENT supplies `share` of the book and carries the entire edge."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    ids: list[str] = []
    other = 40
    for index in range(4):
        outcome = rng.normal(0, 0.05, other)
        predictions.extend(rng.normal(0, 0.05, other))
        actuals.extend(outcome)
        ids.extend([f"ev{index}"] * other)
    viral = int(4 * other * share / (1 - share))
    outcome = rng.normal(0, 0.05, viral)
    predictions.extend(outcome * 0.95 + rng.normal(0, 0.004, viral))
    actuals.extend(outcome)
    ids.extend(["viral"] * viral)
    return predictions, actuals, ids


# --- 1. THE BLOCKING MEASUREMENT: no fold carries event attribution --------------
#
# THE MEASUREMENT IS TAKEN FROM THE TRAINING RUNS, NOT THE MEMORY STORE.
# `data/event_memory.jsonl` is GITIGNORED - 2084 rows on the machine that
# collected them, absent on a fresh clone - so a gate keyed to its length
# measures whose checkout it is running in. THIS ONE FAILED LOCALLY AND PASSED
# IN CI for exactly that reason, the mirror image of the live-chart bug in
# `docs/open-decisions.md` item 6.
#
# X4's blocker was never "the store is empty". It is that NO TRAINING FOLD
# CARRIES PER-OBSERVATION EVENT ATTRIBUTION, so there is nothing to leave out
# in a leave-one-out test. `data/training_runs.jsonl` is TRACKED, so that
# measurement recomputes identically everywhere. The memory count is reported
# as CONTEXT - collecting events is progress toward the fix - but it is the
# attribution that decides, because memories that never reach a fold cannot
# make the robustness test runnable.

memories = load_memories()

runs = load_training_runs()
check(bool(runs), "no training runs are available; X4 cannot state its blocker")

attributed = [
    run["estimator"]
    for run in runs
    if attribution_of(run["folds"][0], EVENT_ROBUSTNESS_AXIS_EVENT) is not None
    or attribution_of(run["folds"][0], EVENT_ROBUSTNESS_AXIS_SOURCE) is not None
]
check(
    not attributed,
    f"{attributed} now carry per-observation event or source attribution. "
    f"This gate's NOT_EVALUATED claim is stale and the real test must run",
)

if runs:
    fold = runs[0]["folds"][0]
    shipped = evaluate_event_robustness(
        fold["predictions"],
        fold["actuals"],
        attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT),
        attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE),
    )
    check(
        shipped["verdict"] == EVENT_ROBUSTNESS_NOT_EVALUATED,
        f"a shipped run reported {shipped['verdict']}; with no event data the "
        f"only honest answer is NOT_EVALUATED",
    )
    check(
        shipped["reason_code"] == ER_REASON_NO_ATTRIBUTION,
        f"a shipped run gave reason {shipped['reason_code']!r}",
    )
    # TWO fixes, not one: ingesting news without carrying the ids still
    # leaves nothing joinable.
    check(
        "ingest news" in shipped["reason"]
        and "onto each validation observation" in shipped["reason"],
        "the NOT_EVALUATED reason does not name BOTH fixes; ingesting news "
        "alone still leaves nothing to join an edge to",
    )
    check(
        shipped.get("worst_ratio") is None and shipped.get("pooled_accuracy") is None,
        "an unevaluated report carried a ratio or an accuracy; those cannot "
        "exist without attribution and must be ABSENT",
    )


# --- 2. ATTRIBUTION IS NEVER SYNTHESISED ------------------------------------------

windowed = {
    "predictions": [1.0] * 10,
    "actuals": [1.0] * 10,
    "validation_start_time": "2025-07-21 13:30:00",
}
for axis in EVENT_ROBUSTNESS_AXES:
    check(
        attribution_of(windowed, axis) is None,
        f"an {axis} was synthesised from the fold; one id for every "
        f"observation leaves leave-one-out nothing to leave, and the gate "
        f"would return ROBUST having tested nothing",
    )


# --- 3. A RAW DROP CANNOT BE THE TEST ---------------------------------------------
# MEASURED: removing an ORDINARY event from a small book drops accuracy
# materially, purely because it is a large share. The RATIO does not move.

small_predictions, small_actuals, small_ids, _ = uniform_book(events=5, n=20, seed=3)
small = carrier_ratio(small_predictions, small_actuals, small_ids)
check(
    small["carrier_share"] > 0.15,
    f"the small-book probe's largest item is only "
    f"{small['carrier_share']:.1%} of observations; it cannot demonstrate "
    f"that a raw drop scales with share",
)
check(
    small["worst_ratio"] <= EVENT_ROBUSTNESS_MAX_RATIO,
    f"an innocent small book scored a ratio of {small['worst_ratio']:.3f}, "
    f"above the {EVENT_ROBUSTNESS_MAX_RATIO} bar; a book with few events is "
    f"not a carried edge",
)

# The ratio must stay scale-free as the book's composition changes.
scale_free = []
for events in (4, 8, 16):
    predictions, actuals, ids, _ = uniform_book(events=events, n=30, seed=21)
    scale_free.append(carrier_ratio(predictions, actuals, ids)["worst_ratio"])
check(
    all(ratio <= EVENT_ROBUSTNESS_MAX_RATIO for ratio in scale_free),
    f"a uniform edge exceeded the bar at some book size: {scale_free}; the "
    f"statistic is not scale-free",
)


# --- 4. THE DETECTION WORKS: a viral event is caught at every share ---------------

for share in (0.2, 0.3, 0.5, 0.6):
    predictions, actuals, ids = viral_book(share)
    report = evaluate_event_robustness(predictions, actuals, ids, None)
    check(
        report["verdict"] == EVENT_CARRIED,
        f"a viral event holding {share:.0%} of the book was "
        f"{report['verdict']}, expected CARRIED",
    )
    check(
        report.get("carrier") == "viral",
        f"at share {share:.0%} the carrier was named "
        f"{report.get('carrier')!r}, expected 'viral'",
    )


# --- 5. AND IT CAN PASS -----------------------------------------------------------

broad = evaluate_event_robustness(*uniform_book())
check(
    broad["verdict"] == EVENT_ROBUST,
    f"a uniformly-spread edge was {broad['verdict']}, expected ROBUST; a gate "
    f"that cannot pass is a constant, not a test",
)


# --- 6. BOTH AXES ARE NEEDED ------------------------------------------------------
# THE DECIDING CASE: a source supplying every third event is INVISIBLE on the
# event axis. This is the check that stops someone dropping an axis later.

carried_predictions, carried_actuals, carried_ids, carried_sources = source_carried_book()
events_only = evaluate_event_robustness(
    carried_predictions, carried_actuals, carried_ids, None
)
both_axes = evaluate_event_robustness(
    carried_predictions, carried_actuals, carried_ids, carried_sources
)
check(
    events_only["verdict"] == EVENT_ROBUST,
    f"the source-carried book was {events_only['verdict']} on the EVENT axis "
    f"alone (ratio {events_only.get('worst_ratio', float('nan')):.3f}); if the "
    f"event axis already catches it, this demonstration is stale",
)
check(
    both_axes["verdict"] == EVENT_CARRIED,
    f"the source-carried book was {both_axes['verdict']} with both axes; the "
    f"source carrier must be caught",
)
check(
    both_axes.get("worst_axis") == EVENT_ROBUSTNESS_AXIS_SOURCE,
    f"the carrier was attributed to the "
    f"{both_axes.get('worst_axis')!r} axis, expected "
    f"{EVENT_ROBUSTNESS_AXIS_SOURCE!r}",
)
check(
    both_axes.get("carrier") == "src_hot",
    f"the source carrier was named {both_axes.get('carrier')!r}",
)


# --- 7. THE BAR SITS IN THE MEASURED GAP, recomputed ------------------------------
# 60 uniform books and 40 carried books here rather than the 480/320 used to
# derive it - the point is that the gap survives, not to re-derive it.

worst_uniform = 0.0
for seed in range(60):
    for events in (4, 5, 10):
        predictions, actuals, ids, _ = uniform_book(events=events, n=30, seed=seed * 7 + events)
        worst_uniform = max(
            worst_uniform, carrier_ratio(predictions, actuals, ids)["worst_ratio"]
        )

# AND the three-item case must be REFUSED rather than scored, because MEASURED
# its null reaches 0.350 and touches the bar.
try:
    carrier_ratio(*uniform_book(events=3, n=30, seed=1)[:3])
except EventRobustnessError:
    pass
else:
    failures.append(
        "a three-item book was scored; MEASURED, removing one of three items "
        "deletes a third of the evidence and the uniform null reaches 0.350, "
        "touching the 0.35 bar"
    )

best_carried = 1.0
for seed in range(40):
    predictions, actuals, ids = viral_book(0.3, seed=seed)
    best_carried = min(
        best_carried, carrier_ratio(predictions, actuals, ids)["worst_ratio"]
    )

check(
    worst_uniform <= EVENT_ROBUSTNESS_MAX_RATIO,
    f"a UNIFORM edge reached a ratio of {worst_uniform:.3f}, above the "
    f"{EVENT_ROBUSTNESS_MAX_RATIO} bar; honest books would be flagged",
)
check(
    best_carried > EVENT_ROBUSTNESS_MAX_RATIO,
    f"a GENUINE carrier fell to a ratio of {best_carried:.3f}, at or below "
    f"the {EVENT_ROBUSTNESS_MAX_RATIO} bar; a real carrier would pass",
)
check(
    worst_uniform < best_carried,
    f"the null ({worst_uniform:.3f}) and carrier ({best_carried:.3f}) "
    f"distributions now overlap; no bar can separate them and the statistic "
    f"must be revisited",
)


# --- 8. Structural refusals -------------------------------------------------------

for label, call in (
    ("an unattributed observation", lambda: group_by_item([1.0], [1.0], [None])),
    ("an empty attribution", lambda: group_by_item([1.0], [1.0], [""])),
    ("an empty book", lambda: group_by_item([], [], [])),
    (
        "two items",
        lambda: carrier_ratio([1.0] * 4, [1.0] * 4, ["a", "a", "b", "b"]),
    ),
    ("an unknown axis", lambda: attribution_of({"events": ["a"]}, "vibes")),
):
    try:
        call()
    except EventRobustnessError:
        continue
    failures.append(f"{label} was accepted; it must be refused")

thin = evaluate_event_robustness([1.0] * 4, [1.0] * 4, ["a", "a", "b", "b"], None)
check(
    thin["verdict"] == EVENT_ROBUSTNESS_NOT_EVALUATED
    and thin["reason_code"] == ER_REASON_TOO_FEW_ITEMS,
    f"a two-item book was {thin['verdict']} ({thin['reason_code']}); with two "
    f"items every removal halves the book",
)


# --- 9. Floors are REUSED, not invented -------------------------------------------

check(
    EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS == EVENT_MEMORY_MIN_ANALOGS,
    "X4 and E6 disagree on how few examples are too few to characterise an "
    "event",
)
check(
    EVENT_ROBUSTNESS_MIN_ITEMS >= 3,
    "fewer than three items cannot be judged",
)


# --- 10. Every report renders and passes its own contract ------------------------

reports = [("broad", broad), ("carried", both_axes), ("thin", thin)]
if runs:
    reports.append(("shipped", shipped))
for label, report in reports:
    problems = event_robustness_problems(report)
    if problems:
        failures.append(f"the {label} report failed its own contract: {problems}")
        break
    lines = render_event_robustness(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_event_robustness returned no lines for {label}")
        break


if failures:
    print("X4 EVENT ROBUSTNESS GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X4 event robustness gate: OK")
print(f"  event memories on disk            {len(memories)}  (context; gitignored)")
print(f"  shipped runs with attribution     0 of {len(runs)}  -> NOT_EVALUATED")
print(f"  viral event                       CARRIED at 20/30/50/60% share")
print(f"  uniform book                      {broad['verdict']}"
      f" (ratio {broad['worst_ratio']:.3f})")
print(f"  source carrier, event axis only   {events_only['verdict']}"
      f" (ratio {events_only['worst_ratio']:.3f}) - MISSED")
print(f"  source carrier, both axes         {both_axes['verdict']}"
      f" (ratio {both_axes['worst_ratio']:.3f} on {both_axes['worst_axis']})")
print(f"  measured gap                      null <= {worst_uniform:.3f}"
      f" < bar {EVENT_ROBUSTNESS_MAX_RATIO} < carrier {best_carried:.3f}")
sys.exit(0)
