"""X6 governance gate — an edge at one horizon is not an edge.

Exits 1 when any of these fails.

THE BLOCKING MEASUREMENT, RECOMPUTED HERE from the tracked training runs: every
run is 20d, so four of the five required horizons have no model. That is a DATA
gap, not a structural one — the labels, the `target_horizon` field and the
trainer argument all already exist.

THE THRESHOLD IS RECOMPUTED TOO, from seeded simulation rather than quoted, so
the 3-of-5 bar keeps being justified on every run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    LABEL_HORIZON_SESSIONS,
    OOS_ALPHA,
    OOS_MIN_OBSERVATIONS,
    TEMPORAL_ALPHA,
    TEMPORAL_FRAGILE,
    TEMPORAL_HORIZONS,
    TEMPORAL_MIN_AGREEING,
    TEMPORAL_MIN_OBSERVATIONS,
    TEMPORAL_NOT_EVALUATED,
    TEMPORAL_REQUIRES_SPREAD,
    TEMPORAL_ROBUST,
)
from core.oos_validation import sampling_band  # noqa: E402
from core.temporal_robustness import (  # noqa: E402
    TR_HORIZON_MISSING,
    TR_HORIZON_NO_EDGE,
    TR_REASON_MISSING_HORIZONS,
    TR_REASON_TOO_FEW_AGREE,
    TemporalRobustnessError,
    evaluate_temporal_robustness,
    horizon_edge,
    render_temporal,
    temporal_problems,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def sweep(accuracies, observations=120):
    return {
        horizon: {"directional_accuracy": accuracy, "observations": observations}
        for horizon, accuracy in zip(TEMPORAL_HORIZONS, accuracies)
    }


# --- 1. THE BLOCKING MEASUREMENT: every trained run is 20d -----------------------

runs = load_training_runs()
check(bool(runs), "no training runs are available; X6 cannot state its blocker")

trained = {r["target_horizon"] for r in runs}
missing_horizons = [h for h in TEMPORAL_HORIZONS if h not in trained]
check(
    len(missing_horizons) == 4,
    f"{len(missing_horizons)} horizon(s) lack a trained model, expected 4 "
    f"({sorted(trained)} present). If horizons were trained, that is the fix "
    f"X6 asks for and this gate's claim is stale - the real sweep must now run",
)

# Every required horizon must at least be LABELABLE, or the fix X6 names is
# not actually available and the blocker is worse than stated.
for horizon in TEMPORAL_HORIZONS:
    check(
        horizon in LABEL_HORIZON_SESSIONS,
        f"{horizon} has no label definition, so X6's 'just train it' next "
        f"action is wrong and the blocker is structural after all",
    )

if runs:
    shipped = evaluate_temporal_robustness(
        {"20d": runs[0]["metrics"]}, estimator=runs[0]["estimator"]
    )
    check(
        shipped["verdict"] == TEMPORAL_NOT_EVALUATED,
        f"the shipped state reported {shipped['verdict']}; with four horizons "
        f"untrained the only honest answer is NOT_EVALUATED",
    )
    check(
        shipped["reason_code"] == TR_REASON_MISSING_HORIZONS,
        f"the shipped state gave reason {shipped['reason_code']!r}",
    )
    check(
        len(shipped["missing_horizons"]) == 4,
        f"{len(shipped['missing_horizons'])} horizons reported missing, "
        f"expected 4",
    )


# --- 2. A MISSING HORIZON IS NOT A NEGATIVE RESULT -------------------------------

outcome, excess = horizon_edge(None, None)
check(
    outcome == TR_HORIZON_MISSING,
    f"an absent horizon was reported as {outcome!r}",
)
check(
    outcome != TR_HORIZON_NO_EDGE,
    "an absent horizon was read as showing no edge; that reports an untested "
    "horizon as a tested one",
)
check(
    excess is None,
    "an absent horizon carried a measured excess; there was nothing to measure",
)

empty = evaluate_temporal_robustness({})
check(
    sorted(empty["horizons"]) == sorted(TEMPORAL_HORIZONS),
    "not every required horizon appears in the report; a horizon nobody tests "
    "is a horizon whose edge is unproven",
)
check(
    all(
        entry["excess_over_coin_flip"] is None
        for entry in empty["horizons"].values()
    ),
    "an unevaluated horizon carried a measurement",
)


# --- 3. ONE STRONG HORIZON IS STILL NOT ROBUST -----------------------------------
# The 24.8% false-positive rate, enforced as a rule.

single = evaluate_temporal_robustness(
    {"20d": {"directional_accuracy": 0.95, "observations": 120}}
)
check(
    single["verdict"] != TEMPORAL_ROBUST,
    f"a single horizon at 0.95 accuracy was {single['verdict']}; MEASURED, a "
    f"noise model clears one horizon 24.80% of the time",
)


# --- 4. THE THRESHOLD, RECOMPUTED from seeded simulation -------------------------
# Not quoted: the 3-of-5 bar is re-justified on every run.

band = sampling_band(120, TEMPORAL_ALPHA)


def trial(seed: int, real_edge: bool) -> int:
    rng = np.random.default_rng(seed)
    cleared = 0
    for _ in range(len(TEMPORAL_HORIZONS)):
        outcome_series = rng.normal(0, 0.05, 120)
        predicted = (
            outcome_series * 0.35 + rng.normal(0, 0.03, 120)
            if real_edge
            else rng.normal(0, 0.05, 120)
        )
        accuracy = float(np.mean(np.sign(predicted) == np.sign(outcome_series)))
        if abs(accuracy - 0.5) > band:
            cleared += 1
    return cleared


null = np.array([trial(seed, False) for seed in range(600)])
real = np.array([trial(seed, True) for seed in range(600)])

false_positive_at_one = float((null >= 1).mean())
false_positive_at_bar = float((null >= TEMPORAL_MIN_AGREEING).mean())
detection_at_bar = float((real >= TEMPORAL_MIN_AGREEING).mean())

check(
    false_positive_at_one > 0.10,
    f"a noise model now clears one horizon only "
    f"{false_positive_at_one:.2%} of the time; the 24.8% figure that "
    f"justifies requiring more than one horizon is stale",
)
check(
    false_positive_at_bar < 0.01,
    f"at the {TEMPORAL_MIN_AGREEING}-horizon bar a noise model passes "
    f"{false_positive_at_bar:.2%} of the time, above 1%; the bar no longer "
    f"controls false positives",
)
check(
    detection_at_bar > 0.95,
    f"at the {TEMPORAL_MIN_AGREEING}-horizon bar a real edge is detected only "
    f"{detection_at_bar:.2%} of the time; the bar is costing real detection",
)
check(
    false_positive_at_bar < false_positive_at_one,
    "requiring more horizons did not reduce the false-positive rate, so the "
    "whole premise of the sweep is wrong",
)


# --- 5. THE GATE CAN PASS AND CAN FAIL -------------------------------------------

robust = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.52, 0.51]))
check(
    robust["verdict"] == TEMPORAL_ROBUST,
    f"three agreeing horizons were {robust['verdict']}; a gate that cannot "
    f"pass is a constant, not a test",
)
fragile = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.52, 0.51, 0.50]))
check(
    fragile["verdict"] == TEMPORAL_FRAGILE
    and fragile["reason_code"] == TR_REASON_TOO_FEW_AGREE,
    f"two agreeing horizons were {fragile['verdict']} "
    f"({fragile['reason_code']}), expected FRAGILE",
)
check(
    fragile["verdict"] != TEMPORAL_NOT_EVALUATED,
    "'tested and too few agreed' was collapsed into 'the sweep never "
    "happened'; they are different answers",
)


# --- 6. The spread rule was TESTED AND DROPPED -----------------------------------
# MEASURED over 2,000 null trials per arm, adjacent agreement (0.50%) and
# spread agreement (0.45%) are indistinguishable. Requiring spread would be an
# unmeasured threshold, so adjacent agreement must still count.

adjacent = evaluate_temporal_robustness(sweep([0.70, 0.68, 0.66, 0.51, 0.50]))
check(
    adjacent["verdict"] == TEMPORAL_ROBUST,
    f"three ADJACENT agreeing horizons were {adjacent['verdict']}; the spread "
    f"requirement was measured and dropped, so this must still pass",
)
check(
    not TEMPORAL_REQUIRES_SPREAD,
    "a spread requirement was reinstated; MEASURED, adjacent and spread "
    "agreement differ by 0.05 percentage points under the null",
)


# --- 7. Thin samples cannot decide, in either direction --------------------------

thin = evaluate_temporal_robustness(sweep([0.95] * 5, observations=OOS_MIN_OBSERVATIONS - 1))
check(
    thin["verdict"] == TEMPORAL_NOT_EVALUATED,
    f"five horizons at 0.95 accuracy on thin samples were {thin['verdict']}; "
    f"below {TEMPORAL_MIN_OBSERVATIONS} observations a horizon cannot decide",
)


# --- 8. Structural refusals -------------------------------------------------------

try:
    evaluate_temporal_robustness(
        {"252d": {"directional_accuracy": 0.7, "observations": 120}}
    )
except TemporalRobustnessError:
    pass
else:
    failures.append(
        "a horizon X6 does not evaluate was accepted; results for an "
        "unevaluated horizon cannot count toward agreement"
    )


# --- 9. Floors are REUSED, not invented ------------------------------------------

check(
    TEMPORAL_MIN_OBSERVATIONS == OOS_MIN_OBSERVATIONS,
    "X6 and X1 disagree on how few observations are too few; a horizon could "
    "be called robust on a sample X1 rejects",
)
check(
    TEMPORAL_ALPHA == OOS_ALPHA,
    "X6 does not test at X1's alpha; a horizon could pass on a looser "
    "standard than the whole model is held to",
)


# --- 10. A conclusion cannot absorb untested horizons ----------------------------

forged = dict(robust)
forged["missing_horizons"] = ["1d"]
check(
    any("never evaluated" in p for p in temporal_problems(forged)),
    "the contract check accepted a ROBUST verdict while a horizon was never "
    "evaluated; that is how an untested horizon gets absorbed into a "
    "conclusion",
)

for label, report in (("robust", robust), ("fragile", fragile), ("empty", empty)):
    problems = temporal_problems(report)
    if problems:
        failures.append(f"the {label} report failed its own contract: {problems}")
        break
    lines = render_temporal(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_temporal returned no lines for {label}")
        break

if runs:
    problems = temporal_problems(shipped)
    check(
        problems == [],
        f"the shipped NOT_EVALUATED report failed its own contract: {problems}",
    )


if failures:
    print("X6 TEMPORAL ROBUSTNESS GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X6 temporal robustness gate: OK")
print(f"  horizons required            {len(TEMPORAL_HORIZONS)}"
      f"  ({', '.join(TEMPORAL_HORIZONS)})")
print(f"  horizons with a model        {len(trained)}  ({', '.join(sorted(trained))})")
print(f"  shipped verdict              {shipped['verdict'] if runs else 'n/a'}")
print(f"  false-positive at 1 horizon  {false_positive_at_one:.2%}  <- why one is not enough")
print(f"  false-positive at {TEMPORAL_MIN_AGREEING} horizons "
      f"{false_positive_at_bar:.2%}   detection {detection_at_bar:.2%}")
sys.exit(0)
