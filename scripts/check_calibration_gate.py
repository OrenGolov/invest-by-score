"""X2 governance gate — an in-sample ECE of zero is not calibration.

Exits 1 when any of these fails.

THE DECIDING MEASUREMENT, RECOMPUTED HERE from the tracked training runs rather
than quoted: fitting the isotonic map on all 120 observations and scoring the
same 120 gives ECE exactly 0.0000 for every estimator. Refitting on the first 60
and scoring the held-out 60 gives 0.166-0.189. The first number is not
optimistic — it is structurally incapable of being anything else.

`data/training_runs.jsonl` is TRACKED, so this runs identically on a fresh clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.calibration import (  # noqa: E402
    calibrated_probability,
    expected_calibration_error,
    fit_calibration,
)
from core.calibration_gate import (  # noqa: E402
    CAL_REASON_ECE,
    CAL_REASON_IN_SAMPLE,
    CAL_REASON_MCE,
    CAL_REASON_THIN_HOLDOUT,
    CalibrationGateError,
    calibration_problems,
    evaluate_calibration,
    is_in_sample,
    noise_floor,
    render_calibration,
)
from core.config import (  # noqa: E402
    CALIBRATION_GATE_APPROVED,
    CALIBRATION_GATE_MAX_ECE,
    CALIBRATION_GATE_MAX_MCE,
    CALIBRATION_GATE_MIN_HOLDOUT,
    CALIBRATION_GATE_NOT_APPROVED,
    CALIBRATION_GATE_NOT_EVALUATED,
    DRIFT_CALIBRATION_GAP,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def well_calibrated(n=400, seed=5):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.random(n) < probs).astype(float)
    return list(probs), list(outcomes)


def out_of_sample(n):
    return {"fit_indices": range(10_000, 10_000 + n), "score_indices": range(n)}


# --- 1. THE MEASUREMENT: in-sample ECE is exactly zero, recomputed ----------------

runs = {r["estimator"]: r for r in load_training_runs()}
check(bool(runs), "no training runs are available; X2 cannot measure anything")

in_sample_eces: dict[str, float] = {}
holdout_eces: dict[str, float] = {}

for name, run in sorted(runs.items()):
    fold = run["folds"][0]
    predictions = np.array(fold["predictions"])
    actuals = np.array(fold["actuals"])

    fitted = fit_calibration(predictions, actuals)
    probs = [calibrated_probability(float(x), fitted) for x in predictions]
    in_sample_eces[name] = expected_calibration_error(probs, actuals)

    half = len(predictions) // 2
    fitted_half = fit_calibration(predictions[:half], actuals[:half])
    held = [calibrated_probability(float(x), fitted_half) for x in predictions[half:]]
    holdout_eces[name] = expected_calibration_error(held, actuals[half:])

if in_sample_eces:
    worst_in_sample = max(in_sample_eces.values())
    check(
        worst_in_sample < 1e-6,
        f"in-sample ECE is no longer ~0 (worst {worst_in_sample:.6f}); the "
        f"measurement this gate rests on has changed and X2's reasoning must "
        f"be revisited",
    )

    best_holdout = min(holdout_eces.values())
    check(
        best_holdout > CALIBRATION_GATE_MAX_ECE,
        f"the best honest holdout ECE is now {best_holdout:.4f}, at or under "
        f"the {CALIBRATION_GATE_MAX_ECE} bar. That is a REAL IMPROVEMENT and "
        f"this gate's quoted 0.166-0.189 is stale - X2's reasoning must be "
        f"revisited rather than left asserting nothing is calibrated",
    )

    # The gap between the two numbers is the whole point of the task.
    for name in sorted(in_sample_eces):
        gap = holdout_eces[name] - in_sample_eces[name]
        check(
            gap > 0.10,
            f"{name}: the honest holdout ECE ({holdout_eces[name]:.4f}) is only "
            f"{gap:.4f} above the in-sample {in_sample_eces[name]:.4f}; the "
            f"in-sample number is supposed to be unable to detect anything",
        )


# --- 2. An in-sample result is NOT_EVALUATED, and never scores --------------------

probs, outcomes = well_calibrated(CALIBRATION_GATE_MIN_HOLDOUT)
in_sample_report = evaluate_calibration(
    probs,
    outcomes,
    fit_indices=range(CALIBRATION_GATE_MIN_HOLDOUT),
    score_indices=range(CALIBRATION_GATE_MIN_HOLDOUT),
)
check(
    in_sample_report["verdict"] == CALIBRATION_GATE_NOT_EVALUATED,
    f"an in-sample calibration was {in_sample_report['verdict']}, expected "
    f"NOT_EVALUATED; MEASURED, its ECE is 0.0000 by construction",
)
check(
    in_sample_report["reason_code"] == CAL_REASON_IN_SAMPLE,
    f"an in-sample calibration gave reason {in_sample_report['reason_code']!r}",
)
check(
    in_sample_report.get("ece") is None,
    "an in-sample calibration REPORTED an ECE; it must not even be computed, "
    "or 0.0000 can leak into a comparison as the best score of all",
)
check(
    in_sample_report["verdict"] != CALIBRATION_GATE_APPROVED,
    "an in-sample calibration was APPROVED",
)


# --- 3. Unknown provenance is NOT out-of-sample -----------------------------------

unknown = evaluate_calibration(probs, outcomes)
check(
    unknown["verdict"] == CALIBRATION_GATE_NOT_EVALUATED,
    f"a calibration with UNKNOWN provenance was {unknown['verdict']}; nobody "
    f"recorded where it came from, which is not the same as out-of-sample",
)
check(
    is_in_sample(None, range(60)) is None,
    "unknown provenance reported a boolean; it is a third state",
)
check(
    is_in_sample(range(60), range(60, 120)) is False,
    "disjoint index sets were not recognised as out-of-sample",
)
check(
    is_in_sample(range(100), range(50, 150)) is True,
    "overlapping index sets were not recognised as in-sample",
)


# --- 4. The gate CAN approve, or refusing proves nothing --------------------------

good_probs, good_outcomes = well_calibrated(400)
approved = evaluate_calibration(good_probs, good_outcomes, **out_of_sample(400))
check(
    approved["verdict"] == CALIBRATION_GATE_APPROVED,
    f"a well-calibrated 400-observation holdout was {approved['verdict']}; a "
    f"gate that cannot approve is a constant, not a test",
)
check(
    calibration_problems(approved) == [],
    f"the approved report failed its own contract: {calibration_problems(approved)}",
)


# --- 5. Overconfidence is caught --------------------------------------------------

shifted = [min(1.0, p + 0.25) for p in good_probs]
bad = evaluate_calibration(shifted, good_outcomes, **out_of_sample(400))
check(
    bad["verdict"] == CALIBRATION_GATE_NOT_APPROVED,
    f"probabilities shifted +0.25 were {bad['verdict']}, expected NOT_APPROVED",
)
check(
    bad["reason_code"] == CAL_REASON_ECE,
    f"an overconfident model gave reason {bad['reason_code']!r}",
)


# A SECOND overconfidence probe, deliberately MILD. The +0.25 shift above trips
# BOTH bars, so it cannot detect the ECE bar being loosened to the MCE bar. This
# one lands strictly between them: ECE 0.1188 fails while MCE 0.1548 passes.

mild = [min(1.0, p + 0.12) for p in good_probs]
mild_report = evaluate_calibration(mild, good_outcomes, **out_of_sample(400))
check(
    mild_report["mce"] <= CALIBRATION_GATE_MAX_MCE,
    f"the mild probe's worst bin ({mild_report['mce']:.4f}) now exceeds the "
    f"{CALIBRATION_GATE_MAX_MCE} bar, so it can no longer isolate the ECE bar",
)
check(
    mild_report["verdict"] == CALIBRATION_GATE_NOT_APPROVED
    and mild_report["reason_code"] == CAL_REASON_ECE,
    f"an ECE of {mild_report['ece']:.4f} - above the "
    f"{CALIBRATION_GATE_MAX_ECE} bar but below the worst-bin bar - was "
    f"{mild_report['verdict']} ({mild_report['reason_code']}); the ECE bar is "
    f"not being enforced at its own level",
)


# --- 5b. ECE is count-weighted, so the worst bin must be checked separately ------
# THE CASE MCE EXISTS FOR: a small region that is completely wrong hides inside
# an acceptable average. MEASURED below - ECE 0.0437 passes the 0.10 bar while
# one bin is off by 0.95.

_rng = np.random.default_rng(9)
_good_n = 395
_good = _rng.uniform(0.4, 0.6, _good_n)
_hidden_probs = list(_good) + [0.95] * 5
_hidden_outcomes = list((_rng.random(_good_n) < _good).astype(float)) + [0.0] * 5
hidden = evaluate_calibration(
    _hidden_probs, _hidden_outcomes, **out_of_sample(len(_hidden_probs))
)
check(
    hidden["ece"] <= CALIBRATION_GATE_MAX_ECE,
    f"the hidden-region probe no longer has a passing ECE ({hidden['ece']:.4f}); "
    f"it cannot test the worst-bin rule unless the average passes first",
)
check(
    hidden["mce"] > CALIBRATION_GATE_MAX_MCE,
    f"the hidden-region probe no longer has a failing worst bin "
    f"({hidden['mce']:.4f}); the probe is inert",
)
check(
    hidden["verdict"] == CALIBRATION_GATE_NOT_APPROVED,
    f"a model whose ECE is {hidden['ece']:.4f} but whose worst bin is off by "
    f"{hidden['mce']:.4f} was {hidden['verdict']}; ECE is count-weighted and "
    f"averaged that region away",
)
check(
    hidden["reason_code"] == CAL_REASON_MCE,
    f"the hidden-region model gave reason {hidden['reason_code']!r}, expected "
    f"{CAL_REASON_MCE!r}",
)


# --- 6. A thin holdout cannot decide, in EITHER direction -------------------------
# Below the noise floor is not a pass: it means the sample cannot tell.

thin_probs, thin_outcomes = well_calibrated(60)
thin = evaluate_calibration(thin_probs, thin_outcomes, **out_of_sample(60))
check(
    thin["verdict"] == CALIBRATION_GATE_NOT_EVALUATED,
    f"a 60-observation holdout was {thin['verdict']}; MEASURED, a perfectly "
    f"calibrated model still posts up to {noise_floor(60):.4f} at that size",
)
check(
    thin["reason_code"] == CAL_REASON_THIN_HOLDOUT,
    f"a thin holdout gave reason {thin['reason_code']!r}",
)

# The trap in its purest form: a tiny sample with a PERFECT ECE.
perfect_but_tiny = evaluate_calibration(
    [0.5] * 20, [1.0] * 10 + [0.0] * 10, **out_of_sample(20)
)
check(
    perfect_but_tiny["verdict"] != CALIBRATION_GATE_APPROVED,
    "a 20-observation holdout with a perfect ECE was APPROVED; a sample that "
    "small cannot demonstrate calibration no matter what number it posts",
)


# --- 7. The noise floor is measured, and it is why 200 is the minimum -------------

check(
    abs(noise_floor(60) - 0.1333) < 1e-4,
    f"the n=60 noise floor is {noise_floor(60):.4f}, not the measured 0.1333",
)
check(
    noise_floor(60) > CALIBRATION_GATE_MAX_ECE,
    "the n=60 noise floor no longer exceeds the ECE bar, which is the entire "
    "reason a 60-observation holdout cannot decide",
)
check(
    noise_floor(CALIBRATION_GATE_MIN_HOLDOUT) < CALIBRATION_GATE_MAX_ECE,
    f"the noise floor at the {CALIBRATION_GATE_MIN_HOLDOUT}-observation "
    f"minimum is not below the {CALIBRATION_GATE_MAX_ECE} bar, so even the "
    f"minimum holdout could not decide",
)
for n, measured in ((100, 0.1000), (150, 0.0800), (200, 0.0700), (500, 0.0440)):
    check(
        abs(noise_floor(n) - measured) < 0.01,
        f"the noise floor at n={n} is {noise_floor(n):.4f}, more than 0.01 from "
        f"the simulated {measured}",
    )


# --- 8. The bars are REUSED, not invented ----------------------------------------

check(
    CALIBRATION_GATE_MAX_ECE == DRIFT_CALIBRATION_GAP,
    f"the ECE bar {CALIBRATION_GATE_MAX_ECE} is not L6's measured drift "
    f"threshold {DRIFT_CALIBRATION_GAP}; a gate looser than the detector that "
    f"retires a model would approve something already known to be drifting",
)
check(
    CALIBRATION_GATE_MAX_MCE > CALIBRATION_GATE_MAX_ECE,
    "the worst-bin bar does not exceed the average bar, so MCE adds nothing",
)


# --- 9. A missing probability is never 0.5 ---------------------------------------

try:
    evaluate_calibration([0.6, None], [1.0, 0.0], **out_of_sample(2))
except CalibrationGateError:
    pass
else:
    failures.append(
        "a missing probability was scored; coercing it to 0.5 turns 'the model "
        "declined to predict' into 'the model predicted a coin flip', "
        "manufacturing calibration evidence from silence"
    )

try:
    evaluate_calibration([0.5], [1.0, 0.0], **out_of_sample(1))
except CalibrationGateError:
    pass
except Exception as exc:  # a library error, not the contract's own refusal
    failures.append(
        f"mismatched counts raised {type(exc).__name__} from underneath "
        f"instead of a CalibrationGateError naming the problem: {exc}"
    )
else:
    failures.append(
        "mismatched probability and outcome counts were scored; silently "
        "zipping them drops observations without saying so"
    )

absent = evaluate_calibration(None, None)
check(
    absent["verdict"] == CALIBRATION_GATE_NOT_EVALUATED,
    "absent probabilities were given a pass/fail verdict",
)


# --- 9b. The contract check itself must catch a forged APPROVED ------------------
# The report is what a caller reads. A tampered or buggy producer that marks an
# in-sample result APPROVED must be caught by the contract check, not only by
# the producer that would never have emitted it.

forged = dict(approved)
forged["in_sample"] = None
check(
    any("out-of-sample provenance" in p for p in calibration_problems(forged)),
    "the contract check accepted an APPROVED report with UNKNOWN provenance; "
    "it is the last line of defence when a producer is wrong",
)

forged_in_sample = dict(approved)
forged_in_sample["in_sample"] = True
check(
    any("out-of-sample provenance" in p for p in calibration_problems(forged_in_sample)),
    "the contract check accepted an APPROVED report that was scored IN SAMPLE",
)

forged_ece = dict(approved)
forged_ece["ece"] = 0.5
check(
    any("above the" in p for p in calibration_problems(forged_ece)),
    "the contract check accepted an APPROVED report whose ECE is above the bar",
)

forged_thin = dict(approved)
forged_thin["observations"] = 60
check(
    any("sampling noise" in p for p in calibration_problems(forged_thin)),
    "the contract check accepted an APPROVED report on a 60-observation holdout",
)


# --- 10. Every report renders and passes its own contract ------------------------

for label, report in (
    ("approved", approved),
    ("overconfident", bad),
    ("thin", thin),
    ("in-sample", in_sample_report),
    ("absent", absent),
):
    problems = calibration_problems(report)
    if problems:
        failures.append(f"the {label} report failed its own contract: {problems}")
        break
    lines = render_calibration(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_calibration returned no lines for {label}")
        break


if failures:
    print("X2 CALIBRATION GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X2 calibration gate: OK")
if in_sample_eces:
    print(f"  estimators measured        {len(in_sample_eces)}")
    print(f"  worst in-sample ECE        {max(in_sample_eces.values()):.6f}  (meaningless by construction)")
    print(f"  best honest holdout ECE    {min(holdout_eces.values()):.4f}  (bar {CALIBRATION_GATE_MAX_ECE})")
print(f"  noise floor at n=60        {noise_floor(60):.4f}  - above the bar, so n=60 cannot decide")
print(f"  minimum holdout            {CALIBRATION_GATE_MIN_HOLDOUT}")
print("  the gate CAN approve       verified on a well-calibrated holdout")
sys.exit(0)
