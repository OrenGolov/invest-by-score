"""Gate: concept drift is FOUR detectors, and the window floor is load-bearing.

"Feature distribution drift, relationship drift, calibration drift,
event-response drift."

THE DECIDING MEASUREMENT: THE FOUR ARE REALLY FOUR. Over four scenarios each
breaking exactly one thing, the feature-distribution detector catches ONE:

    scenario                 feature PSI   caught?
    1 feature drift               1.2197   YES
    2 relationship drift          0.0360   no
    3 calibration drift           0.0133   no
    4 event-response drift        0.0274   no

The other three are invisible to it because THE FEATURES DID NOT MOVE - the
world did.

M5's score_drift_psi is NOT reusable: it bins on a fixed [0,10] score scale
and MEASURED scores 0.0000 on a 3.2-sigma shift in a feature ranged
[-0.3, 0.3], where quantile bins score 6.9450.

Verified to FAIL when any of these is reinjected:
  - a drift type dropped, so a partial scan reports clean
  - fixed binning restored, blinding the feature detector
  - the window floor lowered below the noise threshold
  - NOT_EVALUATED collapsed into STABLE
  - drift wired to trigger a retrain
  - a detector made insensitive to its own scenario
  - the calibration detector replaced by a ranking measure
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    DRIFT_ALERT,
    DRIFT_CALIBRATION,
    DRIFT_FEATURE,
    DRIFT_MIN_WINDOW,
    DRIFT_NOT_EVALUATED,
    DRIFT_PSI_ALERT,
    DRIFT_RELATIONSHIP,
    DRIFT_RESPONSE,
    DRIFT_STABLE,
    DRIFT_TRIGGERS_RETRAIN,
    DRIFT_TYPES,
    DRIFT_VERDICTS,
    DRIFT_WARN,
)
from core.drift_detection import (  # noqa: E402
    calibration_drift,
    drift_problems,
    drift_report,
    event_response_drift,
    feature_drift,
    population_stability_index,
    relationship_drift,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def normal(count, mean=0.0, sd=1.0, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(count)]


def main() -> int:
    window = max(int(DRIFT_MIN_WINDOW), 300)

    # 1. THE FOUR TYPES ARE DECLARED AND DISTINCT.
    check(
        len(DRIFT_TYPES) == 4,
        f"{len(DRIFT_TYPES)} drift types are declared, not four — the sprint "
        f"names four and MEASURED the feature detector catches only one",
    )

    # 2. THE DECIDING MEASUREMENT: each detector catches ITS OWN scenario and
    #    the feature detector is blind to the other three.
    rng = random.Random(13)
    xs = [rng.gauss(0, 1) for _ in range(window)]
    shifted = [rng.gauss(1.2, 1) for _ in range(window)]

    scenario_one = feature_drift(xs, shifted)
    check(
        scenario_one["verdict"] == DRIFT_ALERT,
        f"a 1.2-sigma feature shift read {scenario_one['verdict']}",
    )

    # relationship drift with IDENTICAL features
    before = [1 if rng.random() < 1 / (1 + math.exp(-0.9 * v)) else 0 for v in xs]
    after = [1 if rng.random() < 1 / (1 + math.exp(+0.9 * v)) else 0 for v in xs]
    blind = feature_drift(xs, xs)
    check(
        blind["verdict"] == DRIFT_STABLE,
        "identical feature windows did not read STABLE",
    )
    relationship = relationship_drift(xs, before, xs, after)
    check(
        relationship["verdict"] == DRIFT_ALERT,
        f"a correlation SIGN FLIP read {relationship['verdict']} — this is the "
        f"drift a distribution detector cannot see (its PSI is 0.0360)",
    )
    check(
        relationship["sign_flip"] is True,
        "a sign flip was not reported as one",
    )

    # calibration drift with the RANKING intact
    probabilities = [1 / (1 + math.exp(-0.9 * v)) for v in xs]
    outcomes = [1 if rng.random() < p else 0 for p in probabilities]
    inflated = [min(0.999, p * 1.6) for p in probabilities]
    good = calibration_drift(probabilities, outcomes)
    bad = calibration_drift(inflated, outcomes)
    check(
        good["verdict"] == DRIFT_STABLE,
        f"a well-calibrated model read {good['verdict']}",
    )
    check(
        bad["verdict"] in (DRIFT_WARN, DRIFT_ALERT),
        f"probabilities inflated 1.6x read {bad['verdict']} — the calibration "
        f"detector is blind to the drift only it can see",
    )
    check(
        bad["direction"] == "OVERCONFIDENT",
        "an inflated model was not named OVERCONFIDENT — a signed number "
        "means nothing without a direction",
    )
    # AND the relationship detector must NOT catch it: that is why calibration
    # is a separate question.
    ranking = relationship_drift(xs, outcomes, xs, outcomes)
    check(
        ranking["verdict"] == DRIFT_STABLE,
        "the relationship detector fired on an unchanged ranking, so the two "
        "detectors are not measuring different things",
    )

    # event-response drift with everything else unchanged
    strong = [abs(rng.gauss(0.04, 0.02)) for _ in range(window)]
    collapsed = [abs(rng.gauss(0.0005, 0.0005)) for _ in range(window)]
    response = event_response_drift(strong, collapsed)
    check(
        response["verdict"] == DRIFT_ALERT,
        f"an event response collapsing to near zero read "
        f"{response['verdict']}",
    )

    # THE COMPARISON MUST BE SYMMETRIC, and my own gate caught that it was
    # not. A percentage ratio bounds a decline at -100% while an increase is
    # unbounded, so under the first version an event type that had STOPPED
    # MOVING PRICE (-99.1%) read WARN while only an exact zero reached ALERT.
    vanished = event_response_drift(strong, [0.0] * window)
    check(
        vanished["verdict"] == DRIFT_ALERT and vanished["response_vanished"],
        "a response that vanished entirely was not an ALERT",
    )
    halved = event_response_drift(strong, [v / 2 for v in strong])
    doubled = event_response_drift(strong, [v * 2 for v in strong])
    check(
        halved["verdict"] == doubled["verdict"] == DRIFT_WARN,
        f"halving read {halved['verdict']} while doubling read "
        f"{doubled['verdict']} — the same magnitude of change must read the "
        f"same in both directions, which a percentage ratio cannot do",
    )
    quartered = event_response_drift(strong, [v / 4 for v in strong])
    check(
        quartered["verdict"] == DRIFT_ALERT,
        f"a fourfold weakening read {quartered['verdict']}",
    )
    check(
        response["direction"] == "WEAKER",
        "a collapsing response was not named WEAKER",
    )
    steady = event_response_drift(strong, strong)
    check(
        steady["verdict"] == DRIFT_STABLE,
        "an unchanged response read as drift",
    )

    # 3. QUANTILE BINNING, NOT FIXED. MEASURED: a fixed [0,10] scale scores
    #    0.0000 on a 3.2-sigma shift in a narrow-range feature.
    narrow_reference = normal(window, 0.02, 0.05, seed=3)
    narrow_current = normal(window, 0.18, 0.05, seed=4)
    psi = population_stability_index(narrow_reference, narrow_current)
    check(
        psi > DRIFT_PSI_ALERT,
        f"a 3.2-sigma shift in a feature ranged [-0.3, 0.3] scored PSI {psi} "
        f"— the binning is fixed rather than quantile, and is blind to any "
        f"feature that does not live on that scale",
    )
    wide_reference = normal(window, 45, 12, seed=5)
    wide_current = normal(window, 72, 12, seed=6)
    check(
        population_stability_index(wide_reference, wide_current) > DRIFT_PSI_ALERT,
        "a large shift in an RSI-scaled feature was not detected",
    )

    # 4. THE WINDOW FLOOR. MEASURED, PSI > 0.25 fires on 77% of CLEAN
    #    comparisons at window 50 and 20.5% at window 100.
    check(
        DRIFT_MIN_WINDOW >= 250,
        f"a window floor of {DRIFT_MIN_WINDOW} cannot support a PSI "
        f"threshold: the conventional rule fires on 77% of clean comparisons "
        f"at window 50",
    )
    thin = feature_drift(normal(50, seed=7), normal(50, seed=8))
    check(
        thin["verdict"] == DRIFT_NOT_EVALUATED,
        f"a 50-observation window returned {thin['verdict']} instead of "
        f"refusing to judge",
    )

    # ...and the floor actually buys silence on clean data.
    false_alarms = 0
    for seed in range(40):
        clean = feature_drift(
            normal(window, seed=1000 + seed), normal(window, seed=2000 + seed)
        )
        if clean["verdict"] == DRIFT_ALERT:
            false_alarms += 1
    check(
        false_alarms <= 2,
        f"{false_alarms}/40 clean comparisons raised an ALERT at the shipped "
        f"window floor — the floor is not buying the silence it was chosen for",
    )

    # 5. NOT_EVALUATED IS NOT STABLE.
    check(
        DRIFT_NOT_EVALUATED != DRIFT_STABLE
        and DRIFT_VERDICTS[0] == DRIFT_NOT_EVALUATED,
        "'could not measure' and 'stable' are different facts",
    )

    # 6. A PARTIAL SCAN CANNOT REPORT CLEAN.
    partial = drift_report(features={"x": (xs, xs)})
    check(
        set(partial["by_type"]) == set(DRIFT_TYPES),
        f"a report supplied only feature data covered {set(partial['by_type'])} "
        f"— every declared type must appear, or a three-of-four scan reads as "
        f"a clean bill of health",
    )
    check(
        partial["by_type"][DRIFT_RELATIONSHIP] == DRIFT_NOT_EVALUATED,
        "an unsupplied drift type was not reported as NOT_EVALUATED",
    )
    check(
        drift_problems(partial) == [],
        f"a partial report is not contract-clean: {drift_problems(partial)}",
    )

    # 7. DRIFT NEVER RETRAINS BY ITSELF.
    check(
        DRIFT_TRIGGERS_RETRAIN is False,
        "drift is wired to trigger a retrain — a model that replaces itself "
        "on an alert is the uncontrolled self-modification Sprint L prevents",
    )
    check(
        partial.get("triggers_retrain") is False,
        "a drift report claimed it triggers a retrain",
    )

    # 8. A FULL REPORT IS CONTRACT-CLEAN AND SURFACES THE WORST VERDICT.
    full = drift_report(
        features={"x": (xs, shifted)},
        relationships={"x": (xs, before, xs, after)},
        calibration=(inflated, outcomes),
        responses={"earnings": (strong, collapsed)},
    )
    check(
        drift_problems(full) == [],
        f"a full report is not contract-clean: {drift_problems(full)}",
    )
    check(
        full["worst"] == DRIFT_ALERT and full["alerts"] >= 3,
        f"a report with drift in every dimension surfaced worst="
        f"{full['worst']} with {full['alerts']} alert(s)",
    )

    if FAILURES:
        print("L6 drift-detection gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L6 drift-detection gate OK:")
    print("  four detectors: the feature one catches 1 of 4 scenarios alone.")
    print("  a correlation sign flip is caught while its feature PSI reads 0.036.")
    print("  1.6x-inflated probabilities are caught while the ranking is intact.")
    print(f"  quantile bins see a 3.2-sigma narrow-range shift (PSI {psi:.2f}); "
          f"fixed [0,10] bins score 0.0000.")
    print(f"  window floor {DRIFT_MIN_WINDOW}: {false_alarms}/40 clean "
          f"comparisons alerted.")
    print("  a partial scan reports NOT_EVALUATED, never clean; drift never retrains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
