"""Gate: regime specialization must be EARNED, per regime, on repeated OOS evidence.

"Evaluate separate models per regime (bullish/bearish/range/risk-off/stress)
ONLY IF OOS evidence supports specialization."

The last clause is the whole task, and the system must be able to answer NO.

THE DECIDING MEASUREMENT: SPECIALIZATION LOSES ON THIN DATA EVEN WHEN THE
SKILL GENUINELY DIFFERS. Brier on held-out data, skill differing by regime:
pooled 0.24913 vs specialized 0.25808 at 100 observations. Pooling only stops
winning at ~300. When skill does NOT differ, pooling wins at every size.

A BARE OOS WIN IS NOT EVIDENCE: MEASURED with no real difference anywhere,
specialization still wins 14-23% of single comparisons. A margin plus a fold
majority cuts that to 2-4% while real specialization is still found.

Verified to FAIL when any of these is reinjected:
  - the default flipped from pooled to specialized
  - the margin removed, so a bare win adopts
  - the margin raised past the knee, rejecting real specialization
  - the cell floor dropped below what supports a rate
  - the fold majority reduced to a minority
  - INSUFFICIENT_DATA collapsed into POOLED
  - a regime left unserved instead of falling back to pooled
  - a regime vocabulary of its own instead of the governed labels
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    REGIME_LABELS,
    REGIME_SPEC_INSUFFICIENT,
    REGIME_SPEC_NOT_EVALUATED,
    REGIME_SPEC_POOLED,
    REGIME_SPEC_SPECIALIZED,
    REGIME_SPECIALIZATION_DEFAULT_POOLED,
    REGIME_SPECIALIZATION_FALLBACK_POOLED,
    REGIME_SPECIALIZATION_FOLDS,
    REGIME_SPECIALIZATION_FOLDS_REQUIRED,
    REGIME_SPECIALIZATION_LABELS,
    REGIME_SPECIALIZATION_MARGIN,
    REGIME_SPECIALIZATION_MIN_CELL,
    REGIME_SPECIALIZATION_PER_REGIME,
    REGIME_SPECIALIZATION_VERDICTS,
)
from core.regime_specialization import (  # noqa: E402
    RegimeSpecializationError,
    evaluate_regime,
    render_specialization,
    specialization_problems,
    specialization_report,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def folds(count, size, pooled_p, specialized_p, true_p, seed):
    rng = random.Random(seed)
    built = []
    for _ in range(count):
        outcomes = [1 if rng.random() < true_p else 0 for _ in range(size)]
        built.append(
            {
                "pooled": [pooled_p] * size,
                "specialized": [specialized_p] * size,
                "outcomes": outcomes,
            }
        )
    return built


def main() -> int:
    # 1. THE GOVERNED VOCABULARY, NOT A SECOND ONE (W5).
    check(
        set(REGIME_SPECIALIZATION_LABELS) == set(REGIME_LABELS),
        "L7 is specializing on labels other than the governed five-state "
        "classifier — a second regime vocabulary is the split-brain W5 forbids",
    )
    try:
        specialization_report({"euphoria": []})
        FAILURES.append(
            "an invented regime label was accepted — L7 must specialize on the "
            "governed labels only"
        )
    except RegimeSpecializationError:
        pass

    # 2. THE DEFAULT IS POOLED, and it must be able to answer NO.
    check(
        REGIME_SPECIALIZATION_DEFAULT_POOLED,
        "the default is not pooled. MEASURED, when skill does not differ by "
        "regime, pooling wins at every sample size",
    )
    no_difference = evaluate_regime(
        "range",
        folds(REGIME_SPECIALIZATION_FOLDS, 200, 0.55, 0.55, 0.55, seed=11),
    )
    check(
        no_difference["verdict"] == REGIME_SPEC_POOLED,
        f"a regime with NO real difference read "
        f"{no_difference['verdict']} — the system cannot answer NO",
    )

    # 3. REAL SPECIALIZATION IS STILL FOUND. Strictness must be a filter, not
    #    a gag.
    real = evaluate_regime(
        "bullish",
        folds(REGIME_SPECIALIZATION_FOLDS, 200, 0.55, 0.70, 0.70, seed=12),
    )
    check(
        real["verdict"] == REGIME_SPEC_SPECIALIZED,
        f"a genuinely better per-regime model read {real['verdict']} — the "
        f"rule is a gag rather than a filter",
    )
    check(
        (real.get("median_gain") or 0) > 0,
        "a SPECIALIZED verdict carried no measured gain",
    )

    # 4. A BARE WIN IS NOT ENOUGH. MEASURED, no-difference regimes win
    #    14-23% of single comparisons, so noise must not adopt.
    #
    # THE NOISE MUST BE REALISTIC. A first version of this check compared a
    # pooled and a specialized prediction that were IDENTICAL (0.55 vs 0.55),
    # so the gain was exactly zero and no fold could ever be won whatever the
    # rule was. It passed against a gate with the margin removed AND against
    # one where a single fold sufficed — it was testing nothing. The
    # specialized model must differ the way a real one does: fitted on a
    # finite sample, so it lands near but not on the pooled rate.
    #
    # 200 TRIALS, NOT 40. The margin and the fold majority each do independent
    # work, and a 40-trial check could not see it: MEASURED at 4-of-5 folds,
    # dropping the margin takes noise adoption from 0/400 to 23/400 (5.75%),
    # which 40 trials would miss most of the time. The larger sample is what
    # makes this check able to fail when the margin is removed.
    noise_adoptions = 0
    for seed in range(200):
        rng = random.Random(4000 + seed)
        noisy = []
        for _ in range(REGIME_SPECIALIZATION_FOLDS):
            outcomes = [1 if rng.random() < 0.55 else 0 for _ in range(150)]
            # a per-regime rate estimated from its own finite sample
            fitted = statistics.mean(
                [1 if rng.random() < 0.55 else 0 for _ in range(150)]
            )
            noisy.append(
                {
                    "pooled": [0.55] * 150,
                    "specialized": [fitted] * 150,
                    "outcomes": outcomes,
                }
            )
        if evaluate_regime("bearish", noisy)["verdict"] == REGIME_SPEC_SPECIALIZED:
            noise_adoptions += 1
    check(
        noise_adoptions <= 2,
        f"{noise_adoptions}/200 no-difference regimes were adopted as "
        f"SPECIALIZED — the margin and fold majority are not holding. "
        f"MEASURED, the shipped pair gives 0/400 while dropping the margin "
        f"alone gives 23/400",
    )
    check(
        REGIME_SPECIALIZATION_MARGIN > 0,
        "the margin is zero, so a bare OOS win adopts",
    )
    check(
        REGIME_SPECIALIZATION_FOLDS_REQUIRED * 2 > REGIME_SPECIALIZATION_FOLDS,
        f"requiring {REGIME_SPECIALIZATION_FOLDS_REQUIRED} of "
        f"{REGIME_SPECIALIZATION_FOLDS} folds is a minority — a result that "
        f"fails most of its own folds has not repeated",
    )

    # 5. THE RARE-REGIME PROBLEM. Stress is ~5% of observations; a rate from
    #    ~50 carries a typical error of 0.056 against a 0.08 effect.
    check(
        REGIME_SPECIALIZATION_MIN_CELL >= 60,
        f"a cell floor of {REGIME_SPECIALIZATION_MIN_CELL} cannot support a "
        f"per-regime rate",
    )
    thin = evaluate_regime(
        "stress", folds(REGIME_SPECIALIZATION_FOLDS, 20, 0.55, 0.70, 0.70, seed=13)
    )
    check(
        thin["verdict"] == REGIME_SPEC_INSUFFICIENT,
        f"a regime with 20 observations per fold read {thin['verdict']} — a "
        f"rare regime must be refused, not specialized on noise",
    )
    # ...and INSUFFICIENT is not POOLED: they are different facts.
    check(
        REGIME_SPEC_INSUFFICIENT != REGIME_SPEC_POOLED,
        "'too little data to test' and 'tested, pooling wins' collapsed into "
        "one verdict",
    )
    untested = evaluate_regime("stress", [])
    check(
        untested["verdict"] == REGIME_SPEC_NOT_EVALUATED,
        f"an untested regime read {untested['verdict']} rather than "
        f"NOT_EVALUATED",
    )

    # 6. PER REGIME, NOT ALL-OR-NOTHING. MEASURED, a common regime with a real
    #    difference adopts at 63% while a rare one adopts at 1% on the same
    #    data — an all-or-nothing rule holds the first hostage to the second.
    check(
        REGIME_SPECIALIZATION_PER_REGIME,
        "specialization is not decided per regime",
    )
    mixed = specialization_report(
        {
            "bullish": folds(REGIME_SPECIALIZATION_FOLDS, 200, 0.55, 0.70, 0.70, seed=14),
            "stress": folds(REGIME_SPECIALIZATION_FOLDS, 20, 0.55, 0.70, 0.70, seed=15),
        }
    )
    check(
        mixed["verdicts"]["bullish"] == REGIME_SPEC_SPECIALIZED,
        "a well-evidenced regime was denied its own model because a rare "
        "regime could not be tested",
    )
    check(
        mixed["verdicts"]["stress"] == REGIME_SPEC_INSUFFICIENT,
        "a rare regime was specialized despite insufficient evidence",
    )

    # 7. EVERY REGIME IS SERVED. An unserved regime removes coverage exactly
    #    where it matters most.
    check(
        REGIME_SPECIALIZATION_FALLBACK_POOLED,
        "a regime that does not earn specialization must fall back to pooled",
    )
    check(
        set(mixed["serving"]) == set(REGIME_SPECIALIZATION_LABELS),
        f"only {sorted(mixed['serving'])} are served — every governed regime "
        f"must have a model",
    )
    check(
        mixed["serving"]["stress"] == "pooled",
        "a regime that failed to specialize is not served by the pooled model",
    )
    check(
        mixed["serving"]["bullish"] == "specialized",
        "a regime that earned specialization is not served by its own model",
    )

    # 8. COVERAGE AND CONTRACT.
    check(
        set(mixed["verdicts"]) == set(REGIME_SPECIALIZATION_LABELS),
        "the report does not cover every governed regime",
    )
    check(
        specialization_problems(mixed) == [],
        f"a report is not contract-clean: {specialization_problems(mixed)}",
    )
    empty = specialization_report()
    check(
        specialization_problems(empty) == [],
        f"an empty report is not contract-clean: {specialization_problems(empty)}",
    )
    check(
        all(v == REGIME_SPEC_NOT_EVALUATED for v in empty["verdicts"].values()),
        "an empty report did not report every regime as NOT_EVALUATED",
    )
    check(
        len(render_specialization(mixed)) == len(REGIME_SPECIALIZATION_LABELS),
        "the rendering does not carry one line per regime",
    )
    check(
        REGIME_SPECIALIZATION_VERDICTS[0] == REGIME_SPEC_NOT_EVALUATED
        and REGIME_SPECIALIZATION_VERDICTS[-1] == REGIME_SPEC_SPECIALIZED,
        "the verdicts do not run weakest to strongest",
    )

    if FAILURES:
        print("L7 regime-specialization gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L7 regime-specialization gate OK:")
    print("  the default is POOLED and the system can answer NO.")
    print(f"  {noise_adoptions}/200 no-difference regimes were adopted "
          f"(margin {REGIME_SPECIALIZATION_MARGIN:.1%}, "
          f"{REGIME_SPECIALIZATION_FOLDS_REQUIRED} of "
          f"{REGIME_SPECIALIZATION_FOLDS} folds).")
    print("  a genuinely better per-regime model is still adopted.")
    print(f"  a rare regime below {REGIME_SPECIALIZATION_MIN_CELL} per fold is "
          f"refused, not specialized on noise.")
    print("  decided per regime; every governed regime is served, pooled if unearned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
