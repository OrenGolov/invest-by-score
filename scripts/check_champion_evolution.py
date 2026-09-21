"""Gate: champion replacement needs all seven conditions, and history is immutable.

"Replacement requires credible OOS improvement, acceptable calibration, no
unacceptable false-positive degradation, acceptable risk, regime robustness,
reproducibility, governance approval. Historical forecasts remain immutable."

Most conditions have an owner and are DELEGATED (W5). Three gaps were real.

GAP 1 - "CREDIBLE" WAS UNQUALIFIED. MEASURED, two models of IDENTICAL skill:
at n=250 the challenger beats the champion by 5.6 points on 10% of
comparisons and 10.4 points on 1%. "It improved out of sample" is not
credible evidence on its own.

GAP 2 - RISK WAS NEVER MEASURED AT PROMOTION. With the hit rate held FIXED
and only the SIZE of losing moves changed, a challenger with the SAME hit
rate carried a 28x worse drawdown while losing money, and one with a BETTER
hit rate still lost money. An accuracy-only gate promotes both.

GAP 3 - FALSE POSITIVES ARE NOT THE VETO RATE. A challenger matching the
champion on OVERALL accuracy (0.594 vs 0.597) moved the acted-on
false-positive rate 0.349 -> 0.551.

Verified to FAIL when any of these is reinjected:
  - a condition dropped, so six of seven suffices
  - the credibility bar removed, so a bare OOS win promotes
  - the OOS sample floor lowered below what supports the test
  - the risk condition removed or its bound widened
  - the false-positive bound widened
  - NOT_EVALUATED treated as passing
  - a delegated condition recomputed instead of relayed
  - historical forecasts editable or deletable
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CHAMPION_EVOLUTION_CONDITIONS,
    CHAMPION_EVOLUTION_CREDIBILITY_SIGMA,
    CHAMPION_EVOLUTION_HISTORY_IMMUTABLE,
    CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO,
    CHAMPION_EVOLUTION_MAX_FP_INCREASE,
    CHAMPION_EVOLUTION_MIN_ACTED_CALLS,
    CHAMPION_EVOLUTION_MIN_OOS_SAMPLE,
    CHAMPION_EVOLUTION_OWNED,
    CHAMPION_EVOLUTION_REQUIRED,
    EVO_FAIL,
    EVO_FALSE_POSITIVE,
    EVO_GOVERNANCE,
    EVO_NOT_EVALUATED,
    EVO_OOS,
    EVO_PASS,
    EVO_RISK,
)
from core.champion_evolution import (  # noqa: E402
    credibility,
    evaluate_replacement,
    evolution_problems,
    history_immutability_problems,
    max_drawdown,
    render_evolution,
    risk_condition,
)

FAILURES: list[str] = []
OK = {"outcome": EVO_PASS, "reason": "ok"}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def series(count, hit, loss_size, seed):
    """Hit rate held FIXED; only the SIZE of losing moves changes."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        if rng.random() < hit:
            out.append(abs(rng.gauss(0.010, 0.004)))
        else:
            out.append(-abs(rng.gauss(0.010 * loss_size, 0.004 * loss_size)))
    return out


def calls(true_positives, false_positives):
    return {"true_positives": true_positives, "false_positives": false_positives}


def replacement(**overrides):
    kwargs = {
        "oos_comparison": {
            "candidate_value": 0.62,
            "incumbent_value": 0.55,
            "sample": 2000,
        },
        "incumbent_returns": series(500, 0.58, 1.0, 1),
        "candidate_returns": series(500, 0.62, 1.0, 2),
        "incumbent_calls": calls(130, 70),
        "candidate_calls": calls(140, 60),
        "calibration": OK,
        "regime": OK,
        "reproducibility": OK,
        "governance": OK,
    }
    kwargs.update(overrides)
    return evaluate_replacement(**kwargs)


def main() -> int:
    # 1. SEVEN CONDITIONS, ALL REQUIRED.
    check(
        len(CHAMPION_EVOLUTION_CONDITIONS) == 7,
        f"{len(CHAMPION_EVOLUTION_CONDITIONS)} conditions are declared, not "
        f"seven",
    )
    check(
        set(CHAMPION_EVOLUTION_REQUIRED) == set(CHAMPION_EVOLUTION_CONDITIONS),
        "not every condition is required — the sprint lists them with 'and'",
    )
    clean = replacement()
    check(
        clean["may_replace"] is True,
        f"a replacement meeting every condition was blocked by "
        f"{clean.get('blocked_by')} — nothing could ever be promoted",
    )
    check(
        set(clean["outcomes"]) == set(CHAMPION_EVOLUTION_CONDITIONS),
        "the report does not cover every condition",
    )
    check(
        evolution_problems(clean) == [],
        f"a clean report is not contract-clean: {evolution_problems(clean)}",
    )

    # 2. GAP 1 - CREDIBILITY. A bare OOS win is not evidence.
    check(
        CHAMPION_EVOLUTION_CREDIBILITY_SIGMA >= 1.96,
        "the credibility bar is below the 95% level",
    )
    thin = replacement(
        oos_comparison={
            "candidate_value": 0.60,
            "incumbent_value": 0.55,
            "sample": 250,
        }
    )
    check(
        thin["may_replace"] is False and EVO_OOS in thin["blocked_by"],
        f"a 5-point 'improvement' on 250 observations was accepted — MEASURED, "
        f"two models of IDENTICAL skill differ by 5.6 points on 10% of "
        f"comparisons at that size",
    )
    marginal = credibility(0.56, 0.55, 2000)
    check(
        not marginal["credible"],
        "a 1-point improvement on 2000 observations was called credible",
    )
    strong = credibility(0.62, 0.55, 2000)
    check(
        strong["credible"],
        "a 7-point improvement on 2000 observations was NOT called credible — "
        "the bar is a wall rather than a filter",
    )
    check(
        CHAMPION_EVOLUTION_MIN_OOS_SAMPLE >= 500,
        f"an OOS floor of {CHAMPION_EVOLUTION_MIN_OOS_SAMPLE} cannot support "
        f"the credibility test",
    )
    # THE BAR SCALES WITH THE EVIDENCE: the same gap must be credible at a
    # large sample and not at a small one.
    check(
        credibility(0.59, 0.55, 5000)["credible"]
        and not credibility(0.59, 0.55, 600)["credible"],
        "the credibility bar does not scale with sample size, so it is either "
        "too lax at small n or too strict at large n",
    )

    # 3. GAP 2 - RISK. Accuracy cannot see it.
    check(
        EVO_RISK in CHAMPION_EVOLUTION_OWNED,
        "risk is not owned by L8 — nothing else in the promotion path "
        "measures it",
    )
    risky = replacement(candidate_returns=series(500, 0.62, 3.0, 3))
    check(
        risky["may_replace"] is False and EVO_RISK in risky["blocked_by"],
        "a candidate with BETTER accuracy and a far worse drawdown was "
        "accepted — MEASURED, that model loses money while its hit rate rises",
    )
    # ...and the same hit rate with bigger losses, which accuracy cannot see.
    same_accuracy = risk_condition(
        series(500, 0.58, 1.0, 4), series(500, 0.58, 3.0, 5)
    )
    check(
        same_accuracy["outcome"] == EVO_FAIL,
        "a candidate with an IDENTICAL hit rate and a 28x worse drawdown "
        "passed the risk condition",
    )
    check(
        CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO <= 2.0,
        "the drawdown bound permits doubling the worst loss for accuracy",
    )
    check(
        risk_condition(None, None)["outcome"] == EVO_NOT_EVALUATED,
        "a missing return series was treated as acceptable risk",
    )

    # 4. GAP 3 - FALSE POSITIVES, not the veto rate.
    degraded = replacement(candidate_calls=calls(90, 110))
    check(
        degraded["may_replace"] is False
        and EVO_FALSE_POSITIVE in degraded["blocked_by"],
        "a candidate whose acted-on false-positive rate rose 0.35 -> 0.55 was "
        "accepted — MEASURED, it matches the champion on OVERALL accuracy",
    )
    check(
        CHAMPION_EVOLUTION_MAX_FP_INCREASE <= 0.10,
        "the false-positive allowance is not a bound",
    )
    few = replacement(candidate_calls=calls(5, 2))
    check(
        few["outcomes"][EVO_FALSE_POSITIVE] == EVO_NOT_EVALUATED,
        f"a precision from 7 acted-on calls was treated as a verdict; the "
        f"floor is {CHAMPION_EVOLUTION_MIN_ACTED_CALLS}",
    )

    # 5. NOT_EVALUATED BLOCKS. Absence of evidence is not evidence of safety.
    for missing, label in (
        ({"calibration": None}, "calibration"),
        ({"regime": None}, "regime robustness"),
        ({"reproducibility": None}, "reproducibility"),
        ({"governance": None}, "governance"),
        ({"oos_comparison": None}, "the OOS comparison"),
    ):
        report = replacement(**missing)
        check(
            report["may_replace"] is False,
            f"a replacement proceeded with {label} unevaluated",
        )

    # 6. W5: DELEGATED CONDITIONS ARE RELAYED, NOT RECOMPUTED.
    check(
        EVO_GOVERNANCE not in CHAMPION_EVOLUTION_OWNED,
        "L8 claims to own governance approval — a second copy would let it "
        "approve a promotion M5 would refuse",
    )
    relayed = replacement(
        calibration={"outcome": EVO_FAIL, "reason": "calibration gap 0.31"}
    )
    check(
        relayed["may_replace"] is False,
        "a FAILING delegated verdict did not block the replacement",
    )
    check(
        any(
            c["reason"] == "calibration gap 0.31"
            for c in relayed["conditions"]
        ),
        "a delegated reason was not relayed verbatim — L8 is re-deciding what "
        "another sprint owns",
    )
    source = (REPO_ROOT / "core" / "champion_evolution.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("def calibration_drift", "def approval_check", "def evaluate_promotion"):
        check(
            forbidden not in source,
            f"core/champion_evolution.py defines {forbidden!r} — L8 is "
            f"reimplementing a condition another sprint owns",
        )

    # 7. HISTORICAL FORECASTS REMAIN IMMUTABLE. The one clause that is an
    #    invariant rather than a threshold.
    check(
        CHAMPION_EVOLUTION_HISTORY_IMMUTABLE,
        "historical immutability is switched off — a system that edits its "
        "own history can report any track record it likes",
    )
    before = [
        {"forecast_id": "f1", "value": 0.6, "claim": "POINT", "model_version": "m-1"},
        {"forecast_id": "f2", "value": 0.4, "claim": "POINT", "model_version": "m-1"},
    ]
    check(
        history_immutability_problems(before, before) == [],
        "an unchanged history was reported as mutated",
    )
    edited = [dict(before[0], value=0.9), before[1]]
    check(
        history_immutability_problems(before, edited),
        "a rewritten forecast VALUE was not detected",
    )
    reattributed = [dict(before[0], model_version="m-2"), before[1]]
    check(
        history_immutability_problems(before, reattributed),
        "a forecast re-attributed to a different model was not detected — "
        "promotion must never change which model made a past call",
    )
    check(
        history_immutability_problems(before, [before[0]]),
        "a DELETED historical forecast was not detected; deletion rewrites "
        "the record as surely as an edit does",
    )

    # 8. RENDERING AND CONTRACT.
    check(
        len(render_evolution(clean)) == len(CHAMPION_EVOLUTION_CONDITIONS),
        "the rendering does not carry one line per condition",
    )
    forged = dict(clean)
    forged["outcomes"] = dict(clean["outcomes"])
    forged["outcomes"][EVO_RISK] = EVO_FAIL
    check(
        evolution_problems(forged),
        "a report claiming may_replace while a condition failed was accepted",
    )

    if FAILURES:
        print("L8 champion-evolution gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L8 champion-evolution gate OK:")
    print("  seven conditions, all required; six of seven is not a replacement.")
    print(f"  credibility scales with evidence ({CHAMPION_EVOLUTION_CREDIBILITY_SIGMA} "
          f"sigma, floor {CHAMPION_EVOLUTION_MIN_OOS_SAMPLE}): a 5-point win on "
          f"250 observations is refused.")
    print("  a candidate with the SAME hit rate and a 28x worse drawdown is refused.")
    print("  an acted-on false-positive rate rising 0.35 -> 0.55 is refused.")
    print("  NOT_EVALUATED blocks; delegated verdicts are relayed, not recomputed.")
    print("  a rewritten, re-attributed or deleted historical forecast is detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
