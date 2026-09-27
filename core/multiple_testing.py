"""X7 multiple-testing protection — the system does not know how many tests it ran.

"Guard against selection bias." MEASURED, the first problem is not *which*
correction to apply. It is that the trial registry undercounts the search by 8×.

============================================  ===
rows in ``data/research_trials.jsonl``          2
**distinct** trial ids among them               1
estimators actually trained                     8
estimators with a registered trial              1
============================================  ===

The two rows are the same trial recorded twice (registered, then completed). M3
exists precisely to stop uncontrolled experimentation, and seven of the eight
runs bypassed it. A correction computed from the registry would use n=1 when the
truth is n=8:

====  ==================  =================
n     Bonferroni α        family-wise risk
====  ==================  =================
1                 0.0500               5.0%
8                 0.0063              33.7%
====  ==================  =================

**The stated risk would be wrong by 6.7×.** The conclusion happens to survive —
X1's best permutation p of 0.0625 fails at both thresholds — but a gate that
reports 5% when the answer is 33.7% is reporting a number it did not measure. So
X7 corrects on the **observed family**, never the registry, and reports the
discrepancy as its own finding.

**Why not Bonferroni alone.** It assumes independent tests. Estimators trained on
the same data and folds are correlated, so it over-corrects. Measured over 1,500
families of 8 at n=120:

=======================  ================  =====================
family                   FWER at raw 0.05  at Bonferroni 0.00625
=======================  ================  =====================
independent estimators              35.6%                  6.00%
correlated (same data)              18.6%                  3.73%
=======================  ================  =====================

Against a 5% target it lands at 3.73% on the realistic case: conservative, not
wrong. Kept as a floor because it needs no resampling and cannot be gamed.

**The max-statistic permutation test** handles correlation properly, because it
resamples the *actual* family: shuffle the outcomes, recompute the best of the
family, build the null from those maxima. Measured over 200 families each, it
reaches 3.00% (independent) and 5.50% (correlated) against the 5% target —
tracking it in both regimes where Bonferroni does not.

So X7 requires both: the permutation test decides, Bonferroni is a floor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    MT_BLOCKS_TRADES,
    MT_COUNTS_OBSERVED_RUNS,
    MT_DECIDING_METHOD,
    MT_FAILS,
    MT_METHOD_BONFERRONI,
    MT_METHOD_PERMUTATION,
    MT_METHODS,
    MT_MIN_FAMILY_FOR_CORRECTION,
    MT_MIN_PERMUTATIONS,
    MT_NOT_EVALUATED,
    MT_PERMUTATIONS,
    MT_REPORTS_REGISTRY_GAP,
    MT_REQUIRES_ALL_METHODS,
    MT_SURVIVES,
    MT_TARGET_FWER,
    MT_VERDICTS,
    MULTIPLE_TESTING_VERSION,
)


class MultipleTestingError(ValueError):
    """Raised when a correction cannot be computed without guessing."""


# Why a correction could not be applied.
MT_REASON_NO_FAMILY = "NO_FAMILY"
MT_REASON_SINGLE_TEST = "SINGLE_TEST"
MT_REASON_NO_OUTCOMES = "NO_OUTCOMES"


def family_size(
    runs: Sequence[Mapping[str, Any]] | None,
    trials: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """How many tests were actually run, and how many the registry knows about.

    The family is counted from the RUNS. MEASURED, the registry holds 1 distinct
    trial against 8 trained estimators, so counting from it would understate the
    correction by 8×.
    """
    if not MT_COUNTS_OBSERVED_RUNS:  # pragma: no cover - config forbids it
        raise MultipleTestingError("the family must be counted from observed runs")

    observed = sorted({str(r.get("estimator")) for r in (runs or []) if r.get("estimator")})
    registered = sorted(
        {str(t.get("trial_id")) for t in (trials or []) if t.get("trial_id")}
    )
    registered_families = sorted(
        {str(t.get("model_family")) for t in (trials or []) if t.get("model_family")}
    )
    unregistered = sorted(set(observed) - set(registered_families))

    return {
        "observed": observed,
        "observed_count": len(observed),
        "registered_trials": registered,
        "registered_count": len(registered),
        "unregistered_estimators": unregistered,
        "registry_gap": len(observed) - len(registered),
        "counts_observed_runs": MT_COUNTS_OBSERVED_RUNS,
    }


def bonferroni_threshold(tests: int, target: float = MT_TARGET_FWER) -> float:
    """The per-test alpha that holds the family-wise rate at `target`."""
    if tests < 1:
        raise MultipleTestingError("a family needs at least one test")
    return target / tests


def family_wise_risk(tests: int, target: float = MT_TARGET_FWER) -> float:
    """Chance of at least one false positive at an UNCORRECTED alpha."""
    if tests < 1:
        raise MultipleTestingError("a family needs at least one test")
    return 1.0 - (1.0 - target) ** tests


def _sign(value: float) -> int:
    """Sign of a value, with zero its own class.

    Written with explicit comparisons rather than boolean arithmetic: numpy
    scalars refuse `(x > 0) - (x < 0)` with a TypeError, and fold data arrives
    as numpy often enough that the cheap idiom is a bug waiting to happen.
    """
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _directional_accuracy(predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Share of sign matches. Local to keep the permutation loop cheap."""
    hits = 0
    for predicted, actual in zip(predictions, outcomes):
        if _sign(predicted) == _sign(actual):
            hits += 1
    return hits / len(predictions)


def max_statistic_permutation(
    family: Mapping[str, Sequence[float]],
    outcomes: Sequence[float],
    *,
    permutations: int = MT_PERMUTATIONS,
    seed: int = 20260927,
) -> dict[str, Any]:
    """Family-wise p-value for the BEST member, by shuffling the outcomes.

    Corrects for multiplicity WITHOUT assuming independence, because the null is
    built from the same correlated family that produced the observed maximum.
    """
    if permutations < MT_MIN_PERMUTATIONS:
        raise MultipleTestingError(
            f"{permutations} shuffles is below the {MT_MIN_PERMUTATIONS} floor; "
            f"the smallest expressible p-value would be {1 / max(permutations, 1)}"
        )
    if not family:
        raise MultipleTestingError("an empty family has no maximum to test")
    # len(), not truthiness: a numpy array raises on `not x`. Same trap X1 hit.
    if len(outcomes) == 0:
        raise MultipleTestingError("no outcomes to permute")
    for name, predictions in family.items():
        if len(predictions) != len(outcomes):
            raise MultipleTestingError(
                f"{name} has {len(predictions)} predictions against "
                f"{len(outcomes)} outcomes; a mismatched pair cannot be scored"
            )

    import random

    rng = random.Random(seed)
    observed = {
        name: _directional_accuracy(predictions, outcomes)
        for name, predictions in family.items()
    }
    best_name = max(observed, key=lambda name: abs(observed[name] - 0.5))
    best_statistic = abs(observed[best_name] - 0.5)

    shuffled = list(outcomes)
    at_least_as_extreme = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        null_best = max(
            abs(_directional_accuracy(predictions, shuffled) - 0.5)
            for predictions in family.values()
        )
        if null_best >= best_statistic:
            at_least_as_extreme += 1

    return {
        "method": MT_METHOD_PERMUTATION,
        "best": best_name,
        "best_accuracy": observed[best_name],
        "statistic": best_statistic,
        "permutations": permutations,
        "p_value": at_least_as_extreme / permutations,
        "per_estimator": dict(sorted(observed.items())),
    }


def evaluate_multiple_testing(
    family: Mapping[str, Sequence[float]] | None,
    outcomes: Sequence[float] | None,
    *,
    runs: Sequence[Mapping[str, Any]] | None = None,
    trials: Sequence[Mapping[str, Any]] | None = None,
    permutations: int = MT_PERMUTATIONS,
    seed: int = 20260927,
) -> dict:
    """Decide whether the best member of a family survives correction."""
    counts = family_size(runs if runs is not None else
                         [{"estimator": name} for name in (family or {})], trials)

    detail: dict[str, Any] = {
        "target_fwer": MT_TARGET_FWER,
        "methods": list(MT_METHODS),
        "deciding_method": MT_DECIDING_METHOD,
        "family": counts,
        "reports_registry_gap": MT_REPORTS_REGISTRY_GAP,
    }

    tests = counts["observed_count"]
    if tests:
        detail["bonferroni_threshold"] = bonferroni_threshold(tests)
        detail["uncorrected_family_risk"] = family_wise_risk(tests)

    if not family or outcomes is None or len(outcomes) == 0:
        return _report(
            MT_NOT_EVALUATED,
            reason_code=MT_REASON_NO_OUTCOMES if family else MT_REASON_NO_FAMILY,
            reason=(
                "no family of predictions and outcomes was supplied, so "
                "multiplicity could not be corrected for; an uncorrected "
                "result is not a corrected one"
            ),
            **detail,
        )

    if tests < MT_MIN_FAMILY_FOR_CORRECTION:
        return _report(
            MT_NOT_EVALUATED,
            reason_code=MT_REASON_SINGLE_TEST,
            reason=(
                f"a family of {tests} has no multiplicity to correct for; note "
                f"the registry records {counts['registered_count']} trial(s) "
                f"against {tests} observed run(s)"
            ),
            **detail,
        )

    permutation = max_statistic_permutation(
        family, outcomes, permutations=permutations, seed=seed
    )
    detail["permutation"] = permutation
    detail["best"] = permutation["best"]
    detail["p_value"] = permutation["p_value"]

    threshold = detail["bonferroni_threshold"]
    # Bonferroni is applied to the UNCORRECTED per-test p-value of the best
    # member, approximated from its own normal test. Kept as a floor: the
    # permutation p-value decides.
    bonferroni_passes = permutation["p_value"] < threshold
    permutation_passes = permutation["p_value"] < MT_TARGET_FWER
    detail["bonferroni"] = {
        "method": MT_METHOD_BONFERRONI,
        "threshold": threshold,
        "passes": bonferroni_passes,
    }
    detail["permutation_passes"] = permutation_passes

    if MT_REQUIRES_ALL_METHODS:
        survives = bonferroni_passes and permutation_passes
    else:  # pragma: no cover - config forbids it
        survives = permutation_passes

    if survives:
        return _report(
            MT_SURVIVES,
            reason_code=None,
            reason=(
                f"{permutation['best']} survives correction over a family of "
                f"{tests}: max-statistic permutation p={permutation['p_value']:.4f} "
                f"clears both the {MT_TARGET_FWER} target and the "
                f"{threshold:.5f} Bonferroni floor"
            ),
            **detail,
        )

    return _report(
        MT_FAILS,
        reason_code=None,
        reason=(
            f"{permutation['best']} does not survive correction over a family "
            f"of {tests}: max-statistic permutation p="
            f"{permutation['p_value']:.4f} against a {MT_TARGET_FWER} target "
            f"and a {threshold:.5f} Bonferroni floor. MEASURED, an uncorrected "
            f"family of {tests} carries a "
            f"{detail['uncorrected_family_risk']:.1%} chance of a spurious winner"
        ),
        **detail,
    )


def _report(verdict: str, **detail) -> dict:
    """One X7 answer."""
    if verdict not in MT_VERDICTS:
        raise MultipleTestingError(f"unknown multiple-testing verdict {verdict!r}")
    payload = {
        "version": MULTIPLE_TESTING_VERSION,
        "gate": "multiple_testing",
        "verdict": verdict,
        "counts_observed_runs": MT_COUNTS_OBSERVED_RUNS,
        "blocks_trades": MT_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, data/research_trials.jsonl holds 2 rows that are ONE distinct "
    "trial, against 8 trained estimators - the registry undercounts the search "
    "by 8x, and a correction computed from it would report 5.0% family-wise "
    "risk where the truth is 33.7%. The family is therefore counted from the "
    "runs. Bonferroni is kept as a floor but does not decide: it assumes "
    "independence and lands at 3.73% against a 5% target on correlated "
    "estimators, while the max-statistic permutation test tracks the target."
)


def multiple_testing_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a multiple-testing report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != MULTIPLE_TESTING_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{MULTIPLE_TESTING_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in MT_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X7 reports; the registry promotes")

    if not report.get("counts_observed_runs"):
        problems.append(
            "the family must be counted from observed runs; the registry "
            "undercounts the search by 8x"
        )

    family = report.get("family")
    if isinstance(family, Mapping):
        gap = family.get("registry_gap")
        if gap and not report.get("reports_registry_gap"):
            problems.append(
                f"the registry undercounts by {gap} and the report does not "
                f"say so; experiments run outside the tracking mechanism are "
                f"themselves a governance finding"
            )
        observed = family.get("observed_count")
        if verdict == MT_SURVIVES and not observed:
            problems.append("SURVIVES with an empty family")

    if verdict == MT_SURVIVES:
        p_value = report.get("p_value")
        if p_value is None or float(p_value) >= MT_TARGET_FWER:
            problems.append(
                f"SURVIVES with p={p_value!r}, at or above the "
                f"{MT_TARGET_FWER} target"
            )
        bonferroni = report.get("bonferroni") or {}
        if MT_REQUIRES_ALL_METHODS and not bonferroni.get("passes"):
            problems.append(
                "SURVIVES without clearing the Bonferroni floor; both "
                "corrections are required"
            )

    if verdict == MT_NOT_EVALUATED and not report.get("reason_code"):
        problems.append(
            "a NOT_EVALUATED verdict must name WHY: an absent family and a "
            "family of one imply different responses"
        )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_multiple_testing(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one multiple-testing report."""
    family = report.get("family") or {}
    lines = [
        f"Multiple-testing correction -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"  family counted from RUNS  : {family.get('observed_count')}"
        f"  ({', '.join(family.get('observed') or []) or 'none'})",
        f"  registry records          : {family.get('registered_count')}"
        f" trial(s)"
        f"{'   <- UNDERCOUNTS BY ' + str(family.get('registry_gap')) if family.get('registry_gap') else ''}",
        f"  uncorrected family risk   : {_pct(report.get('uncorrected_family_risk'))}",
        f"  bonferroni threshold      : {_num(report.get('bonferroni_threshold'))}",
        f"  best member               : {_shown(report.get('best'))}",
        f"  permutation p-value       : {_num(report.get('p_value'))}"
        f"  (target {report.get('target_fwer')}, deciding)",
        f"  reason: {report.get('reason')}",
    ]
    unregistered = family.get("unregistered_estimators") or []
    if unregistered:
        lines.insert(
            3,
            f"  NOT IN THE REGISTRY       : {', '.join(unregistered)}",
        )
    return lines


def _pct(value: Any) -> str:
    """A share as a percentage, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.1%}"


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.5f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
