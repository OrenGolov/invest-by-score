"""L8 champion evolution — seven conditions, all required, history immutable.

"Replacement requires credible OOS improvement, acceptable calibration, no
unacceptable false-positive degradation, acceptable risk, regime robustness,
reproducibility, governance approval. Historical forecasts remain immutable."

**Most conditions have an owner already**, and L8 sequences rather than
rebuilds them (W5): calibration is L6's, regime robustness is L7's,
reproducibility is M8's, governance is M5's. Three gaps were real.

**"Credible" was unqualified.** M5 requires the candidate to beat the incumbent
out of sample but not by how much relative to noise. MEASURED, two models of
IDENTICAL skill: at n=250 the challenger beats the champion by 5.6 points on
10% of comparisons and 10.4 points on 1%. So the improvement is compared
against its own standard error, and the bar scales with the evidence.

**Risk was never measured at promotion.** MEASURED with the hit rate held
fixed and only the size of losing moves changed, a challenger with the *same*
hit rate carried a **28× worse drawdown**, and one with a *better* hit rate
still lost money. An accuracy-only gate promotes both.

**False positives are not the veto rate.** M5 watches how often the policy
vetoes; L8 needs how often an acted-on call was wrong. MEASURED, a challenger
matching the champion on overall accuracy (0.594 vs 0.597) moved the BUY
false-positive rate 0.349 → 0.551.

**Historical forecasts remain immutable.** This is the one clause that is not
a threshold but an invariant: a system that can edit what a past forecast said
can report any track record it likes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.config import (
    CHAMPION_EVOLUTION_CONDITIONS,
    CHAMPION_EVOLUTION_CREDIBILITY_SIGMA,
    CHAMPION_EVOLUTION_HISTORY_IMMUTABLE,
    CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO,
    CHAMPION_EVOLUTION_MAX_FP_INCREASE,
    CHAMPION_EVOLUTION_MAX_TAIL_RATIO,
    CHAMPION_EVOLUTION_MIN_ACTED_CALLS,
    CHAMPION_EVOLUTION_MIN_OOS_SAMPLE,
    CHAMPION_EVOLUTION_OUTCOMES,
    CHAMPION_EVOLUTION_OWNED,
    CHAMPION_EVOLUTION_REQUIRED,
    CHAMPION_EVOLUTION_VERSION,
    EVO_CALIBRATION,
    EVO_FAIL,
    EVO_FALSE_POSITIVE,
    EVO_GOVERNANCE,
    EVO_NOT_EVALUATED,
    EVO_OOS,
    EVO_PASS,
    EVO_REGIME,
    EVO_REPRODUCIBILITY,
    EVO_RISK,
)


class ChampionEvolutionError(ValueError):
    """Raised when an evolution request is structurally invalid."""


def _condition(name: str, outcome: str, reason: str, **detail) -> dict:
    if name not in CHAMPION_EVOLUTION_CONDITIONS:
        raise ChampionEvolutionError(f"{name!r} is not an L8 condition")
    if outcome not in CHAMPION_EVOLUTION_OUTCOMES:
        raise ChampionEvolutionError(f"{outcome!r} is not an L8 outcome")
    record = {"condition": name, "outcome": outcome, "reason": reason}
    record.update(detail)
    return record


def max_drawdown(returns: Sequence[float]) -> float:
    """Worst peak-to-trough decline of the cumulative return path."""
    peak = cumulative = 0.0
    worst = 0.0
    for value in returns or []:
        cumulative += float(value)
        peak = max(peak, cumulative)
        worst = min(worst, cumulative - peak)
    return worst


def left_tail(returns: Sequence[float], quantile: float = 0.05) -> float | None:
    """The 5th-percentile return: how bad a bad day is."""
    ordered = sorted(float(value) for value in returns or [])
    if not ordered:
        return None
    return ordered[min(int(quantile * len(ordered)), len(ordered) - 1)]


def credibility(
    candidate_value: float,
    incumbent_value: float,
    sample: int,
    *,
    sigma: float = CHAMPION_EVOLUTION_CREDIBILITY_SIGMA,
    min_sample: int = CHAMPION_EVOLUTION_MIN_OOS_SAMPLE,
) -> dict:
    """Is the OOS improvement large relative to what luck produces?

    The bar SCALES WITH THE EVIDENCE rather than being a fixed number of
    points. MEASURED, the p99 of the null gap between two identical models is
    +0.1040 at n=250 and +0.0365 at n=2000, so a fixed percentage bar would be
    far too lax at small n and far too strict at large n.
    """
    if sample < min_sample:
        return {
            "credible": False,
            "reason": (
                f"{sample} out-of-sample observations, below the {min_sample} "
                f"floor. MEASURED, detection of a real 3-point edge at the "
                f"p<0.01 bar is 6.3% at n=250 — the test cannot find anything"
            ),
            "sample": sample,
        }

    gap = float(candidate_value) - float(incumbent_value)
    # Standard error of a difference in rates, at the conservative p=0.5.
    standard_error = math.sqrt(2.0 * 0.25 / sample)
    required = sigma * standard_error
    return {
        "credible": bool(gap >= required),
        "gap": round(gap, 6),
        "required": round(required, 6),
        "sigma": sigma,
        "standard_error": round(standard_error, 6),
        "sample": sample,
        "reason": (
            f"improvement {gap:+.4f} against the {required:.4f} needed at "
            f"{sigma} sigma on {sample} observations"
        ),
    }


def oos_condition(comparison: Mapping[str, Any] | None) -> dict:
    """Condition 1: a CREDIBLE out-of-sample improvement."""
    if not comparison:
        return _condition(
            EVO_OOS,
            EVO_NOT_EVALUATED,
            "no out-of-sample comparison was supplied",
        )
    candidate = comparison.get("candidate_value")
    incumbent = comparison.get("incumbent_value")
    sample = comparison.get("sample")
    if candidate is None or incumbent is None or sample is None:
        return _condition(
            EVO_OOS,
            EVO_NOT_EVALUATED,
            "the comparison lacks a candidate value, an incumbent value or a "
            "sample size, so credibility cannot be assessed",
        )
    verdict = credibility(float(candidate), float(incumbent), int(sample))
    return _condition(
        EVO_OOS,
        EVO_PASS if verdict["credible"] else EVO_FAIL,
        verdict["reason"],
        **{k: v for k, v in verdict.items() if k not in ("credible", "reason")},
    )


def risk_condition(
    incumbent_returns: Sequence[float] | None,
    candidate_returns: Sequence[float] | None,
    *,
    max_drawdown_ratio: float = CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO,
    max_tail_ratio: float = CHAMPION_EVOLUTION_MAX_TAIL_RATIO,
) -> dict:
    """Condition 4: acceptable risk, which accuracy cannot see.

    MEASURED with the hit rate held FIXED and only the size of losing moves
    changed, a challenger with the same hit rate carried a 28x worse drawdown
    while losing money.
    """
    if not incumbent_returns or not candidate_returns:
        return _condition(
            EVO_RISK,
            EVO_NOT_EVALUATED,
            "return series for both the incumbent and the candidate are "
            "required — nothing in the promotion path measured risk before L8",
        )

    before_dd = abs(max_drawdown(incumbent_returns))
    after_dd = abs(max_drawdown(candidate_returns))
    before_tail = left_tail(incumbent_returns)
    after_tail = left_tail(candidate_returns)

    problems: list[str] = []
    drawdown_ratio = None
    if before_dd > 0:
        drawdown_ratio = after_dd / before_dd
        if drawdown_ratio > max_drawdown_ratio:
            problems.append(
                f"drawdown {before_dd:.4f} -> {after_dd:.4f} "
                f"({drawdown_ratio:.2f}x, bound {max_drawdown_ratio}x)"
            )
    elif after_dd > 0:
        problems.append(
            f"the incumbent had no drawdown and the candidate has {after_dd:.4f}"
        )

    tail_ratio = None
    if before_tail is not None and after_tail is not None and before_tail < 0:
        tail_ratio = abs(after_tail) / abs(before_tail)
        if tail_ratio > max_tail_ratio:
            problems.append(
                f"left tail {before_tail:.4f} -> {after_tail:.4f} "
                f"({tail_ratio:.2f}x, bound {max_tail_ratio}x)"
            )

    detail = {
        "incumbent_drawdown": round(before_dd, 6),
        "candidate_drawdown": round(after_dd, 6),
        "drawdown_ratio": None if drawdown_ratio is None else round(drawdown_ratio, 4),
        "incumbent_tail": None if before_tail is None else round(before_tail, 6),
        "candidate_tail": None if after_tail is None else round(after_tail, 6),
        "tail_ratio": None if tail_ratio is None else round(tail_ratio, 4),
    }
    if problems:
        return _condition(EVO_RISK, EVO_FAIL, "; ".join(problems), **detail)
    return _condition(
        EVO_RISK,
        EVO_PASS,
        f"drawdown {before_dd:.4f} -> {after_dd:.4f}, left tail "
        f"{before_tail if before_tail is None else round(before_tail, 4)} -> "
        f"{after_tail if after_tail is None else round(after_tail, 4)}",
        **detail,
    )


def false_positive_condition(
    incumbent: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    *,
    max_increase: float = CHAMPION_EVOLUTION_MAX_FP_INCREASE,
    min_calls: int = CHAMPION_EVOLUTION_MIN_ACTED_CALLS,
) -> dict:
    """Condition 3: acted-on calls must not get worse.

    NOT the veto rate, which is M5's. MEASURED, a challenger matching the
    champion on OVERALL accuracy moved the BUY false-positive rate
    0.349 -> 0.551 — every acted-on call got worse and accuracy could not
    see it.
    """
    if not incumbent or not candidate:
        return _condition(
            EVO_FALSE_POSITIVE,
            EVO_NOT_EVALUATED,
            "acted-on call counts for both models are required",
        )

    def rate(record: Mapping[str, Any]) -> tuple[float | None, int]:
        true_positive = int(record.get("true_positives") or 0)
        false_positive = int(record.get("false_positives") or 0)
        total = true_positive + false_positive
        if total <= 0:
            return None, 0
        return false_positive / total, total

    before, before_n = rate(incumbent)
    after, after_n = rate(candidate)
    if before is None or after is None:
        return _condition(
            EVO_FALSE_POSITIVE,
            EVO_NOT_EVALUATED,
            "one of the models made no acted-on calls, so precision cannot be "
            "compared",
        )
    if before_n < min_calls or after_n < min_calls:
        return _condition(
            EVO_FALSE_POSITIVE,
            EVO_NOT_EVALUATED,
            f"fewer than {min_calls} acted-on calls on one side "
            f"(incumbent {before_n}, candidate {after_n}); a precision from a "
            f"handful of calls is noise",
            incumbent_calls=before_n,
            candidate_calls=after_n,
        )

    increase = after - before
    detail = {
        "incumbent_false_positive_rate": round(before, 6),
        "candidate_false_positive_rate": round(after, 6),
        "increase": round(increase, 6),
        "incumbent_calls": before_n,
        "candidate_calls": after_n,
    }
    if increase > max_increase:
        return _condition(
            EVO_FALSE_POSITIVE,
            EVO_FAIL,
            f"acted-on false-positive rate {before:.4f} -> {after:.4f} "
            f"(+{increase:.4f}, bound +{max_increase})",
            **detail,
        )
    return _condition(
        EVO_FALSE_POSITIVE,
        EVO_PASS,
        f"acted-on false-positive rate {before:.4f} -> {after:.4f} "
        f"({increase:+.4f})",
        **detail,
    )


_DELEGATED = {
    EVO_CALIBRATION: "core.drift_detection.calibration_drift (L6)",
    EVO_REGIME: "core.regime_specialization (L7)",
    EVO_REPRODUCIBILITY: "M8 reproducibility guarantee",
    EVO_GOVERNANCE: "core.promotion human_approval (M5)",
}


def _delegated(name: str, supplied: Mapping[str, Any] | None) -> dict:
    """Lift a delegated verdict, preserving it verbatim.

    L8 NEVER recomputes a delegated condition. It sequences the seven; the
    owning sprint decides what passes.
    """
    owner = _DELEGATED[name]
    if supplied is None:
        return _condition(
            name,
            EVO_NOT_EVALUATED,
            f"no verdict was supplied by {owner}, so this condition has found "
            f"nothing rather than found the candidate acceptable",
            delegated_to=owner,
        )
    outcome = str(supplied.get("outcome") or EVO_NOT_EVALUATED)
    if outcome not in CHAMPION_EVOLUTION_OUTCOMES:
        outcome = EVO_NOT_EVALUATED
    reason = str(supplied.get("reason") or "").strip()
    return _condition(
        name,
        outcome,
        reason or f"{owner} reported {outcome} with no reason recorded",
        delegated_to=owner,
    )


def evaluate_replacement(
    *,
    oos_comparison: Mapping[str, Any] | None = None,
    incumbent_returns: Sequence[float] | None = None,
    candidate_returns: Sequence[float] | None = None,
    incumbent_calls: Mapping[str, Any] | None = None,
    candidate_calls: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    regime: Mapping[str, Any] | None = None,
    reproducibility: Mapping[str, Any] | None = None,
    governance: Mapping[str, Any] | None = None,
) -> dict:
    """All seven conditions. Replacement is allowed only when every one passes."""
    conditions = [
        oos_condition(oos_comparison),
        _delegated(EVO_CALIBRATION, calibration),
        false_positive_condition(incumbent_calls, candidate_calls),
        risk_condition(incumbent_returns, candidate_returns),
        _delegated(EVO_REGIME, regime),
        _delegated(EVO_REPRODUCIBILITY, reproducibility),
        _delegated(EVO_GOVERNANCE, governance),
    ]
    ordered = sorted(
        conditions, key=lambda item: CHAMPION_EVOLUTION_CONDITIONS.index(item["condition"])
    )
    blocking = [
        item
        for item in ordered
        if item["condition"] in CHAMPION_EVOLUTION_REQUIRED
        and item["outcome"] != EVO_PASS
    ]
    return {
        "version": CHAMPION_EVOLUTION_VERSION,
        "conditions": ordered,
        "outcomes": {item["condition"]: item["outcome"] for item in ordered},
        "may_replace": not blocking,
        "blocked_by": [item["condition"] for item in blocking],
        "blocking_reasons": [
            f"{item['condition']}: {item['reason']}" for item in blocking
        ],
        "owned": tuple(CHAMPION_EVOLUTION_OWNED),
        "history_immutable": CHAMPION_EVOLUTION_HISTORY_IMMUTABLE,
        "note": (
            "Every condition is required — the sprint lists them with 'and'. "
            "may_replace means the evidence allows a replacement, not that one "
            "has happened; crowning remains a human act under the L5 chain, "
            "and the champion tenure bound still applies."
        ),
    }


def history_immutability_problems(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Did a promotion rewrite any historical forecast?

    THE ONE CLAUSE THAT IS NOT A THRESHOLD. A system that can edit what a past
    forecast said, or which model made it, can report any track record it
    likes — and every measurement in this repository would become
    unverifiable. Checked by comparison rather than assumed.
    """
    problems: list[str] = []
    indexed = {
        str(row.get("forecast_id")): row for row in before or [] if row.get("forecast_id")
    }
    seen: set[str] = set()
    for row in after or []:
        forecast_id = str(row.get("forecast_id") or "")
        if not forecast_id:
            continue
        seen.add(forecast_id)
        original = indexed.get(forecast_id)
        if original is None:
            continue
        for field in ("value", "claim", "as_of", "horizon", "forecast_version", "model_version"):
            if field in original and original.get(field) != row.get(field):
                problems.append(
                    f"{forecast_id}: {field} was rewritten "
                    f"{original.get(field)!r} -> {row.get(field)!r}"
                )
    missing = sorted(set(indexed) - seen)
    if missing:
        problems.append(
            f"{len(missing)} historical forecast(s) disappeared ({missing[:3]}"
            f"{'...' if len(missing) > 3 else ''}) — deletion rewrites the "
            f"record as surely as an edit does"
        )
    return problems


def evolution_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an evolution report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    outcomes = report.get("outcomes") or {}
    missing = [name for name in CHAMPION_EVOLUTION_CONDITIONS if name not in outcomes]
    if missing:
        problems.append(
            f"the report omits {missing} — every condition is required, and a "
            f"replacement satisfying six is not a replacement"
        )
    for condition in report.get("conditions") or []:
        if condition.get("outcome") not in CHAMPION_EVOLUTION_OUTCOMES:
            problems.append(f"{condition.get('condition')}: unknown outcome")
        if not str(condition.get("reason") or "").strip():
            problems.append(f"{condition.get('condition')}: no reason given")

    failing = [
        name for name, outcome in outcomes.items() if outcome != EVO_PASS
    ]
    if failing and report.get("may_replace"):
        problems.append(
            f"may_replace is true while {sorted(failing)} did not pass"
        )
    if not report.get("history_immutable", False):
        problems.append(
            "the report does not assert historical immutability — a system "
            "that edits its own history can report any track record"
        )
    return problems


def render_evolution(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per condition."""
    return [
        f"  {str(item.get('condition')):32s} {str(item.get('outcome')):14s} "
        f"{item.get('reason')}"
        for item in report.get("conditions") or []
    ]
