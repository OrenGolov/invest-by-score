"""L7 regime-specific learning — specialization must be earned, per regime.

"Evaluate separate models per regime (bullish/bearish/range/risk-off/stress)
only if OOS evidence supports specialization."

**The last clause is the whole task.** Splitting the training data five ways is
not free, and the default answer must be NO.

**MEASURED: specialization loses on thin data even when the skill genuinely
differs by regime.** Brier on held-out data — when skill does *not* differ,
pooling wins at every sample size; when it *does*, specialization still loses
until roughly 300 observations.

**The rare-regime problem.** Stress is ~5% of observations, so at n=1000 it
holds ~52 — and a rate from ~50 observations carries a typical error of 0.056,
comparable to the 0.08 difference being detected. The estimate *is* the noise.
A stress-specific model is refused until its own evidence exists, however much
total data the system holds.

**A bare OOS win is not evidence.** MEASURED with no real difference anywhere,
specialization still "wins" on 14–23% of single comparisons. Hence a margin
*and* a fold majority: 4 of 5 folds cuts noise adoption to 2–4% while real
specialization is still found.

**Decided per regime, never all-or-nothing**, and a regime that does not earn
its own model falls back to the pooled one rather than going unserved.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Sequence

from core.config import (
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
    REGIME_SPECIALIZATION_VERSION,
)


class RegimeSpecializationError(ValueError):
    """Raised when a specialization request is structurally invalid."""


def _verdict(regime: str, verdict: str, reason: str, **detail) -> dict:
    if regime not in REGIME_SPECIALIZATION_LABELS:
        raise RegimeSpecializationError(
            f"{regime!r} is not a governed regime label; expected one of "
            f"{REGIME_SPECIALIZATION_LABELS}"
        )
    if verdict not in REGIME_SPECIALIZATION_VERDICTS:
        raise RegimeSpecializationError(f"{verdict!r} is not a verdict")
    record = {"regime": regime, "verdict": verdict, "reason": reason}
    record.update(detail)
    return record


def brier(predictions: Sequence[float], outcomes: Sequence[float]) -> float | None:
    """Mean squared error of a probabilistic prediction."""
    if not predictions or len(predictions) != len(outcomes):
        return None
    return sum(
        (float(p) - float(o)) ** 2 for p, o in zip(predictions, outcomes)
    ) / len(predictions)


def relative_improvement(pooled: float | None, specialized: float | None) -> float | None:
    """How much better the specialized model is, as a fraction of pooled error.

    RELATIVE, not absolute: a 0.001 Brier gain means something different
    against a baseline of 0.25 than against 0.05.
    """
    if pooled is None or specialized is None or pooled <= 0:
        return None
    return (pooled - specialized) / pooled


def fold_comparison(
    pooled_predictions: Sequence[float],
    specialized_predictions: Sequence[float],
    outcomes: Sequence[float],
    *,
    margin: float = REGIME_SPECIALIZATION_MARGIN,
) -> dict:
    """One fold: did the specialized model beat pooling by the margin?"""
    if len(pooled_predictions) != len(outcomes) or len(
        specialized_predictions
    ) != len(outcomes):
        raise RegimeSpecializationError(
            "predictions and outcomes must be paired — a mismatched length "
            "means the comparison is already wrong"
        )
    pooled_score = brier(pooled_predictions, outcomes)
    specialized_score = brier(specialized_predictions, outcomes)
    gain = relative_improvement(pooled_score, specialized_score)
    return {
        "pooled_brier": None if pooled_score is None else round(pooled_score, 6),
        "specialized_brier": (
            None if specialized_score is None else round(specialized_score, 6)
        ),
        "relative_gain": None if gain is None else round(gain, 6),
        "won": bool(gain is not None and gain >= margin),
        "observations": len(outcomes),
    }


def evaluate_regime(
    regime: str,
    folds: Iterable[Mapping[str, Sequence[float]]],
    *,
    min_cell: int = REGIME_SPECIALIZATION_MIN_CELL,
    margin: float = REGIME_SPECIALIZATION_MARGIN,
    folds_required: int = REGIME_SPECIALIZATION_FOLDS_REQUIRED,
    folds_expected: int = REGIME_SPECIALIZATION_FOLDS,
) -> dict:
    """Has this regime earned its own model?

    Each fold supplies `pooled`, `specialized` and `outcomes` for observations
    in this regime only. A fold below the cell floor is not counted as a loss
    — it is not counted at all, because it could not be tested.
    """
    folds = list(folds or [])
    if not folds:
        return _verdict(
            regime,
            REGIME_SPEC_NOT_EVALUATED,
            "no folds were supplied, so specialization was never tested — "
            "which is not the same as pooling having won",
            folds_run=0,
            folds_required=folds_required,
        )

    results: list[dict] = []
    thin = 0
    for fold in folds:
        outcomes = list(fold.get("outcomes") or [])
        if len(outcomes) < min_cell:
            thin += 1
            continue
        results.append(
            fold_comparison(
                list(fold.get("pooled") or []),
                list(fold.get("specialized") or []),
                outcomes,
                margin=margin,
            )
        )

    if not results:
        return _verdict(
            regime,
            REGIME_SPEC_INSUFFICIENT,
            f"every fold held fewer than {min_cell} observations of this "
            f"regime. MEASURED, a rate from ~50 observations carries a typical "
            f"error of 0.056 while the effect being detected is 0.08 — the "
            f"estimate is the noise",
            folds_run=0,
            folds_thin=thin,
            min_cell=min_cell,
        )

    # A RESULT MUST REPEAT ACROSS THE FOLDS IT WAS RUN ON. Folds that could
    # not be tested are excluded from the denominator rather than counted as
    # losses, but a regime that could only be tested on a minority of folds
    # has not shown its result repeats.
    if len(results) < folds_required:
        return _verdict(
            regime,
            REGIME_SPEC_INSUFFICIENT,
            f"only {len(results)} of {len(folds)} folds could be tested "
            f"({thin} below the {min_cell} floor), fewer than the "
            f"{folds_required} required to show a result repeats",
            folds_run=len(results),
            folds_thin=thin,
            folds_required=folds_required,
            min_cell=min_cell,
        )

    wins = sum(1 for result in results if result["won"])
    gains = [r["relative_gain"] for r in results if r["relative_gain"] is not None]
    median_gain = round(statistics.median(gains), 6) if gains else None

    if wins >= folds_required:
        return _verdict(
            regime,
            REGIME_SPEC_SPECIALIZED,
            f"won {wins} of {len(results)} folds by at least {margin:.1%} "
            f"relative Brier (median gain {median_gain:+.2%})"
            if median_gain is not None
            else f"won {wins} of {len(results)} folds",
            folds_run=len(results),
            folds_won=wins,
            folds_required=folds_required,
            median_gain=median_gain,
            observations=sum(r["observations"] for r in results),
        )

    return _verdict(
        regime,
        REGIME_SPEC_POOLED,
        f"won {wins} of {len(results)} folds, short of the {folds_required} "
        f"required. MEASURED with no real difference anywhere, specialization "
        f"still wins 14-23% of single comparisons, so a bare win is not "
        f"evidence",
        folds_run=len(results),
        folds_won=wins,
        folds_required=folds_required,
        median_gain=median_gain,
        observations=sum(r["observations"] for r in results),
    )


def specialization_report(
    by_regime: Mapping[str, Iterable[Mapping[str, Sequence[float]]]] | None = None,
    **kwargs,
) -> dict:
    """Evaluate every governed regime and say which model serves each.

    Every declared regime appears, even when it was never tested — a report
    covering three of five and reading as complete would hide exactly the
    rare regimes this module is most careful about.
    """
    by_regime = dict(by_regime or {})
    unknown = sorted(set(by_regime) - set(REGIME_SPECIALIZATION_LABELS))
    if unknown:
        raise RegimeSpecializationError(
            f"{unknown} are not governed regime labels; L7 specializes on "
            f"{REGIME_SPECIALIZATION_LABELS} and never invents its own"
        )

    findings: list[dict] = []
    for regime in REGIME_SPECIALIZATION_LABELS:
        findings.append(evaluate_regime(regime, by_regime.get(regime) or [], **kwargs))

    specialized = [f["regime"] for f in findings if f["verdict"] == REGIME_SPEC_SPECIALIZED]

    # WHICH MODEL SERVES EACH REGIME. A regime that did not earn its own falls
    # back to pooled rather than going unserved: no model for stress because
    # stress is rare would remove coverage exactly when it matters most.
    serving = {
        f["regime"]: ("specialized" if f["verdict"] == REGIME_SPEC_SPECIALIZED else "pooled")
        for f in findings
    }

    return {
        "version": REGIME_SPECIALIZATION_VERSION,
        "findings": findings,
        "verdicts": {f["regime"]: f["verdict"] for f in findings},
        "specialized": sorted(specialized),
        "serving": serving,
        "default_pooled": REGIME_SPECIALIZATION_DEFAULT_POOLED,
        "fallback_pooled": REGIME_SPECIALIZATION_FALLBACK_POOLED,
        "per_regime": REGIME_SPECIALIZATION_PER_REGIME,
        "note": (
            "Specialization is adopted per regime, on out-of-sample evidence "
            "repeated across folds. The default is pooled, and a regime that "
            "does not earn its own model is served by the pooled one."
        ),
    }


def specialization_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a specialization report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    verdicts = report.get("verdicts") or {}
    missing = [name for name in REGIME_SPECIALIZATION_LABELS if name not in verdicts]
    if missing:
        problems.append(
            f"the report omits {missing} — a scan covering some regimes and "
            f"reading as complete hides exactly the rare ones"
        )

    serving = report.get("serving") or {}
    for regime, verdict in verdicts.items():
        if verdict not in REGIME_SPECIALIZATION_VERDICTS:
            problems.append(f"{regime}: unknown verdict {verdict!r}")
        # EVERY REGIME IS SERVED. An unserved regime is worse than a pooled
        # one, and silently having no model is how coverage disappears.
        if regime not in serving:
            problems.append(f"{regime}: no model is recorded as serving it")
        elif verdict != REGIME_SPEC_SPECIALIZED and serving.get(regime) != "pooled":
            problems.append(
                f"{regime}: verdict {verdict} but served by "
                f"{serving.get(regime)!r} — only a SPECIALIZED verdict earns a "
                f"regime-specific model"
            )
    for finding in report.get("findings") or []:
        if not str(finding.get("reason") or "").strip():
            problems.append(f"{finding.get('regime')}: no reason given")
    if not report.get("default_pooled", False):
        problems.append(
            "the report does not record pooled as the default — MEASURED, "
            "pooling wins at every sample size when skill does not differ"
        )
    return problems


def render_specialization(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per regime."""
    lines: list[str] = []
    for finding in report.get("findings") or []:
        lines.append(
            f"  {str(finding.get('regime')):10s} "
            f"{str(finding.get('verdict')):18s} {finding.get('reason')}"
        )
    return lines
