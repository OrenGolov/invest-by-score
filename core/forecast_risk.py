"""R4 forecast-adjusted risk — how much of this size has the forecast earned?

R2 sized a position from correlation alone: how much of this can the
portfolio absorb? That question never asks whether the forecast motivating the
trade is worth acting on. A 51% P(up) drawn from two observations and a 51%
drawn from eight hundred produce the identical R2 size.

**THE DECIDING MEASUREMENT: confidence-scaling helps in one direction and
HURTS in the other.** On a candidate correlated with what is held and more
volatile, scaling the size by confidence walked portfolio volatility from
+19.64% down to +0.98% — exactly the intended effect. On an *uncorrelated*
candidate the same scaling walked it from -19.23% to -0.99%, monotonically at
every level tested. It did not reduce risk; it removed a benefit. Shrinking a
low-confidence diversifier makes the portfolio worse.

**So the haircut applies only to candidates that INCREASE portfolio risk.** A
risk-reducing candidate keeps R2's size regardless of confidence, because the
portfolio improves either way and a weak forecast is not a reason to decline
a free reduction in variance.

**Confidence adjusts R2's answer; it never replaces it.** MEASURED, the
ordering is not close: a HIGH-confidence concentrator at 20% raised volatility
+19.78%, while a LOW-confidence diversifier at 5% LOWERED it -4.82%. The
candidate's relationship to the portfolio dominates the forecast's strength.
A system that sized on confidence alone would rank these backwards.

**A haircut, never a multiplier.** Confidence scales size DOWNWARD only. A
confident forecast earns R2's size and never more than it, because R2's size
is already the most the portfolio can absorb — multiplying past it would
breach the risk budget on the strength of the model liking its own forecast.

**Below the floor the answer is NO_TRADE, not a token position.** A
confidence-scaled size decays smoothly toward zero, which silently produces
positions too small to be real. R2 already refuses a trade under
SIZING_MIN_WEIGHT; R4 refuses the forecast before it gets there.

**UNMEASURABLE is not low.** MEASURED, calibration, model_agreement and
model_drift are all UNMEASURABLE today because no model is registered, and F7
excludes them from its weighting rather than scoring them 0.0. R4 inherits
that: it never converts "not measured" into a haircut. A forecast is not less
reliable for living in a system that has not yet trained a model.

**R4 adjusts a size. It does not decide whether to trade** — that is R7, with
the whole portfolio in view.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    FORECAST_RISK_BLOCKS_TRADES,
    FORECAST_RISK_HAIRCUT_INCREASERS_ONLY,
    FORECAST_RISK_MAX_MULTIPLIER,
    FORECAST_RISK_MIN_CONFIDENCE,
    FORECAST_RISK_UNMEASURABLE_IS_NOT_LOW,
    FORECAST_RISK_VERSION,
    FRISK_NO_TRADE,
    FRISK_NOT_EVALUATED,
    FRISK_REDUCED,
    FRISK_UNCHANGED,
    FRISK_VERDICTS,
    SIZING_MIN_WEIGHT,
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
)


class ForecastRiskError(ValueError):
    """Raised when a forecast-adjusted risk request is structurally invalid."""


def _result(verdict: str, reason: str, **detail) -> dict:
    """An R4 answer. The verdict and its reason always travel together."""
    if verdict not in FRISK_VERDICTS:
        raise ForecastRiskError(f"unknown R4 verdict {verdict!r}")
    payload = {
        "version": FORECAST_RISK_VERSION,
        "verdict": verdict,
        "reason": reason,
        "blocks_trades": FORECAST_RISK_BLOCKS_TRADES,
    }
    payload.update(detail)
    return payload


def confidence_of(assessment: Mapping[str, Any] | None) -> float | None:
    """The confidence F7 measured, or None when it measured nothing.

    None is a real answer and is never coerced to 0.0: "no confidence could be
    computed" and "the forecast is worthless" are different statements, and
    0.0 would silently turn the first into the second.
    """
    if assessment is None:
        return None
    if not isinstance(assessment, Mapping):
        raise ForecastRiskError("a confidence assessment must be a mapping")
    value = assessment.get("confidence")
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ForecastRiskError(
            f"confidence {value!r} is not a number"
        ) from None
    if not 0.0 <= confidence <= 1.0:
        raise ForecastRiskError(
            f"confidence {confidence!r} lies outside [0, 1]"
        )
    return confidence


def multiplier_for(confidence: float | None) -> float | None:
    """The size multiplier a confidence earns. Never above 1.0.

    Identity, capped: the confidence IS the fraction of R2's size the forecast
    has earned. A curve would need its own evidence, and none was measured;
    MEASURED, the volatility response to size was linear in the tested range
    (+19.64%, +14.72%, +9.80%, +4.89%, +0.98% at 100/75/50/25/5% of size), so
    a linear haircut tracks the risk it is meant to control.
    """
    if confidence is None:
        return None
    return min(float(confidence), FORECAST_RISK_MAX_MULTIPLIER)


def adjust_size(
    proposal: Mapping[str, Any],
    assessment: Mapping[str, Any] | None,
    *,
    min_confidence: float = FORECAST_RISK_MIN_CONFIDENCE,
    min_weight: float = SIZING_MIN_WEIGHT,
) -> dict:
    """Adjust an R2 sizing proposal by how much the forecast has earned.

    The proposal's own verdict is respected: R4 never revives a position R2
    refused, and never invents one R2 could not evaluate.
    """
    if not isinstance(proposal, Mapping):
        raise ForecastRiskError("an R2 sizing proposal is required")
    if not 0.0 < min_confidence < 1.0:
        raise ForecastRiskError("min_confidence must lie inside (0, 1)")

    ticker = proposal.get("ticker")
    sizing_verdict = proposal.get("verdict")
    proposed = proposal.get("size")

    # R2 SPOKE FIRST AND R4 DOES NOT OVERRULE IT. A refused position stays
    # refused however confident the forecast: the portfolio cannot absorb it,
    # and conviction does not change what the covariance says.
    if sizing_verdict in (SIZING_REFUSED, SIZING_NOT_EVALUATED) or proposed is None:
        return _result(
            FRISK_NOT_EVALUATED,
            (
                f"R2 returned {sizing_verdict!r} with no size to adjust; a "
                f"forecast cannot revive a position the portfolio has already "
                f"refused"
            ),
            ticker=ticker,
            sizing_verdict=sizing_verdict,
            confidence=confidence_of(assessment),
        )

    confidence = confidence_of(assessment)
    increase = proposal.get("volatility_increase")
    # R2 reports this directly; fall back to the sign of the increase so a
    # hand-built proposal without the flag still routes correctly.
    diversifying = proposal.get("diversifying")
    if diversifying is None:
        diversifying = increase is not None and float(increase) < 0

    base = {
        "ticker": ticker,
        "sizing_verdict": sizing_verdict,
        "proposed_size": round(float(proposed), 6),
        "confidence": confidence,
        "diversifying": bool(diversifying),
        "volatility_increase": increase,
    }

    # NO CONFIDENCE IS NOT LOW CONFIDENCE. Without a measurement there is
    # nothing to take a haircut from, and inventing one would be the
    # Number(x ?? 0) failure the rest of the system refuses.
    if confidence is None:
        return _result(
            FRISK_NOT_EVALUATED,
            (
                "no confidence was measured for this forecast, so there is "
                "nothing to adjust the size by. A missing confidence scored "
                "as 0.0 would refuse every unassessed trade, and scored as "
                "1.0 would wave it through"
            ),
            **base,
        )

    # A RISK REDUCER KEEPS ITS SIZE. MEASURED, applying the haircut here
    # degraded the portfolio monotonically at every confidence level tested:
    # -19.23%, -14.54%, -9.76%, -4.91%, -0.99%.
    if diversifying and FORECAST_RISK_HAIRCUT_INCREASERS_ONLY:
        return _result(
            FRISK_UNCHANGED,
            (
                f"this candidate REDUCES portfolio volatility "
                f"({float(increase):+.2%}) and keeps R2's size. MEASURED, "
                f"shrinking a risk reducer by confidence removed the benefit "
                f"rather than the risk"
                if increase is not None
                else "this candidate reduces portfolio volatility and keeps "
                "R2's size"
            ),
            adjusted_size=round(float(proposed), 6),
            multiplier=1.0,
            **base,
        )

    # BELOW THE FLOOR, THE HONEST ANSWER IS NO TRADE. A size that decays
    # smoothly toward zero produces positions too small to be real.
    if confidence < min_confidence:
        return _result(
            FRISK_NO_TRADE,
            (
                f"confidence {confidence:.3f} is below the {min_confidence:.2f} "
                f"floor, and this candidate raises portfolio risk. A scaled "
                f"size here would be a token position, not a smaller bet"
            ),
            adjusted_size=None,
            multiplier=None,
            **base,
        )

    multiplier = multiplier_for(confidence)
    adjusted = float(proposed) * multiplier

    # A HAIRCUT MUST NOT PRODUCE AN UNTRADEABLE POSITION. R2 refuses below
    # min_weight; R4 must not sneak under it by another route.
    if adjusted < min_weight:
        return _result(
            FRISK_NO_TRADE,
            (
                f"confidence {confidence:.3f} scales {float(proposed):.2%} to "
                f"{adjusted:.3%}, below the {min_weight:.1%} minimum tradeable "
                f"weight. R2 refuses a position this small and R4 does not "
                f"reach it by another route"
            ),
            adjusted_size=None,
            multiplier=round(multiplier, 6),
            **base,
        )

    verdict = FRISK_UNCHANGED if multiplier >= 1.0 else FRISK_REDUCED
    return _result(
        verdict,
        (
            f"full confidence earns R2's {float(proposed):.2%} in full"
            if verdict == FRISK_UNCHANGED
            else f"confidence {confidence:.3f} takes R2's {float(proposed):.2%} "
            f"to {adjusted:.2%} on a candidate that raises portfolio risk"
        ),
        adjusted_size=round(adjusted, 6),
        multiplier=round(multiplier, 6),
        **base,
    )


def forecast_risk_report(
    proposals: Mapping[str, Mapping[str, Any]],
    assessments: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    min_confidence: float = FORECAST_RISK_MIN_CONFIDENCE,
) -> dict:
    """Adjust a set of R2 proposals, one per candidate."""
    if not isinstance(proposals, Mapping):
        raise ForecastRiskError("a mapping of ticker to R2 proposal is required")
    assessments = assessments or {}

    adjusted: dict[str, dict] = {}
    for ticker in sorted(proposals):
        adjusted[ticker] = adjust_size(
            proposals[ticker],
            assessments.get(ticker),
            min_confidence=min_confidence,
        )

    counts: dict[str, int] = {v: 0 for v in FRISK_VERDICTS}
    for result in adjusted.values():
        counts[result["verdict"]] += 1

    return {
        "version": FORECAST_RISK_VERSION,
        "candidates": len(adjusted),
        "adjusted": adjusted,
        "counts": counts,
        "min_confidence": min_confidence,
        "blocks_trades": FORECAST_RISK_BLOCKS_TRADES,
        "note": (
            "Confidence adjusts R2's size; it never replaces the correlation "
            "analysis. MEASURED, a HIGH-confidence concentrator at 20% raised "
            "portfolio volatility +19.78% while a LOW-confidence diversifier "
            "at 5% lowered it -4.82%."
        ),
    }


def forecast_risk_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an R4 report or a single adjustment. Empty is clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("blocks_trades"):
        problems.append(
            "R4 claims to block trades — it adjusts a size, and whether to "
            "trade at all is R7's decision"
        )

    results = report.get("adjusted")
    if results is None:
        results = {report.get("ticker"): report}
    elif not isinstance(results, Mapping):
        return ["adjusted is not a mapping"]

    for ticker, result in results.items():
        if not isinstance(result, Mapping):
            problems.append(f"{ticker}: adjustment is not a mapping")
            continue
        verdict = result.get("verdict")
        if verdict not in FRISK_VERDICTS:
            problems.append(f"{ticker}: unknown verdict {verdict!r}")
            continue
        if not str(result.get("reason") or "").strip():
            problems.append(f"{ticker}: verdict with no reason")

        size = result.get("adjusted_size")
        # THE SHAPE RULE (inherited R1/R2/R3): a size exists IFF one was
        # found. 0.0 reads as "propose nothing", the same number a refusal
        # would produce, so the two must not share a representation.
        if verdict in (FRISK_NOT_EVALUATED, FRISK_NO_TRADE):
            if size is not None:
                problems.append(
                    f"{ticker}: {verdict} carried a size of {size!r} — a "
                    f"refusal and a zero-size position are different answers"
                )
        else:
            if size is None:
                problems.append(f"{ticker}: {verdict} carried no size")
            elif float(size) <= 0:
                problems.append(f"{ticker}: {verdict} carried a size of {size!r}")

        multiplier = result.get("multiplier")
        if multiplier is not None:
            if float(multiplier) > FORECAST_RISK_MAX_MULTIPLIER + 1e-9:
                problems.append(
                    f"{ticker}: multiplier {multiplier} exceeds "
                    f"{FORECAST_RISK_MAX_MULTIPLIER} — confidence is a "
                    f"haircut, not a multiplier"
                )
            if float(multiplier) < 0:
                problems.append(f"{ticker}: negative multiplier {multiplier}")
            proposed = result.get("proposed_size")
            if proposed is not None and size is not None:
                expected = float(proposed) * float(multiplier)
                if abs(expected - float(size)) > 1e-6:
                    problems.append(
                        f"{ticker}: adjusted size {size} does not equal "
                        f"{proposed} x {multiplier}"
                    )

        # A RISK REDUCER MUST NOT BE HAIRCUT.
        if result.get("diversifying") and verdict == FRISK_REDUCED:
            problems.append(
                f"{ticker}: a risk-REDUCING candidate was cut by confidence. "
                f"MEASURED, that removes the benefit rather than the risk"
            )
        if result.get("diversifying") and verdict == FRISK_NO_TRADE:
            problems.append(
                f"{ticker}: a risk-REDUCING candidate was refused on "
                f"confidence, declining a free reduction in variance"
            )

        # A SIZE MUST NEVER GROW.
        proposed = result.get("proposed_size")
        if proposed is not None and size is not None:
            if float(size) > float(proposed) + 1e-9:
                problems.append(
                    f"{ticker}: adjusted size {size} exceeds R2's "
                    f"{proposed} — R2's size is the most the portfolio can "
                    f"absorb"
                )
    return problems


def render_forecast_risk(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per candidate."""
    lines: list[str] = []
    results = report.get("adjusted")
    if not isinstance(results, Mapping):
        results = {report.get("ticker"): report}
    for ticker, result in sorted(results.items(), key=lambda kv: str(kv[0])):
        if not isinstance(result, Mapping):
            continue
        confidence = result.get("confidence")
        shown_conf = "—" if confidence is None else f"{float(confidence):.3f}"
        size = result.get("adjusted_size")
        shown_size = "—" if size is None else f"{float(size):6.2%}"
        proposed = result.get("proposed_size")
        shown_prop = "—" if proposed is None else f"{float(proposed):6.2%}"
        lines.append(
            f"  {str(ticker):8s} conf {shown_conf}   R2 {shown_prop}   "
            f"R4 {shown_size}   {result.get('verdict')}"
        )
    return lines
