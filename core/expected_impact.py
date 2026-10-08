"""R5 expected portfolio impact — what the portfolio looks like afterwards.

R2 answered "how much of this can the portfolio absorb?" and R4 answered "how
much has the forecast earned?". Neither answers what the portfolio actually
LOOKS LIKE after the trade, which is the question a decision rests on.

**THE DECIDING MEASUREMENT: volatility and concentration disagree about half
the time.** Across 20 seeded markets and 2 candidates, 21 of 40 trades — 52%
— moved portfolio volatility and risk concentration in OPPOSITE directions.
On the reference market a correlated candidate raised volatility +5.00% while
*improving* risk concentration (risk-HHI 0.5000 -> 0.3736), and a diversifier
cut volatility -20.56% while leaving concentration essentially untouched
(0.5005). A single risk number cannot answer both questions.

**So R5 reports both and rules on neither.** Each dimension carries its own
direction and its own magnitude. Collapsing them into one verdict would mean
silently picking a winner on the 52% of trades where they disagree, and the
choice would be invisible to the reader. R7 weighs them with the whole
portfolio in view.

**R5 PROJECTS RISK, NEVER RETURN.** There is no trained forecasting model:
`build_joint_forecast` reports UNAVAILABLE, and the snapshot contract already
records that `expected_return` "needs a trained model, and none exists". Risk
is measurable from the covariance; return is not. An expected-return impact
would be a confident claim about a quantity nobody computed — and a 0.0 would
render as "flat" under the `Number(x ?? 0)` idiom, which is the precise hazard
F3, F4, F5, F6 and the snapshot contract each guard against.

**The before/after comparison is made on ONE covariance.** Both states are
evaluated against the same matrix, so a difference is attributable to the
trade rather than to a change in the estimation window. Comparing a
before-state on one window with an after-state on another would measure the
window.

**Impact is projected, not enforced.** Whether the trade happens is R7's
decision.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.config import (
    EXPECTED_IMPACT_VERSION,
    IMPACT_BLOCKS_TRADES,
    IMPACT_DIMENSION_CONCENTRATION,
    IMPACT_DIMENSION_LARGEST,
    IMPACT_DIMENSION_VOLATILITY,
    IMPACT_DIMENSIONS,
    IMPACT_DIRECTIONS,
    IMPACT_IMPROVES,
    IMPACT_MATERIAL_CONCENTRATION,
    IMPACT_MATERIAL_VOLATILITY,
    IMPACT_NEUTRAL,
    IMPACT_NOT_EVALUATED,
    IMPACT_PROJECTS_RETURN,
    IMPACT_RETURN_REASON,
    IMPACT_WORSENS,
)
from core.position_exposure import (
    portfolio_variance,
    risk_contributions,
)


class ExpectedImpactError(ValueError):
    """Raised when an impact projection request is structurally invalid."""


def blend(held: Mapping[str, float], ticker: str, size: float) -> dict[str, float]:
    """The portfolio after adding `size` of `ticker`, funded pro rata.

    Existing holdings are scaled by (1 - size) rather than one of them being
    sold, because which holding funds the trade is a separate decision with
    its own consequences. Pro rata keeps the comparison about the CANDIDATE.
    """
    if not 0.0 <= size < 1.0:
        raise ExpectedImpactError(f"size {size!r} must lie in [0, 1)")
    ticker = str(ticker).upper()
    blended = {t: float(w) * (1.0 - size) for t, w in held.items() if t != ticker}
    existing = float(held.get(ticker, 0.0)) * (1.0 - size)
    blended[ticker] = existing + size
    return blended


def concentration_of(shares: Mapping[str, float]) -> float | None:
    """Herfindahl index over risk contributions. None when there is nothing.

    None rather than 0.0 for the reason R3 gives: an index over an empty set
    reads as perfect diversification, which is the opposite of unknown.
    """
    values = [float(v) for v in shares.values() if v is not None]
    if not values:
        return None
    return round(sum(v * v for v in values), 8)


def _state(
    weights: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> dict:
    """The three measurable properties of a portfolio state."""
    variance = portfolio_variance(weights, tickers, cov)
    shares = risk_contributions(weights, tickers, cov)
    concentration = concentration_of(shares)
    largest = max(shares.values()) if shares else None
    return {
        "volatility": round(math.sqrt(max(variance, 0.0)), 8),
        "risk_concentration": concentration,
        "largest_risk_share": None if largest is None else round(float(largest), 6),
        "risk_contributions": {t: round(float(v), 6) for t, v in sorted(shares.items())},
    }


def _direction(
    before: float | None, after: float | None, material: float
) -> tuple[str, float | None]:
    """Which way this dimension moved, and by how much.

    Lower is better on every dimension R5 projects — less volatility, less
    concentration, a smaller largest share — so a decrease IMPROVES.
    """
    if before is None or after is None:
        return IMPACT_NOT_EVALUATED, None
    change = float(after) - float(before)
    if abs(change) < material:
        return IMPACT_NEUTRAL, round(change, 8)
    return (IMPACT_IMPROVES if change < 0 else IMPACT_WORSENS), round(change, 8)


def project_impact(
    held: Mapping[str, float],
    ticker: str,
    size: float,
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> dict:
    """What the portfolio looks like after adding `size` of `ticker`.

    Both states are measured against the SAME covariance, so any difference is
    attributable to the trade rather than to a change in the estimation window.
    """
    if not isinstance(held, Mapping):
        raise ExpectedImpactError("a mapping of held weights is required")
    ticker = str(ticker).upper()
    if ticker not in tickers:
        raise ExpectedImpactError(
            f"{ticker!r} has no return history here, so its effect on the "
            f"portfolio cannot be measured"
        )

    before = _state(held, tickers, cov)
    after = _state(blend(held, ticker, size), tickers, cov)

    if before["volatility"] <= 0:
        return {
            "version": EXPECTED_IMPACT_VERSION,
            "ticker": ticker,
            "size": round(float(size), 6),
            "before": before,
            "after": after,
            "dimensions": {
                name: {
                    "direction": IMPACT_NOT_EVALUATED,
                    "before": before.get(name),
                    "after": after.get(name),
                    "change": None,
                    "relative_change": None,
                    "reason": (
                        "the held portfolio has no measurable volatility, so "
                        "a relative impact has nothing to be relative to"
                    ),
                }
                for name in IMPACT_DIMENSIONS
            },
            "projects_return": IMPACT_PROJECTS_RETURN,
            "return_reason": IMPACT_RETURN_REASON,
            "blocks_trades": IMPACT_BLOCKS_TRADES,
            "note": _NOTE,
        }

    # Volatility is judged on its RELATIVE move, matching R2's risk budget, so
    # the two modules agree about what "a lot" means. Concentration is judged
    # on its ABSOLUTE move, because HHI is already a unitless share.
    vol_change = after["volatility"] - before["volatility"]
    vol_relative = vol_change / before["volatility"]
    if abs(vol_relative) < IMPACT_MATERIAL_VOLATILITY:
        vol_direction = IMPACT_NEUTRAL
    else:
        vol_direction = IMPACT_IMPROVES if vol_relative < 0 else IMPACT_WORSENS

    conc_direction, conc_change = _direction(
        before["risk_concentration"],
        after["risk_concentration"],
        IMPACT_MATERIAL_CONCENTRATION,
    )
    largest_direction, largest_change = _direction(
        before["largest_risk_share"],
        after["largest_risk_share"],
        IMPACT_MATERIAL_CONCENTRATION,
    )

    dimensions = {
        IMPACT_DIMENSION_VOLATILITY: {
            "direction": vol_direction,
            "before": before["volatility"],
            "after": after["volatility"],
            "change": round(vol_change, 8),
            "relative_change": round(vol_relative, 6),
            "threshold": IMPACT_MATERIAL_VOLATILITY,
            "reason": (
                f"portfolio volatility moves {vol_relative:+.2%}, "
                f"{'inside' if vol_direction == IMPACT_NEUTRAL else 'beyond'} "
                f"the {IMPACT_MATERIAL_VOLATILITY:.0%} materiality threshold"
            ),
        },
        IMPACT_DIMENSION_CONCENTRATION: {
            "direction": conc_direction,
            "before": before["risk_concentration"],
            "after": after["risk_concentration"],
            "change": conc_change,
            "relative_change": None,
            "threshold": IMPACT_MATERIAL_CONCENTRATION,
            "reason": (
                f"risk concentration moves {conc_change:+.4f} on the "
                f"Herfindahl scale. MEASURED, this disagrees with the "
                f"volatility direction on 52% of trades, which is why both "
                f"are reported"
                if conc_change is not None
                else "risk concentration could not be measured"
            ),
        },
        IMPACT_DIMENSION_LARGEST: {
            "direction": largest_direction,
            "before": before["largest_risk_share"],
            "after": after["largest_risk_share"],
            "change": largest_change,
            "relative_change": None,
            "threshold": IMPACT_MATERIAL_CONCENTRATION,
            "reason": (
                f"the largest single risk share moves {largest_change:+.4f}"
                if largest_change is not None
                else "the largest risk share could not be measured"
            ),
        },
    }

    return {
        "version": EXPECTED_IMPACT_VERSION,
        "ticker": ticker,
        "size": round(float(size), 6),
        "before": before,
        "after": after,
        "dimensions": dimensions,
        "disagreement": _disagreement(dimensions),
        "projects_return": IMPACT_PROJECTS_RETURN,
        "return_reason": IMPACT_RETURN_REASON,
        "blocks_trades": IMPACT_BLOCKS_TRADES,
        "note": _NOTE,
    }


_NOTE = (
    "R5 projects RISK, never RETURN: no trained model exists, and a 0.0 "
    "return impact would render as 'flat' to any consumer that coalesces "
    "nulls. It reports each dimension separately and rules on none, because "
    "MEASURED, volatility and concentration moved in opposite directions on "
    "21 of 40 trades."
)


def _disagreement(dimensions: Mapping[str, Mapping[str, Any]]) -> dict:
    """Whether the dimensions point different ways, stated explicitly.

    This is the finding, not a footnote: a reader who sees only "volatility
    improves" would act on half the evidence.
    """
    directions = {
        name: detail.get("direction")
        for name, detail in dimensions.items()
        if detail.get("direction") in (IMPACT_IMPROVES, IMPACT_WORSENS)
    }
    improving = sorted(n for n, d in directions.items() if d == IMPACT_IMPROVES)
    worsening = sorted(n for n, d in directions.items() if d == IMPACT_WORSENS)
    disagree = bool(improving and worsening)
    return {
        "disagree": disagree,
        "improving": improving,
        "worsening": worsening,
        "reason": (
            f"this trade IMPROVES {improving} and WORSENS {worsening}. "
            f"MEASURED, that happens on 52% of trades, so neither direction "
            f"alone describes the outcome"
            if disagree
            else "every material dimension points the same way"
        ),
    }


def impact_report(
    held: Mapping[str, float],
    proposals: Mapping[str, Mapping[str, Any]],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> dict:
    """Project the impact of each sized proposal, one per candidate."""
    if not isinstance(proposals, Mapping):
        raise ExpectedImpactError("a mapping of ticker to proposal is required")

    projections: dict[str, dict] = {}
    for ticker in sorted(proposals):
        proposal = proposals[ticker] or {}
        size = proposal.get("adjusted_size")
        if size is None:
            size = proposal.get("size")
        if size is None:
            # NO SIZE, NO PROJECTION. Projecting a zero-size trade would
            # report "no impact", which is indistinguishable from a trade
            # that genuinely changes nothing.
            projections[ticker] = {
                "version": EXPECTED_IMPACT_VERSION,
                "ticker": ticker,
                "size": None,
                "dimensions": {
                    name: {
                        "direction": IMPACT_NOT_EVALUATED,
                        "before": None,
                        "after": None,
                        "change": None,
                        "relative_change": None,
                        "reason": (
                            "no size was proposed, so there is no trade to "
                            "project. A zero-size projection would report "
                            "'no impact', which is a different claim"
                        ),
                    }
                    for name in IMPACT_DIMENSIONS
                },
                "projects_return": IMPACT_PROJECTS_RETURN,
                "return_reason": IMPACT_RETURN_REASON,
                "blocks_trades": IMPACT_BLOCKS_TRADES,
                "note": _NOTE,
            }
            continue
        projections[ticker] = project_impact(held, ticker, float(size), tickers, cov)

    disagreeing = sorted(
        t
        for t, p in projections.items()
        if (p.get("disagreement") or {}).get("disagree")
    )
    return {
        "version": EXPECTED_IMPACT_VERSION,
        "candidates": len(projections),
        "projections": projections,
        "dimensions": list(IMPACT_DIMENSIONS),
        "disagreeing": disagreeing,
        "projects_return": IMPACT_PROJECTS_RETURN,
        "return_reason": IMPACT_RETURN_REASON,
        "blocks_trades": IMPACT_BLOCKS_TRADES,
        "note": _NOTE,
    }


def impact_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an impact report or a single projection."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("blocks_trades"):
        problems.append(
            "R5 claims to block trades — it projects an impact, and whether "
            "to trade is R7's decision"
        )
    if report.get("projects_return"):
        problems.append(
            "R5 claims to project an expected return — no trained model "
            "exists, and a fabricated 0.0 renders as 'flat' to any consumer "
            "that coalesces nulls"
        )
    if not str(report.get("return_reason") or "").strip():
        problems.append("the refusal to project return carries no reason")

    projections = report.get("projections")
    if projections is None:
        projections = {report.get("ticker"): report}
    elif not isinstance(projections, Mapping):
        return ["projections is not a mapping"]

    for ticker, projection in projections.items():
        if not isinstance(projection, Mapping):
            problems.append(f"{ticker}: projection is not a mapping")
            continue
        dimensions = projection.get("dimensions") or {}
        for name in IMPACT_DIMENSIONS:
            if name not in dimensions:
                problems.append(
                    f"{ticker}: dimension {name!r} is missing — MEASURED, "
                    f"volatility and concentration disagree on 52% of trades, "
                    f"so each must be reported"
                )
                continue
            detail = dimensions[name]
            direction = detail.get("direction")
            if direction not in IMPACT_DIRECTIONS:
                problems.append(f"{ticker}/{name}: unknown direction {direction!r}")
            if not str(detail.get("reason") or "").strip():
                problems.append(f"{ticker}/{name}: direction with no reason")
            # THE SHAPE RULE: a change exists IFF it was measured.
            if direction == IMPACT_NOT_EVALUATED:
                if detail.get("change") is not None:
                    problems.append(
                        f"{ticker}/{name}: NOT_EVALUATED carried a change of "
                        f"{detail.get('change')!r}"
                    )
            elif detail.get("change") is None:
                problems.append(f"{ticker}/{name}: {direction} carried no change")
            # DIRECTION MUST MATCH THE SIGN. Lower is better on every
            # dimension R5 projects.
            change = detail.get("change")
            if change is not None:
                if direction == IMPACT_IMPROVES and float(change) > 0:
                    problems.append(
                        f"{ticker}/{name}: IMPROVES with a +{change} change — "
                        f"lower is better on every dimension R5 projects"
                    )
                if direction == IMPACT_WORSENS and float(change) < 0:
                    problems.append(
                        f"{ticker}/{name}: WORSENS with a {change} change"
                    )

        # A DISAGREEMENT MUST BE STATED, not left for the reader to spot.
        disagreement = projection.get("disagreement")
        if disagreement is not None:
            directions = {
                n: d.get("direction") for n, d in dimensions.items()
            }
            improving = any(d == IMPACT_IMPROVES for d in directions.values())
            worsening = any(d == IMPACT_WORSENS for d in directions.values())
            if bool(improving and worsening) != bool(disagreement.get("disagree")):
                problems.append(
                    f"{ticker}: the dimensions disagree but the report does "
                    f"not say so — a reader seeing one direction would act on "
                    f"half the evidence"
                )
    return problems


def render_impact(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one block per candidate."""
    lines: list[str] = []
    projections = report.get("projections")
    if not isinstance(projections, Mapping):
        projections = {report.get("ticker"): report}
    for ticker, projection in sorted(projections.items(), key=lambda kv: str(kv[0])):
        if not isinstance(projection, Mapping):
            continue
        size = projection.get("size")
        shown = "—" if size is None else f"{float(size):.2%}"
        lines.append(f"  {str(ticker):8s} size {shown}")
        for name in IMPACT_DIMENSIONS:
            detail = (projection.get("dimensions") or {}).get(name) or {}
            change = detail.get("change")
            shown_change = "—" if change is None else f"{float(change):+.4f}"
            lines.append(
                f"      {name:20s} {str(detail.get('direction')):14s} "
                f"{shown_change}"
            )
        disagreement = projection.get("disagreement") or {}
        if disagreement.get("disagree"):
            lines.append(
                f"      !! improves {disagreement.get('improving')} but "
                f"worsens {disagreement.get('worsening')}"
            )
    return lines
