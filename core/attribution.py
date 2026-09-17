"""Confounder / attribution engine (Sprint E5).

The binding rule:

    **Never automatically claim causality.** Decompose observed movement into
    market component + sector component + stock-specific component +
    event-associated residual.

E4 measures what happened. E5 asks how much of it was the market, how much
was the sector, how much was the company, and how much is left over near the
event. That leftover is EVENT-ASSOCIATED, never event-caused — a residual is
a question, not an answer.

    E4 reaction (stock / benchmark / sector returns)
        |
        v
    decompose()      stock = market + sector_excess + stock_specific
        |
        v
    attribute()      -> confounded / unexplained / event_associated /
                        inconclusive, with the reason stated

Design decisions worth stating:

- **The sector component is EXCESS over market, not the raw sector return.**
  A sector ETF already contains market beta, so subtracting the raw sector
  return would count the market twice and silently push the error into the
  residual — the one number nobody would notice was wrong. The
  decomposition is additive by construction and a test asserts it.
- **A residual must clear two bars, not one.** It must be a large SHARE of
  the movement (common factors did not explain it) and large in MAGNITUDE
  relative to the stock's own baseline volatility (it is not noise on a
  quiet day). A residual that is 90% of a 0.1% move is not a finding.
- **Overlapping events confound by default.** If another event sits in the
  window, no single one of them can claim the residual. This is the most
  common confounder in event studies and the easiest to forget, so it is
  checked rather than assumed away.
- **Every verdict is about CONFOUNDING, not significance.** `event_associated`
  means other explanations were ruled out enough to make the association
  interesting. It does not mean the event caused the move, and the phrasing
  of every output says so.
- **Missing context is `inconclusive`, never `event_associated`.** Without a
  benchmark there is no market component to remove, so the residual is just
  the raw return wearing a different name.

Pure and deterministic: no wall-clock, no network, no randomness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from core.config import (
    ATTRIBUTION_CONFOUNDED,
    ATTRIBUTION_EVENT_ASSOCIATED,
    ATTRIBUTION_INCONCLUSIVE,
    ATTRIBUTION_MAX_OVERLAPPING_EVENTS,
    ATTRIBUTION_MIN_RESIDUAL_SHARE,
    ATTRIBUTION_MIN_RESIDUAL_SIGMA,
    ATTRIBUTION_UNEXPLAINED,
    EVENT_ATTRIBUTION_VERSION,
)

_CAUSALITY_NOTE = (
    "event-associated, NOT event-caused: a residual is movement other factors "
    "did not explain, which is a question rather than an answer "
    "(master context section 37)"
)


class AttributionError(ValueError):
    """Raised when a decomposition cannot be performed as requested."""


@dataclass
class Decomposition:
    """One horizon's movement split into additive components."""

    horizon: str
    stock_return: float
    market: float | None = None
    sector_excess: float | None = None
    stock_specific: float | None = None
    residual_share: float | None = None
    residual_sigma: float | None = None
    components_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_additive(self, tolerance: float = 1e-8) -> bool:
        """Whether the components sum back to the observed movement."""
        if not self.components_available:
            return False
        total = (self.market or 0.0) + (self.sector_excess or 0.0) + (
            self.stock_specific or 0.0
        )
        return abs(total - self.stock_return) <= tolerance


@dataclass
class Attribution:
    """A decomposition plus a confounding verdict."""

    horizon: str
    verdict: str
    decomposition: Decomposition
    reason: str = ""
    overlapping_events: list[str] = field(default_factory=list)
    attribution_version: str = EVENT_ATTRIBUTION_VERSION
    causality_note: str = _CAUSALITY_NOTE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decomposition"] = self.decomposition.to_dict()
        return payload

    def is_event_associated(self) -> bool:
        return self.verdict == ATTRIBUTION_EVENT_ASSOCIATED


def decompose(
    stock_return: float,
    market_return: float | None,
    sector_return: float | None = None,
    horizon: str = "",
    baseline_volatility: float | None = None,
    sessions: int = 1,
) -> Decomposition:
    """Split an observed movement into additive components.

        stock = market + sector_excess + stock_specific

    `sector_excess` is the sector's return BEYOND the market. Using the raw
    sector return would count market beta twice, because a sector ETF already
    contains it, and the surplus would land in the residual.

    Without a market return there is no decomposition to make: the residual
    would just be the raw return relabelled, so the components stay absent.
    """
    if market_return is None:
        return Decomposition(
            horizon=horizon,
            stock_return=round(float(stock_return), 8),
            components_available=False,
        )

    # Round FIRST, then derive the residual from the rounded parts. Rounding
    # each component independently leaves up to ~1e-8 of error that the
    # residual silently absorbs, which breaks exact additivity — found by a
    # property test over random shapes rather than by inspection.
    observed = round(float(stock_return), 8)
    market = round(float(market_return), 8)
    sector_excess = (
        round(float(sector_return) - float(market_return), 8)
        if sector_return is not None else 0.0
    )
    stock_specific = round(observed - market - sector_excess, 8)

    # Share of the TOTAL absolute movement that the components did not
    # explain. Absolute terms, because a +3% market and a -3% stock-specific
    # move is a large decomposition, not a quiet one.
    magnitudes = abs(market) + abs(sector_excess) + abs(stock_specific)
    residual_share = (
        round(abs(stock_specific) / magnitudes, 6) if magnitudes > 0 else None
    )

    # Magnitude relative to the stock's own normal daily dispersion, scaled
    # over the window. A residual that is a large share of a tiny move is
    # noise on a quiet day.
    residual_sigma = None
    if baseline_volatility:
        window_sigma = float(baseline_volatility) * max(sessions, 1) ** 0.5
        if window_sigma > 0:
            residual_sigma = round(abs(stock_specific) / window_sigma, 6)

    return Decomposition(
        horizon=horizon,
        stock_return=observed,
        market=market,
        sector_excess=sector_excess,
        stock_specific=stock_specific,
        residual_share=residual_share,
        residual_sigma=residual_sigma,
        components_available=True,
    )


def attribute(
    decomposition: Decomposition,
    overlapping_events: Iterable[str] | None = None,
) -> Attribution:
    """Judge whether a residual can be associated with the event.

    Every verdict is about CONFOUNDING, never significance. `event_associated`
    means common factors and overlapping events were ruled out enough to make
    the association worth recording — it does not mean the event caused the
    move, and the causality note travels with the answer.
    """
    overlaps = sorted(str(event) for event in (overlapping_events or []))

    if not decomposition.components_available:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_INCONCLUSIVE,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason=(
                "no market return available, so there is no market component to "
                "remove — the residual would be the raw return under a different "
                "name"
            ),
        )

    if len(overlaps) > ATTRIBUTION_MAX_OVERLAPPING_EVENTS:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_CONFOUNDED,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason=(
                f"{len(overlaps)} other event(s) fall inside this window "
                f"({overlaps[:3]}) — no single event can claim the residual"
            ),
        )

    share = decomposition.residual_share
    sigma = decomposition.residual_sigma

    if share is None:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_INCONCLUSIVE,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason="the movement is exactly zero; there is nothing to attribute",
        )

    if share < ATTRIBUTION_MIN_RESIDUAL_SHARE:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_CONFOUNDED,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason=(
                f"market and sector explain most of the movement (residual is "
                f"{share:.1%} of it, below {ATTRIBUTION_MIN_RESIDUAL_SHARE:.0%}) — "
                f"this looks like a common-factor move, not a company-specific one"
            ),
        )

    if sigma is None:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_UNEXPLAINED,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason=(
                "the residual dominates, but without a baseline volatility there "
                "is no way to tell a real move from a quiet-day flicker"
            ),
        )

    if sigma < ATTRIBUTION_MIN_RESIDUAL_SIGMA:
        return Attribution(
            horizon=decomposition.horizon,
            verdict=ATTRIBUTION_UNEXPLAINED,
            decomposition=decomposition,
            overlapping_events=overlaps,
            reason=(
                f"the residual is {share:.1%} of the movement but only "
                f"{sigma:.2f} baseline sigma (below "
                f"{ATTRIBUTION_MIN_RESIDUAL_SIGMA}) — a large share of a small "
                f"move is noise"
            ),
        )

    return Attribution(
        horizon=decomposition.horizon,
        verdict=ATTRIBUTION_EVENT_ASSOCIATED,
        decomposition=decomposition,
        overlapping_events=overlaps,
        reason=(
            f"the residual is {share:.1%} of the movement and {sigma:.2f} "
            f"baseline sigma, with no overlapping events — common factors do "
            f"not explain it"
        ),
    )


def attribute_study(
    study,
    overlapping_events: Iterable[str] | None = None,
) -> dict[str, Attribution]:
    """Decompose and attribute every horizon of an E4 study.

    Reads the baseline volatility the study already computed, so the sigma
    test uses the same definition of "normal" the study did.
    """
    if not getattr(study, "reactions", None):
        raise AttributionError(
            f"study for {getattr(study, 'ticker', '?')} has no reactions to attribute"
        )

    baseline_volatility = (getattr(study, "baseline", {}) or {}).get("daily_volatility")
    attributions: dict[str, Attribution] = {}
    for horizon, reaction in study.reactions.items():
        decomposition = decompose(
            stock_return=reaction["stock_return"],
            market_return=reaction.get("benchmark_return"),
            sector_return=reaction.get("sector_return"),
            horizon=horizon,
            baseline_volatility=baseline_volatility,
            sessions=max(int(reaction.get("sessions", 1)), 1),
        )
        attributions[horizon] = attribute(decomposition, overlapping_events)
    return attributions


def find_overlapping_events(
    event,
    all_events: Iterable[Any],
    sessions: int,
    sessions_per_day: float = 1.0,
) -> list[str]:
    """Other events for the same entity inside this event's window.

    Overlapping events are the most common confounder in an event study and
    the easiest to forget. Same entity only: an unrelated company's news is
    not a confounder for this one.
    """
    import pandas as pd

    target_id = getattr(event, "event_id", None)
    entity = str(getattr(event, "entity", "")).upper()
    try:
        start = pd.Timestamp(getattr(event, "published_time", ""))
    except (ValueError, TypeError):
        return []
    end = start + pd.Timedelta(days=float(sessions) / sessions_per_day)

    overlapping: list[str] = []
    for other in all_events:
        if getattr(other, "event_id", None) == target_id:
            continue
        if str(getattr(other, "entity", "")).upper() != entity:
            continue
        try:
            stamp = pd.Timestamp(getattr(other, "published_time", ""))
        except (ValueError, TypeError):
            continue
        if start < stamp <= end:
            overlapping.append(str(getattr(other, "event_id", "")))
    return sorted(overlapping)


def attribution_report(attributions: Iterable[Attribution]) -> dict[str, Any]:
    """Coverage summary over a batch of attributions.

    The share that is `confounded` is the number worth watching: a high one
    means most apparent reactions are common-factor moves, which is the
    normal and healthy finding rather than a defect.
    """
    items = list(attributions)
    by_verdict: dict[str, int] = {}
    for attribution in items:
        by_verdict[attribution.verdict] = by_verdict.get(attribution.verdict, 0) + 1
    associated = [a for a in items if a.is_event_associated()]
    return {
        "attribution_version": EVENT_ATTRIBUTION_VERSION,
        "total": len(items),
        "event_associated": len(associated),
        "by_verdict": dict(sorted(by_verdict.items())),
        "association_rate": round(len(associated) / len(items), 4) if items else 0.0,
        "min_residual_share": ATTRIBUTION_MIN_RESIDUAL_SHARE,
        "min_residual_sigma": ATTRIBUTION_MIN_RESIDUAL_SIGMA,
        "causality_note": _CAUSALITY_NOTE,
    }
