"""Event revision / contradiction learning (Sprint E7).

Stores event chains — initial claim → correction → confirmation → reversal —
because **the evolution itself is training information**.

A story that was reported, corrected, then reversed is a fundamentally
different kind of evidence from one reported once that stood. Treating the
four reports as four independent events would:

- quadruple-count a single story in every base rate;
- hide that the market learned the claim was wrong;
- let a reversed claim keep the weight of a confirmed one.

    events (E1)
        |
        v
    build_chains()      group by entity + type inside a window
        |
        v
    EventChain          links classified, status settled
        |
        +-- reliability_weight()   reversed weighs less than unconfirmed
        +-- chain_report()         how often this source gets reversed

Design decisions worth stating:

- **A reversal weighs LESS than an unconfirmed claim.** A reversed story is
  not merely uninformative — it is positive evidence the source got it
  wrong. Treating it as neutral would lose the most useful signal in the
  chain.
- **`contested` is a real state.** A claim both confirmed and reversed is
  unresolved, not an average of the two. Collapsing it toward the middle
  would be exactly the "contradictory evidence averaging to neutral" failure
  N1 already refuses at the article level.
- **Classification is conservative.** Only an explicit directional flip is a
  reversal; same-direction repetition is a confirmation or duplicate. When
  the relationship is unclear the link is `duplicate`, which carries no new
  information rather than an invented one.
- **The chain is keyed on the INITIAL claim.** Downstream learning should
  count one story once, at the moment it first became knowable, with its
  eventual fate attached.

Pure and deterministic: no wall-clock, no network, no randomness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from core.config import (
    CHAIN_LINKS,
    CHAIN_LINK_CONFIRMATION,
    CHAIN_LINK_CORRECTION,
    CHAIN_LINK_DUPLICATE,
    CHAIN_LINK_INITIAL,
    CHAIN_LINK_REVERSAL,
    CHAIN_RELIABILITY_WEIGHT,
    CHAIN_STATUS_CONFIRMED,
    CHAIN_STATUS_CONTESTED,
    CHAIN_STATUS_CORRECTED,
    CHAIN_STATUS_OPEN,
    CHAIN_STATUS_REVERSED,
    EVENT_CHAIN_VERSION,
    EVENT_CHAIN_WINDOW_DAYS,
    EVENT_DIRECTION_CONTRADICTORY,
    EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_NEUTRAL,
    EVENT_DIRECTION_POSITIVE,
)


class EventChainError(ValueError):
    """Raised when a chain cannot be built or is invalid."""


@dataclass
class ChainLink:
    """One event's place in a chain."""

    event_id: str
    published_time: str
    link_type: str
    direction: str
    source: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventChain:
    """One story and everything that happened to it."""

    chain_id: str
    ticker: str
    event_type: str
    initial_time: str
    initial_direction: str
    links: list[ChainLink] = field(default_factory=list)
    status: str = CHAIN_STATUS_OPEN
    chain_version: str = EVENT_CHAIN_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["links"] = [link.to_dict() for link in self.links]
        return payload

    def reliability_weight(self) -> float:
        """How much this chain's claim should count.

        A reversed claim weighs less than an unconfirmed one: a reversal is
        positive evidence the source got it wrong, not an absence of
        evidence.
        """
        return CHAIN_RELIABILITY_WEIGHT[self.status]

    def was_revised(self) -> bool:
        """Whether the story changed after its initial claim."""
        return any(
            link.link_type in (CHAIN_LINK_CORRECTION, CHAIN_LINK_REVERSAL)
            for link in self.links
        )

    def link_counts(self) -> dict[str, int]:
        counts = {link_type: 0 for link_type in CHAIN_LINKS}
        for link in self.links:
            counts[link.link_type] += 1
        return counts


def _opposed(first: str, second: str) -> bool:
    """Whether two directions point opposite ways.

    CONTRADICTORY is not an opposite of anything — it is already an unresolved
    state, and pairing it with a direction would invent a flip that the
    evidence does not contain.
    """
    directional = {EVENT_DIRECTION_POSITIVE, EVENT_DIRECTION_NEGATIVE}
    return first in directional and second in directional and first != second


def classify_link(earlier, later) -> tuple[str, str]:
    """How `later` relates to `earlier`. Returns (link_type, reason).

    Conservative by design: only an explicit directional flip is a reversal,
    and an unclear relationship is a `duplicate` carrying no new information
    rather than an invented one.
    """
    earlier_direction = str(getattr(earlier, "direction", EVENT_DIRECTION_NEUTRAL))
    later_direction = str(getattr(later, "direction", EVENT_DIRECTION_NEUTRAL))

    if _opposed(earlier_direction, later_direction):
        return (
            CHAIN_LINK_REVERSAL,
            f"direction flipped from {earlier_direction} to {later_direction} — "
            f"the earlier claim was wrong",
        )

    if later_direction == EVENT_DIRECTION_CONTRADICTORY:
        return (
            CHAIN_LINK_CORRECTION,
            "the later report is internally contradictory, which amends the "
            "original claim without reversing it",
        )

    earlier_source = str(getattr(earlier, "source", ""))
    later_source = str(getattr(later, "source", ""))
    if later_source and later_source != earlier_source:
        return (
            CHAIN_LINK_CONFIRMATION,
            f"independent corroboration from {later_source!r} in the same "
            f"direction",
        )

    return (
        CHAIN_LINK_DUPLICATE,
        "same source, same direction — a repetition carrying no new information",
    )


def _settle_status(links: list[ChainLink]) -> str:
    """The chain's settled state, read from its links.

    `contested` is a real state, not an average: a claim both confirmed and
    reversed is unresolved, and collapsing it toward the middle would repeat
    the "contradictory evidence averages to neutral" failure N1 refuses at
    the article level.
    """
    types = {link.link_type for link in links}
    reversed_ = CHAIN_LINK_REVERSAL in types
    confirmed = CHAIN_LINK_CONFIRMATION in types

    if reversed_ and confirmed:
        return CHAIN_STATUS_CONTESTED
    if reversed_:
        return CHAIN_STATUS_REVERSED
    if CHAIN_LINK_CORRECTION in types:
        return CHAIN_STATUS_CORRECTED
    if confirmed:
        return CHAIN_STATUS_CONFIRMED
    return CHAIN_STATUS_OPEN


def build_chains(
    events: Iterable[Any],
    window_days: int = EVENT_CHAIN_WINDOW_DAYS,
) -> list[EventChain]:
    """Group events into chains by entity, type and time proximity.

    Events beyond the window are separate stories rather than an evolution of
    one, so the window is what stops an entire year of earnings reports
    collapsing into a single chain.
    """
    import pandas as pd

    if window_days < 1:
        raise EventChainError("the chain window must be at least a day")

    ordered = []
    for event in events:
        try:
            stamp = pd.Timestamp(getattr(event, "published_time", ""))
        except (ValueError, TypeError):
            # An unparseable timestamp cannot be placed in a sequence, and
            # guessing its position would invent an evolution.
            continue
        ordered.append((stamp, event))
    ordered.sort(key=lambda pair: (pair[0], str(getattr(pair[1], "event_id", ""))))

    chains: list[EventChain] = []
    # Per key: (chain, initial_stamp, previous_event). The window is measured
    # from the INITIAL claim so a chain cannot grow without bound, but each
    # link is classified against the PREVIOUS one so the story is read as a
    # sequence. Classifying everything against the initial claim made a
    # corroborated reversal look like a second reversal.
    open_chains: dict[tuple[str, str], tuple[EventChain, Any, Any]] = {}

    for stamp, event in ordered:
        key = (
            str(getattr(event, "entity", "")).upper(),
            str(getattr(event, "event_type", "other")),
        )
        existing = open_chains.get(key)

        if existing is not None:
            chain, initial_stamp, previous_event = existing
            if (stamp - initial_stamp).days <= window_days:
                link_type, reason = classify_link(previous_event, event)
                chain.links.append(ChainLink(
                    event_id=str(getattr(event, "event_id", "")),
                    published_time=str(getattr(event, "published_time", "")),
                    link_type=link_type,
                    direction=str(getattr(event, "direction", EVENT_DIRECTION_NEUTRAL)),
                    source=str(getattr(event, "source", "")),
                    reason=reason,
                ))
                chain.status = _settle_status(chain.links)
                open_chains[key] = (chain, initial_stamp, event)
                continue

        chain = EventChain(
            chain_id=str(getattr(event, "event_id", "")),
            ticker=key[0],
            event_type=key[1],
            initial_time=str(getattr(event, "published_time", "")),
            initial_direction=str(getattr(event, "direction", EVENT_DIRECTION_NEUTRAL)),
            links=[ChainLink(
                event_id=str(getattr(event, "event_id", "")),
                published_time=str(getattr(event, "published_time", "")),
                link_type=CHAIN_LINK_INITIAL,
                direction=str(getattr(event, "direction", EVENT_DIRECTION_NEUTRAL)),
                source=str(getattr(event, "source", "")),
                reason="first report of this story",
            )],
        )
        chains.append(chain)
        open_chains[key] = (chain, stamp, event)

    return chains


def chain_problems(chain: EventChain) -> list[str]:
    """Validate a built chain."""
    problems: list[str] = []
    if not chain.chain_id:
        problems.append("chain_id is required")
    if not chain.links:
        problems.append("a chain must contain at least its initial claim")
    elif chain.links[0].link_type != CHAIN_LINK_INITIAL:
        problems.append(
            "the first link must be the initial claim — a chain is keyed on the "
            "moment the story first became knowable"
        )
    for index, link in enumerate(chain.links):
        if link.link_type not in CHAIN_LINKS:
            problems.append(f"link[{index}] has unknown type {link.link_type!r}")
        if index > 0 and link.link_type == CHAIN_LINK_INITIAL:
            problems.append(f"link[{index}] is a second initial claim")
    if chain.status != _settle_status(chain.links):
        problems.append(
            f"status {chain.status!r} disagrees with the links "
            f"({_settle_status(chain.links)!r})"
        )
    return problems


def deduplicate_events(events: Iterable[Any], chains: list[EventChain]) -> list[str]:
    """The event ids that should count once — one per chain.

    Counting every report of a story independently would multiply a single
    claim through every base rate downstream. This returns the initial claim
    of each chain, which is the moment the story first became knowable.
    """
    return [chain.chain_id for chain in chains]


def chain_report(chains: list[EventChain]) -> dict[str, Any]:
    """Summary over a batch of chains.

    `reversal_rate` is the number worth watching: it is how often this
    source's claims turn out to be wrong, which is the training information
    E7 exists to expose.
    """
    by_status: dict[str, int] = {}
    revised = 0
    for chain in chains:
        by_status[chain.status] = by_status.get(chain.status, 0) + 1
        if chain.was_revised():
            revised += 1
    reversed_count = by_status.get(CHAIN_STATUS_REVERSED, 0) + by_status.get(
        CHAIN_STATUS_CONTESTED, 0
    )
    return {
        "chain_version": EVENT_CHAIN_VERSION,
        "chains": len(chains),
        "events": sum(len(chain.links) for chain in chains),
        "by_status": dict(sorted(by_status.items())),
        "revised": revised,
        "reversal_rate": (
            round(reversed_count / len(chains), 4) if chains else 0.0
        ),
        "window_days": EVENT_CHAIN_WINDOW_DAYS,
        "detail": (
            "a chain counts a story once; the reversal rate is how often this "
            "source's claims turn out to be wrong"
        ),
    }
