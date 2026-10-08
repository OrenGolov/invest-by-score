"""Influential person intelligence (Sprint E3).

E3 tracks WHO speaks: identity, role, organization, historical relevance,
topic specialization, source credibility, statement frequency, novelty and
historical market impact. The eventual goal is to learn

    actor x topic x company/sector  ->  historical market response

That learning belongs to E4 (event study) and E6 (event memory), which
measure what actually happened. E3's job is to make the actor a first-class,
versioned object so those measurements have something stable to attach to —
and, just as importantly, to refuse to pretend it has measured anything yet.

    ActorRecord (curated identity)
        |
        v
    ActorProfile  <- observations accrued from events
        |
        +-- credibility()          prior, or measured once there is a record
        +-- relevance_for(topic)   specialisation-adjusted
        +-- impact_summary()       UNMEASURED until ACTOR_MIN_OBSERVATIONS

Design decisions worth stating:

- **Base credibility is a PRIOR, labelled as one.** There is no market-impact
  history yet, so `credibility()` returns a prior and says so. Presenting a
  guessed number as a measurement is the exact failure M6 exists to prevent,
  and an actor's credibility feeding a score is a worse place to do it than
  most.
- **Insider vs external is a real distinction.** A CEO's guidance IS company
  guidance; an analyst's opinion is ABOUT the company, not from it. That
  decides whether a statement can be treated as the company speaking, so it
  is modelled rather than inferred from a name.
- **Impact is UNMEASURED below `ACTOR_MIN_OBSERVATIONS`.** A two-statement
  track record is noise, and an actor's apparent "historical impact" computed
  from three events would be the most confidently wrong number in the
  system.
- **Topic specialisation is bounded.** A CEO on their own product line is
  stronger evidence than the same CEO on macro policy, but the adjustment is
  capped so specialisation informs rather than dominates.
- **Statement frequency matters in both directions.** Someone who comments
  daily carries less information per statement than someone who speaks
  rarely; that is a real effect and is recorded, not scored, until there is
  evidence for how to weight it.

Pure and deterministic: no wall-clock, no network, no randomness.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from core.config import (
    ACTOR_BASE_CREDIBILITY,
    ACTOR_MIN_OBSERVATIONS,
    ACTOR_OFF_TOPIC_PENALTY,
    ACTOR_REGISTRY_VERSION,
    ACTOR_STANDINGS,
    ACTOR_STANDING_EXTERNAL,
    ACTOR_STANDING_INSIDER,
    ACTOR_STANDING_UNKNOWN,
    ACTOR_TOPIC_BONUS,
)


class ActorIntelligenceError(ValueError):
    """Raised when an actor record or observation is invalid."""


@dataclass(frozen=True)
class ActorRecord:
    """A curated influential person: who they are and what they speak for."""

    name: str
    role: str
    organization: str
    standing: str = ACTOR_STANDING_UNKNOWN
    # Tickers this actor can speak FOR (an insider) or authoritatively ABOUT.
    speaks_for: tuple[str, ...] = ()
    # Event types this actor is a subject-matter authority on.
    topics: tuple[str, ...] = ()

    def is_insider_for(self, ticker: str) -> bool:
        """Whether a statement from this actor IS the company speaking."""
        return (
            self.standing == ACTOR_STANDING_INSIDER
            and str(ticker).upper() in self.speaks_for
        )


@dataclass
class ActorObservation:
    """One recorded statement by an actor, and what followed.

    `abnormal_return` is deliberately optional: E3 records statements before
    E4 exists to measure their reactions, and a missing measurement must stay
    missing rather than defaulting to zero.
    """

    actor: str
    event_type: str
    ticker: str
    published_time: str
    novelty: float = 0.0
    abnormal_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActorProfile:
    """An actor plus everything observed about them so far."""

    record: ActorRecord
    observations: list[ActorObservation] = field(default_factory=list)
    registry_version: str = ACTOR_REGISTRY_VERSION

    # --- credibility ------------------------------------------------------

    def credibility(self, ticker: str | None = None) -> dict[str, Any]:
        """Source credibility, with its basis stated.

        Returns `basis: "prior"` until there are enough observations to
        measure anything. The basis is part of the answer because a prior and
        a measurement must never be confused by a consumer.
        """
        standing = self.record.standing
        if ticker is not None and self.record.is_insider_for(ticker):
            standing = ACTOR_STANDING_INSIDER
        elif ticker is not None and standing == ACTOR_STANDING_INSIDER:
            # An insider at one company is an OUTSIDER at another. A CEO
            # commenting on a rival is not that rival's insider.
            standing = ACTOR_STANDING_EXTERNAL

        value = ACTOR_BASE_CREDIBILITY[standing]
        return {
            "value": round(float(value), 4),
            "basis": "prior",
            "standing": standing,
            "observations": len(self.observations),
            "detail": (
                f"no measured track record yet ({len(self.observations)} of "
                f"{ACTOR_MIN_OBSERVATIONS} observations needed); this is a "
                f"prior from standing {standing!r}, not a measurement"
            ),
        }

    # --- topic specialisation ---------------------------------------------

    def relevance_for(self, event_type: str, base: float = 1.0) -> dict[str, Any]:
        """Adjust relevance by whether this is the actor's subject.

        A CEO on their own product line is stronger evidence than the same
        CEO on macro policy. Bounded so specialisation informs a judgement
        rather than dominating it.
        """
        if not self.record.topics:
            return {
                "value": round(float(base), 4),
                "adjustment": 0.0,
                "reason": "no declared topic specialisation; relevance unchanged",
            }
        on_topic = str(event_type) in self.record.topics
        adjustment = ACTOR_TOPIC_BONUS if on_topic else -ACTOR_OFF_TOPIC_PENALTY
        value = max(0.0, min(1.0, float(base) + adjustment))
        return {
            "value": round(value, 4),
            "adjustment": round(adjustment, 4),
            "reason": (
                f"{event_type!r} is within {self.record.name}'s declared topics"
                if on_topic
                else (
                    f"{event_type!r} is outside {self.record.name}'s declared "
                    f"topics {list(self.record.topics)} — speaking off-subject "
                    f"is weaker evidence"
                )
            ),
        }

    # --- statement frequency ----------------------------------------------

    def statement_frequency(self) -> dict[str, Any]:
        """How often this actor speaks, and about what.

        Recorded, not scored. Someone commenting daily plausibly carries less
        information per statement than someone who speaks rarely, but that is
        a hypothesis for E6 to test — weighting it now would bake in a guess.
        """
        by_type: dict[str, int] = {}
        by_ticker: dict[str, int] = {}
        for observation in self.observations:
            by_type[observation.event_type] = by_type.get(observation.event_type, 0) + 1
            by_ticker[observation.ticker] = by_ticker.get(observation.ticker, 0) + 1
        return {
            "total_statements": len(self.observations),
            "by_event_type": dict(sorted(by_type.items())),
            "by_ticker": dict(sorted(by_ticker.items())),
            "distinct_tickers": len(by_ticker),
            "note": (
                "frequency is recorded, not scored — how to weight a frequent "
                "commentator is a question for E6 to answer with evidence"
            ),
        }

    def mean_novelty(self) -> float | None:
        """Average novelty of this actor's statements, or None if unobserved."""
        values = [float(o.novelty) for o in self.observations]
        return round(statistics.fmean(values), 4) if values else None

    # --- historical market impact -----------------------------------------

    def impact_summary(self, ticker: str | None = None) -> dict[str, Any]:
        """Measured market impact, or an explicit refusal to claim one.

        This is where E3 is most likely to be misused, so the refusal is
        structural: below `ACTOR_MIN_OBSERVATIONS` the answer is
        `status: "unmeasured"` with no numbers at all, rather than a mean
        over three events that a consumer might treat as a finding.
        """
        relevant = [
            o for o in self.observations
            if o.abnormal_return is not None
            and (ticker is None or o.ticker == str(ticker).upper())
        ]
        scope = f"{self.record.name} x {ticker}" if ticker else self.record.name

        if len(relevant) < ACTOR_MIN_OBSERVATIONS:
            return {
                "status": "unmeasured",
                "scope": scope,
                "observations": len(relevant),
                "required": ACTOR_MIN_OBSERVATIONS,
                "detail": (
                    f"{len(relevant)} measured reactions is below the "
                    f"{ACTOR_MIN_OBSERVATIONS} required — a track record this "
                    f"short is noise, and reporting a mean would invite "
                    f"treating it as a finding"
                ),
            }

        returns = [float(o.abnormal_return) for o in relevant]
        if not returns:
            # Defensive: the threshold check above already guarantees this is
            # non-empty, but a statistics call on an empty list raises deep in
            # the stdlib and would mask the real problem from a caller.
            raise ActorIntelligenceError(
                f"no measured reactions for {scope} — impact cannot be computed"
            )
        return {
            "status": "measured",
            "scope": scope,
            "observations": len(relevant),
            "mean_abnormal_return": round(statistics.fmean(returns), 6),
            "median_abnormal_return": round(statistics.median(returns), 6),
            "stdev_abnormal_return": (
                round(statistics.pstdev(returns), 6) if len(returns) > 1 else 0.0
            ),
            "positive_share": round(sum(1 for r in returns if r > 0) / len(returns), 4),
            "detail": (
                "association only — an abnormal return following a statement is "
                "not evidence the statement caused it (master context section 37)"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.record.name,
            "role": self.record.role,
            "organization": self.record.organization,
            "standing": self.record.standing,
            "speaks_for": list(self.record.speaks_for),
            "topics": list(self.record.topics),
            "credibility": self.credibility(),
            "statement_frequency": self.statement_frequency(),
            "mean_novelty": self.mean_novelty(),
            "impact": self.impact_summary(),
            "registry_version": self.registry_version,
        }


def actor_problems(record: ActorRecord) -> list[str]:
    """Validate a curated actor record."""
    problems: list[str] = []
    if not str(record.name or "").strip():
        problems.append("name is required")
    if not str(record.role or "").strip():
        problems.append("role is required — an unattributed title is not an identity")
    if not str(record.organization or "").strip():
        problems.append("organization is required")
    if record.standing not in ACTOR_STANDINGS:
        problems.append(
            f"standing {record.standing!r} is not known (known: {sorted(ACTOR_STANDINGS)})"
        )
    if record.standing == ACTOR_STANDING_INSIDER and not record.speaks_for:
        problems.append(
            f"{record.name!r} is an insider but speaks_for no ticker — an insider "
            f"is an insider AT somewhere"
        )
    return problems


def build_actor_registry(
    entity_registry: dict | None = None,
) -> dict[str, ActorProfile]:
    """Build actor profiles from the E2 entity registry.

    Derived rather than duplicated: every executive already listed in the
    entity registry becomes an insider actor for that company. Maintaining a
    second hand-written list of the same people is how the two drift apart.
    """
    from core.entity_resolution import build_default_entity_registry

    active = entity_registry if entity_registry is not None else build_default_entity_registry()
    profiles: dict[str, ActorProfile] = {}
    for ticker, record in sorted(active.items()):
        for person in record.executives:
            existing = profiles.get(person)
            if existing is not None:
                # The same person can be an officer of more than one company.
                existing.record = ActorRecord(
                    name=existing.record.name,
                    role=existing.record.role,
                    organization=existing.record.organization,
                    standing=existing.record.standing,
                    speaks_for=tuple(sorted({*existing.record.speaks_for, ticker})),
                    topics=existing.record.topics,
                )
                continue
            profiles[person] = ActorProfile(
                record=ActorRecord(
                    name=person,
                    role="executive",
                    organization=record.legal_name,
                    standing=ACTOR_STANDING_INSIDER,
                    speaks_for=(ticker,),
                    # Company officers are authorities on company matters, not
                    # on macro. Declared narrowly on purpose.
                    topics=(
                        "earnings", "guidance", "product_launch",
                        "strategic_announcement", "management_commentary",
                    ),
                )
            )
    return profiles


def record_observation(
    profiles: dict[str, ActorProfile],
    observation: ActorObservation,
) -> ActorProfile:
    """Attach an observed statement to its actor."""
    profile = profiles.get(observation.actor)
    if profile is None:
        raise ActorIntelligenceError(
            f"actor {observation.actor!r} is not in the registry — an unknown "
            f"speaker cannot accrue a track record"
        )
    profile.observations.append(observation)
    return profile


def observations_from_events(events: Iterable[Any]) -> list[ActorObservation]:
    """Extract actor observations from canonical E1 events.

    Events with no named actor are skipped: an anonymous statement has no
    actor to attribute a track record to.
    """
    observations: list[ActorObservation] = []
    for event in events:
        actor = str(getattr(event, "actor", "") or "").strip()
        if not actor:
            continue
        observations.append(ActorObservation(
            actor=actor,
            event_type=str(getattr(event, "event_type", "other")),
            ticker=str(getattr(event, "entity", "")).upper(),
            published_time=str(getattr(event, "published_time", "")),
            novelty=float(getattr(event, "novelty", 0.0) or 0.0),
        ))
    return observations


def registry_report(profiles: dict[str, ActorProfile]) -> dict[str, Any]:
    """Coverage summary over the actor registry."""
    measured = sum(
        1 for profile in profiles.values()
        if profile.impact_summary()["status"] == "measured"
    )
    by_standing: dict[str, int] = {}
    for profile in profiles.values():
        standing = profile.record.standing
        by_standing[standing] = by_standing.get(standing, 0) + 1
    return {
        "registry_version": ACTOR_REGISTRY_VERSION,
        "actors": len(profiles),
        "by_standing": dict(sorted(by_standing.items())),
        "with_measured_impact": measured,
        "unmeasured": len(profiles) - measured,
        "min_observations": ACTOR_MIN_OBSERVATIONS,
        "note": (
            "impact stays unmeasured until E4 supplies abnormal returns; the "
            "actors exist so those measurements have somewhere to attach"
        ),
    }
