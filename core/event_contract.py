"""Canonical event object (Sprint E1).

Sprint E exists to learn how events actually move the chart. That requires
ONE event shape every downstream stage agrees on: the event study (E4), the
confounder engine (E5) and event memory (E6) must all read the same object,
or they will quietly disagree about what an "event" even is — and a
disagreement like that surfaces as a mysterious modelling result rather than
an error.

    news article (N1 pipeline)
        |
        v
    event_from_article()   <- adapts, never re-classifies
        |
        v
    Event  --> event_problems()  --> event memory / study / attribution

The schema, exactly as E1 specifies: event_id, published_time,
effective_time, entity, entities_affected, actor, actor_type, event_type,
source, source_quality, novelty, relevance, direction, magnitude,
confidence, evidence.

Design decisions worth stating:

- **This adapts N1, it does not replace it.** Classification, tone and
  source quality already exist and are already tested; re-deriving them here
  would create a second, divergent opinion about the same article. The
  adapter maps an N1 article onto the canonical shape and nothing more.
- **`published_time` and `effective_time` are distinct.** When a company
  announces on Monday that a plant closed on Friday, the PIT rule keys on
  publication — that is when the market could know — while the effect dates
  from Friday. Collapsing them is how a backtest silently uses future
  information, so both are recorded and `effective_time` defaults to
  `published_time` only when genuinely unknown.
- **`event_id` is deterministic**, derived from the content that defines the
  event. The same article ingested twice is the same event, so event memory
  cannot double-count a story that two providers carried.
- **CONTRADICTORY is a direction**, not an averaging failure. Credible
  evidence pointing both ways keeps that label all the way into event
  memory (master context section 15).
- **`magnitude` is a bounded, unitless claim size — NOT an expected return.**
  Calling it a return would invite treating an unvalidated number as a
  forecast before the F-sprint exists to make real ones.
- **No evidence, no event.** An event citing no source record is a rumour,
  and `event_problems` refuses it.

Pure and deterministic: no wall-clock, no network, no randomness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import (
    ACTOR_TYPES,
    ACTOR_TYPE_COMPANY,
    ACTOR_TYPE_JOURNALIST,
    ACTOR_TYPE_UNKNOWN,
    EVENT_DIRECTIONS,
    EVENT_DIRECTION_CONTRADICTORY,
    EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_NEUTRAL,
    EVENT_DIRECTION_POSITIVE,
    EVENT_MAGNITUDE_MAX,
    EVENT_MAGNITUDE_MIN,
    EVENT_MIN_EVIDENCE,
    EVENT_SCHEMA_VERSION,
)


class EventContractError(ValueError):
    """Raised when an event violates the canonical schema."""


@dataclass
class Event:
    """One canonical event. Immutable once built."""

    entity: str
    published_time: str
    event_type: str
    # THE OUTLET that reported it ("CNBC", "Biztoc.com"), not the provider
    # that delivered it. MEASURED 2026-10-04, this field held the constant
    # "newsapi_news" for every event ever built, because news_adapter.py
    # passed its provider id at seven call sites. The per-article
    # `source_name` was captured in the raw ledger and then dropped, so L4's
    # source-reliability estimator was handed ONE source 1,637 times and
    # correctly reported REGISTRY_PRIOR - it had no variation to learn from.
    # 369 distinct outlets appeared in a single day's articles.
    source: str
    evidence: list[dict[str, str]] = field(default_factory=list)

    # Effective time defaults to publication ONLY when the real effective
    # time is unknown; see the module docstring on why they are distinct.
    effective_time: str | None = None
    entities_affected: list[str] = field(default_factory=list)
    actor: str = ""
    actor_type: str = ACTOR_TYPE_UNKNOWN
    # WHICH PIPELINE DELIVERED IT ("newsapi_news", later "finnhub_news").
    # Separate from `source` because they answer different questions: the
    # provider is a property of our plumbing, the outlet is a property of
    # the world. Only the second one can have a track record worth learning.
    provider: str = ""
    source_quality: float = 0.0
    novelty: float = 0.0
    relevance: float = 0.0
    direction: str = EVENT_DIRECTION_NEUTRAL
    magnitude: float = 0.0
    confidence: float = 0.0
    event_id: str = ""
    schema_version: str = EVENT_SCHEMA_VERSION
    # E2: how this event's entity was resolved. Recorded so a consumer can
    # see WHY the event was attributed to this entity — and so an event that
    # failed resolution is visibly different from one that resolved cleanly,
    # rather than both arriving as ordinary events.
    entity_resolution: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effective_time is None:
            self.effective_time = self.published_time
        if self.entity and self.entity not in self.entities_affected:
            # The primary entity is always affected by its own event.
            self.entities_affected = [self.entity, *self.entities_affected]
        if not self.event_id:
            self.event_id = self.canonical_id()

    def canonical_id(self) -> str:
        """Deterministic identity over the content that defines the event.

        Excludes the scores (novelty, relevance, confidence, magnitude):
        those are judgements ABOUT the event that may be recomputed as the
        pipeline improves. The event itself is the same event.

        **EXCLUDES `source` AND `provider` — WHO DELIVERED A STORY IS NOT
        WHAT HAPPENED.** This module's contract is that "event memory cannot
        double-count a story that two providers carried", and that is now
        literally true: the SAME article reaching us through NewsAPI and
        through Finnhub has one identity, where hashing the provider would
        have made two.

        Excluding `source` is also what lets it become the outlet at all.
        While it held the constant "newsapi_news" the choice was invisible;
        the moment it holds 369 distinct outlet names, hashing it would
        fragment identity by reporter.

        **WHAT THIS DOES NOT DO, measured rather than assumed.** It does NOT
        merge two outlets covering one story:

            CNBC    /x -> cb5a2514...
            Reuters /y -> 9c46abc4...

        because `evidence` carries each article's `source_record_id` (its
        URL) and those differ. Same-story-across-outlets is a SEMANTIC
        clustering problem — same entity, same window, same event type — and
        it belongs to E7's chains, which already exist to count a story
        once. Identity here stays syntactic and deterministic; pretending
        otherwise would be a dedup claim this function cannot honour.
        """
        payload = {
            "entity": str(self.entity).upper(),
            "published_time": str(self.published_time),
            "effective_time": str(self.effective_time),
            "event_type": str(self.event_type),
            "actor": str(self.actor),
            # Tolerate malformed entries here so identity can always be
            # computed; event_problems() is what reports them. Raising from
            # __post_init__ instead would crash before validation could
            # explain what was wrong.
            "evidence": sorted(
                str(entry.get("source_record_id", "")) if isinstance(entry, dict) else str(entry)
                for entry in self.evidence or []
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_point_in_time_valid(self, as_of: str) -> bool:
        """Whether this event was knowable at `as_of`.

        Keys on PUBLICATION, never on effective time: the market cannot act
        on an effect it has not been told about.
        """
        return str(self.published_time) <= str(as_of)

    def is_entity_resolved(self) -> bool:
        """Whether the entity attribution is strong enough for training.

        An event with no recorded resolution is treated as UNRESOLVED rather
        than assumed good: E2's rule is that bad resolution must not enter
        training silently, and an absent resolution is the most silent case
        of all.
        """
        return bool((self.entity_resolution or {}).get("training_eligible"))

    def is_backdated(self) -> bool:
        """Whether the effect predates its publication.

        Common and legitimate — a Monday announcement of a Friday event —
        but it must be visible, because an event study anchored on the wrong
        timestamp measures the wrong window.
        """
        return str(self.effective_time) < str(self.published_time)


def _in_unit_band(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and 0.0 <= number <= 1.0


def event_problems(event: Event) -> list[str]:
    """Validate one event against the canonical schema.

    Returns diff-precise problem strings (empty = conforming) so callers can
    assert or gate, matching the convention used by the feature and model
    registries.
    """
    problems: list[str] = []

    if not str(event.entity or "").strip():
        problems.append("entity is required — an event must be about something")
    if not str(event.published_time or "").strip():
        problems.append("published_time is required — it is the PIT key")
    if not str(event.event_type or "").strip():
        problems.append("event_type is required")
    if not str(event.source or "").strip():
        problems.append("source is required")

    if event.actor_type not in ACTOR_TYPES:
        problems.append(
            f"actor_type {event.actor_type!r} is not a known type "
            f"(known: {sorted(ACTOR_TYPES)})"
        )
    if event.direction not in EVENT_DIRECTIONS:
        problems.append(
            f"direction {event.direction!r} is not a known direction "
            f"(known: {sorted(EVENT_DIRECTIONS)})"
        )

    for name in ("source_quality", "novelty", "relevance", "confidence"):
        if not _in_unit_band(getattr(event, name)):
            problems.append(f"{name} must be a number in [0, 1], got {getattr(event, name)!r}")

    try:
        magnitude = float(event.magnitude)
    except (TypeError, ValueError):
        problems.append(f"magnitude must be numeric, got {event.magnitude!r}")
    else:
        if magnitude != magnitude or not (
            EVENT_MAGNITUDE_MIN <= magnitude <= EVENT_MAGNITUDE_MAX
        ):
            problems.append(
                f"magnitude {magnitude} is outside "
                f"[{EVENT_MAGNITUDE_MIN}, {EVENT_MAGNITUDE_MAX}] — it is a bounded "
                f"claim size, not an expected return"
            )

    evidence = event.evidence or []
    if len(evidence) < EVENT_MIN_EVIDENCE:
        problems.append(
            f"an event must cite at least {EVENT_MIN_EVIDENCE} source record — "
            f"an unsourced event is a rumour"
        )
    for index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            problems.append(f"evidence[{index}] is not a dict")
            continue
        if not str(entry.get("source_record_id", "")).strip():
            problems.append(f"evidence[{index}] has no source_record_id")

    if event.entity and event.entity not in (event.entities_affected or []):
        problems.append(
            "entities_affected must include the primary entity — an event about "
            "a company affects that company"
        )

    if event.published_time and event.effective_time:
        # A future-dated EFFECT is legitimate (a scheduled closure); a
        # future-dated PUBLICATION is not, and the PIT filter catches that
        # at query time rather than here.
        pass

    if event.event_id and event.event_id != event.canonical_id():
        problems.append(
            "event_id does not match the event's content — a changed event is a "
            "NEW event, not an edit"
        )
    return problems


def require_valid_event(event: Event) -> Event:
    """Fail-closed helper: return the event or raise."""
    problems = event_problems(event)
    if problems:
        raise EventContractError(
            f"event {event.event_id!r} violates the canonical schema: "
            + "; ".join(problems)
        )
    return event


def _direction_from_tone(tone: Any) -> str:
    """Map N1's signed tone onto the canonical direction vocabulary."""
    try:
        value = float(tone)
    except (TypeError, ValueError):
        return EVENT_DIRECTION_NEUTRAL
    if value != value:
        return EVENT_DIRECTION_NEUTRAL
    if value > 0.0:
        return EVENT_DIRECTION_POSITIVE
    if value < 0.0:
        return EVENT_DIRECTION_NEGATIVE
    return EVENT_DIRECTION_NEUTRAL


def _actor_type_for(event_type: str, actor: str) -> str:
    """Infer the actor type conservatively.

    Only company-issued event types are claimed with confidence; everything
    else stays `unknown` rather than guessing. E2/E3 will resolve actors
    properly — inventing an actor_type now would put unearned confidence into
    the training data those sprints consume.
    """
    if not str(actor or "").strip():
        return ACTOR_TYPE_JOURNALIST if event_type else ACTOR_TYPE_UNKNOWN
    if event_type in ("earnings", "guidance", "product_launch", "strategic_announcement"):
        return ACTOR_TYPE_COMPANY
    return ACTOR_TYPE_UNKNOWN


def event_from_article(
    article: dict,
    entity: str,
    source: str,
    effective_time: str | None = None,
) -> Event:
    """Adapt one N1 pipeline article onto the canonical event shape.

    Adapts only — it never re-classifies. The N1 pipeline already decided
    the category, tone, relevance and source weight, and re-deriving them
    here would create a second, divergent opinion about the same article.

    **THE OUTLET COMES FROM THE ARTICLE; `source` IS THE PROVIDER.** The
    caller passes its pipeline id ("newsapi_news") and that lands in
    `Event.provider`. `Event.source` is taken from the article's own
    `source_name` — the outlet that actually reported it.

    MEASURED 2026-10-04, before this change: `source_name` was captured on
    every one of 1,637 articles and then discarded here, so every event in
    the system recorded the string "newsapi_news" as its source. L4's
    source-reliability estimator was therefore handed one source with no
    variation and correctly reported REGISTRY_PRIOR for everything. 369
    distinct outlets were present in that same day's data.

    `source` remains the fallback when an article attests no outlet, so a
    provider that omits the field degrades to today's behaviour rather than
    producing an event with no source at all (which `event_problems` would
    refuse).
    """
    if not isinstance(article, dict):
        raise EventContractError(f"article must be a dict, got {type(article).__name__}")

    published = article.get("published_time")
    if not str(published or "").strip():
        raise EventContractError(
            "article has no published_time — without it the event cannot be "
            "point-in-time filtered"
        )

    record_id = str(article.get("source_record_id", "")).strip()
    if not record_id:
        raise EventContractError("article has no source_record_id — an event needs evidence")

    event_type = str(article.get("category") or "other")
    actor = str(article.get("actor") or "")
    tone = article.get("tone")

    # The outlet, falling back to the provider when the article attests none.
    outlet = str(article.get("source_name") or "").strip() or str(source)

    return Event(
        entity=str(entity).upper(),
        published_time=str(published),
        effective_time=effective_time,
        event_type=event_type,
        source=outlet,
        provider=str(source),
        actor=actor,
        actor_type=_actor_type_for(event_type, actor),
        source_quality=float(article.get("source_weight") or 0.0),
        novelty=float(article.get("novelty") or 0.0),
        relevance=float(article.get("relevance") or 0.0),
        direction=_direction_from_tone(tone),
        # Magnitude is the ABSOLUTE tone: how large a claim, independent of
        # which way it points. Direction already carries the sign.
        magnitude=min(EVENT_MAGNITUDE_MAX, abs(float(tone or 0.0))),
        confidence=float(article.get("source_weight") or 0.0) * float(article.get("relevance") or 0.0),
        evidence=[{
            "source_record_id": record_id,
            "reason": (
                f"{event_type}; tone {tone}; relevance {article.get('relevance')}; "
                f"source weight {article.get('source_weight')}"
            ),
        }],
    )


def _resolution_payload(article: dict, entity: str) -> dict[str, Any]:
    """Resolve an article's entity and record the verdict (E2)."""
    from core.entity_resolution import resolve_article

    resolution = resolve_article(article, entity)
    payload = resolution.to_dict()
    payload["training_eligible"] = resolution.is_training_eligible()
    return payload


def events_from_news_snapshot(
    snapshot: dict,
    effective_times: dict[str, str] | None = None,
    require_resolved_entity: bool = False,
) -> list[Event]:
    """Adapt every article in an N1 news snapshot into canonical events.

    A non-OK snapshot yields NO events rather than degraded ones: the
    fail-closed rule that governs the news agent governs its events too. A
    CONTRADICTORY snapshot is the one exception worth naming — its articles
    still become events, because the contradiction is information the event
    study needs, not a reason to discard the evidence.
    """
    status = str((snapshot or {}).get("status", "UNAVAILABLE"))
    if status not in ("OK", "CONTRADICTORY"):
        return []

    entity = str(snapshot.get("ticker", "")).upper()
    source = str(snapshot.get("source_id", "unknown"))
    effective_times = effective_times or {}

    events: list[Event] = []
    for article in snapshot.get("articles") or []:
        record_id = str(article.get("source_record_id", "")).strip()
        if not record_id or not str(article.get("published_time") or "").strip():
            # Incomplete articles are skipped, never patched into events —
            # a fabricated timestamp would defeat the PIT filter.
            continue
        event = event_from_article(
            article, entity, source, effective_times.get(record_id)
        )
        event.entity_resolution = _resolution_payload(article, entity)
        if require_resolved_entity and not event.is_entity_resolved():
            # Fail-closed mode for training data: an event whose entity could
            # not be resolved is dropped rather than labelled. The resolution
            # verdict stays available via resolve_article for coverage
            # reporting, so the gap is measurable, not merely absent.
            continue
        if status == "CONTRADICTORY":
            event.direction = EVENT_DIRECTION_CONTRADICTORY
            event.event_id = event.canonical_id()
        events.append(event)
    return events


def events_as_of(events: list[Event], as_of: str) -> list[Event]:
    """PIT filter: only events published at or before `as_of`.

    Keys on publication, never on effective time — the market cannot act on
    an effect it has not been told about.
    """
    return [event for event in events if event.is_point_in_time_valid(as_of)]
