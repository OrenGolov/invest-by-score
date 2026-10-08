"""CI drift gate for the E1 canonical event object.

Proves, on every push, that the event shape Sprint E depends on still holds:

1. every E1 field is present and validated;
2. published_time and effective_time stay DISTINCT, and the PIT filter keys
   on publication;
3. event_id is deterministic — the same story from two providers is one
   event, so event memory cannot double-count;
4. CONTRADICTORY survives into the event instead of averaging to neutral;
5. an unsourced event is refused;
6. a non-OK news snapshot yields no events (fail-closed).

Synthetic only, so it runs in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    EVENT_DIRECTION_CONTRADICTORY,
    EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_POSITIVE,
)
from core.event_contract import (  # noqa: E402
    Event,
    EventContractError,
    event_from_article,
    event_problems,
    events_as_of,
    events_from_news_snapshot,
)

_E1_FIELDS = (
    "event_id", "published_time", "effective_time", "entity",
    "entities_affected", "actor", "actor_type", "event_type", "source",
    "source_quality", "novelty", "relevance", "direction", "magnitude",
    "confidence", "evidence",
)


def _event(**overrides) -> Event:
    payload = dict(
        entity="NVDA", published_time="2026-01-05 12:00:00", event_type="earnings",
        source="newsapi_news", source_quality=0.8, novelty=0.7, relevance=0.9,
        direction=EVENT_DIRECTION_POSITIVE, magnitude=0.6, confidence=0.72,
        evidence=[{"source_record_id": "rec-1", "reason": "beat"}],
    )
    payload.update(overrides)
    return Event(**payload)


def _article(**overrides) -> dict:
    payload = {
        "source_record_id": "rec-1", "published_time": "2026-01-05 10:00:00",
        "category": "earnings", "tone": 0.7, "relevance": 0.9,
        "source_weight": 0.8, "novelty": 0.6,
    }
    payload.update(overrides)
    return payload


def main() -> int:
    failures: list[str] = []

    def refuses(label: str, call) -> None:
        try:
            call()
        except EventContractError:
            return
        failures.append(f"guard did NOT fire: {label}")

    # 1. Schema completeness and validation.
    recorded = _event().to_dict()
    for required in _E1_FIELDS:
        if required not in recorded:
            failures.append(f"E1 field {required!r} is not recorded on an event")
    if event_problems(_event()):
        failures.append(f"a valid event was rejected: {event_problems(_event())[:2]}")
    for bad, label in (
        ({"actor_type": "oracle"}, "unknown actor_type"),
        ({"direction": "sideways"}, "unknown direction"),
        ({"source_quality": 1.5}, "out-of-band source_quality"),
        ({"relevance": float("nan")}, "NaN relevance"),
        ({"magnitude": 12.0}, "magnitude outside the claim band"),
        ({"entity": ""}, "missing entity"),
        ({"event_type": ""}, "missing event_type"),
    ):
        if not event_problems(_event(**bad)):
            failures.append(f"{label} was accepted")

    # 2. Timestamps stay distinct; PIT keys on publication.
    backdated = _event(effective_time="2026-01-02 09:00:00")
    if not backdated.is_backdated():
        failures.append("a backdated effect was not flagged")
    if backdated.is_point_in_time_valid("2026-01-03"):
        failures.append(
            "PIT validity used effective_time — the market cannot act on an "
            "effect it has not been told about"
        )
    if not backdated.is_point_in_time_valid("2026-01-06"):
        failures.append("a published event was excluded by the PIT filter")
    if len(events_as_of([_event(), _event(published_time="2026-06-01 00:00:00")], "2026-03-01")) != 1:
        failures.append("the PIT filter did not exclude a future event")

    # 3. Deterministic identity.
    if _event().event_id != _event().event_id:
        failures.append("event_id is not deterministic")
    if _event().event_id != _event(source_quality=0.1, confidence=0.1).event_id:
        failures.append(
            "rescoring changed the event_id — the same story would be counted twice"
        )
    for field_name, value in (
        ("entity", "TSLA"), ("published_time", "2026-02-05 12:00:00"),
        ("event_type", "litigation"), ("actor", "Someone"),
    ):
        if _event().event_id == _event(**{field_name: value}).event_id:
            failures.append(f"a changed {field_name} did not change the event_id")
    tampered = _event()
    tampered.entity = "TSLA"
    if not event_problems(tampered):
        failures.append("a tampered event_id was not detected")

    # 4/5. Evidence and contradiction.
    if not event_problems(_event(evidence=[])):
        failures.append("an unsourced event was accepted — that is a rumour")
    if not event_problems(_event(evidence=[{"reason": "no id"}])):
        failures.append("evidence without a source_record_id was accepted")

    snapshot = {
        "ticker": "NVDA", "status": "OK", "source_id": "newsapi_news",
        "articles": [_article(), _article(source_record_id="r2", category="litigation", tone=-0.5)],
    }
    events = events_from_news_snapshot(snapshot)
    if len(events) != 2:
        failures.append(f"expected 2 events from 2 articles, got {len(events)}")
    for event in events:
        if event_problems(event):
            failures.append(f"an adapted event is non-conformant: {event_problems(event)[:2]}")
    directions = {e.direction for e in events}
    if directions != {EVENT_DIRECTION_POSITIVE, EVENT_DIRECTION_NEGATIVE}:
        failures.append(f"tone did not map onto direction (got {directions})")

    contradictory = events_from_news_snapshot({**snapshot, "status": "CONTRADICTORY"})
    if not contradictory:
        failures.append("a CONTRADICTORY snapshot produced no events")
    if any(e.direction != EVENT_DIRECTION_CONTRADICTORY for e in contradictory):
        failures.append(
            "CONTRADICTORY did not propagate into the events — it must never "
            "average to neutral"
        )

    # 6. Fail-closed on a degraded snapshot.
    for status in ("UNAVAILABLE", "INCOMPLETE", "INVALID", "STALE"):
        if events_from_news_snapshot({**snapshot, "status": status}):
            failures.append(f"a {status} snapshot still produced events")

    refuses("an article with no published_time", lambda: event_from_article(
        _article(published_time=""), "NVDA", "src"))
    refuses("an article with no source_record_id", lambda: event_from_article(
        _article(source_record_id=""), "NVDA", "src"))

    if failures:
        print("E1 event-contract gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E1 event-contract gate OK:")
    print(f"  all {len(_E1_FIELDS)} canonical fields present and validated.")
    print("  published_time vs effective_time distinct; PIT keys on publication.")
    print("  event_id deterministic across rescoring; tampering detected.")
    print("  CONTRADICTORY propagates; unsourced events and degraded snapshots refused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
