"""Canonical event object tests (Sprint E1).

E1 schema, pinned: event_id, published_time, effective_time, entity,
entities_affected, actor, actor_type, event_type, source, source_quality,
novelty, relevance, direction, magnitude, confidence, evidence.

Beyond field presence, three properties decide whether Sprint E can trust
its own inputs:

- published_time and effective_time are DISTINCT, and the PIT filter keys on
  publication — the market cannot act on an effect it has not been told of;
- event_id is deterministic, so the same story from two providers is one
  event rather than two;
- CONTRADICTORY survives into the event, instead of averaging to neutral.
"""

from __future__ import annotations

import unittest

from core.config import (
    ACTOR_TYPE_COMPANY,
    ACTOR_TYPE_UNKNOWN,
    EVENT_DIRECTION_CONTRADICTORY,
    EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_NEUTRAL,
    EVENT_DIRECTION_POSITIVE,
    EVENT_SCHEMA_VERSION,
)
from core.event_contract import (
    Event,
    EventContractError,
    event_from_article,
    event_problems,
    events_as_of,
    events_from_news_snapshot,
    require_valid_event,
)


def _event(**overrides) -> Event:
    payload = dict(
        entity="NVDA",
        published_time="2026-01-05 12:00:00",
        event_type="earnings",
        source="newsapi_news",
        source_quality=0.8,
        novelty=0.7,
        relevance=0.9,
        direction=EVENT_DIRECTION_POSITIVE,
        magnitude=0.6,
        confidence=0.72,
        evidence=[{"source_record_id": "rec-1", "reason": "earnings beat"}],
    )
    payload.update(overrides)
    return Event(**payload)


def _article(**overrides) -> dict:
    payload = {
        "source_record_id": "rec-1",
        "published_time": "2026-01-05 10:00:00",
        "category": "earnings",
        "tone": 0.7,
        "relevance": 0.9,
        "source_weight": 0.8,
        "novelty": 0.6,
    }
    payload.update(overrides)
    return payload


class TestSchema(unittest.TestCase):
    def test_every_e1_field_is_present(self) -> None:
        recorded = _event().to_dict()
        for required in (
            "event_id", "published_time", "effective_time", "entity",
            "entities_affected", "actor", "actor_type", "event_type", "source",
            "source_quality", "novelty", "relevance", "direction", "magnitude",
            "confidence", "evidence",
        ):
            with self.subTest(field=required):
                self.assertIn(required, recorded)

    def test_a_valid_event_has_no_problems(self) -> None:
        self.assertEqual(event_problems(_event()), [])

    def test_schema_version_is_stamped(self) -> None:
        self.assertEqual(_event().schema_version, EVENT_SCHEMA_VERSION)

    def test_required_fields_are_enforced(self) -> None:
        for field_name in ("entity", "published_time", "event_type", "source"):
            with self.subTest(missing=field_name):
                problems = event_problems(_event(**{field_name: ""}))
                self.assertTrue(any(field_name in p for p in problems))

    def test_unknown_actor_type_is_refused(self) -> None:
        problems = event_problems(_event(actor_type="oracle"))
        self.assertTrue(any("actor_type" in p for p in problems))

    def test_unknown_direction_is_refused(self) -> None:
        problems = event_problems(_event(direction="sideways"))
        self.assertTrue(any("direction" in p for p in problems))

    def test_unit_scores_must_be_in_band(self) -> None:
        for field_name in ("source_quality", "novelty", "relevance", "confidence"):
            for bad in (-0.1, 1.5, float("nan"), "high"):
                with self.subTest(field=field_name, value=bad):
                    self.assertTrue(event_problems(_event(**{field_name: bad})))

    def test_magnitude_is_a_bounded_claim_not_a_return(self) -> None:
        """A 12% expected return is not a valid magnitude."""
        problems = event_problems(_event(magnitude=0.12 * 100))
        self.assertTrue(any("not an expected return" in p for p in problems))

    def test_require_valid_event_raises(self) -> None:
        with self.assertRaises(EventContractError):
            require_valid_event(_event(direction="sideways"))


class TestEvidence(unittest.TestCase):
    def test_an_unsourced_event_is_refused(self) -> None:
        """An event citing nothing is a rumour."""
        problems = event_problems(_event(evidence=[]))
        self.assertTrue(any("rumour" in p for p in problems))

    def test_evidence_without_a_record_id_is_refused(self) -> None:
        problems = event_problems(_event(evidence=[{"reason": "someone said so"}]))
        self.assertTrue(any("source_record_id" in p for p in problems))

    def test_non_dict_evidence_is_refused(self) -> None:
        self.assertTrue(event_problems(_event(evidence=["rec-1"])))


class TestTimestamps(unittest.TestCase):
    def test_effective_time_defaults_to_publication(self) -> None:
        self.assertEqual(_event().effective_time, "2026-01-05 12:00:00")

    def test_effective_time_can_precede_publication(self) -> None:
        """Monday's announcement of Friday's plant closure."""
        event = _event(effective_time="2026-01-02 09:00:00")
        self.assertEqual(event_problems(event), [])
        self.assertTrue(event.is_backdated())

    def test_a_same_day_event_is_not_backdated(self) -> None:
        self.assertFalse(_event().is_backdated())

    def test_pit_validity_keys_on_publication(self) -> None:
        """The market cannot act on an effect it has not been told about."""
        event = _event(effective_time="2026-01-02 09:00:00")
        self.assertFalse(event.is_point_in_time_valid("2026-01-03"))
        self.assertTrue(event.is_point_in_time_valid("2026-01-06"))

    def test_pit_filter_excludes_future_events(self) -> None:
        events = [
            _event(published_time="2026-01-01 00:00:00"),
            _event(published_time="2026-06-01 00:00:00"),
        ]
        self.assertEqual(len(events_as_of(events, "2026-03-01")), 1)

    def test_an_event_published_exactly_at_as_of_is_included(self) -> None:
        self.assertEqual(len(events_as_of([_event()], "2026-01-05 12:00:00")), 1)


class TestDeterministicIdentity(unittest.TestCase):
    def test_the_same_event_hashes_identically(self) -> None:
        self.assertEqual(_event().event_id, _event().event_id)

    def test_the_same_story_from_two_providers_is_one_event(self) -> None:
        """Event memory must not double-count a syndicated story."""
        first = _event()
        second = _event(source_quality=0.5, relevance=0.4, confidence=0.2)
        self.assertEqual(first.event_id, second.event_id)

    def test_defining_content_changes_the_id(self) -> None:
        for field_name, value in (
            ("entity", "TSLA"),
            ("published_time", "2026-02-05 12:00:00"),
            ("effective_time", "2026-01-01 00:00:00"),
            ("event_type", "litigation"),
            ("actor", "Jensen Huang"),
            ("source", "other_provider"),
        ):
            with self.subTest(changed=field_name):
                self.assertNotEqual(_event().event_id, _event(**{field_name: value}).event_id)

    def test_different_evidence_changes_the_id(self) -> None:
        other = _event(evidence=[{"source_record_id": "rec-2"}])
        self.assertNotEqual(_event().event_id, other.event_id)

    def test_a_tampered_id_is_detected(self) -> None:
        event = _event()
        event.entity = "TSLA"
        self.assertTrue(any("NEW event" in p for p in event_problems(event)))


class TestEntities(unittest.TestCase):
    def test_the_primary_entity_is_always_affected(self) -> None:
        self.assertIn("NVDA", _event().entities_affected)

    def test_additional_entities_are_preserved(self) -> None:
        event = _event(entities_affected=["AMD", "INTC"])
        self.assertEqual(event.entities_affected, ["NVDA", "AMD", "INTC"])

    def test_the_primary_entity_is_not_duplicated(self) -> None:
        event = _event(entities_affected=["NVDA"])
        self.assertEqual(event.entities_affected.count("NVDA"), 1)


class TestNewsAdapter(unittest.TestCase):
    """Adapts N1; never re-classifies it."""

    def test_an_article_becomes_a_conforming_event(self) -> None:
        event = event_from_article(_article(), "NVDA", "newsapi_news")
        self.assertEqual(event_problems(event), [])
        self.assertEqual(event.event_type, "earnings")

    def test_tone_maps_onto_direction(self) -> None:
        for tone, expected in (
            (0.7, EVENT_DIRECTION_POSITIVE),
            (-0.5, EVENT_DIRECTION_NEGATIVE),
            (0.0, EVENT_DIRECTION_NEUTRAL),
            (None, EVENT_DIRECTION_NEUTRAL),
        ):
            with self.subTest(tone=tone):
                event = event_from_article(_article(tone=tone), "NVDA", "src")
                self.assertEqual(event.direction, expected)

    def test_magnitude_is_the_absolute_tone(self) -> None:
        """Direction carries the sign; magnitude carries the size."""
        event = event_from_article(_article(tone=-0.5), "NVDA", "src")
        self.assertEqual(event.magnitude, 0.5)
        self.assertEqual(event.direction, EVENT_DIRECTION_NEGATIVE)

    def test_company_event_types_infer_a_company_actor(self) -> None:
        event = event_from_article(
            _article(category="earnings", actor="NVIDIA Corp"), "NVDA", "src"
        )
        self.assertEqual(event.actor_type, ACTOR_TYPE_COMPANY)

    def test_an_unclear_actor_stays_unknown(self) -> None:
        """E2/E3 resolve actors; guessing now would fabricate confidence."""
        event = event_from_article(
            _article(category="macro_shock", actor="Some Person"), "NVDA", "src"
        )
        self.assertEqual(event.actor_type, ACTOR_TYPE_UNKNOWN)

    def test_an_article_without_a_timestamp_is_refused(self) -> None:
        with self.assertRaises(EventContractError) as ctx:
            event_from_article(_article(published_time=""), "NVDA", "src")
        self.assertIn("point-in-time", str(ctx.exception))

    def test_an_article_without_a_record_id_is_refused(self) -> None:
        with self.assertRaises(EventContractError):
            event_from_article(_article(source_record_id=""), "NVDA", "src")

    def test_a_non_dict_article_is_refused(self) -> None:
        with self.assertRaises(EventContractError):
            event_from_article("an article", "NVDA", "src")


class TestSnapshotAdapter(unittest.TestCase):
    def _snapshot(self, **overrides) -> dict:
        payload = {
            "ticker": "NVDA",
            "status": "OK",
            "source_id": "newsapi_news",
            "articles": [
                _article(source_record_id="r1"),
                _article(source_record_id="r2", category="litigation", tone=-0.5),
            ],
        }
        payload.update(overrides)
        return payload

    def test_every_article_becomes_an_event(self) -> None:
        events = events_from_news_snapshot(self._snapshot())
        self.assertEqual(len(events), 2)
        for event in events:
            with self.subTest(event=event.event_type):
                self.assertEqual(event_problems(event), [])

    def test_a_non_ok_snapshot_yields_no_events(self) -> None:
        """Fail-closed: the rule governing the news agent governs its events."""
        for status in ("UNAVAILABLE", "INCOMPLETE", "INVALID", "STALE"):
            with self.subTest(status=status):
                self.assertEqual(events_from_news_snapshot(self._snapshot(status=status)), [])

    def test_contradictory_propagates_into_the_events(self) -> None:
        """It must never average to neutral."""
        events = events_from_news_snapshot(self._snapshot(status="CONTRADICTORY"))
        self.assertEqual(len(events), 2)
        for event in events:
            with self.subTest(event=event.event_id):
                self.assertEqual(event.direction, EVENT_DIRECTION_CONTRADICTORY)
                self.assertEqual(event_problems(event), [])

    def test_incomplete_articles_are_skipped_not_patched(self) -> None:
        """A fabricated timestamp would defeat the PIT filter."""
        snapshot = self._snapshot(articles=[
            _article(source_record_id="r1"),
            _article(source_record_id="", published_time="2026-01-05 10:00:00"),
            _article(source_record_id="r3", published_time=""),
        ])
        self.assertEqual(len(events_from_news_snapshot(snapshot)), 1)

    def test_an_empty_snapshot_yields_nothing(self) -> None:
        self.assertEqual(events_from_news_snapshot({}), [])

    def test_effective_times_can_be_supplied(self) -> None:
        events = events_from_news_snapshot(
            self._snapshot(), effective_times={"r1": "2026-01-01 00:00:00"}
        )
        backdated = [e for e in events if e.is_backdated()]
        self.assertEqual(len(backdated), 1)


if __name__ == "__main__":
    unittest.main()
