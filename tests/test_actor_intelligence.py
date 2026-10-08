"""Influential person intelligence tests (Sprint E3).

E3 tracks identity, role, organization, historical relevance, topic
specialization, source credibility, statement frequency, novelty and
historical market impact.

The property that matters most here is the one about NOT knowing things.
E3 exists before E4 measures any market reaction, so an actor's "historical
impact" is a number the system is strongly tempted to invent. These tests pin
that it refuses: below the observation threshold the answer is `unmeasured`
with no numbers at all, and `credibility()` labels its output a prior rather
than a measurement.
"""

from __future__ import annotations

import unittest

from core.config import (
    ACTOR_BASE_CREDIBILITY,
    ACTOR_MIN_OBSERVATIONS,
    ACTOR_REGISTRY_VERSION,
    ACTOR_STANDING_EXTERNAL,
    ACTOR_STANDING_INSIDER,
    ACTOR_STANDING_UNKNOWN,
)
from core.actor_intelligence import (
    ActorIntelligenceError,
    ActorObservation,
    ActorProfile,
    ActorRecord,
    actor_problems,
    build_actor_registry,
    observations_from_events,
    record_observation,
    registry_report,
)
from core.event_contract import Event


def _profile(**overrides) -> ActorProfile:
    payload = dict(
        name="Jensen Huang",
        role="executive",
        organization="NVIDIA Corporation",
        standing=ACTOR_STANDING_INSIDER,
        speaks_for=("NVDA",),
        topics=("earnings", "guidance", "product_launch"),
    )
    payload.update(overrides)
    return ActorProfile(record=ActorRecord(**payload))


def _observations(count: int, abnormal_return: float | None = 0.02) -> list[ActorObservation]:
    return [
        ActorObservation(
            actor="Jensen Huang", event_type="earnings", ticker="NVDA",
            published_time=f"2026-01-{(index % 28) + 1:02d} 00:00:00",
            novelty=0.5, abnormal_return=abnormal_return,
        )
        for index in range(count)
    ]


class TestActorRecords(unittest.TestCase):
    def test_a_valid_record_has_no_problems(self) -> None:
        self.assertEqual(actor_problems(_profile().record), [])

    def test_identity_fields_are_required(self) -> None:
        for field_name in ("name", "role", "organization"):
            with self.subTest(missing=field_name):
                problems = actor_problems(_profile(**{field_name: ""}).record)
                self.assertTrue(any(field_name in p for p in problems))

    def test_unknown_standing_is_refused(self) -> None:
        problems = actor_problems(_profile(standing="oracle").record)
        self.assertTrue(any("standing" in p for p in problems))

    def test_an_insider_must_name_a_company(self) -> None:
        """An insider is an insider AT somewhere."""
        problems = actor_problems(_profile(speaks_for=()).record)
        self.assertTrue(any("insider is an insider AT" in p for p in problems))

    def test_insider_status_is_company_specific(self) -> None:
        record = _profile().record
        self.assertTrue(record.is_insider_for("NVDA"))
        self.assertFalse(record.is_insider_for("AMD"))


class TestCredibilityIsAPrior(unittest.TestCase):
    """The system must not present a guess as a measurement."""

    def test_credibility_declares_its_basis(self) -> None:
        self.assertEqual(_profile().credibility("NVDA")["basis"], "prior")

    def test_the_detail_says_it_is_not_a_measurement(self) -> None:
        detail = _profile().credibility("NVDA")["detail"]
        self.assertIn("not a measurement", detail)

    def test_an_insider_starts_more_credible_about_their_own_company(self) -> None:
        profile = _profile()
        self.assertGreater(
            profile.credibility("NVDA")["value"], profile.credibility("AMD")["value"]
        )

    def test_an_insider_elsewhere_is_external_here(self) -> None:
        """A CEO commenting on a rival is not that rival's insider."""
        result = _profile().credibility("AMD")
        self.assertEqual(result["standing"], ACTOR_STANDING_EXTERNAL)
        self.assertEqual(result["value"], ACTOR_BASE_CREDIBILITY[ACTOR_STANDING_EXTERNAL])

    def test_an_unknown_actor_is_least_credible(self) -> None:
        profile = _profile(standing=ACTOR_STANDING_UNKNOWN, speaks_for=())
        self.assertEqual(
            profile.credibility()["value"], ACTOR_BASE_CREDIBILITY[ACTOR_STANDING_UNKNOWN]
        )


class TestTopicSpecialisation(unittest.TestCase):
    def test_on_topic_raises_relevance(self) -> None:
        result = _profile().relevance_for("earnings", base=0.8)
        self.assertGreater(result["adjustment"], 0)
        self.assertIn("within", result["reason"])

    def test_off_topic_lowers_relevance(self) -> None:
        """A CEO on macro policy is weaker evidence than on their own results."""
        result = _profile().relevance_for("macro_shock", base=0.8)
        self.assertLess(result["adjustment"], 0)
        self.assertIn("off-subject", result["reason"])

    def test_the_adjustment_is_bounded_to_the_unit_band(self) -> None:
        self.assertLessEqual(_profile().relevance_for("earnings", base=1.0)["value"], 1.0)
        self.assertGreaterEqual(_profile().relevance_for("macro_shock", base=0.0)["value"], 0.0)

    def test_no_declared_topics_leaves_relevance_unchanged(self) -> None:
        result = _profile(topics=()).relevance_for("anything", base=0.77)
        self.assertEqual(result["value"], 0.77)
        self.assertEqual(result["adjustment"], 0.0)


class TestImpactRefusal(unittest.TestCase):
    """The number E3 is most tempted to invent."""

    def test_impact_is_unmeasured_with_no_observations(self) -> None:
        summary = _profile().impact_summary()
        self.assertEqual(summary["status"], "unmeasured")

    def test_unmeasured_impact_reports_no_numbers_at_all(self) -> None:
        """Not a mean over three events a consumer might treat as a finding."""
        summary = _profile().impact_summary()
        for key in ("mean_abnormal_return", "median_abnormal_return", "positive_share"):
            with self.subTest(key=key):
                self.assertNotIn(key, summary)

    def test_one_below_the_threshold_is_still_unmeasured(self) -> None:
        profile = _profile()
        profile.observations = _observations(ACTOR_MIN_OBSERVATIONS - 1)
        self.assertEqual(profile.impact_summary()["status"], "unmeasured")

    def test_at_the_threshold_impact_is_measured(self) -> None:
        profile = _profile()
        profile.observations = _observations(ACTOR_MIN_OBSERVATIONS)
        summary = profile.impact_summary()
        self.assertEqual(summary["status"], "measured")
        self.assertEqual(summary["mean_abnormal_return"], 0.02)

    def test_observations_without_a_reaction_do_not_count(self) -> None:
        """E3 records statements before E4 can measure them."""
        profile = _profile()
        profile.observations = _observations(ACTOR_MIN_OBSERVATIONS * 2, abnormal_return=None)
        self.assertEqual(profile.impact_summary()["status"], "unmeasured")

    def test_measured_impact_disclaims_causality(self) -> None:
        """An abnormal return after a statement is association, not cause."""
        profile = _profile()
        profile.observations = _observations(ACTOR_MIN_OBSERVATIONS)
        self.assertIn("association only", profile.impact_summary()["detail"])

    def test_impact_can_be_scoped_to_one_company(self) -> None:
        profile = _profile()
        profile.observations = _observations(ACTOR_MIN_OBSERVATIONS)
        self.assertEqual(profile.impact_summary("NVDA")["status"], "measured")
        self.assertEqual(profile.impact_summary("AMD")["status"], "unmeasured")


class TestStatementFrequency(unittest.TestCase):
    def test_frequency_counts_by_type_and_ticker(self) -> None:
        profile = _profile()
        profile.observations = _observations(5)
        frequency = profile.statement_frequency()
        self.assertEqual(frequency["total_statements"], 5)
        self.assertEqual(frequency["by_event_type"]["earnings"], 5)
        self.assertEqual(frequency["distinct_tickers"], 1)

    def test_frequency_is_recorded_not_scored(self) -> None:
        """Weighting a frequent commentator is a question for E6."""
        self.assertIn("not scored", _profile().statement_frequency()["note"])

    def test_mean_novelty_is_none_when_unobserved(self) -> None:
        self.assertIsNone(_profile().mean_novelty())

    def test_mean_novelty_is_reported_once_observed(self) -> None:
        profile = _profile()
        profile.observations = _observations(4)
        self.assertEqual(profile.mean_novelty(), 0.5)


class TestRegistryFromEntities(unittest.TestCase):
    """Derived from E2 rather than maintained twice."""

    def setUp(self) -> None:
        self.profiles = build_actor_registry()

    def test_executives_become_insider_actors(self) -> None:
        self.assertIn("Jensen Huang", self.profiles)
        record = self.profiles["Jensen Huang"].record
        self.assertEqual(record.standing, ACTOR_STANDING_INSIDER)
        self.assertIn("NVDA", record.speaks_for)

    def test_every_derived_record_is_valid(self) -> None:
        for name, profile in sorted(self.profiles.items()):
            with self.subTest(actor=name):
                self.assertEqual(actor_problems(profile.record), [])

    def test_funds_contribute_no_actors(self) -> None:
        """A fund has no officer who speaks for it."""
        organizations = {p.record.organization for p in self.profiles.values()}
        self.assertNotIn("Vanguard S&P 500 ETF", organizations)

    def test_officers_are_authorities_on_company_matters_not_macro(self) -> None:
        topics = self.profiles["Jensen Huang"].record.topics
        self.assertIn("earnings", topics)
        self.assertNotIn("macro_shock", topics)

    def test_the_report_states_impact_is_unmeasured(self) -> None:
        report = registry_report(self.profiles)
        self.assertGreater(report["actors"], 50)
        self.assertEqual(report["with_measured_impact"], 0)
        self.assertEqual(report["unmeasured"], report["actors"])

    def test_registry_version_is_stamped(self) -> None:
        self.assertEqual(registry_report(self.profiles)["registry_version"], ACTOR_REGISTRY_VERSION)


class TestObservationRecording(unittest.TestCase):
    def test_an_observation_attaches_to_its_actor(self) -> None:
        profiles = {"Jensen Huang": _profile()}
        record_observation(profiles, _observations(1)[0])
        self.assertEqual(len(profiles["Jensen Huang"].observations), 1)

    def test_an_unknown_actor_is_refused(self) -> None:
        """An unknown speaker cannot accrue a track record."""
        with self.assertRaises(ActorIntelligenceError):
            record_observation({}, _observations(1)[0])

    def test_observations_are_extracted_from_events(self) -> None:
        events = [
            Event(
                entity="NVDA", published_time="2026-01-05 00:00:00",
                event_type="earnings", source="src", actor="Jensen Huang",
                novelty=0.6, evidence=[{"source_record_id": "r1"}],
            )
        ]
        observations = observations_from_events(events)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].actor, "Jensen Huang")
        self.assertEqual(observations[0].ticker, "NVDA")

    def test_anonymous_events_are_skipped(self) -> None:
        """An anonymous statement has no actor to attribute a record to."""
        events = [
            Event(
                entity="NVDA", published_time="2026-01-05 00:00:00",
                event_type="earnings", source="src",
                evidence=[{"source_record_id": "r1"}],
            )
        ]
        self.assertEqual(observations_from_events(events), [])


if __name__ == "__main__":
    unittest.main()
