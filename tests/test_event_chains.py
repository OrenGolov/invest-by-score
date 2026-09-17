"""Event revision / contradiction learning tests (Sprint E7).

Stores event chains — initial claim → correction → confirmation → reversal —
because the evolution itself is training information.

The properties that make a chain useful rather than decorative:

- a story counts ONCE. Treating four reports of one claim as four events
  would multiply it through every base rate downstream;
- a REVERSED claim weighs less than an unconfirmed one. A reversal is
  positive evidence the source was wrong, not an absence of evidence;
- `contested` (confirmed AND reversed) is a real state, not an average —
  collapsing it toward the middle is the same failure N1 refuses at the
  article level.
"""

from __future__ import annotations

import unittest

from core.config import (
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
    EVENT_DIRECTION_POSITIVE,
)
from core.event_chains import (
    EventChainError,
    build_chains,
    chain_problems,
    chain_report,
    classify_link,
    deduplicate_events,
)
from core.event_contract import Event


def _event(
    index: int,
    day: int,
    direction: str = EVENT_DIRECTION_NEGATIVE,
    source: str = "reuters",
    month: int = 1,
    entity: str = "NVDA",
    event_type: str = "litigation",
) -> Event:
    return Event(
        entity=entity,
        published_time=f"2026-{month:02d}-{day:02d} 10:00:00",
        event_type=event_type,
        source=source,
        direction=direction,
        evidence=[{"source_record_id": f"r{index}"}],
    )


class TestLinkClassification(unittest.TestCase):
    def test_a_directional_flip_is_a_reversal(self) -> None:
        link, reason = classify_link(
            _event(1, 5, EVENT_DIRECTION_NEGATIVE), _event(2, 7, EVENT_DIRECTION_POSITIVE)
        )
        self.assertEqual(link, CHAIN_LINK_REVERSAL)
        self.assertIn("was wrong", reason)

    def test_a_new_source_in_the_same_direction_is_a_confirmation(self) -> None:
        link, _ = classify_link(
            _event(1, 5, source="reuters"), _event(2, 6, source="bloomberg")
        )
        self.assertEqual(link, CHAIN_LINK_CONFIRMATION)

    def test_the_same_source_repeating_is_a_duplicate(self) -> None:
        """A repetition carries no new information."""
        link, _ = classify_link(
            _event(1, 5, source="reuters"), _event(2, 6, source="reuters")
        )
        self.assertEqual(link, CHAIN_LINK_DUPLICATE)

    def test_a_contradictory_report_is_a_correction(self) -> None:
        link, _ = classify_link(
            _event(1, 5, EVENT_DIRECTION_NEGATIVE),
            _event(2, 6, EVENT_DIRECTION_CONTRADICTORY),
        )
        self.assertEqual(link, CHAIN_LINK_CORRECTION)

    def test_contradictory_is_not_treated_as_an_opposite(self) -> None:
        """It is already unresolved; pairing it would invent a flip."""
        link, _ = classify_link(
            _event(1, 5, EVENT_DIRECTION_CONTRADICTORY),
            _event(2, 6, EVENT_DIRECTION_POSITIVE, source="bloomberg"),
        )
        self.assertNotEqual(link, CHAIN_LINK_REVERSAL)


class TestChainBuilding(unittest.TestCase):
    def test_a_lone_event_forms_an_open_chain(self) -> None:
        chains = build_chains([_event(1, 5)])
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].status, CHAIN_STATUS_OPEN)

    def test_the_first_link_is_always_the_initial_claim(self) -> None:
        chain = build_chains([_event(1, 5), _event(2, 6, source="bloomberg")])[0]
        self.assertEqual(chain.links[0].link_type, CHAIN_LINK_INITIAL)

    def test_the_chain_is_keyed_on_the_initial_claim(self) -> None:
        first = _event(1, 5)
        chain = build_chains([first, _event(2, 6, source="bloomberg")])[0]
        self.assertEqual(chain.chain_id, first.event_id)
        self.assertEqual(chain.initial_time, first.published_time)

    def test_events_are_chained_regardless_of_input_order(self) -> None:
        later, earlier = _event(2, 8, source="bloomberg"), _event(1, 5)
        chain = build_chains([later, earlier])[0]
        self.assertEqual(chain.chain_id, earlier.event_id)

    def test_events_beyond_the_window_are_separate_stories(self) -> None:
        chains = build_chains([_event(1, 5), _event(2, 5, month=3)])
        self.assertEqual(len(chains), 2)

    def test_events_inside_the_window_join_one_chain(self) -> None:
        chains = build_chains([
            _event(1, 5), _event(2, 5 + EVENT_CHAIN_WINDOW_DAYS - 1, source="bloomberg")
        ])
        self.assertEqual(len(chains), 1)

    def test_different_entities_never_share_a_chain(self) -> None:
        chains = build_chains([_event(1, 5, entity="NVDA"), _event(2, 6, entity="AMD")])
        self.assertEqual(len(chains), 2)

    def test_different_event_types_never_share_a_chain(self) -> None:
        chains = build_chains([
            _event(1, 5, event_type="litigation"),
            _event(2, 6, event_type="earnings"),
        ])
        self.assertEqual(len(chains), 2)

    def test_an_unparseable_timestamp_is_skipped(self) -> None:
        """Guessing its position would invent an evolution."""
        broken = _event(1, 5)
        broken.published_time = "not a date"
        self.assertEqual(build_chains([broken]), [])

    def test_a_zero_window_is_refused(self) -> None:
        with self.assertRaises(EventChainError):
            build_chains([_event(1, 5)], window_days=0)

    def test_every_built_chain_is_valid(self) -> None:
        chains = build_chains([
            _event(1, 5), _event(2, 6, source="bloomberg"),
            _event(3, 8, EVENT_DIRECTION_POSITIVE),
        ])
        for chain in chains:
            with self.subTest(chain=chain.chain_id):
                self.assertEqual(chain_problems(chain), [])


class TestSettledStatus(unittest.TestCase):
    def _status(self, events) -> str:
        return build_chains(events)[0].status

    def test_an_unconfirmed_claim_is_open(self) -> None:
        self.assertEqual(self._status([_event(1, 5)]), CHAIN_STATUS_OPEN)

    def test_corroboration_confirms(self) -> None:
        self.assertEqual(
            self._status([_event(1, 5), _event(2, 6, source="bloomberg")]),
            CHAIN_STATUS_CONFIRMED,
        )

    def test_a_flip_reverses(self) -> None:
        self.assertEqual(
            self._status([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)]),
            CHAIN_STATUS_REVERSED,
        )

    def test_a_contradictory_follow_up_corrects(self) -> None:
        self.assertEqual(
            self._status([_event(1, 5), _event(2, 6, EVENT_DIRECTION_CONTRADICTORY)]),
            CHAIN_STATUS_CORRECTED,
        )

    def test_confirmed_and_reversed_is_contested_not_averaged(self) -> None:
        """The same failure N1 refuses at the article level."""
        status = self._status([
            _event(1, 5), _event(2, 6, source="bloomberg"),
            _event(3, 8, EVENT_DIRECTION_POSITIVE),
        ])
        self.assertEqual(status, CHAIN_STATUS_CONTESTED)

    def test_a_duplicate_alone_leaves_the_chain_open(self) -> None:
        """Repetition from the same source settles nothing."""
        self.assertEqual(
            self._status([_event(1, 5), _event(2, 6, source="reuters")]),
            CHAIN_STATUS_OPEN,
        )


class TestReliabilityWeighting(unittest.TestCase):
    def test_a_reversal_weighs_less_than_an_unconfirmed_claim(self) -> None:
        """A reversal is evidence the source was wrong, not an absence."""
        self.assertLess(
            CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_REVERSED],
            CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_OPEN],
        )

    def test_confirmation_outweighs_everything(self) -> None:
        for status in (CHAIN_STATUS_OPEN, CHAIN_STATUS_CORRECTED,
                       CHAIN_STATUS_CONTESTED, CHAIN_STATUS_REVERSED):
            with self.subTest(status=status):
                self.assertGreater(
                    CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_CONFIRMED],
                    CHAIN_RELIABILITY_WEIGHT[status],
                )

    def test_a_contested_claim_outweighs_a_reversed_one(self) -> None:
        self.assertGreater(
            CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_CONTESTED],
            CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_REVERSED],
        )

    def test_the_chain_exposes_its_weight(self) -> None:
        chain = build_chains([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)])[0]
        self.assertEqual(
            chain.reliability_weight(), CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_REVERSED]
        )

    def test_revision_is_detectable(self) -> None:
        revised = build_chains([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)])[0]
        stood = build_chains([_event(1, 5)])[0]
        self.assertTrue(revised.was_revised())
        self.assertFalse(stood.was_revised())


class TestDeduplication(unittest.TestCase):
    def test_one_story_counts_once(self) -> None:
        """Four reports of one claim must not become four events."""
        events = [
            _event(1, 5), _event(2, 6, source="bloomberg"),
            _event(3, 7, source="ft"), _event(4, 8, EVENT_DIRECTION_POSITIVE),
        ]
        chains = build_chains(events)
        self.assertEqual(len(deduplicate_events(events, chains)), 1)

    def test_the_surviving_id_is_the_initial_claim(self) -> None:
        first = _event(1, 5)
        events = [first, _event(2, 6, source="bloomberg")]
        chains = build_chains(events)
        self.assertEqual(deduplicate_events(events, chains), [first.event_id])

    def test_separate_stories_each_count(self) -> None:
        events = [_event(1, 5), _event(2, 5, month=3)]
        chains = build_chains(events)
        self.assertEqual(len(deduplicate_events(events, chains)), 2)


class TestValidation(unittest.TestCase):
    def test_a_chain_without_links_is_refused(self) -> None:
        chain = build_chains([_event(1, 5)])[0]
        chain.links = []
        self.assertTrue(chain_problems(chain))

    def test_a_second_initial_claim_is_refused(self) -> None:
        chain = build_chains([_event(1, 5), _event(2, 6, source="bloomberg")])[0]
        chain.links[1].link_type = CHAIN_LINK_INITIAL
        self.assertTrue(any("second initial" in p for p in chain_problems(chain)))

    def test_a_status_disagreeing_with_the_links_is_refused(self) -> None:
        chain = build_chains([_event(1, 5)])[0]
        chain.status = CHAIN_STATUS_CONFIRMED
        self.assertTrue(any("disagrees" in p for p in chain_problems(chain)))

    def test_an_unknown_link_type_is_refused(self) -> None:
        chain = build_chains([_event(1, 5)])[0]
        chain.links[0].link_type = "rumour"
        self.assertTrue(chain_problems(chain))


class TestReport(unittest.TestCase):
    def test_the_reversal_rate_is_reported(self) -> None:
        """How often this source's claims turn out to be wrong."""
        chains = build_chains([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)])
        self.assertEqual(chain_report(chains)["reversal_rate"], 1.0)

    def test_a_contested_chain_counts_toward_the_reversal_rate(self) -> None:
        chains = build_chains([
            _event(1, 5), _event(2, 6, source="bloomberg"),
            _event(3, 8, EVENT_DIRECTION_POSITIVE),
        ])
        self.assertGreater(chain_report(chains)["reversal_rate"], 0.0)

    def test_the_report_counts_chains_and_events_separately(self) -> None:
        chains = build_chains([
            _event(1, 5), _event(2, 6, source="bloomberg"), _event(3, 7, source="ft"),
        ])
        report = chain_report(chains)
        self.assertEqual(report["chains"], 1)
        self.assertEqual(report["events"], 3)

    def test_an_empty_batch_does_not_divide_by_zero(self) -> None:
        self.assertEqual(chain_report([])["reversal_rate"], 0.0)

    def test_the_version_is_stamped(self) -> None:
        self.assertEqual(chain_report([])["chain_version"], EVENT_CHAIN_VERSION)


if __name__ == "__main__":
    unittest.main()
