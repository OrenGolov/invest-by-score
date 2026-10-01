"""A9 news event tests — the operator's examples, graded from realized outcomes.

THE LOAD-BEARING PROPERTY: the headline never sets the priority. A headline is the
most tempting thing in this system to grade directly — "beats expectations by 8%"
reads urgent — and whether that KIND of event has ever moved the price is a
different question. `test_tone_does_not_change_the_verdict` and
`test_an_unrateable_type_is_unknown_not_low` are the tests that hold that line.

MEASURED against the shipped store (2,921 memories): only `earnings` carries a 20d
response, for 1,954 outcomes with a median absolute move of 5.48%. Every other type
has 1d responses only, so at the 20d horizon they are genuinely unrateable and must
report UNKNOWN_IMPACT rather than a guess.
"""

from __future__ import annotations

import unittest

from core.alert_priority import grade
from core.config import (
    ALERT_PRIORITY_MEDIUM,
    ALERT_PRIORITY_VERY_HIGH,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    EVENT_IMPACT_HIGH,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_MIN_MEDIAN_MOVE,
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    NEWS_EVENT_MIN_RELEVANCE,
)
from core.event_memory import EventMemory
from core.news_event_alert import (
    NEWS_EVENT_NOT_EVALUATED,
    NewsEventAlertError,
    leading_event,
    news_event_alert,
    summary_for,
    title_for,
)


def article(
    headline,
    *,
    ticker="AVGO",
    summary="",
    company="Broadcom",
    source="Reuters",
    published="2026-10-01T20:30:00+00:00",
):
    return {
        "headline": headline,
        "summary": summary,
        "ticker": ticker,
        "company_name": company,
        "published_time": published,
        "source": source,
        "url": "https://example.invalid/story",
    }


def snapshot(articles, status="OK", reason=None):
    out = {"status": status, "credible": list(articles)}
    if reason:
        out["reason"] = reason
    return out


def memory(event_type, move, *, index=0, horizon="20d"):
    """One remembered outcome of a given type, with a response at `horizon`."""
    return EventMemory(
        event_id=f"{event_type}-{index}",
        ticker="AVGO",
        published_time="2026-01-01T00:00:00+00:00",
        event_type=event_type,
        direction="positive",
        response={horizon: {"abnormal_return": move}},
    )


def many(event_type, move, count=20, horizon="20d"):
    return [
        memory(event_type, move, index=index, horizon=horizon)
        for index in range(count)
    ]


class TheOperatorsExamplesTests(unittest.TestCase):
    """The subject lines from the specification, produced end to end."""

    def test_an_earnings_event_with_history_reads_like_the_example(self):
        # "Very High: AVGO - Earnings Results"
        alert = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings above estimates",
                              summary="Revenue beat by 8%.")]),
            "AVGO",
            memories=many("earnings", 0.055),
            as_of="2026-10-01",
        )
        self.assertEqual(alert["event_type"], "earnings")
        self.assertEqual(alert["title"], "Earnings Results")
        self.assertEqual(alert["verdict"], EVENT_IMPACT_HIGH)
        self.assertEqual(grade(alert)["priority"], ALERT_PRIORITY_VERY_HIGH)

    def test_a_partnership_is_classified_as_a_strategic_announcement(self):
        alert = news_event_alert(
            snapshot([article("Nvidia announces strategic partnership on AI",
                              ticker="NVDA", company="Nvidia",
                              summary="A joint venture to build AI infrastructure.")]),
            "NVDA",
            memories=many("strategic_announcement", 0.06),
            as_of="2026-10-01",
        )
        self.assertEqual(alert["event_type"], "strategic_announcement")
        self.assertEqual(alert["title"], "Strategic Announcement")

    def test_each_taxonomy_entry_has_a_human_title(self):
        # A subject line must read "Earnings Results", not "earnings".
        for event_type in (
            "earnings",
            "guidance",
            "litigation",
            "regulation",
            "product_launch",
            "macro_shock",
            "m_and_a",
            "strategic_announcement",
            "management_commentary",
            "other",
        ):
            with self.subTest(event_type=event_type):
                title = title_for(event_type)
                self.assertTrue(title)
                self.assertNotEqual(title, event_type)

    def test_an_unknown_type_falls_back_rather_than_crashing(self):
        self.assertEqual(title_for("something_new"), "Company News")


class TheHeadlineNeverSetsThePriorityTests(unittest.TestCase):
    """A9's reason for existing, stated as tests."""

    def test_tone_does_not_change_the_verdict(self):
        # A cheerful press release and a grim one about the SAME event type move
        # the price by the same measured amount.
        memories = many("earnings", 0.055)
        positive = news_event_alert(
            snapshot([article("Broadcom quarterly earnings beat expectations",
                              summary="Excellent results, superb growth.")]),
            "AVGO", memories=memories, as_of="2026-10-01",
        )
        negative = news_event_alert(
            snapshot([article("Broadcom quarterly earnings miss expectations",
                              summary="Poor results, declining sales.")]),
            "AVGO", memories=memories, as_of="2026-10-01",
        )
        # Both must classify as the SAME type for the comparison to mean
        # anything: a differing type would change the verdict legitimately, and
        # the test would prove nothing about tone.
        self.assertEqual(positive["event_type"], negative["event_type"])
        self.assertEqual(positive["verdict"], negative["verdict"])
        self.assertEqual(
            grade(positive)["priority"], grade(negative)["priority"]
        )

    def test_but_tone_is_still_reported(self):
        # It is direction, which a reader acts on differently from size.
        memories = many("earnings", 0.055)
        positive = news_event_alert(
            snapshot([article("Broadcom quarterly earnings beat estimates",
                              summary="Excellent strong growth and superb demand.")]),
            "AVGO", memories=memories, as_of="2026-10-01",
        )
        negative = news_event_alert(
            snapshot([article("Broadcom quarterly earnings miss estimates",
                              summary="Poor weak declining demand, worrying outlook.")]),
            "AVGO", memories=memories, as_of="2026-10-01",
        )
        self.assertGreater(positive["tone"], negative["tone"])

    def test_a_dramatic_headline_about_a_low_impact_type_stays_low(self):
        # The verdict follows the TYPE's history, not the wording.
        alert = news_event_alert(
            snapshot([article("Broadcom CEO makes explosive shock comments",
                              summary="Management said demand is strong.")]),
            "AVGO",
            memories=many("management_commentary", 0.004),
            as_of="2026-10-01",
        )
        self.assertEqual(alert["verdict"], EVENT_IMPACT_LOW)
        self.assertEqual(alert["severity"], ALERT_SEVERITY_INFO)

    def test_the_threshold_is_the_shared_one_not_a_local_copy(self):
        just_below = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings")]),
            "AVGO",
            memories=many("earnings", EVENT_IMPACT_MIN_MEDIAN_MOVE - 0.001),
            as_of="2026-10-01",
        )
        just_above = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings")]),
            "AVGO",
            memories=many("earnings", EVENT_IMPACT_MIN_MEDIAN_MOVE + 0.001),
            as_of="2026-10-01",
        )
        self.assertEqual(just_below["verdict"], EVENT_IMPACT_LOW)
        self.assertEqual(just_above["verdict"], EVENT_IMPACT_HIGH)

    def test_a_high_impact_event_is_warn_and_a_low_one_is_info(self):
        high = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings")]),
            "AVGO", memories=many("earnings", 0.08), as_of="2026-10-01",
        )
        low = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings")]),
            "AVGO", memories=many("earnings", 0.004), as_of="2026-10-01",
        )
        self.assertEqual(high["severity"], ALERT_SEVERITY_WARN)
        self.assertEqual(low["severity"], ALERT_SEVERITY_INFO)


class AnUnreadableProviderIsNotAQuietDayTests(unittest.TestCase):
    """The distinction the operator most needs, and the one silence destroys."""

    def test_an_unavailable_provider_reports_not_evaluated(self):
        alert = news_event_alert(
            {"status": "UNAVAILABLE", "reason": "HTTP 429 rate limited"},
            "AVGO", memories=many("earnings", 0.055), as_of="2026-10-01",
        )
        self.assertEqual(alert["verdict"], NEWS_EVENT_NOT_EVALUATED)
        self.assertNotEqual(alert["verdict"], EVENT_IMPACT_NO_EVENT)
        self.assertFalse(alert["measured"])

    def test_an_unavailable_provider_carries_no_priority(self):
        # A blind detector must not be graded, and must not be buried as Low.
        alert = news_event_alert(
            {"status": "UNAVAILABLE", "reason": "no API key"},
            "AVGO", as_of="2026-10-01",
        )
        self.assertIsNone(grade(alert)["priority"])

    def test_the_providers_reason_is_carried_through(self):
        alert = news_event_alert(
            {"status": "UNAVAILABLE", "reason": "HTTP 429 rate limited"},
            "AVGO", as_of="2026-10-01",
        )
        self.assertIn("429", alert["reason"])

    def test_a_missing_snapshot_is_not_evaluated(self):
        alert = news_event_alert(None, "AVGO", as_of="2026-10-01")
        self.assertEqual(alert["verdict"], NEWS_EVENT_NOT_EVALUATED)

    def test_a_readable_but_empty_day_is_a_real_negative(self):
        alert = news_event_alert(snapshot([]), "AVGO", as_of="2026-10-01")
        self.assertEqual(alert["verdict"], EVENT_IMPACT_NO_EVENT)
        self.assertTrue(alert["measured"])

    def test_the_two_negatives_are_distinguishable(self):
        could_not_look = news_event_alert(
            {"status": "UNAVAILABLE"}, "AVGO", as_of="2026-10-01"
        )
        nothing_happened = news_event_alert(
            snapshot([]), "AVGO", as_of="2026-10-01"
        )
        self.assertNotEqual(
            could_not_look["verdict"], nothing_happened["verdict"]
        )
        self.assertNotEqual(
            could_not_look["measured"], nothing_happened["measured"]
        )


class AnUnrateableTypeIsUnknownNotLowTests(unittest.TestCase):
    """MEASURED in A3: a median |move| at n=3 spans a five-fold range."""

    def test_too_few_analogs_gives_unknown(self):
        alert = news_event_alert(
            snapshot([article("Broadcom raises its guidance for the year")]),
            "AVGO",
            memories=many("guidance", 0.05, count=2),
            as_of="2026-10-01",
        )
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)
        self.assertNotEqual(alert["verdict"], EVENT_IMPACT_LOW)

    def test_unknown_reports_no_median_at_all(self):
        # An unrated type has no median, not a median of zero.
        alert = news_event_alert(
            snapshot([article("Broadcom raises its guidance")]),
            "AVGO", memories=many("guidance", 0.05, count=2), as_of="2026-10-01",
        )
        self.assertIsNone(alert["median_abs_move"])
        self.assertFalse(alert["measured"])

    def test_unknown_grades_medium_rather_than_being_dropped(self):
        alert = news_event_alert(
            snapshot([article("Broadcom raises its guidance")]),
            "AVGO", memories=many("guidance", 0.05, count=2), as_of="2026-10-01",
        )
        self.assertEqual(grade(alert)["priority"], ALERT_PRIORITY_MEDIUM)

    def test_no_memories_at_all_gives_unknown(self):
        alert = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings")]),
            "AVGO", memories=[], as_of="2026-10-01",
        )
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)

    def test_memories_of_a_different_type_do_not_rate_this_one(self):
        # 1,954 earnings outcomes say nothing about a litigation event.
        alert = news_event_alert(
            snapshot([article("Broadcom sued in a class action over labelling")]),
            "AVGO", memories=many("earnings", 0.055, count=100), as_of="2026-10-01",
        )
        self.assertEqual(alert["event_type"], "litigation")
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)

    def test_memories_at_another_horizon_do_not_rate_this_one(self):
        # MEASURED on the shipped store: every non-earnings type carries a 1d
        # response and nothing at 20d, so at 20d they are genuinely unrateable.
        alert = news_event_alert(
            snapshot([article("Intel unveils a new data centre chip",
                              ticker="INTC", company="Intel")]),
            "INTC",
            memories=many("product_launch", 0.05, count=40, horizon="1d"),
            as_of="2026-10-01",
            horizon="20d",
        )
        self.assertEqual(alert["verdict"], EVENT_IMPACT_UNKNOWN)


class RelevanceTests(unittest.TestCase):
    def test_a_company_name_match_is_admitted(self):
        # Most financial journalism writes "Broadcom", not "AVGO". A 1.0-only
        # floor would discard nearly every real article.
        found = leading_event(
            snapshot([article("Broadcom reports quarterly earnings", ticker="")]),
            "AVGO",
        )
        self.assertTrue(found["found"])
        self.assertEqual(found["relevance"], NEWS_EVENT_MIN_RELEVANCE)

    def test_an_exact_ticker_match_scores_highest(self):
        found = leading_event(
            snapshot([article("AVGO reports quarterly earnings")]), "AVGO"
        )
        self.assertEqual(found["relevance"], 1.0)

    def test_an_unrelated_article_is_excluded(self):
        alert = news_event_alert(
            snapshot([article("Some other firm reports earnings",
                              ticker="ZZZZ", company="Totally Different Inc")]),
            "AVGO", memories=many("earnings", 0.055), as_of="2026-10-01",
        )
        self.assertEqual(alert["verdict"], EVENT_IMPACT_NO_EVENT)

    def test_the_most_relevant_article_wins_not_the_newest(self):
        # The newest is often a syndicated copy; relevance is the selector.
        found = leading_event(
            snapshot(
                [
                    article("Broadcom mentioned in a sector roundup", ticker=""),
                    article("AVGO reports quarterly earnings",
                            published="2026-09-30T10:00:00+00:00"),
                ]
            ),
            "AVGO",
        )
        self.assertEqual(found["relevance"], 1.0)
        self.assertIn("AVGO", found["headline"])

    def test_a_malformed_article_does_not_lose_the_day(self):
        found = leading_event(
            snapshot([{"nonsense": True},
                      article("Broadcom reports quarterly earnings")]),
            "AVGO",
        )
        self.assertTrue(found["found"])

    def test_a_non_mapping_snapshot_raises(self):
        with self.assertRaises(NewsEventAlertError):
            leading_event(["not", "a", "mapping"], "AVGO")

    def test_an_empty_ticker_raises(self):
        with self.assertRaises(NewsEventAlertError):
            news_event_alert(snapshot([]), "")


class SummaryBlockTests(unittest.TestCase):
    def setUp(self):
        self.alert = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings above estimates",
                              summary="Revenue beat by 8%.")]),
            "AVGO",
            memories=many("earnings", 0.055),
            as_of="2026-10-01",
        )
        self.summary = summary_for(self.alert)

    def test_every_section_the_template_needs_is_present(self):
        for key in ("who", "what", "when", "why"):
            with self.subTest(key=key):
                self.assertTrue(self.summary[key])

    def test_the_source_is_named_in_who(self):
        self.assertIn("Reuters", self.summary["who"])

    def test_the_why_block_quotes_the_measurement(self):
        joined = " ".join(self.summary["why"])
        # The ANALOG COUNT and the horizon must both be quoted. The number is
        # the fixture's, not the shipped store's: asserting 1,954 here would
        # make the test depend on data/event_memory.jsonl, which is gitignored
        # and absent on a fresh clone -- the checkout-dependence bug class.
        self.assertIn(f"{self.alert['analogs']} remembered", joined)
        self.assertIn("20d", joined)

    def test_the_why_block_states_that_tone_is_direction_not_size(self):
        joined = " ".join(self.summary["why"])
        self.assertIn("direction rather than size", joined)

    def test_an_unrateable_event_says_so_rather_than_filling_the_gap(self):
        alert = news_event_alert(
            snapshot([article("Broadcom raises its guidance")]),
            "AVGO", memories=many("guidance", 0.05, count=2), as_of="2026-10-01",
        )
        joined = " ".join(summary_for(alert)["why"])
        self.assertIn("too few", joined.lower())

    def test_a_non_mapping_alert_raises(self):
        with self.assertRaises(NewsEventAlertError):
            summary_for(["nope"])


class EndToEndTests(unittest.TestCase):
    def test_the_alert_stores_and_renders_as_the_operator_specified(self):
        from core.alert_email import render_text, subject_for
        from core.alert_store import build_record

        alert = news_event_alert(
            snapshot([article("Broadcom reports quarterly earnings above estimates",
                              summary="Revenue beat by 8%.")]),
            "AVGO",
            memories=many("earnings", 0.055),
            as_of="2026-10-01",
        )
        graded = grade(alert)
        record = build_record(
            alert,
            graded,
            title=alert["title"],
            summary=summary_for(alert),
            detected_at="2026-10-01T20:45:00+00:00",
        )
        self.assertEqual(
            subject_for(record), "Very High: AVGO - Earnings Results"
        )
        body = render_text(record)
        for needle in ("Action:", "Who:", "What:", "When:", "Why it Matters:"):
            with self.subTest(section=needle):
                self.assertIn(needle, body)

    def test_the_same_event_on_two_days_shares_an_id(self):
        from core.alert_store import build_record

        memories = many("earnings", 0.055)
        ids = set()
        for as_of, at in (
            ("2026-10-01", "2026-10-01T20:45:00+00:00"),
            ("2026-10-02", "2026-10-02T20:45:00+00:00"),
        ):
            alert = news_event_alert(
                snapshot([article("Broadcom reports quarterly earnings")]),
                "AVGO", memories=memories, as_of=as_of,
            )
            ids.add(
                build_record(
                    alert, grade(alert), title=alert["title"], detected_at=at
                )["alert_id"]
            )
        self.assertEqual(len(ids), 1, "an unchanged event must not re-email")

    def test_a_news_alert_never_recommends_a_direction(self):
        from core.config import ALERT_DIRECTIONAL_ACTIONS

        for move, count in ((0.08, 40), (0.004, 40), (0.05, 2)):
            alert = news_event_alert(
                snapshot([article("Broadcom reports quarterly earnings")]),
                "AVGO", memories=many("earnings", move, count=count),
                as_of="2026-10-01",
            )
            with self.subTest(move=move, count=count):
                self.assertNotIn(
                    grade(alert)["action"], ALERT_DIRECTIONAL_ACTIONS
                )


if __name__ == "__main__":
    unittest.main()
