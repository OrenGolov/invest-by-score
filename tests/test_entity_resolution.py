"""Entity resolution tests (Sprint E2).

The E2 rule, pinned: "Bad resolution must not silently enter training."

The emphasis is on *silently*. Bad resolution is unavoidable — headlines are
ambiguous and companies share words. What these tests pin is that a failed
resolution is DISTINGUISHABLE from a correct exclusion, and cannot reach a
training row without being seen.

The specific defect E2 fixes: the N1 heuristic returned 0.0 both for
"Broadcom announces a deal" (correctly not about NVDA) and for "Jensen Huang
announces a GPU" (about NVDA, but unresolvable). Those are different
failures and now have different answers.
"""

from __future__ import annotations

import unittest

from core.config import (
    ENTITY_MATCH_ALIAS,
    ENTITY_MATCH_AMBIGUOUS,
    ENTITY_MATCH_EXECUTIVE,
    ENTITY_MATCH_LEGAL_NAME,
    ENTITY_MATCH_METHODS,
    ENTITY_MATCH_NONE,
    ENTITY_MATCH_TICKER,
    ENTITY_MIN_TRAINING_CONFIDENCE,
    ENTITY_RESOLVER_VERSION,
)
from core.entity_resolution import (
    EntityRecord,
    EntityResolutionError,
    build_default_entity_registry,
    registry_problems,
    resolution_report,
    resolve_article,
    resolve_entity,
)
from core.event_contract import events_from_news_snapshot


class TestRegistry(unittest.TestCase):
    def test_the_default_registry_is_valid(self) -> None:
        self.assertEqual(registry_problems(build_default_entity_registry()), [])

    def test_a_shared_alias_is_reported(self) -> None:
        """It would make every mention of that alias ambiguous."""
        registry = {
            "AAA": EntityRecord("AAA", "Alpha Inc.", ("Apex",)),
            "BBB": EntityRecord("BBB", "Beta Inc.", ("Apex",)),
        }
        self.assertTrue(any("claimed by both" in p for p in registry_problems(registry)))

    def test_an_empty_legal_name_is_reported(self) -> None:
        registry = {"AAA": EntityRecord("AAA", "")}
        self.assertTrue(any("legal_name" in p for p in registry_problems(registry)))

    def test_a_key_mismatch_is_reported(self) -> None:
        registry = {"AAA": EntityRecord("BBB", "Beta Inc.")}
        self.assertTrue(any("does not match" in p for p in registry_problems(registry)))


class TestMatchMethods(unittest.TestCase):
    def test_ticker_in_text_resolves_strongest(self) -> None:
        resolution = resolve_entity("NVDA beats earnings", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_TICKER)
        self.assertEqual(resolution.confidence, 1.0)

    def test_legal_name_resolves(self) -> None:
        resolution = resolve_entity("NVIDIA Corporation reports results", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_LEGAL_NAME)

    def test_alias_resolves(self) -> None:
        resolution = resolve_entity("Nvidia beats earnings", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_ALIAS)

    def test_a_provider_tag_is_trusted(self) -> None:
        """The provider asserted the association; that is the strongest signal."""
        resolution = resolve_entity("no names here", "NVDA", provider_ticker="NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_TICKER)
        self.assertTrue(resolution.matched)

    def test_every_method_carries_a_confidence(self) -> None:
        from core.config import ENTITY_MATCH_CONFIDENCE

        for method in ENTITY_MATCH_METHODS:
            with self.subTest(method=method):
                self.assertIn(method, ENTITY_MATCH_CONFIDENCE)

    def test_resolver_version_is_stamped(self) -> None:
        self.assertEqual(resolve_entity("Nvidia", "NVDA").resolver_version, ENTITY_RESOLVER_VERSION)


class TestTheDefectE2Fixes(unittest.TestCase):
    """An unresolvable article must not look like an irrelevant one."""

    def test_an_executive_mention_now_resolves(self) -> None:
        """Previously 0.0 — indistinguishable from 'not about NVDA'."""
        resolution = resolve_entity("Jensen Huang announces a new GPU", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_EXECUTIVE)
        self.assertTrue(resolution.matched)

    def test_an_executive_is_weaker_evidence_than_a_ticker(self) -> None:
        """A CEO is often quoted about the industry, not the company."""
        executive = resolve_entity("Jensen Huang announces a new GPU", "NVDA")
        ticker = resolve_entity("NVDA announces a new GPU", "NVDA")
        self.assertLess(executive.confidence, ticker.confidence)

    def test_a_different_company_resolves_to_none_with_the_reason(self) -> None:
        resolution = resolve_entity("Broadcom announces a deal", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_NONE)
        self.assertIn("AVGO", resolution.candidates)
        self.assertIn("mentions", resolution.reason)

    def test_no_entity_at_all_resolves_to_none_with_a_different_reason(self) -> None:
        resolution = resolve_entity("The company said it will expand", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_NONE)
        self.assertEqual(resolution.candidates, [])

    def test_the_two_none_cases_are_distinguishable(self) -> None:
        """Both are excluded, but for different reasons — and it shows."""
        other_company = resolve_entity("Broadcom announces a deal", "NVDA")
        nothing = resolve_entity("The company said it will expand", "NVDA")
        self.assertNotEqual(other_company.reason, nothing.reason)
        self.assertNotEqual(other_company.candidates, nothing.candidates)


class TestAmbiguity(unittest.TestCase):
    def test_two_named_entities_resolve_to_ambiguous(self) -> None:
        resolution = resolve_entity("Nvidia and AMD both rallied", "NVDA")
        self.assertEqual(resolution.method, ENTITY_MATCH_AMBIGUOUS)
        self.assertFalse(resolution.matched)

    def test_ambiguity_carries_zero_confidence(self) -> None:
        self.assertEqual(resolve_entity("Nvidia and AMD rallied", "NVDA").confidence, 0.0)

    def test_ambiguity_names_the_other_candidates(self) -> None:
        resolution = resolve_entity("Nvidia and AMD both rallied", "NVDA")
        self.assertIn("AMD", resolution.candidates)
        self.assertIn("AMD", resolution.reason)

    def test_ambiguity_is_not_training_eligible(self) -> None:
        """Attributing it to one company would be a guess."""
        self.assertFalse(resolve_entity("Nvidia and AMD rallied", "NVDA").is_training_eligible())


class TestShortTickerGuard(unittest.TestCase):
    """The V/BE class of error: short tickers are ordinary English words."""

    def test_a_bare_short_ticker_does_not_match(self) -> None:
        self.assertFalse(resolve_entity("The V shape recovery continued", "V").matched)

    def test_a_short_ticker_resolves_by_name(self) -> None:
        resolution = resolve_entity("Visa raised its outlook", "V")
        self.assertTrue(resolution.matched)
        self.assertEqual(resolution.method, ENTITY_MATCH_ALIAS)

    def test_bloom_energy_needs_its_name_not_be(self) -> None:
        self.assertFalse(resolve_entity("This will be a strong quarter", "BE").matched)
        self.assertTrue(resolve_entity("Bloom Energy raised guidance", "BE").matched)

    def test_a_long_ticker_still_matches_bare(self) -> None:
        self.assertTrue(resolve_entity("NVDA rallied", "NVDA").matched)


class TestPhraseBoundaries(unittest.TestCase):
    def test_scattered_words_do_not_match_a_multi_word_name(self) -> None:
        """'Advanced Micro Devices' must not match words strewn about."""
        text = "Advanced sensors, micro lenses and other devices shipped"
        self.assertFalse(resolve_entity(text, "AMD").matched)

    def test_a_substring_does_not_match(self) -> None:
        """'Metaverse' is not 'Meta'."""
        self.assertFalse(resolve_entity("Metaverse investment slowed", "META").matched)

    def test_matching_is_case_insensitive(self) -> None:
        for text in ("nvidia beats", "NVIDIA beats", "NvIdIa beats"):
            with self.subTest(text=text):
                self.assertTrue(resolve_entity(text, "NVDA").matched)


class TestTrainingEligibility(unittest.TestCase):
    def test_strong_matches_are_eligible(self) -> None:
        for text in ("NVDA rallied", "NVIDIA Corporation rallied", "Nvidia rallied"):
            with self.subTest(text=text):
                self.assertTrue(resolve_entity(text, "NVDA").is_training_eligible())

    def test_an_executive_match_sits_at_the_threshold(self) -> None:
        resolution = resolve_entity("Jensen Huang spoke", "NVDA")
        self.assertEqual(resolution.confidence, ENTITY_MIN_TRAINING_CONFIDENCE)
        self.assertTrue(resolution.is_training_eligible())

    def test_failed_resolutions_are_never_eligible(self) -> None:
        for text in ("Broadcom deal", "The company expanded", "Nvidia and AMD rallied"):
            with self.subTest(text=text):
                self.assertFalse(resolve_entity(text, "NVDA").is_training_eligible())

    def test_an_unregistered_ticker_is_refused_not_guessed(self) -> None:
        resolution = resolve_entity("ZZZZ soared", "ZZZZ")
        self.assertFalse(resolution.matched)
        self.assertIn("not in the entity registry", resolution.reason)

    def test_an_empty_ticker_raises(self) -> None:
        with self.assertRaises(EntityResolutionError):
            resolve_entity("anything", "")


class TestArticleResolution(unittest.TestCase):
    def test_an_article_resolves_across_its_fields(self) -> None:
        article = {"headline": "Results", "summary": "Nvidia beat estimates"}
        self.assertTrue(resolve_article(article, "NVDA").matched)

    def test_a_provider_tagged_article_resolves(self) -> None:
        self.assertTrue(resolve_article({"ticker": "NVDA", "headline": "x"}, "NVDA").matched)

    def test_a_non_dict_article_raises(self) -> None:
        with self.assertRaises(EntityResolutionError):
            resolve_article("an article", "NVDA")


class TestCoverageReport(unittest.TestCase):
    def test_the_report_separates_none_from_ambiguous(self) -> None:
        """They need different fixes: a registry gap vs hard source material."""
        resolutions = [
            resolve_entity("NVDA up", "NVDA"),
            resolve_entity("Nvidia and AMD up", "NVDA"),
            resolve_entity("nothing here", "NVDA"),
        ]
        report = resolution_report(resolutions)
        self.assertEqual(report["by_method"][ENTITY_MATCH_AMBIGUOUS], 1)
        self.assertEqual(report["by_method"][ENTITY_MATCH_NONE], 1)

    def test_the_report_counts_eligibility(self) -> None:
        resolutions = [resolve_entity("NVDA up", "NVDA"), resolve_entity("nothing", "NVDA")]
        report = resolution_report(resolutions)
        self.assertEqual(report["training_eligible"], 1)
        self.assertEqual(report["rejected"], 1)
        self.assertEqual(report["eligibility_rate"], 0.5)

    def test_an_empty_batch_does_not_divide_by_zero(self) -> None:
        self.assertEqual(resolution_report([])["eligibility_rate"], 0.0)


class TestEventIntegration(unittest.TestCase):
    """Bad resolution must not silently enter training."""

    def _snapshot(self) -> dict:
        return {
            "ticker": "NVDA",
            "status": "OK",
            "source_id": "newsapi_news",
            "articles": [
                {
                    "source_record_id": "r1", "published_time": "2026-01-05 10:00:00",
                    "category": "earnings", "tone": 0.7, "relevance": 0.9,
                    "source_weight": 0.8, "headline": "Nvidia beats earnings",
                },
                {
                    "source_record_id": "r2", "published_time": "2026-01-05 11:00:00",
                    "category": "other", "tone": 0.1, "relevance": 0.2,
                    "source_weight": 0.5, "headline": "The company will expand",
                },
            ],
        }

    def test_events_carry_their_resolution(self) -> None:
        for event in events_from_news_snapshot(self._snapshot()):
            with self.subTest(event=event.event_id):
                self.assertIn("method", event.entity_resolution)
                self.assertIn("confidence", event.entity_resolution)

    def test_a_resolved_and_an_unresolved_event_are_distinguishable(self) -> None:
        events = events_from_news_snapshot(self._snapshot())
        self.assertEqual([e.is_entity_resolved() for e in events], [True, False])

    def test_strict_mode_drops_unresolved_events(self) -> None:
        strict = events_from_news_snapshot(self._snapshot(), require_resolved_entity=True)
        self.assertEqual(len(strict), 1)
        self.assertTrue(all(e.is_entity_resolved() for e in strict))

    def test_an_event_without_a_resolution_is_treated_as_unresolved(self) -> None:
        """An absent resolution is the most silent failure of all."""
        from core.event_contract import Event

        event = Event(
            entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
            source="src", evidence=[{"source_record_id": "r1"}],
        )
        self.assertFalse(event.is_entity_resolved())


if __name__ == "__main__":
    unittest.main()


class TestPortfolioCoverage(unittest.TestCase):
    """The registry must cover the portfolio it is asked to resolve.

    At 15 entries, 81% of the portfolio resolved to `none` — the resolver
    was working correctly and reporting that it had no idea who most of
    these companies were.
    """

    def setUp(self) -> None:
        from fetch_data import PORTFOLIO_TICKERS

        self.portfolio = PORTFOLIO_TICKERS
        self.registry = build_default_entity_registry()

    def test_every_portfolio_ticker_is_registered(self) -> None:
        missing = sorted(set(self.portfolio) - set(self.registry))
        self.assertEqual(missing, [], f"unregistered portfolio tickers: {missing}")

    def test_every_legal_name_matches_itself(self) -> None:
        """Regression: a name ending in "." could never match.

        The trailing \b required a word character after the period, so
        "Apple Inc." failed against text containing "Apple Inc." — 24 of 74
        entries were unmatchable.
        """
        from core.entity_resolution import _phrase_present

        for ticker, record in sorted(self.registry.items()):
            with self.subTest(ticker=ticker):
                self.assertTrue(
                    _phrase_present(record.legal_name, f"{record.legal_name} reported results"),
                    f"{record.legal_name!r} cannot match itself",
                )

    def test_funds_declare_no_executives(self) -> None:
        """A fund has no officer who speaks for it."""
        for ticker in ("VOO", "SOXX", "CIBR", "NASA"):
            with self.subTest(ticker=ticker):
                self.assertEqual(self.registry[ticker].executives, ())

    def test_portfolio_wide_eligibility_is_high(self) -> None:
        resolutions = [
            resolve_entity(f"{self.registry[t].legal_name} reported quarterly results", t)
            for t in self.portfolio
        ]
        report = resolution_report(resolutions)
        self.assertGreater(report["eligibility_rate"], 0.95)

    def test_the_remaining_rejection_is_genuine_ambiguity(self) -> None:
        """CCEP's name contains KO's — refusing to guess is correct."""
        resolution = resolve_entity(
            "Coca-Cola Europacific Partners reported results", "CCEP"
        )
        self.assertEqual(resolution.method, ENTITY_MATCH_AMBIGUOUS)
        self.assertIn("KO", resolution.candidates)


class TestTickerCaseSensitivity(unittest.TestCase):
    """A ticker is evidence only when written as a symbol.

    Length alone does not separate a symbol from an ordinary word: CAT is
    three characters, so it passed the short-ticker guard and matched the
    word "cat" with full ticker confidence. Adding RCAT ("Red Cat Holdings")
    made that collision reachable from real headlines.
    """

    def test_lowercase_word_is_not_a_ticker_match(self) -> None:
        resolution = resolve_entity("The cat sat on the mat", "CAT")
        self.assertEqual(resolution.method, ENTITY_MATCH_NONE)
        self.assertFalse(resolution.matched)

    def test_uppercase_symbol_still_matches(self) -> None:
        resolution = resolve_entity("CAT raises full-year guidance", "CAT")
        self.assertEqual(resolution.method, ENTITY_MATCH_TICKER)
        self.assertTrue(resolution.matched)

    def test_titlecase_word_inside_a_company_name_is_not_a_ticker(self) -> None:
        """The RCAT regression: "Red Cat" must not drag in CAT."""
        resolution = resolve_entity("Red Cat Holdings wins an Army drone contract", "RCAT")
        self.assertEqual(resolution.method, ENTITY_MATCH_LEGAL_NAME)
        self.assertNotIn("CAT", resolution.candidates)

    def test_legal_names_remain_case_insensitive(self) -> None:
        """Only ticker matching is case-sensitive; prose is not."""
        resolution = resolve_entity("caterpillar inc. reported results", "CAT")
        self.assertEqual(resolution.method, ENTITY_MATCH_LEGAL_NAME)


class TestNewPortfolioMembers(unittest.TestCase):
    """RCAT and ASTS were added to the portfolio; resolution must cover them."""

    def test_asts_resolves_by_alias(self) -> None:
        resolution = resolve_entity("AST SpaceMobile reported quarterly results", "ASTS")
        self.assertTrue(resolution.matched)
        self.assertTrue(resolution.is_training_eligible())

    def test_rcat_resolves_by_alias(self) -> None:
        resolution = resolve_entity("Red Cat Holdings reported quarterly results", "RCAT")
        self.assertTrue(resolution.matched)
        self.assertTrue(resolution.is_training_eligible())

    def test_unrelated_news_is_still_excluded(self) -> None:
        """The negative control: coverage must not become over-matching."""
        for ticker in ("RCAT", "ASTS"):
            with self.subTest(ticker=ticker):
                resolution = resolve_entity("Nvidia announces a new GPU", ticker)
                self.assertEqual(resolution.method, ENTITY_MATCH_NONE)
