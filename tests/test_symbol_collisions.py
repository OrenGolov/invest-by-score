"""Symbol-collision entity resolution: both failure directions, locked in.

MEASURED over 6,360 captured articles. For the 15 holdings whose ticker is an
English word or a colliding abbreviation, a bare-token symbol match admitted 116
articles of which **93 were not about the company at all**:

    ARM    "35-Years-Owned 1972 Chevrolet Corvette Coupe Project"
    CAT    "27 Meowing Memes Ministering Mood Boosts for Cat People Like You"
    KO     "Tyson Fury vs Anthony Joshua ... smiles then KO"
    MP     "Proposed deal to resolve Drumcree dispute, says DUP MP"
    KEEL   "keel-workflow 1.25.0"
    V      10 of 10 admitted articles were collisions

After requiring corroboration: **116 -> 10** on those tickers, **1,633 unchanged**
on the other 62.

THIS SUITE GUARDS BOTH DIRECTIONS, because the fix went wrong in each of them in
turn:

* TOO PERMISSIVE. An unbounded marker match let "rally" match inside "liteRALLY"
  and "neutRALLY", readmitting a baseball report and a developer-jargon article.
* TOO STRICT. Then the boundary fix lost its `\\b` anchors to shell escaping and
  became literal backspace characters, so the pattern matched NOTHING -- which
  rejected genuine bare-symbol coverage and, worse, invalidated the measurement
  taken against it without any test failing.

So `test_a_marker_matches_only_on_word_boundaries` and
`test_a_marker_still_matches_as_a_standalone_word` are a matched pair. Either one
alone passes while the code is broken in the other direction.
"""

from __future__ import annotations

import unittest

from core.config import (
    NEWS_COLLISION_PRONE_TICKERS,
    NEWS_FINANCE_MARKERS,
)
from core.news_adapter import _corroborated, _MARKER_PATTERNS, resolve_relevance


def article(headline, *, summary="", ticker=None, company=None):
    """A record in the shape the provider actually delivers.

    `ticker` and `company_name` default to None because MEASURED on live
    NewsAPI records that is what they are -- the provider does not resolve the
    entity for us, which is the whole reason this resolution exists.
    """
    return {
        "headline": headline,
        "summary": summary,
        "ticker": ticker,
        "company_name": company,
    }


class TheMeasuredFalsePositivesAreRejectedTests(unittest.TestCase):
    """Every one of these was admitted at relevance 1.0 before the fix."""

    CASES = (
        ("AEP", "[0day-rubbish] MultiTech Conduit AEP 6.3.6 Authenticated import_config"),
        ("ANET", "CTF-ANET: Clinical Time Frequency Aware-Attention Network"),
        ("ARM", "35-Years-Owned 1972 Chevrolet Corvette Coupe Project"),
        ("ARM", "HQGZQL Dual Magnifying Glass with Swing Arm Base and LED Light"),
        ("BE", "Ricky Hatton left estate after his tragic death, new figures show"),
        ("CAT", "27 Meowing Memes Ministering Mood Boosts for Cat People Like You"),
        ("CAT", "Hilary Duff is Releasing a New Song on Friday: CATastrophe"),
        ("CEG", "Proton Experimental brings fixes for Crysis, CEG-protected games"),
        ("KEEL", "keel-workflow 1.25.0"),
        ("KO", "Tyson Fury vs Anthony Joshua kicks off with tickling then KO"),
        ("KO", "ko-tasks-server 1.1.1"),
        ("MP", "Proposed deal to resolve Drumcree dispute, says DUP MP"),
        ("NOW", "Surface Pro 13-Inch (11th Edition) $1,747 at the Microsoft Store"),
        ("NU", "No More Windows For The Dutch Government, Ze Kiezen Nu Linux"),
        ("V", "The letter V appears in this sentence about nothing at all"),
    )

    def test_every_measured_collision_scores_zero(self):
        for ticker, headline in self.CASES:
            with self.subTest(ticker=ticker, headline=headline[:40]):
                self.assertEqual(
                    resolve_relevance(article(headline), ticker),
                    0.0,
                    "a bare symbol in unrelated text is not evidence about the "
                    "holding",
                )


class GenuineCoverageIsStillAdmittedTests(unittest.TestCase):
    """The other direction. A rule that drops real coverage is not a fix."""

    BY_NAME = (
        ("CAT", "Caterpillar raises full-year guidance on strong demand"),
        ("KO", "Coca-Cola reports quarterly earnings above estimates"),
        ("V", "Visa shares rally after analyst upgrade"),
        ("NOW", "ServiceNow stock jumps on revenue beat"),
        ("ARM", "Arm Holdings earnings beat as royalty revenue climbs"),
        ("AEP", "American Electric Power declares dividend"),
        ("MP", "MP Materials shares surge on rare earth deal"),
        ("BE", "Bloom Energy stock rises on fuel cell order"),
        ("GLW", "Corning lifts outlook on display demand"),
        ("TER", "Teradyne quarterly revenue tops estimates"),
        ("NU", "Nubank profit climbs on Brazil growth"),
        ("CEG", "Constellation Energy signs nuclear power deal"),
        ("ANET", "Arista Networks stock rallies on cloud orders"),
        ("STX", "Seagate raises guidance on data centre demand"),
    )

    BY_MARKER = (
        # Bare symbol plus a finance marker, with no company name. This is the
        # case the broken boundary pattern silently rejected.
        ("STX", "STX stock upgraded by analyst on price target raise"),
        ("KEEL", "KEEL stock jumps after drilling results"),
        ("V", "V shares hit a record on quarterly earnings"),
        ("CAT", "CAT dividend raised as profit beats estimates"),
    )

    def test_an_article_naming_the_company_is_admitted(self):
        for ticker, headline in self.BY_NAME:
            with self.subTest(ticker=ticker):
                self.assertGreaterEqual(
                    resolve_relevance(article(headline), ticker), 0.7
                )

    def test_a_symbol_with_a_finance_marker_is_admitted(self):
        for ticker, headline in self.BY_MARKER:
            with self.subTest(ticker=ticker):
                self.assertGreaterEqual(
                    resolve_relevance(article(headline), ticker),
                    0.7,
                    "a bare symbol alongside finance language is genuine "
                    "coverage; rejecting it was the second failure direction",
                )

    def test_naming_the_company_scores_at_least_as_high_as_the_symbol(self):
        # The company name is the strongest signal available and must never
        # score below the bare symbol it replaces.
        named = resolve_relevance(
            article("Caterpillar raises full-year guidance"), "CAT"
        )
        symbol = resolve_relevance(
            article("CAT raises full-year guidance"), "CAT"
        )
        self.assertGreaterEqual(named, symbol)


class TheWordBoundaryPairTests(unittest.TestCase):
    """A matched pair. Either test alone passes while the code is broken."""

    def test_a_marker_matches_only_on_word_boundaries(self):
        # "rally" inside "literally" admitted a baseball report for ARM.
        self.assertFalse(
            _corroborated("ARM", "the team literally had to drop a bullpen arm")
        )
        self.assertFalse(
            _corroborated("KO", "programmer sounds neutrally traditional")
        )

    def test_a_marker_still_matches_as_a_standalone_word(self):
        # The boundary fix lost its anchors to escaping and matched NOTHING,
        # which no single-direction test would have caught.
        self.assertTrue(_corroborated("V", "V shares rally after the open"))
        self.assertTrue(_corroborated("STX", "STX stock upgraded by an analyst"))

    def test_every_marker_pattern_carries_real_boundaries(self):
        # Asserted on the compiled pattern, because the bug was that the
        # anchors became literal control characters that still "looked" right.
        self.assertEqual(len(_MARKER_PATTERNS), len(NEWS_FINANCE_MARKERS))
        for pattern in _MARKER_PATTERNS:
            with self.subTest(pattern=pattern.pattern):
                self.assertTrue(pattern.pattern.startswith("\\b"))
                self.assertTrue(pattern.pattern.endswith("\\b"))
                self.assertNotIn("\x08", pattern.pattern)


class TheRuleIsNarrowlyScopedTests(unittest.TestCase):
    """62 of the 77 holdings must be completely unaffected."""

    def test_a_normal_ticker_needs_no_corroboration(self):
        # NVDA and MSFT do not collide with ordinary English, so requiring
        # corroboration there would cost coverage for no gain. MEASURED: 1,633
        # admitted articles across 48 normal tickers, unchanged by the fix.
        for ticker in ("NVDA", "MSFT", "AAPL", "AMZN", "TSLA", "GOOGL"):
            with self.subTest(ticker=ticker):
                self.assertEqual(
                    resolve_relevance(
                        article(f"{ticker} appears in this unrelated sentence"),
                        ticker,
                    ),
                    1.0,
                )

    def test_a_provider_asserted_match_needs_no_corroboration(self):
        # The provider resolved the entity itself, which is a stronger claim
        # than the symbol happening to appear in the text.
        self.assertEqual(
            resolve_relevance(
                article("Cat memes for cat people", ticker="CAT"), "CAT"
            ),
            1.0,
        )

    def test_only_the_listed_tickers_are_affected(self):
        self.assertEqual(len(NEWS_COLLISION_PRONE_TICKERS), 15)
        for ticker in NEWS_COLLISION_PRONE_TICKERS:
            with self.subTest(ticker=ticker):
                self.assertTrue(ticker.isupper())


class TheConfigRefusesWhatWouldGoDarkTests(unittest.TestCase):
    """A listed ticker with no corroborator would reject ALL of its news."""

    def test_every_listed_ticker_has_a_corroborating_name(self):
        for ticker, names in NEWS_COLLISION_PRONE_TICKERS.items():
            with self.subTest(ticker=ticker):
                self.assertTrue(
                    names,
                    f"{ticker} would go permanently dark: every article about "
                    f"it would be rejected",
                )

    def test_corroborating_names_are_lower_case(self):
        # Matching is done on lowered text, so an upper-case entry would never
        # match and the holding would go dark without any error.
        for ticker, names in NEWS_COLLISION_PRONE_TICKERS.items():
            for name in names:
                with self.subTest(ticker=ticker, name=name):
                    self.assertEqual(name, name.lower())

    def test_markers_are_lower_case(self):
        for marker in NEWS_FINANCE_MARKERS:
            with self.subTest(marker=marker):
                self.assertEqual(marker, marker.lower())

    def test_the_marker_list_is_not_empty(self):
        # With no markers, a holding whose coverage never prints its full
        # company name would go dark.
        self.assertTrue(NEWS_FINANCE_MARKERS)


if __name__ == "__main__":
    unittest.main()
