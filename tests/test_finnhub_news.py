"""Finnhub company news: the normaliser, the dispatcher, and the fallback.

**WHY A SECOND PROVIDER.** MEASURED, NewsAPI's free tier allows 100 requests/day
against a need of roughly 150, which cost five consecutive nights of permanently
lost news and still forces the collector to rotate 40 of 75 tickers per run.
NewsAPI's only paid tier is $449/month for 55x more quota than this system uses.
Finnhub's free tier is 60 calls/minute.

**THE PROPERTY THAT MATTERS MOST** is `test_an_unrelated_article_is_not_claimed`.
Finnhub returns a `related` field naming the tickers an article is about, and the
normaliser populates `ticker` ONLY when the provider names the one requested.
Trusting the request parameter instead would assert an entity resolution the
provider never made — and `resolve_relevance` grants a provider-asserted match
1.0 without corroboration, so a wrong claim there bypasses the collision guard
entirely.

Fixtures are recorded payload shapes, not live calls: a test that needs the
network is a test that fails on a plane.
"""

from __future__ import annotations

import json
import os
import unittest
import urllib.error
from datetime import datetime, timezone
from unittest import mock

from core.config import (
    FINNHUB_API_KEY_ENV,
    FINNHUB_MAX_ARTICLES,
    FINNHUB_NEWS_URL,
    FINNHUB_SOURCE_ID,
    NEWS_MAX_ARTICLES,
    NEWS_PROVIDER_API_KEY_ENV,
    NEWS_PROVIDER_ORDER,
    NEWSAPI_SOURCE_ID,
)
from core.finnhub_news import available, fetch_company_news, normalize

AS_OF = datetime(2026, 10, 3, tzinfo=timezone.utc)

# 2026-10-01T18:00:00Z, inside the 7-day lookback window from AS_OF.
# Computed rather than guessed: my first value was six days off, which the
# ISO-rendering test caught.
WHEN = 1790877600


def entry(
    headline="Broadcom reports quarterly earnings above estimates",
    *,
    related="AVGO",
    summary="Revenue beat by 8%.",
    source="Reuters",
    stamp=WHEN,
    identifier=1,
    url="https://example.invalid/story",
):
    """One article in the shape Finnhub's /company-news actually returns."""
    return {
        "category": "company",
        "datetime": stamp,
        "headline": headline,
        "id": identifier,
        "image": "https://example.invalid/img.png",
        "related": related,
        "source": source,
        "summary": summary,
        "url": url,
    }


class TheNormaliserTests(unittest.TestCase):
    def test_it_produces_the_canonical_record_shape(self):
        records = normalize([entry()], "AVGO")
        self.assertEqual(len(records), 1)
        for field in (
            "source_record_id",
            "published_time",
            "headline",
            "summary",
            "url",
            "source_name",
            "ticker",
            "company_name",
            "source_quality",
            "tone",
        ):
            with self.subTest(field=field):
                self.assertIn(field, records[0])

    def test_the_unix_timestamp_becomes_an_iso_string(self):
        # The point-in-time filter parses ISO strings; a raw integer would be
        # unparseable and the article silently dropped.
        published = normalize([entry()], "AVGO")[0]["published_time"]
        self.assertTrue(published.startswith("2026-10-01T"))
        self.assertTrue(published.endswith("+00:00"))

    def test_a_provider_named_ticker_is_claimed(self):
        self.assertEqual(normalize([entry(related="AVGO")], "AVGO")[0]["ticker"], "AVGO")

    def test_a_ticker_in_a_multi_symbol_list_is_claimed(self):
        records = normalize([entry(related="AAPL,AVGO,MSFT")], "AVGO")
        self.assertEqual(records[0]["ticker"], "AVGO")

    def test_an_unrelated_article_is_not_claimed(self):
        """THE LOAD-BEARING TEST.

        `resolve_relevance` grants a provider-asserted match 1.0 WITHOUT
        corroboration, because the provider resolving the entity is stronger
        evidence than a symbol appearing in text. So claiming a ticker the
        provider did not name would bypass the collision guard entirely — the
        exact failure the guard was built to stop.
        """
        records = normalize([entry(related="AAPL,MSFT")], "AVGO")
        self.assertIsNone(records[0]["ticker"])

    def test_an_empty_related_field_claims_nothing(self):
        self.assertIsNone(normalize([entry(related="")], "AVGO")[0]["ticker"])

    def test_the_match_is_case_insensitive(self):
        self.assertEqual(normalize([entry(related="avgo")], "AVGO")[0]["ticker"], "AVGO")

    def test_a_headline_less_entry_is_skipped(self):
        self.assertEqual(normalize([entry(headline="")], "AVGO"), [])

    def test_a_non_dict_entry_is_skipped(self):
        self.assertEqual(normalize(["not a dict", None, 7], "AVGO"), [])

    def test_an_entry_with_no_timestamp_is_skipped(self):
        # Defaulting to "now" would make the article pass any point-in-time
        # check by construction.
        for stamp in (None, 0, "", "yesterday", -5):
            with self.subTest(stamp=stamp):
                self.assertEqual(normalize([entry(stamp=stamp)], "AVGO"), [])

    def test_one_bad_entry_does_not_lose_the_good_ones(self):
        records = normalize(
            [entry(identifier=1), {"nonsense": True}, entry(identifier=2)], "AVGO"
        )
        self.assertEqual(len(records), 2)

    def test_an_empty_payload_is_empty_not_an_error(self):
        self.assertEqual(normalize([], "AVGO"), [])
        self.assertEqual(normalize(None, "AVGO"), [])


class TheFetchDispositionTests(unittest.TestCase):
    """It never raises: a failure is an explicit disposition, never empty data."""

    def test_no_key_reports_key_required(self):
        with mock.patch.dict(os.environ, {FINNHUB_API_KEY_ENV: ""}):
            result = fetch_company_news("AVGO", AS_OF)
        self.assertEqual(result["status"], "provider_key_required")
        self.assertEqual(result["records"], [])

    def test_available_reflects_the_key(self):
        with mock.patch.dict(os.environ, {FINNHUB_API_KEY_ENV: ""}):
            self.assertFalse(available())
        with mock.patch.dict(os.environ, {FINNHUB_API_KEY_ENV: "k"}):
            self.assertTrue(available())

    def _fetch(self, payload=None, *, error=None):
        with mock.patch.dict(os.environ, {FINNHUB_API_KEY_ENV: "test-key"}):
            if error is not None:
                with mock.patch("urllib.request.urlopen", side_effect=error):
                    return fetch_company_news("AVGO", AS_OF)
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(payload).encode()
            with mock.patch("urllib.request.urlopen", return_value=response):
                with mock.patch("json.load", return_value=payload):
                    return fetch_company_news("AVGO", AS_OF)

    def test_a_good_payload_returns_records(self):
        result = self._fetch([entry()])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["source_id"], FINNHUB_SOURCE_ID)

    def test_a_rate_limit_is_named_separately(self):
        # 429 means STOP ASKING, not "try again". Conflating it with a transient
        # error is how a quota overage deepens.
        error = urllib.error.HTTPError(
            FINNHUB_NEWS_URL, 429, "Too Many Requests", {}, None
        )
        result = self._fetch(error=error)
        self.assertEqual(result["status"], "provider_request_failed")
        self.assertIn("429", result["reason"])
        self.assertIn("stop rather than retry", result["reason"])

    def test_another_http_error_is_reported_with_its_code(self):
        error = urllib.error.HTTPError(FINNHUB_NEWS_URL, 503, "Down", {}, None)
        result = self._fetch(error=error)
        self.assertEqual(result["status"], "provider_request_failed")
        self.assertIn("503", result["reason"])

    def test_a_network_failure_does_not_raise(self):
        result = self._fetch(error=OSError("connection reset"))
        self.assertEqual(result["status"], "provider_request_failed")
        self.assertEqual(result["records"], [])

    def test_an_error_object_is_not_read_as_zero_articles(self):
        # The endpoint returns a bare array; a dict is an error body, and
        # treating it as an empty list would report "no news" on a bad key.
        result = self._fetch({"error": "Invalid API key"})
        self.assertEqual(result["status"], "provider_request_failed")
        self.assertIn("Invalid API key", result["reason"])

    def test_records_are_newest_first_before_the_cap(self):
        # Truncating an arbitrary order would silently drop the most recent
        # articles, which are the ones that matter.
        payload = [
            entry(identifier=i, stamp=WHEN - i * 3600, headline=f"story {i}")
            for i in range(5)
        ]
        records = self._fetch(list(reversed(payload)))["records"]
        stamps = [r["published_time"] for r in records]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_the_payload_is_capped(self):
        payload = [
            entry(identifier=i, stamp=WHEN - i * 60, headline=f"story {i}")
            for i in range(FINNHUB_MAX_ARTICLES + 25)
        ]
        self.assertEqual(len(self._fetch(payload)["records"]), FINNHUB_MAX_ARTICLES)

    def test_the_key_is_sent_but_never_returned(self):
        # A credential echoed into a result could reach a log or an email body.
        result = self._fetch([entry()])
        self.assertNotIn("test-key", json.dumps(result, default=str))


class TheDispatcherTests(unittest.TestCase):
    """`fetch_news_records` walks the provider order and reports provenance."""

    def test_a_provider_with_no_key_is_skipped_not_failed(self):
        from core.news_adapter import fetch_news_records

        with mock.patch.dict(
            os.environ,
            {FINNHUB_API_KEY_ENV: "", NEWS_PROVIDER_API_KEY_ENV: ""},
        ):
            result = fetch_news_records("AVGO", AS_OF)
        # Nothing was ASKED, so this is "not configured", not "refused".
        self.assertEqual(result["status"], "provider_key_required")
        self.assertEqual(result["records"], [])

    def test_not_configured_is_distinguished_from_refused(self):
        """CAUGHT BY PROBE.

        The first version keyed the status on whether `attempts` was non-empty,
        but a SKIPPED provider appends an attempt too. A machine with no keys
        reported `provider_request_failed`, which reads as "the provider refused
        us" and sends a reader hunting an outage that never happened.
        """
        from core.news_adapter import fetch_news_records

        with mock.patch.dict(
            os.environ,
            {FINNHUB_API_KEY_ENV: "", NEWS_PROVIDER_API_KEY_ENV: ""},
        ):
            result = fetch_news_records("AVGO", AS_OF)
        self.assertNotEqual(result["status"], "provider_request_failed")
        self.assertIn("not configured", result["reason"])

    def test_finnhub_answers_first_when_configured(self):
        from core.news_adapter import fetch_news_records

        with mock.patch.dict(os.environ, {FINNHUB_API_KEY_ENV: "k"}):
            with mock.patch(
                "core.finnhub_news.fetch_company_news",
                return_value={
                    "status": "ok",
                    "records": [{"headline": "x"}],
                    "reason": "",
                },
            ) as finnhub:
                with mock.patch(
                    "core.news_adapter.fetch_provider_articles"
                ) as newsapi:
                    result = fetch_news_records("AVGO", AS_OF)
        finnhub.assert_called_once()
        newsapi.assert_not_called()
        self.assertEqual(result["source_id"], FINNHUB_SOURCE_ID)

    def test_newsapi_covers_for_a_finnhub_failure(self):
        # Finnhub is North America only, so the fallback is load-bearing rather
        # than decorative: without it a non-covered holding would read as having
        # no news instead of as unlooked-at.
        from core.news_adapter import fetch_news_records

        with mock.patch.dict(
            os.environ,
            {FINNHUB_API_KEY_ENV: "k", NEWS_PROVIDER_API_KEY_ENV: "n"},
        ):
            with mock.patch(
                "core.finnhub_news.fetch_company_news",
                return_value={
                    "status": "provider_request_failed",
                    "records": [],
                    "reason": "not covered",
                },
            ):
                with mock.patch(
                    "core.news_adapter.fetch_provider_articles",
                    return_value={
                        "status": "ok",
                        "records": [{"headline": "y"}],
                        "reason": "",
                    },
                ):
                    result = fetch_news_records("AVGO", AS_OF)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source_id"], NEWSAPI_SOURCE_ID)

    def test_every_failure_is_named_in_the_reason(self):
        # "News is unavailable" without saying which provider failed is
        # unactionable.
        from core.news_adapter import fetch_news_records

        with mock.patch.dict(
            os.environ,
            {FINNHUB_API_KEY_ENV: "k", NEWS_PROVIDER_API_KEY_ENV: "n"},
        ):
            with mock.patch(
                "core.finnhub_news.fetch_company_news",
                return_value={
                    "status": "provider_request_failed",
                    "records": [],
                    "reason": "finnhub down",
                },
            ):
                with mock.patch(
                    "core.news_adapter.fetch_provider_articles",
                    return_value={
                        "status": "provider_request_failed",
                        "records": [],
                        "reason": "newsapi down",
                    },
                ):
                    result = fetch_news_records("AVGO", AS_OF)
        self.assertIn("finnhub down", result["reason"])
        self.assertIn("newsapi down", result["reason"])


class TheConfigContractTests(unittest.TestCase):
    def test_the_two_providers_have_distinct_source_ids(self):
        # Otherwise the W6 ledger cannot say which one supplied a record.
        self.assertNotEqual(FINNHUB_SOURCE_ID, NEWSAPI_SOURCE_ID)

    def test_the_declared_newsapi_id_matches_the_adapter(self):
        # Config declares it rather than importing from the adapter, because
        # config must not depend on a module that imports config. This test is
        # what keeps the two from drifting.
        from core.news_adapter import NEWS_SOURCE_ID

        self.assertEqual(NEWSAPI_SOURCE_ID, NEWS_SOURCE_ID)

    def test_the_article_cap_matches_newsapi(self):
        # The aggregation and contradiction thresholds were measured against
        # that volume, so a per-provider cap changes what a decision rests on.
        self.assertEqual(FINNHUB_MAX_ARTICLES, NEWS_MAX_ARTICLES)

    def test_newsapi_remains_in_the_order_as_a_fallback(self):
        self.assertIn(NEWSAPI_SOURCE_ID, NEWS_PROVIDER_ORDER)

    def test_finnhub_leads_the_order(self):
        self.assertEqual(NEWS_PROVIDER_ORDER[0], FINNHUB_SOURCE_ID)

    def test_the_endpoint_is_https(self):
        # The key travels in the query string.
        self.assertTrue(FINNHUB_NEWS_URL.startswith("https://"))

    def test_the_endpoint_is_company_news_not_market_news(self):
        # market-news returns general headlines with no `related` symbols, which
        # is the field that makes entity resolution provider-side.
        self.assertIn("company-news", FINNHUB_NEWS_URL)


if __name__ == "__main__":
    unittest.main()
