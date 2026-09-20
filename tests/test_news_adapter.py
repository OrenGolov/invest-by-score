"""Sprint N1: news adapter — ingestion, classification, contradiction, wiring.

Hermetic by construction: the provider fetch is patched (recorded-fixture
style), raw-store writes are mocked, and no test reads the wall clock. The
three N1 acceptance gates are covered explicitly:

- a positive headline from a zero-quality source cannot raise confidence;
- a contradictory cluster yields the explicit CONTRADICTORY status;
- as_of filtering is provable via a future-dated-article test.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from datetime import datetime
from unittest.mock import patch

from core import config as core_config
from core.news_adapter import (
    NEWS_SOURCE_ID,
    _normalize_provider_payload,
    aggregate_articles,
    build_news_snapshot,
    classify_event,
    detect_contradictions,
    fetch_provider_articles,
    lexicon_tone,
    pit_filter,
    resolve_news_provider,
    resolve_relevance,
    resolve_tone,
    source_weight,
)
from core.news_contract import fetch_news_snapshot
from core.orchestrator import orchestrate_score
from core.score_engine import _ensemble_blend, build_score

AS_OF = "2024-01-05 12:00:00"
KEY_ENV = {"NEWS_PROVIDER_API_KEY": "test-key"}


def _record(rid="u1", published="2024-01-05 10:00:00", headline="TEST beats earnings expectations",
            tone=None, quality=None, ticker=None, company_name=None, summary=""):
    return {
        "source_record_id": rid,
        "published_time": published,
        "headline": headline,
        "summary": summary,
        "url": f"https://example.com/{rid}",
        "source_name": "Example Wire",
        "ticker": ticker,
        "company_name": company_name,
        "source_quality": quality,
        "tone": tone,
    }


def _snapshot(records, ticker="TEST", as_of=AS_OF):
    with patch.dict(os.environ, KEY_ENV), \
            patch("core.news_adapter.append_raw_records", return_value=True), \
            patch("core.news_adapter.fetch_provider_articles",
                  return_value={"status": "ok", "records": records, "reason": ""}):
        return build_news_snapshot(ticker, as_of)


class NoKeyContractTests(unittest.TestCase):
    """The no-key path must stay byte-for-byte identical to the pre-N1 stub."""

    LEGACY_UNAVAILABLE = {
        "ticker": "MSFT",
        "as_of": "2024-01-02",
        "status": "UNAVAILABLE",
        "source_id": "news_provider_unconfigured",
        "source_confidence": 0.0,
        "published_time": None,
        "calculation_version": "news-contract-v1",
        "lookback_period": "N/A",
        "sentiment_score": None,
        "articles": [],
        "reason": "No verified news provider is connected. Sentiment is not inferred from price or technical indicators.",
    }

    def test_no_key_contract_is_byte_for_byte_the_legacy_stub(self):
        # The environment must be CLEARED, as the sibling test below does.
        # Without it this asserted the no-key contract while a real key was
        # present, so it passed only by accident of the machine it ran on —
        # and started failing the day NEWS_PROVIDER_API_KEY was configured.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                fetch_news_snapshot("MSFT", "2024-01-02"), self.LEGACY_UNAVAILABLE
            )
            self.assertNotIn("pipeline", fetch_news_snapshot("MSFT", "2024-01-02"))

    def test_provider_resolution_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            resolution = resolve_news_provider("MSFT", "2024-01-02")
        self.assertEqual(resolution["status"], "provider_key_required")
        self.assertEqual(resolution["source_confidence"], 0.0)

    def test_provider_resolution_with_key_is_live(self):
        with patch.dict(os.environ, KEY_ENV):
            resolution = resolve_news_provider("MSFT", "2024-01-02")
        self.assertEqual(resolution["status"], "live_provider")
        self.assertEqual(resolution["source_id"], NEWS_SOURCE_ID)

    def test_fetch_without_key_is_an_explicit_disposition(self):
        with patch.dict(os.environ, {}, clear=True):
            disposition = fetch_provider_articles("MSFT", datetime(2024, 1, 2, 12, 0, 0))
        self.assertEqual(disposition["status"], "provider_key_required")
        self.assertEqual(disposition["records"], [])

    def test_failed_request_degrades_to_unavailable_never_fabricates(self):
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.news_adapter.urllib.request.urlopen", side_effect=OSError("boom")):
            snapshot = build_news_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(snapshot["sentiment_score"])
        self.assertEqual(snapshot["source_id"], NEWS_SOURCE_ID)
        self.assertIn("unavailable", snapshot["reason"].lower())

    def test_empty_provider_response_is_explicitly_unavailable(self):
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.news_adapter.append_raw_records", return_value=True), \
                patch("core.news_adapter.fetch_provider_articles",
                      return_value={"status": "ok", "records": [], "reason": ""}):
            snapshot = build_news_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(snapshot["sentiment_score"])


class EventClassificationTests(unittest.TestCase):
    """N1 taxonomy: earnings, guidance, regulation, litigation, product launch,
    M&A, macro shock, strategic announcement, management commentary, other."""

    CASES = {
        "TEST tops quarterly earnings estimates": "earnings",
        "TEST raises full-year guidance": "guidance",
        "TEST sued by investors over disclosure": "litigation",
        "TEST faces regulator probe into practices": "regulation",
        "TEST launches new flagship device": "product_launch",
        "TEST slides as inflation accelerates": "macro_shock",
        "TEST announces acquisition of a rival": "m_and_a",
        "TEST signs strategic partnership overseas": "strategic_announcement",
        "TEST ceo comments on demand trends": "management_commentary",
        "TEST office on the river is scenic": "other",
    }

    def test_every_taxonomy_category_is_reachable(self):
        for headline, expected in self.CASES.items():
            with self.subTest(headline=headline):
                self.assertEqual(classify_event(headline), expected)

    def test_classification_is_deterministic(self):
        for headline in self.CASES:
            self.assertEqual(classify_event(headline), classify_event(headline))


class ToneTests(unittest.TestCase):
    def test_positive_negative_and_neutral(self):
        self.assertGreater(lexicon_tone("TEST profits surge"), 0.0)
        self.assertLess(lexicon_tone("TEST demand plunges"), 0.0)
        self.assertEqual(lexicon_tone("TEST opens a new office"), 0.0)

    def test_negation_flips_polarity(self):
        self.assertGreater(lexicon_tone("TEST does beat expectations"), 0.0)
        self.assertLess(lexicon_tone("TEST does not beat expectations"), 0.0)

    def test_tone_stays_in_bounds(self):
        for text in ("TEST beat beat beat record", "TEST miss miss fail plunge fraud", "TEST quiet day"):
            self.assertGreaterEqual(lexicon_tone(text), -1.0)
            self.assertLessEqual(lexicon_tone(text), 1.0)

    def test_provider_tone_is_honored_and_stamped(self):
        tone, derivation = resolve_tone(_record(tone=-0.42))
        self.assertEqual(tone, -0.42)
        self.assertEqual(derivation, "provider")

    def test_out_of_range_provider_tone_falls_back_to_lexicon(self):
        tone, derivation = resolve_tone(_record(headline="TEST profits surge", tone=7.0))
        self.assertGreater(tone, 0.0)
        self.assertTrue(derivation.startswith("lexicon:"))

    def test_missing_tone_uses_lexicon_with_version_stamp(self):
        tone, derivation = resolve_tone(_record(headline="TEST profits surge"))
        self.assertGreater(tone, 0.0)
        self.assertEqual(derivation, f"lexicon:{core_config.NEWS_TONE_LEXICON_VERSION}")


class EntityRelevanceTests(unittest.TestCase):
    def test_exact_ticker_token_match(self):
        self.assertEqual(resolve_relevance(_record(headline="Big win for TEST shareholders"), "TEST"), 1.0)

    def test_provider_matched_ticker(self):
        self.assertEqual(resolve_relevance(_record(headline="Good day", ticker="TEST"), "TEST"), 1.0)

    def test_company_name_match_scores_lower(self):
        relevance = resolve_relevance(
            _record(headline="Testing Corp said nothing today", company_name="Testing Corp"), "TST",
        )
        self.assertEqual(relevance, 0.7)

    def test_off_entity_article_has_zero_relevance(self):
        self.assertEqual(resolve_relevance(_record(headline="Unrelated markets round-up"), "TEST"), 0.0)


class PitFilterTests(unittest.TestCase):
    def test_future_dated_articles_are_rejected(self):
        as_of = datetime(2024, 1, 5, 12, 0, 0)
        eligible, rejected = pit_filter([_record(rid="f", published="2024-01-06 09:00:00")], as_of)
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]["reason"], "future_dated")

    def test_boundary_publication_exactly_at_as_of_is_eligible(self):
        as_of = datetime(2024, 1, 5, 12, 0, 0)
        eligible, rejected = pit_filter([_record(rid="b", published="2024-01-05 12:00:00")], as_of)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(rejected, [])

    def test_unparseable_publication_time_is_rejected(self):
        as_of = datetime(2024, 1, 5, 12, 0, 0)
        eligible, rejected = pit_filter([_record(rid="x", published="not-a-date")], as_of)
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]["reason"], "unparseable_published_time")


class SourceQualityTests(unittest.TestCase):
    def test_fresh_article_carries_full_base_confidence(self):
        weight = source_weight(_record(), datetime(2024, 1, 5, 12, 0, 0), datetime(2024, 1, 5, 12, 0, 0))
        self.assertEqual(weight, core_config.NEWS_BASE_SOURCE_CONFIDENCE)

    def test_recency_decay_follows_half_life(self):
        weight = source_weight(_record(), datetime(2024, 1, 2, 12, 0, 0), datetime(2024, 1, 5, 12, 0, 0))
        self.assertAlmostEqual(weight, core_config.NEWS_BASE_SOURCE_CONFIDENCE * 0.5, places=6)

    def test_zero_quality_source_has_zero_weight(self):
        weight = source_weight(
            _record(quality=0.0), datetime(2024, 1, 5, 12, 0, 0), datetime(2024, 1, 5, 12, 0, 0),
        )
        self.assertEqual(weight, 0.0)

    def test_malformed_quality_fails_closed_to_zero(self):
        weight = source_weight(
            _record(quality="high"), datetime(2024, 1, 5, 12, 0, 0), datetime(2024, 1, 5, 12, 0, 0),
        )
        self.assertEqual(weight, 0.0)


class ContradictionTests(unittest.TestCase):
    @staticmethod
    def _credible(rid, tone, category="earnings", day="2024-01-05", weight=0.75):
        return {
            "source_record_id": rid,
            "category": category,
            "tone": tone,
            "cluster_date": day,
            "source_weight": weight,
            "relevance": 1.0,
        }

    def test_opposite_sign_cluster_beyond_tolerance_is_contradictory(self):
        contradictions = detect_contradictions([self._credible("a", 0.9), self._credible("b", -0.9)])
        self.assertEqual(len(contradictions), 1)
        cluster = contradictions[0]
        self.assertEqual(cluster["positive"]["source_record_ids"], ["a"])
        self.assertEqual(cluster["negative"]["source_record_ids"], ["b"])

    def test_small_disagreement_is_not_contradictory(self):
        self.assertEqual(detect_contradictions([self._credible("a", 0.5), self._credible("b", -0.05)]), [])

    def test_same_sign_cluster_is_not_contradictory(self):
        self.assertEqual(detect_contradictions([self._credible("a", 0.9), self._credible("b", 0.1)]), [])

    def test_different_days_or_categories_do_not_cluster(self):
        self.assertEqual(detect_contradictions([
            self._credible("a", 0.9, day="2024-01-04"),
            self._credible("b", -0.9, day="2024-01-05"),
        ]), [])
        self.assertEqual(detect_contradictions([
            self._credible("a", 0.9, category="earnings"),
            self._credible("b", -0.9, category="guidance"),
        ]), [])


class AggregationTests(unittest.TestCase):
    @staticmethod
    def _article(rid, tone, weight=0.75, relevance=1.0):
        return {"source_record_id": rid, "tone": tone, "source_weight": weight, "relevance": relevance}

    def test_weighted_mean_respects_source_weight(self):
        sentiment, _ = aggregate_articles([
            self._article("a", 1.0, weight=0.8),
            self._article("b", -1.0, weight=0.2),
        ])
        self.assertEqual(sentiment, 0.6)

    def test_confidence_capped_by_weakest_constituent(self):
        articles = [
            self._article("a", 1.0, weight=0.9),
            self._article("b", 1.0, weight=0.9),
            self._article("c", 1.0, weight=0.3),
        ]
        _, confidence = aggregate_articles(articles)
        self.assertEqual(confidence, 0.3)

    def test_empty_input_aggregates_to_nothing(self):
        self.assertEqual(aggregate_articles([]), (None, 0.0))


class BuildSnapshotTests(unittest.TestCase):
    """Full pipeline behavior through the patched provider seam."""

    def test_ok_snapshot_aggregates_credible_articles(self):
        snapshot = _snapshot([
            _record(rid="u2", published="2024-01-04 10:00:00", headline="TEST launches new chip", tone=0.4),
            _record(rid="u1", published="2024-01-05 10:00:00", headline="TEST beats earnings expectations", tone=0.8),
        ])
        self.assertEqual(snapshot["status"], "OK")
        self.assertGreater(snapshot["sentiment_score"], 0.0)
        self.assertLessEqual(snapshot["sentiment_score"], 1.0)
        self.assertGreater(snapshot["source_confidence"], 0.0)
        self.assertEqual(snapshot["published_time"], "2024-01-05 10:00:00")
        self.assertEqual(snapshot["lookback_period"], "7d")
        self.assertEqual(snapshot["pipeline"]["counts"]["categories"], {"earnings": 1, "product_launch": 1})
        self.assertEqual(snapshot["pipeline"]["pipeline_version"], core_config.NEWS_PIPELINE_VERSION)
        for article in snapshot["articles"]:
            for key in ("source_id", "source_record_id", "published_time", "headline", "url",
                        "category", "tone", "relevance", "source_weight", "tone_derivation"):
                self.assertIn(key, article)

    def test_positive_headline_from_zero_quality_source_cannot_raise_confidence(self):
        clean = _snapshot([_record(rid="ok1", headline="TEST record profits", tone=1.0)])
        polluted = _snapshot([
            _record(rid="ok1", headline="TEST record profits", tone=1.0),
            _record(rid="zero", headline="TEST beats everything dramatically", tone=1.0, quality=0.0),
        ])
        self.assertEqual(clean["status"], "OK")
        self.assertEqual(polluted["status"], "OK")
        self.assertEqual(clean["sentiment_score"], polluted["sentiment_score"])
        self.assertEqual(clean["source_confidence"], polluted["source_confidence"])
        zero_article = next(a for a in polluted["articles"] if a["source_record_id"] == "zero")
        self.assertFalse(zero_article["included_in_aggregation"])
        self.assertEqual(zero_article["exclusion_reason"], "zero_source_weight")

    def test_only_zero_quality_articles_yield_incomplete(self):
        snapshot = _snapshot([_record(rid="z", headline="TEST record profits", tone=1.0, quality=0.0)])
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertIsNone(snapshot["sentiment_score"])
        self.assertEqual(snapshot["source_confidence"], 0.0)

    def test_contradictory_cluster_yields_explicit_status_not_a_neutral_average(self):
        snapshot = _snapshot([
            _record(rid="pos", published="2024-01-05 10:00:00", headline="TEST beats earnings estimates", tone=0.9),
            _record(rid="neg", published="2024-01-05 11:00:00", headline="TEST misses earnings estimates", tone=-0.9),
        ])
        self.assertEqual(snapshot["status"], "CONTRADICTORY")
        self.assertIsNone(snapshot["sentiment_score"])
        self.assertEqual(snapshot["source_confidence"], core_config.NEWS_CONTRADICTION_CONFIDENCE_FLOOR)
        cluster = snapshot["pipeline"]["contradictions"][0]
        self.assertEqual(cluster["positive"]["source_record_ids"], ["pos"])
        self.assertEqual(cluster["negative"]["source_record_ids"], ["neg"])

    def test_zero_quality_headline_cannot_drive_a_contradiction(self):
        snapshot = _snapshot([
            _record(rid="pos", published="2024-01-05 10:00:00", headline="TEST beats earnings estimates", tone=0.9),
            _record(rid="neg", published="2024-01-05 11:00:00", headline="TEST misses earnings estimates", tone=-0.9, quality=0.0),
        ])
        self.assertEqual(snapshot["status"], "OK")

    def test_future_dated_article_invalidates_the_payload(self):
        snapshot = _snapshot([
            _record(rid="future", published="2024-01-06 09:00:00", headline="TEST beats earnings estimates"),
            _record(rid="ok", published="2024-01-05 09:00:00", headline="TEST record profits", tone=1.0),
        ])
        self.assertEqual(snapshot["status"], "INVALID")
        self.assertIsNone(snapshot["sentiment_score"])
        self.assertEqual(snapshot["source_confidence"], 0.0)
        rejected = snapshot["pipeline"]["rejected"]
        self.assertEqual([entry["reason"] for entry in rejected], ["future_dated"])
        self.assertNotIn("future", {a["source_record_id"] for a in snapshot["articles"]})

    def test_off_entity_articles_are_kept_as_evidence_but_not_aggregated(self):
        snapshot = _snapshot([
            _record(rid="on", headline="TEST record profits", tone=1.0),
            _record(rid="off", headline="Generic markets round-up today", tone=1.0),
        ])
        self.assertEqual(snapshot["status"], "OK")
        by_id = {a["source_record_id"]: a for a in snapshot["articles"]}
        self.assertFalse(by_id["off"]["included_in_aggregation"])
        self.assertEqual(by_id["off"]["exclusion_reason"], "zero_relevance")

    def test_duplicate_headlines_are_deduplicated_by_novelty(self):
        snapshot = _snapshot([
            _record(rid="first", published="2024-01-05 09:00:00", headline="TEST record profits", tone=1.0),
            _record(rid="second", published="2024-01-05 10:00:00", headline="TEST Record Profits!", tone=-1.0),
        ])
        self.assertEqual(snapshot["status"], "OK")
        by_id = {a["source_record_id"]: a for a in snapshot["articles"]}
        self.assertEqual(by_id["second"]["exclusion_reason"], "duplicate_headline")
        # The duplicate's negative tone never pulls the aggregate toward zero.
        self.assertEqual(snapshot["sentiment_score"], 1.0)


class EnsembleWiringTests(unittest.TestCase):
    """Born-wired rule: the news agent enters the ensemble with real weight."""

    @staticmethod
    def _contributions(news):
        return {
            "market_data": {"score_current": 10.0, "score_long": 10.0, "status": "OK", "note": ""},
            "technical_analysis": {"score_current": 8.0, "score_long": 6.0, "status": "OK", "note": ""},
            "fundamental_analysis": {"score_current": 5.0, "score_long": 5.0, "status": "OK", "note": ""},
            "news_intelligence": news,
            "sentiment": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "macroeconomic": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "market_regime": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        }

    def test_news_ok_gets_its_dedicated_weight(self):
        current, long_term, breakdown = _ensemble_blend(
            self._contributions({"score_current": 9.0, "score_long": None, "status": "OK", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        self.assertAlmostEqual(current, (0.70 * 8.0 + 0.10 * 5.0 + 0.10 * 9.0) / 0.90, places=2)
        self.assertAlmostEqual(long_term, (0.70 * 6.0 + 0.20 * 5.0) / 0.90, places=2)
        news = breakdown["agents"]["news_intelligence"]
        self.assertAlmostEqual(news["effective_weight_current"], 0.10 / 0.90, places=6)
        self.assertTrue(news["eligible_current"])
        self.assertFalse(news["eligible_long"])

    def test_unavailable_news_renormalizes_instead_of_counting_neutral(self):
        current, _, breakdown = _ensemble_blend(
            self._contributions({"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        self.assertAlmostEqual(current, (0.70 * 8.0 + 0.10 * 5.0) / 0.80, places=2)
        news = breakdown["agents"]["news_intelligence"]
        self.assertEqual(news["effective_weight_current"], 0.0)
        self.assertFalse(news["eligible_current"])

    def test_contradictory_news_is_ineligible_and_keeps_its_status(self):
        _, _, breakdown = _ensemble_blend(
            self._contributions({"score_current": None, "score_long": None, "status": "CONTRADICTORY", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        entry = breakdown["agents"]["news_intelligence"]
        self.assertEqual(entry["status"], "CONTRADICTORY")
        self.assertFalse(entry["eligible_current"])


class ScoreEngineIntegrationTests(unittest.TestCase):
    """The news contract flows into build_score through its own ensemble line."""

    @staticmethod
    def _ok_news(sentiment):
        return {
            "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": "OK",
            "source_id": NEWS_SOURCE_ID, "source_confidence": 0.7,
            "published_time": "2024-01-02 09:00:00",
            "calculation_version": "news-contract-v1", "lookback_period": "7d",
            "sentiment_score": sentiment, "articles": [], "reason": "",
            "pipeline": {"pipeline_version": core_config.NEWS_PIPELINE_VERSION},
        }

    def test_news_line_moves_the_ensemble_not_the_technical_view(self):
        with patch("core.score_engine.fetch_news_snapshot", return_value=self._ok_news(0.8)):
            with_news = build_score("MSFT", "2024-01-02", persist_audit=False)
        baseline = build_score("MSFT", "2024-01-02", persist_audit=False)
        news_line = with_news.ensemble_breakdown["agents"]["news_intelligence"]
        self.assertEqual(news_line["score_current"], 9.0)
        self.assertAlmostEqual(news_line["effective_weight_current"], 0.10 / 0.90, places=6)
        self.assertEqual(
            baseline.ensemble_breakdown["agents"]["news_intelligence"]["effective_weight_current"], 0.0,
        )
        # Anti-double-count: the technical view is identical with and without news.
        self.assertEqual(
            with_news.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
            baseline.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
        )

    def test_contradictory_news_forces_no_trade_through_the_veto_path(self):
        contradictory = self._ok_news(0.0)
        contradictory["status"] = "CONTRADICTORY"
        contradictory["sentiment_score"] = None
        contradictory["source_confidence"] = core_config.NEWS_CONTRADICTION_CONFIDENCE_FLOOR
        with patch("core.score_engine.fetch_news_snapshot", return_value=contradictory):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(decision.mode, "NO_TRADE")
        self.assertIn("agent_status_no_trade", decision.veto_reasons)
        news_agent = next(a for a in decision.agent_outputs if a.agent == "news_intelligence")
        self.assertEqual(news_agent.status, "CONTRADICTORY")
        self.assertEqual(news_agent.confidence, core_config.NEWS_CONTRADICTION_CONFIDENCE_FLOOR)

    def test_default_run_keeps_news_unavailable_and_ineligible(self):
        result = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(result.news_snapshot["status"], "UNAVAILABLE")
        entry = result.ensemble_breakdown["agents"]["news_intelligence"]
        self.assertFalse(entry["eligible_current"])
        self.assertIsNone(entry["score_current"])

    def test_news_determinism(self):
        with patch("core.score_engine.fetch_news_snapshot", return_value=self._ok_news(0.5)):
            first = build_score("MSFT", "2024-01-02", persist_audit=False)
            second = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(first.replay_metadata["replay_hash"], second.replay_metadata["replay_hash"])


class RawLedgerTests(unittest.TestCase):
    def test_provider_fetch_appends_raw_records_before_use(self):
        records = [_record(rid="r1")]
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.news_adapter.append_raw_records") as append_mock, \
                patch("core.news_adapter.fetch_provider_articles",
                      return_value={"status": "ok", "records": records, "reason": ""}):
            build_news_snapshot("TEST", AS_OF)
        append_mock.assert_called_once()
        self.assertEqual(append_mock.call_args.kwargs["source_id"], NEWS_SOURCE_ID)
        self.assertEqual(append_mock.call_args.kwargs["request_key"], "TEST_2024-01-05")
        self.assertEqual(append_mock.call_args.kwargs["records"], records)

    def test_no_key_path_never_touches_the_raw_ledger(self):
        with patch("core.news_adapter.append_raw_records") as append_mock:
            build_news_snapshot("MSFT", "2024-01-02")
        append_mock.assert_not_called()


class ProviderNormalizationTests(unittest.TestCase):
    @staticmethod
    def _fake_response(payload):
        buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return buffer.read()

        return FakeResponse()

    def test_newsapi_shape_is_normalized_and_malformed_rows_skipped(self):
        payload = {
            "status": "ok",
            "articles": [
                {"source": {"name": "Example"}, "title": "TEST beats earnings",
                 "url": "https://example.com/a", "publishedAt": "2024-01-05T10:00:00Z",
                 "description": "Strong quarter"},
                {"no": "headline"},
                "not-a-dict",
            ],
        }
        records = _normalize_provider_payload(payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_record_id"], "https://example.com/a")
        self.assertEqual(records[0]["published_time"], "2024-01-05T10:00:00Z")
        self.assertEqual(records[0]["source_name"], "Example")
        self.assertIsNone(records[0]["tone"])

    def test_fetch_parses_provider_response(self):
        payload = {"status": "ok", "articles": [
            {"source": {"name": "W"}, "title": "TEST wins", "url": "u", "publishedAt": "2024-01-05T10:00:00Z"},
        ]}
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.news_adapter.urllib.request.urlopen", return_value=self._fake_response(payload)):
            disposition = fetch_provider_articles("TEST", datetime(2024, 1, 5, 12, 0, 0))
        self.assertEqual(disposition["status"], "ok")
        self.assertEqual(len(disposition["records"]), 1)

    def test_fetch_rejects_error_payload(self):
        payload = {"status": "error", "message": "rate limited"}
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.news_adapter.urllib.request.urlopen", return_value=self._fake_response(payload)):
            disposition = fetch_provider_articles("TEST", datetime(2024, 1, 5, 12, 0, 0))
        self.assertEqual(disposition["status"], "provider_request_failed")


class RegistryTests(unittest.TestCase):
    def test_news_registry_entry_exists_and_is_not_healthy_until_keyed(self):
        from fetch_data import SOURCE_REGISTRY, get_provider_health_matrix

        entry = SOURCE_REGISTRY["news"]
        self.assertEqual(entry["status"], "provider_key_required")
        self.assertEqual(entry["source_id"], NEWS_SOURCE_ID)
        health = get_provider_health_matrix()["news"]
        self.assertEqual(health["status"], "provider_key_required")
        self.assertEqual(health["health_score"], 0.0)


if __name__ == "__main__":
    unittest.main()