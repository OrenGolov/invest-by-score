"""Sprint N2: sentiment placeholder — anti-proxying and contract pinning.

The N2 acceptance criterion is a contract test that pins the UNAVAILABLE
shape byte-for-byte, so a future provider cannot silently change the public
schema. The anti-proxying rule is enforced both structurally (the fetch
signature accepts no market/news inputs) and behaviorally (identical output
regardless of anything happening elsewhere in the pipeline).
"""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import config as core_config
from core.news_adapter import NEWS_SOURCE_ID
from core.orchestrator import _derive_agent_statuses, orchestrate_score
from core.score_engine import build_score
from core.sentiment_contract import SENTIMENT_SOURCE_ID, fetch_sentiment_snapshot

AS_OF = "2024-01-05 12:00:00"


class SentimentContractTests(unittest.TestCase):
    """N2 acceptance: the UNAVAILABLE shape is pinned byte-for-byte."""

    PINNED_UNAVAILABLE = {
        "ticker": "MSFT",
        "as_of": "2024-01-02",
        "status": "UNAVAILABLE",
        "source_id": "sentiment_provider_unconfigured",
        "source_confidence": 0.0,
        "published_time": None,
        "calculation_version": "sentiment-contract-v1",
        "lookback_period": "N/A",
        "sentiment_score": None,
        "derivation": "none",
        "intended_inputs": ["social_volume", "tone_trend", "disagreement", "manipulation_flags"],
        "reason": (
            "No legitimate social/positioning sentiment provider is connected. "
            "Sentiment is never inferred from RSI, price direction, technical "
            "indicators, or the news score (anti-proxying rule, Sprint N2); a "
            "news-derived design would have to be labeled derived_from_news with "
            "confidence scaled accordingly."
        ),
    }

    def test_unavailable_shape_is_pinned_byte_for_byte(self):
        self.assertEqual(fetch_sentiment_snapshot("MSFT", "2024-01-02"), self.PINNED_UNAVAILABLE)

    def test_signature_accepts_only_ticker_and_as_of(self):
        # Structural anti-proxying: there is no parameter through which price,
        # technical indicators, or the news snapshot could enter.
        self.assertEqual(
            list(inspect.signature(fetch_sentiment_snapshot).parameters),
            ["ticker", "as_of"],
        )

    def test_output_is_identical_regardless_of_ticker_or_repeat_calls(self):
        first = fetch_sentiment_snapshot("msft", AS_OF)
        second = fetch_sentiment_snapshot("MSFT", AS_OF)
        third = fetch_sentiment_snapshot("TSLA", AS_OF)
        self.assertEqual(first, second)
        self.assertEqual(first["ticker"], "MSFT")
        self.assertEqual(third["status"], "UNAVAILABLE")
        self.assertIsNone(third["sentiment_score"])

    def test_derivation_labels_the_placeholder_state(self):
        snapshot = fetch_sentiment_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["derivation"], "none")
        self.assertEqual(snapshot["source_id"], SENTIMENT_SOURCE_ID)
        self.assertEqual(snapshot["calculation_version"], core_config.SENTIMENT_CONTRACT_VERSION)


class SentimentEnsembleWiringTests(unittest.TestCase):
    """Zero weight, explicit status, renormalization unaffected."""

    def test_sentiment_weight_stays_zero_in_both_horizons(self):
        self.assertEqual(core_config.ENSEMBLE_WEIGHTS_CURRENT["sentiment"], 0.0)
        self.assertEqual(core_config.ENSEMBLE_WEIGHTS_LONG["sentiment"], 0.0)

    def test_build_score_carries_the_placeholder_and_zero_contribution(self):
        result = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(result.sentiment_snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(result.sentiment_snapshot["sentiment_score"])
        self.assertEqual(result.sentiment_snapshot["derivation"], "none")
        entry = result.ensemble_breakdown["agents"]["sentiment"]
        self.assertEqual(entry["status"], "UNAVAILABLE")
        self.assertEqual(entry["effective_weight_current"], 0.0)
        self.assertFalse(entry["eligible_current"])
        self.assertIn("anti-proxy", entry["note"])

    def test_replay_metadata_hashes_the_sentiment_contract(self):
        result = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertIn("sentiment_snapshot_hash", result.replay_metadata)
        self.assertIn(SENTIMENT_SOURCE_ID, result.replay_metadata["source_record_ids"])


class SentimentOrchestratorTests(unittest.TestCase):
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

    def test_sentiment_agent_is_in_the_roster(self):
        decision = orchestrate_score("MSFT", "2024-01-02")
        names = [agent.agent for agent in decision.agent_outputs]
        self.assertIn("sentiment", names)
        self.assertIn("macroeconomic", names)
        # N4: the regime agent is the ninth decision agent (docs/agents.md roster).
        self.assertIn("market_regime", names)
        self.assertEqual(len(names), 9)
        sentiment_agent = next(a for a in decision.agent_outputs if a.agent == "sentiment")
        self.assertEqual(sentiment_agent.status, "UNAVAILABLE")
        self.assertEqual(sentiment_agent.model_version, "sentiment-contract-v1")
        self.assertEqual(sentiment_agent.payload["derivation"], "none")
        self.assertEqual(
            sentiment_agent.payload["intended_inputs"],
            ["social_volume", "tone_trend", "disagreement", "manipulation_flags"],
        )
        self.assertTrue(sentiment_agent.input_hash)

    def test_sentiment_sits_outside_the_data_agent_posture_loop(self):
        score_stub = SimpleNamespace(
            confidence_breakdown={"factors": [{"name": "freshness", "value": 1.0}]},
            news_snapshot={"status": "OK"},
        )
        snapshot_stub = {"data_quality": {"score": 80.0}, "source_contract": {"timestamp_valid": True}}
        statuses = _derive_agent_statuses(
            snapshot_stub,
            {"point_in_time_valid": True, "source_status": "live_provider"},
            score_stub,
        )
        self.assertEqual(
            set(statuses),
            {"market_data", "technical_analysis", "fundamental_analysis", "news_intelligence",
             "macroeconomic", "market_regime"},
        )

    def test_sentiment_placeholder_does_not_change_a_healthy_news_run(self):
        # With news OK, the only remaining degraded agent is the sentiment
        # placeholder. Its zero weight and posture-loop exemption mean the
        # decision must behave exactly as the live agents dictate — the
        # placeholder neither lifts the mode toward PAPER nor adds vetoes.
        with patch("core.score_engine.fetch_news_snapshot", return_value=self._ok_news(0.5)):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertNotIn("agent_status_no_trade", decision.veto_reasons)
        sentiment_agent = next(a for a in decision.agent_outputs if a.agent == "sentiment")
        self.assertEqual(sentiment_agent.status, "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()