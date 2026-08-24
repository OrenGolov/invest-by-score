from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from agents.market_data_agent import fetch_market_snapshot
from core.agent_contracts import AgentContract, OrchestrationDecision
from core.audit_store import get_audit_events, get_decision_by_replay_hash, persist_decision_audit
from core.orchestrator import orchestrate_score
from core.score_engine import build_score
from fetch_data import fetch_fundamental_snapshot, get_provider_health_matrix, resolve_fundamental_provider


class ScoringEngineTests(unittest.TestCase):
    def test_score_output_shape(self):
        result = build_score("MSFT", "2024-01-02")
        payload = result.to_dict()
        self.assertEqual(result.ticker, "MSFT")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 10.0)
        self.assertIn("action", payload)
        self.assertIsInstance(result.risk_flags, list)
        self.assertIn("moving_averages", payload)
        self.assertIn("rsi", payload)
        self.assertIn("volatility", payload)
        self.assertIn("market_context", payload)
        self.assertIn("data_quality", payload)
        self.assertIn("source_metadata", payload)
        self.assertIn("recommended_actions", payload)
        self.assertIn("latest_financial_report", payload)
        self.assertIn("next_expected_report", payload)
        self.assertIn("insights", payload)
        self.assertIn("scoring_breakdown", payload)
        self.assertIn("source_reliability", payload)
        self.assertIn("technical_features", payload)
        self.assertIn("feature_metadata", payload)
        self.assertIn("governance", payload)
        self.assertIn("risk_gate_passed", payload["governance"])
        self.assertIn("evidence_status", payload["governance"])
        self.assertIn("gate_reasons", payload["governance"])
        self.assertIn("evidence_ledger", payload)
        self.assertIn("status", payload["evidence_ledger"])
        self.assertIn("fundamental_score", payload)
        self.assertIn("fundamental_features", payload)
        self.assertIn("valuation_quality", payload["fundamental_features"])
        self.assertIn("current_time_score", payload)
        self.assertIn("long_term_score", payload)
        self.assertGreaterEqual(payload["current_time_score"], 0.0)
        self.assertGreaterEqual(payload["long_term_score"], 0.0)
        self.assertIn("200d", payload["moving_averages"])
        self.assertIn("150d", payload["moving_averages"])
        self.assertIn("100d", payload["moving_averages"])
        self.assertIn("50d", payload["moving_averages"])
        self.assertIn("score", payload["data_quality"])
        self.assertIn("feature_family", payload["feature_metadata"])
        self.assertIn("trend_regime", payload["technical_features"])
        self.assertIn("source", payload["source_metadata"])
        self.assertIn("primary", payload["recommended_actions"])
        self.assertIn("publication_date", payload["next_expected_report"])
        self.assertIn("bullish_signals", payload["insights"])
        self.assertIn("weighted_contributions", payload["scoring_breakdown"])

    def test_market_snapshot_reports_point_in_time_and_quality(self):
        snapshot = fetch_market_snapshot("MSFT", "2024-01-02")
        self.assertEqual(snapshot["as_of"], "2024-01-02 00:00:00")
        self.assertIn("point_in_time_cutoff", snapshot)
        self.assertIn("future_bars_excluded", snapshot)
        self.assertIn("data_quality", snapshot)
        self.assertIn("score", snapshot["data_quality"])
        self.assertIsInstance(snapshot["data_quality"]["flags"], list)
        self.assertIsInstance(snapshot["future_bars_excluded"], int)

    def test_fundamental_snapshot_uses_real_feed_and_respects_as_of(self):
        snapshot = fetch_fundamental_snapshot("MSFT", "2024-01-02")
        self.assertIn("source", snapshot)
        self.assertIn("source_type", snapshot)
        self.assertIn("latest_close", snapshot)
        self.assertIn("valuation_metrics", snapshot)
        self.assertIn("point_in_time_valid", snapshot)
        self.assertIsInstance(snapshot["point_in_time_valid"], bool)
        self.assertLessEqual(pd.Timestamp(snapshot["as_of"]), pd.Timestamp("2024-01-02 23:59:59"))
        self.assertIn("as_of", snapshot)
        self.assertIn("source_status", snapshot)
        self.assertTrue(snapshot["point_in_time_valid"])

    def test_sprint5_orchestration_contracts_are_typed_and_replayable(self):
        result = orchestrate_score("MSFT", "2024-01-02")
        self.assertIsInstance(result, OrchestrationDecision)
        self.assertEqual(result.ticker, "MSFT")
        self.assertEqual(result.as_of, "2024-01-02 00:00:00")
        self.assertIn("analysis_only", result.mode.lower())
        self.assertGreaterEqual(len(result.agent_outputs), 3)
        self.assertTrue(all(isinstance(agent, AgentContract) for agent in result.agent_outputs))
        self.assertTrue(all(agent.input_hash for agent in result.agent_outputs))
        self.assertIn("market_data", result.agent_outputs[0].agent)
        self.assertIn("status", result.agent_outputs[0].payload)
        replay = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(result.to_dict(), replay.to_dict())

    def test_future_as_of_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch_market_snapshot("MSFT", "2099-01-01")

    def test_no_market_data_is_a_validation_error(self):
        with self.assertRaises(ValueError):
            fetch_market_snapshot("ZZZZ", "2024-01-02")

    def test_future_dated_fundamental_payload_is_rejected(self):
        future_payload = {
            "LastDivDate": "2099-01-01",
            "PERatio": "30",
            "PriceToBookRatio": "4.5",
            "MarketCapitalization": "1000000000",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return b'{"LastDivDate": "2099-01-01", "PERatio": "30", "PriceToBookRatio": "4.5", "MarketCapitalization": "1000000000"}'

        with patch("fetch_data.urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(ValueError):
                fetch_fundamental_snapshot("MSFT", "2024-01-02")

    def test_low_quality_data_forces_analysis_only(self):
        result = orchestrate_score("MSFT", "2024-01-02")
        self.assertIn("analysis_only", result.mode.lower())
        self.assertIn("analysis_only", result.action.lower())

    def test_fundamental_provider_resolution_has_explicit_fallbacks(self):
        with patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": ""}, clear=False):
            provider = resolve_fundamental_provider("MSFT", "2024-01-02")
            self.assertIn("provider", provider)
            self.assertIn("status", provider)
            self.assertEqual(provider["status"], "provider_key_required")

    def test_decision_audit_persistence_writes_event_record(self):
        result = build_score("MSFT", "2024-01-02")
        event = persist_decision_audit({
            "ticker": result.ticker,
            "as_of": result.as_of,
            "mode": "ANALYSIS_ONLY",
            "action": result.action,
            "score": result.score,
            "confidence": result.confidence,
            "replay_hash": "abc123",
            "source_quality": {
                "market_confidence": 0.8,
                "fundamental_confidence": 0.55,
                "effective_confidence": 0.68,
            },
        })
        self.assertIn("event_id", event)
        self.assertIn("ticker", event)
        self.assertEqual(event["ticker"], "MSFT")
        self.assertIn("replay_hash", event)

    def test_score_contract_includes_richer_source_quality_and_replay_metadata(self):
        result = build_score("MSFT", "2024-01-02")
        self.assertIn("source_quality", result.to_dict())
        self.assertIn("replay_metadata", result.to_dict())
        self.assertIn("effective_confidence", result.source_quality)
        self.assertIn("replay_hash", result.replay_metadata)

    def test_provider_health_matrix_tracks_quality_and_status(self):
        health = get_provider_health_matrix()
        self.assertIn("market_data", health)
        self.assertIn("fundamentals", health)
        self.assertIn("health_score", health["market_data"])
        self.assertIn("status", health["fundamentals"])

    def test_audit_retrieval_and_replay_lookup_work(self):
        event = persist_decision_audit({
            "ticker": "MSFT",
            "as_of": "2024-01-02 00:00:00",
            "mode": "ANALYSIS_ONLY",
            "action": "ANALYSIS_ONLY",
            "score": 6.5,
            "confidence": 0.75,
            "replay_hash": "replay-lookup-test",
            "source_quality": {"effective_confidence": 0.69},
        })
        self.assertIn("event_id", event)
        self.assertTrue(get_audit_events(limit=5))
        self.assertIn("replay_hash", get_decision_by_replay_hash("replay-lookup-test")[0])


if __name__ == "__main__":
    unittest.main()
