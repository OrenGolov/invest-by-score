from __future__ import annotations

import unittest

from agents.market_data_agent import fetch_market_snapshot
from core.score_engine import build_score


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


if __name__ == "__main__":
    unittest.main()
