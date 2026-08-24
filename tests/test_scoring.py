from __future__ import annotations

import unittest

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
        self.assertIn("200d", payload["moving_averages"])
        self.assertIn("150d", payload["moving_averages"])
        self.assertIn("100d", payload["moving_averages"])
        self.assertIn("50d", payload["moving_averages"])
        self.assertIn("score", payload["data_quality"])
        self.assertIn("source", payload["source_metadata"])


if __name__ == "__main__":
    unittest.main()
