from __future__ import annotations

import unittest

from core.score_engine import build_score


class ScoringEngineTests(unittest.TestCase):
    def test_score_output_shape(self):
        result = build_score("MSFT", "2024-01-02")
        self.assertEqual(result.ticker, "MSFT")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 10.0)
        self.assertIn("action", result.to_dict())
        self.assertIsInstance(result.risk_flags, list)


if __name__ == "__main__":
    unittest.main()
