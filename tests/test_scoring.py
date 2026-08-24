import unittest

from core.score_engine import build_score


class ScoringEngineTests(unittest.TestCase):
    def test_score_output_shape(self):
        result = build_score("AAPL", "2026-08-21")
        self.assertIn("ticker", result)
        self.assertIn("score", result)
        self.assertIn("confidence", result)
        self.assertIn("explanation", result)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 10)
        self.assertEqual(result["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
