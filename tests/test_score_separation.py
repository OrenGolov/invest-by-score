from __future__ import annotations

import unittest

from core.config import MAX_SCORE, MIN_SCORE
from core.news_contract import fetch_news_snapshot
from core.schemas import MarketSnapshot
from core.score_engine import _score_current_time, _score_long_term, build_score

NEUTRAL_NEWS = fetch_news_snapshot("TEST", "2024-01-02 00:00:00")


def _base_snapshot(**overrides) -> dict:
    snapshot = {
        "ticker": "TEST",
        "as_of": "2024-01-02 00:00:00",
        "close": 100.0,
        "volume": 1_000_000.0,
        "avg_volume_20d": 1_000_000.0,
        "volume_ratio_20d": 1.0,
        "change_1d": 0.0,
        "change_5d": 0.0,
        "change_20d": 0.0,
        "change_60d": 0.0,
        "trend_vs_20d_mean": 0.0,
        "rsi": 50.0,
        "volatility": 0.01,
        "price_vs_ma_50": 0.0,
        "price_vs_ma_100": 0.0,
        "price_vs_ma_150": 0.0,
        "price_vs_ma_200": 0.0,
        "moving_averages": {"50d": 100.0, "100d": 100.0, "150d": 100.0, "200d": 100.0},
    }
    snapshot.update(overrides)
    return snapshot


BULLISH = _base_snapshot(
    change_1d=0.005, change_5d=0.01, change_20d=0.02, trend_vs_20d_mean=0.015,
    rsi=60.0, volume_ratio_20d=1.1, price_vs_ma_50=0.01, price_vs_ma_100=0.008,
    change_60d=0.03, price_vs_ma_150=0.02, price_vs_ma_200=0.03, volatility=0.005,
    moving_averages={"50d": 101.0, "100d": 100.0, "150d": 103.0, "200d": 100.0},
)
BEARISH = _base_snapshot(
    change_1d=-0.005, change_5d=-0.01, change_20d=-0.02, trend_vs_20d_mean=-0.015,
    rsi=40.0, volume_ratio_20d=0.9, price_vs_ma_50=-0.01, price_vs_ma_100=-0.008,
    change_60d=-0.03, price_vs_ma_150=-0.02, price_vs_ma_200=-0.03, volatility=0.02,
    moving_averages={"50d": 99.0, "100d": 100.0, "150d": 97.0, "200d": 100.0},
)
NEUTRAL = _base_snapshot()


class FeatureGroupSeparationTests(unittest.TestCase):
    """Current-time and long-term scores must be computed from disjoint
    feature groups, so neither can silently become an alias of the other.
    """

    def test_current_time_score_ignores_long_term_only_fields(self):
        mutated = _base_snapshot(
            change_60d=0.5,
            price_vs_ma_150=0.3,
            price_vs_ma_200=0.3,
            volatility=0.5,
            moving_averages={"50d": 100.0, "100d": 100.0, "150d": 140.0, "200d": 70.0},
        )
        self.assertEqual(
            _score_current_time(_base_snapshot(), NEUTRAL_NEWS),
            _score_current_time(mutated, NEUTRAL_NEWS),
        )

    def test_long_term_score_ignores_current_only_fields(self):
        mutated = _base_snapshot(
            change_1d=0.05, change_5d=0.05, change_20d=0.05, trend_vs_20d_mean=0.05,
            rsi=90.0, volume_ratio_20d=3.0,
            price_vs_ma_50=0.3, price_vs_ma_100=0.3,
            moving_averages={"50d": 140.0, "100d": 140.0, "150d": 100.0, "200d": 100.0},
        )
        self.assertEqual(_score_long_term(_base_snapshot()), _score_long_term(mutated))

    def test_scores_are_never_aliases_across_regimes(self):
        for label, snapshot in (("bullish", BULLISH), ("bearish", BEARISH), ("neutral", NEUTRAL)):
            current = _score_current_time(snapshot, NEUTRAL_NEWS)
            long_term = _score_long_term(snapshot)
            self.assertNotEqual(current, long_term, msg=f"{label} case collapsed current/long-term into an alias")

    def test_scores_are_not_a_fixed_offset_of_each_other(self):
        diffs = set()
        for snapshot in (BULLISH, BEARISH, NEUTRAL):
            current = _score_current_time(snapshot, NEUTRAL_NEWS)
            long_term = _score_long_term(snapshot)
            diffs.add(round(current - long_term, 4))
        self.assertGreater(len(diffs), 1, "current - long_term was constant across regimes; scores may be a linear alias")

    def test_bullish_case_pushes_both_scores_above_neutral_base(self):
        current = _score_current_time(BULLISH, NEUTRAL_NEWS)
        long_term = _score_long_term(BULLISH)
        self.assertGreater(current, 4.0)
        self.assertGreater(long_term, 4.0)
        self.assertNotEqual(current, long_term)

    def test_bearish_case_pushes_both_scores_below_neutral_base(self):
        current = _score_current_time(BEARISH, NEUTRAL_NEWS)
        long_term = _score_long_term(BEARISH)
        self.assertLess(current, 4.0)
        self.assertLess(long_term, 4.0)
        self.assertNotEqual(current, long_term)

    def test_neutral_case_still_diverges_due_to_disjoint_risk_treatment(self):
        current = _score_current_time(NEUTRAL, NEUTRAL_NEWS)
        long_term = _score_long_term(NEUTRAL)
        self.assertNotEqual(current, long_term)

    def test_scores_diverge_even_when_current_time_is_fully_clamped(self):
        extreme_current_only = _base_snapshot(
            change_1d=5.0, change_5d=5.0, change_20d=5.0, trend_vs_20d_mean=5.0,
            rsi=100.0, volume_ratio_20d=10.0, price_vs_ma_50=5.0, price_vs_ma_100=5.0,
            moving_averages={"50d": 600.0, "100d": 100.0, "150d": 100.0, "200d": 100.0},
        )
        current = _score_current_time(extreme_current_only, NEUTRAL_NEWS)
        long_term = _score_long_term(extreme_current_only)
        self.assertEqual(current, MAX_SCORE)
        self.assertLess(long_term, MAX_SCORE - 1.0)
        self.assertNotEqual(current, long_term)

    def test_scores_diverge_even_when_long_term_is_fully_clamped(self):
        extreme_long_term_only = _base_snapshot(
            change_60d=5.0, price_vs_ma_150=5.0, price_vs_ma_200=5.0, volatility=0.0,
            moving_averages={"50d": 100.0, "100d": 100.0, "150d": 600.0, "200d": 100.0},
        )
        current = _score_current_time(extreme_long_term_only, NEUTRAL_NEWS)
        long_term = _score_long_term(extreme_long_term_only)
        self.assertEqual(long_term, MAX_SCORE)
        self.assertLess(current, MAX_SCORE - 1.0)
        self.assertNotEqual(current, long_term)

    def test_scores_stay_in_bounds(self):
        for snapshot in (BULLISH, BEARISH, NEUTRAL):
            current = _score_current_time(snapshot, NEUTRAL_NEWS)
            long_term = _score_long_term(snapshot)
            for value in (current, long_term):
                self.assertGreaterEqual(value, MIN_SCORE)
                self.assertLessEqual(value, MAX_SCORE)


class DeterminismTests(unittest.TestCase):
    def test_same_ticker_and_timestamp_is_deterministic(self):
        first = build_score("MSFT", "2024-01-02")
        second = build_score("MSFT", "2024-01-02")
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.current_time_score, second.current_time_score)
        self.assertEqual(first.long_term_score, second.long_term_score)
        self.assertEqual(first.replay_metadata["replay_hash"], second.replay_metadata["replay_hash"])
        self.assertEqual(first.replay_metadata["snapshot_hash"], second.replay_metadata["snapshot_hash"])
        self.assertEqual(first.replay_metadata["news_snapshot_hash"], second.replay_metadata["news_snapshot_hash"])


class NewsContractTests(unittest.TestCase):
    def test_news_contract_is_an_explicit_unavailable_placeholder(self):
        snapshot = fetch_news_snapshot("MSFT", "2024-01-02")
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(snapshot["sentiment_score"])
        self.assertEqual(snapshot["articles"], [])
        self.assertEqual(snapshot["source_confidence"], 0.0)
        self.assertIn("source_id", snapshot)
        self.assertIn("calculation_version", snapshot)

    def test_news_contract_is_identical_regardless_of_price_indicators(self):
        first = fetch_news_snapshot("MSFT", "2024-01-02")
        second = fetch_news_snapshot("MSFT", "2024-01-02")
        self.assertEqual(first, second)

    def test_news_contribution_is_zero_while_unavailable(self):
        snapshot = _base_snapshot(rsi=95.0, change_1d=0.2, price_vs_ma_50=0.5)
        with_news = _score_current_time(snapshot, fetch_news_snapshot("TEST", "2024-01-02"))
        without_news_field = _score_current_time(snapshot, {"status": "UNAVAILABLE"})
        self.assertEqual(with_news, without_news_field)

    def test_score_engine_never_derives_news_from_price(self):
        result = build_score("MSFT", "2024-01-02")
        self.assertEqual(result.news_snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(result.news_snapshot["sentiment_score"])
        self.assertNotIn("news_sentiment_proxy", result.scoring_breakdown["weighted_contributions"])


class MarketSnapshotContractTests(unittest.TestCase):
    def test_market_snapshot_features_carry_full_provenance(self):
        snapshot = _base_snapshot()
        snapshot["features"] = {
            "rsi": {
                "name": "rsi",
                "value": 50.0,
                "as_of": snapshot["as_of"],
                "source_id": "yahoo_finance_chart",
                "published_time": "2024-01-02 00:00:00",
                "calculation_version": "market-feature-v1",
                "lookback_period": "14d",
            }
        }
        snapshot["source_contract"] = {"source_id": "yahoo_finance_chart"}
        snapshot["last_valid_bar"] = "2024-01-02 00:00:00"
        canonical = MarketSnapshot.from_dict(snapshot)
        self.assertEqual(canonical.ticker, "TEST")
        rsi_feature = canonical.features["rsi"]
        for attr in ("as_of", "source_id", "published_time", "calculation_version", "lookback_period"):
            self.assertTrue(getattr(rsi_feature, attr), f"feature contract missing {attr}")

    def test_real_market_snapshot_features_satisfy_the_contract(self):
        from agents.market_data_agent import fetch_market_snapshot

        raw = fetch_market_snapshot("MSFT", "2024-01-02")
        canonical = MarketSnapshot.from_dict(raw)
        self.assertTrue(canonical.features)
        for name, feature in canonical.features.items():
            self.assertEqual(feature.as_of, raw["as_of"], name)
            self.assertTrue(feature.source_id, name)
            self.assertTrue(feature.published_time, name)
            self.assertTrue(feature.calculation_version, name)
            self.assertTrue(feature.lookback_period, name)


if __name__ == "__main__":
    unittest.main()
