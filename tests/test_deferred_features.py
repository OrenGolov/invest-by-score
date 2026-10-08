"""Sprint N6: slotted feature gaps — ATR(14), 60d trend slope, 50/100/150/200d
trailing returns, and the breadth registry placeholder.

The N6 rule: registered with full provenance, but DEFERRED — none of these
features may enter a scorer without a weight and a test. These tests pin:

- value correctness against hand-computed / independent definitions
  (true-range definition with a first-session NaN, exact least-squares
  slope on linear ramps, the anchored _pct_change convention);
- strict-window semantics (ATR/slope stay None until the full window
  exists — a shorter mean/fit must never be published under the same name);
- full provenance on every feature contract (as_of, source_id,
  published_time, calculation_version, lookback_period);
- point-in-time behavior (future bars excluded before computation);
- scorer neutrality (mutating the new fields cannot move any scorer, and
  the names are absent from the scored feature lists);
- the breadth placeholder: provider_key_required, zero confidence, never
  inferred from price.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from agents.market_data_agent import (
    _average_true_range,
    _pct_change,
    _trend_slope,
    fetch_market_snapshot,
)
from core.config import MARKET_FEATURE_VERSION
from core.score_engine import (
    CURRENT_SCORE_FEATURES,
    LONG_TERM_SCORE_FEATURES,
    _score_current_time,
    _score_long_term,
)
from fetch_data import SOURCE_REGISTRY

DEFERRED_FEATURE_NAMES = (
    "change_50d", "change_100d", "change_150d", "change_200d", "atr_14", "trend_slope_60d",
)
DEFERRED_LOOKBACKS = {
    "change_50d": "50d", "change_100d": "100d", "change_150d": "150d",
    "change_200d": "200d", "atr_14": "14d", "trend_slope_60d": "60d",
}


def _ramp_closes(sessions, step, base=100.0):
    return [base + step * index for index in range(sessions)]


def _frame(closes, highs=None, lows=None, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    if highs is None:
        highs = list(closes)
    if lows is None:
        lows = list(closes)
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": list(highs),
            "Low": list(lows),
            "Close": list(closes),
            "Volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )


def _independent_true_ranges(frame):
    """Reference TR implementation written straight from the definition."""
    ranges = []
    closes = [float(value) for value in frame["Close"]]
    highs = [float(value) for value in frame["High"]]
    lows = [float(value) for value in frame["Low"]]
    for position in range(1, len(closes)):
        ranges.append(
            max(
                highs[position] - lows[position],
                abs(highs[position] - closes[position - 1]),
                abs(lows[position] - closes[position - 1]),
            )
        )
    return ranges


class AverageTrueRangeTests(unittest.TestCase):
    def test_constant_true_range_frame_gives_the_exact_value(self):
        # high = close + 1, low = close - 1, constant closes: every true
        # range is exactly 2.0 (high-low dominates), so ATR(14) == 2.0.
        frame = _frame([100.0] * 320, highs=[101.0] * 320, lows=[99.0] * 320)
        self.assertEqual(_average_true_range(frame, window=14), 2.0)

    def test_gap_session_enters_the_true_range_window(self):
        # Rising closes with high = close + 0.5, low = close - 0.5 give a
        # true range of 1.5 per session (|high - prev_close| = 1.5 on a +1.0
        # ramp); a final-session gap (high = close + 4.0) contributes
        # max(4.5, 5.0, 0.5) = 5.0 as the newest observation:
        # ATR(14) = (13 * 1.5 + 5.0) / 14.
        sessions = 320
        closes = _ramp_closes(sessions, 1.0)
        highs = [close + 0.5 for close in closes]
        lows = [close - 0.5 for close in closes]
        highs[-1] = closes[-1] + 4.0
        frame = _frame(closes, highs=highs, lows=lows)
        expected = (13 * 1.5 + 5.0) / 14
        self.assertAlmostEqual(_average_true_range(frame, window=14), expected, places=4)
        # Cross-check against the independent definition over the window.
        reference = sum(_independent_true_ranges(frame)[-14:]) / 14
        self.assertAlmostEqual(_average_true_range(frame, window=14), reference, places=4)

    def test_strict_window_none_until_fifteen_sessions(self):
        short = _frame([100.0] * 10, highs=[101.0] * 10, lows=[99.0] * 10)
        self.assertIsNone(_average_true_range(short, window=14))
        # Exactly window + 1 sessions: the first full window exists.
        exact = _frame([100.0] * 15, highs=[101.0] * 15, lows=[99.0] * 15)
        self.assertEqual(_average_true_range(exact, window=14), 2.0)

    def test_first_session_never_counts_as_a_full_range(self):
        # A huge first-session high-low must not enter the mean: TR[0] is
        # undefined (no previous close), so a 15-session frame averages
        # exactly the 14 subsequent true ranges.
        frame = _frame([100.0] * 15, highs=[1000.0] + [101.0] * 14, lows=[1.0] + [99.0] * 14)
        self.assertEqual(_average_true_range(frame, window=14), 2.0)


class TrendSlopeTests(unittest.TestCase):
    def test_linear_ramp_slope_is_exact(self):
        slope = _trend_slope(_frame(_ramp_closes(320, 0.5))["Close"], window=60)
        self.assertAlmostEqual(slope, 0.5, places=9)

    def test_declining_ramp_slope_is_negative_and_exact(self):
        frame = _frame(_ramp_closes(320, -0.25, base=400.0))
        self.assertAlmostEqual(_trend_slope(frame["Close"], window=60), -0.25, places=9)

    def test_slope_uses_only_the_last_sixty_sessions(self):
        # A steep early segment must not contaminate the 60-session fit.
        closes = _ramp_closes(300, 5.0) + _ramp_closes(60, 0.5, base=100.0 + 5.0 * 299)
        self.assertAlmostEqual(_trend_slope(_frame(closes)["Close"], window=60), 0.5, places=9)

    def test_strict_window_none_until_sixty_sessions(self):
        self.assertIsNone(_trend_slope(_frame(_ramp_closes(59, 0.5))["Close"], window=60))
        self.assertAlmostEqual(
            _trend_slope(_frame(_ramp_closes(60, 0.5))["Close"], window=60), 0.5, places=9
        )


class TrailingReturnTests(unittest.TestCase):
    def test_trailing_returns_follow_the_anchored_convention(self):
        frame = _frame(_ramp_closes(320, 0.5))
        for periods in (50, 100, 150, 200):
            with self.subTest(periods=periods):
                expected = float(frame["Close"].iloc[-1] / frame["Close"].iloc[-(periods + 1)]) - 1.0
                self.assertAlmostEqual(_pct_change(frame["Close"], periods), expected, places=12)

    def test_insufficient_anchor_history_reads_zero_by_family_convention(self):
        # change_20d/60d already define the family behavior: 0.0 until the
        # anchor session exists. The 50/100/150/200d returns follow it.
        frame = _frame(_ramp_closes(60, 0.5))
        for periods in (100, 150, 200):
            self.assertEqual(_pct_change(frame["Close"], periods), 0.0, periods)
        # 50d needs 51 sessions: exactly 60 is enough.
        self.assertAlmostEqual(
            _pct_change(frame["Close"], 50),
            float(frame["Close"].iloc[-1] / frame["Close"].iloc[-51]) - 1.0,
            places=12,
        )


class SnapshotFeatureContractTests(unittest.TestCase):
    """The slotted features carry full provenance inside the snapshot."""

    @classmethod
    def _snapshot(cls, closes=None):
        frame = _frame(closes if closes is not None else _ramp_closes(320, 0.5))
        as_of = frame.index[-1].strftime("%Y-%m-%d")
        with patch("agents.market_data_agent.fetch_price_history", return_value=frame):
            return fetch_market_snapshot("TEST", as_of)

    def test_snapshot_exposes_all_six_features_with_provenance(self):
        snapshot = self._snapshot()
        for name in DEFERRED_FEATURE_NAMES:
            with self.subTest(feature=name):
                self.assertIn(name, snapshot["features"], "feature contract missing")
                contract = snapshot["features"][name]
                self.assertIn(name, snapshot, "top-level value missing")
                self.assertEqual(contract["name"], name)
                self.assertEqual(contract["as_of"], snapshot["as_of"])
                self.assertEqual(contract["source_id"], "yahoo_finance_chart")
                self.assertEqual(contract["published_time"], snapshot["last_valid_bar"])
                self.assertEqual(contract["calculation_version"], MARKET_FEATURE_VERSION)
                self.assertEqual(contract["lookback_period"], DEFERRED_LOOKBACKS[name])
                self.assertEqual(contract["value"], snapshot[name])

    def test_canonical_market_snapshot_parses_the_contracts(self):
        from core.schemas import MarketSnapshot

        raw = self._snapshot()
        canonical = MarketSnapshot.from_dict(raw)
        for name in DEFERRED_FEATURE_NAMES:
            contract = canonical.features[name]
            self.assertEqual(contract.calculation_version, MARKET_FEATURE_VERSION, name)
            self.assertTrue(contract.lookback_period, name)

    def test_point_in_time_future_bars_cannot_change_the_features(self):
        frame = _frame(_ramp_closes(320, 0.5))
        extended = _frame(
            _ramp_closes(322, 0.5),
            start=frame.index[0].strftime("%Y-%m-%d"),
        )
        as_of = frame.index[-1].strftime("%Y-%m-%d")
        with patch("agents.market_data_agent.fetch_price_history", return_value=frame):
            truncated = fetch_market_snapshot("TEST", as_of)
        with patch("agents.market_data_agent.fetch_price_history", return_value=extended):
            filtered = fetch_market_snapshot("TEST", as_of)
        self.assertEqual(filtered["future_bars_excluded"], 2)
        for name in DEFERRED_FEATURE_NAMES:
            self.assertEqual(filtered[name], truncated[name], name)
            self.assertEqual(filtered["features"][name], truncated["features"][name], name)

    def test_short_history_is_explicit_not_fabricated(self):
        frame = _frame(_ramp_closes(10, 0.5))
        as_of = frame.index[-1].strftime("%Y-%m-%d")
        with patch("agents.market_data_agent.fetch_price_history", return_value=frame):
            snapshot = fetch_market_snapshot("TEST", as_of)
        self.assertIsNone(snapshot["atr_14"])           # strict window: no silent redefinition
        self.assertIsNone(snapshot["trend_slope_60d"])  # strict window
        self.assertIsNotNone(snapshot["features"]["atr_14"])  # contract still present
        for name in ("change_50d", "change_100d", "change_150d", "change_200d"):
            self.assertEqual(snapshot[name], 0.0, name)  # family convention


class ScorerNeutralityTests(unittest.TestCase):
    """N6 rule: none of the slotted features may enter a scorer yet."""

    @staticmethod
    def _base_snapshot():
        return {
            "ticker": "TEST",
            "as_of": "2024-01-02 00:00:00",
            "close": 100.0,
            "volume": 1_000_000.0,
            "avg_volume_20d": 1_000_000.0,
            "volume_ratio_20d": 1.0,
            "change_1d": 0.005,
            "change_5d": 0.01,
            "change_20d": 0.02,
            "change_60d": 0.03,
            "trend_vs_20d_mean": 0.015,
            "rsi": 55.0,
            "volatility": 0.005,
            "price_vs_ma_50": 0.01,
            "price_vs_ma_100": 0.008,
            "price_vs_ma_150": 0.02,
            "price_vs_ma_200": 0.03,
            "moving_averages": {"50d": 101.0, "100d": 100.0, "150d": 103.0, "200d": 100.0},
        }

    def test_deferred_names_are_absent_from_scored_feature_lists(self):
        for name in DEFERRED_FEATURE_NAMES:
            self.assertNotIn(name, CURRENT_SCORE_FEATURES, name)
            self.assertNotIn(name, LONG_TERM_SCORE_FEATURES, name)

    def test_mutating_the_new_fields_cannot_move_any_scorer(self):
        base = self._base_snapshot()
        injected = dict(
            base,
            atr_14=99.0,
            trend_slope_60d=-42.0,
            change_50d=9.9,
            change_100d=-9.9,
            change_150d=0.55,
            change_200d=-0.55,
        )
        self.assertEqual(_score_current_time(base), _score_current_time(injected))
        self.assertEqual(_score_long_term(base), _score_long_term(injected))


class BreadthRegistryPlaceholderTests(unittest.TestCase):
    """Breadth/participation is deferred, never inferred from price."""

    def test_breadth_registry_entry_is_an_explicit_placeholder(self):
        breadth = SOURCE_REGISTRY["breadth"]
        self.assertEqual(breadth["status"], "provider_key_required")
        self.assertEqual(breadth["source_id"], "breadth_provider_required")
        self.assertEqual(breadth["domain"], "market_breadth")
        self.assertEqual(breadth["base_confidence"], 0.0)

    def test_placeholder_cannot_masquerade_as_live_coverage(self):
        for domain, entry in SOURCE_REGISTRY.items():
            if domain == "breadth":
                self.assertNotEqual(entry["status"], "live_provider")


if __name__ == "__main__":
    unittest.main()
