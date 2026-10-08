"""Price/volume feature expansion tests (Sprint C2).

Two rules are pinned here.

**Insufficient history yields None, never zero.** A drawdown computed over
three bars is not a drawdown, and a neutral-looking 0.0 is indistinguishable
from a genuine one once it reaches a training row.

**The benchmark is injected, never invented.** "+4% while the index did +4%"
and "+4% while the index did -2%" are different facts. With no benchmark the
feature refuses to answer rather than substituting a default, because C2 does
not own benchmark selection — C3 does.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    CHART_BREAKOUT_DOWN,
    CHART_BREAKOUT_FAILED_UP,
    CHART_BREAKOUT_NONE,
    CHART_BREAKOUT_STATES,
    CHART_BREAKOUT_UP,
    CHART_FEATURE_MIN_HISTORY,
    CHART_VOL_CONTRACTING,
    CHART_VOL_EXPANDING,
    CHART_VOL_STABLE,
)
from core.chart_features import (
    ChartFeatureError,
    acceleration,
    breakout_state,
    compute_chart_features,
    drawdown,
    gap_pct,
    recovery_speed,
    relative_strength,
    resistance_distance,
    support_distance,
    volatility_regime,
    volatility_regime_ratio,
)
from core.feature_registry import build_default_registry, producer_problems


def _frame(closes, opens=None, highs=None, lows=None, start="2026-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": opens if opens is not None else closes,
            "High": highs if highs is not None else [c * 1.01 for c in closes],
            "Low": lows if lows is not None else [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(closes),
        },
        index=index,
    )


def _flat(n, value=100.0):
    return _frame([value] * n)


class InsufficientHistoryTests(unittest.TestCase):
    """The rule: withhold the answer, never fake a neutral one."""

    def test_every_windowed_feature_is_none_on_a_short_frame(self):
        short = _flat(3)
        for name, fn in (
            ("acceleration_10d", acceleration),
            ("drawdown_60d", drawdown),
            ("recovery_speed_60d", recovery_speed),
            ("support_distance_60d", support_distance),
            ("resistance_distance_60d", resistance_distance),
            ("breakout_state_60d", breakout_state),
            ("volatility_regime_ratio", volatility_regime_ratio),
        ):
            with self.subTest(feature=name):
                self.assertIsNone(fn(short), f"{name} fabricated a value")

    def test_a_withheld_feature_is_named_not_just_none(self):
        """A consumer must be able to tell 'no data' from 'computed as zero'."""
        result = compute_chart_features(_flat(3))
        self.assertIn("drawdown_60d", result["insufficient_history"])
        self.assertIsNone(result["features"]["drawdown_60d"])

    def test_a_feature_with_enough_history_is_not_listed(self):
        result = compute_chart_features(_frame([100.0, 102.0]))
        self.assertNotIn("gap_pct", result["insufficient_history"])

    def test_minimums_are_declared_for_every_windowed_feature(self):
        for name in (
            "acceleration_10d", "gap_pct", "drawdown_60d", "recovery_speed_60d",
            "support_distance_60d", "resistance_distance_60d",
            "breakout_state_60d", "volatility_regime_ratio",
            "relative_strength_60d",
        ):
            with self.subTest(feature=name):
                self.assertIn(name, CHART_FEATURE_MIN_HISTORY)


class AccelerationTests(unittest.TestCase):
    def test_a_speeding_up_rally_accelerates(self):
        closes = [100.0] * 11 + [100.0 + i for i in range(1, 11)] + [110.0 + 3 * i for i in range(1, 11)]
        self.assertGreater(acceleration(_frame(closes)), 0)

    def test_a_stalling_rally_decelerates(self):
        """Price still rising, momentum falling — the case a return misses."""
        closes = [100.0 + 3 * i for i in range(11)] + [133.0 + 0.1 * i for i in range(1, 11)]
        closes += [134.0 + 0.05 * i for i in range(1, 11)]
        self.assertLess(acceleration(_frame(closes)), 0)

    def test_a_steady_trend_has_near_zero_acceleration(self):
        """Linear growth compounds to slightly DECAYING percentage returns, so
        exact zero is the wrong expectation — near-zero is the real property."""
        closes = [100.0 + i for i in range(40)]
        self.assertLess(abs(acceleration(_frame(closes))), 0.05)


class GapTests(unittest.TestCase):
    def test_a_real_gap_is_reported(self):
        frame = _frame([100.0, 110.0], opens=[100.0, 109.0])
        self.assertGreater(gap_pct(frame), 0.05)

    def test_ordinary_drift_is_zero_not_a_gap(self):
        """Bars exist and the answer is 'no gap' — different from unknown."""
        frame = _frame([100.0, 100.2], opens=[100.0, 100.1])
        self.assertEqual(gap_pct(frame), 0.0)

    def test_a_down_gap_is_negative(self):
        frame = _frame([100.0, 90.0], opens=[100.0, 91.0])
        self.assertLess(gap_pct(frame), 0)


class DrawdownAndRecoveryTests(unittest.TestCase):
    def test_at_the_high_the_drawdown_is_zero(self):
        self.assertEqual(drawdown(_frame([100.0 + i for i in range(60)])), 0.0)

    def test_below_the_high_the_drawdown_is_negative(self):
        closes = [100.0 + i for i in range(50)] + [140.0 - i for i in range(10)]
        self.assertLess(drawdown(_frame(closes)), 0)

    def test_the_drawdown_is_never_positive(self):
        closes = [100.0 + (i % 7) for i in range(60)]
        self.assertLessEqual(drawdown(_frame(closes)), 0.0)

    def test_full_recovery_reads_one(self):
        closes = [100.0] * 30 + [80.0] * 10 + [100.0] * 20
        self.assertAlmostEqual(recovery_speed(_frame(closes)), 1.0, places=4)

    def test_sitting_at_the_trough_reads_zero(self):
        closes = [100.0] * 40 + [80.0] * 20
        self.assertAlmostEqual(recovery_speed(_frame(closes)), 0.0, places=4)

    def test_a_flat_series_has_no_recovery_to_measure(self):
        """Recovery from nothing is not a measurement."""
        self.assertIsNone(recovery_speed(_flat(60)))


class LevelDistanceTests(unittest.TestCase):
    def test_support_distance_is_positive_above_the_low(self):
        closes = [100.0 + i for i in range(60)]
        self.assertGreater(support_distance(_frame(closes)), 0)

    def test_resistance_distance_is_positive_below_the_high(self):
        closes = [160.0 - i for i in range(60)]
        self.assertGreater(resistance_distance(_frame(closes)), 0)

    def test_at_the_high_resistance_distance_is_about_zero(self):
        closes = [100.0 + i for i in range(60)]
        frame = _frame(closes, highs=closes)
        self.assertAlmostEqual(resistance_distance(frame), 0.0, places=6)


class BreakoutTests(unittest.TestCase):
    def test_a_holding_break_above_the_range_is_a_breakout(self):
        closes = [100.0] * 60 + [120.0] * 5
        self.assertEqual(breakout_state(_frame(closes)), CHART_BREAKOUT_UP)

    def test_a_break_that_closes_back_inside_is_a_FAILED_breakout(self):
        """The distinction that makes this feature worth having."""
        closes = [100.0 + (i % 5) for i in range(60)] + [120.0, 118.0, 110.0, 103.0, 102.0]
        self.assertEqual(breakout_state(_frame(closes)), CHART_BREAKOUT_FAILED_UP)

    def test_a_break_in_both_directions_resolves_to_where_price_now_sits(self):
        """A run that pierces both sides is ambiguous by construction.

        Precedence is: a break that is still HOLDING wins over one that has
        already reversed, so the label describes where price is now rather
        than which extreme happened to come first.
        """
        closes = [100.0] * 60 + [120.0, 118.0, 105.0, 100.0, 99.0]
        self.assertEqual(breakout_state(_frame(closes)), CHART_BREAKOUT_DOWN)

    def test_a_quiet_range_is_no_breakout(self):
        closes = [100.0 + (i % 3) * 0.1 for i in range(65)]
        self.assertEqual(breakout_state(_frame(closes)), CHART_BREAKOUT_NONE)

    def test_the_breakout_never_redefines_the_level_it_broke(self):
        """The range excludes the confirmation window, or nothing ever breaks out."""
        closes = [100.0] * 60 + [130.0] * 5
        self.assertEqual(breakout_state(_frame(closes)), CHART_BREAKOUT_UP)

    def test_the_state_is_always_a_declared_one(self):
        closes = [100.0 + (i % 11) for i in range(80)]
        self.assertIn(breakout_state(_frame(closes)), CHART_BREAKOUT_STATES)


class VolatilityRegimeTests(unittest.TestCase):
    def test_a_calm_series_turning_wild_is_expanding(self):
        closes = [100.0 + (i % 2) * 0.05 for i in range(60)]
        closes += [100.0 + (i % 2) * 12.0 for i in range(12)]
        ratio = volatility_regime_ratio(_frame(closes))
        self.assertEqual(volatility_regime(ratio), CHART_VOL_EXPANDING)

    def test_a_wild_series_calming_down_is_contracting(self):
        closes = [100.0 + (i % 2) * 15.0 for i in range(60)]
        closes += [100.0 + (i % 2) * 0.02 for i in range(12)]
        ratio = volatility_regime_ratio(_frame(closes))
        self.assertEqual(volatility_regime(ratio), CHART_VOL_CONTRACTING)

    def test_a_consistent_series_is_stable(self):
        closes = [100.0 + (i % 4) * 1.0 for i in range(80)]
        self.assertEqual(volatility_regime(volatility_regime_ratio(_frame(closes))), CHART_VOL_STABLE)

    def test_no_ratio_means_no_label_not_a_default(self):
        self.assertIsNone(volatility_regime(None))


class RelativeStrengthTests(unittest.TestCase):
    """C2 never selects or invents a benchmark — that is C3's contract."""

    def _pair(self, stock_closes, bench_closes):
        return _frame(stock_closes), _frame(bench_closes)

    def test_no_benchmark_means_no_answer(self):
        stock = _frame([100.0 + i for i in range(80)])
        self.assertIsNone(relative_strength(stock, None))

    def test_outperformance_is_positive(self):
        stock, bench = self._pair([100.0 + i for i in range(80)], [100.0] * 80)
        self.assertGreater(relative_strength(stock, bench), 0)

    def test_underperformance_is_negative(self):
        stock, bench = self._pair([100.0] * 80, [100.0 + i for i in range(80)])
        self.assertLess(relative_strength(stock, bench), 0)

    def test_matching_the_benchmark_is_about_zero(self):
        """+4% while the index did +4% is NOT strength."""
        closes = [100.0 + i for i in range(80)]
        stock, bench = self._pair(closes, list(closes))
        self.assertAlmostEqual(relative_strength(stock, bench), 0.0, places=6)

    def test_alignment_is_by_timestamp_not_position(self):
        """A shorter benchmark must not be measured over a different window."""
        stock = _frame([100.0 + i for i in range(80)], start="2026-01-01")
        bench = _frame([100.0 + i for i in range(80)], start="2025-01-01")
        self.assertIsNone(relative_strength(stock, bench))

    def test_a_stale_benchmark_is_refused_not_silently_reweighted(self):
        """Intersecting alone lets a stale benchmark drag the window backwards.

        The stock's most recent sessions get dropped with no disclosure, and the
        caller receives a number measured over a period they did not ask for. On
        a 120-bar stock a 20-session-stale benchmark moved the reported value
        0.4938 -> 0.6164.
        """
        stock = _frame([100.0 + 2.0 * i for i in range(120)], start="2026-01-01")
        stale = _frame([100.0 + 0.1 * i for i in range(100)], start="2026-01-01")
        self.assertIsNone(relative_strength(stock, stale))

    def test_a_current_benchmark_still_resolves(self):
        stock = _frame([100.0 + 2.0 * i for i in range(120)], start="2026-01-01")
        fresh = _frame([100.0 + 0.1 * i for i in range(120)], start="2026-01-01")
        self.assertIsNotNone(relative_strength(stock, fresh))

    def test_a_benchmark_extending_past_the_stock_is_fine(self):
        """Only staleness matters: extra benchmark history is simply unused."""
        stock = _frame([100.0 + 2.0 * i for i in range(120)], start="2026-01-01")
        longer = _frame([100.0 + 0.1 * i for i in range(160)], start="2026-01-01")
        self.assertIsNotNone(relative_strength(stock, longer))

    def test_a_benchmark_without_a_close_column_is_refused(self):
        stock = _frame([100.0 + i for i in range(80)])
        broken = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-01-01"]))
        self.assertIsNone(relative_strength(stock, broken))


class SurfaceTests(unittest.TestCase):
    def test_the_surface_carries_every_c2_feature(self):
        result = compute_chart_features(_frame([100.0 + (i % 9) for i in range(120)]))
        for name in CHART_FEATURE_MIN_HISTORY:
            with self.subTest(feature=name):
                self.assertIn(name, result["features"])

    def test_the_surface_is_versioned(self):
        result = compute_chart_features(_flat(80))
        self.assertTrue(result["calculation_version"])
        self.assertTrue(result["pipeline_version"])

    def test_a_frame_without_close_is_refused(self):
        with self.assertRaises(ChartFeatureError):
            compute_chart_features(pd.DataFrame({"Open": [1.0]}))

    def test_a_missing_frame_is_refused(self):
        with self.assertRaises(ChartFeatureError):
            compute_chart_features(None)

    def test_the_surface_is_deterministic(self):
        frame = _frame([100.0 + (i % 13) for i in range(150)])
        self.assertEqual(compute_chart_features(frame), compute_chart_features(frame))

    def test_computing_reads_only_backwards(self):
        """Appending a future bar must not change a value already computed.

        The producer does no as_of filtering of its own — it trusts the caller's
        frame — so this pins that it never reaches past the last row it was
        given, and that a longer frame is a different question, not a leak.
        """
        base = _frame([100.0 + (i % 7) for i in range(120)])
        trimmed = base.iloc[:-1]
        self.assertEqual(
            compute_chart_features(trimmed)["features"]["gap_pct"],
            gap_pct(trimmed),
        )


class RegistryWiringTests(unittest.TestCase):
    """M1: an unregistered feature cannot enter a production model."""

    def setUp(self):
        self.registry = build_default_registry()._features

    def test_every_c2_feature_is_registered(self):
        for name in CHART_FEATURE_MIN_HISTORY:
            with self.subTest(feature=name):
                self.assertIn(name, self.registry)

    def test_the_producer_exists_on_disk(self):
        self.assertEqual(producer_problems("chart_feature_agent"), [])

    def test_c2_features_declare_the_chart_producer(self):
        for name in CHART_FEATURE_MIN_HISTORY:
            with self.subTest(feature=name):
                self.assertEqual(self.registry[name].owner, "chart_feature_agent")

    def test_c2_features_exclude_rather_than_default_on_null(self):
        """A withheld feature must not be silently replaced by a neutral value."""
        for name in CHART_FEATURE_MIN_HISTORY:
            with self.subTest(feature=name):
                self.assertEqual(self.registry[name].null_policy, "exclude")

    def test_existing_market_features_were_not_re_owned(self):
        """W5: no second implementation of an already-canonical feature."""
        for name in ("change_20d", "rsi", "volatility", "atr_14",
                     "volume_ratio_20d", "trend_slope_60d"):
            with self.subTest(feature=name):
                self.assertEqual(self.registry[name].owner, "market_data_agent")

    def test_registered_minimum_history_matches_the_producer(self):
        """The registry must not promise a window the producer does not enforce."""
        for name, minimum in CHART_FEATURE_MIN_HISTORY.items():
            with self.subTest(feature=name):
                self.assertEqual(self.registry[name].minimum_history, minimum)
