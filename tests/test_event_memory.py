"""Event memory tests (Sprint E6).

Stores event, context, chart state, historical analogs and the
1D/5D/20D/60D response as institutional memory.

This is the store the forecasting sprints will retrieve from without
re-deriving anything, so its value depends entirely on not remembering
things that were never true. These tests pin the refusals:

- an unmeasured event never enters memory — it teaches nothing while
  counting toward every analog total;
- a memory without chart state is a log, not institutional memory;
- re-recording cannot inflate the analog count, which would silently make
  every historical base rate wrong;
- a "typical response" from too few examples is refused outright.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.config import (
    EVENT_MEMORY_CHART_FIELDS,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    EVENT_MEMORY_RESPONSE_HORIZONS,
    EVENT_MEMORY_VERSION,
)
from core.attribution import attribute_study
from core.event_contract import Event
from core.event_memory import (
    EventMemory,
    EventMemoryError,
    analog_summary,
    build_memory,
    chart_similarity,
    find_analogs,
    load_memories,
    load_memory_objects,
    memory_problems,
    memory_report,
    remember,
)
from core.event_study import study_event


def _frame(bars: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, bars))
    return pd.DataFrame(
        {
            "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, bars).astype(float),
        },
        index=pd.bdate_range("2024-01-01", periods=bars),
    )


def _snapshot(**overrides) -> dict:
    payload = {
        "close": 120.0, "rsi": 55.0, "volatility": 0.02,
        "volume_ratio_20d": 1.1, "atr_14": 2.5, "trend_slope_60d": 0.3,
        "trend_vs_20d_mean": 0.02, "market_regime": "bullish",
        "change_5d": 0.01, "change_20d": 0.05, "change_60d": 0.12,
        "price_vs_ma_50": 1.03, "price_vs_ma_200": 1.15,
    }
    payload.update(overrides)
    return payload


def _memory(event_id: str = "e1", **overrides) -> EventMemory:
    payload = dict(
        event_id=event_id, ticker="NVDA", published_time="2026-01-05 00:00:00",
        event_type="earnings", direction="positive",
        chart_state=_snapshot(),
        response={"20d": {"abnormal_return": 0.03, "stock_return": 0.05}},
        attribution={"20d": "event_associated"},
    )
    payload.update(overrides)
    return EventMemory(**payload)


class MemoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "event_memory.jsonl"
        self.addCleanup(self._tmp.cleanup)


class TestWhatIsRemembered(MemoryTestCase):
    """Event, context, chart state and response — all four."""

    def setUp(self) -> None:
        super().setUp()
        self.frame = _frame()
        self.benchmark = _frame(seed=1)
        self.event = Event(
            entity="NVDA",
            published_time=self.frame.index[200].strftime("%Y-%m-%d %H:%M:%S"),
            event_type="earnings", source="news", actor="Jensen Huang",
            evidence=[{"source_record_id": "r1"}],
        )
        self.study = study_event(self.event, self.frame, self.benchmark)

    def _built(self) -> EventMemory:
        return build_memory(
            self.event, self.study, attribute_study(self.study), _snapshot()
        )

    def test_the_event_itself_is_remembered(self) -> None:
        memory = self._built()
        self.assertEqual(memory.event_id, self.event.event_id)
        self.assertEqual(memory.ticker, "NVDA")
        self.assertEqual(memory.event_type, "earnings")
        self.assertEqual(memory.actor, "Jensen Huang")

    def test_the_chart_state_is_remembered(self) -> None:
        chart = self._built().chart_state
        for field_name in EVENT_MEMORY_CHART_FIELDS:
            with self.subTest(field=field_name):
                self.assertIn(field_name, chart)

    def test_the_context_is_remembered(self) -> None:
        context = self._built().context
        for key in ("benchmark", "sector", "model", "baseline_volatility", "market_regime"):
            with self.subTest(key=key):
                self.assertIn(key, context)

    def test_every_horizon_response_is_remembered(self) -> None:
        response = self._built().response
        self.assertEqual(set(response), set(EVENT_MEMORY_RESPONSE_HORIZONS))
        for horizon, values in response.items():
            with self.subTest(horizon=horizon):
                for key in ("stock_return", "abnormal_return",
                            "volatility_ratio", "volume_ratio"):
                    self.assertIn(key, values)

    def test_the_attribution_verdict_travels_with_the_memory(self) -> None:
        """A confounded response is not the same thing as an associated one."""
        memory = self._built()
        self.assertTrue(memory.attribution)
        for horizon in memory.attribution:
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, memory.response)

    def test_the_version_is_stamped(self) -> None:
        self.assertEqual(self._built().memory_version, EVENT_MEMORY_VERSION)


class TestRefusals(MemoryTestCase):
    def test_an_unmeasured_study_is_refused(self) -> None:
        """It teaches nothing while counting toward every analog total."""
        frame = _frame()
        early = Event(
            entity="NVDA", published_time=frame.index[5].strftime("%Y-%m-%d %H:%M:%S"),
            event_type="earnings", source="news",
            evidence=[{"source_record_id": "r1"}],
        )
        study = study_event(early, frame, _frame(seed=1))
        with self.assertRaises(EventMemoryError) as ctx:
            build_memory(early, study, {}, _snapshot())
        self.assertIn("teaches nothing", str(ctx.exception))

    def test_a_memory_without_chart_state_is_refused(self) -> None:
        """Institutional memory that cannot be compared is just a log."""
        problems = memory_problems(_memory(chart_state={}))
        self.assertTrue(any("chart_state" in p for p in problems))

    def test_a_memory_without_a_response_is_refused(self) -> None:
        problems = memory_problems(_memory(response={}))
        self.assertTrue(any("teaches nothing" in p for p in problems))

    def test_identity_fields_are_required(self) -> None:
        for field_name in ("event_id", "ticker", "published_time", "event_type"):
            with self.subTest(missing=field_name):
                problems = memory_problems(_memory(**{field_name: ""}))
                self.assertTrue(any(field_name in p for p in problems))

    def test_an_attribution_without_a_response_is_refused(self) -> None:
        problems = memory_problems(_memory(attribution={"5d": "event_associated"}))
        self.assertTrue(any("no response was measured" in p for p in problems))

    def test_an_incomplete_memory_is_never_written(self) -> None:
        with self.assertRaises(EventMemoryError):
            remember(_memory(chart_state={}), self.path)
        self.assertFalse(self.path.exists())


class TestImmutabilityAndDeduplication(MemoryTestCase):
    def test_re_recording_does_not_inflate_the_count(self) -> None:
        """Otherwise every historical base rate silently becomes wrong."""
        memory = _memory()
        for _ in range(3):
            remember(memory, self.path)
        self.assertEqual(len(load_memories(self.path)), 1)

    def test_distinct_events_are_both_kept(self) -> None:
        remember(_memory("e1"), self.path)
        remember(_memory("e2"), self.path)
        self.assertEqual(len(load_memories(self.path)), 2)

    def test_a_malformed_line_raises_loudly(self) -> None:
        self.path.write_text("{not json\n", encoding="utf-8")
        with self.assertRaises(EventMemoryError):
            load_memories(self.path)

    def test_a_missing_store_reads_empty(self) -> None:
        self.assertEqual(load_memories(self.path), [])

    def test_memories_round_trip_as_objects(self) -> None:
        remember(_memory(), self.path)
        loaded = load_memory_objects(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].response_at("20d"), 0.03)


class TestSimilarity(unittest.TestCase):
    def test_an_identical_chart_scores_one(self) -> None:
        self.assertEqual(chart_similarity(_snapshot(), _snapshot()), 1.0)

    def test_a_different_chart_scores_lower(self) -> None:
        other = _snapshot(rsi=20.0, change_20d=-0.15, market_regime="bearish")
        self.assertLess(chart_similarity(_snapshot(), other), 1.0)

    def test_similarity_is_scale_free(self) -> None:
        """A $500 stock and a $50 stock with the same shape are comparable."""
        expensive = _snapshot(close=500.0)
        cheap = _snapshot(close=50.0)
        # Only `close` differs; every other field matches exactly.
        self.assertGreater(chart_similarity(expensive, cheap), 0.9)

    def test_a_missing_field_is_not_agreement(self) -> None:
        partial = {"rsi": 55.0}
        self.assertEqual(chart_similarity(partial, {"rsi": 55.0}), 1.0)
        self.assertEqual(chart_similarity({}, _snapshot()), 0.0)

    def test_regime_matching_is_categorical(self) -> None:
        same = chart_similarity({"market_regime": "bullish"}, {"market_regime": "bullish"})
        different = chart_similarity({"market_regime": "bullish"}, {"market_regime": "stress"})
        self.assertEqual(same, 1.0)
        self.assertEqual(different, 0.0)


class TestAnalogRetrieval(MemoryTestCase):
    def _pool(self, count: int, event_type: str = "earnings") -> list[EventMemory]:
        return [
            _memory(
                f"e{index}", event_type=event_type,
                chart_state=_snapshot(rsi=55.0 + index * 0.2),
                response={"20d": {"abnormal_return": 0.01 * (index - 3)}},
            )
            for index in range(count)
        ]

    def test_similar_setups_are_retrieved(self) -> None:
        analogs = find_analogs(_snapshot(), "earnings", self._pool(8))
        self.assertGreater(len(analogs), 0)

    def test_analogs_are_sorted_by_similarity(self) -> None:
        analogs = find_analogs(_snapshot(), "earnings", self._pool(8))
        scores = [entry["similarity"] for entry in analogs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_different_event_type_is_not_a_comparable(self) -> None:
        """A regulatory action is not a comparable for an earnings surprise."""
        pool = self._pool(8, event_type="regulation")
        self.assertEqual(find_analogs(_snapshot(), "earnings", pool), [])

    def test_a_dissimilar_chart_is_excluded(self) -> None:
        pool = [_memory("e1", chart_state=_snapshot(
            rsi=5.0, change_20d=-0.5, change_60d=-0.6, market_regime="stress",
            price_vs_ma_50=0.6, price_vs_ma_200=0.5, trend_slope_60d=-0.9,
        ))]
        self.assertEqual(find_analogs(_snapshot(), "earnings", pool), [])

    def test_the_event_itself_can_be_excluded(self) -> None:
        pool = self._pool(8)
        analogs = find_analogs(_snapshot(), "earnings", pool, exclude_event_id="e0")
        self.assertNotIn("e0", [entry["memory"].event_id for entry in analogs])

    def test_the_similarity_threshold_is_respected(self) -> None:
        analogs = find_analogs(_snapshot(), "earnings", self._pool(8))
        for entry in analogs:
            with self.subTest(event=entry["memory"].event_id):
                self.assertGreaterEqual(entry["similarity"], EVENT_MEMORY_MIN_SIMILARITY)


class TestAnalogSummary(MemoryTestCase):
    def _analogs(self, count: int) -> list[dict]:
        return [
            {
                "similarity": 0.9,
                "memory": _memory(
                    f"e{index}",
                    response={"20d": {"abnormal_return": 0.01 * (index - 2)}},
                ),
            }
            for index in range(count)
        ]

    def test_too_few_analogs_refuses_to_summarise(self) -> None:
        """A typical response from two examples is not typical of anything."""
        summary = analog_summary(self._analogs(EVENT_MEMORY_MIN_ANALOGS - 1))
        self.assertEqual(summary["status"], "insufficient_analogs")

    def test_an_insufficient_summary_reports_no_statistics(self) -> None:
        summary = analog_summary(self._analogs(2))
        for key in ("median_response", "mean_response", "positive_share"):
            with self.subTest(key=key):
                self.assertNotIn(key, summary)

    def test_enough_analogs_produce_a_summary(self) -> None:
        summary = analog_summary(self._analogs(EVENT_MEMORY_MIN_ANALOGS))
        self.assertEqual(summary["status"], "measured")
        self.assertIn("median_response", summary)
        self.assertIn("dispersion", summary)

    def test_the_summary_disclaims_being_a_forecast(self) -> None:
        summary = analog_summary(self._analogs(EVENT_MEMORY_MIN_ANALOGS))
        self.assertIn("not a forecast", summary["detail"])

    def test_analogs_without_the_horizon_do_not_count(self) -> None:
        analogs = [
            {"similarity": 0.9, "memory": _memory(f"e{i}", response={"5d": {"abnormal_return": 0.01}})}
            for i in range(EVENT_MEMORY_MIN_ANALOGS + 2)
        ]
        self.assertEqual(analog_summary(analogs, "20d")["status"], "insufficient_analogs")

    def test_the_event_associated_share_is_reported(self) -> None:
        """A confounded history is weaker evidence than an associated one."""
        summary = analog_summary(self._analogs(EVENT_MEMORY_MIN_ANALOGS))
        self.assertIn("event_associated_share", summary)


class TestReport(MemoryTestCase):
    def test_the_report_counts_the_store(self) -> None:
        remember(_memory("e1"), self.path)
        remember(_memory("e2", event_type="guidance"), self.path)
        report = memory_report(self.path)
        self.assertEqual(report["memories"], 2)
        self.assertEqual(report["by_event_type"]["guidance"], 1)

    def test_an_empty_store_reports_zero(self) -> None:
        self.assertEqual(memory_report(self.path)["memories"], 0)


if __name__ == "__main__":
    unittest.main()


class TestNearZeroSimilarity(unittest.TestCase):
    """Regression: relative similarity collapsed near zero.

    Dividing the gap by max(|a|,|b|) meant two nearly-flat values
    (+0.0008 vs -0.0006) scored 0.0 — both mean "flat", but the measure
    called them completely dissimilar. Two near-identical charts scored
    0.654 and fell below the 0.7 retrieval bar, so analogs were silently
    missed with no way to report the miss.
    """

    def _near_identical(self) -> tuple[dict, dict]:
        left = {
            "rsi": 55.0, "volatility": 0.020, "change_5d": 0.0008,
            "change_20d": 0.051, "trend_slope_60d": 0.0012,
            "market_regime": "bullish",
        }
        right = {
            "rsi": 55.5, "volatility": 0.021, "change_5d": -0.0006,
            "change_20d": 0.052, "trend_slope_60d": -0.0009,
            "market_regime": "bullish",
        }
        return left, right

    def test_near_identical_charts_clear_the_retrieval_bar(self) -> None:
        left, right = self._near_identical()
        self.assertGreater(chart_similarity(left, right), EVENT_MEMORY_MIN_SIMILARITY)

    def test_two_flat_readings_are_similar(self) -> None:
        """Both are flat; opposite tiny signs do not make them opposites."""
        self.assertGreater(
            chart_similarity({"trend_slope_60d": 0.0012}, {"trend_slope_60d": -0.0009}),
            0.9,
        )

    def test_genuinely_opposite_moves_still_score_zero(self) -> None:
        """The fix must not blur real differences."""
        self.assertEqual(
            chart_similarity({"change_20d": 0.05}, {"change_20d": -0.05}), 0.0
        )

    def test_overbought_and_oversold_remain_dissimilar(self) -> None:
        self.assertEqual(chart_similarity({"rsi": 70.0}, {"rsi": 30.0}), 0.0)

    def test_identical_charts_still_score_one(self) -> None:
        left, _ = self._near_identical()
        self.assertEqual(chart_similarity(left, left), 1.0)

    def test_a_declared_field_without_a_scale_falls_back_to_relative(self) -> None:
        """Not every chart field needs an absolute scale."""
        self.assertGreater(
            chart_similarity(
                {"close": 100.0}, {"close": 100.2}, fields=("close",)
            ),
            0.9,
        )

    def test_a_field_outside_the_chart_state_is_ignored(self) -> None:
        """Only declared chart fields are compared; anything else is not
        part of the chart state and contributes nothing."""
        self.assertEqual(
            chart_similarity({"unknown_field": 10.0}, {"unknown_field": 10.5}), 0.0
        )
