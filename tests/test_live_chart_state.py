"""Live chart-state tests (Sprint L blocker).

The behaviour under test is that a LIVE forecast is anchored to today's chart
and REFUSES when it cannot be, rather than silently reaching for the most
recent stored state.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from core.chart_features import ChartFeatureError, chart_state
from core.config import (
    EVENT_MEMORY_CHART_FIELDS,
    LIVE_CHART_FALLBACK_TO_MEMORY,
    LIVE_CHART_MAX_STALENESS_DAYS,
    LIVE_CHART_MIN_HISTORY,
)


def frame(bars: int, start: str = "2024-01-01", base: float = 100.0):
    index = pd.bdate_range(start=start, periods=bars)
    closes = [base + step * 0.1 for step in range(bars)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value * 1.01 for value in closes],
            "Low": [value * 0.99 for value in closes],
            "Close": closes,
            "Volume": [1_000_000] * bars,
        },
        index=index,
    )


class MinimumHistoryTests(unittest.TestCase):
    def test_a_short_frame_yields_none_not_an_approximation(self):
        # A price_vs_ma_200 over 40 bars is not a 200-day mean, and nothing
        # downstream could tell the difference.
        self.assertIsNone(chart_state(frame(LIVE_CHART_MIN_HISTORY - 1)))

    def test_a_sufficient_frame_yields_a_state(self):
        self.assertIsNotNone(chart_state(frame(LIVE_CHART_MIN_HISTORY + 1)))

    def test_the_minimum_covers_the_longest_window(self):
        self.assertGreaterEqual(LIVE_CHART_MIN_HISTORY, 210)


class PointInTimeTests(unittest.TestCase):
    def test_a_position_reads_only_backwards(self):
        full = frame(400)
        self.assertEqual(chart_state(full, 250), chart_state(full.iloc[:251]))

    def test_a_later_position_differs_from_an_earlier_one(self):
        full = frame(400)
        self.assertNotEqual(chart_state(full, 250), chart_state(full, 399))

    def test_the_default_position_is_the_last_bar(self):
        full = frame(400)
        self.assertEqual(chart_state(full), chart_state(full, len(full) - 1))

    def test_a_position_outside_the_frame_is_refused(self):
        with self.assertRaises(ChartFeatureError):
            chart_state(frame(300), 5_000)

    def test_a_descending_frame_is_refused_not_sorted(self):
        with self.assertRaises(ChartFeatureError):
            chart_state(frame(300).iloc[::-1])


class OneBuilderTests(unittest.TestCase):
    """W5: the backfill and the live run share one definition."""

    def test_the_backfill_delegates(self):
        import build_event_memory

        data = frame(300)
        self.assertEqual(
            build_event_memory._chart_snapshot(data, 280), chart_state(data, 280)
        )

    def test_the_state_covers_the_memory_chart_fields(self):
        produced = set(chart_state(frame(300)) or {})
        self.assertEqual(produced, set(EVENT_MEMORY_CHART_FIELDS))


class RefusalTests(unittest.TestCase):
    """A live run refuses rather than reaching for a stored state."""

    def setUp(self):
        import fetch_data
        import run_forecasts

        self.run_forecasts = run_forecasts
        self.fetch_data = fetch_data
        self.original = fetch_data.fetch_price_history

    def tearDown(self):
        self.fetch_data.fetch_price_history = self.original

    def _starve(self):
        def dead(*_args, **_kwargs):
            raise RuntimeError("feed down")

        self.fetch_data.fetch_price_history = dead

    def test_an_unobtainable_ticker_gets_no_state(self):
        self._starve()
        states, refusals = self.run_forecasts.live_chart_states(
            ["AAPL"], "2026-09-21"
        )
        self.assertEqual(states, {})
        self.assertIn("AAPL", refusals)

    def test_a_refusal_carries_its_reason(self):
        self._starve()
        _, refusals = self.run_forecasts.live_chart_states(["AAPL"], "2026-09-21")
        self.assertTrue(refusals["AAPL"].strip())

    def test_record_run_forecasts_nothing_without_live_bars(self):
        self._starve()
        result = self.run_forecasts.record_run(
            ["AAPL", "VOO"], "2026-09-21", dry_run=True, path=None
        )
        self.assertEqual(result.get("attempted", 0), 0)
        self.assertEqual(result.get("recorded", 0), 0)

    def test_the_refused_tickers_stay_in_the_report(self):
        # `record_run` returns early when the event-memory store is empty —
        # correctly, since it refuses to forecast from nothing — and that
        # early return reaches no chart stage, so it carries no refusals.
        #
        # THE STORE IS GITIGNORED, so this test passed on my machine (2084
        # memories) and failed on every CI run (a fresh clone has none). A
        # test that depends on untracked data is testing a working directory,
        # not the repository.
        self._starve()
        result = self.run_forecasts.record_run(
            ["AAPL"], "2026-09-21", dry_run=True, path=None
        )
        if str(result.get("reason") or "").strip():
            self.assertEqual(result.get("attempted", 0), 0)
        else:
            self.assertIn("AAPL", result.get("chart_refusals", {}))


class ContractTests(unittest.TestCase):
    def test_fallback_to_memory_is_off(self):
        self.assertFalse(LIVE_CHART_FALLBACK_TO_MEMORY)

    def test_the_staleness_bound_is_a_live_bound(self):
        self.assertLessEqual(LIVE_CHART_MAX_STALENESS_DAYS, 30)
        self.assertGreaterEqual(LIVE_CHART_MAX_STALENESS_DAYS, 1)


if __name__ == "__main__":
    unittest.main()
