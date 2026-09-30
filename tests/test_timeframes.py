"""Multi-timeframe representation tests (Sprint C1).

The C1 rule, pinned: every timeframe answers for the SAME as_of, and each
answers only from bars that had already CLOSED.

The specific hazard these tests exist for: a provider labels a bar at period
START. The week of 2026-09-14 arrives labelled `2026-09-14`, so at
`as_of = 2026-09-15` a naive `index <= as_of` filter keeps it — and its Close
is Friday's close, which had not happened yet. That is a future-data leak
wearing a past timestamp, and it would silently contaminate every feature C2
builds on top of this.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    TIMEFRAME_MIN_BARS,
    TIMEFRAME_ORDER,
    TIMEFRAME_SPECS,
    TIMEFRAME_TREND_DOWN,
    TIMEFRAME_TREND_FLAT,
    TIMEFRAME_TREND_UP,
)
from core.timeframes import (
    STATUS_INCOMPLETE,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    TimeframeError,
    build_alignment,
    build_timeframe_snapshot,
    build_timeframe_state,
    classify_trend,
    eligible_bars,
    timeframe_snapshot_problems,
    worst_status,
)

AS_OF = "2026-09-15"


def _frame(dates, closes=None):
    index = pd.to_datetime(dates)
    closes = closes if closes is not None else [100.0 + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(index),
        },
        index=index,
    )


class BarEligibilityTests(unittest.TestCase):
    """The leak this sprint exists to prevent."""

    def test_a_still_forming_weekly_bar_is_excluded(self):
        """THE C1 defect: a bar labelled in the past that closes in the future."""
        frame = _frame(["2026-08-31", "2026-09-07", "2026-09-14"], [100.0, 200.0, 999.0])
        eligible, unclosed, future = eligible_bars(frame, AS_OF, "7D")
        self.assertNotIn(999.0, list(eligible["Close"]))
        self.assertEqual(unclosed, 1)
        self.assertEqual(future, 0)

    def test_the_excluded_bar_is_counted_never_silently_dropped(self):
        """A timeframe that quietly shrinks looks identical to one with no data."""
        frame = _frame(["2026-08-31", "2026-09-07", "2026-09-14"])
        _, unclosed, _ = eligible_bars(frame, AS_OF, "7D")
        self.assertGreater(unclosed, 0)

    def test_a_bar_opening_after_as_of_is_future_not_unclosed(self):
        """The two exclusion reasons are different and are reported separately."""
        frame = _frame(["2026-09-07", "2026-09-28"])
        _, unclosed, future = eligible_bars(frame, AS_OF, "7D")
        self.assertEqual(future, 1)
        self.assertEqual(unclosed, 0)

    def test_a_bar_closing_exactly_at_as_of_is_eligible(self):
        """Boundary: closed AT the instant counts as knowable."""
        frame = _frame(["2026-09-08"])  # +7D == 2026-09-15 == as_of
        eligible, unclosed, _ = eligible_bars(frame, AS_OF, "7D")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(unclosed, 0)

    def test_a_partial_trailing_bar_is_excluded(self):
        """Providers append a stub row; a 2026-09-17 row is not a month."""
        frame = _frame(["2026-07-01", "2026-08-01", "2026-09-14"])
        eligible, unclosed, _ = eligible_bars(frame, AS_OF, "31D")
        self.assertEqual(
            [str(i.date()) for i in eligible.index], ["2026-07-01", "2026-08-01"]
        )
        self.assertEqual(unclosed, 1)

    def test_an_empty_frame_is_handled_without_raising(self):
        eligible, unclosed, future = eligible_bars(pd.DataFrame(), AS_OF, "1D")
        self.assertTrue(eligible.empty)
        self.assertEqual((unclosed, future), (0, 0))

    def test_a_coarser_clock_excludes_strictly_more(self):
        """The same bar can be closed daily and unclosed weekly."""
        frame = _frame(["2026-09-14"])
        daily, _, _ = eligible_bars(frame, AS_OF, "1D")
        weekly, _, _ = eligible_bars(frame, AS_OF, "7D")
        self.assertEqual(len(daily), 1)
        self.assertEqual(len(weekly), 0)


class TrendClassificationTests(unittest.TestCase):
    def test_a_rise_beyond_the_band_is_up(self):
        self.assertEqual(classify_trend(100.0, 110.0), TIMEFRAME_TREND_UP)

    def test_a_fall_beyond_the_band_is_down(self):
        self.assertEqual(classify_trend(100.0, 90.0), TIMEFRAME_TREND_DOWN)

    def test_drift_inside_the_band_is_flat(self):
        """Rounding-level movement is not a trend."""
        self.assertEqual(classify_trend(100.0, 100.5), TIMEFRAME_TREND_FLAT)

    def test_a_nonpositive_base_is_flat_not_a_division_error(self):
        self.assertEqual(classify_trend(0.0, 50.0), TIMEFRAME_TREND_FLAT)


class TimeframeStateTests(unittest.TestCase):
    def test_enough_closed_bars_produce_a_trend(self):
        dates = pd.date_range("2026-01-01", periods=40, freq="D")
        state = build_timeframe_state("daily", _frame(dates), AS_OF)
        self.assertEqual(state.status, STATUS_OK)
        self.assertEqual(state.trend, TIMEFRAME_TREND_UP)

    def test_too_few_bars_is_incomplete_with_no_trend(self):
        """A slope fitted to two points is arithmetic, not evidence."""
        state = build_timeframe_state("daily", _frame(["2026-09-01", "2026-09-02"]), AS_OF)
        self.assertEqual(state.status, STATUS_INCOMPLETE)
        self.assertIsNone(state.trend)

    def test_an_incomplete_timeframe_still_reports_the_bars_it_has(self):
        """Withhold the verdict, not the evidence."""
        state = build_timeframe_state("daily", _frame(["2026-09-01", "2026-09-02"]), AS_OF)
        self.assertEqual(state.bar_count, 2)
        self.assertIsNotNone(state.last_close)

    def test_a_failed_fetch_is_unavailable(self):
        state = build_timeframe_state("daily", None, AS_OF, fetch_failed=True)
        self.assertEqual(state.status, STATUS_UNAVAILABLE)
        self.assertIsNone(state.trend)

    def test_a_missing_close_column_is_invalid(self):
        broken = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-09-01"]))
        state = build_timeframe_state("daily", broken, AS_OF)
        self.assertEqual(state.status, STATUS_INVALID)

    def test_only_unclosed_bars_is_incomplete_not_ok(self):
        state = build_timeframe_state("weekly", _frame(["2026-09-14"]), AS_OF)
        self.assertEqual(state.status, STATUS_INCOMPLETE)
        self.assertEqual(state.bar_count, 0)

    def test_an_unknown_timeframe_is_refused(self):
        with self.assertRaises(TimeframeError):
            build_timeframe_state("fortnightly", _frame(["2026-09-01"]), AS_OF)

    def test_yearly_resamples_rather_than_inventing_a_provider_interval(self):
        """W5: one price truth. Yearly is built from the monthly bars."""
        self.assertEqual(TIMEFRAME_SPECS["yearly"]["interval"], "1mo")
        self.assertTrue(TIMEFRAME_SPECS["yearly"]["resample"])
        dates = pd.date_range("2019-01-01", periods=80, freq="MS")
        state = build_timeframe_state("yearly", _frame(dates), AS_OF)
        self.assertEqual(state.status, STATUS_OK)
        # 80 monthly bars from 2019 collapse to far fewer yearly bars.
        self.assertLess(state.bar_count, 12)


class AlignmentTests(unittest.TestCase):
    def _state(self, name, trend, status=STATUS_OK):
        return build_timeframe_state(
            name,
            _frame(pd.date_range("2026-01-01", periods=60, freq="D"),
                   [100.0 + i for i in range(60)] if trend == TIMEFRAME_TREND_UP
                   else [200.0 - i for i in range(60)]),
            AS_OF,
        )

    def test_agreeing_timeframes_are_aligned(self):
        states = {
            "daily": self._state("daily", TIMEFRAME_TREND_UP),
            "weekly": self._state("weekly", TIMEFRAME_TREND_UP),
        }
        alignment = build_alignment(states)
        self.assertTrue(alignment["aligned"])
        self.assertEqual(alignment["dominant_trend"], TIMEFRAME_TREND_UP)

    def test_disagreeing_timeframes_are_conflicted(self):
        states = {
            "daily": self._state("daily", TIMEFRAME_TREND_UP),
            "weekly": self._state("weekly", TIMEFRAME_TREND_DOWN),
        }
        alignment = build_alignment(states)
        self.assertTrue(alignment["conflicted"])
        self.assertIsNone(alignment["dominant_trend"])

    def test_a_timeframe_without_a_trend_does_not_vote(self):
        """An INCOMPLETE clock has no opinion; it must not create agreement."""
        states = {
            "daily": self._state("daily", TIMEFRAME_TREND_UP),
            "weekly": build_timeframe_state("weekly", _frame(["2026-09-14"]), AS_OF),
        }
        alignment = build_alignment(states)
        self.assertIn("weekly", alignment["silent_timeframes"])
        self.assertNotIn("weekly", alignment["trends"])

    def test_a_single_voting_timeframe_is_not_alignment(self):
        """One clock agreeing with itself is not corroboration."""
        states = {"daily": self._state("daily", TIMEFRAME_TREND_UP)}
        self.assertFalse(build_alignment(states)["aligned"])


class SnapshotTests(unittest.TestCase):
    @staticmethod
    def _fetcher(frames):
        def fetch(ticker, period, interval):
            if interval not in frames:
                raise RuntimeError(f"no fixture for {interval}")
            return frames[interval]
        return fetch

    def _healthy_frames(self):
        return {
            "1h": _frame(pd.date_range("2026-09-01", periods=100, freq="h")),
            "1d": _frame(pd.date_range("2026-01-01", periods=200, freq="D")),
            "1wk": _frame(pd.date_range("2025-01-06", periods=80, freq="W-MON")),
            "1mo": _frame(pd.date_range("2018-01-01", periods=90, freq="MS")),
        }

    def test_every_timeframe_answers_for_the_same_as_of(self):
        """C1's actual requirement, stated as a test."""
        snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=self._fetcher(self._healthy_frames()))
        self.assertEqual(set(snapshot["timeframes"]), set(TIMEFRAME_ORDER))
        self.assertEqual(snapshot["as_of"], "2026-09-15 00:00:00")

    def test_a_healthy_snapshot_is_ok_and_contract_clean(self):
        snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=self._fetcher(self._healthy_frames()))
        self.assertEqual(snapshot["status"], STATUS_OK)
        self.assertEqual(timeframe_snapshot_problems(snapshot), [])

    def test_the_snapshot_status_is_the_worst_timeframe(self):
        """A multi-timeframe view is only as good as its weakest clock."""
        frames = self._healthy_frames()
        frames["1wk"] = _frame(["2026-09-14"])  # unclosed only -> INCOMPLETE
        snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=self._fetcher(frames))
        self.assertEqual(snapshot["status"], STATUS_INCOMPLETE)
        self.assertIn("weekly", snapshot["reason"])

    def test_a_provider_failure_degrades_rather_than_raising(self):
        def failing(ticker, period, interval):
            raise RuntimeError("provider down")
        snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=failing)
        self.assertEqual(snapshot["status"], STATUS_UNAVAILABLE)
        for state in snapshot["timeframes"].values():
            self.assertEqual(state["status"], STATUS_UNAVAILABLE)

    def test_the_snapshot_is_deterministic(self):
        fetcher = self._fetcher(self._healthy_frames())
        first = build_timeframe_snapshot("MSFT", AS_OF, fetcher=fetcher)
        second = build_timeframe_snapshot("MSFT", AS_OF, fetcher=fetcher)
        self.assertEqual(first, second)

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(TimeframeError):
            build_timeframe_snapshot("  ", AS_OF)

    def test_the_snapshot_is_versioned(self):
        snapshot = build_timeframe_snapshot("MSFT", AS_OF, fetcher=self._fetcher(self._healthy_frames()))
        self.assertTrue(snapshot["calculation_version"])
        self.assertTrue(snapshot["pipeline_version"])

    def test_an_earlier_as_of_never_sees_more_bars(self):
        """Monotonicity: moving as_of back cannot reveal information."""
        fetcher = self._fetcher(self._healthy_frames())
        late = build_timeframe_snapshot("MSFT", "2026-09-15", fetcher=fetcher)
        early = build_timeframe_snapshot("MSFT", "2026-06-15", fetcher=fetcher)
        for name in TIMEFRAME_ORDER:
            with self.subTest(timeframe=name):
                self.assertLessEqual(
                    early["timeframes"][name]["bar_count"],
                    late["timeframes"][name]["bar_count"],
                )


class ContractValidationTests(unittest.TestCase):
    def test_worst_status_picks_the_most_severe(self):
        states = {
            "a": build_timeframe_state("daily", _frame(pd.date_range("2026-01-01", periods=40)), AS_OF),
            "b": build_timeframe_state("daily", None, AS_OF, fetch_failed=True),
        }
        self.assertEqual(worst_status(states), STATUS_UNAVAILABLE)

    def test_no_states_is_unavailable_not_ok(self):
        self.assertEqual(worst_status({}), STATUS_UNAVAILABLE)

    def test_a_missing_timeframe_is_a_contract_problem(self):
        snapshot = {
            "ticker": "MSFT", "as_of": AS_OF, "status": STATUS_OK,
            "calculation_version": "v1", "timeframes": {"daily": {"status": STATUS_OK, "trend": "up"}},
        }
        problems = timeframe_snapshot_problems(snapshot)
        self.assertTrue(any("weekly" in problem for problem in problems))

    def test_an_ok_timeframe_without_a_trend_is_a_contract_problem(self):
        helper = SnapshotTests()
        snapshot = build_timeframe_snapshot(
            "MSFT", AS_OF, fetcher=helper._fetcher(helper._healthy_frames()),
        )
        snapshot["timeframes"]["daily"]["trend"] = None
        self.assertTrue(any("no trend" in problem for problem in timeframe_snapshot_problems(snapshot)))

    def test_min_bars_covers_every_timeframe(self):
        self.assertEqual(set(TIMEFRAME_MIN_BARS), set(TIMEFRAME_ORDER))


class PartialBucketDisclosureTests(unittest.TestCase):
    """A resampled bucket is LABELLED at period end.

    So the final yearly bucket can be stamped 2025-12-31 while its newest
    source bar is 2025-06-01 — six months of label that is not data. Left
    undisclosed, `last_bar_time` reports a date after the real history and a
    year-over-year comparison puts half a year against full years.
    """

    @staticmethod
    def _monthly(n=101, start="2018-01-01"):
        index = pd.date_range(start, periods=n, freq="MS")
        closes = [100.0 + i for i in range(n)]
        return pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes,
             "Close": closes, "Volume": [1] * n},
            index=index,
        )

    def test_a_partial_final_yearly_bucket_is_flagged(self):
        state = build_timeframe_state("yearly", self._monthly(), "2026-06-15")
        self.assertTrue(state.partial_final_bucket)

    def test_the_true_last_source_bar_is_reported(self):
        state = build_timeframe_state("yearly", self._monthly(), "2026-06-15")
        self.assertIsNotNone(state.last_source_bar)
        self.assertLess(
            pd.Timestamp(state.last_source_bar), pd.Timestamp(state.last_bar_time),
            "a partial bucket must expose a source bar earlier than its label",
        )

    def test_the_partial_bucket_explains_itself(self):
        state = build_timeframe_state("yearly", self._monthly(), "2026-06-15")
        self.assertIn("partial period", state.reason)

    def test_a_non_resampled_timeframe_is_never_flagged_partial(self):
        """Daily bars are not bucketed, so the label IS the bar."""
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        state = build_timeframe_state("daily", _frame(dates), "2026-09-15")
        self.assertFalse(state.partial_final_bucket)
        self.assertIsNone(state.last_source_bar)

    def test_a_complete_final_bucket_is_not_flagged(self):
        """When the source history ends exactly at a year end, nothing is partial."""
        index = pd.date_range("2018-01-01", periods=96, freq="MS")  # through 2025-12
        closes = [100.0 + i for i in range(96)]
        frame = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes,
             "Close": closes, "Volume": [1] * 96},
            index=index,
        )
        state = build_timeframe_state("yearly", frame, "2027-06-15")
        self.assertFalse(state.partial_final_bucket)
