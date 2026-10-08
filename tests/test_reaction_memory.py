"""Chart reaction memory tests (Sprint C7).

What C7 retains for a significant event: the structural picture BEFORE it, and
the 1h / intraday / 1d / 5d / 20d / 60d reaction after — so a later setup can be
matched against history instead of guessed at.

Three rules are pinned here.

**The 1h channel is never invented.** Providers serve about a month of hourly
bars while a 60d reaction needs sixty sessions after the event, so for any event
old enough to have a 60d reaction the hourly data does not exist. `1h` reads
UNAVAILABLE rather than being interpolated from daily bars. A real bug found
during live verification: an event PREDATING the hourly window matched every bar
(`index >= target` is true throughout), so argmax picked bar 0 and reported an
unrelated hour six months later as the reaction.

**The before-picture excludes the event bar.** Including it would let the
reaction describe its own setup, which is how a memory learns to predict the
past.

**Retrieval matches structure first.** Two charts with identical numbers but
different phases are not analogs, and averaging their outcomes yields a base
rate for a situation that never occurred.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    REACTION_HORIZONS,
    REACTION_HOURLY_MAX_ENTRY_GAP_HOURS,
    REACTION_INTRADAY_HORIZONS,
    REACTION_MIN_ANALOGS,
    REACTION_PRE_EVENT_SESSIONS,
    REACTION_REQUIRED_HORIZONS,
    REACTION_STATUS_INCOMPLETE,
    REACTION_STATUS_OK,
    REACTION_STATUS_UNAVAILABLE,
)
from core.reaction_memory import (
    ReactionMemoryError,
    analog_response,
    before_picture,
    before_similarity,
    build_reaction_memory,
    find_reaction_analogs,
    hourly_reaction,
    memory_problems,
)

EVENT = "2026-06-01"


def _daily(n=200, start="2025-09-01", base=100.0, step=0.15):
    closes = []
    for i in range(n):
        leg = (i % 14) / 14
        wave = 7.0 * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(base + wave + step * i)
    index = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.012 for c in closes],
            "Low": [c * 0.988 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n,
        },
        index=index,
    )


def _hourly(n=40, start="2026-06-01 13:30:00"):
    closes = [100.0 + 0.5 * i for i in range(n)]
    index = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes],
            "Close": closes,
            "Volume": [10_000] * n,
        },
        index=index,
    )


def _study(horizons=("intraday", "1d", "5d", "20d", "60d"), value=0.02):
    return {
        "status": "OK",
        "reactions": {
            horizon: {"horizon": horizon, "stock_return": value, "abnormal_return": value / 2}
            for horizon in horizons
        },
    }


class HourlyReactionTests(unittest.TestCase):
    """The channel that must never be fabricated."""

    def test_a_recent_event_measures_an_hourly_reaction(self):
        result = hourly_reaction(_hourly(), EVENT + " 13:30:00")
        self.assertEqual(result["status"], REACTION_STATUS_OK)
        self.assertIsNotNone(result["stock_return"])

    def test_no_hourly_frame_is_unavailable_not_zero(self):
        result = hourly_reaction(None, EVENT)
        self.assertEqual(result["status"], REACTION_STATUS_UNAVAILABLE)
        self.assertIsNone(result["stock_return"])

    def test_an_event_predating_the_hourly_window_is_refused(self):
        """The live bug: every bar satisfies `index >= target`, so argmax picks
        bar 0 and an unrelated hour months later becomes 'the reaction'."""
        result = hourly_reaction(_hourly(start="2026-09-01 13:30:00"), "2026-03-02")
        self.assertEqual(result["status"], REACTION_STATUS_UNAVAILABLE)
        self.assertIn("predates the hourly window", result["reason"])

    def test_the_entry_gap_bound_is_enforced(self):
        inside = hourly_reaction(_hourly(start="2026-06-01 20:00:00"), "2026-06-01 13:30:00")
        self.assertEqual(inside["status"], REACTION_STATUS_OK)
        beyond_hours = REACTION_HOURLY_MAX_ENTRY_GAP_HOURS + 48
        outside = hourly_reaction(
            _hourly(start="2026-06-01 13:30:00"),
            pd.Timestamp("2026-06-01 13:30:00") - pd.Timedelta(hours=beyond_hours),
        )
        self.assertEqual(outside["status"], REACTION_STATUS_UNAVAILABLE)

    def test_an_event_after_the_last_hourly_bar_is_unavailable(self):
        result = hourly_reaction(_hourly(), "2030-01-01")
        self.assertEqual(result["status"], REACTION_STATUS_UNAVAILABLE)

    def test_too_little_history_after_the_event_is_unavailable(self):
        """One bar cannot produce a one-hour forward return."""
        result = hourly_reaction(_hourly(n=1), EVENT + " 13:30:00")
        self.assertEqual(result["status"], REACTION_STATUS_UNAVAILABLE)

    def test_a_payload_without_close_is_unavailable(self):
        broken = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-06-01 13:30:00"]))
        self.assertEqual(hourly_reaction(broken, EVENT)["status"], REACTION_STATUS_UNAVAILABLE)


class BeforePictureTests(unittest.TestCase):
    def test_the_before_picture_is_structural_not_numeric_only(self):
        before = before_picture(_daily(), EVENT)
        self.assertEqual(before["status"], REACTION_STATUS_OK)
        self.assertIn("phase", before)
        self.assertIn("swing_structure", before)

    def test_the_event_bar_is_excluded(self):
        """Including it would let the reaction describe its own setup."""
        before = before_picture(_daily(), EVENT)
        self.assertLess(pd.Timestamp(before["last_bar"]), pd.Timestamp(EVENT))

    def test_the_window_is_bounded(self):
        before = before_picture(_daily(n=400, start="2025-01-01"), EVENT)
        self.assertLessEqual(before["bars"], REACTION_PRE_EVENT_SESSIONS)

    def test_no_prior_bars_is_unavailable(self):
        before = before_picture(_daily(n=20, start="2026-07-01"), EVENT)
        self.assertEqual(before["status"], REACTION_STATUS_UNAVAILABLE)

    def test_a_frame_without_close_is_refused(self):
        with self.assertRaises(ReactionMemoryError):
            before_picture(pd.DataFrame({"Open": [1.0]}), EVENT)


class MemoryAssemblyTests(unittest.TestCase):
    def test_every_declared_horizon_answers(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        for horizon in REACTION_HORIZONS:
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, memory["reactions"])

    def test_e4_reactions_are_consumed_not_recomputed(self):
        """W5: abnormal return stays owned by the event study."""
        memory = build_reaction_memory(
            "TEST", EVENT, _daily(), event_study=_study(value=0.077)
        )
        self.assertEqual(memory["reactions"]["5d"]["stock_return"], 0.077)

    def test_a_missing_intraday_horizon_does_not_degrade_the_memory(self):
        """Expected on an older event, so it is noted rather than penalised."""
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        self.assertEqual(memory["reactions"]["1h"]["status"], REACTION_STATUS_UNAVAILABLE)
        self.assertEqual(memory["status"], REACTION_STATUS_OK)
        self.assertEqual(memory["missing_intraday_horizons"], list(REACTION_INTRADAY_HORIZONS))

    def test_a_missing_required_horizon_makes_the_memory_incomplete(self):
        memory = build_reaction_memory(
            "TEST", EVENT, _daily(), event_study=_study(horizons=("1d", "5d"))
        )
        self.assertEqual(memory["status"], REACTION_STATUS_INCOMPLETE)
        self.assertIn("20d", memory["missing_required_horizons"])

    def test_no_event_study_leaves_every_daily_horizon_unmeasured(self):
        memory = build_reaction_memory("TEST", EVENT, _daily())
        self.assertEqual(memory["status"], REACTION_STATUS_INCOMPLETE)
        for horizon in REACTION_REQUIRED_HORIZONS:
            with self.subTest(horizon=horizon):
                self.assertIsNone(memory["reactions"][horizon]["stock_return"])

    def test_an_hourly_frame_populates_the_1h_channel(self):
        memory = build_reaction_memory(
            "TEST", EVENT + " 13:30:00", _daily(),
            event_study=_study(), hourly_frame=_hourly(),
        )
        self.assertEqual(memory["reactions"]["1h"]["status"], REACTION_STATUS_OK)

    def test_the_memory_is_hashed_and_deterministic(self):
        first = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        second = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        self.assertTrue(first["memory_hash"])
        self.assertEqual(first["memory_hash"], second["memory_hash"])

    def test_a_different_reaction_changes_the_hash(self):
        first = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study(value=0.02))
        second = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study(value=0.09))
        self.assertNotEqual(first["memory_hash"], second["memory_hash"])

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(ReactionMemoryError):
            build_reaction_memory("  ", EVENT, _daily())

    def test_the_memory_is_versioned(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        self.assertTrue(memory["schema_version"])
        self.assertTrue(memory["calculation_version"])


class SimilarityTests(unittest.TestCase):
    """Structure first: same numbers, different phase is not an analog."""

    def test_the_same_phase_and_numbers_scores_high(self):
        picture = {"phase": "breakout", "consolidation_range": 0.05,
                   "volatility_regime_ratio": 1.1, "gap_pct": 0.0}
        self.assertGreater(before_similarity(picture, dict(picture)), 0.95)

    def test_a_different_phase_costs_more_than_numeric_drift(self):
        left = {"phase": "breakout", "consolidation_range": 0.05,
                "volatility_regime_ratio": 1.1, "gap_pct": 0.0}
        same_phase_drifted = {**left, "consolidation_range": 0.07}
        other_phase_identical = {**left, "phase": "failed_breakout"}
        self.assertGreater(
            before_similarity(left, same_phase_drifted),
            before_similarity(left, other_phase_identical),
        )

    def test_a_missing_phase_scores_no_phase_credit(self):
        left = {"phase": "breakout", "consolidation_range": 0.05}
        self.assertLess(before_similarity(left, {"consolidation_range": 0.05}), 1.0)

    def test_an_empty_picture_scores_zero(self):
        self.assertEqual(before_similarity({}, {"phase": "breakout"}), 0.0)


class RetrievalTests(unittest.TestCase):
    def _memory(self, phase, value, status=REACTION_STATUS_OK):
        return {
            "ticker": "T",
            "event_time": EVENT,
            "status": status,
            "before": {
                "status": REACTION_STATUS_OK,
                "phase": phase,
                "consolidation_range": 0.05,
                "volatility_regime_ratio": 1.1,
                "gap_pct": 0.0,
            },
            "reactions": {
                "20d": {"status": REACTION_STATUS_OK, "stock_return": value},
            },
        }

    def test_matching_setups_are_retrieved(self):
        setup = {"phase": "breakout", "consolidation_range": 0.05,
                 "volatility_regime_ratio": 1.1, "gap_pct": 0.0}
        memories = [self._memory("breakout", 0.05), self._memory("consolidation", -0.03)]
        analogs = find_reaction_analogs(setup, memories)
        self.assertEqual(len(analogs), 1)
        self.assertEqual(analogs[0]["before"]["phase"], "breakout")

    def test_analogs_are_ordered_by_similarity(self):
        setup = {"phase": "breakout", "consolidation_range": 0.05,
                 "volatility_regime_ratio": 1.1, "gap_pct": 0.0}
        close = self._memory("breakout", 0.05)
        looser = self._memory("breakout", 0.02)
        looser["before"]["consolidation_range"] = 0.09
        analogs = find_reaction_analogs(setup, [looser, close])
        self.assertGreaterEqual(analogs[0]["similarity"], analogs[1]["similarity"])

    def test_a_memory_without_a_before_picture_cannot_match(self):
        setup = {"phase": "breakout", "consolidation_range": 0.05}
        blind = self._memory("breakout", 0.05)
        blind["before"]["status"] = REACTION_STATUS_UNAVAILABLE
        self.assertEqual(find_reaction_analogs(setup, [blind]), [])

    def test_no_memories_yields_no_analogs(self):
        self.assertEqual(find_reaction_analogs({"phase": "breakout"}, []), [])


class AnalogResponseTests(unittest.TestCase):
    def _analog(self, value):
        return {"reactions": {"20d": {"status": REACTION_STATUS_OK, "stock_return": value}}}

    def test_enough_analogs_produce_a_median(self):
        analogs = [self._analog(v) for v in (0.02, 0.05, 0.08, -0.01)]
        response = analog_response(analogs, "20d")
        self.assertTrue(response["sufficient"])
        self.assertIsNotNone(response["median_response"])

    def test_too_few_analogs_refuse_a_median(self):
        """Two observations are an anecdote, not a base rate."""
        analogs = [self._analog(0.02)] * (REACTION_MIN_ANALOGS - 1)
        response = analog_response(analogs, "20d")
        self.assertFalse(response["sufficient"])
        self.assertIsNone(response["median_response"])
        self.assertIn("anecdote", response["reason"])

    def test_the_share_positive_is_reported(self):
        analogs = [self._analog(v) for v in (0.02, 0.05, -0.08, -0.01)]
        self.assertAlmostEqual(analog_response(analogs, "20d")["share_positive"], 0.5)

    def test_unmeasured_reactions_do_not_count(self):
        analogs = [self._analog(0.02), self._analog(0.05), self._analog(0.08)]
        analogs.append({"reactions": {"20d": {"status": REACTION_STATUS_UNAVAILABLE,
                                              "stock_return": None}}})
        self.assertEqual(analog_response(analogs, "20d")["observations"], 3)

    def test_an_unknown_horizon_is_refused(self):
        with self.assertRaises(ReactionMemoryError):
            analog_response([], "7d")


class ContractValidationTests(unittest.TestCase):
    def test_a_healthy_memory_is_contract_clean(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        self.assertEqual(memory_problems(memory), [])

    def test_an_ok_horizon_without_a_return_is_caught(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        memory["reactions"]["5d"]["stock_return"] = None
        self.assertTrue(any("no return" in p for p in memory_problems(memory)))

    def test_an_unavailable_horizon_reporting_a_return_is_caught(self):
        """The fabricated-value shape this module exists to prevent."""
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        memory["reactions"]["1h"]["stock_return"] = 0.01
        self.assertTrue(any("still reports a return" in p for p in memory_problems(memory)))

    def test_an_ok_memory_missing_a_required_horizon_is_caught(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        memory["reactions"]["20d"]["status"] = REACTION_STATUS_UNAVAILABLE
        memory["reactions"]["20d"]["stock_return"] = None
        self.assertTrue(any("required horizon" in p for p in memory_problems(memory)))

    def test_a_missing_horizon_key_is_caught(self):
        memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())
        del memory["reactions"]["60d"]
        self.assertTrue(any("60d" in p for p in memory_problems(memory)))
