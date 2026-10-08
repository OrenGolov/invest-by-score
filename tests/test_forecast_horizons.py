"""Forecast horizon tests (Sprint F2).

F2 fixes the horizon set: 1d, 5d, 20d, 60d, 120d, 252d. What these tests pin is
that the set cannot quietly narrow, and that adding a long horizon does not
silently break the machinery that depends on it.

**Adding 252d moved two things nobody would have guessed**, and both are now
derived rather than hardcoded:

- the label builder's calendar coverage was 130 days, sized for a 60-session
  maximum. A 252-session window spans about a year, so long labels would have
  reported pending forever because their future bars were never fetched;
- the backtest embargo must cover the longest horizon or a validation fold sees
  bars that shaped a training row's outcome.

**Pending is not missing.** A 252d forecast made last month is unfinished, not
wrong, and conflating the two would make every young long-horizon forecast look
like a failure.
"""

from __future__ import annotations

import unittest

from core import config as core_config
from core.config import (
    FORECAST_HORIZONS,
    FORECAST_LONG_HORIZON_SESSIONS,
    FORECAST_REQUIRED_HORIZONS,
    LABEL_HORIZON_SESSIONS,
)
from core.forecast_horizons import (
    HORIZON_STATUS_PENDING,
    HORIZON_STATUS_SCORABLE,
    HORIZON_STATUS_UNAVAILABLE,
    ForecastHorizonError,
    horizon_calendar_days,
    horizon_contract,
    horizon_problems,
    horizon_readiness,
    horizon_sessions,
    is_long_horizon,
    known_horizons,
    readiness_report,
    required_history_sessions,
)


def _labels(matured=(), pending=()):
    return {"matured_horizons": list(matured), "pending_horizons": list(pending)}


class HorizonSetTests(unittest.TestCase):
    def test_every_f2_horizon_is_declared(self):
        for horizon in ("1d", "5d", "20d", "60d", "120d", "252d"):
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, known_horizons())

    def test_horizons_are_ordered_shortest_first(self):
        """A joint forecast (F3) reads in time order, and '120d' sorts before
        '1d' alphabetically — so ordering must be by sessions, not by name."""
        sessions = [horizon_sessions(name) for name in known_horizons()]
        self.assertEqual(sessions, sorted(sessions))

    def test_every_forecast_horizon_has_a_label_horizon(self):
        """A horizon with no label could never be scored against an outcome."""
        for horizon in known_horizons():
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, LABEL_HORIZON_SESSIONS)

    def test_an_unknown_horizon_is_refused_not_defaulted(self):
        with self.assertRaises(ForecastHorizonError):
            horizon_sessions("90d")

    def test_session_counts_match_the_names(self):
        self.assertEqual(horizon_sessions("1d"), 1)
        self.assertEqual(horizon_sessions("252d"), 252)

    def test_calendar_days_scale_with_sessions(self):
        """252 trading sessions is about a year, not 252 days."""
        self.assertGreater(horizon_calendar_days("252d"), 300)
        self.assertLess(horizon_calendar_days("5d"), 15)

    def test_long_horizons_are_flagged(self):
        self.assertTrue(is_long_horizon("120d"))
        self.assertTrue(is_long_horizon("252d"))
        self.assertFalse(is_long_horizon("20d"))

    def test_the_long_threshold_is_declared(self):
        self.assertGreater(FORECAST_LONG_HORIZON_SESSIONS, 0)


class DerivedConfigTests(unittest.TestCase):
    """The couplings that adding 252d exposed."""

    def test_calendar_coverage_reaches_past_the_longest_horizon(self):
        """It was 130 days against a 60-session max; a 252-session horizon
        spans about a year, so a fixed 130 would never fetch its future bars."""
        longest = max(LABEL_HORIZON_SESSIONS.values())
        self.assertGreaterEqual(core_config.LABEL_CALENDAR_COVERAGE_DAYS, longest)

    def test_the_embargo_covers_the_longest_horizon(self):
        """Otherwise a validation fold sees bars that shaped a training row."""
        self.assertGreaterEqual(
            core_config.BACKTEST_EMBARGO_SESSIONS,
            max(LABEL_HORIZON_SESSIONS.values()),
        )

    def test_folds_are_at_least_as_wide_as_the_embargo(self):
        self.assertGreaterEqual(
            core_config.BACKTEST_FOLD_SESSIONS, core_config.BACKTEST_EMBARGO_SESSIONS
        )

    def test_required_history_includes_the_forward_window(self):
        """The part people forget: a 252d label needs a year of bars AFTER the
        prediction time, so a dataset that merely reaches as_of cannot score it."""
        self.assertEqual(required_history_sessions("252d", warmup_sessions=70), 322)


class ReadinessTests(unittest.TestCase):
    def test_a_matured_horizon_is_scorable(self):
        readiness = horizon_readiness(_labels(matured=["20d"]), "20d")
        self.assertEqual(readiness["status"], HORIZON_STATUS_SCORABLE)
        self.assertEqual(readiness["reason"], "")

    def test_a_pending_horizon_explains_the_wait(self):
        """Unfinished is not wrong, and the difference must be visible."""
        readiness = horizon_readiness(_labels(pending=["252d"]), "252d")
        self.assertEqual(readiness["status"], HORIZON_STATUS_PENDING)
        self.assertIn("has not closed", readiness["reason"])

    def test_an_absent_horizon_is_unavailable_not_pending(self):
        readiness = horizon_readiness(_labels(), "20d")
        self.assertEqual(readiness["status"], HORIZON_STATUS_UNAVAILABLE)
        self.assertTrue(readiness["reason"])

    def test_a_report_covers_every_declared_horizon(self):
        report = readiness_report(_labels(matured=["1d"], pending=["5d"]))
        self.assertEqual(set(report["horizons"]), set(FORECAST_HORIZONS))

    def test_report_lists_are_in_time_order(self):
        """'120d' must not sort before '1d'."""
        report = readiness_report(_labels(matured=list(FORECAST_HORIZONS)))
        self.assertEqual(report["scorable"], list(FORECAST_HORIZONS))

    def test_a_report_separates_scorable_from_pending(self):
        report = readiness_report(
            _labels(matured=["1d", "5d", "20d", "60d"], pending=["120d", "252d"])
        )
        self.assertEqual(report["scorable"], ["1d", "5d", "20d", "60d"])
        self.assertEqual(report["pending"], ["120d", "252d"])

    def test_a_healthy_report_is_contract_clean(self):
        report = readiness_report(_labels(matured=list(FORECAST_HORIZONS)))
        self.assertEqual(horizon_problems(report), [])

    def test_a_report_missing_a_required_horizon_is_caught(self):
        report = readiness_report(_labels(matured=list(FORECAST_HORIZONS)))
        del report["horizons"]["252d"]
        self.assertTrue(any("252d" in p for p in horizon_problems(report)))

    def test_a_non_scorable_horizon_without_a_reason_is_caught(self):
        report = readiness_report(_labels(pending=["252d"]))
        report["horizons"]["252d"]["reason"] = ""
        self.assertTrue(any("explain" in p for p in horizon_problems(report)))

    def test_long_horizons_are_reported(self):
        report = readiness_report(_labels(matured=list(FORECAST_HORIZONS)))
        self.assertIn("252d", report["long_horizons"])


class LiveLabelTests(unittest.TestCase):
    """Against the real V1 label builder."""

    def test_long_horizons_mature_on_deep_history(self):
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2024-06-15")
        if labels.get("status") not in ("OK", "PARTIAL"):
            self.skipTest("label builder unavailable in this environment")
        report = readiness_report(labels)
        for horizon in FORECAST_REQUIRED_HORIZONS:
            with self.subTest(horizon=horizon):
                self.assertIn(horizon, report["scorable"])

    def test_a_recent_as_of_leaves_long_horizons_pending(self):
        """Not a failure — the window simply has not closed."""
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2026-06-15")
        if labels.get("status") not in ("OK", "PARTIAL"):
            self.skipTest("label builder unavailable in this environment")
        report = readiness_report(labels)
        self.assertIn("252d", report["pending"])
        self.assertIn("20d", report["scorable"])
