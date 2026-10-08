"""Sprint V1: leakage-safe outcome labels.

Hermetic: the price fetch is patched with synthetic frames (no network, no
wall-clock dependence in the label math). Coverage:

- the spec acceptance: a decision dated 10 sessions ago has 1d/5d labels and
  20d/60d = null (PARTIAL) — partially elapsed horizons are never emitted;
- off-by-one safety at every horizon boundary (5 bars after as_of matures
  5d; 4 does not);
- leakage safety: bars beyond a horizon's window cannot change that
  horizon's label; bars inside do; pre-as_of bars other than the entry
  close cannot either;
- intraday as_of handling (window starts strictly after the as_of moment);
- label values: forward returns, anchored realized vol, risk-adjusted
  outcome (None for 1d / zero-vol windows), adverse excursion (lows vs
  entry close, window-bounded), label_20d_up at the configured threshold;
- eligibility from the data's latest bar, never wall-clock;
- versioning (OUTCOME_LABEL_VERSION stamped everywhere), record/labels
  hashes, determinism;
- the append-only outcomes store: idempotent recompute, supersede-on-change,
  pending horizons never persisted, latest-per-horizon resolution,
  integrity on malformed lines.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core import config as core_config
from core.labels import (
    append_outcome_records,
    build_outcome_labels,
    latest_outcome_labels,
    load_outcome_records,
)
from fetch_data import TickerFetchError


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


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * index for index in range(sessions)]


def _as_of(frame, position):
    """as_of string for the bar at `position` (supports negative indexing)."""
    return frame.index[position].strftime("%Y-%m-%d %H:%M:%S")


def _labels_for(frame, as_of):
    with patch("core.labels.fetch_price_history", return_value=frame):
        return build_outcome_labels("TEST", as_of)


# Derived: a fixture must be long enough to MATURE the longest horizon, plus
# room for an entry bar. A fixed 130 sessions worked only while 60d was the
# maximum; F2's 252d horizon made every such fixture silently unable to mature.
_LONGEST = max(core_config.LABEL_HORIZON_SESSIONS.values())
_FIXTURE_SESSIONS = _LONGEST + 70


class ForwardReturnBoundaryTests(unittest.TestCase):
    """Spec acceptance: matured vs null at exact horizon boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.frame = _frame(_ramp(_FIXTURE_SESSIONS))

    def test_all_horizons_mature_when_sixty_sessions_exist(self):
        frame = self.frame
        # Enough forward sessions for the LONGEST horizon, derived: a fixed -61
        # matured everything only while 60d was the maximum.
        longest = max(core_config.LABEL_HORIZON_SESSIONS.values())
        as_of = _as_of(frame, -(longest + 1))
        labels = _labels_for(frame, as_of)
        self.assertEqual(labels["status"], "OK")
        self.assertEqual(
            sorted(labels["matured_horizons"]),
            sorted(core_config.LABEL_HORIZON_SESSIONS),
        )
        self.assertEqual(labels["pending_horizons"], [])
        entry_position = len(frame) - (longest + 1)
        entry_close = float(frame["Close"].iloc[entry_position])
        self.assertEqual(labels["entry_close"], round(entry_close, 4))
        self.assertEqual(labels["entry_bar"], _as_of(frame, entry_position))
        for horizon, sessions in core_config.LABEL_HORIZON_SESSIONS.items():
            expected = float(frame["Close"].iloc[entry_position + sessions]) / entry_close - 1.0
            with self.subTest(horizon=horizon):
                self.assertAlmostEqual(
                    labels["horizons"][horizon]["forward_return"], expected, places=6
                )
                self.assertEqual(
                    labels["horizons"][horizon]["exit_bar"],
                    _as_of(frame, entry_position + sessions),
                )

    def test_spec_acceptance_ten_sessions_ago(self):
        frame = self.frame
        as_of = _as_of(frame, -11)  # 10 sessions after as_of
        labels = _labels_for(frame, as_of)
        self.assertEqual(labels["status"], "PARTIAL")
        self.assertEqual(labels["matured_horizons"], ["1d", "5d"])
        expected_pending = [
            name for name, sessions in core_config.LABEL_HORIZON_SESSIONS.items()
            if sessions > 10
        ]
        self.assertEqual(sorted(labels["pending_horizons"]), sorted(expected_pending))
        for name in expected_pending:
            self.assertIsNone(labels["horizons"][name], name)
        entry_position = len(frame) - 11
        expected_5d = float(frame["Close"].iloc[entry_position + 5]) / float(frame["Close"].iloc[entry_position]) - 1.0
        self.assertAlmostEqual(labels["horizons"]["5d"]["forward_return"], expected_5d, places=6)
        self.assertIn("no partial-window leakage", labels["reason"])

    def test_off_by_one_safety_at_the_five_day_boundary(self):
        frame = self.frame
        matured_at_five = _labels_for(frame, _as_of(frame, -6))   # 5 bars after
        pending_at_four = _labels_for(frame, _as_of(frame, -5))   # 4 bars after
        self.assertIn("5d", matured_at_five["matured_horizons"])
        self.assertIsNotNone(matured_at_five["horizons"]["5d"])
        self.assertIn("5d", pending_at_four["pending_horizons"])
        self.assertIsNone(pending_at_four["horizons"]["5d"])
        self.assertEqual(pending_at_four["status"], "PARTIAL")

    def test_one_day_label_uses_the_next_session_not_the_entry(self):
        frame = self.frame
        as_of = _as_of(frame, -11)
        labels = _labels_for(frame, as_of)
        entry_position = len(frame) - 11
        self.assertEqual(labels["horizons"]["1d"]["exit_bar"], _as_of(frame, entry_position + 1))
        # A zero-day window (entry vs entry) would read 0.0; it must not.
        self.assertNotEqual(labels["horizons"]["1d"]["forward_return"], 0.0)

    def test_intraday_as_of_starts_the_window_after_the_moment(self):
        frame = self.frame
        intraday_as_of = frame.index[100].strftime("%Y-%m-%d") + " 12:00:00"
        labels = _labels_for(frame, intraday_as_of)
        self.assertEqual(labels["entry_bar"], _as_of(frame, 100))
        self.assertEqual(labels["horizons"]["1d"]["exit_bar"], _as_of(frame, 101))

    def test_eligibility_comes_from_the_data_not_the_clock(self):
        # A frozen historical frame: the same dataset state must produce the
        # same labels on every run (no wall-clock in the label math).
        frame = self.frame
        as_of = _as_of(frame, -61)
        self.assertEqual(_labels_for(frame, as_of), _labels_for(frame, as_of))


class LeakageSafetyTests(unittest.TestCase):
    """The defining property: labels use ONLY bars in (as_of, as_of + h]."""

    @classmethod
    def setUpClass(cls):
        cls.frame = _frame(_ramp(_FIXTURE_SESSIONS))
        cls.as_of = _as_of(cls.frame, -(_LONGEST + 1))  # every horizon matured
        cls.entry_position = len(cls.frame) - (_LONGEST + 1)

    def test_bars_beyond_a_horizon_cannot_change_its_label(self):
        poisoned = self.frame.copy()
        # Tenfold everything from entry+6 onward: inside the 20d/60d windows,
        # strictly outside the 1d/5d windows.
        poison_from = self.entry_position + 6
        for column in ("Open", "High", "Low", "Close"):
            poisoned.loc[poisoned.index[poison_from:], column] *= 10.0
        baseline = _labels_for(self.frame, self.as_of)
        poisoned_labels = _labels_for(poisoned, self.as_of)
        for horizon in ("1d", "5d"):
            with self.subTest(horizon=horizon):
                self.assertEqual(
                    poisoned_labels["horizons"][horizon],
                    baseline["horizons"][horizon],
                    f"bars beyond the {horizon} window leaked into its label",
                )
        for horizon in ("20d", "60d"):
            with self.subTest(horizon=horizon):
                self.assertNotEqual(
                    poisoned_labels["horizons"][horizon]["forward_return"],
                    baseline["horizons"][horizon]["forward_return"],
                    f"the {horizon} window should have seen the poisoned bars",
                )

    def test_bars_inside_the_window_do_change_the_label(self):
        # The 5d forward return is exit-close vs entry-close, so poisoning an
        # INTERIOR bar cannot move it — but it must move the window's
        # realized volatility, and poisoning the EXIT bar must move the
        # return. Both are the leakage-sensitive quantities.
        interior = self.frame.copy()
        for column in ("Open", "High", "Low", "Close"):
            interior.loc[interior.index[self.entry_position + 3], column] *= 2.0
        baseline = _labels_for(self.frame, self.as_of)
        interior_labels = _labels_for(interior, self.as_of)
        self.assertNotEqual(
            interior_labels["horizons"]["5d"]["realized_vol"],
            baseline["horizons"]["5d"]["realized_vol"],
            "an interior window bar must change realized vol",
        )
        self.assertEqual(
            interior_labels["horizons"]["5d"]["forward_return"],
            baseline["horizons"]["5d"]["forward_return"],
            "interior bars cannot move an exit-close-based return",
        )
        exit_poisoned = self.frame.copy()
        for column in ("Open", "High", "Low", "Close"):
            exit_poisoned.loc[exit_poisoned.index[self.entry_position + 5], column] *= 2.0
        exit_labels = _labels_for(exit_poisoned, self.as_of)
        self.assertNotEqual(
            exit_labels["horizons"]["5d"]["forward_return"],
            baseline["horizons"]["5d"]["forward_return"],
            "poisoning the exit bar must change the forward return",
        )

    def test_pre_as_of_bars_other_than_the_entry_cannot_change_labels(self):
        # The label's only pre-as_of input is the entry close. Rewriting
        # ancient history (position 5) must leave every label untouched.
        poisoned = self.frame.copy()
        for column in ("Open", "High", "Low", "Close"):
            poisoned.loc[poisoned.index[5], column] = 3.0
        baseline = _labels_for(self.frame, self.as_of)
        poisoned_labels = _labels_for(poisoned, self.as_of)
        self.assertEqual(baseline["horizons"], poisoned_labels["horizons"])
        self.assertEqual(baseline["labels_hash"], poisoned_labels["labels_hash"])

    def test_realized_vol_window_is_also_bounded(self):
        poisoned = self.frame.copy()
        poison_from = self.entry_position + 6
        for column in ("Open", "High", "Low", "Close"):
            poisoned.loc[poisoned.index[poison_from:], column] *= 10.0
        baseline = _labels_for(self.frame, self.as_of)
        poisoned_labels = _labels_for(poisoned, self.as_of)
        self.assertEqual(
            poisoned_labels["horizons"]["5d"]["realized_vol"],
            baseline["horizons"]["5d"]["realized_vol"],
        )


class LabelValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = _frame(_ramp(_FIXTURE_SESSIONS))
        cls.as_of = _as_of(cls.frame, -(_LONGEST + 1))

    def test_label_20d_up_follows_the_configured_threshold(self):
        labels = _labels_for(self.frame, self.as_of)
        record = labels["horizons"]["20d"]
        self.assertEqual(
            record["label_up"],
            record["forward_return"] > core_config.OUTCOME_LABEL_UP_THRESHOLD,
        )
        self.assertTrue(record["label_up"])  # rising ramp

        declining = _frame(_ramp(_FIXTURE_SESSIONS, step=-1.0, base=400.0))
        down = _labels_for(declining, _as_of(declining, -61))
        self.assertFalse(down["horizons"]["20d"]["label_up"])

    def test_adverse_excursion_is_the_worst_low_vs_entry_close_in_the_window(self):
        frame = self.frame.copy()
        entry_position = len(frame) - (_LONGEST + 1)
        entry_close = float(frame["Close"].iloc[entry_position])
        # Deep dip 10 sessions into the 20d window...
        dip_position = entry_position + 10
        frame.loc[frame.index[dip_position], "Low"] = entry_close * 0.90
        labels = _labels_for(frame, self.as_of)
        self.assertAlmostEqual(labels["horizons"]["20d"]["adverse_excursion"], -0.10, places=6)
        # ...and a deeper dip BEYOND the 20d window must not move it.
        beyond = frame.copy()
        beyond.loc[beyond.index[entry_position + 40], "Low"] = entry_close * 0.50
        labels_beyond = _labels_for(beyond, self.as_of)
        self.assertAlmostEqual(
            labels_beyond["horizons"]["20d"]["adverse_excursion"], -0.10, places=6
        )

    def test_risk_adjusted_outcome_is_return_over_realized_vol(self):
        # A well-conditioned frame: the perfect ramp's vol sits at the 6dp
        # rounding floor, so this test oscillates around a gentle trend.
        oscillating = _frame([
            200.0 + 0.5 * index + (2.0 if index % 2 else -2.0)
            for index in range(130)
        ])
        as_of = _as_of(oscillating, -61)
        labels = _labels_for(oscillating, as_of)
        record = labels["horizons"]["5d"]
        entry_position = len(oscillating) - 61
        closes = [float(oscillating["Close"].iloc[entry_position + i]) for i in range(6)]
        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        expected_vol = float(pd.Series(returns).std(ddof=0))
        self.assertGreater(expected_vol, 0.001)  # away from the rounding floor
        self.assertAlmostEqual(record["realized_vol"], expected_vol, places=6)
        # The published block is self-consistent: the ratio is recomputable
        # from the published numerator and denominator alone.
        self.assertEqual(
            record["risk_adjusted"],
            round(record["forward_return"] / record["realized_vol"], 4),
        )
        self.assertAlmostEqual(
            record["risk_adjusted"], record["forward_return"] / expected_vol, places=3
        )
        # 1d has a single return: dispersion undefined -> None, never a fake.
        self.assertIsNone(labels["horizons"]["1d"]["risk_adjusted"])

    def test_zero_volatility_window_yields_none_not_infinity(self):
        flat = _frame([100.0] * 130)
        labels = _labels_for(flat, _as_of(flat, -61))
        record = labels["horizons"]["5d"]
        self.assertEqual(record["forward_return"], 0.0)
        self.assertEqual(record["realized_vol"], 0.0)
        self.assertIsNone(record["risk_adjusted"])
        self.assertFalse(labels["horizons"]["20d"]["label_up"])  # 0.0 is not > 0.0

    def test_version_stamped_everywhere(self):
        labels = _labels_for(self.frame, self.as_of)
        self.assertEqual(labels["label_version"], core_config.OUTCOME_LABEL_VERSION)
        for name, record in labels["horizons"].items():
            # An unmatured horizon is explicitly None, not a stub record.
            if record is None:
                self.assertIn(name, labels["pending_horizons"])
                continue
            self.assertEqual(record["label_version"], core_config.OUTCOME_LABEL_VERSION)
        self.assertEqual(len(labels["horizons"]["1d"]["record_hash"]), 64)
        self.assertEqual(len(labels["labels_hash"]), 64)


class FailureStateTests(unittest.TestCase):
    """Fail-closed statuses: never a fabricated label."""

    def test_future_as_of_raises_like_the_market_agent(self):
        with self.assertRaises(ValueError):
            with patch("core.labels.fetch_price_history", return_value=_frame(_ramp(_FIXTURE_SESSIONS))):
                build_outcome_labels("TEST", "2099-01-01")

    def test_fetch_failure_is_unavailable(self):
        with patch(
            "core.labels.fetch_price_history",
            side_effect=TickerFetchError("TEST: network request failed"),
        ):
            labels = build_outcome_labels("TEST", "2024-01-02")
        self.assertEqual(labels["status"], "UNAVAILABLE")
        self.assertEqual(labels["horizons"], {})
        self.assertEqual(labels["labels_hash"], "")

    def test_empty_frame_is_unavailable(self):
        with patch("core.labels.fetch_price_history", return_value=pd.DataFrame()):
            labels = build_outcome_labels("TEST", "2024-01-02")
        self.assertEqual(labels["status"], "UNAVAILABLE")

    def test_as_of_before_first_bar_is_unavailable(self):
        frame = _frame(_ramp(_FIXTURE_SESSIONS))
        labels = _labels_for(frame, "2021-01-01")
        self.assertEqual(labels["status"], "UNAVAILABLE")
        self.assertIn("No bars at or before", labels["reason"])

    def test_pending_decision_at_the_latest_bar(self):
        frame = _frame(_ramp(_FIXTURE_SESSIONS))
        labels = _labels_for(frame, _as_of(frame, -1))
        self.assertEqual(labels["status"], "PENDING")
        self.assertEqual(labels["matured_horizons"], [])
        self.assertEqual(
            len(labels["pending_horizons"]), len(core_config.LABEL_HORIZON_SESSIONS)
        )
        self.assertEqual(labels["labels_hash"], "")


class OutcomeStorageTests(unittest.TestCase):
    """Append-only outcomes.jsonl: idempotent, supersedes, latest-wins."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "outcomes.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _record(self, horizon="5d", forward_return=0.05, ticker="TEST", as_of="2024-01-02 00:00:00"):
        from core.labels import _record_hash

        record = {
            "ticker": ticker,
            "as_of": as_of,
            "horizon": horizon,
            "label_version": core_config.OUTCOME_LABEL_VERSION,
            "status": "OK",
            "entry_close": 100.0,
            "forward_return": forward_return,
            "window_sessions": int(horizon.rstrip("d")),
        }
        record["record_hash"] = _record_hash(record)
        return record

    def test_append_and_load_roundtrip(self):
        records = [self._record(h) for h in ("1d", "5d", "20d", "60d")]
        appended = append_outcome_records(records, path=self.path)
        self.assertEqual(appended, 4)
        loaded = load_outcome_records(ticker="TEST", as_of="2024-01-02", path=self.path)
        self.assertEqual(len(loaded), 4)
        horizons = {r["horizon"] for r in loaded}
        self.assertEqual(horizons, {"1d", "5d", "20d", "60d"})

    def test_recompute_is_idempotent(self):
        records = [self._record(h) for h in ("1d", "5d")]
        append_outcome_records(records, path=self.path)
        self.assertEqual(append_outcome_records(records, path=self.path), 0)
        self.assertEqual(len(load_outcome_records(path=self.path)), 2)

    def test_changed_value_appends_with_supersedes(self):
        original = self._record("5d", forward_return=0.05)
        append_outcome_records([original], path=self.path)
        revised = self._record("5d", forward_return=0.07)
        self.assertEqual(append_outcome_records([revised], path=self.path), 1)
        loaded = load_outcome_records(horizon="5d", path=self.path)
        self.assertEqual(len(loaded), 2)  # history preserved, never mutated
        self.assertEqual(loaded[-1]["supersedes"], original["record_hash"])
        self.assertEqual(loaded[-1]["forward_return"], 0.07)
        latest = latest_outcome_labels("TEST", "2024-01-02", path=self.path)
        self.assertEqual(latest["horizons"]["5d"]["forward_return"], 0.07)

    def test_pending_records_are_never_persisted(self):
        pending = self._record("20d")
        pending["status"] = "PARTIAL"
        self.assertEqual(append_outcome_records([pending], path=self.path), 0)
        self.assertEqual(load_outcome_records(path=self.path), [])

    def test_latest_resolution_statuses(self):
        records = [self._record(h) for h in ("1d", "5d")]
        append_outcome_records(records, path=self.path)
        resolved = latest_outcome_labels("TEST", "2024-01-02", path=self.path)
        self.assertEqual(resolved["status"], "PARTIAL")
        self.assertEqual(resolved["count"], 2)
        self.assertEqual(
            sorted(resolved["pending_horizons"]),
            sorted(set(core_config.LABEL_HORIZON_SESSIONS) - {"1d", "5d"}),
        )
        empty = latest_outcome_labels("TEST", "2030-01-01", path=self.path)
        self.assertEqual(empty["status"], "PENDING")

    def test_key_isolation_between_tickers_and_versions(self):
        other_ticker = self._record("5d", ticker="OTHER")
        other_version = self._record("5d")
        other_version["label_version"] = "outcome-label-v2"
        from core.labels import _record_hash

        other_version["record_hash"] = _record_hash(other_version)
        append_outcome_records([other_ticker, other_version], path=self.path)
        self.assertEqual(len(load_outcome_records(ticker="OTHER", path=self.path)), 1)
        # The TEST-ticker record only exists under the v2 label version.
        self.assertEqual(
            len(load_outcome_records(ticker="TEST", label_version="outcome-label-v1", path=self.path)),
            0,
        )
        self.assertEqual(
            len(load_outcome_records(ticker="TEST", label_version="outcome-label-v2", path=self.path)),
            1,
        )

    def test_malformed_line_violates_integrity_loudly(self):
        records = [self._record("5d")]
        append_outcome_records(records, path=self.path)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with self.assertRaises(ValueError):
            load_outcome_records(path=self.path)

    def test_build_and_persist_roundtrip(self):
        frame = _frame(_ramp(_FIXTURE_SESSIONS))
        as_of = _as_of(frame, -61)
        with patch("core.labels.fetch_price_history", return_value=frame):
            first = build_outcome_labels("TEST", as_of)
        from core.labels import build_and_persist_outcome_labels

        with patch("core.labels.fetch_price_history", return_value=frame):
            persisted = build_and_persist_outcome_labels("TEST", as_of, path=self.path)
        self.assertEqual(persisted["persisted"], 4)
        with patch("core.labels.fetch_price_history", return_value=frame):
            again = build_and_persist_outcome_labels("TEST", as_of, path=self.path)
        self.assertEqual(again["persisted"], 0)  # idempotent recompute
        self.assertEqual(first["labels_hash"], persisted["labels_hash"])


if __name__ == "__main__":
    unittest.main()


class AdverseExcursionFloorTests(unittest.TestCase):
    """v2: a drawdown cannot be positive.

    `adverse_excursion` answers "worst DRAWDOWN within the horizon", and a
    drawdown is a loss from entry. On a gap-up every low sits ABOVE the entry
    close, so an unfloored `min()` returned the smallest GAIN — a different
    quantity wearing the same name, and one that violated the F1 bound
    (-1.0, 0.0) the target is scored against. Measured: 9 of 300 real 20d
    labels on live market data.
    """

    @staticmethod
    def _gap_up_frame():
        """Every bar after entry trades strictly above the entry close."""
        closes = [100.0] * 30 + [112.0 + i * 0.1 for i in range(_LONGEST + 40)]
        frame = _frame(closes)
        # Lows above the entry close: the price never traded back down.
        frame["Low"] = [c * 0.999 for c in closes]
        return frame

    def test_a_gap_up_reports_zero_drawdown_not_a_gain(self):
        frame = self._gap_up_frame()
        labels = _labels_for(frame, _as_of(frame, -(_LONGEST + 1)))
        excursion = labels["horizons"]["20d"]["adverse_excursion"]
        self.assertLessEqual(excursion, 0.0)

    def test_the_excursion_never_violates_its_declared_bound(self):
        from core.forecast_targets import bounds_problems

        frame = self._gap_up_frame()
        labels = _labels_for(frame, _as_of(frame, -(_LONGEST + 1)))
        excursion = labels["horizons"]["20d"]["adverse_excursion"]
        self.assertEqual(bounds_problems("adverse_excursion", excursion), [])

    def test_a_real_drawdown_is_still_measured(self):
        """The floor must not swallow genuine losses."""
        frame = self.__class__._gap_up_frame()
        entry_position = len(frame) - (_LONGEST + 1)
        entry_close = float(frame["Close"].iloc[entry_position])
        frame.loc[frame.index[entry_position + 5], "Low"] = entry_close * 0.88
        labels = _labels_for(frame, _as_of(frame, -(_LONGEST + 1)))
        self.assertAlmostEqual(
            labels["horizons"]["20d"]["adverse_excursion"], -0.12, places=4
        )

    def test_the_label_version_records_the_change(self):
        self.assertEqual(core_config.OUTCOME_LABEL_VERSION, "outcome-label-v2")
