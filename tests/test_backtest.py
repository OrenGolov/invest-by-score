"""Sprint V2: walk-forward validation harness.

Hermetic: the offline replay seam injects cached frames and forces
provider-gated domains to their no-key UNAVAILABLE contracts, so the LIVE
scoring path replays offline and deterministically. Coverage:

- cost model: liquidity buckets (fail-closed on unknown), decomposition,
  side asymmetry, square-root impact sensitivity;
- metrics: pure functions with documented None conventions, determinism;
- fold geometry: embargo >= max label horizon enforced, train/embargo/
  validation boundaries, holdout never touched by a fold;
- manifests: required fields, seed=None legitimate, unknown version
  rejected, deterministic run hash;
- engine: manifest validity, structure, embargo gap, holdout isolation,
  deterministic re-runs (identical inputs -> identical results), cost
  parameters demonstrably changing results, t+1-open execution and cost
  application on the trade accounting (scripted scores);
- leakage: a deliberately shifted (one-bar-early) injected label is
  detected by canonical hash verification and the whole run is rejected
  before any metric exists.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core import config as core_config
from core.backtest import costs as bt_costs
from core.backtest import metrics as bt_metrics
from core.backtest.engine import (
    BacktestLeakageError,
    _avg_dollar_volume,
    _replay_window,
    build_walk_forward_folds,
    offline_replay_seam,
    run_walk_forward_backtest,
)
from core.backtest.manifest import MANIFEST_VERSION, build_manifest, validate_manifest
from core.labels import _record_hash, build_outcome_labels


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": list(closes),
            "Low": list(closes),
            "Close": list(closes),
            "Volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * index for index in range(sessions)]


class CostModelTests(unittest.TestCase):
    def test_liquidity_buckets_and_fail_closed_unknown(self):
        table = bt_costs.COST_TABLE_V1
        self.assertEqual(bt_costs.liquidity_bucket(None, table), "micro")
        self.assertEqual(bt_costs.liquidity_bucket(500_000.0, table), "micro")
        self.assertEqual(bt_costs.liquidity_bucket(5_000_000.0, table), "small")
        self.assertEqual(bt_costs.liquidity_bucket(50_000_000.0, table), "mid")
        self.assertEqual(bt_costs.liquidity_bucket(500_000_000.0, table), "large")

    def test_side_costs_decompose_and_grow_with_participation(self):
        low = bt_costs.total_side_cost_bps(0.0001, 500_000_000.0)
        high = bt_costs.total_side_cost_bps(0.5, 500_000_000.0)
        self.assertEqual(low["bucket"], "large")
        self.assertGreater(high["impact_bps"], low["impact_bps"])
        self.assertEqual(low["commission_bps"], high["commission_bps"])
        self.assertAlmostEqual(
            low["total_bps"],
            low["half_spread_bps"] + low["impact_bps"] + low["commission_bps"],
            places=4,
        )

    def test_execution_price_asymmetry_between_sides(self):
        buy = bt_costs.execution_price(100.0, "buy", 0.001, 50_000_000.0)
        sell = bt_costs.execution_price(100.0, "sell", 0.001, 50_000_000.0)
        self.assertGreater(buy, 100.0)
        self.assertLess(sell, 100.0)
        with self.assertRaises(ValueError):
            bt_costs.execution_price(100.0, "hold", 0.001, 50_000_000.0)
        with self.assertRaises(ValueError):
            bt_costs.execution_price(0.0, "buy", 0.001, 50_000_000.0)


class MetricsTests(unittest.TestCase):
    def test_max_drawdown_known_curve(self):
        equity = [100.0, 120.0, 90.0, 95.0, 60.0, 110.0]
        self.assertAlmostEqual(bt_metrics.max_drawdown(equity), 0.5, places=9)

    def test_sharpe_none_on_zero_dispersion(self):
        self.assertIsNone(bt_metrics.sharpe_ratio([0.01, 0.01, 0.01]))
        self.assertIsNone(bt_metrics.sharpe_ratio([0.01]))

    def test_sharpe_positive_for_consistent_gains(self):
        returns = [0.001] * 10 + [0.002] * 10
        self.assertGreater(bt_metrics.sharpe_ratio(returns), 0.0)

    def test_sortino_ignores_upside_and_none_without_downside(self):
        self.assertIsNone(bt_metrics.sortino_ratio([0.01, 0.02, 0.015]))
        mixed = [0.01, -0.005, 0.02, -0.01]
        self.assertGreater(bt_metrics.sortino_ratio(mixed), 0.0)

    def test_cagr_and_calmar(self):
        curve = [100.0 * (1.10 ** (i / 252)) for i in range(253)]
        cagr_value = bt_metrics.cagr(curve)
        self.assertAlmostEqual(cagr_value, 0.10, places=2)
        self.assertEqual(bt_metrics.max_drawdown(curve), 0.0)  # monotonic
        self.assertIsNone(bt_metrics.calmar_ratio(cagr_value, 0.0))
        self.assertAlmostEqual(bt_metrics.calmar_ratio(0.20, 0.10), 2.0, places=6)

    def test_win_rate_profit_factor_and_exposure(self):
        self.assertIsNone(bt_metrics.win_rate([]))
        self.assertIsNone(bt_metrics.profit_factor([]))
        self.assertIsNone(bt_metrics.profit_factor([0.05, 0.03]))  # no losses
        self.assertAlmostEqual(bt_metrics.win_rate([0.05, -0.02, 0.03]), 2 / 3, places=6)
        self.assertAlmostEqual(bt_metrics.profit_factor([0.05, -0.02, 0.03]), 4.0, places=6)
        self.assertAlmostEqual(bt_metrics.exposure([1, 1, 0, 0]), 0.5, places=9)
        self.assertAlmostEqual(bt_metrics.turnover(4, 100), 0.04, places=9)

    def test_metrics_block_is_deterministic(self):
        first = bt_metrics.compute_metrics([0.01, -0.002], [100.0, 100.8], [0, 1], [0.008], 2, 0, 1, 2)
        second = bt_metrics.compute_metrics([0.01, -0.002], [100.0, 100.8], [0, 1], [0.008], 2, 0, 1, 2)
        self.assertEqual(first, second)
        self.assertEqual(first["metrics_version"], "backtest-metrics-v1")


class FoldGeometryTests(unittest.TestCase):
    def test_embargo_below_max_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            build_walk_forward_folds(400, fold_sessions=50, embargo_sessions=59, holdout_sessions=40)

    def test_fold_geometry_boundaries_and_holdout_isolation(self):
        geometry = build_walk_forward_folds(300, fold_sessions=60, embargo_sessions=60, holdout_sessions=40)
        self.assertEqual(geometry["holdout"], [260, 299])
        self.assertEqual(len(geometry["folds"]), 2)
        for fold in geometry["folds"]:
            train_end = fold["train"][1]
            validation_start, validation_end = fold["validation"]
            # The embargo gap sits strictly between train end and validation.
            self.assertEqual(validation_start - train_end - 1, fold["embargo_sessions"])
            self.assertEqual(validation_end - validation_start + 1, 60)
            # Validation never touches the holdout tail.
            self.assertLess(validation_end, geometry["holdout"][0])
        # Consecutive folds advance by exactly one fold length.
        self.assertEqual(
            geometry["folds"][1]["validation"][0] - geometry["folds"][0]["validation"][0], 60
        )

    def test_too_short_history_is_rejected(self):
        with self.assertRaises(ValueError):
            build_walk_forward_folds(150, fold_sessions=60, embargo_sessions=60, holdout_sessions=40)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        frame = _frame(_ramp(300))
        first = build_manifest(
            "TEST", frame,
            {"embargo": 60, "fold_sessions": 60},
            {"ensemble": "ensemble-v3", "outcome_label": "outcome-label-v1"},
            {"price_history": "cached_frame_injection"},
        )
        second = build_manifest(
            "TEST", frame,
            {"embargo": 60, "fold_sessions": 60},
            {"ensemble": "ensemble-v3", "outcome_label": "outcome-label-v1"},
            {"price_history": "cached_frame_injection"},
        )
        self.assertEqual(first, second)
        self.assertEqual(first["manifest_version"], MANIFEST_VERSION)
        self.assertEqual(validate_manifest(first), [])

    def test_changed_inputs_change_the_run_hash(self):
        frame = _frame(_ramp(300))
        base = build_manifest("TEST", frame, {"embargo": 60}, {}, {})
        changed = build_manifest("TEST", frame, {"embargo": 61}, {}, {})
        self.assertNotEqual(base["run_hash"], changed["run_hash"])

    def test_validation_rejects_incomplete_or_unknown_manifests(self):
        problems = validate_manifest({"manifest_version": MANIFEST_VERSION})
        self.assertTrue(any("missing field" in problem for problem in problems))
        unknown = {"manifest_version": "backtest-manifest-v0", "run_hash": "x", "code_commit": "x",
                   "versions": {"a": "1"}, "data_digest": "x", "config": {"a": 1}, "seed": None,
                   "provider_overrides": {"a": "b"}}
        self.assertTrue(any("unknown manifest_version" in problem for problem in validate_manifest(unknown)))


class ReplayWindowTests(unittest.TestCase):
    """Trade accounting: t+1-open execution, costs, round trip, boundary."""

    def test_scripted_scores_drive_a_costed_round_trip(self):
        frame = _frame(_ramp(30, step=1.0))
        entry_as_of = frame.index[3].strftime("%Y-%m-%d %H:%M:%S")
        exit_as_of = frame.index[20].strftime("%Y-%m-%d %H:%M:%S")
        script = {entry_as_of: 7.5, exit_as_of: 3.0}

        def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
            return SimpleNamespace(score=script.get(as_of, 5.0), action="ANALYSIS_ONLY")

        with offline_replay_seam({"TEST": frame}), \
                patch("core.backtest.engine.build_score", fake_build_score):
            replay = _replay_window(
                "TEST", frame, 0, 29,
                initial_capital=100_000.0, trade_notional=50_000.0,
                injected_labels=None, cost_table=bt_costs.COST_TABLE_V1,
            )
        # One entry (decision t=3 -> execution t=4 open) and one exit
        # (decision t=20 -> execution t=21 open): a full round trip.
        self.assertEqual(replay["position_changes"], 2)
        self.assertEqual(len(replay["trades"]), 1)
        trade = replay["trades"][0]
        self.assertEqual(trade["entry_bar"], frame.index[4].strftime("%Y-%m-%d %H:%M:%S"))
        self.assertEqual(trade["exit_bar"], frame.index[21].strftime("%Y-%m-%d %H:%M:%S"))
        # Position 3: the 20-session volume window doesn't exist yet, so the
        # engine takes the fail-closed worst-case cost path.
        expected_entry = bt_costs.execution_price(
            float(frame["Open"].iloc[4]), "buy", None, None
        )
        expected_exit = bt_costs.execution_price(
            float(frame["Open"].iloc[21]), "sell",
            50_000.0 / _avg_dollar_volume(frame, 20), _avg_dollar_volume(frame, 20),
        )
        self.assertEqual(trade["entry_price"], expected_entry)
        self.assertEqual(trade["exit_price"], expected_exit)
        self.assertGreater(trade["entry_price"], float(frame["Open"].iloc[4]))  # buy pays costs
        self.assertLess(trade["exit_price"], float(frame["Open"].iloc[21]))     # sell receives less
        self.assertAlmostEqual(trade["return"], expected_exit / expected_entry - 1.0, places=6)
        # The final-bar signal (t=29) cannot execute inside the window.
        self.assertEqual(replay["decisions"][-1]["target_position"], 0)
        self.assertEqual(replay["position_changes"], 2)


class WalkForwardEngineTests(unittest.TestCase):
    """Full harness runs over the live scoring path, offline."""

    @classmethod
    def setUpClass(cls):
        # Rise then fall: the signal enters on the ramp and exits on the break.
        closes = _ramp(140, step=2.0) + _ramp(140, step=-2.0, base=100.0 + 2.0 * 139)
        cls.frame = _frame(closes)
        cls.geometry_kwargs = dict(fold_sessions=50, embargo_sessions=60, holdout_sessions=40)
        cls.default_run = run_walk_forward_backtest("TEST", cls.frame, **cls.geometry_kwargs)
        cls.second_run = run_walk_forward_backtest("TEST", cls.frame, **cls.geometry_kwargs)
        expensive_table = {
            **bt_costs.COST_TABLE_V1,
            "spread_bps": {"micro": 250.0, "small": 120.0, "mid": 60.0, "large": 100.0},
            "impact_coefficient_bps": 5000.0,
            "commission_bps": 5.0,
        }
        cls.expensive_run = run_walk_forward_backtest(
            "TEST", cls.frame, cost_table=expensive_table, **cls.geometry_kwargs
        )

    def test_manifest_is_valid_and_complete(self):
        manifest = self.default_run["manifest"]
        self.assertEqual(self.default_run["manifest_issues"], [])
        self.assertEqual(manifest["manifest_version"], MANIFEST_VERSION)
        self.assertEqual(len(manifest["run_hash"]), 64)
        self.assertIn("ensemble", manifest["versions"])
        self.assertIn("outcome_label", manifest["versions"])
        self.assertIn("cost_table", manifest["versions"])
        self.assertIn("strategy", manifest["versions"])
        self.assertIsNone(manifest["seed"])

    def test_structure_folds_holdout_and_labels(self):
        run = self.default_run
        self.assertEqual(run["label_alignment"], "verified")
        self.assertEqual(len(run["folds"]), 2)
        holdout_start = run["geometry"]["holdout"][0]
        for fold in run["folds"]:
            self.assertEqual(fold["evaluation"], "validation_fold")
            self.assertEqual(
                fold["validation"][0] - fold["train"][1] - 1, fold["embargo_sessions"]
            )
            self.assertLess(fold["validation"][1], holdout_start)
            self.assertGreater(fold["metrics"]["decision_count"], 0)
        self.assertEqual(run["holdout"]["evaluation"], "holdout_once")
        self.assertGreater(run["aggregate"]["label_evaluated"], 0)
        self.assertIsNotNone(run["aggregate"]["label_hit_rate"])
        self.assertIn("rejection_rate", run["aggregate"]["metrics"])

    def test_identical_inputs_rerun_to_identical_results(self):
        self.assertEqual(self.default_run, self.second_run)

    def test_cost_parameters_demonstrably_change_results(self):
        self.assertNotEqual(
            self.default_run["aggregate"]["final_equity"],
            self.expensive_run["aggregate"]["final_equity"],
        )
        self.assertNotEqual(
            self.default_run["manifest"]["run_hash"],
            self.expensive_run["manifest"]["run_hash"],
        )


class LeakageRejectionTests(unittest.TestCase):
    """A one-bar-early injected label is detected and the run is rejected."""

    @staticmethod
    def _first_decision(frame):
        return frame.index[110].strftime("%Y-%m-%d %H:%M:%S")

    def test_canonical_injected_labels_pass(self):
        frame = _frame(_ramp(260, step=1.0))
        decision_positions = list(range(110, 210)) + list(range(220, 260))
        injected = {}
        with offline_replay_seam({"TEST": frame}):
            for position in decision_positions:
                as_of = frame.index[position].strftime("%Y-%m-%d %H:%M:%S")
                canonical = build_outcome_labels("TEST", as_of)
                injected[as_of] = canonical["horizons"]

        def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
            return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

        with patch("core.backtest.engine.build_score", fake_build_score):
            result = run_walk_forward_backtest(
                "TEST", frame,
                fold_sessions=50, embargo_sessions=60, holdout_sessions=40,
                injected_labels=injected,
            )
        self.assertEqual(result["label_alignment"], "verified")

    def test_one_bar_early_label_is_rejected_before_any_metric(self):
        frame = _frame(_ramp(260, step=1.0))
        first_decision = self._first_decision(frame)
        with offline_replay_seam({"TEST": frame}):
            canonical = build_outcome_labels("TEST", first_decision)
        leaked_5d = dict(canonical["horizons"]["5d"])
        # Shift the exit one bar early: the classic off-by-one leak.
        leaked_5d["exit_bar"] = frame.index[114].strftime("%Y-%m-%d %H:%M:%S")
        leaked_5d["forward_return"] = round(
            float(frame["Close"].iloc[114]) / canonical["entry_close"] - 1.0, 6
        )
        leaked_5d["record_hash"] = _record_hash(leaked_5d)
        injected = {
            first_decision: {
                "1d": canonical["horizons"]["1d"],
                "5d": leaked_5d,
                "20d": canonical["horizons"]["20d"],
                "60d": canonical["horizons"]["60d"],
            }
        }

        def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
            return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

        with patch("core.backtest.engine.build_score", fake_build_score):
            with self.assertRaises(BacktestLeakageError) as ctx:
                run_walk_forward_backtest(
                    "TEST", frame,
                    fold_sessions=50, embargo_sessions=60, holdout_sessions=40,
                    injected_labels=injected,
                )
        self.assertIn("leaked_labels_detected", str(ctx.exception))
        self.assertIn("5d", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

