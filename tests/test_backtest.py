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

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core import config as core_config
from core.backtest import costs as bt_costs
from core.backtest import metrics as bt_metrics
from core.backtest.engine import (
    BacktestLeakageError,
    BacktestManifestError,
    _avg_dollar_volume,
    _replay_window,
    build_walk_forward_folds,
    offline_replay_seam,
    run_walk_forward_backtest,
)
from core.backtest.manifest import (
    MANIFEST_VERSION,
    build_manifest,
    load_manifest_by_run_hash,
    load_run_manifests,
    persist_run_manifest,
    validate_manifest,
)
from core.forecast_horizons import horizon_sessions
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


# Derived from config: the embargo must cover the longest label horizon, so a
# fixture hardcoding 60 silently breaks the moment a longer horizon is declared
# (F2 added 252d and did exactly that).
_EMBARGO = core_config.BACKTEST_EMBARGO_SESSIONS
# Sessions a fold-running fixture needs: embargo + a validation window + a
# holdout tail, with headroom. Fixed 260/300 stopped fitting at 252d.
_RUN_SESSIONS = _EMBARGO + 150


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


class CostModelV2Tests(unittest.TestCase):
    """Sprint V3: the volatility-adjusted, clamped, itemized cost model."""

    def test_version_stamped_tables(self):
        self.assertEqual(bt_costs.COST_TABLE_VERSION, "backtest-cost-table-v2")
        self.assertEqual(bt_costs.COST_TABLE_V1["cost_table_version"], "backtest-cost-table-v1")
        self.assertEqual(bt_costs.COST_TABLE_V2["cost_table_version"], "backtest-cost-table-v2")

    def test_unknown_cost_table_version_is_rejected(self):
        broken = {**bt_costs.COST_TABLE_V2, "cost_table_version": "backtest-cost-table-vX"}
        with self.assertRaises(ValueError):
            bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.02, broken)

    def test_impact_scales_with_realized_volatility(self):
        calm = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.005)
        wild = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.06)
        self.assertGreater(wild["impact_bps"], calm["impact_bps"])
        self.assertAlmostEqual(calm["vol_factor"], 0.25, places=4)   # 0.005/0.02 clipped to the floor
        self.assertAlmostEqual(wild["vol_factor"], 3.0, places=4)    # 0.06/0.02
        self.assertFalse(calm["clamped"])
        self.assertFalse(wild["clamped"])

    def test_vol_factor_clip_bounds(self):
        floor = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.0001)
        cap = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.20)
        self.assertEqual(floor["vol_factor"], bt_costs.COST_TABLE_V2["impact_vol_factor_min"])
        self.assertEqual(cap["vol_factor"], bt_costs.COST_TABLE_V2["impact_vol_factor_max"])

    def test_missing_volatility_uses_the_documented_neutral_factor(self):
        costs = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, None)
        self.assertAlmostEqual(costs["vol_factor"], 1.0, places=9)
        self.assertFalse(costs["vol_available"])

    def test_floor_and_cap_clamp_the_total(self):
        table = {**bt_costs.COST_TABLE_V2, "min_total_side_cost_bps": 5.0, "max_total_side_cost_bps": 10.0}
        floored = bt_costs.total_side_cost_bps(0.00001, 500_000_000.0, 0.001, table)
        capped = bt_costs.total_side_cost_bps(1.0, 100_000.0, 0.08, table)
        self.assertEqual(floored["total_bps"], 5.0)
        self.assertTrue(floored["clamped"])
        self.assertEqual(capped["total_bps"], 10.0)
        self.assertTrue(capped["clamped"])
        unclamped = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.02)
        self.assertFalse(unclamped["clamped"])

    def test_execution_cost_record_is_fully_itemized(self):
        record = bt_costs.execution_cost_record(
            "buy", 100.0, 0.001, 100_000_000.0, 0.02, 50_000.0
        )
        for field in ("side", "open_price", "executed_price", "order_notional",
                      "participation", "avg_dollar_volume", "daily_vol", "bucket",
                      "half_spread_bps", "impact_bps", "vol_factor", "commission_bps",
                      "total_bps", "clamped", "cost_notional"):
            self.assertIn(field, record, field)
        self.assertAlmostEqual(
            record["executed_price"],
            record["open_price"] * (1 + record["total_bps"] / 10_000.0),
            places=6,
        )
        self.assertAlmostEqual(record["cost_notional"], 50_000.0 * record["total_bps"] / 10_000.0, places=2)

    def test_v1_table_still_replays_under_the_historical_assumptions(self):
        costs = bt_costs.total_side_cost_bps(0.001, 100_000_000.0, 0.08, bt_costs.COST_TABLE_V1)
        # V1 has no volatility term and no clamp: its record carries no vol
        # fields at all (no fake neutral), and impact ignores daily_vol.
        self.assertNotIn("vol_factor", costs)
        self.assertNotIn("vol_available", costs)
        self.assertEqual(
            costs["impact_bps"],
            round(bt_costs.COST_TABLE_V1["impact_coefficient_bps"] * math.sqrt(0.001), 4),
        )
        self.assertFalse(costs["clamped"])


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


class OfflineSeamTests(unittest.TestCase):
    """A replay seam that leaves a provider live is not a seam.

    MEASURED 2026-10-07: the seam cleared NEWS_PROVIDER_API_KEY and FRED_API_KEY
    and left ALPHAVANTAGE_API_KEY set, so an "offline" backtest made live
    fundamentals calls. Alpha Vantage allows 25 requests/day and throttles per
    second, and a throttled call does not raise -- it returns no metrics and the
    scorer substitutes neutral defaults. Two identical runs could therefore
    differ: the first got real values, the second got defaults.

    `test_identical_inputs_rerun_to_identical_results` failed in a full-suite
    run and passed in isolation, which is the signature of exactly this.
    """

    def test_every_provider_key_is_cleared_inside_the_seam(self):
        import os

        from core.backtest.engine import offline_replay_seam

        keys = ("NEWS_PROVIDER_API_KEY", "FRED_API_KEY", "ALPHAVANTAGE_API_KEY")
        with patch.dict(os.environ, {key: "live-looking-value" for key in keys}):
            with offline_replay_seam({"TEST": pd.DataFrame()}):
                for key in keys:
                    with self.subTest(provider_key=key):
                        self.assertEqual(
                            os.getenv(key), "",
                            f"{key} leaks into an offline replay, so a backtest "
                            f"can depend on what a provider served that day",
                        )

    def test_the_seam_restores_the_environment_afterwards(self):
        # Clearing a key permanently would break the collector that runs next.
        import os

        from core.backtest.engine import offline_replay_seam

        with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "restore-me"}):
            with offline_replay_seam({"TEST": pd.DataFrame()}):
                pass
            self.assertEqual(os.getenv("ALPHAVANTAGE_API_KEY"), "restore-me")


class FoldGeometryTests(unittest.TestCase):
    def test_embargo_below_max_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            build_walk_forward_folds(400, fold_sessions=50, embargo_sessions=59, holdout_sessions=40)

    def test_the_embargo_is_judged_against_the_horizon_being_trained(self):
        # MEASURED 2026-10-06: the gate always took the max over EVERY declared
        # horizon (252, the 12m label), so a 20d fit was refused unless it
        # embargoed a full year. `scripts/train.py` failed on its own documented
        # command. A dataset trains one horizon; that is the one at risk.
        geometry = build_walk_forward_folds(
            600, fold_sessions=80, embargo_sessions=60, holdout_sessions=60,
            target_horizon="20d",
        )
        self.assertTrue(geometry["folds"])

    def test_an_embargo_shorter_than_its_own_horizon_is_still_rejected(self):
        # The leakage rule itself is unchanged: a validation label must not
        # overlap the training window, whichever horizon is being trained.
        for horizon in ("120d", "252d"):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    build_walk_forward_folds(
                        2000, fold_sessions=80, embargo_sessions=60,
                        holdout_sessions=60, target_horizon=horizon,
                    )

    def test_an_undeclared_horizon_is_refused_rather_than_ignored(self):
        # Silently falling back to the max would turn a typo into a surprising
        # geometry; naming the mistake is cheaper than debugging the folds.
        with self.assertRaises(ValueError):
            build_walk_forward_folds(
                600, fold_sessions=80, embargo_sessions=300,
                holdout_sessions=60, target_horizon="21d",
            )

    def test_omitting_the_horizon_keeps_the_conservative_default(self):
        # Callers that do not know their horizon must still be held to the
        # longest declared one.
        with self.assertRaises(ValueError):
            build_walk_forward_folds(
                600, fold_sessions=80, embargo_sessions=60, holdout_sessions=60,
            )

    def test_fold_geometry_boundaries_and_holdout_isolation(self):
        sessions = _EMBARGO * 5
        geometry = build_walk_forward_folds(
            sessions, fold_sessions=60, embargo_sessions=_EMBARGO, holdout_sessions=40
        )
        # Derived from the session count, not pinned to a literal: the holdout is
        # always the final 40 sessions wherever the series ends.
        self.assertEqual(geometry["holdout"], [sessions - 40, sessions - 1])
        self.assertGreaterEqual(len(geometry["folds"]), 2)
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
            build_walk_forward_folds(_EMBARGO + 20, fold_sessions=60, embargo_sessions=_EMBARGO, holdout_sessions=40)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        frame = _frame(_ramp(_RUN_SESSIONS))
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
        frame = _frame(_ramp(_RUN_SESSIONS))
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
                injected_labels=None, cost_table=bt_costs.COST_TABLE_V2,
            )
        # One entry (decision t=3 -> execution t=4 open) and one exit
        # (decision t=20 -> execution t=21 open): a full round trip.
        self.assertEqual(replay["position_changes"], 2)
        self.assertEqual(len(replay["trades"]), 1)
        trade = replay["trades"][0]
        self.assertEqual(trade["entry_bar"], frame.index[4].strftime("%Y-%m-%d %H:%M:%S"))
        self.assertEqual(trade["exit_bar"], frame.index[21].strftime("%Y-%m-%d %H:%M:%S"))
        # Position 3: the 20-session volume window doesn't exist yet, so the
        # engine takes the fail-closed worst-case cost path (micro bucket,
        # participation worst case). Volatility is also unavailable there.
        expected_entry = bt_costs.execution_cost_record(
            "buy", float(frame["Open"].iloc[4]), None, None, None, 50_000.0,
            table=bt_costs.COST_TABLE_V2,
        )
        # The exit has both volume and volatility history: full v2 model.
        exit_participation = 50_000.0 / _avg_dollar_volume(frame, 20)
        expected_exit = bt_costs.execution_cost_record(
            "sell", float(frame["Open"].iloc[21]), exit_participation,
            _avg_dollar_volume(frame, 20), bt_costs.realized_vol_daily(frame, 20),
            50_000.0, table=bt_costs.COST_TABLE_V2,
        )
        self.assertEqual(trade["entry_price"], expected_entry["executed_price"])
        self.assertEqual(trade["exit_price"], expected_exit["executed_price"])
        self.assertEqual(trade["entry_costs"], {
            key: expected_entry[key] for key in (
                "bucket", "half_spread_bps", "impact_bps", "vol_factor",
                "commission_bps", "total_bps", "clamped", "cost_notional")
        })
        self.assertEqual(trade["exit_costs"], {
            key: expected_exit[key] for key in (
                "bucket", "half_spread_bps", "impact_bps", "vol_factor",
                "commission_bps", "total_bps", "clamped", "cost_notional")
        })
        self.assertGreater(trade["entry_price"], float(frame["Open"].iloc[4]))  # buy pays costs
        self.assertLess(trade["exit_price"], float(frame["Open"].iloc[21]))     # sell receives less
        self.assertAlmostEqual(trade["return"], expected_exit["executed_price"] / expected_entry["executed_price"] - 1.0, places=6)
        # The final-bar signal (t=29) cannot execute inside the window.
        self.assertEqual(replay["decisions"][-1]["target_position"], 0)
        self.assertEqual(replay["position_changes"], 2)
        # V3: every execution carries its itemized cost record.
        self.assertEqual(len(replay["executions"]), 2)
        self.assertTrue(all(execution["vol_available"] is not None for execution in replay["executions"]))
        self.assertGreater(replay["total_cost_notional"], 0.0)
        self.assertIsNotNone(replay["cost_drag"])


class WalkForwardEngineTests(unittest.TestCase):
    """Full harness runs over the live scoring path, offline."""

    @classmethod
    def setUpClass(cls):
        # Rise then fall: the signal enters on the ramp and exits on the break.
        # Sized from the embargo: a fold spans embargo + fold + holdout, so a
        # fixed 280-session ramp stopped fitting the moment F2's 252d horizon
        # widened the embargo. Half ramps up, half back down.
        _leg = max(140, _EMBARGO + 120)
        closes = _ramp(_leg, step=2.0) + _ramp(_leg, step=-2.0, base=100.0 + 2.0 * (_leg - 1))
        cls.frame = _frame(closes)
        cls.geometry_kwargs = dict(fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.manifest_store = Path(cls._tmp.name) / "backtest_runs.jsonl"
        cls.default_run = run_walk_forward_backtest(
            "TEST", cls.frame, manifest_store_path=cls.manifest_store, **cls.geometry_kwargs
        )
        cls.second_run = run_walk_forward_backtest(
            "TEST", cls.frame, manifest_store_path=cls.manifest_store, **cls.geometry_kwargs
        )
        expensive_table = {
            **bt_costs.COST_TABLE_V1,
            "spread_bps": {"micro": 250.0, "small": 120.0, "mid": 60.0, "large": 100.0},
            "impact_coefficient_bps": 5000.0,
            "commission_bps": 5.0,
        }
        cls.expensive_run = run_walk_forward_backtest(
            "TEST", cls.frame, cost_table=expensive_table,
            manifest_store_path=cls.manifest_store, **cls.geometry_kwargs
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

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
        # Count is a function of the fixture length, not the contract; the
        # per-fold invariants below are what this test is actually about.
        self.assertGreaterEqual(len(run["folds"]), 2)
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

    def test_costs_are_explicitly_attributed(self):
        # V3: no cost may hide inside a price — every execution carries its
        # itemized record, and the aggregate states the total drag.
        run = self.default_run
        self.assertGreater(run["aggregate"]["execution_count"], 0)
        self.assertGreater(run["aggregate"]["total_cost_notional"], 0.0)
        self.assertIsNotNone(run["aggregate"]["avg_execution_cost_bps"])
        self.assertIsNotNone(run["aggregate"]["cost_drag"])
        self.assertGreater(run["aggregate"]["cost_drag"], 0.0)
        executions = [
            execution
            for fold in run["folds"]
            for execution in fold["executions"]
        ]
        self.assertEqual(len(executions), run["aggregate"]["execution_count"])
        for execution in executions:
            with self.subTest(bar=execution["decision_bar"], side=execution["side"]):
                self.assertIn(execution["side"], ("buy", "sell"))
                self.assertIn("bucket", execution)
                self.assertIn("half_spread_bps", execution)
                self.assertIn("impact_bps", execution)
                self.assertIn("commission_bps", execution)
                self.assertIn("total_bps", execution)
                self.assertIn("cost_notional", execution)
                self.assertAlmostEqual(
                    execution["executed_price"],
                    execution["open_price"] * (1 + execution["total_bps"] / 10_000.0)
                    if execution["side"] == "buy"
                    else execution["open_price"] * (1 - execution["total_bps"] / 10_000.0),
                    places=6,
                )

    def test_expensive_table_pays_more_per_execution(self):
        self.assertGreater(
            self.expensive_run["aggregate"]["avg_execution_cost_bps"],
            self.default_run["aggregate"]["avg_execution_cost_bps"],
        )
        self.assertGreater(
            self.expensive_run["aggregate"]["cost_drag"],
            self.default_run["aggregate"]["cost_drag"],
        )


class LeakageRejectionTests(unittest.TestCase):
    """A one-bar-early injected label is detected and the run is rejected."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest_store = Path(self._tmp.name) / "backtest_runs.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _first_decision(frame):
        """The first bar the engine actually SCORES.

        Not merely past the embargo: a decision inside the embargo gap is never
        evaluated, so tampering with its label proved nothing. Read the fold
        geometry instead of guessing an offset.
        """
        geometry = build_walk_forward_folds(
            len(frame), fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40
        )
        first_validation_start = geometry["folds"][0]["validation"][0]
        return frame.index[first_validation_start].strftime("%Y-%m-%d %H:%M:%S")

    def test_canonical_injected_labels_pass(self):
        frame = _frame(_ramp(_RUN_SESSIONS, step=1.0))
        # Every bar: this test is about LEAK DETECTION, not fold arithmetic, so
        # covering the whole frame keeps it valid whatever the geometry becomes.
        # A fixed 110..260 range silently stopped covering the decisions once
        # F2's 252d horizon widened the embargo.
        decision_positions = list(range(len(frame)))
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
                fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40,
                injected_labels=injected,
                manifest_store_path=self.manifest_store,
            )
        self.assertEqual(result["label_alignment"], "verified")
        # V4: the completed run persisted its manifest.
        self.assertTrue(result["manifest_persisted"])
        persisted = load_manifest_by_run_hash(
            result["manifest"]["run_hash"], path=self.manifest_store
        )
        self.assertIsNotNone(persisted)
        self.assertEqual(validate_manifest(persisted), [])

    def test_one_bar_early_label_is_rejected_before_any_metric(self):
        frame = _frame(_ramp(_RUN_SESSIONS, step=1.0))
        # Inject canonical labels for EVERY decision, then tamper with exactly
        # one. Injecting only the first left later decisions unlabelled, so the
        # engine aborted on a missing label before reaching the tampered one --
        # a green test that proved nothing about leak detection.
        injected = {}
        with offline_replay_seam({"TEST": frame}):
            for position in range(len(frame)):
                as_of = frame.index[position].strftime("%Y-%m-%d %H:%M:%S")
                injected[as_of] = build_outcome_labels("TEST", as_of)["horizons"]

        first_decision = self._first_decision(frame)
        canonical = injected[first_decision]
        entry_position = frame.index.get_loc(
            pd.Timestamp(canonical["5d"]["entry_bar"])
        )
        # Shift the exit one bar early: the classic off-by-one leak.
        early_position = entry_position + horizon_sessions("5d") - 1
        leaked_5d = dict(canonical["5d"])
        leaked_5d["exit_bar"] = frame.index[early_position].strftime("%Y-%m-%d %H:%M:%S")
        leaked_5d["forward_return"] = round(
            float(frame["Close"].iloc[early_position]) / float(leaked_5d["entry_close"]) - 1.0, 6
        )
        leaked_5d["record_hash"] = _record_hash(leaked_5d)
        injected[first_decision] = {**canonical, "5d": leaked_5d}

        def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
            return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

        with patch("core.backtest.engine.build_score", fake_build_score):
            with self.assertRaises(BacktestLeakageError) as ctx:
                run_walk_forward_backtest(
                    "TEST", frame,
                    fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40,
                    injected_labels=injected,
                    manifest_store_path=self.manifest_store,
                )
        self.assertIn("leaked_labels_detected", str(ctx.exception))
        self.assertIn("5d", str(ctx.exception))
        # V4: an aborted run never reaches the manifest store — only
        # completed runs exist.
        self.assertEqual(load_run_manifests(path=self.manifest_store), [])


class ManifestMandateTests(unittest.TestCase):
    """V4: run manifests are mandatory — persisted, validated, enforced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest_store = Path(self._tmp.name) / "backtest_runs.jsonl"
        self.frame = _frame(_ramp(_RUN_SESSIONS, step=1.0))

    def tearDown(self):
        self._tmp.cleanup()

    def _scripted_run(self):
        """A fast, fully deterministic harness run (scripted scores)."""

        def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
            return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

        with patch("core.backtest.engine.build_score", fake_build_score):
            return run_walk_forward_backtest(
                "TEST", self.frame,
                fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40,
                manifest_store_path=self.manifest_store,
            )

    def test_every_completed_run_persists_a_valid_manifest(self):
        result = self._scripted_run()
        self.assertTrue(result["manifest_persisted"])
        persisted = load_run_manifests(path=self.manifest_store)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0], result["manifest"])
        self.assertEqual(validate_manifest(persisted[0]), [])
        self.assertEqual(
            load_manifest_by_run_hash(result["manifest"]["run_hash"], path=self.manifest_store),
            result["manifest"],
        )
        self.assertIsNone(
            load_manifest_by_run_hash("no-such-hash", path=self.manifest_store)
        )

    def test_identical_rerun_is_idempotent_in_the_store(self):
        first = self._scripted_run()
        second = self._scripted_run()
        self.assertEqual(first, second)  # byte-identical rerun
        persisted = load_run_manifests(path=self.manifest_store)
        self.assertEqual(len(persisted), 1)  # one run hash -> one record
        self.assertEqual(persisted[0]["run_hash"], first["manifest"]["run_hash"])

    def test_engine_refuses_to_run_without_a_valid_manifest(self):
        broken = {"manifest_version": MANIFEST_VERSION}  # missing everything else
        with patch("core.backtest.engine.build_manifest", return_value=broken):
            with self.assertRaises(BacktestManifestError) as ctx:
                self._scripted_run()
        self.assertIn("invalid run manifest", str(ctx.exception))
        # The refusal happens before any work: nothing was persisted, and
        # the incomplete manifest was never stored either.
        self.assertEqual(load_run_manifests(path=self.manifest_store), [])

    def test_store_rejects_same_hash_with_different_content(self):
        result = self._scripted_run()
        manifest = result["manifest"]
        tampered = {**manifest, "code_commit": "tampered"}
        with self.assertRaises(ValueError) as ctx:
            persist_run_manifest(tampered, path=self.manifest_store)
        self.assertIn("integrity violation", str(ctx.exception))
        # The original record is untouched.
        self.assertEqual(
            load_manifest_by_run_hash(manifest["run_hash"], path=self.manifest_store),
            manifest,
        )

    def test_store_integrity_is_loud_on_malformed_lines(self):
        self._scripted_run()
        with self.manifest_store.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with self.assertRaises(ValueError):
            load_run_manifests(path=self.manifest_store)

    def test_aborted_runs_persist_nothing(self):
        first_decision = self.frame.index[110].strftime("%Y-%m-%d %H:%M:%S")
        with offline_replay_seam({"TEST": self.frame}):
            canonical = build_outcome_labels("TEST", first_decision)
        leaked_5d = dict(canonical["horizons"]["5d"])
        leaked_5d["exit_bar"] = self.frame.index[114].strftime("%Y-%m-%d %H:%M:%S")
        leaked_5d["forward_return"] = round(
            float(self.frame["Close"].iloc[114]) / canonical["entry_close"] - 1.0, 6
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
            with self.assertRaises(BacktestLeakageError):
                run_walk_forward_backtest(
                    "TEST", self.frame,
                    fold_sessions=50, embargo_sessions=_EMBARGO, holdout_sessions=40,
                    injected_labels=injected,
                    manifest_store_path=self.manifest_store,
                )
        self.assertEqual(load_run_manifests(path=self.manifest_store), [])


if __name__ == "__main__":
    unittest.main()

