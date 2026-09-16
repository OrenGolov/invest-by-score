"""Sprint N4: five-state regime classifier + STRESS -> NO_TRADE coupling.

Hermetic: the price-history fetch is patched with synthetic frames (no
network, no wall-clock dependence in the classification path), raw stores
are untouched. Coverage: boundary tests at every threshold/percentile via
the pure rule evaluator; frame-level classification (bullish, bearish,
range, risk_off trend/vol branches, stress); transition-risk flips; PIT
future-bar exclusion; failure-state statuses (UNAVAILABLE / INVALID /
INCOMPLETE, never neutral labels); risk-policy coupling (stress / missing /
unknown labels veto); risk_off momentum dampening mirrored in the scoring
breakdown; and the orchestrator-level guarantee that a stress snapshot
cannot reach PAPER regardless of score.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core import config as core_config
from core.orchestrator import _derive_agent_statuses, orchestrate_score
from core.regime_agent import (
    REGIME_CLASSIFIER_VERSION,
    build_regime_snapshot,
    classify_regime,
    evaluate_rules,
)
from core.regime_contract import fetch_regime_snapshot
from core.risk_policy import evaluate_risk_policy
from core.score_engine import _score_current_time, build_score
from fetch_data import TickerFetchError


def _frame(closes, highs=None, start="2022-01-03"):
    """Daily OHLCV frame from a close sequence (highs default to closes)."""
    index = pd.date_range(start, periods=len(closes), freq="B")
    if highs is None:
        highs = list(closes)
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": list(highs),
            "Low": list(closes),
            "Close": list(closes),
            "Volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )


def _rules(**overrides):
    """Scalar rule-chain input: calm, above-trend defaults overridable."""
    scalars = {
        "realized_vol": 0.010,
        "vol_q80": 0.030,
        "vol_q95": 0.060,
        "drawdown_from_60d_high": 0.03,
        "price_vs_ma_50": 0.05,
        "price_vs_ma_200": 0.08,
        "ma_50": 105.0,
        "ma_200": 100.0,
        "close": 110.0,
        "change_20d": 0.03,
        "change_60d": 0.06,
    }
    scalars.update(overrides)
    return evaluate_rules(**scalars)


class RuleChainBoundaryTests(unittest.TestCase):
    """Acceptance: boundary tests at each threshold/percentile."""

    def test_all_labels_are_from_the_versioned_set(self):
        self.assertEqual(
            core_config.REGIME_LABELS, ("bullish", "bearish", "range", "risk_off", "stress")
        )
        self.assertEqual(core_config.REGIME_REQUIRED_SESSIONS, 283)

    def test_stress_requires_vol_strictly_above_q95_and_drawdown_strictly_exceeded(self):
        # Exactly at both boundaries: neither is "above" -> not stress.
        at_boundary = _rules(realized_vol=0.060, drawdown_from_60d_high=0.15)
        self.assertEqual(at_boundary["label"], "risk_off")
        self.assertFalse(at_boundary["rule_trace"]["stress"]["triggered"])
        # A hair past both: stress.
        past = _rules(realized_vol=0.060001, drawdown_from_60d_high=0.150001)
        self.assertEqual(past["label"], "stress")
        self.assertTrue(past["rule_trace"]["stress"]["triggered"])

    def test_stress_requires_both_conditions(self):
        vol_only = _rules(realized_vol=0.070)  # above q95, drawdown tiny
        self.assertEqual(vol_only["label"], "risk_off")
        dd_only = _rules(drawdown_from_60d_high=0.30)  # deep drawdown, calm vol
        self.assertEqual(dd_only["label"], "bullish")

    def test_risk_off_vol_branch_uses_strict_80th_percentile(self):
        at = _rules(realized_vol=0.030, drawdown_from_60d_high=0.10)
        self.assertNotEqual(at["label"], "risk_off")
        self.assertEqual(at["label"], "bullish")
        above = _rules(realized_vol=0.030001, drawdown_from_60d_high=0.10)
        self.assertEqual(above["label"], "risk_off")

    def test_risk_off_trend_branch_requires_aligned_negative_momentum(self):
        base = {
            "realized_vol": 0.001, "vol_q80": 0.030, "vol_q95": 0.060,
            "price_vs_ma_50": -0.05, "price_vs_ma_200": -0.10,
            "ma_50": 90.0, "ma_200": 100.0, "close": 89.0,
        }
        aligned = _rules(**base, change_20d=-0.01, change_60d=-0.05)
        self.assertEqual(aligned["label"], "risk_off")
        self.assertTrue(aligned["rule_trace"]["risk_off"]["ma_downtrend"])
        self.assertTrue(aligned["rule_trace"]["risk_off"]["aligned_negative_momentum"])
        # 20d recovered while 60d is still negative: not aligned -> falls to bearish.
        recovered = _rules(**base, change_20d=0.01, change_60d=-0.05)
        self.assertEqual(recovered["label"], "bearish")

    def test_range_boundary_is_strictly_inside_the_flat_band(self):
        inside = _rules(price_vs_ma_50=0.019999, price_vs_ma_200=-0.019999,
                        ma_50=100.0, ma_200=100.0, close=100.0)
        self.assertEqual(inside["label"], "range")
        at_band = _rules(price_vs_ma_50=0.019999, price_vs_ma_200=0.020,
                         ma_50=100.0, ma_200=100.0, close=100.0)
        self.assertEqual(at_band["label"], "bullish")

    def test_stress_precedes_everything(self):
        crashing = _rules(realized_vol=0.070, drawdown_from_60d_high=0.30,
                          price_vs_ma_50=0.0, price_vs_ma_200=0.0,
                          ma_50=100.0, ma_200=100.0, close=100.0,
                          change_20d=-0.2, change_60d=-0.4)
        self.assertEqual(crashing["label"], "stress")
        self.assertEqual(crashing["rule_trace"]["precedence"], "stress > risk_off > range > bearish > bullish")

    def test_bearish_requires_below_both_mas_without_risk_off_signals(self):
        bearish = _rules(realized_vol=0.001, price_vs_ma_50=-0.06, price_vs_ma_200=-0.09,
                         ma_50=95.0, ma_200=100.0, close=91.0,
                         change_20d=0.01, change_60d=-0.02)
        self.assertEqual(bearish["label"], "bearish")
        above = _rules(realized_vol=0.001, price_vs_ma_50=0.06, price_vs_ma_200=0.09,
                       ma_50=105.0, ma_200=100.0, close=109.0)
        self.assertEqual(above["label"], "bullish")


# --- Scenario frames (>= 283 sessions so every strict window is populated) ------
#
# Volatility structure is deliberately non-degenerate: storm blocks (large
# alternating returns) set the trailing vol percentiles far above the calm
# tail (constant returns), so every strict threshold comparison has margins
# orders of magnitude wider than float noise.

def _series_from_returns(returns, base=100.0):
    closes = [base]
    for daily_return in returns:
        closes.append(closes[-1] * (1.0 + daily_return))
    return closes[1:]


STORM_UP = [0.004 if index % 2 == 0 else -0.002 for index in range(290)]   # avg +0.1%, vol ~0.3%
CALM_UP = [0.001] * 30
STORM_DOWN = [-0.005 if index % 2 == 0 else 0.001 for index in range(260)]  # avg -0.2%, vol ~0.3%
CALM_RECOVER = [0.0008] * 25
CALM_DOWN = [-0.001] * 30

BULLISH = _frame(_series_from_returns(STORM_UP + CALM_UP))
RANGE = _frame([100.0] * 320)  # exact zeros: vol, quantiles, and MA distances are exactly 0.0
RISK_OFF_TREND = _frame(_series_from_returns(STORM_DOWN + CALM_DOWN))

RISK_OFF_VOL_TAIL = []
_price = 100.0
for step in (0.03, -0.03, 0.03, -0.03, 0.03, -0.02):
    _price = _price * (1.0 + step)
    RISK_OFF_VOL_TAIL.append(_price)
RISK_OFF_VOL = _frame([100.0] * 330 + RISK_OFF_VOL_TAIL)

# Crash: 14 down-pairs (-1.6%/+0.3%) -> ~16.6% below the 60-session high,
# deep enough for the stress drawdown condition; short enough that the
# trailing transition window still mixes pre-stress states with stress.
STRESS_TAIL = _series_from_returns(
    [-0.016 if index % 2 == 0 else 0.003 for index in range(28)]
)
STRESS = _frame([100.0] * 300 + STRESS_TAIL)

BEARISH = _frame(
    _series_from_returns(STORM_UP + STORM_DOWN + CALM_RECOVER)
)

AS_OF = BULLISH.index[-1].strftime("%Y-%m-%d %H:%M:%S")


class FrameClassificationTests(unittest.TestCase):
    """classify_regime over synthetic PIT frames."""

    def test_bullish_uptrend(self):
        result = classify_regime(BULLISH)
        self.assertTrue(result["computable"])
        self.assertEqual(result["label"], "bullish")
        self.assertGreaterEqual(result["probability_proxy"], 0.0)
        self.assertLessEqual(result["probability_proxy"], 1.0)
        self.assertEqual(result["transition_risk"]["flips"], 0)
        self.assertEqual(set(result["recent_labels"]), {"bullish"})

    def test_flat_market_is_range(self):
        result = classify_regime(RANGE)
        self.assertEqual(result["label"], "range")
        self.assertEqual(result["probability_proxy"], 1.0)  # dead flat
        self.assertEqual(result["transition_risk"]["flips"], 0)

    def test_sustained_downtrend_is_risk_off_by_trend_branch(self):
        result = classify_regime(RISK_OFF_TREND)
        self.assertEqual(result["label"], "risk_off")
        trace = result["rule_trace"]["risk_off"]
        self.assertTrue(trace["ma_downtrend"])
        self.assertTrue(trace["aligned_negative_momentum"])
        self.assertFalse(trace["vol_above_q80"])

    def test_vol_spike_without_drawdown_is_risk_off_by_vol_branch(self):
        result = classify_regime(RISK_OFF_VOL)
        self.assertEqual(result["label"], "risk_off")
        trace = result["rule_trace"]["risk_off"]
        self.assertTrue(trace["vol_above_q80"])
        self.assertFalse(result["rule_trace"]["stress"]["triggered"])

    def test_crash_is_stress(self):
        result = classify_regime(STRESS)
        self.assertEqual(result["label"], "stress")
        trace = result["rule_trace"]["stress"]
        self.assertTrue(trace["vol_above_q95"])
        self.assertTrue(trace["drawdown_exceeded"])
        self.assertGreater(result["inputs"]["drawdown_from_60d_high"], 0.15)
        # Transition risk: the trailing window mixes the pre-crash state with
        # the crash states, so at least one regime flip is counted.
        self.assertGreaterEqual(result["transition_risk"]["flips"], 1)
        self.assertLessEqual(len(result["recent_labels"]), 20)

    def test_recovery_below_mas_is_bearish(self):
        result = classify_regime(BEARISH)
        self.assertEqual(result["label"], "bearish")
        self.assertFalse(result["rule_trace"]["risk_off"]["triggered"])
        self.assertTrue(result["rule_trace"]["bearish"]["triggered"])

    def test_short_history_is_not_computable(self):
        result = classify_regime(BULLISH.iloc[:150])
        self.assertFalse(result["computable"])
        self.assertIsNone(result["label"])
        self.assertEqual(result["shortfall"]["required_sessions"], 283)

    def test_proxy_stays_on_the_unit_interval_across_scenarios(self):
        for frame in (BULLISH, RANGE, RISK_OFF_TREND, RISK_OFF_VOL, STRESS, BEARISH):
            result = classify_regime(frame)
            with self.subTest(label=result["label"]):
                self.assertGreaterEqual(result["probability_proxy"], 0.0)
                self.assertLessEqual(result["probability_proxy"], 1.0)


class RegimeSnapshotStatusTests(unittest.TestCase):
    """build_regime_snapshot: PIT behavior and failure-state statuses."""

    def test_ok_snapshot_shape_and_provenance(self):
        with patch("core.regime_agent.fetch_price_history", return_value=BULLISH):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(snapshot["regime"], "bullish")
        self.assertEqual(snapshot["source_id"], "yahoo_finance_chart")
        self.assertEqual(snapshot["calculation_version"], "regime-contract-v1")
        self.assertEqual(snapshot["as_of"], AS_OF)
        self.assertEqual(snapshot["published_time"], AS_OF)
        self.assertEqual(snapshot["pipeline"]["classifier_version"], REGIME_CLASSIFIER_VERSION)
        self.assertIn("0 future bar(s) excluded", snapshot["reason"])
        self.assertIn("sessions", snapshot["lookback_period"])

    def test_future_bars_are_excluded_before_classification(self):
        # The frame extends past as_of; the classification must be identical
        # to the truncated frame and the exclusion must be surfaced.
        extended = _frame(
            list(BULLISH["Close"]) + [200.0, 205.0],
            start=BULLISH.index[0].strftime("%Y-%m-%d"),
        )
        with patch("core.regime_agent.fetch_price_history", return_value=extended):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        truncated = classify_regime(BULLISH)
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(snapshot["regime"], truncated["label"])
        self.assertEqual(snapshot["inputs"], truncated["inputs"])
        self.assertEqual(snapshot["transition_risk"], truncated["transition_risk"])
        self.assertIn("2 future bar(s) excluded", snapshot["reason"])

    def test_fetch_failure_is_unavailable_never_neutral(self):
        with patch(
            "core.regime_agent.fetch_price_history",
            side_effect=TickerFetchError("TEST: network request failed"),
        ):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(snapshot["regime"])
        self.assertEqual(snapshot["source_confidence"], 0.0)
        self.assertEqual(snapshot["source_id"], "regime_source_unavailable")

    def test_empty_payload_is_unavailable(self):
        with patch("core.regime_agent.fetch_price_history", return_value=pd.DataFrame()):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertIsNone(snapshot["regime"])

    def test_schema_violation_is_invalid(self):
        broken = BULLISH.drop(columns=["High"])
        with patch("core.regime_agent.fetch_price_history", return_value=broken):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        self.assertEqual(snapshot["status"], "INVALID")
        self.assertIsNone(snapshot["regime"])

    def test_short_history_is_incomplete_with_explicit_none_regime(self):
        with patch("core.regime_agent.fetch_price_history", return_value=BULLISH.iloc[:150]):
            snapshot = build_regime_snapshot("TEST", AS_OF)
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertIsNone(snapshot["regime"])
        self.assertIsNone(snapshot["probability_proxy"])
        self.assertIn("283 required", snapshot["reason"])

    def test_contract_entry_point_delegates(self):
        with patch("core.regime_agent.fetch_price_history", return_value=RANGE):
            snapshot = fetch_regime_snapshot("TEST", "2024-06-28")
        self.assertEqual(snapshot["regime"], "range")

class RegimeRiskPolicyCouplingTests(unittest.TestCase):
    """N4 governance coupling: stress (and missing evidence) veto."""

    @staticmethod
    def _context(**overrides):
        context = {
            "market_data_quality": 85.0,
            "market_source_confidence": 0.8,
            "market_timestamp_valid": True,
            "fundamental_point_in_time_valid": True,
            "fundamental_source_confidence": 0.9,
            "fundamental_source_status": "live_provider",
            "score": 6.5,
            "action": "PAPER",
            "confidence": 0.8,
            "market_regime": "bullish",
            "confidence_breakdown": {
                "total_penalty": 0.0,
                "factors": [
                    {"name": "freshness", "value": 1.0},
                    {"name": "volatility_regime", "value": 0.9},
                ],
            },
        }
        context.update(overrides)
        return context

    def _rule(self, evaluation):
        return next(rule for rule in evaluation["rules"] if rule["rule_id"] == "market_regime_stress")

    def test_stress_rule_is_veto_severity_in_config(self):
        self.assertEqual(core_config.RISK_POLICY_V2["market_regime_stress"]["severity"], "veto")

    def test_stress_label_triggers_the_veto(self):
        evaluation = evaluate_risk_policy(self._context(market_regime="stress"))
        rule = self._rule(evaluation)
        self.assertTrue(rule["triggered"])
        self.assertEqual(rule["severity"], "veto")
        self.assertIn("market_regime_stress", evaluation["veto_rule_ids"])

    def test_missing_label_fails_closed(self):
        evaluation = evaluate_risk_policy(self._context(market_regime=None))
        rule = self._rule(evaluation)
        self.assertTrue(rule["triggered"])
        self.assertIn("missing", rule["detail"])

    def test_unknown_label_fails_closed(self):
        evaluation = evaluate_risk_policy(self._context(market_regime="crisis"))
        self.assertIn("market_regime_stress", evaluation["veto_rule_ids"])

    def test_non_stress_labels_do_not_veto(self):
        for label in ("bullish", "bearish", "range", "risk_off"):
            with self.subTest(label=label):
                evaluation = evaluate_risk_policy(self._context(market_regime=label))
                self.assertFalse(self._rule(evaluation)["triggered"])
                self.assertNotIn("market_regime_stress", evaluation["veto_rule_ids"])


def _regime_stub(status, label=None, proxy=None):
    """A versioned regime snapshot as build_score would produce it."""
    snapshot = {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": status,
        "source_id": "yahoo_finance_chart", "source_confidence": 0.8 if status == "OK" else 0.0,
        "published_time": "2024-01-02 00:00:00", "calculation_version": "regime-contract-v1",
        "lookback_period": "300 sessions", "regime": label, "probability_proxy": proxy,
        "transition_risk": {"window_sessions": 20, "labeled_sessions": 20, "flips": 0,
                            "flip_rate": 0.0, "labels": [label or "bullish"] * 20},
        "inputs": {}, "rule_trace": {}, "reason": "stub",
    }
    if status == "OK":
        snapshot["pipeline"] = {
            "pipeline_version": "regime-pipeline-v1",
            "classifier_version": "regime-classifier-v1",
            "counts": {"sessions_used": 300, "labeled_sessions": 18, "transition_labels": 20},
        }
    return snapshot


class MomentumDampeningTests(unittest.TestCase):
    """N4: risk_off dampens momentum coefficients; the breakdown mirrors it."""

    @staticmethod
    def _snapshot(**overrides):
        snapshot = {
            "close": 100.0,
            "change_1d": 0.01,
            "change_5d": 0.02,
            "change_20d": 0.05,
            "change_60d": 0.10,
            "trend_vs_20d_mean": 0.015,
            "rsi": 55.0,
            "volume": 1_000_000.0,
            "avg_volume_20d": 900_000.0,
            "volume_ratio_20d": 1.1,
            "price_vs_ma_50": 0.04,
            "price_vs_ma_100": 0.03,
            "price_vs_ma_150": 0.02,
            "price_vs_ma_200": 0.01,
            "volatility": 0.01,
            "moving_averages": {"50d": 104.0, "100d": 103.0, "150d": 102.0, "200d": 101.0},
        }
        snapshot.update(overrides)
        return snapshot

    def test_dampening_scales_momentum_toward_zero(self):
        baseline = _score_current_time(self._snapshot())
        dampened = _score_current_time(
            self._snapshot(), momentum_damping=core_config.REGIME_RISKOFF_MOMENTUM_DAMPING
        )
        # Positive momentum: dampening pulls the score toward the 4.0 base.
        self.assertLess(dampened, baseline)
        self.assertGreater(dampened, 4.0)
        # Negative momentum: dampening pulls the penalty toward the base too.
        bearish = self._snapshot(change_1d=-0.01, change_5d=-0.02, change_20d=-0.05, trend_vs_20d_mean=-0.015)
        self.assertGreater(
            _score_current_time(bearish, momentum_damping=core_config.REGIME_RISKOFF_MOMENTUM_DAMPING),
            _score_current_time(bearish),
        )

    def test_dampening_never_touches_non_momentum_terms(self):
        # Zero momentum: damping is a no-op, so RSI/MA/volume terms are provably untouched.
        flat = self._snapshot(change_1d=0.0, change_5d=0.0, change_20d=0.0, trend_vs_20d_mean=0.0)
        self.assertEqual(
            _score_current_time(flat),
            _score_current_time(flat, momentum_damping=core_config.REGIME_RISKOFF_MOMENTUM_DAMPING),
        )

    def test_damping_must_be_a_real_number(self):
        with self.assertRaises(TypeError):
            _score_current_time(self._snapshot(), momentum_damping={"status": "OK"})

    def test_scoring_breakdown_mirrors_the_dampening(self):
        from core.score_engine import _build_scoring_breakdown

        baseline = _build_scoring_breakdown(self._snapshot(), 6.0, 6.0, 5.0)
        dampened = _build_scoring_breakdown(
            self._snapshot(), 6.0, 6.0, 5.0, momentum_damping=core_config.REGIME_RISKOFF_MOMENTUM_DAMPING
        )
        for term in ("momentum_1d", "momentum_5d", "momentum_20d", "trend_vs_20d_mean"):
            self.assertAlmostEqual(
                round(baseline["current_time_breakdown"][term] * core_config.REGIME_RISKOFF_MOMENTUM_DAMPING, 2),
                dampened["current_time_breakdown"][term],
                places=2,
                msg=term,
            )
        for term in ("price_vs_50d_ma", "rsi_signal", "volume_confirmation"):
            self.assertEqual(
                baseline["current_time_breakdown"][term], dampened["current_time_breakdown"][term], msg=term
            )
        self.assertEqual(
            dampened["regime_momentum_damping"], core_config.REGIME_RISKOFF_MOMENTUM_DAMPING
        )
        self.assertTrue(any("Risk-off" in driver for driver in dampened["score_change_drivers"]))
        self.assertFalse(any("Risk-off" in driver for driver in baseline["score_change_drivers"]))


class ScoreEngineRegimeIntegrationTests(unittest.TestCase):
    """build_score carries the regime snapshot; the ensemble weight stays 0."""

    def test_build_score_carries_the_regime_line_and_hash(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish", 0.42)):
            result = build_score("MSFT", "2024-01-02", persist_audit=False)
        line = result.ensemble_breakdown["agents"]["market_regime"]
        self.assertEqual(line["status"], "OK")
        self.assertIsNone(line["score_current"])
        self.assertFalse(line["eligible_current"])
        self.assertEqual(line["effective_weight_current"], 0.0)
        self.assertIn("regime_snapshot_hash", result.replay_metadata)
        self.assertEqual(result.market_regime_snapshot["regime"], "bullish")
        # Zero weight: the effective weights still sum to exactly 1.0.
        for horizon in ("current", "long"):
            total = sum(
                entry[f"effective_weight_{horizon}"]
                for entry in result.ensemble_breakdown["agents"].values()
            )
            self.assertAlmostEqual(total, 1.0, places=6, msg=horizon)

    def test_risk_off_regime_dampens_the_published_current_time_score(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish", 0.9)):
            bullish = build_score("MSFT", "2024-01-02", persist_audit=False)
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "risk_off", 0.5)):
            risk_off = build_score("MSFT", "2024-01-02", persist_audit=False)
        # The dampening only scales momentum terms; the long-term view is identical.
        self.assertEqual(risk_off.long_term_score, bullish.long_term_score)
        self.assertEqual(
            risk_off.scoring_breakdown["regime_momentum_damping"],
            core_config.REGIME_RISKOFF_MOMENTUM_DAMPING,
        )
        self.assertEqual(bullish.scoring_breakdown["regime_momentum_damping"], 1.0)
        # With positive 20d momentum in the cached MSFT frame, the dampened
        # current-time score must sit strictly below the bullish run's.
        self.assertLess(risk_off.current_time_score, bullish.current_time_score)

    def test_regime_unavailable_does_not_change_the_technical_view(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish")):
            bullish = build_score("MSFT", "2024-01-02", persist_audit=False)
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("UNAVAILABLE")):
            unavailable = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(unavailable.current_time_score, bullish.current_time_score)
        self.assertEqual(unavailable.ensemble_breakdown["agents"]["market_regime"]["status"], "UNAVAILABLE")


class OrchestratorRegimeCouplingTests(unittest.TestCase):
    """A stress snapshot cannot reach PAPER regardless of score (acceptance)."""

    def test_stress_regime_forces_no_trade(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "stress", 0.8)):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(decision.mode, "NO_TRADE")
        self.assertIn("market_regime_stress", decision.veto_reasons)
        regime_agent = next(agent for agent in decision.agent_outputs if agent.agent == "market_regime")
        self.assertEqual(regime_agent.status, "OK")  # classified successfully...
        self.assertEqual(regime_agent.payload["regime"], "stress")  # ...and gated.
        risk_agent = next(agent for agent in decision.agent_outputs if agent.agent == "risk_management")
        self.assertEqual(risk_agent.status, "VETO")

    def test_bullish_regime_adds_no_veto(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish", 0.9)):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertNotIn("market_regime_stress", decision.veto_reasons)
        regime_agent = next(agent for agent in decision.agent_outputs if agent.agent == "market_regime")
        self.assertEqual(regime_agent.status, "OK")

    def test_unavailable_regime_fails_closed_to_no_trade(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("UNAVAILABLE")):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(decision.mode, "NO_TRADE")
        self.assertIn("market_regime_stress", decision.veto_reasons)
        regime_agent = next(agent for agent in decision.agent_outputs if agent.agent == "market_regime")
        self.assertEqual(regime_agent.status, "UNAVAILABLE")
        self.assertIn("regime_not_fully_usable", regime_agent.warnings)

    def test_regime_agent_is_the_ninth_roster_member(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish")):
            decision = orchestrate_score("MSFT", "2024-01-02")
        names = [agent.agent for agent in decision.agent_outputs]
        self.assertEqual(len(names), 9)
        self.assertEqual(names.index("market_regime"), 6)  # after macro, before risk
        self.assertEqual(decision.replay_hash, decision.replay_hash)


def _statuses_stub(market_regime_status):
    return SimpleNamespace(
        confidence_breakdown={"factors": [{"name": "freshness", "value": 1.0}]},
        news_snapshot={"status": "OK"},
        macro_snapshot={"status": "OK"},
        market_regime_snapshot={"status": market_regime_status},
    )


class AgentStatusDerivationTests(unittest.TestCase):
    def test_derive_agent_statuses_includes_market_regime(self):
        statuses = _derive_agent_statuses(
            {"data_quality": {"score": 85.0}, "source_contract": {"timestamp_valid": True}},
            {"point_in_time_valid": True},
            _statuses_stub("OK"),
        )
        self.assertEqual(statuses["market_regime"], "OK")

    def test_missing_regime_snapshot_defaults_to_unavailable(self):
        statuses = _derive_agent_statuses(
            {"data_quality": {"score": 85.0}, "source_contract": {"timestamp_valid": True}},
            {"point_in_time_valid": True},
            SimpleNamespace(confidence_breakdown={"factors": []}, news_snapshot={"status": "OK"}),
        )
        self.assertEqual(statuses["market_regime"], "UNAVAILABLE")

    def test_technical_status_inherits_market_data_status(self):
        """Regression: technical_analysis used to be hardcoded OK.

        The technical agent derives every signal from the market snapshot and
        carries the largest ensemble weight, so reporting OK on top of a
        degraded snapshot let a failed agent masquerade as OK in the audit
        record (master context section 32).
        """
        for quality, timestamp_valid, expected in (
            (85.0, True, "OK"),
            (10.0, True, "INCOMPLETE"),
            (85.0, False, "INVALID"),
        ):
            with self.subTest(quality=quality, timestamp_valid=timestamp_valid):
                statuses = _derive_agent_statuses(
                    {
                        "data_quality": {"score": quality},
                        "source_contract": {"timestamp_valid": timestamp_valid},
                    },
                    {"point_in_time_valid": True},
                    _statuses_stub("OK"),
                )
                self.assertEqual(statuses["market_data"], expected)
                self.assertEqual(statuses["technical_analysis"], expected)


if __name__ == "__main__":
    unittest.main()


