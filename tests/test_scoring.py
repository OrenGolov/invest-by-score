from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from agents.market_data_agent import _pct_change, fetch_market_snapshot
from core import config as core_config
from core.agent_contracts import AgentContract, OrchestrationDecision
from core.audit_store import get_audit_events, get_decision_by_replay_hash, get_decision_by_ticker_and_as_of, persist_decision_audit
from agents.technical_agent import score_technical
from core.audit_policy import evaluate_audit_policy, stable_hash
from core.orchestrator import _derive_agent_statuses, _select_mode, orchestrate_score
from core.schemas import STATUS_POSTURE, AgentStatus, status_posture, worst_status
from core.raw_store import append_raw_records, load_raw_records, rebuild_price_frame
from core.risk_policy import evaluate_risk_policy
from core.score_engine import _compute_confidence, _ensemble_blend, _score_current_time, _score_long_term, build_score
from fetch_data import fetch_fundamental_snapshot, get_provider_health_matrix, resolve_fundamental_provider


class ScoringEngineTests(unittest.TestCase):
    def test_score_output_shape(self):
        result = build_score("MSFT", "2024-01-02")
        payload = result.to_dict()
        self.assertEqual(result.ticker, "MSFT")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 10.0)
        self.assertIn("action", payload)
        self.assertIsInstance(result.risk_flags, list)
        self.assertIn("moving_averages", payload)
        self.assertIn("rsi", payload)
        self.assertIn("volatility", payload)
        self.assertIn("market_context", payload)
        self.assertIn("data_quality", payload)
        self.assertIn("source_metadata", payload)
        self.assertIn("recommended_actions", payload)
        self.assertIn("latest_financial_report", payload)
        self.assertIn("next_expected_report", payload)
        self.assertIn("insights", payload)
        self.assertIn("scoring_breakdown", payload)
        self.assertIn("source_reliability", payload)
        self.assertIn("technical_features", payload)
        self.assertIn("feature_metadata", payload)
        self.assertIn("governance", payload)
        self.assertIn("risk_gate_passed", payload["governance"])
        self.assertIn("evidence_status", payload["governance"])
        self.assertIn("gate_reasons", payload["governance"])
        self.assertIn("evidence_ledger", payload)
        self.assertIn("status", payload["evidence_ledger"])
        self.assertIn("fundamental_score", payload)
        self.assertIn("fundamental_features", payload)
        self.assertIn("valuation_quality", payload["fundamental_features"])
        self.assertIn("current_time_score", payload)
        self.assertIn("long_term_score", payload)
        self.assertGreaterEqual(payload["current_time_score"], 0.0)
        self.assertGreaterEqual(payload["long_term_score"], 0.0)
        self.assertIn("200d", payload["moving_averages"])
        self.assertIn("150d", payload["moving_averages"])
        self.assertIn("100d", payload["moving_averages"])
        self.assertIn("50d", payload["moving_averages"])
        self.assertIn("score", payload["data_quality"])
        self.assertIn("feature_family", payload["feature_metadata"])
        self.assertIn("trend_regime", payload["technical_features"])
        self.assertIn("source", payload["source_metadata"])
        self.assertIn("primary", payload["recommended_actions"])
        self.assertIn("publication_date", payload["next_expected_report"])
        self.assertIn("bullish_signals", payload["insights"])
        self.assertIn("weighted_contributions", payload["scoring_breakdown"])

    def test_current_and_long_term_scores_diverge_by_regime(self):
        per_ticker = {ticker: build_score(ticker, "2024-01-02") for ticker in ["MSFT", "AAPL", "NVDA", "TSLA", "META"]}
        current_values = {ticker: result.current_time_score for ticker, result in per_ticker.items()}
        long_values = {ticker: result.long_term_score for ticker, result in per_ticker.items()}
        self.assertGreater(len(set(round(value, 2) for value in current_values.values())), 1)
        self.assertGreater(len(set(round(value, 2) for value in long_values.values())), 1)
        self.assertTrue(any(abs(result.current_time_score - result.long_term_score) > 0.05 for result in per_ticker.values()))

    def test_market_snapshot_reports_point_in_time_and_quality(self):
        snapshot = fetch_market_snapshot("MSFT", "2024-01-02")
        self.assertEqual(snapshot["as_of"], "2024-01-02 00:00:00")
        self.assertIn("point_in_time_cutoff", snapshot)
        self.assertIn("future_bars_excluded", snapshot)
        self.assertIn("data_quality", snapshot)
        self.assertIn("score", snapshot["data_quality"])
        self.assertIn("chart_series", snapshot)
        self.assertTrue(snapshot["chart_series"])
        self.assertIn("ma_50", snapshot["chart_series"][0])
        self.assertIsInstance(snapshot["data_quality"]["flags"], list)
        self.assertIsInstance(snapshot["future_bars_excluded"], int)

    def test_fundamental_snapshot_uses_real_feed_and_respects_as_of(self):
        snapshot = fetch_fundamental_snapshot("MSFT", "2024-01-02")
        self.assertIn("source", snapshot)
        self.assertIn("source_type", snapshot)
        self.assertIn("latest_close", snapshot)
        self.assertIn("valuation_metrics", snapshot)
        self.assertIn("point_in_time_valid", snapshot)
        self.assertIsInstance(snapshot["point_in_time_valid"], bool)
        self.assertLessEqual(pd.Timestamp(snapshot["as_of"]), pd.Timestamp("2024-01-02 23:59:59"))
        self.assertIn("as_of", snapshot)
        self.assertIn("source_status", snapshot)
        self.assertTrue(snapshot["point_in_time_valid"])

    def test_sprint5_orchestration_contracts_are_typed_and_replayable(self):
        result = orchestrate_score("MSFT", "2024-01-02")
        self.assertIsInstance(result, OrchestrationDecision)
        self.assertEqual(result.ticker, "MSFT")
        self.assertEqual(result.as_of, "2024-01-02 00:00:00")
        self.assertIn("analysis_only", result.mode.lower())
        self.assertGreaterEqual(len(result.agent_outputs), 3)
        self.assertTrue(all(isinstance(agent, AgentContract) for agent in result.agent_outputs))
        self.assertTrue(all(agent.input_hash for agent in result.agent_outputs))
        self.assertIn("market_data", result.agent_outputs[0].agent)
        self.assertIn("status", result.agent_outputs[0].payload)
        score_result = build_score("MSFT", "2024-01-02")
        self.assertEqual(result.current_time_score, score_result.current_time_score)
        self.assertEqual(result.long_term_score, score_result.long_term_score)
        replay = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(result.to_dict(), replay.to_dict())

    def test_future_as_of_is_rejected(self):
        with self.assertRaises(ValueError):
            fetch_market_snapshot("MSFT", "2099-01-01")

    def test_no_market_data_is_a_validation_error(self):
        with self.assertRaises(ValueError):
            fetch_market_snapshot("ZZZZ", "2024-01-02")

    def test_future_dated_fundamental_payload_is_rejected(self):
        future_payload = {
            "LastDivDate": "2099-01-01",
            "PERatio": "30",
            "PriceToBookRatio": "4.5",
            "MarketCapitalization": "1000000000",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return b'{"LastDivDate": "2099-01-01", "PERatio": "30", "PriceToBookRatio": "4.5", "MarketCapitalization": "1000000000"}'

        with patch("fetch_data.urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(ValueError):
                fetch_fundamental_snapshot("MSFT", "2024-01-02")

    def test_low_quality_data_forces_analysis_only(self):
        result = orchestrate_score("MSFT", "2024-01-02")
        self.assertIn("analysis_only", result.mode.lower())
        self.assertIn("analysis_only", result.action.lower())

    def test_fundamental_provider_resolution_has_explicit_fallbacks(self):
        with patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": ""}, clear=False):
            provider = resolve_fundamental_provider("MSFT", "2024-01-02")
            self.assertIn("provider", provider)
            self.assertIn("status", provider)
            self.assertEqual(provider["status"], "provider_key_required")

    def test_decision_audit_persistence_writes_event_record(self):
        result = build_score("MSFT", "2024-01-02")
        event = persist_decision_audit({
            "ticker": result.ticker,
            "as_of": result.as_of,
            "mode": "ANALYSIS_ONLY",
            "action": result.action,
            "score": result.score,
            "confidence": result.confidence,
            "replay_hash": "abc123",
            "source_quality": {
                "market_confidence": 0.8,
                "fundamental_confidence": 0.55,
                "effective_confidence": 0.68,
            },
        })
        self.assertIn("event_id", event)
        self.assertIn("ticker", event)
        self.assertEqual(event["ticker"], "MSFT")
        self.assertIn("replay_hash", event)

    def test_score_contract_includes_richer_source_quality_and_replay_metadata(self):
        result = build_score("MSFT", "2024-01-02")
        self.assertIn("source_quality", result.to_dict())
        self.assertIn("replay_metadata", result.to_dict())
        self.assertIn("effective_confidence", result.source_quality)
        self.assertIn("replay_hash", result.replay_metadata)

    def test_provider_health_matrix_tracks_quality_and_status(self):
        health = get_provider_health_matrix()
        self.assertIn("market_data", health)
        self.assertIn("fundamentals", health)
        self.assertIn("health_score", health["market_data"])
        self.assertIn("status", health["fundamentals"])

    def test_audit_retrieval_and_replay_lookup_work(self):
        event = persist_decision_audit({
            "ticker": "MSFT",
            "as_of": "2024-01-02 00:00:00",
            "mode": "ANALYSIS_ONLY",
            "action": "ANALYSIS_ONLY",
            "score": 6.5,
            "confidence": 0.75,
            "replay_hash": "replay-lookup-test",
            "source_quality": {"effective_confidence": 0.69},
        })
        self.assertIn("event_id", event)
        self.assertTrue(get_audit_events(limit=5))
        self.assertIn("replay_hash", get_decision_by_replay_hash("replay-lookup-test")[0])

    def test_replay_lookup_by_ticker_and_as_of(self):
        persist_decision_audit({
            "ticker": "AAPL",
            "as_of": "2024-01-03 00:00:00",
            "mode": "ANALYSIS_ONLY",
            "action": "ANALYSIS_ONLY",
            "score": 7.1,
            "confidence": 0.8,
            "replay_hash": "replay-lookup-by-key",
            "source_quality": {"effective_confidence": 0.77},
        })
        matches = get_decision_by_ticker_and_as_of("aapl", "2024-01-03 00:00:00")
        self.assertTrue(matches)
        self.assertEqual(matches[0]["ticker"], "AAPL")


class ConfidenceModelTests(unittest.TestCase):
    """Regressions for the evidence-based confidence model (confidence-v2).

    These tests are hermetic: _compute_confidence operates on synthetic
    snapshots so factor sensitivity is proven without any network access.
    """

    @staticmethod
    def _snapshot(**overrides):
        base = {
            "as_of": "2026-08-27 00:00:00",
            "last_valid_bar": "2026-08-26 00:00:00",
            "source_confidence": 0.8,
            "data_quality": {"score": 85.0, "bars_available": 250},
            "volatility": 0.02,
            "change_5d": 0.01,
            "change_20d": 0.03,
            "trend_vs_20d_mean": 0.004,
            "price_vs_ma_50": 0.01,
            "price_vs_ma_200": 0.05,
        }
        base.update(overrides)
        return base

    def _confidence(self, snapshot=None, fundamental=None, flags=()):
        confidence, breakdown = _compute_confidence(
            snapshot or self._snapshot(),
            fundamental if fundamental is not None else {
                "source_status": "live_provider", "source_confidence": 0.9
            },
            list(flags),
            governance_risk_gate_passed=True,
        )
        return confidence, breakdown

    def test_breakdown_shape_and_bounds(self):
        confidence, breakdown = self._confidence()
        self.assertEqual(breakdown["calculation_version"], "evidence-confidence-v2")
        self.assertEqual(round(sum(f["weight"] for f in breakdown["factors"]), 4), 1.0)
        self.assertEqual(len(breakdown["factors"]), 6)
        self.assertLessEqual(confidence, breakdown["cap"])
        self.assertGreaterEqual(confidence, breakdown["floor"])
        factors = {f["name"] for f in breakdown["factors"]}
        self.assertEqual(
            factors,
            {"data_quality", "source_reliability", "signal_agreement", "freshness", "history_coverage", "volatility_regime"},
        )

    def test_data_quality_factor_is_sensitivity_tested(self):
        high, _ = self._confidence(self._snapshot(data_quality={"score": 90.0, "bars_available": 250}))
        low, _ = self._confidence(self._snapshot(data_quality={"score": 40.0, "bars_available": 250}))
        self.assertGreater(high, low)

    def test_stale_bars_reduce_confidence(self):
        fresh, _ = self._confidence()
        stale, stale_breakdown = self._confidence(self._snapshot(last_valid_bar="2026-07-10 00:00:00"))
        self.assertGreater(fresh, stale)
        freshness = next(f for f in stale_breakdown["factors"] if f["name"] == "freshness")
        self.assertEqual(freshness["value"], 0.0)

    def test_chaotic_volatility_reduces_confidence(self):
        calm, _ = self._confidence(self._snapshot(volatility=0.015))
        chaotic, _ = self._confidence(self._snapshot(volatility=0.09))
        self.assertGreater(calm, chaotic)


    def test_conflicting_signals_reduce_agreement_below_perfect(self):
        aligned, aligned_breakdown = self._confidence()
        agreement_aligned = next(f for f in aligned_breakdown["factors"] if f["name"] == "signal_agreement")
        self.assertAlmostEqual(agreement_aligned["value"], 1.0)
        mixed, mixed_breakdown = self._confidence(self._snapshot(price_vs_ma_200=-0.05))
        agreement_mixed = next(f for f in mixed_breakdown["factors"] if f["name"] == "signal_agreement")
        self.assertLess(agreement_mixed["value"], agreement_aligned["value"])
        self.assertLess(mixed, aligned)

    def test_penalty_applied_once_per_condition(self):
        _, duplicate_flags = self._confidence(flags=["Weak momentum", "Weak momentum"])
        self.assertEqual(duplicate_flags["total_penalty"], 0.10)
        _, all_flags = self._confidence(flags=["Weak momentum", "Low volume"])
        self.assertAlmostEqual(all_flags["total_penalty"], 0.18)

    def test_fundamental_weakness_penalized_once_not_twice(self):
        _, missing_fundamentals = self._confidence(fundamental={"source_status": "provider_key_required", "source_confidence": 0.0})
        self.assertTrue(missing_fundamentals["fundamental_weak"])
        weak_entries = [p for p in missing_fundamentals["penalties"] if p["name"] == "Fundamental source weak or invalid"]
        self.assertEqual(len(weak_entries), 1)
        self.assertEqual(weak_entries[0]["magnitude"], 0.10)

    def test_live_fundamentals_avoid_penalty_and_raise_confidence(self):
        strong, strong_breakdown = self._confidence(fundamental={"source_status": "live_provider", "source_confidence": 0.9})
        weak, _ = self._confidence(fundamental={"source_status": "fallback_estimate", "source_confidence": 0.5})
        self.assertFalse(strong_breakdown["fundamental_weak"])
        self.assertEqual(strong_breakdown["total_penalty"], 0.0)
        self.assertGreater(strong, weak)

    def test_governance_failure_adds_explicit_penalty(self):
        confidence_ok, _ = self._confidence()
        confidence_failed, failed_breakdown = _compute_confidence(
            self._snapshot(),
            {"source_status": "live_provider", "source_confidence": 0.9},
            [],
            governance_risk_gate_passed=False,
        )
        gate_penalties = [p for p in failed_breakdown["penalties"] if p["name"] == "Governance risk gate failed"]
        self.assertEqual(len(gate_penalties), 1)
        self.assertEqual(gate_penalties[0]["magnitude"], 0.05)
        self.assertFalse(failed_breakdown["risk_gate_passed"])
        self.assertLess(confidence_failed, confidence_ok)

    def test_identical_inputs_are_deterministic(self):
        first = self._confidence()
        second = self._confidence()
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_build_score_exposes_breakdown_in_payload(self):
        result = build_score("MSFT", "2024-01-02")
        payload = result.to_dict()
        self.assertIn("confidence_breakdown", payload)
        self.assertEqual(payload["confidence_breakdown"]["calculation_version"], "evidence-confidence-v2")
        self.assertLessEqual(abs(payload["confidence_breakdown"]["value"] - payload["confidence"]), 0.005)


class MarketDataIntegrityTests(unittest.TestCase):
    """Regressions for market-data feature integrity defects."""

    def test_pct_change_compares_previous_bar_not_itself(self):
        series = pd.Series([10.0, 11.0, 12.0])
        self.assertAlmostEqual(_pct_change(series, 1), 12.0 / 11.0 - 1.0)
        ramp = pd.Series([float(value) for value in range(1, 22)])
        self.assertAlmostEqual(_pct_change(ramp, 5), 21.0 / 16.0 - 1.0)

    def test_pct_change_short_series_is_neutral(self):
        self.assertEqual(_pct_change(pd.Series([10.0]), 1), 0.0)
        self.assertEqual(_pct_change(pd.Series(dtype=float), 1), 0.0)
        # A zero anchor bar must yield a neutral contribution, not a blow-up.
        self.assertEqual(_pct_change(pd.Series([0.0, 9.0]), 1), 0.0)

    @staticmethod
    def _history_frame(days):
        index = pd.DatetimeIndex([pd.Timestamp(day) for day in days])
        values = [10.0 + position * 0.1 for position in range(len(index))]
        return pd.DataFrame(
            {"Open": values, "High": values, "Low": values, "Close": values, "Volume": [250_000.0] * len(index)},
            index=index,
        )

    def test_weekends_and_holidays_are_not_flagged_as_gaps(self):
        # Consecutive calendar sessions never leave more than a 4-day hole,
        # so ordinary weekends/holiday clusters must not degrade quality.
        days = pd.date_range("2026-07-13", periods=35, freq="D").tolist()
        frame = MarketDataIntegrityTests._history_frame(days)
        with patch("agents.market_data_agent.fetch_price_history", return_value=frame):
            snapshot = fetch_market_snapshot("TESTX", "2026-08-17")
        self.assertEqual(snapshot["gaps_detected"], 0)
        self.assertNotIn("market_gap_detected", snapshot["data_quality"]["flags"])

    def test_multi_day_coverage_stall_is_flagged(self):
        days = pd.date_range("2026-06-15", periods=72, freq="D").tolist()
        hole_start, hole_end = pd.Timestamp("2026-07-20"), pd.Timestamp("2026-08-10")
        surviving = [day for day in days if not (hole_start <= day <= hole_end)]
        frame = MarketDataIntegrityTests._history_frame(surviving)
        with patch("agents.market_data_agent.fetch_price_history", return_value=frame):
            snapshot = fetch_market_snapshot("TESTX", "2026-08-26")
        self.assertEqual(snapshot["gaps_detected"], 1)
        self.assertIn("market_gap_detected", snapshot["data_quality"]["flags"])


class EnsembleBlendTests(unittest.TestCase):
    """W1: versioned ensemble wiring — weights, renormalization, breakdown."""

    @staticmethod
    def _contributions(**overrides):
        base = {
            "market_data": {"score_current": 10.0, "score_long": 10.0, "status": "OK", "note": "informational"},
            "technical_analysis": {"score_current": 8.0, "score_long": 6.0, "status": "OK", "note": ""},
            "fundamental_analysis": {"score_current": 5.0, "score_long": 5.0, "status": "OK", "note": ""},
            "news_intelligence": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "sentiment": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "macroeconomic": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "market_regime": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        }
        base.update(overrides)
        return base

    def _blend(self, contributions=None):
        return _ensemble_blend(
            contributions if contributions is not None else self._contributions(),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )

    def test_weight_sets_share_keys_sum_to_one_and_are_non_negative(self):
        for weights in (core_config.ENSEMBLE_WEIGHTS_CURRENT, core_config.ENSEMBLE_WEIGHTS_LONG):
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
            self.assertTrue(all(weight >= 0.0 for weight in weights.values()))
        self.assertEqual(set(core_config.ENSEMBLE_WEIGHTS_CURRENT), set(core_config.ENSEMBLE_WEIGHTS_LONG))
        self.assertEqual(
            set(core_config.ENSEMBLE_WEIGHTS_CURRENT),
            {"market_data", "technical_analysis", "fundamental_analysis", "news_intelligence", "sentiment", "macroeconomic", "market_regime"},
        )
        self.assertTrue(core_config.ENSEMBLE_VERSION)

    def test_validation_helper_rejects_invalid_weight_sets(self):
        validate = core_config._validate_ensemble_weights
        complete = dict(core_config.ENSEMBLE_WEIGHTS_CURRENT)
        with self.assertRaises(ValueError):
            validate("empty", {})
        with self.assertRaises(ValueError):
            validate("key_mismatch", {key: value for key, value in complete.items() if key != "sentiment"})
        with self.assertRaises(ValueError):
            validate("bad_sum", {**complete, "technical_analysis": 0.5})
        with self.assertRaises(ValueError):
            validate("negative", {**complete, "sentiment": -0.1})

    def test_blend_matches_weighted_math(self):
        current, long_term, breakdown = self._blend()
        self.assertEqual(current, 0.85 * 8.0 + 0.15 * 5.0)
        self.assertEqual(long_term, 0.75 * 6.0 + 0.25 * 5.0)
        self.assertEqual(breakdown["current_time_score"], current)
        self.assertEqual(breakdown["long_term_score"], long_term)
        self.assertFalse(breakdown["no_eligible_agents"])
        technical = breakdown["agents"]["technical_analysis"]
        self.assertAlmostEqual(technical["effective_weight_current"], 0.85, places=9)
        contributions_sum = (
            technical["contribution_current"]
            + breakdown["agents"]["fundamental_analysis"]["contribution_current"]
        )
        self.assertAlmostEqual(contributions_sum, current, places=9)


    def test_renormalization_when_fundamental_unavailable(self):
        contributions = self._contributions(
            fundamental_analysis={"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": "future-dated payload"},
        )
        current, long_term, breakdown = self._blend(contributions)
        self.assertEqual(current, 8.0)
        self.assertEqual(long_term, 6.0)
        fundamental = breakdown["agents"]["fundamental_analysis"]
        self.assertAlmostEqual(fundamental["raw_weight_current"], 0.15, places=9)
        self.assertEqual(fundamental["effective_weight_current"], 0.0)
        self.assertFalse(fundamental["eligible_current"])
        self.assertAlmostEqual(
            sum(entry["effective_weight_current"] for entry in breakdown["agents"].values()), 1.0, places=9
        )
        self.assertAlmostEqual(
            sum(entry["effective_weight_long"] for entry in breakdown["agents"].values()), 1.0, places=9
        )

    def test_zero_weight_informational_agent_cannot_move_score(self):
        baseline_current, baseline_long, _ = self._blend()
        degraded_current, degraded_long, degraded_breakdown = self._blend(
            self._contributions(
                market_data={"score_current": 0.0, "score_long": 0.0, "status": "INCOMPLETE", "note": "degraded"},
            )
        )
        self.assertEqual(degraded_current, baseline_current)
        self.assertEqual(degraded_long, baseline_long)
        market_entry = degraded_breakdown["agents"]["market_data"]
        self.assertEqual(market_entry["contribution_current"], 0.0)
        self.assertEqual(market_entry["status"], "INCOMPLETE")

    def test_none_score_is_excluded_not_coerced_to_zero(self):
        contributions = self._contributions(
            technical_analysis={"score_current": None, "score_long": 6.0, "status": "OK", "note": ""},
        )
        current, long_term, breakdown = self._blend(contributions)
        self.assertEqual(current, 5.0)  # fundamental alone at effective weight 1.0
        self.assertEqual(long_term, 0.75 * 6.0 + 0.25 * 5.0)
        technical = breakdown["agents"]["technical_analysis"]
        self.assertFalse(technical["eligible_current"])
        self.assertTrue(technical["eligible_long"])

    def test_no_eligible_agents_flags_and_zeroes(self):
        contributions = self._contributions()
        for info in contributions.values():
            info["status"] = "UNAVAILABLE"
            info["score_current"] = None
            info["score_long"] = None
        current, long_term, breakdown = self._blend(contributions)
        self.assertEqual(current, 0.0)
        self.assertEqual(long_term, 0.0)
        self.assertTrue(breakdown["no_eligible_agents"])

    def test_weight_changes_move_the_score(self):
        contributions = self._contributions()
        all_technical = {agent: (1.0 if agent == "technical_analysis" else 0.0) for agent in core_config.ENSEMBLE_WEIGHTS_CURRENT}
        all_fundamental = {agent: (1.0 if agent == "fundamental_analysis" else 0.0) for agent in core_config.ENSEMBLE_WEIGHTS_CURRENT}
        tech_current, _, _ = _ensemble_blend(contributions, all_technical, core_config.ENSEMBLE_WEIGHTS_LONG)
        fund_current, _, _ = _ensemble_blend(contributions, all_fundamental, core_config.ENSEMBLE_WEIGHTS_LONG)
        self.assertEqual(tech_current, 8.0)
        self.assertEqual(fund_current, 5.0)

    def test_identical_inputs_are_deterministic(self):
        self.assertEqual(self._blend(), self._blend())

    def test_build_score_exposes_ensemble_breakdown(self):
        result = build_score("MSFT", "2024-01-02")
        breakdown = result.ensemble_breakdown
        self.assertEqual(breakdown["calculation_version"], "ensemble-v1")
        self.assertFalse(breakdown["no_eligible_agents"])
        self.assertEqual(len(breakdown["agents"]), 7)
        self.assertAlmostEqual(
            sum(entry["effective_weight_current"] for entry in breakdown["agents"].values()), 1.0, places=6
        )
        self.assertAlmostEqual(
            sum(entry["effective_weight_long"] for entry in breakdown["agents"].values()), 1.0, places=6
        )
        self.assertEqual(breakdown["current_time_score"], result.current_time_score)
        self.assertEqual(breakdown["long_term_score"], result.long_term_score)
        technical = breakdown["agents"]["technical_analysis"]
        self.assertTrue(technical["eligible_current"])
        self.assertIn("canonical", technical["note"])


class RiskPolicyTests(unittest.TestCase):
    """W2: fail-closed risk policy — rules, severities, mode invariant."""

    LEGACY_VETO_IDS = {
        "data_quality_below_threshold",
        "market_source_confidence_below_threshold",
        "future_dated_market_data",
        "future_dated_fundamental_payload",
        "fundamental_source_confidence_below_threshold",
        "analysis_only_mode",
        "score_below_threshold",
    }

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

    def test_clean_context_triggers_nothing(self):
        evaluation = evaluate_risk_policy(self._context())
        self.assertFalse(evaluation["veto"])
        self.assertEqual(evaluation["veto_rule_ids"], [])
        self.assertEqual(evaluation["warning_rule_ids"], [])
        self.assertEqual(evaluation["policy_version"], "risk-policy-v2")
        self.assertTrue(all(not rule["triggered"] for rule in evaluation["rules"]))

    def test_every_veto_rule_triggers_when_forced(self):
        forcing = {
            "data_quality_below_threshold": {"market_data_quality": 50.0},
            "market_source_confidence_below_threshold": {"market_source_confidence": 0.5},
            "future_dated_market_data": {"market_timestamp_valid": False},
            "future_dated_fundamental_payload": {"fundamental_point_in_time_valid": False},
            "fundamental_source_confidence_below_threshold": {
                "fundamental_source_confidence": 0.5,
                "fundamental_source_status": "provider_key_required",
            },
            "score_below_threshold": {"score": 4.0},
            "analysis_only_mode": {"action": "ANALYSIS_ONLY"},
            "confidence_below_minimum": {"confidence": 0.3},
        }
        for rule_id, overrides in forcing.items():
            with self.subTest(rule_id=rule_id):
                evaluation = evaluate_risk_policy(self._context(**overrides))
                rule = next(rule for rule in evaluation["rules"] if rule["rule_id"] == rule_id)
                self.assertTrue(rule["triggered"], rule["detail"])
                self.assertEqual(rule["severity"], "veto")
                self.assertIn(rule_id, evaluation["veto_rule_ids"])
                self.assertTrue(evaluation["veto"])

    def test_fail_closed_missing_inputs_trigger_every_rule(self):
        evaluation = evaluate_risk_policy({})
        self.assertTrue(evaluation["veto"])
        self.assertTrue(all(rule["triggered"] for rule in evaluation["rules"]))
        self.assertIn("missing", " ".join(rule["detail"] for rule in evaluation["rules"]).lower())
        expected = {
            rule_id for rule_id, spec in core_config.RISK_POLICY_V2.items()
            if spec["severity"] == "veto"
        }
        self.assertEqual(set(evaluation["veto_rule_ids"]), expected)

    def test_warning_rules_do_not_veto(self):
        evaluation = evaluate_risk_policy(self._context(
            confidence_breakdown={
                "total_penalty": 0.20,
                "factors": [
                    {"name": "freshness", "value": 0.2},
                    {"name": "volatility_regime", "value": 0.1},
                ],
            },
        ))
        self.assertFalse(evaluation["veto"])
        self.assertEqual(evaluation["veto_rule_ids"], [])
        self.assertEqual(
            set(evaluation["warning_rule_ids"]),
            {"confidence_penalty_budget_exceeded", "freshness_degraded", "volatility_regime_elevated"},
        )


    def test_legacy_veto_ids_are_preserved_on_degraded_context(self):
        evaluation = evaluate_risk_policy(self._context(
            market_data_quality=50.0,
            market_source_confidence=0.5,
            market_timestamp_valid=False,
            fundamental_point_in_time_valid=False,
            fundamental_source_confidence=0.1,
            fundamental_source_status="provider_key_required",
            score=4.0,
            action="ANALYSIS_ONLY",
            confidence=0.8,
        ))
        self.assertTrue(self.LEGACY_VETO_IDS.issubset(set(evaluation["veto_rule_ids"])))

    def test_confidence_boundary_passes_at_exact_threshold(self):
        at_threshold = evaluate_risk_policy(self._context(confidence=0.35))
        below_threshold = evaluate_risk_policy(self._context(confidence=0.34))
        confidence_rule_at = next(rule for rule in at_threshold["rules"] if rule["rule_id"] == "confidence_below_minimum")
        confidence_rule_below = next(rule for rule in below_threshold["rules"] if rule["rule_id"] == "confidence_below_minimum")
        self.assertFalse(confidence_rule_at["triggered"])
        self.assertTrue(confidence_rule_below["triggered"])

    def test_select_mode_invariant_no_paper_while_veto_fires(self):
        for rule_id in core_config.RISK_POLICY_V2:
            with self.subTest(rule_id=rule_id):
                self.assertNotEqual(_select_mode("PAPER", [rule_id], True), "PAPER")
        self.assertEqual(_select_mode("PAPER", [], True), "PAPER")
        self.assertEqual(_select_mode("ANALYSIS_ONLY", [], True), "ANALYSIS_ONLY")
        self.assertEqual(_select_mode("PAPER", [], False), "NO_TRADE")

    def test_orchestrator_risk_agent_carries_policy_evaluation(self):
        decision = orchestrate_score("MSFT", "2024-01-02")
        risk_agent = next(agent for agent in decision.agent_outputs if agent.agent == "risk_management")
        payload = risk_agent.payload
        self.assertEqual(payload["policy_version"], "risk-policy-v2")
        self.assertEqual(risk_agent.model_version, "risk-policy-v2")
        self.assertIn(risk_agent.status, {"OK", "VETO"})
        self.assertTrue(payload["rules"])
        for rule in payload["rules"]:
            self.assertIn(rule["severity"], {"veto", "warning"})
            self.assertIsInstance(rule["triggered"], bool)
            self.assertTrue(rule["detail"])
        self.assertEqual(decision.veto_reasons, payload["veto_rule_ids"])


class AuditPolicyTests(unittest.TestCase):
    """W3: auditor as an independent evidence/replay validator."""

    @staticmethod
    def _agents(**tamper):
        names = ["market_data", "technical_analysis", "fundamental_analysis", "news_intelligence", "risk_management"]
        agents = []
        for name in names:
            agent = {
                "agent": name,
                "status": "OK",
                "evidence": [{"source_record_id": f"{name}_source", "reason": "ok"}],
                "input_hash": "a" * 64,
            }
            if name in tamper:
                agent.update(tamper[name])
            agents.append(agent)
        return agents

    @staticmethod
    def _fake_result(**overrides):
        result = {
            "ticker": "MSFT",
            "as_of": "2026-08-27 00:00:00",
            "score": 7.0,
            "current_time_score": 7.5,
            "long_term_score": 6.5,
            "confidence": 0.8,
            "confidence_breakdown": {"value": 0.8, "calculation_version": "evidence-confidence-v2"},
            "ensemble_breakdown": {
                "current_time_score": 7.5,
                "long_term_score": 6.5,
                "no_eligible_agents": False,
                "agents": {
                    "technical_analysis": {"effective_weight_current": 1.0, "effective_weight_long": 1.0},
                },
            },
            "governance": {"risk_gate_passed": True},
            "evidence_ledger": {"status": "ready"},
            "replay_metadata": {"audit_event_id": "evt-1"},
        }
        result.update(overrides)
        return result

    def _context(self, **overrides):
        snapshot = overrides.pop("snapshot", {"ticker": "MSFT", "close": 100.0})
        context = {
            "ticker": "MSFT",
            "as_of": "2026-08-27 00:00:00",
            "agents": self._agents(),
            "expected_input_hashes": {
                "market_data": "a" * 64,
                "technical_analysis": "a" * 64,
                "fundamental_analysis": "a" * 64,
                "news_intelligence": "a" * 64,
                "risk_management": "a" * 64,
            },
            "snapshot": snapshot,
            "snapshot_hash": stable_hash(snapshot),
            "replay_hash": "r" * 64,
            "first_result": self._fake_result(),
            "second_result": self._fake_result(),
            "confidence": 0.8,
            "confidence_breakdown": {"value": 0.8, "calculation_version": "evidence-confidence-v2"},
            "ensemble_breakdown": self._fake_result()["ensemble_breakdown"],
            "current_time_score": 7.5,
            "long_term_score": 6.5,
            "governance": {"risk_gate_passed": True},
            "evidence_ledger": {"status": "ready"},
        }
        context.update(overrides)
        return context


    def test_clean_context_passes_all_checks(self):
        evaluation = evaluate_audit_policy(self._context())
        self.assertFalse(evaluation["veto"])
        self.assertEqual(evaluation["veto_check_ids"], [])
        self.assertEqual(evaluation["policy_version"], "audit-policy-v1")
        self.assertTrue(all(finding["passed"] for finding in evaluation["findings"]))

    def test_tampered_agent_hash_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(
            agents=self._agents(technical_analysis={"input_hash": "b" * 64}),
        ))
        self.assertTrue(evaluation["veto"])
        self.assertIn("agent_input_hash_integrity", evaluation["veto_check_ids"])

    def test_tampered_snapshot_hash_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(snapshot_hash="deadbeef"))
        self.assertTrue(evaluation["veto"])
        self.assertIn("snapshot_hash_integrity", evaluation["veto_check_ids"])

    def test_determinism_mismatch_vetoes_with_path(self):
        evaluation = evaluate_audit_policy(self._context(
            second_result=self._fake_result(score=9.9),
        ))
        self.assertTrue(evaluation["veto"])
        self.assertIn("determinism_probe", evaluation["veto_check_ids"])
        probe = next(f for f in evaluation["findings"] if f["check_id"] == "determinism_probe")
        self.assertIn("$.score", probe["detail"])

    def test_missing_probe_build_fails_closed(self):
        evaluation = evaluate_audit_policy(self._context(second_result=None))
        self.assertTrue(evaluation["veto"])
        self.assertIn("determinism_probe", evaluation["veto_check_ids"])

    def test_calibration_out_of_band_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(confidence=0.99))
        self.assertTrue(evaluation["veto"])
        self.assertIn("calibration_sanity", evaluation["veto_check_ids"])

    def test_calibration_breakdown_disagreement_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(
            confidence_breakdown={"value": 0.5, "calculation_version": "evidence-confidence-v2"},
        ))
        self.assertTrue(evaluation["veto"])
        self.assertIn("calibration_sanity", evaluation["veto_check_ids"])

    def test_ensemble_inconsistency_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(
            ensemble_breakdown=self._fake_result(ensemble_breakdown={})["ensemble_breakdown"],
        ))
        self.assertTrue(evaluation["veto"])
        self.assertIn("ensemble_consistency", evaluation["veto_check_ids"])

    def test_evidence_sufficiency_vetoes(self):
        evaluation = evaluate_audit_policy(self._context(
            agents=self._agents(market_data={"evidence": []}),
        ))
        self.assertTrue(evaluation["veto"])
        self.assertIn("evidence_sufficiency", evaluation["veto_check_ids"])

    def test_ledger_mismatch_is_warning_only(self):
        evaluation = evaluate_audit_policy(self._context(
            governance={"risk_gate_passed": False},
        ))
        self.assertFalse(evaluation["veto"])
        self.assertIn("evidence_ledger_consistency", evaluation["warning_check_ids"])

    def test_orchestrator_audit_agent_passes_on_clean_run(self):
        decision = orchestrate_score("MSFT", "2024-01-02")
        audit_agent = next(agent for agent in decision.agent_outputs if agent.agent == "performance_auditor")
        self.assertEqual(audit_agent.status, "OK")
        self.assertEqual(audit_agent.payload["policy_version"], "audit-policy-v1")
        self.assertFalse(audit_agent.payload["veto"])
        self.assertNotIn("auditor_veto", decision.veto_reasons)
        self.assertTrue(all(finding["passed"] for finding in audit_agent.payload["findings"]))

    def test_auditor_veto_blocks_paper(self):
        self.assertNotEqual(_select_mode("PAPER", ["auditor_veto"], True), "PAPER")


class StatusTaxonomyTests(unittest.TestCase):
    """W4: failure-state taxonomy — validation, posture mapping, propagation."""

    @staticmethod
    def _score_result_stub(confidence_breakdown=None, news_status="UNAVAILABLE"):
        factors = confidence_breakdown or {"factors": [{"name": "freshness", "value": 1.0}]}
        return SimpleNamespace(
            confidence_breakdown=factors,
            news_snapshot={"status": news_status},
        )

    @staticmethod
    def _snapshot(**over):
        snapshot = {
            "data_quality": {"score": 85.0},
            "source_contract": {"timestamp_valid": True},
        }
        snapshot.update(over)
        return snapshot

    @staticmethod
    def _fundamental(pit_valid=True):
        return {"point_in_time_valid": pit_valid}

    def test_agent_contract_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            AgentContract(agent="x", ticker="X", as_of="t", status="BANANA")

    def test_agent_contract_accepts_every_taxonomy_status(self):
        for status in AgentStatus:
            with self.subTest(status=status.value):
                contract = AgentContract(agent="x", ticker="X", as_of="t", status=status.value)
                self.assertEqual(contract.status, status.value)

    def test_status_posture_table(self):
        expected = {
            "OK": "PAPER",
            "UNAVAILABLE": "ANALYSIS_ONLY",
            "STALE": "ANALYSIS_ONLY",
            "INCOMPLETE": "ANALYSIS_ONLY",
            "CONTRADICTORY": "NO_TRADE",
            "INVALID": "NO_TRADE",
            "VETO": "NO_TRADE",
        }
        self.assertEqual(STATUS_POSTURE, {AgentStatus(status): posture for status, posture in expected.items()})
        for status, posture in expected.items():
            with self.subTest(status=status):
                self.assertEqual(status_posture(status), posture)

    def test_worst_status_ordering(self):
        self.assertEqual(worst_status([]), "OK")
        self.assertEqual(worst_status(["OK", "OK"]), "OK")
        self.assertEqual(worst_status(["OK", "STALE"]), "STALE")
        self.assertEqual(worst_status(["INCOMPLETE", "STALE"]), "STALE")
        self.assertEqual(worst_status(["STALE", "CONTRADICTORY"]), "CONTRADICTORY")
        self.assertEqual(worst_status(["CONTRADICTORY", "INVALID"]), "INVALID")
        self.assertEqual(worst_status(["OK", "INVALID", "STALE"]), "INVALID")

    def test_derive_healthy_snapshot_is_ok(self):
        statuses = _derive_agent_statuses(self._snapshot(), self._fundamental(), self._score_result_stub())
        self.assertEqual(statuses["market_data"], "OK")
        self.assertEqual(statuses["fundamental_analysis"], "OK")
        self.assertEqual(statuses["technical_analysis"], "OK")

    def test_derive_future_dated_market_data_is_invalid(self):
        statuses = _derive_agent_statuses(
            self._snapshot(source_contract={"timestamp_valid": False}),
            self._fundamental(),
            self._score_result_stub(),
        )
        self.assertEqual(statuses["market_data"], "INVALID")

    def test_derive_stale_freshness(self):
        stub = self._score_result_stub({"factors": [{"name": "freshness", "value": 0.2}]})
        statuses = _derive_agent_statuses(self._snapshot(), self._fundamental(), stub)
        self.assertEqual(statuses["market_data"], "STALE")

    def test_derive_low_quality_is_incomplete(self):
        statuses = _derive_agent_statuses(
            self._snapshot(data_quality={"score": 50.0}),
            self._fundamental(),
            self._score_result_stub(),
        )
        self.assertEqual(statuses["market_data"], "INCOMPLETE")

    def test_derive_missing_quality_is_incomplete(self):
        statuses = _derive_agent_statuses(
            self._snapshot(data_quality={}),
            self._fundamental(),
            self._score_result_stub(),
        )
        self.assertEqual(statuses["market_data"], "INCOMPLETE")

    def test_derive_future_dated_fundamentals_are_invalid(self):
        statuses = _derive_agent_statuses(self._snapshot(), self._fundamental(pit_valid=False), self._score_result_stub())
        self.assertEqual(statuses["fundamental_analysis"], "INVALID")

    def test_posture_propagation_blocks_paper_on_invalid_agent(self):
        self.assertNotEqual(_select_mode("PAPER", ["agent_status_no_trade"], True), "PAPER")

    def test_orchestrator_statuses_are_valid_taxonomy_values(self):
        decision = orchestrate_score("MSFT", "2024-01-02")
        for agent in decision.agent_outputs:
            with self.subTest(agent=agent.agent):
                AgentStatus(agent.status)  # raises on unknown status


class TechnicalTruthTests(unittest.TestCase):
    """W5: one technical truth — score_technical is the canonical blend."""

    @staticmethod
    def _snapshot(**over):
        snapshot = {
            "close": 100.0,
            "change_1d": 0.01,
            "change_5d": 0.02,
            "change_20d": 0.05,
            "change_60d": 0.10,
            "trend_vs_20d_mean": 0.01,
            "rsi": 55.0,
            "volume": 1_000_000.0,
            "avg_volume_20d": 900_000.0,
            "volume_ratio_20d": 1.1,
            "volatility": 0.02,
            "price_vs_ma_50": 0.02,
            "price_vs_ma_100": 0.03,
            "price_vs_ma_150": 0.04,
            "price_vs_ma_200": 0.05,
            "moving_averages": {"50d": 98.0, "100d": 97.0, "150d": 96.0, "200d": 95.0},
        }
        snapshot.update(over)
        return snapshot

    def test_score_technical_is_canonical_blend(self):
        snapshot = self._snapshot()
        news = {"status": "UNAVAILABLE"}
        expected = round((_score_current_time(snapshot, news) + _score_long_term(snapshot)) / 2.0, 2)
        self.assertEqual(score_technical(snapshot, news), expected)

    def test_score_technical_defaults_to_unavailable_news(self):
        snapshot = self._snapshot()
        self.assertEqual(
            score_technical(snapshot),
            score_technical(snapshot, {"status": "UNAVAILABLE"}),
        )

    def test_news_snapshot_flows_through_the_agent_view(self):
        snapshot = self._snapshot()
        with_news = score_technical(snapshot, {"status": "OK", "sentiment_score": 1.0})
        without_news = score_technical(snapshot)
        self.assertGreater(with_news, without_news)

    def test_orchestrator_technical_agent_matches_canonical_blend(self):
        result = build_score("MSFT", "2024-01-02")
        decision = orchestrate_score("MSFT", "2024-01-02")
        technical_agent = next(a for a in decision.agent_outputs if a.agent == "technical_analysis")
        technical = result.ensemble_breakdown["agents"]["technical_analysis"]
        expected = round((technical["score_current"] + technical["score_long"]) / 2.0, 2)
        self.assertEqual(technical_agent.score, expected)
        # Rounding envelope around the exact blend of the canonical views.
        self.assertLessEqual(
            abs(technical_agent.score - (technical["score_current"] + technical["score_long"]) / 2.0),
            0.005,
        )


class RawStoreTests(unittest.TestCase):
    """W6: append-only raw store — append, supersede, fail-soft, rebuild."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch("core.raw_store.RAW_STORE_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_append_then_load_round_trip(self):
        records = [{"bar_time": "2026-08-25 13:30:00", "Close": 100.0}]
        self.assertTrue(append_raw_records("yahoo_finance_chart", "MSFT_1y_1d", records))
        entries = load_raw_records("yahoo_finance_chart", "MSFT_1y_1d")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["record"]["Close"], 100.0)
        self.assertNotIn("superseded_by", entries[0])
        for key in ("ingested_time", "payload_sha256", "record"):
            self.assertIn(key, entries[0])

    def test_request_key_filtering(self):
        append_raw_records("yahoo_finance_chart", "MSFT_1y_1d", [{"v": 1}])
        append_raw_records("yahoo_finance_chart", "VOO_6mo_1d", [{"v": 2}])
        self.assertEqual(len(load_raw_records("yahoo_finance_chart", "MSFT_1y_1d")), 1)
        self.assertEqual(len(load_raw_records("yahoo_finance_chart", "VOO_6mo_1d")), 1)
        self.assertEqual(load_raw_records("yahoo_finance_chart", "UNKNOWN"), [])

    def test_supersede_on_refetch_marks_older_versions(self):
        append_raw_records("yahoo_finance_chart", "K", [{"bar_time": "2026-08-25 13:30:00", "Close": 100.0}])
        append_raw_records("yahoo_finance_chart", "K", [{"bar_time": "2026-08-25 13:30:00", "Close": 101.0}])
        entries = load_raw_records("yahoo_finance_chart", "K", record_key_field="bar_time")
        self.assertEqual(len(entries), 2)
        superseded = [entry for entry in entries if "superseded_by" in entry]
        self.assertEqual(len(superseded), 1)
        self.assertEqual(superseded[0]["record"]["Close"], 100.0)
        self.assertNotEqual(superseded[0]["payload_sha256"], superseded[0]["superseded_by"])

    def test_write_failure_never_raises(self):
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("not a directory")
        with patch("core.raw_store.RAW_STORE_DIR", blocker):
            self.assertFalse(append_raw_records("yahoo_finance_chart", "K", [{"v": 1}]))

    def test_rebuild_price_frame_uses_latest_versions(self):
        append_raw_records("yahoo_finance_chart", "K", [
            {"bar_time": "2026-08-25 13:30:00", "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 100.0, "Volume": 10.0},
            {"bar_time": "2026-08-26 13:30:00", "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 101.0, "Volume": 11.0},
        ])
        append_raw_records("yahoo_finance_chart", "K", [
            {"bar_time": "2026-08-25 13:30:00", "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 105.0, "Volume": 12.0},
        ])
        frame = rebuild_price_frame("yahoo_finance_chart", "K")
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.index.name, "Date")
        self.assertEqual(float(frame.loc["2026-08-25 13:30:00", "Close"]), 105.0)
        self.assertEqual(float(frame.loc["2026-08-26 13:30:00", "Close"]), 101.0)


class RawStoreRebuildTests(unittest.TestCase):
    """W6 acceptance: the market snapshot rebuilds purely from raw records."""

    def test_market_snapshot_rebuilds_from_raw_store(self):
        cache_path = Path("data") / "MSFT_1y_1d.parquet"
        if not cache_path.exists():
            self.skipTest("MSFT 1y cache not present on this machine")
        baseline_frame = pd.read_parquet(cache_path)
        baseline = fetch_market_snapshot("MSFT", "2026-08-27")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("core.raw_store.RAW_STORE_DIR", Path(tmp)):
                rows = [
                    {
                        "bar_time": index.strftime("%Y-%m-%d %H:%M:%S"),
                        "Open": float(row["Open"]),
                        "High": float(row["High"]),
                        "Low": float(row["Low"]),
                        "Close": float(row["Close"]),
                        "Volume": float(row["Volume"]),
                    }
                    for index, row in baseline_frame.iterrows()
                ]
                self.assertTrue(append_raw_records("yahoo_finance_chart", "MSFT_1y_1d", rows))
                rebuilt = rebuild_price_frame("yahoo_finance_chart", "MSFT_1y_1d")
        self.assertIsNotNone(rebuilt)
        pd.testing.assert_frame_equal(rebuilt, baseline_frame, check_dtype=False, check_freq=False)

        cache_path.unlink()
        rebuilt.to_parquet(cache_path)
        rebuilt_snapshot = fetch_market_snapshot("MSFT", "2026-08-27")

        for key in (
            "close", "rsi", "volatility", "change_1d", "change_5d", "change_20d",
            "change_60d", "trend_vs_20d_mean", "volume_ratio_20d",
            "price_vs_ma_50", "price_vs_ma_100", "price_vs_ma_150", "price_vs_ma_200",
        ):
            self.assertEqual(baseline[key], rebuilt_snapshot[key], key)
        self.assertEqual(baseline["moving_averages"], rebuilt_snapshot["moving_averages"])
        self.assertEqual(baseline["data_quality"]["score"], rebuilt_snapshot["data_quality"]["score"])
        self.assertEqual(baseline["data_quality"]["flags"], rebuilt_snapshot["data_quality"]["flags"])


if __name__ == "__main__":
    unittest.main()
