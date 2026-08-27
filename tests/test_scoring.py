from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from agents.market_data_agent import fetch_market_snapshot
from core.agent_contracts import AgentContract, OrchestrationDecision
from core.audit_store import get_audit_events, get_decision_by_replay_hash, get_decision_by_ticker_and_as_of, persist_decision_audit
from core.orchestrator import orchestrate_score
from core.score_engine import build_score, _compute_confidence
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


if __name__ == "__main__":
    unittest.main()
