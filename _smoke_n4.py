"""Temporary N4 smoke check — deleted after the sprint lands."""
from unittest.mock import patch

import pandas as pd

from core.config import REGIME_RISKOFF_MOMENTUM_DAMPING
from core.orchestrator import orchestrate_score
from core.regime_agent import build_regime_snapshot
from core.regime_contract import fetch_regime_snapshot
from core.risk_policy import evaluate_risk_policy


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1_000_000.0] * len(closes)},
        index=index,
    )


def _series_from_returns(returns, base=100.0):
    closes = [base]
    for daily_return in returns:
        closes.append(closes[-1] * (1.0 + daily_return))
    return closes[1:]


STORM_UP = [0.004 if i % 2 == 0 else -0.002 for i in range(290)]
CALM_UP = [0.001] * 30
STRESS_TAIL = _series_from_returns([-0.016 if i % 2 == 0 else 0.003 for i in range(28)])

BULLISH = _frame(_series_from_returns(STORM_UP + CALM_UP))
AS_OF = BULLISH.index[-1].strftime("%Y-%m-%d %H:%M:%S")
STRESS = _frame([100.0] * 300 + STRESS_TAIL)
AS_OF_STRESS = STRESS.index[-1].strftime("%Y-%m-%d %H:%M:%S")


def _regime_stub(status, label=None, proxy=None):
    stub = {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": status,
        "source_id": "yahoo_finance_chart", "source_confidence": 0.8,
        "published_time": "2024-01-02 00:00:00", "calculation_version": "regime-contract-v1",
        "lookback_period": "300 sessions", "regime": label, "probability_proxy": proxy,
        "transition_risk": {"window_sessions": 20, "labeled_sessions": 20, "flips": 0,
                            "flip_rate": 0.0, "labels": [label or "bullish"] * 20},
        "inputs": {}, "rule_trace": {}, "reason": "stub",
    }
    if status == "OK":
        stub["pipeline"] = {
            "pipeline_version": "regime-pipeline-v1",
            "classifier_version": "regime-classifier-v1",
            "counts": {"sessions_used": 300, "labeled_sessions": 18, "transition_labels": 20},
        }
    return stub


with patch("core.regime_agent.fetch_price_history", return_value=BULLISH):
    snapshot = fetch_regime_snapshot("TEST", AS_OF)
assert snapshot["status"] == "OK" and snapshot["regime"] == "bullish", snapshot["status"]
assert snapshot["pipeline"]["classifier_version"] == "regime-classifier-v1"
print("1. OK path (bullish uptrend):", snapshot["regime"], "proxy", snapshot["probability_proxy"])

with patch("core.regime_agent.fetch_price_history", return_value=STRESS):
    stress = build_regime_snapshot("TEST", AS_OF_STRESS)
assert stress["regime"] == "stress", stress["regime"]
assert stress["inputs"]["drawdown_from_60d_high"] > 0.15
assert stress["transition_risk"]["flips"] >= 1
print("2. stress path: drawdown", stress["inputs"]["drawdown_from_60d_high"],
      "flips", stress["transition_risk"]["flips"])

with patch("core.regime_agent.fetch_price_history", side_effect=RuntimeError("down")):
    unavailable = build_regime_snapshot("TEST", AS_OF)
assert unavailable["status"] == "UNAVAILABLE" and unavailable["regime"] is None
print("3. UNAVAILABLE path: explicit None regime (never neutral)")

with patch("core.regime_agent.fetch_price_history", return_value=BULLISH.iloc[:150]):
    incomplete = build_regime_snapshot("TEST", AS_OF)
assert incomplete["status"] == "INCOMPLETE" and incomplete["regime"] is None
print("4. INCOMPLETE path: short history ->", incomplete["status"])

healthy = {"market_data_quality": 85.0, "market_source_confidence": 0.8, "market_timestamp_valid": True,
           "fundamental_point_in_time_valid": True, "fundamental_source_confidence": 0.9,
           "fundamental_source_status": "live_provider", "score": 6.5, "action": "PAPER",
           "confidence": 0.8, "market_regime": "bullish",
           "confidence_breakdown": {"total_penalty": 0.0, "factors": []}}
assert "market_regime_stress" not in evaluate_risk_policy(healthy)["veto_rule_ids"]
assert "market_regime_stress" in evaluate_risk_policy({**healthy, "market_regime": "stress"})["veto_rule_ids"]
assert "market_regime_stress" in evaluate_risk_policy({k: v for k, v in healthy.items() if k != "market_regime"})["veto_rule_ids"]
print("5. veto coupling: bullish clean / stress vetoes / missing label fails closed")

with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "stress", 0.8)):
    stress_decision = orchestrate_score("MSFT", "2024-01-02")
assert stress_decision.mode == "NO_TRADE" and "market_regime_stress" in stress_decision.veto_reasons
print("6. stress snapshot cannot reach PAPER:", stress_decision.mode)

with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "bullish", 0.9)):
    bullish_decision = orchestrate_score("MSFT", "2024-01-02")
assert "market_regime_stress" not in bullish_decision.veto_reasons
assert len(bullish_decision.agent_outputs) == 9
regime_agent = next(a for a in bullish_decision.agent_outputs if a.agent == "market_regime")
assert regime_agent.status == "OK" and regime_agent.payload["classifier_version"] == "regime-classifier-v1"
print("7. bullish run: 9 agents, regime agent born wired, mode", bullish_decision.mode)

with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub("OK", "risk_off", 0.5)):
    risk_off = orchestrate_score("MSFT", "2024-01-02")
assert "market_regime_stress" not in risk_off.veto_reasons
print("8. risk_off run: dampening", REGIME_RISKOFF_MOMENTUM_DAMPING,
      "current_time_score", risk_off.current_time_score, "vs bullish", bullish_decision.current_time_score)

print("ALL SMOKE CHECKS PASSED")