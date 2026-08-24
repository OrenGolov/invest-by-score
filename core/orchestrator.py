from __future__ import annotations

import hashlib
import json

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.agent_contracts import AgentContract, OrchestrationDecision
from core.score_engine import build_score
from fetch_data import fetch_fundamental_snapshot


def _stable_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def orchestrate_score(ticker: str, as_of: str, timestamp: str | None = None) -> OrchestrationDecision:
    """Run the typed, point-in-time agent contract layer for a requested as_of snapshot."""
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    fundamental_snapshot = fetch_fundamental_snapshot(ticker, as_of, use_cache=True)
    technical_score = score_technical(snapshot)
    score_result = build_score(ticker, as_of, timestamp)

    market_payload = {
        "status": "OK",
        "source": snapshot.get("source"),
        "source_confidence": snapshot.get("source_confidence"),
        "future_bars_excluded": snapshot.get("future_bars_excluded", 0),
        "point_in_time_valid": bool(snapshot.get("source_contract", {}).get("timestamp_valid", True)),
    }
    market_agent = AgentContract(
        agent="market_data",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="OK",
        score=round(float(snapshot.get("data_quality", {}).get("score", 0.0)) / 10.0, 2),
        confidence=float(snapshot.get("source_confidence", 0.0)),
        uncertainty={"lower": 0.2, "upper": 0.8},
        evidence=[{"source_record_id": snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart"), "reason": "Market snapshot was filtered to bars at or before the as_of timestamp."}],
        model_version="market-data-v1",
        input_hash=_stable_hash(snapshot),
        warnings=[] if snapshot.get("data_quality", {}).get("score", 0.0) >= 60.0 else ["quality_below_threshold"],
        payload=market_payload,
    )

    technical_payload = {
        "status": "OK",
        "score": technical_score,
        "rsi": snapshot.get("rsi"),
        "trend_regime": snapshot.get("market_regime"),
    }
    technical_agent = AgentContract(
        agent="technical_analysis",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="OK",
        score=technical_score,
        confidence=max(0.0, min(1.0, score_result.confidence)),
        uncertainty={"lower": 4.0, "upper": 8.0},
        evidence=[{"source_record_id": snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart"), "reason": "Technical signals were computed from on-or-before as_of bars only."}],
        model_version="technical-v1",
        input_hash=_stable_hash(snapshot),
        warnings=[] if technical_score > 0 else ["technical_signal_unavailable"],
        payload=technical_payload,
    )

    fundamental_payload = {
        "status": fundamental_snapshot.get("source_status", "unknown"),
        "source_confidence": fundamental_snapshot.get("source_confidence", 0.0),
        "point_in_time_valid": fundamental_snapshot.get("point_in_time_valid", True),
        "valuation_metrics": fundamental_snapshot.get("valuation_metrics", {}),
    }
    fundamental_agent = AgentContract(
        agent="fundamental_analysis",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="OK" if fundamental_snapshot.get("point_in_time_valid", True) else "UNAVAILABLE",
        score=float(score_result.fundamental_score),
        confidence=float(fundamental_snapshot.get("source_confidence", 0.0)),
        uncertainty={"lower": 0.1, "upper": 0.9},
        evidence=[{"source_record_id": fundamental_snapshot.get("source", "provider_key_required"), "reason": "Fundamental metrics were gated by source availability and timestamp validity."}],
        model_version="fundamental-v1",
        input_hash=_stable_hash(fundamental_snapshot),
        warnings=[] if fundamental_snapshot.get("point_in_time_valid", True) else ["future_dated_fundamental_payload"],
        payload=fundamental_payload,
    )

    mode = "ANALYSIS_ONLY"
    if score_result.action != "ANALYSIS_ONLY" and score_result.governance.get("risk_gate_passed"):
        mode = "PAPER"

    decision = OrchestrationDecision(
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        mode=mode,
        action=score_result.action,
        score=float(score_result.score),
        confidence=float(score_result.confidence),
        agent_outputs=[market_agent, technical_agent, fundamental_agent],
        summary=(
            "Typed, point-in-time orchestrator run for the requested timestamp. "
            "The system remains analysis-only unless quality and source checks pass."
        ),
    )
    return decision
