from __future__ import annotations

import hashlib
import json

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.agent_contracts import AgentContract, NoTradeDecision, OrchestrationDecision
from core.score_engine import build_score
from fetch_data import fetch_fundamental_snapshot


def _stable_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_veto_reasons(snapshot: dict, fundamental_snapshot: dict, score_result) -> list[str]:
    reasons: list[str] = []
    if float(snapshot.get("data_quality", {}).get("score", 0.0)) < 60.0:
        reasons.append("data_quality_below_threshold")
    if float(snapshot.get("source_confidence", 0.0)) < 0.7:
        reasons.append("market_source_confidence_below_threshold")
    if not bool(snapshot.get("source_contract", {}).get("timestamp_valid", True)):
        reasons.append("future_dated_market_data")
    if not bool(fundamental_snapshot.get("point_in_time_valid", True)):
        reasons.append("future_dated_fundamental_payload")
    if float((fundamental_snapshot or {}).get("source_confidence", 0.0)) < 0.7 and (fundamental_snapshot or {}).get("source_status") != "live_provider":
        reasons.append("fundamental_source_confidence_below_threshold")
    if score_result.action == "ANALYSIS_ONLY":
        reasons.append("analysis_only_mode")
    if float(score_result.score) < 5.5:
        reasons.append("score_below_threshold")
    return reasons


def orchestrate_score(ticker: str, as_of: str, timestamp: str | None = None) -> OrchestrationDecision:
    """Run the typed, point-in-time agent contract layer for a requested as_of snapshot."""
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    fundamental_snapshot = fetch_fundamental_snapshot(ticker, as_of, use_cache=True)
    technical_score = score_technical(snapshot)
    score_result = build_score(ticker, as_of, timestamp)

    source_record_ids = [
        str(snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart")),
        str(fundamental_snapshot.get("source", "provider_key_required")),
    ]
    snapshot_hash = _stable_hash(snapshot)
    replay_hash = _stable_hash({"ticker": ticker.upper(), "as_of": snapshot["as_of"], "market": snapshot, "fundamentals": fundamental_snapshot})

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
        input_hash=snapshot_hash,
        warnings=[] if snapshot.get("data_quality", {}).get("score", 0.0) >= 60.0 else ["quality_below_threshold"],
        payload=market_payload,
        source_record_id=str(snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart")),
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
        input_hash=snapshot_hash,
        warnings=[] if technical_score > 0 else ["technical_signal_unavailable"],
        payload=technical_payload,
        source_record_id=str(snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart")),
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
        source_record_id=str(fundamental_snapshot.get("source", "provider_key_required")),
    )

    risk_agent = AgentContract(
        agent="risk_management",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="OK",
        score=float(score_result.score),
        confidence=max(0.0, min(1.0, float(score_result.confidence))),
        uncertainty={"lower": 0.0, "upper": 0.5},
        evidence=[{"source_record_id": "risk_policy", "reason": "Risk gate enforces minimum quality, confidence, and timestamp thresholds."}],
        model_version="risk-v1",
        input_hash=snapshot_hash,
        warnings=[],
        payload={"status": "OK", "veto_ready": True, "risk_flags": score_result.risk_flags},
        source_record_id="risk_policy",
    )

    audit_agent = AgentContract(
        agent="performance_auditor",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="OK",
        score=float(score_result.score),
        confidence=max(0.0, min(1.0, float(score_result.confidence))),
        uncertainty={"lower": 0.0, "upper": 0.3},
        evidence=[{"source_record_id": "audit_policy", "reason": "Audit checks the score for evidence sufficiency and replayability."}],
        model_version="audit-v1",
        input_hash=replay_hash,
        warnings=[],
        payload={"status": "OK", "replay_hash": replay_hash, "evidence_status": score_result.governance.get("evidence_status", "sufficient")},
        source_record_id="audit_policy",
    )

    veto_reasons = _build_veto_reasons(snapshot, fundamental_snapshot, score_result)
    mode = "ANALYSIS_ONLY"
    action = score_result.action
    if score_result.action == "ANALYSIS_ONLY":
        mode = "ANALYSIS_ONLY"
        action = "ANALYSIS_ONLY"
    elif not veto_reasons and score_result.governance.get("risk_gate_passed"):
        mode = "PAPER"
        action = "PAPER"
    else:
        mode = "NO_TRADE"
        action = "NO_TRADE"

    decision = OrchestrationDecision(
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        mode=mode,
        action=action,
        score=float(score_result.score),
        confidence=float(score_result.confidence),
        current_time_score=float(score_result.current_time_score),
        long_term_score=float(score_result.long_term_score),
        agent_outputs=[market_agent, technical_agent, fundamental_agent, risk_agent, audit_agent],
        summary=(
            "Typed, point-in-time orchestrator run for the requested timestamp. "
            "The system remains analysis-only unless quality and source checks pass."
        ),
        veto_reasons=veto_reasons,
        replay_hash=replay_hash,
        snapshot_hash=snapshot_hash,
        decision_type="NO_TRADE" if mode == "NO_TRADE" else "score",
        source_record_ids=source_record_ids,
    )
    if mode == "NO_TRADE":
        return decision
    return decision
