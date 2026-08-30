from __future__ import annotations

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.agent_contracts import AgentContract, NoTradeDecision, OrchestrationDecision
from core.audit_policy import evaluate_audit_policy, stable_hash
from core.config import RISK_POLICY_V2
from core.risk_policy import evaluate_risk_policy
from core.schemas import STATUS_POSTURE, AgentStatus, status_posture, worst_status
from core.score_engine import build_score
from fetch_data import fetch_fundamental_snapshot


def _stable_hash(payload: dict) -> str:
    """Canonical hashing owned by the audit layer (single implementation)."""
    return stable_hash(payload)


def _build_risk_context(snapshot: dict, fundamental_snapshot: dict, score_result) -> dict:
    """Assemble the risk-policy evaluation context from persisted inputs."""
    return {
        "market_data_quality": snapshot.get("data_quality", {}).get("score"),
        "market_source_confidence": snapshot.get("source_confidence"),
        "market_timestamp_valid": bool(snapshot.get("source_contract", {}).get("timestamp_valid", True)),
        "fundamental_point_in_time_valid": fundamental_snapshot.get("point_in_time_valid", True),
        "fundamental_source_confidence": fundamental_snapshot.get("source_confidence"),
        "fundamental_source_status": fundamental_snapshot.get("source_status"),
        "score": score_result.score,
        "action": score_result.action,
        "confidence": score_result.confidence,
        "confidence_breakdown": score_result.confidence_breakdown,
    }


def _select_mode(action: str, veto_rule_ids: list[str], risk_gate_passed: bool) -> str:
    """Single place deciding the decision posture.

    Invariant: PAPER requires zero triggered veto-severity rules AND a passed
    risk gate; ANALYSIS_ONLY propagates the scoring engine's posture; anything
    else is NO_TRADE. Fail-closed by construction.
    """
    if action == "ANALYSIS_ONLY":
        return "ANALYSIS_ONLY"
    if not veto_rule_ids and risk_gate_passed:
        return "PAPER"
    return "NO_TRADE"


def _derive_agent_statuses(snapshot: dict, fundamental_snapshot: dict, score_result) -> dict[str, str]:
    """Map raw decision signals onto the AgentStatus taxonomy (W4).

    Thresholds are re-read from the risk policy table so degradation levels
    are defined in exactly one place. Pure and deterministic.
    """
    freshness_threshold = float(RISK_POLICY_V2["freshness_degraded"]["minimum_freshness_factor"])
    quality_threshold = float(RISK_POLICY_V2["data_quality_below_threshold"]["minimum_market_data_quality"])
    freshness = next(
        (
            float(factor["value"])
            for factor in (score_result.confidence_breakdown or {}).get("factors", [])
            if factor.get("name") == "freshness" and factor.get("value") is not None
        ),
        None,
    )
    market_quality_value = snapshot.get("data_quality", {}).get("score")
    market_quality = float(market_quality_value) if market_quality_value is not None else None
    timestamp_valid = bool(snapshot.get("source_contract", {}).get("timestamp_valid", True))

    if not timestamp_valid:
        market_status = AgentStatus.INVALID.value
    elif freshness is not None and freshness < freshness_threshold:
        market_status = AgentStatus.STALE.value
    elif market_quality is None or market_quality < quality_threshold:
        market_status = AgentStatus.INCOMPLETE.value
    else:
        market_status = AgentStatus.OK.value

    fundamental_pit_valid = bool(fundamental_snapshot.get("point_in_time_valid", True))
    fundamental_status = AgentStatus.OK.value if fundamental_pit_valid else AgentStatus.INVALID.value

    news_status = str((score_result.news_snapshot or {}).get("status", "UNAVAILABLE"))

    return {
        "market_data": market_status,
        "technical_analysis": AgentStatus.OK.value,
        "fundamental_analysis": fundamental_status,
        "news_intelligence": news_status,
    }


def orchestrate_score(ticker: str, as_of: str, timestamp: str | None = None) -> OrchestrationDecision:
    """Run the typed, point-in-time agent contract layer for a requested as_of snapshot."""
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    fundamental_snapshot = fetch_fundamental_snapshot(ticker, as_of, use_cache=True)
    technical_score = score_technical(snapshot)
    score_result = build_score(ticker, as_of, timestamp)
    probe_result = build_score(ticker, as_of, timestamp, persist_audit=False)

    source_record_ids = [
        str(snapshot.get("source_contract", {}).get("source_id", "yahoo_finance_chart")),
        str(fundamental_snapshot.get("source", "provider_key_required")),
    ]
    snapshot_hash = _stable_hash(snapshot)
    replay_hash = _stable_hash({"ticker": ticker.upper(), "as_of": snapshot["as_of"], "market": snapshot, "fundamentals": fundamental_snapshot})
    risk_evaluation = evaluate_risk_policy(_build_risk_context(snapshot, fundamental_snapshot, score_result))
    agent_statuses = _derive_agent_statuses(snapshot, fundamental_snapshot, score_result)

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
        status=agent_statuses["market_data"],
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
        status=agent_statuses["fundamental_analysis"],
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

    triggered_rules = [rule for rule in risk_evaluation["rules"] if rule["triggered"]]
    risk_agent = AgentContract(
        agent="risk_management",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="VETO" if risk_evaluation["veto"] else "OK",
        score=float(score_result.score),
        confidence=max(0.0, min(1.0, float(score_result.confidence))),
        uncertainty={"lower": 0.0, "upper": 0.5},
        evidence=[{
            "source_record_id": "risk_policy",
            "reason": (
                f"Risk policy {risk_evaluation['policy_version']}: "
                f"{len(triggered_rules)} of {len(risk_evaluation['rules'])} rules triggered "
                f"({len(risk_evaluation['veto_rule_ids'])} veto-severity)."
            ),
        }],
        model_version=risk_evaluation["policy_version"],
        input_hash=snapshot_hash,
        warnings=[rule["rule_id"] for rule in triggered_rules],
        payload={
            "status": "VETO" if risk_evaluation["veto"] else "OK",
            "policy_version": risk_evaluation["policy_version"],
            "veto": risk_evaluation["veto"],
            "veto_rule_ids": risk_evaluation["veto_rule_ids"],
            "warning_rule_ids": risk_evaluation["warning_rule_ids"],
            "rules": risk_evaluation["rules"],
            "risk_flags": score_result.risk_flags,
        },
        source_record_id="risk_policy",
    )

    news_snapshot = score_result.news_snapshot
    news_agent = AgentContract(
        agent="news_intelligence",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status=news_snapshot.get("status", "UNAVAILABLE"),
        score=0.0,
        confidence=float(news_snapshot.get("source_confidence", 0.0)),
        uncertainty={"lower": 0.0, "upper": 0.0},
        evidence=[{"source_record_id": news_snapshot.get("source_id", "news_provider_unconfigured"), "reason": news_snapshot.get("reason", "")}],
        model_version=news_snapshot.get("calculation_version", "news-contract-v1"),
        input_hash=_stable_hash(news_snapshot),
        warnings=[] if news_snapshot.get("status") == "OK" else ["news_provider_unconfigured"],
        payload={"status": news_snapshot.get("status", "UNAVAILABLE"), "sentiment_score": news_snapshot.get("sentiment_score")},
        source_record_id=str(news_snapshot.get("source_id", "news_provider_unconfigured")),
    )

    audit_evaluation = evaluate_audit_policy({
        "ticker": ticker.upper(),
        "as_of": snapshot["as_of"],
        "agents": [agent.to_dict() for agent in (market_agent, technical_agent, fundamental_agent, news_agent, risk_agent)],
        "expected_input_hashes": {
            "market_data": snapshot_hash,
            "technical_analysis": snapshot_hash,
            "fundamental_analysis": _stable_hash(fundamental_snapshot),
            "news_intelligence": _stable_hash(news_snapshot),
            "risk_management": snapshot_hash,
        },
        "snapshot": snapshot,
        "snapshot_hash": snapshot_hash,
        "replay_hash": replay_hash,
        "first_result": score_result.to_dict(),
        "second_result": probe_result.to_dict(),
        "confidence": score_result.confidence,
        "confidence_breakdown": score_result.confidence_breakdown,
        "ensemble_breakdown": score_result.ensemble_breakdown,
        "current_time_score": score_result.current_time_score,
        "long_term_score": score_result.long_term_score,
        "governance": score_result.governance,
        "evidence_ledger": score_result.evidence_ledger,
    })
    failed_checks = [finding for finding in audit_evaluation["findings"] if not finding["passed"]]
    audit_agent = AgentContract(
        agent="performance_auditor",
        ticker=ticker.upper(),
        as_of=snapshot["as_of"],
        status="VETO" if audit_evaluation["veto"] else "OK",
        score=float(score_result.score),
        confidence=max(0.0, min(1.0, float(score_result.confidence))),
        uncertainty={"lower": 0.0, "upper": 0.3},
        evidence=[{
            "source_record_id": "audit_policy",
            "reason": (
                f"Audit policy {audit_evaluation['policy_version']}: "
                f"{len(audit_evaluation['findings']) - len(failed_checks)} of {len(audit_evaluation['findings'])} checks passed."
            ),
        }],
        model_version=audit_evaluation["policy_version"],
        input_hash=replay_hash,
        warnings=[finding["check_id"] for finding in failed_checks],
        payload={
            "status": "VETO" if audit_evaluation["veto"] else "OK",
            "policy_version": audit_evaluation["policy_version"],
            "veto": audit_evaluation["veto"],
            "veto_check_ids": audit_evaluation["veto_check_ids"],
            "warning_check_ids": audit_evaluation["warning_check_ids"],
            "findings": audit_evaluation["findings"],
            "replay_hash": replay_hash,
            "snapshot_hash": snapshot_hash,
            "evidence_status": score_result.governance.get("evidence_status", "sufficient"),
        },
        source_record_id="audit_policy",
    )

    veto_reasons = list(risk_evaluation["veto_rule_ids"])
    if audit_evaluation["veto"]:
        veto_reasons.append("auditor_veto")
    data_agent_posture = status_posture(worst_status(
        agent_statuses[name] for name in ("market_data", "technical_analysis", "fundamental_analysis", "news_intelligence")
    ))
    if data_agent_posture == "NO_TRADE":
        veto_reasons.append("agent_status_no_trade")
    effective_action = "ANALYSIS_ONLY" if data_agent_posture == "ANALYSIS_ONLY" else score_result.action
    mode = _select_mode(effective_action, veto_reasons, bool(score_result.governance.get("risk_gate_passed")))
    action = score_result.action
    if mode == "ANALYSIS_ONLY":
        action = "ANALYSIS_ONLY"
    elif mode == "PAPER":
        action = "PAPER"
    elif mode == "NO_TRADE":
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
        agent_outputs=[market_agent, technical_agent, fundamental_agent, news_agent, risk_agent, audit_agent],
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
