from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentContract:
    agent: str
    ticker: str
    as_of: str
    status: str = "OK"
    score: float = 0.0
    confidence: float = 0.0
    uncertainty: dict[str, float] = field(default_factory=lambda: {"lower": 0.0, "upper": 0.0})
    evidence: list[dict[str, str]] = field(default_factory=list)
    model_version: str = "v1.0"
    input_hash: str = ""
    warnings: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    source_record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VetoRecord:
    agent: str
    status: str
    reason: str
    severity: str = "warning"
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoTradeDecision:
    ticker: str
    as_of: str
    mode: str = "NO_TRADE"
    action: str = "NO_TRADE"
    reason: str = ""
    veto_reasons: list[str] = field(default_factory=list)
    replay_hash: str = ""
    snapshot_hash: str = ""
    current_time_score: float = 0.0
    long_term_score: float = 0.0
    evidence: list[dict[str, str]] = field(default_factory=list)
    agent_outputs: list[AgentContract] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationDecision:
    ticker: str
    as_of: str
    mode: str
    action: str
    score: float
    confidence: float
    current_time_score: float = 0.0
    long_term_score: float = 0.0
    agent_outputs: list[AgentContract] = field(default_factory=list)
    summary: str = ""
    veto_reasons: list[str] = field(default_factory=list)
    replay_hash: str = ""
    snapshot_hash: str = ""
    decision_type: str = "score"
    source_record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
