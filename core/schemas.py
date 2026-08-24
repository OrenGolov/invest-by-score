from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass
class MarketSnapshot:
    ticker: str
    as_of: str
    latest_close: float
    recent_returns: List[float] = field(default_factory=list)
    trend: str = "neutral"
    volatility: float = 0.0


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
    evidence: list[dict[str, str]] = field(default_factory=list)
    agent_outputs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreResult:
    ticker: str
    as_of: str
    score: float
    current_time_score: float = 0.0
    long_term_score: float = 0.0
    confidence: float = 0.0
    explanation: str = ""
    risk_flags: list[str] = field(default_factory=list)
    action: str = "ANALYSIS_ONLY"
    moving_averages: dict[str, float] = field(default_factory=dict)
    rsi: float | None = None
    volatility: float | None = None
    market_context: dict[str, float | str | dict | list] = field(default_factory=dict)
    data_quality: dict[str, float | int | list[str]] = field(default_factory=dict)
    source_metadata: dict[str, str | float | dict | list] = field(default_factory=dict)
    recommended_actions: dict[str, object] = field(default_factory=dict)
    latest_financial_report: dict[str, object] = field(default_factory=dict)
    next_expected_report: dict[str, object] = field(default_factory=dict)
    insights: dict[str, object] = field(default_factory=dict)
    scoring_breakdown: dict[str, float | str | dict | list] = field(default_factory=dict)
    source_reliability: dict[str, object] = field(default_factory=dict)
    technical_features: dict[str, object] = field(default_factory=dict)
    feature_metadata: dict[str, object] = field(default_factory=dict)
    governance: dict[str, object] = field(default_factory=dict)
    evidence_ledger: dict[str, object] = field(default_factory=dict)
    fundamental_score: float = 0.0
    fundamental_features: dict[str, object] = field(default_factory=dict)
    source_quality: dict[str, object] = field(default_factory=dict)
    replay_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
