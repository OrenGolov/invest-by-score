from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureContract:
    """Point-in-time provenance for a single computed feature.

    Every feature that feeds a score must be traceable back to the data
    it was computed from: when it is valid for (as_of), where it came
    from (source_id), when that source data was published, which formula
    version produced it, and how much history it was computed over.
    """

    name: str
    value: Any
    as_of: str
    source_id: str
    published_time: str | None
    calculation_version: str
    lookback_period: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketSnapshot:
    """Canonical, typed view of a point-in-time market snapshot.

    `agents.market_data_agent.fetch_market_snapshot` remains the wire
    format (a plain dict, for JSON serialization and backward
    compatibility with existing consumers). This dataclass is the single
    canonical contract that dict is required to satisfy: use
    `MarketSnapshot.from_dict(snapshot)` to validate/typed-access it.
    """

    ticker: str
    as_of: str
    source_id: str
    published_time: str | None
    calculation_version: str
    lookback_period: str
    close: float
    volume: float
    moving_averages: dict[str, float] = field(default_factory=dict)
    rsi: float | None = None
    volatility: float | None = None
    features: dict[str, FeatureContract] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketSnapshot":
        source_contract = data.get("source_contract", {}) or {}
        raw_features = data.get("features") or {}
        features = {
            name: (payload if isinstance(payload, FeatureContract) else FeatureContract(**payload))
            for name, payload in raw_features.items()
        }
        return cls(
            ticker=str(data["ticker"]),
            as_of=str(data["as_of"]),
            source_id=str(source_contract.get("source_id", data.get("source", "unknown"))),
            published_time=data.get("last_valid_bar"),
            calculation_version=str(data.get("calculation_version", "market-snapshot-v1")),
            lookback_period=str(data.get("lookback_period", "variable")),
            close=float(data.get("close", 0.0)),
            volume=float(data.get("volume", 0.0)),
            moving_averages=dict(data.get("moving_averages", {})),
            rsi=data.get("rsi"),
            volatility=data.get("volatility"),
            features=features,
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "source_id": self.source_id,
            "published_time": self.published_time,
            "calculation_version": self.calculation_version,
            "lookback_period": self.lookback_period,
            "close": self.close,
            "volume": self.volume,
            "moving_averages": self.moving_averages,
            "rsi": self.rsi,
            "volatility": self.volatility,
            "features": {name: contract.to_dict() for name, contract in self.features.items()},
        }


@dataclass
class NewsSnapshot:
    """Point-in-time news/sentiment contract.

    Until a verified news provider is connected, every field stays in an
    explicit UNAVAILABLE state. This must never be backfilled from price
    or technical indicators (RSI, momentum, etc.) — that would silently
    disguise a missing data source as a real signal.
    """

    ticker: str
    as_of: str
    status: str
    source_id: str
    source_confidence: float
    published_time: str | None
    calculation_version: str
    lookback_period: str
    sentiment_score: float | None = None
    articles: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

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
    news_snapshot: dict[str, object] = field(default_factory=dict)
    confidence_breakdown: dict[str, object] = field(default_factory=dict)
    ensemble_breakdown: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
