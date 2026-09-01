from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class AgentStatus(str, Enum):
    """Failure-state taxonomy for every agent contract (W4).

    Mapping rules:
    - OK: the agent ran on eligible data with no material degradation.
    - UNAVAILABLE: provider unconfigured or empty response (e.g. the news
      stub until a verified provider connects).
    - STALE: the newest eligible input is too old versus as_of (freshness
      factor below the risk-policy threshold).
    - INCOMPLETE: data quality below the governance threshold or critical
      fields missing (close, valuation metrics).
    - CONTRADICTORY: reserved for same-domain sources disagreeing beyond
      tolerance (activates with the Sprint N adapters).
    - INVALID: timestamp violations (future-dated payload), schema
      violations, or impossible values — the data cannot be trusted.
    - VETO: governance posture for risk/auditor agents that blocked the
      decision (not a data state).

    STATUS_POSTURE drives propagation: INVALID and CONTRADICTORY force
    NO_TRADE; UNAVAILABLE, STALE and INCOMPLETE force the ANALYSIS_ONLY
    floor; OK permits PAPER subject to all other governance gates.
    """

    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    CONTRADICTORY = "CONTRADICTORY"
    INVALID = "INVALID"
    VETO = "VETO"


STATUS_POSTURE = {
    AgentStatus.OK: "PAPER",
    AgentStatus.UNAVAILABLE: "ANALYSIS_ONLY",
    AgentStatus.STALE: "ANALYSIS_ONLY",
    AgentStatus.INCOMPLETE: "ANALYSIS_ONLY",
    AgentStatus.CONTRADICTORY: "NO_TRADE",
    AgentStatus.INVALID: "NO_TRADE",
    AgentStatus.VETO: "NO_TRADE",
}

_AGENT_STATUS_SEVERITY = {
    AgentStatus.OK: 0,
    AgentStatus.UNAVAILABLE: 1,
    AgentStatus.INCOMPLETE: 2,
    AgentStatus.STALE: 3,
    AgentStatus.CONTRADICTORY: 4,
    AgentStatus.INVALID: 5,
    AgentStatus.VETO: 5,
}


def status_posture(status: str) -> str:
    """Return the worst decision posture a single agent status permits."""
    return STATUS_POSTURE[AgentStatus(status)]


def worst_status(statuses: Iterable[str]) -> str:
    """Combine agent statuses into the most severe one (empty -> OK)."""
    values = list(statuses)
    if not values:
        return AgentStatus.OK.value
    return max((AgentStatus(status) for status in values), key=lambda status: _AGENT_STATUS_SEVERITY[status]).value


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

    Wire format of `core.news_contract.fetch_news_snapshot` (the adapter in
    `core/news_adapter.py` owns construction). Without a verified provider
    every field stays in an explicit UNAVAILABLE state, byte-for-byte
    identical to the pre-N1 placeholder. This must never be backfilled from
    price or technical indicators (RSI, momentum, etc.) — that would
    silently disguise a missing data source as a real signal. Non-UNAVAILABLE
    paths (Sprint N1) add a `pipeline` provenance block on top of these
    fields; the UNAVAILABLE path never carries one.
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
class SentimentSnapshot:
    """Point-in-time social/positioning sentiment contract (N2 placeholder).

    Distinct from news tone: this agent is meant to consume retail and
    institutional positioning signals — social volume, tone trend,
    disagreement, and manipulation flags. Until a legitimate provider
    exists, every field stays in an explicit UNAVAILABLE state.

    Anti-proxying rule (Sprint N2): sentiment is never inferred from RSI,
    price direction, technical indicators, or the news score. `derivation`
    records the provenance: `"none"` while no value exists, `"provider"`
    once a legitimate provider connects, and `"derived_from_news"` — with
    confidence scaled by 0.5 relative to the news evidence consumed — for
    any explicitly designed news-derived feature. Silent proxying is
    prohibited; the contract test pins this shape.
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
    derivation: str = "none"
    intended_inputs: list[str] = field(
        default_factory=lambda: ["social_volume", "tone_trend", "disagreement", "manipulation_flags"],
    )
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroSnapshot:
    """Point-in-time macroeconomic context (N3) — vintage-aware, PIT-filtered.

    Economic data is published with a lag (CPI ~month+12d, GDP ~month+30d).
    This snapshot captures the latest available release for each series
    *at the decision point* (as_of), never the reference period end.
    Revisions append as new records in raw_store (W6).

    Every series carries provenance: source record ids, publication timestamp,
    and status (OK, UNAVAILABLE, or a specific reason for missing data).
    Missing series degrades confidence via MACRO_MISSING_SERIES_PENALTY,
    never silently zero-fills to neutral.

    Sector loadings (static GICS mappings per N3) show macro factor exposures
    for risk-on/off scenario attribution. Per-series contributions track which
    releases drove the regime classification.
    """

    ticker: str
    as_of: str
    status: str
    source_id: str
    source_confidence: float
    published_time: str | None
    calculation_version: str
    regime: str  # "risk_on" | "neutral" | "risk_off"
    regime_score: float  # [0, 1]: 0 = risk-off, 0.5 = neutral, 1.0 = risk-on
    series_values: dict[str, float] = field(default_factory=dict)  # logical_id -> value
    series_credibility: dict[str, dict] = field(default_factory=dict)  # logical_id -> {status, details}
    sector_loadings: dict[str, float] = field(default_factory=dict)  # "rates" | "energy" | "usd" -> sensitivity
    per_series_contributions: list[dict] = field(default_factory=list)  # [{series_id, value, credibility, source_record_ids}, ...]
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
    sentiment_snapshot: dict[str, object] = field(default_factory=dict)
    confidence_breakdown: dict[str, object] = field(default_factory=dict)
    ensemble_breakdown: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
