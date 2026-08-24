from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class MarketSnapshot:
    ticker: str
    as_of: str
    latest_close: float
    recent_returns: List[float] = field(default_factory=list)
    trend: str = "neutral"
    volatility: float = 0.0


@dataclass
class ScoreResult:
    ticker: str
    as_of: str
    score: float
    confidence: float
    explanation: str
    risk_flags: list[str] = field(default_factory=list)
    action: str = "ANALYSIS_ONLY"
    moving_averages: dict[str, float] = field(default_factory=dict)
    rsi: float | None = None
    volatility: float | None = None
    market_context: dict[str, float | str | dict | list] = field(default_factory=dict)
    data_quality: dict[str, float | int | list[str]] = field(default_factory=dict)
    source_metadata: dict[str, str | float | dict | list] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
