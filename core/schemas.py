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

    def to_dict(self) -> dict:
        return asdict(self)
