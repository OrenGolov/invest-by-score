from __future__ import annotations

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.config import DEFAULT_ACTION, MAX_SCORE
from core.schemas import ScoreResult


def build_score(ticker: str, as_of: str, timestamp: str | None = None) -> ScoreResult:
    """Build a simple score for a ticker at a given point-in-time."""
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    raw_score = score_technical(snapshot)
    capped_score = max(0.0, min(MAX_SCORE, raw_score))

    risk_flags = []
    action = DEFAULT_ACTION
    if capped_score < 3.0:
        risk_flags.append("Weak momentum")
        action = "ANALYSIS_ONLY"
    if snapshot.get("volume", 0.0) < snapshot.get("avg_volume_20d", 0.0) * 0.5:
        risk_flags.append("Low volume")
        action = "ANALYSIS_ONLY"
    if snapshot.get("change_20d", 0.0) < -0.15:
        risk_flags.append("Downtrend")
        action = "ANALYSIS_ONLY"

    confidence = min(0.95, 0.5 + (capped_score / MAX_SCORE) * 0.45)
    if risk_flags:
        confidence = max(0.35, confidence - 0.2)

    explanation = (
        f"Momentum-based technical score using recent price changes and volume. "
        f"Current price is {snapshot['close']:.2f} with 20-day momentum {snapshot.get('change_20d', 0.0):.2%}."
    )

    return ScoreResult(
        ticker=snapshot["ticker"],
        as_of=snapshot["as_of"],
        score=round(capped_score, 2),
        confidence=round(confidence, 2),
        explanation=explanation,
        risk_flags=risk_flags,
        action=action,
    )
