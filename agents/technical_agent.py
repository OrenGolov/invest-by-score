from __future__ import annotations

from core.config import MAX_SCORE, MIN_SCORE
from core.schemas import MarketSnapshot


def score_technical(snapshot: MarketSnapshot) -> dict:
    recent_returns = snapshot.recent_returns
    if not recent_returns:
        score = 5.0
        explanation = "Insufficient trend data; neutral technical stance."
        return {"score": score, "confidence": 0.4, "explanation": explanation}

    avg_return = sum(recent_returns) / len(recent_returns)
    trend_bias = 1.0 if snapshot.trend == "bullish" else -1.0 if snapshot.trend == "bearish" else 0.0
    raw = 5.0 + (avg_return * 100) * 1.5 + trend_bias * 1.5

    if raw < MIN_SCORE:
        raw = MIN_SCORE
    if raw > MAX_SCORE:
        raw = MAX_SCORE

    confidence = min(0.95, max(0.45, 0.5 + abs(avg_return) * 8))
    explanation = (
        f"Trend is {snapshot.trend}. "
        f"Recent 5-day average return is {avg_return:.2%}. "
        f"Latest close is {snapshot.latest_close:.2f}."
    )

    return {
        "score": round(raw, 2),
        "confidence": round(confidence, 2),
        "explanation": explanation,
    }
