from __future__ import annotations

from core.config import MAX_SCORE


def score_technical(snapshot: dict) -> float:
    """Compute a multi-factor technical score from momentum, trend, RSI, and volatility."""
    change_1d = float(snapshot.get("change_1d", 0.0))
    change_5d = float(snapshot.get("change_5d", 0.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    trend_vs_20d_mean = float(snapshot.get("trend_vs_20d_mean", 0.0))
    volume = float(snapshot.get("volume", 0.0))
    avg_volume_20d = float(snapshot.get("avg_volume_20d", 1.0))
    rsi = float(snapshot.get("rsi", 50.0))
    volatility = float(snapshot.get("volatility", 0.0))
    moving_averages = snapshot.get("moving_averages", {})

    ma_200 = float(moving_averages.get("200d", 0.0))
    ma_150 = float(moving_averages.get("150d", 0.0))
    ma_100 = float(moving_averages.get("100d", 0.0))
    ma_50 = float(moving_averages.get("50d", 0.0))
    close = float(snapshot.get("close", 0.0))

    score = 5.0
    score += max(-2.0, min(2.0, change_20d * 30.0))
    score += max(-2.0, min(2.0, change_5d * 25.0))
    score += max(-1.5, min(1.5, change_1d * 20.0))
    score += max(-1.5, min(1.5, trend_vs_20d_mean * 20.0))

    if ma_50 and ma_200:
        score += max(-1.5, min(1.5, ((ma_50 - ma_200) / ma_200) * 50.0))
    if ma_100 and ma_150:
        score += max(-1.0, min(1.0, ((ma_100 - ma_150) / ma_150) * 20.0))

    score += max(-1.5, min(1.5, ((rsi - 50.0) / 50.0) * 1.5))

    volume_ratio = volume / avg_volume_20d if avg_volume_20d else 1.0
    score += max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.0))

    if volatility > 0:
        score -= min(1.5, volatility * 30.0)

    if close <= 0:
        score = 0.0

    score = max(0.0, min(MAX_SCORE, score))
    return round(score, 2)
