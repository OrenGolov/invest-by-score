from __future__ import annotations

from core.config import MAX_SCORE


def score_technical(snapshot: dict) -> float:
    """Compute a simple technical score from price momentum and volume behavior."""
    change_1d = float(snapshot.get("change_1d", 0.0))
    change_5d = float(snapshot.get("change_5d", 0.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    trend_vs_20d_mean = float(snapshot.get("trend_vs_20d_mean", 0.0))
    volume = float(snapshot.get("volume", 0.0))
    avg_volume_20d = float(snapshot.get("avg_volume_20d", 1.0))

    score = 5.0
    score += max(-2.0, min(2.0, change_20d * 30.0))
    score += max(-2.0, min(2.0, change_5d * 25.0))
    score += max(-1.5, min(1.5, change_1d * 20.0))
    score += max(-1.5, min(1.5, trend_vs_20d_mean * 20.0))

    volume_ratio = volume / avg_volume_20d if avg_volume_20d else 1.0
    score += max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.0))

    score = max(0.0, min(MAX_SCORE, score))
    return round(score, 2)
