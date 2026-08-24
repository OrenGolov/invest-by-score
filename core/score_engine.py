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
    if float(snapshot.get("rsi", 50.0)) < 30.0 or float(snapshot.get("rsi", 50.0)) > 70.0:
        risk_flags.append("RSI extreme")

    confidence = min(0.95, 0.5 + (capped_score / MAX_SCORE) * 0.45)
    if risk_flags:
        confidence = max(0.35, confidence - 0.2)

    explanation = (
        f"Multi-factor technical score using moving averages, RSI, volatility, momentum, and volume. "
        f"Current price is {snapshot['close']:.2f}; 20-day momentum is {snapshot.get('change_20d', 0.0):.2%}; RSI is {snapshot.get('rsi', 50.0):.1f}."
    )

    market_context = {
        "close": snapshot.get("close"),
        "volume_ratio_20d": snapshot.get("volume_ratio_20d"),
        "price_vs_ma_50": snapshot.get("price_vs_ma_50"),
        "price_vs_ma_100": snapshot.get("price_vs_ma_100"),
        "price_vs_ma_200": snapshot.get("price_vs_ma_200"),
        "market_regime": snapshot.get("market_regime"),
        "trend_vs_20d_mean": snapshot.get("trend_vs_20d_mean"),
    }

    data_quality = snapshot.get("data_quality", {})
    source_metadata = {
        "source": snapshot.get("source"),
        "source_type": snapshot.get("source_type"),
        "source_confidence": snapshot.get("source_confidence"),
        "last_valid_bar": snapshot.get("last_valid_bar"),
        "first_valid_bar": snapshot.get("first_valid_bar"),
    }

    return ScoreResult(
        ticker=snapshot["ticker"],
        as_of=snapshot["as_of"],
        score=round(capped_score, 2),
        confidence=round(confidence, 2),
        explanation=explanation,
        risk_flags=risk_flags,
        action=action,
        moving_averages=snapshot.get("moving_averages", {}),
        rsi=snapshot.get("rsi"),
        volatility=snapshot.get("volatility"),
        market_context=market_context,
        data_quality=data_quality,
        source_metadata=source_metadata,
    )
