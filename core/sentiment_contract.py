from __future__ import annotations

from core.config import SENTIMENT_CONTRACT_VERSION
from core.schemas import SentimentSnapshot

SENTIMENT_SOURCE_ID = "sentiment_provider_unconfigured"
SENTIMENT_REASON = (
    "No legitimate social/positioning sentiment provider is connected. "
    "Sentiment is never inferred from RSI, price direction, technical "
    "indicators, or the news score (anti-proxying rule, Sprint N2); a "
    "news-derived design would have to be labeled derived_from_news with "
    "confidence scaled accordingly."
)


def fetch_sentiment_snapshot(ticker: str, as_of: str) -> dict:
    """Return the point-in-time social/positioning sentiment contract.

    Sprint N2 typed placeholder. The signature deliberately accepts ONLY the
    ticker and the as-of timestamp: there is no parameter through which
    price, technical indicators, or the news snapshot could enter, so silent
    proxying is impossible by construction. Callers treat `status` as the
    source of truth and apply zero weight while it reads UNAVAILABLE.

    Derivation contract:
    - `derivation: "none"` — no sentiment value exists (current state).
    - `"provider"` — a legitimate social/positioning provider connected.
    - `"derived_from_news"` — an explicitly designed news-derived feature;
      the confidence MUST then be scaled by 0.5 relative to the news
      evidence it consumed (Sprint N2 spec). Silent proxying is prohibited.
    """
    return SentimentSnapshot(
        ticker=ticker.upper(),
        as_of=str(as_of),
        status="UNAVAILABLE",
        source_id=SENTIMENT_SOURCE_ID,
        source_confidence=0.0,
        published_time=None,
        calculation_version=SENTIMENT_CONTRACT_VERSION,
        lookback_period="N/A",
        sentiment_score=None,
        derivation="none",
        intended_inputs=["social_volume", "tone_trend", "disagreement", "manipulation_flags"],
        reason=SENTIMENT_REASON,
    ).to_dict()