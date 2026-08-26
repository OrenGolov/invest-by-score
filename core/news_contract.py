from __future__ import annotations

from core.config import NEWS_CONTRACT_VERSION
from core.schemas import NewsSnapshot


def fetch_news_snapshot(ticker: str, as_of: str) -> dict:
    """Return the point-in-time news/sentiment contract for a ticker.

    No verified news provider is connected yet, so this always returns an
    explicit UNAVAILABLE contract with a null sentiment score. It must
    never derive a sentiment value from price, RSI, volume, or any other
    technical indicator — that would present fabricated data as if it
    were real news coverage. Once a real provider (e.g. a licensed news
    or filings feed) is wired in, this function is the single place that
    changes: callers already treat `status` as the source of truth and
    apply zero weight to the score while it reads UNAVAILABLE.
    """
    return NewsSnapshot(
        ticker=ticker.upper(),
        as_of=str(as_of),
        status="UNAVAILABLE",
        source_id="news_provider_unconfigured",
        source_confidence=0.0,
        published_time=None,
        calculation_version=NEWS_CONTRACT_VERSION,
        lookback_period="N/A",
        sentiment_score=None,
        articles=[],
        reason="No verified news provider is connected. Sentiment is not inferred from price or technical indicators.",
    ).to_dict()
