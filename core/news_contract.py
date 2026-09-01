from __future__ import annotations

from core.news_adapter import build_news_snapshot


def fetch_news_snapshot(ticker: str, as_of: str) -> dict:
    """Return the point-in-time news/sentiment contract for a ticker.

    Thin, stable entry point over `core.news_adapter.build_news_snapshot`
    (Sprint N1). Callers treat `status` as the source of truth:

    - UNAVAILABLE: no provider key configured (byte-for-byte the legacy
      placeholder), the provider request failed, or it returned no
      articles. Sentiment is never derived from price, RSI, volume, or any
      other technical indicator — that would present fabricated data as if
      it were real news coverage.
    - OK: verified articles were ingested, PIT-filtered, entity-resolved,
      classified, weighted, and aggregated inside the point-in-time window.
    - CONTRADICTORY: credible same-day, same-category articles disagree
      beyond tolerance — surfaced explicitly, never averaged to neutral.
    - INVALID: the provider payload violated the timestamp policy
      (future-dated or unparseable publication times).
    - INCOMPLETE: articles exist but none are credible and on-entity
      enough to aggregate.
    """
    return build_news_snapshot(ticker, as_of)
