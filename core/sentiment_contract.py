from __future__ import annotations

from core.config import SENTIMENT_CONTRACT_VERSION, SENTIMENT_DERIVED_CONFIDENCE_SCALE
from core.schemas import SentimentSnapshot

SENTIMENT_SOURCE_ID = "sentiment_provider_unconfigured"
SENTIMENT_DERIVED_SOURCE_ID = "sentiment_derived_from_news"
SENTIMENT_REASON = (
    "No legitimate social/positioning sentiment provider is connected. "
    "Sentiment is never inferred from RSI, price direction, technical "
    "indicators, or the news score (anti-proxying rule, Sprint N2); a "
    "news-derived design would have to be labeled derived_from_news with "
    "confidence scaled accordingly."
)


def _unavailable_snapshot(ticker: str, as_of: str, reason: str = SENTIMENT_REASON) -> dict:
    """Build the canonical UNAVAILABLE placeholder (single construction point)."""
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
        reason=reason,
    ).to_dict()


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
    - `"derived_from_news"` — an explicitly designed news-derived feature
      (the ONLY sanctioned path is `derive_sentiment_from_news` below);
      the confidence MUST then be scaled by
      `SENTIMENT_DERIVED_CONFIDENCE_SCALE` relative to the news evidence it
      consumed (Sprint N2 spec). Silent proxying is prohibited.
    """
    return _unavailable_snapshot(ticker, as_of)


def _is_valid_news_evidence(news_snapshot: dict) -> bool:
    """Fail-closed evidence gate for the explicit news-derived path.

    Only a fully healthy, in-range news payload may be consumed: status OK,
    a real (non-bool) numeric sentiment_score inside [-1, 1], and a
    source_confidence inside [0, 1]. Anything else — missing fields, a
    degraded status (UNAVAILABLE/INCOMPLETE/STALE/CONTRADICTORY/INVALID),
    or malformed numbers — is refused, because sentiment must never be
    fabricated from a failed or contradictory agent.
    """
    if not isinstance(news_snapshot, dict):
        return False
    if news_snapshot.get("status") != "OK":
        return False
    score = news_snapshot.get("sentiment_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    if not -1.0 <= float(score) <= 1.0:
        return False
    confidence = news_snapshot.get("source_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    return 0.0 <= float(confidence) <= 1.0


def derive_sentiment_from_news(news_snapshot: dict, ticker: str, as_of: str) -> dict:
    """Explicit, labeled news-derived sentiment — the ONLY sanctioned path.

    The anti-proxying rule (Sprint N2) prohibits SILENT inference of
    sentiment from the news score; it does not prohibit an explicitly
    designed and documented derived feature. This function is that design,
    made enforceable:

    - Fail-closed: any news snapshot that is not fully healthy (status OK
      with an in-range numeric sentiment_score and source_confidence) yields
      the standard UNAVAILABLE placeholder — degraded or contradictory news
      evidence can never be laundered into a sentiment value.
    - Explicit labeling: the payload is stamped
      `derivation: "derived_from_news"` and
      `source_id: "sentiment_derived_from_news"`, so no downstream reader
      can mistake it for provider sentiment.
    - Dependency in confidence: `source_confidence` is the news snapshot's
      confidence scaled by `SENTIMENT_DERIVED_CONFIDENCE_SCALE` (0.5) — the
      single-source dependency is priced in, not just documented.

    NOT wired into the production pipeline: the live path remains
    `fetch_sentiment_snapshot` (UNAVAILABLE). Promoting this function into
    a production model additionally requires a canonical feature-registry
    entry (Sprint M1 rule: an unregistered feature cannot enter a
    production model) and an explicit ensemble-weight decision.

    Pure function: reads only the news snapshot argument; no fetches, no
    wall clock, no hidden state.
    """
    if not _is_valid_news_evidence(news_snapshot):
        blocked = (
            news_snapshot.get("status", "malformed")
            if isinstance(news_snapshot, dict)
            else "malformed"
        )
        return _unavailable_snapshot(
            ticker,
            as_of,
            reason=(
                "News-derived sentiment refused (fail-closed): the news "
                f"snapshot is not fully healthy (status={blocked!r}); sentiment "
                "is never fabricated from degraded or contradictory evidence."
            ),
        )

    scaled_confidence = round(
        float(news_snapshot["source_confidence"]) * SENTIMENT_DERIVED_CONFIDENCE_SCALE, 6
    )
    return SentimentSnapshot(
        ticker=str(ticker).upper(),
        as_of=str(as_of),
        status="OK",
        source_id=SENTIMENT_DERIVED_SOURCE_ID,
        source_confidence=scaled_confidence,
        published_time=news_snapshot.get("published_time"),
        calculation_version=SENTIMENT_CONTRACT_VERSION,
        lookback_period=str(news_snapshot.get("lookback_period", "N/A")),
        sentiment_score=float(news_snapshot["sentiment_score"]),
        derivation="derived_from_news",
        intended_inputs=["news_snapshot.sentiment_score", "news_snapshot.source_confidence"],
        reason=(
            "Explicit news-derived sentiment feature (anti-proxying rule "
            "satisfied by labeling, not silent inference): value equals the OK "
            "news snapshot's aggregated tone; confidence = news source_confidence "
            f"x {SENTIMENT_DERIVED_CONFIDENCE_SCALE} "
            "(SENTIMENT_DERIVED_CONFIDENCE_SCALE) to price the single-source "
            "dependency."
        ),
    ).to_dict()