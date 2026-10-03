"""Finnhub company news — the provider that fits the actual need.

**WHY A SECOND PROVIDER AT ALL.** MEASURED, NewsAPI's free tier allows 100
requests/day and the portfolio needs roughly 150: 77 tickers for the collector
plus ad-hoc scoring. The gap caused five consecutive nights of permanently lost
news before the schedule was fixed, and still forces the collector to rotate 40 of
75 eligible tickers per run. NewsAPI's only paid tier is **$449/month for 250,000
requests** — roughly 55x more quota than this system needs, at $5,388/year.

Finnhub's free tier is **60 calls per minute**, about 86,000/day. The constraint
simply disappears, at no cost.

**IT ALSO FIXES THE ENTITY-RESOLUTION PROBLEM AT THE SOURCE.** Finnhub's
`/company-news` is queried BY SYMBOL and returns a `related` field naming the
tickers each article is about. NewsAPI is a keyword search: asking it for "V"
returns every article containing the letter V, which is why MEASURED 93 of 116
articles on collision-prone tickers were about Chevrolet Corvettes, cat memes and
boxing matches. A provider-resolved symbol is a different and stronger claim than
a token appearing in text, and `resolve_relevance` already treats it as such.

**WHAT IS DELIBERATELY NOT CHANGED.** The canonical record shape, the
point-in-time filter, the tone lexicon, the category classifier, the credibility
gates and the W6 raw ledger are all shared. This module FETCHES and NORMALISES;
every judgement downstream is the one that already existed. Two providers with two
pipelines would be a W5 violation, and the measurements behind those gates would
no longer describe what runs.

**NORTH AMERICA ONLY**, per Finnhub's own documentation. A holding it cannot cover
falls back to NewsAPI rather than being reported as having no news — "we could not
look here" is not "nothing happened", which is the distinction A9 exists to keep.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from core.config import (
    FINNHUB_API_KEY_ENV,
    FINNHUB_MAX_ARTICLES,
    FINNHUB_NEWS_URL,
    FINNHUB_SOURCE_ID,
    NEWS_LOOKBACK_DAYS,
    NEWS_PROVIDER_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("finnhub_news")


def available() -> bool:
    """Whether a Finnhub key is configured. Never raises."""
    return bool(os.getenv(FINNHUB_API_KEY_ENV))


def _published(entry: dict) -> str | None:
    """A UNIX timestamp rendered as the ISO string the PIT filter expects.

    Returns None when the stamp is missing or unparseable, so the record is
    dropped rather than defaulting to "now" — an article dated to the moment of
    the fetch would pass any point-in-time check by construction.
    """
    raw = entry.get("datetime")
    # Narrowed to the types `int()` actually accepts. The `in (None, "", 0)`
    # guard above reads as a None check but does not narrow for a type checker,
    # and a dict or list reaching int() would raise TypeError rather than being
    # skipped -- which this function exists to prevent.
    if not isinstance(raw, (int, float, str)) or isinstance(raw, bool):
        return None
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def normalize(payload: list, ticker: str) -> list[dict]:
    """Map a Finnhub payload onto the canonical article records.

    Defensive in the same way as the NewsAPI normaliser: a non-dict entry, a
    missing headline or an unreadable timestamp is SKIPPED, so one malformed row
    can never quietly become evidence.

    The decisive difference is `ticker`. Finnhub resolved the entity itself, so
    the field is populated — and `resolve_relevance` then scores a
    provider-asserted match at 1.0 without requiring corroboration, because the
    provider making the claim is stronger evidence than a symbol appearing in
    text.
    """
    records: list[dict] = []
    for index, entry in enumerate(payload or []):
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("headline") or "").strip()
        if not headline:
            continue
        published = _published(entry)
        if published is None:
            continue

        # `related` is a comma-separated symbol list. The requested ticker is
        # only claimed when the provider actually names it: trusting the request
        # parameter instead would assert an entity resolution the provider never
        # made, which is precisely the claim this field exists to carry.
        related = {
            part.strip().upper()
            for part in str(entry.get("related") or "").split(",")
            if part.strip()
        }
        resolved = ticker.upper() if ticker.upper() in related else None

        records.append({
            "source_record_id": str(
                entry.get("id") or entry.get("url") or f"{headline[:80]}#{index}"
            ),
            "published_time": published,
            "headline": headline,
            "summary": str(entry.get("summary") or ""),
            "url": str(entry.get("url") or ""),
            "source_name": str(entry.get("source") or ""),
            "ticker": resolved,
            "company_name": None,
            "source_quality": None,
            "tone": None,
        })
    return records


def fetch_company_news(
    ticker: str,
    as_of_dt: datetime,
    timeout: float = NEWS_PROVIDER_TIMEOUT_SECONDS,
) -> dict:
    """Fetch raw Finnhub records for the point-in-time window.

    Returns the same disposition shape as `news_adapter.fetch_provider_articles`
    so the two are interchangeable to every caller:
    ``{"status", "records", "reason"}``. Never raises — a failed request is an
    explicit disposition, never empty data pretending to be coverage.
    """
    api_key = os.getenv(FINNHUB_API_KEY_ENV)
    if not api_key:
        return {
            "status": "provider_key_required",
            "records": [],
            "reason": f"No {FINNHUB_API_KEY_ENV} configured.",
        }

    window_start = (as_of_dt - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    url = (
        f"{FINNHUB_NEWS_URL}"
        f"?symbol={urllib.parse.quote(ticker)}"
        f"&from={window_start}"
        f"&to={as_of_dt.strftime('%Y-%m-%d')}"
        f"&token={urllib.parse.quote(api_key)}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "invest-by-score/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # 429 IS NAMED SEPARATELY. It is the one failure that means "stop
        # asking" rather than "try again", and conflating it with a transient
        # error is how a quota overage deepens.
        reason = (
            "finnhub rate limit reached (HTTP 429); the run should stop rather "
            "than retry"
            if exc.code == 429
            else f"finnhub request failed: HTTP {exc.code}"
        )
        LOGGER.warning("finnhub_request_failed: %s", reason)
        return {"status": "provider_request_failed", "records": [], "reason": reason}
    except (
        urllib.error.URLError,
        json.JSONDecodeError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.warning("finnhub_request_failed: %s", exc)
        return {
            "status": "provider_request_failed",
            "records": [],
            "reason": f"finnhub request failed: {exc}",
        }

    if not isinstance(payload, list):
        # The endpoint returns a bare JSON array. A dict here is an error body
        # ({"error": "..."}), which must not be read as zero articles.
        message = (
            payload.get("error")
            if isinstance(payload, dict)
            else f"unexpected payload type {type(payload).__name__}"
        )
        LOGGER.warning("finnhub_rejected_request: %s", message)
        return {
            "status": "provider_request_failed",
            "records": [],
            "reason": f"finnhub rejected the request: {message}",
        }

    records = normalize(payload, ticker)
    # Newest first, then capped. Ordering BEFORE the cap matters: truncating an
    # arbitrary order would silently drop the most recent articles.
    records.sort(key=lambda row: str(row.get("published_time") or ""), reverse=True)
    return {
        "status": "ok",
        "records": records[:FINNHUB_MAX_ARTICLES],
        "reason": "",
        "source_id": FINNHUB_SOURCE_ID,
    }
