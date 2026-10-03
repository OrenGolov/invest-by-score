"""News adapter (Sprint N1) — real ingestion, classification, contradiction.

Pipeline (every stage a pure, deterministic function; `build_news_snapshot`
wires them in order):

    NEWS -> PIT FILTER -> ENTITY RESOLUTION -> EVENT CLASSIFICATION
         -> SOURCE QUALITY -> RELEVANCE/NOVELTY -> DIRECTION/MAGNITUDE
         -> CONTRADICTION DETECTION -> EVIDENCE-BACKED OUTPUT

Governance rules enforced here:

- Provider gate: without NEWS_PROVIDER_API_KEY the contract is the explicit
  UNAVAILABLE snapshot, byte-for-byte identical to the pre-N1 stub. A missing
  provider is a status, never a neutral score.
- Point-in-time: only articles with published_time <= as_of are eligible.
  Future-dated or unparseable publication times invalidate the payload and
  force agent status INVALID (timestamp violation — fail closed).
- No fabrication: tone is provider-supplied or lexicon-derived (stamped per
  record via `tone_derivation`); it is never inferred from price, RSI,
  volume, or any other technical indicator.
- Source quality: per-article weight = registry base confidence x per-source
  quality x exponential recency decay. A zero-quality source cannot raise
  confidence and cannot drive a contradiction (credibility gate).
- Relevance/novelty: relevance-0 (off-entity) and duplicate-headline articles
  stay in the evidence but are excluded from aggregation.
- Contradiction v1: a same-day, same-category cluster of credible articles
  whose positive/negative mean tones are opposite-sign with |delta| above
  NEWS_CONTRADICTION_TONE_DELTA yields status CONTRADICTORY with both sides
  surfaced. Contradiction floors confidence and never averages to a
  reassuring neutral.
- Aggregation: relevance- and source-weighted, recency-decayed tone mean;
  confidence = f(sample count, mean source weight, dispersion) capped by the
  weakest contributing source.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from datetime import datetime, timedelta, timezone

from core.config import (
    NEWS_AGGREGATOR_VERSION,
    NEWS_BASE_SOURCE_CONFIDENCE,
    FINNHUB_API_KEY_ENV,
    FINNHUB_SOURCE_ID,
    NEWS_CLASSIFIER_VERSION,
    NEWS_COLLISION_PRONE_TICKERS,
    NEWS_CONTRADICTION_CONFIDENCE_FLOOR,
    NEWS_CONTRADICTION_TONE_DELTA,
    NEWS_FINANCE_MARKERS,
    NEWS_CONTRACT_VERSION,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_ARTICLES,
    NEWS_PIPELINE_VERSION,
    NEWS_PROVIDER_API_KEY_ENV,
    NEWS_PROVIDER_ORDER,
    NEWS_PROVIDER_TIMEOUT_SECONDS,
    NEWS_PROVIDER_URL,
    NEWS_RECENCY_HALF_LIFE_DAYS,
    NEWS_TONE_LEXICON_VERSION,
)
from core.raw_store import append_raw_records
from core.schemas import NewsSnapshot

LOGGER = logging.getLogger("core.news_adapter")

NEWS_SOURCE_ID = "newsapi_news"
NEWS_PROVIDER_NAME = "NewsAPI"
UNAVAILABLE_SOURCE_ID = "news_provider_unconfigured"
UNAVAILABLE_REASON = (
    "No verified news provider is connected. Sentiment is not inferred from "
    "price or technical indicators."
)

# --- Event classification v1 (curated pattern sets as data constants) -----------
# Order is semantic: first matching category wins, and a headline matching none
# falls through to `other`. The taxonomy lives here — not sprinkled through
# logic — so it can be reviewed and versioned as one artifact.
NEWS_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("earnings", (
        r"\bearnings\b", r"\bquarterly (results|report|profit)\b", r"\bq[1-4] (results|earnings)\b",
        r"\beps\b", r"\brevenue (beat|miss)\b", r"\bprofit (beat|miss|warning)\b",
    )),
    ("guidance", (
        r"\bguidance\b", r"\boutlook\b", r"\bforecast\b",
        r"\brais(?:es|ed|ing) (?:its )?(?:guidance|outlook|forecast)\b",
        r"\bcut(?:s|ting)? (?:its )?(?:guidance|outlook|forecast)\b",
    )),
    ("litigation", (
        r"\blawsuit\b", r"\bsued\b", r"\bsues\b", r"\blitigation\b", r"\bclass action\b",
        r"\bsettlement\b", r"\bcourt\b", r"\bindict\w+\b",
        r"\b(?:derivative|counters?|defamation) suit\b", r"\bfiles? (?:a )?suit\b",
        r"\bsuit (?:names|alleges|accuses|against)\b",
    )),
    ("regulation", (
        r"\bregulator\w*\b", r"\bregulation\b", r"\bantitrust\b",
        r"\bsec (?:probe|investigation|charges|files|reporting|rules|requirements|fines|settlement|subpoena)\b",
        r"\bfined\b", r"\bpenalt\w+\b",
        r"\bcompliance\b", r"\bbanned\b",
    )),
    ("product_launch", (
        r"\blaunche?s?\b", r"\blaunching\b", r"\bunveil\w*\b", r"\bdebut\w*\b",
        r"\bannounces? new (?:product|device|chip|model|service)\b",
        r"\breleas(?:es|ed|ing) (?:the )?new\b",
    )),
    ("macro_shock", (
        r"\binflation\b", r"\brate (?:hike|hikes|cut|cuts)\b", r"\bfed\b", r"\bfomc\b",
        r"\bcpi\b", r"\bgdp\b", r"\brecession\b", r"\bjobs report\b", r"\bunemployment\b",
        r"\btariff\w*\b", r"\binterest rates\b",
    )),
    ("m_and_a", (
        r"\bacquisition\b", r"\bacquire[sd]?\b", r"\bmerger\b", r"\bmerges\b",
        r"\bbuyout\b", r"\btakeover\b", r"\bdivest\w*\b",
    )),
    ("strategic_announcement", (
        r"\bstrategic (?:partnership|review|alliance)\b", r"\brestructur\w+\b",
        r"\bspin-?off\b", r"\bjoint venture\b", r"\bexpansion (?:plan|into)\b",
    )),
    ("management_commentary", (
        r"\b(?:ceo|cfo|coo)\b", r"\bexecutive (?:said|says|comments)\b",
        r"\bmanagement (?:said|says|comments|notes)\b", r"\bin an interview\b",
        r"\bon the (?:earnings )?call\b", r"\bcomments on\b",
    )),
)
NEWS_CATEGORY_COMPILED: tuple[tuple[str, tuple[re.Pattern, ...]], ...] = tuple(
    (category, tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns))
    for category, patterns in NEWS_CATEGORY_PATTERNS
)

# --- Tone lexicon v1 (curated data constants) ------------------------------------
# Exact matching on lowercased word tokens. A negator within the three tokens
# preceding a signal term flips that term's polarity. Tone is the net polarity
# normalized by the number of signal terms, so it stays within [-1, 1].
NEWS_TONE_POSITIVE: frozenset[str] = frozenset({
    "beat", "beats", "surge", "surges", "surged", "soar", "soars", "soared",
    "record", "upgrade", "upgrades", "upgraded", "raises", "raised", "growth",
    "profit", "profits", "strong", "stronger", "wins", "approved", "approval",
    "tops", "exceeds", "rally", "rallies", "gain", "gains", "jump", "jumps",
    "boost", "breakthrough", "expands", "success", "rebound", "outperform",
    "buyback", "dividend", "recovery",
})
NEWS_TONE_NEGATIVE: frozenset[str] = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "sink", "sinks",
    "slump", "slumps", "drop", "drops", "fall", "falls", "fell", "cut", "cuts",
    "downgrade", "downgrades", "downgraded", "probe", "probes", "investigation",
    "lawsuit", "recall", "recalls", "weak", "weaker", "warning", "warns",
    "decline", "declines", "loss", "losses", "fail", "fails", "halted",
    "bankruptcy", "fraud", "tumble", "tumbles", "layoffs", "shortfall",
    "delayed", "sued", "sues", "fined", "crisis",
})
NEWS_TONE_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "without", "denies", "denied", "denying", "never", "cannot", "cant",
})


def _parse_timestamp(value: object) -> datetime | None:
    """Parse a timestamp into naive UTC. Returns None when unparseable."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def classify_event(text: str) -> str:
    """EVENT CLASSIFICATION v1: map text onto the taxonomy; first match wins."""
    haystack = str(text or "")
    for category, patterns in NEWS_CATEGORY_COMPILED:
        if any(pattern.search(haystack) for pattern in patterns):
            return category
    return "other"


def lexicon_tone(text: str) -> float:
    """Tone lexicon v1 in [-1, 1]: net polarity over signal terms.

    Negation handling: a negator within the three tokens preceding a signal
    term flips that term's polarity. Pure and deterministic.
    """
    tokens = re.findall(r"[a-z]+", str(text or "").lower())
    polarity = 0
    signal_terms = 0
    for index, token in enumerate(tokens):
        sign = 0
        if token in NEWS_TONE_POSITIVE:
            sign = 1
        elif token in NEWS_TONE_NEGATIVE:
            sign = -1
        if sign == 0:
            continue
        signal_terms += 1
        window = tokens[max(0, index - 3):index]
        if any(negator in window for negator in NEWS_TONE_NEGATORS):
            sign = -sign
        polarity += sign
    if signal_terms == 0:
        return 0.0
    return round(polarity / signal_terms, 4)


def resolve_tone(record: dict) -> tuple[float, str]:
    """DIRECTION/MAGNITUDE v1: provider tone when usable, lexicon otherwise.

    Returns (tone, derivation). The derivation is stamped on the per-article
    record so no consumer can confuse provider tone with the v1 lexicon
    fallback. An out-of-range provider tone is never trusted.
    """
    provider_tone = record.get("tone")
    if provider_tone is not None:
        try:
            tone = float(provider_tone)
        except (TypeError, ValueError):
            tone = None
        if tone is not None and -1.0 <= tone <= 1.0:
            return round(tone, 4), "provider"
        LOGGER.warning("news_provider_tone_out_of_range: %r", provider_tone)
    text = f"{record.get('headline', '')} {record.get('summary', '')}"
    return lexicon_tone(text), f"lexicon:{NEWS_TONE_LEXICON_VERSION}"


# Finance markers, precompiled WITH WORD BOUNDARIES, once at import.
#
# THE SUBSTRING BUG, CAUGHT TWICE. An unbounded match let "rally" match
# inside "liteRALLY" and "neutRALLY", admitting a baseball report for ARM
# and a developer-jargon article for KO. The fix for THAT was written through
# a shell heredoc, which turned its word-boundary anchors into literal
# BACKSPACE characters -- so the pattern then matched nothing instead of too
# much, rejecting two genuine articles AND invalidating the measurement taken
# against it.
#
# Hence one visible place, built once, with no interpolation at the call site.
_WORD_BOUNDARY = chr(92) + "b"
_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(_WORD_BOUNDARY + re.escape(marker) + _WORD_BOUNDARY)
    for marker in NEWS_FINANCE_MARKERS
)


def _corroborated(ticker: str, text: str) -> bool:
    """Does anything beyond the bare symbol tie this text to the company?

    Two independent signals, either of which suffices:

    * THE COMPANY NAME. "Caterpillar" in an article mentioning CAT is decisive.
    * A FINANCE MARKER. MEASURED, the marker list is deliberately broad and that
      breadth does NOT reintroduce the collisions -- a cat-meme listicle contains
      none of "stock", "earnings", "analyst" or "nasdaq", while genuine coverage
      contains at least one. A narrow list would drop real articles instead.
    """
    low = text.lower()
    # A company NAME may match as a substring: "coca-cola" and "arm holdings"
    # are multi-word phrases, and bounding them adds nothing.
    if any(name in low for name in NEWS_COLLISION_PRONE_TICKERS.get(ticker, ())):
        return True
    return any(pattern.search(low) for pattern in _MARKER_PATTERNS)


def resolve_relevance(record: dict, ticker: str) -> float:
    """ENTITY RESOLUTION v1: how clearly the article is about `ticker`.

    1.0 when the ticker matches (provider-matched or present as a token in
    the text); 0.7 for a company-name token match; 0.0 otherwise. Relevance
    0 keeps the article in the evidence but excludes it from aggregation.

    A BARE SYMBOL IS NOT ENOUGH FOR A COLLISION-PRONE TICKER. MEASURED over
    6,360 captured articles, 93 of the 116 admitted on the 15 symbols that are
    English words or colliding abbreviations were not about the company at all
    -- a Chevrolet Corvette for ARM, cat memes for CAT, boxing for KO, a PyPI
    package for KEEL, 10 of 10 for V. Those symbols require corroboration by the
    company name or a finance marker; every other ticker is unaffected.
    """
    provider_ticker = str(record.get("ticker") or "").strip().upper()
    if provider_ticker == ticker:
        # A PROVIDER-ASSERTED match needs no corroboration: the provider resolved
        # the entity itself, which is a different and stronger claim than the
        # symbol happening to appear in the text.
        return 1.0
    text = f"{record.get('headline', '')} {record.get('summary', '')}"
    tokens = set(re.findall(r"[A-Za-z0-9]+", text.upper()))
    if ticker in tokens:
        if ticker in NEWS_COLLISION_PRONE_TICKERS and not _corroborated(
            ticker, text
        ):
            # The symbol appears but nothing confirms the article is about the
            # company. Falls through to the company-name check below, which is
            # the stricter test and the only one that can still admit it.
            pass
        else:
            return 1.0
    name = str(record.get("company_name") or "").strip().upper()
    if name:
        name_tokens = {token for token in re.findall(r"[A-Za-z0-9]+", name) if len(token) > 1}
        if name_tokens and name_tokens.issubset(tokens):
            return 0.7

    # THE CONFIGURED COMPANY NAME, for a collision-prone ticker whose article
    # never prints the symbol at all.
    #
    # CAUGHT BY PROBE: "Caterpillar raises full-year guidance" has no bare CAT
    # token, and the branch above reads `record["company_name"]`, which the real
    # NewsAPI payload sets to None -- MEASURED on live records. So six genuine
    # articles (Caterpillar, Coca-Cola, Visa, ServiceNow, American Electric
    # Power, Bloom Energy) had NO path to admission once the symbol rule
    # tightened. Naming the company is the strongest possible signal and must
    # never score lower than the bare symbol it replaces.
    if ticker in NEWS_COLLISION_PRONE_TICKERS:
        low = text.lower()
        if any(
            candidate in low
            for candidate in NEWS_COLLISION_PRONE_TICKERS[ticker]
        ):
            return 1.0
    return 0.0


def pit_filter(records: list[dict], as_of_dt: datetime) -> tuple[list[tuple[dict, datetime]], list[dict]]:
    """PIT FILTER: keep records with published_time <= as_of (inclusive).

    Unparseable and future-dated records are rejected with explicit reasons;
    their presence in a provider payload invalidates it (status INVALID).
    """
    eligible: list[tuple[dict, datetime]] = []
    rejected: list[dict] = []
    for record in records:
        published_dt = _parse_timestamp(record.get("published_time"))
        if published_dt is None:
            rejected.append({
                "source_record_id": str(record.get("source_record_id", "")),
                "headline": str(record.get("headline", "")),
                "reason": "unparseable_published_time",
            })
            continue
        if published_dt > as_of_dt:
            rejected.append({
                "source_record_id": str(record.get("source_record_id", "")),
                "headline": str(record.get("headline", "")),
                "reason": "future_dated",
            })
            continue
        eligible.append((record, published_dt))
    return eligible, rejected


def source_weight(record: dict, published_dt: datetime, as_of_dt: datetime) -> float:
    """SOURCE QUALITY v1: base confidence x per-source quality x recency decay.

    - Per-source quality defaults to the neutral factor 1.0 when the provider
      attests nothing (documented neutral semantics; an explicit 0.0 still
      means zero-quality). A malformed attestation fails closed to 0.0.
    - Recency decay is exponential with NEWS_RECENCY_HALF_LIFE_DAYS half-life;
      v1 counts calendar days between publication and as_of.
    """
    quality = record.get("source_quality")
    if quality is None:
        quality_factor = 1.0
    else:
        try:
            quality_factor = max(0.0, min(1.0, float(quality)))
        except (TypeError, ValueError):
            quality_factor = 0.0
    age_days = max(0.0, (as_of_dt - published_dt).total_seconds() / 86400.0)
    decay = 0.5 ** (age_days / NEWS_RECENCY_HALF_LIFE_DAYS)
    return round(min(1.0, NEWS_BASE_SOURCE_CONFIDENCE * quality_factor) * decay, 6)


def resolve_news_provider(ticker: str, as_of: str) -> dict:
    """Resolve the active news provider, mirroring the fundamentals pattern."""
    # EITHER provider's key makes the pipeline live. Checking only NewsAPI's
    # would report "no provider" on a machine configured solely for Finnhub.
    api_key = os.getenv(NEWS_PROVIDER_API_KEY_ENV) or os.getenv(
        FINNHUB_API_KEY_ENV
    )
    if api_key:
        return {
            "provider": NEWS_PROVIDER_NAME,
            "source_type": "news",
            "source_id": NEWS_SOURCE_ID,
            "status": "live_provider",
            "fallback_rank": 1,
            "fallbacks": [NEWS_PROVIDER_NAME],
            "source_confidence": NEWS_BASE_SOURCE_CONFIDENCE,
            "selection_reason": f"{NEWS_PROVIDER_API_KEY_ENV} configured; provider selected for live news coverage.",
            "ticker": ticker.upper(),
            "as_of": as_of,
        }
    return {
        "provider": "news_provider_unconfigured",
        "source_type": "news",
        "source_id": UNAVAILABLE_SOURCE_ID,
        "status": "provider_key_required",
        "fallback_rank": 0,
        "fallbacks": [NEWS_PROVIDER_NAME],
        "source_confidence": 0.0,
        "selection_reason": (
            f"No {NEWS_PROVIDER_API_KEY_ENV} configured; the news contract stays "
            "explicitly UNAVAILABLE until a trusted provider is available."
        ),
        "ticker": ticker.upper(),
        "as_of": as_of,
    }


def _normalize_provider_payload(payload: dict) -> list[dict]:
    """Map a provider payload onto canonical article records.

    Defensive by design: non-dict entries and entries without a headline are
    skipped, so one malformed row can never silently become data.
    """
    records: list[dict] = []
    for index, entry in enumerate(payload.get("articles") or []):
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("title") or "").strip()
        if not headline:
            continue
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        records.append({
            "source_record_id": str(entry.get("url") or f"{headline[:80]}#{index}"),
            "published_time": entry.get("publishedAt"),
            "headline": headline,
            "summary": str(entry.get("description") or ""),
            "url": str(entry.get("url") or ""),
            "source_name": str(source.get("name") or ""),
            "ticker": None,
            "company_name": None,
            "source_quality": None,
            "tone": None,
        })
    return records


def fetch_provider_articles(ticker: str, as_of_dt: datetime, timeout: float = NEWS_PROVIDER_TIMEOUT_SECONDS) -> dict:
    """Fetch raw provider records for the point-in-time window.

    Returns {"status": "ok"|"provider_key_required"|"provider_request_failed",
    "records": [...], "reason": str}. Never raises: a failed request is an
    explicit disposition, never empty data pretending to be coverage.
    """
    api_key = os.getenv(NEWS_PROVIDER_API_KEY_ENV)
    if not api_key:
        return {
            "status": "provider_key_required",
            "records": [],
            "reason": f"No {NEWS_PROVIDER_API_KEY_ENV} configured.",
        }
    window_start = (as_of_dt - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    url = (
        f"{NEWS_PROVIDER_URL}"
        f"?q={urllib.parse.quote(ticker)}"
        f"&from={window_start}"
        f"&to={as_of_dt.strftime('%Y-%m-%d')}"
        f"&language=en&sortBy=publishedAt&pageSize={NEWS_MAX_ARTICLES}"
        f"&apikey={api_key}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "invest-by-score/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.warning("news_provider_request_failed: %s", exc)
        return {"status": "provider_request_failed", "records": [], "reason": f"news provider request failed: {exc}"}
    if str(payload.get("status", "ok")).lower() != "ok":
        reason = f"provider rejected the request: {payload.get('message', 'unknown error')}"
        LOGGER.warning("news_provider_rejected_request: %s", reason)
        return {"status": "provider_request_failed", "records": [], "reason": reason}
    return {"status": "ok", "records": _normalize_provider_payload(payload), "reason": ""}


def detect_contradictions(credible: list[dict]) -> list[dict]:
    """CONTRADICTION DETECTION v1 over credible articles.

    A contradiction is a same-day (UTC date of publication), same-category
    cluster whose positive and negative mean tones are opposite-sign with
    |delta| strictly above NEWS_CONTRADICTION_TONE_DELTA. Both sides are
    surfaced with their source record ids. Zero-quality articles never reach
    this stage (credibility gate), so one shaky headline cannot make news
    contradictory.
    """
    clusters: dict[tuple[str, str], list[dict]] = {}
    for article in credible:
        clusters.setdefault((article["cluster_date"], article["category"]), []).append(article)
    contradictions: list[dict] = []
    for (cluster_date, category), group in sorted(clusters.items()):
        if len(group) < 2:
            continue
        positives = [article for article in group if article["tone"] > 0.0]
        negatives = [article for article in group if article["tone"] < 0.0]
        if not positives or not negatives:
            continue
        positive_mean = sum(article["tone"] for article in positives) / len(positives)
        negative_mean = sum(article["tone"] for article in negatives) / len(negatives)
        delta = abs(positive_mean - negative_mean)
        if delta <= NEWS_CONTRADICTION_TONE_DELTA:
            continue
        contradictions.append({
            "cluster_date": cluster_date,
            "category": category,
            "tone_delta": round(delta, 4),
            "positive": {
                "mean_tone": round(positive_mean, 4),
                "source_record_ids": [article["source_record_id"] for article in positives],
            },
            "negative": {
                "mean_tone": round(negative_mean, 4),
                "source_record_ids": [article["source_record_id"] for article in negatives],
            },
        })
    return contradictions


def aggregate_articles(credible: list[dict]) -> tuple[float | None, float]:
    """DIRECTION/MAGNITUDE + aggregation v1.

    sentiment_score is the relevance- and source-weighted, recency-decayed
    tone mean. confidence = f(sample count, mean source weight, dispersion),
    capped by the weakest contributing source. Returns (sentiment, confidence);
    (None, 0.0) when nothing is aggregatable.
    """
    if not credible:
        return None, 0.0
    total_weight = sum(article["source_weight"] * article["relevance"] for article in credible)
    if total_weight <= 0.0:
        return None, 0.0
    sentiment = sum(
        article["source_weight"] * article["relevance"] * article["tone"]
        for article in credible
    ) / total_weight
    count_term = min(1.0, len(credible) / 5.0)
    mean_weight = sum(article["source_weight"] for article in credible) / len(credible)
    dispersion = min(1.0, sum(abs(article["tone"] - sentiment) for article in credible) / len(credible))
    dispersion_term = max(0.0, 1.0 - dispersion)
    weakest_source = min(article["source_weight"] for article in credible)
    confidence = max(0.0, min(count_term, mean_weight, dispersion_term, weakest_source))
    return round(sentiment, 4), round(confidence, 4)


def enrich_articles(
    eligible: list[tuple[dict, datetime]],
    ticker: str,
    as_of_dt: datetime,
) -> list[dict]:
    """Score, classify and gate point-in-time-eligible records. PURE.

    Extracted from `build_news_snapshot` so a caller that already HAS the raw
    records -- the alert runner, reading what the collector captured into the W6
    ledger -- can enrich them without spending a provider request.

    MEASURED: the alert runner calling `build_news_snapshot` per ticker logged 62
    HTTP 429s over 77 tickers, because each call is one request and the collector
    had already spent ~40 of the free tier's 100 that day. Copying this pipeline
    into the runner would have been a W5 violation; both callers use this.

    Nothing here touches the network. Excluded articles STAY in the output with
    an explicit `exclusion_reason`, because evidence that was considered and
    rejected is different from evidence that never existed.
    """
    enriched: list[dict] = []
    for record, published_dt in eligible:
        tone, derivation = resolve_tone(record)
        enriched.append({
            "source_id": NEWS_SOURCE_ID,
            "source_record_id": str(record.get("source_record_id", "")),
            "published_time": str(record.get("published_time") or ""),
            "published_dt": published_dt,
            "cluster_date": published_dt.date().isoformat(),
            "headline": str(record.get("headline", "")),
            "url": str(record.get("url") or ""),
            "category": classify_event(f"{record.get('headline', '')} {record.get('summary', '')}"),
            "tone": tone,
            "tone_derivation": derivation,
            "relevance": resolve_relevance(record, ticker),
            "source_weight": source_weight(record, published_dt, as_of_dt),
            "included_in_aggregation": True,
            "exclusion_reason": "",
        })

    # Deterministic order (recency, then record id) before the novelty dedupe
    # so `first wins` is reproducible.
    enriched.sort(key=lambda article: (article["published_dt"], article["source_record_id"]))
    seen_headlines: set[str] = set()
    for article in enriched:
        normalized = re.sub(r"[^a-z0-9]+", " ", article["headline"].lower()).strip()
        if normalized and normalized in seen_headlines:
            article["included_in_aggregation"] = False
            article["exclusion_reason"] = "duplicate_headline"
        elif normalized:
            seen_headlines.add(normalized)

    # NOVELTY + relevance/credibility gates: excluded articles stay in the
    # evidence with an explicit reason, but never enter the aggregation.
    for article in enriched:
        if not article["included_in_aggregation"]:
            continue
        if article["relevance"] <= 0.0:
            article["included_in_aggregation"] = False
            article["exclusion_reason"] = "zero_relevance"
        elif article["source_weight"] <= 0.0:
            article["included_in_aggregation"] = False
            article["exclusion_reason"] = "zero_source_weight"

    return enriched


def enrich_captured_news(raw: Mapping[str, Any], ticker: str, as_of: str) -> dict:
    """A snapshot built from ALREADY-CAPTURED records, with no provider call.

    The alert runner reads the day's news out of the W6 raw ledger and needs it
    in the same shape `build_news_snapshot` produces, so A9 reads one payload
    format regardless of where the records came from.

    The point-in-time filter still applies: a record published after `as_of` is
    rejected here exactly as it would be on a live fetch, because replaying a
    ledger must not relax the policy that governed its capture.
    """
    records = raw.get("records") if isinstance(raw, Mapping) else None
    if not isinstance(records, list) or not records:
        return _unavailable_snapshot(
            ticker, as_of, reason="no captured news records", source_id=NEWS_SOURCE_ID
        )

    # A BARE DATE MEANS THE WHOLE DAY, NOT MIDNIGHT.
    #
    # CAUGHT BY TEST: `_parse_timestamp("2026-10-01")` returns 00:00:00, so an
    # article published at 18:00 that same day was rejected as FUTURE-DATED and
    # the replay reported INVALID with zero articles. Since the collector stores
    # under a bare-date request key, that would have discarded essentially every
    # captured article -- the replay path would have looked like a provider
    # outage on every single ticker.
    #
    # The live path never hit this because `build_news_snapshot` receives an
    # as_of that already carries a time.
    text = str(as_of).strip()
    as_of_dt = (
        _parse_timestamp(f"{text}T23:59:59Z")
        if len(text) == 10
        else _parse_timestamp(text)
    )
    if as_of_dt is None:
        return _unavailable_snapshot(
            ticker, as_of, reason=f"unparseable as_of {as_of!r}",
            source_id=NEWS_SOURCE_ID,
        )

    eligible, rejected = pit_filter(list(records), as_of_dt)
    enriched = enrich_articles(eligible, ticker, as_of_dt)
    credible = [a for a in enriched if a["included_in_aggregation"]]
    contradictions = detect_contradictions(credible)

    if rejected:
        status, reason = "INVALID", (
            f"{len(rejected)} captured record(s) violate the point-in-time "
            f"policy; rejected fail-closed"
        )
    elif contradictions:
        status, reason = "CONTRADICTORY", (
            f"{len(contradictions)} same-day, same-category contradiction "
            f"cluster(s) exceed the tone tolerance"
        )
    elif not credible:
        status, reason = "INCOMPLETE", (
            "no credible, on-entity captured articles inside the window"
        )
    else:
        status, reason = "OK", ""

    return {
        "ticker": ticker,
        "as_of": as_of,
        "status": status,
        "reason": reason,
        "source_id": NEWS_SOURCE_ID,
        "articles": [
            {key: value for key, value in article.items() if key != "published_dt"}
            for article in enriched
        ],
        "replayed_from_ledger": True,
    }


def _unavailable_snapshot(ticker: str, as_of: str, reason: str = UNAVAILABLE_REASON, source_id: str = UNAVAILABLE_SOURCE_ID) -> dict:
    """The explicit UNAVAILABLE contract.

    The no-key call must stay byte-for-byte identical to the pre-N1 stub;
    tests pin the exact dict.
    """
    return NewsSnapshot(
        ticker=ticker,
        as_of=str(as_of),
        status="UNAVAILABLE",
        source_id=source_id,
        source_confidence=0.0,
        published_time=None,
        calculation_version=NEWS_CONTRACT_VERSION,
        lookback_period="N/A",
        sentiment_score=None,
        articles=[],
        reason=reason,
    ).to_dict()


def fetch_news_records(
    ticker: str,
    as_of_dt: datetime,
    timeout: float = NEWS_PROVIDER_TIMEOUT_SECONDS,
) -> dict:
    """Fetch from the first provider in NEWS_PROVIDER_ORDER that answers.

    Returns the usual disposition plus ``source_id`` naming the provider that
    actually supplied the records, so the W6 ledger can record provenance rather
    than filing everything under one id.

    A provider with no key configured is SKIPPED rather than treated as a
    failure: "not configured here" and "asked and refused" are different facts,
    and only the second is a provider outage.

    Finnhub leads when its key is present -- its free tier is 60 calls/minute
    against NewsAPI's 100/day, and it resolves the entity itself. NewsAPI stays
    as a genuine fallback because Finnhub covers North American companies only.
    """
    attempts: list[str] = []
    for source_id in NEWS_PROVIDER_ORDER:
        if source_id == FINNHUB_SOURCE_ID:
            from core.finnhub_news import fetch_company_news

            result = fetch_company_news(ticker, as_of_dt, timeout=timeout)
        elif source_id == NEWS_SOURCE_ID:
            result = fetch_provider_articles(ticker, as_of_dt, timeout=timeout)
        else:  # pragma: no cover - the config validator forbids this
            continue

        status = result.get("status")
        if status == "ok":
            result["source_id"] = source_id
            result["attempts"] = attempts
            return result
        if status == "provider_key_required":
            attempts.append(f"{source_id}: not configured")
            continue
        attempts.append(f"{source_id}: {result.get('reason', 'failed')}")

    # EVERY provider is exhausted. The reason names each one, because "news is
    # unavailable" without saying which provider failed is unactionable.
    #
    # NOT CONFIGURED IS NOT FAILED. CAUGHT BY PROBE: the first version keyed on
    # whether `attempts` was non-empty, but a skipped provider appends an attempt
    # too -- so a machine with NO keys at all reported `provider_request_failed`,
    # which reads as "the provider refused us" and sends a reader looking for an
    # outage that never happened. The status now turns on whether anything was
    # actually ASKED.
    asked = any("not configured" not in note for note in attempts)
    return {
        "status": "provider_request_failed" if asked else "provider_key_required",
        "records": [],
        "reason": (
            "; ".join(attempts)
            if attempts
            else "no news provider is configured"
        ),
        "source_id": NEWS_SOURCE_ID,
        "attempts": attempts,
    }


def build_news_snapshot(ticker: str, as_of: str, timeout: float = NEWS_PROVIDER_TIMEOUT_SECONDS) -> dict:
    """Run the full N1 news pipeline for one ticker at one point in time."""
    ticker = str(ticker).upper()
    as_of_text = str(as_of)
    as_of_dt = _parse_timestamp(as_of_text)
    if as_of_dt is None:
        raise ValueError(f"news adapter: unparseable as_of {as_of_text!r}")

    resolution = resolve_news_provider(ticker, as_of_text)
    if resolution.get("status") != "live_provider":
        return _unavailable_snapshot(ticker, as_of_text)

    fetched = fetch_news_records(ticker, as_of_dt, timeout=timeout)
    # THE PROVIDER THAT ANSWERED, not a hardcoded one. Filing a Finnhub record
    # under the NewsAPI id would make the two indistinguishable in the very
    # store that exists to prove provenance.
    answering_source = str(fetched.get("source_id") or NEWS_SOURCE_ID)
    if fetched["status"] != "ok":
        return _unavailable_snapshot(
            ticker,
            as_of_text,
            reason=f"News provider unavailable: {fetched['reason']}",
            source_id=answering_source,
        )

    raw_records = fetched["records"]
    append_raw_records(
        source_id=answering_source,
        request_key=f"{ticker}_{as_of_dt.date().isoformat()}",
        records=raw_records,
    )

    if not raw_records:
        return _unavailable_snapshot(
            ticker,
            as_of_text,
            reason="Provider returned no articles inside the point-in-time window.",
            source_id=NEWS_SOURCE_ID,
        )

    eligible, rejected = pit_filter(raw_records, as_of_dt)
    enriched = enrich_articles(eligible, ticker, as_of_dt)

    credible = [article for article in enriched if article["included_in_aggregation"]]
    contradictions = detect_contradictions(credible)
    sentiment, confidence = aggregate_articles(credible)

    if rejected:
        status = "INVALID"
        reason_text = (
            "News provider payload violated the point-in-time policy "
            f"({len(rejected)} future-dated or unparseable publication times); rejected fail-closed."
        )
        sentiment = None
        confidence = 0.0
    elif contradictions:
        status = "CONTRADICTORY"
        reason_text = (
            f"{len(contradictions)} same-day, same-category contradiction cluster(s) exceed "
            "the tone tolerance; credible evidence disagrees, so no neutral average is published."
        )
        sentiment = None
        confidence = NEWS_CONTRADICTION_CONFIDENCE_FLOOR
    elif not credible:
        status = "INCOMPLETE"
        reason_text = "No credible, on-entity articles inside the point-in-time window; nothing to aggregate."
        sentiment = None
        confidence = 0.0
    else:
        status = "OK"
        reason_text = ""

    newest = max((article["published_dt"] for article in credible), default=None)
    published_time_text = None
    if newest is not None:
        published_time_text = next(
            article["published_time"] for article in credible if article["published_dt"] == newest
        )

    articles_public = [
        {
            "source_id": article["source_id"],
            "source_record_id": article["source_record_id"],
            "published_time": article["published_time"],
            "headline": article["headline"],
            "url": article["url"],
            "category": article["category"],
            "tone": article["tone"],
            "tone_derivation": article["tone_derivation"],
            "relevance": article["relevance"],
            "source_weight": article["source_weight"],
            "included_in_aggregation": article["included_in_aggregation"],
            "exclusion_reason": article["exclusion_reason"],
        }
        for article in enriched[:NEWS_MAX_ARTICLES]
    ]

    category_counts: dict[str, int] = {}
    for article in credible:
        category_counts[article["category"]] = category_counts.get(article["category"], 0) + 1

    pipeline = {
        "pipeline_version": NEWS_PIPELINE_VERSION,
        "classifier_version": NEWS_CLASSIFIER_VERSION,
        "tone_lexicon_version": NEWS_TONE_LEXICON_VERSION,
        "aggregator_version": NEWS_AGGREGATOR_VERSION,
        "provider": resolution,
        "counts": {
            "fetched": len(raw_records),
            "eligible": len(eligible),
            "rejected": len(rejected),
            "credible": len(credible),
            "contradictions": len(contradictions),
            "categories": dict(sorted(category_counts.items())),
        },
        "contradictions": contradictions,
        "rejected": rejected[:NEWS_MAX_ARTICLES],
    }

    snapshot = NewsSnapshot(
        ticker=ticker,
        as_of=as_of_text,
        status=status,
        source_id=NEWS_SOURCE_ID,
        source_confidence=float(confidence),
        published_time=published_time_text,
        calculation_version=NEWS_CONTRACT_VERSION,
        lookback_period=f"{NEWS_LOOKBACK_DAYS}d",
        sentiment_score=sentiment,
        articles=articles_public,
        reason=reason_text,
    ).to_dict()
    # Provenance beyond the legacy dataclass fields — never attached to the
    # UNAVAILABLE path, which stays byte-for-byte with the pre-N1 stub.
    snapshot["pipeline"] = pipeline
    return snapshot