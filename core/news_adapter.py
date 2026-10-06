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
from datetime import datetime, timedelta, timezone

from core.config import (
    NEWS_AGGREGATOR_VERSION,
    NEWS_BASE_SOURCE_CONFIDENCE,
    NEWS_CLASSIFIER_VERSION,
    NEWS_CONTRADICTION_CONFIDENCE_FLOOR,
    NEWS_CONTRADICTION_TONE_DELTA,
    NEWS_CONTRACT_VERSION,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_ARTICLES,
    NEWS_PIPELINE_VERSION,
    NEWS_PROVIDER_API_KEY_ENV,
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

# --- Why the news was unavailable (the FAILURE TAXONOMY) ------------------------
#
# Callers used to tell a quota day from an outage by searching the reason
# TEXT for "429". MEASURED, that is wrong in both directions:
#
#   * urllib raises HTTPError for a 429, whose str() is "HTTP Error 429:
#     Too Many Requests" -- so the substring happened to work.
#   * NewsAPI ALSO signals its rolling-window limit as HTTP 200 with a JSON
#     body {"status":"error","code":"rateLimited"}, whose message contains no
#     "429" at all. That day was classified as a generic provider failure,
#     alerted as an outage, and sent the operator to check a network that
#     was fine.
#   * A ticker named "429 Inc" in a reason string would read as a quota day.
#
# So the KIND travels as its own field, decided once here where the HTTP
# status and the body are both in scope, and every downstream consumer
# switches on it instead of re-deriving it from prose.
FAILURE_NONE = ""
FAILURE_QUOTA = "quota_exceeded"
FAILURE_AUTH = "authentication_failed"
FAILURE_UNAVAILABLE = "provider_unavailable"
FAILURE_NO_KEY = "provider_key_required"
FAILURE_UNKNOWN = "unknown_error"

# The operator-facing sentence for each kind: what happened, and what to do.
# A notification the reader cannot act on is how a channel gets muted, so
# every kind names an action -- including "none, this clears itself".
FAILURE_GUIDANCE: dict[str, str] = {
    FAILURE_QUOTA: (
        "the news provider's request quota is spent. NewsAPI's free tier "
        "allows 100 requests/24h, enforced as 50/12h on a rolling window, so "
        "this clears on its own without intervention. The ticker keeps its "
        "place in the rotation and is retried on the next run."
    ),
    FAILURE_AUTH: (
        "the news provider rejected the API key. Check that "
        "NEWS_PROVIDER_API_KEY is set to a currently valid key -- this does "
        "NOT clear on its own, and no news is captured until it is fixed."
    ),
    FAILURE_UNAVAILABLE: (
        "the news provider could not be reached (network or provider-side "
        "outage). The key is not implicated. This usually clears on its own; "
        "if it persists for more than a day, check connectivity."
    ),
    FAILURE_NO_KEY: (
        "no news provider is configured. Until NEWS_PROVIDER_API_KEY is set, "
        "no OBSERVED event memory can exist and the model runs on price-"
        "derived features alone."
    ),
    FAILURE_UNKNOWN: (
        "the news provider failed for a reason this code does not recognise. "
        "The provider's own message is quoted above; it is reported verbatim "
        "rather than guessed at."
    ),
}

# NewsAPI's documented error codes, mapped to our kinds. Matched on the
# machine-readable `code` field, never on the human-readable `message`,
# because the message is prose the provider may reword at any time.
_PROVIDER_ERROR_CODES: dict[str, str] = {
    "ratelimited": FAILURE_QUOTA,
    "maximumresultsreached": FAILURE_QUOTA,
    "apikeydisabled": FAILURE_AUTH,
    "apikeyexhausted": FAILURE_QUOTA,
    "apikeyinvalid": FAILURE_AUTH,
    "apikeymissing": FAILURE_NO_KEY,
    "unauthorized": FAILURE_AUTH,
}


def classify_http_status(status: int) -> str:
    """Map an HTTP status onto a failure kind.

    429 is the quota signal. 401/403 are credential problems, which must be
    distinguished because one clears itself and the other never does.
    """
    if status == 429:
        return FAILURE_QUOTA
    if status in (401, 403):
        return FAILURE_AUTH
    if status >= 500 or status == 408:
        return FAILURE_UNAVAILABLE
    return FAILURE_UNKNOWN


def classify_provider_error(code: str, message: str = "") -> str:
    """Map a provider error BODY onto a failure kind.

    `code` is matched first because it is machine-readable. The message is
    consulted only as a last resort, for providers that omit a code -- and
    the substrings chosen are ones that cannot plausibly appear in an
    unrelated error.
    """
    kind = _PROVIDER_ERROR_CODES.get(str(code or "").strip().lower())
    if kind:
        return kind
    text = str(message or "").lower()
    if "too many requests" in text or "rate limit" in text or "quota" in text:
        return FAILURE_QUOTA
    if "api key" in text or "apikey" in text or "unauthorized" in text:
        return FAILURE_AUTH
    return FAILURE_UNKNOWN

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


def resolve_relevance(record: dict, ticker: str) -> float:
    """ENTITY RESOLUTION v1: how clearly the article is about `ticker`.

    1.0 when the ticker matches (provider-matched or present as a token in
    the text); 0.7 for a company-name token match; 0.0 otherwise. Relevance
    0 keeps the article in the evidence but excludes it from aggregation.
    """
    provider_ticker = str(record.get("ticker") or "").strip().upper()
    if provider_ticker == ticker:
        return 1.0
    tokens = set(re.findall(
        r"[A-Za-z0-9]+",
        f"{record.get('headline', '')} {record.get('summary', '')}".upper(),
    ))
    if ticker in tokens:
        return 1.0
    name = str(record.get("company_name") or "").strip().upper()
    if name:
        name_tokens = {token for token in re.findall(r"[A-Za-z0-9]+", name) if len(token) > 1}
        if name_tokens and name_tokens.issubset(tokens):
            return 0.7
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
    api_key = os.getenv(NEWS_PROVIDER_API_KEY_ENV)
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
            "failure_kind": FAILURE_NO_KEY,
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
    except urllib.error.HTTPError as exc:
        # HTTPError carries the STATUS CODE, which is the only trustworthy
        # quota signal. It is caught before URLError (its own base class) so
        # the code is read rather than thrown away into a string.
        kind = classify_http_status(int(getattr(exc, "code", 0) or 0))
        body_code = ""
        try:  # the body often names the reason more precisely than the status
            body = json.loads(exc.read().decode("utf-8", "replace"))
            body_code = str(body.get("code") or "")
            if kind in (FAILURE_UNKNOWN, FAILURE_UNAVAILABLE) and body_code:
                kind = classify_provider_error(body_code, body.get("message", ""))
        except Exception:  # noqa: BLE001 - an unreadable body is not fatal
            pass
        LOGGER.warning(
            "news_provider_http_error: status=%s kind=%s code=%s",
            getattr(exc, "code", "?"), kind, body_code or "-",
        )
        return {
            "status": "provider_request_failed",
            "records": [],
            "reason": f"news provider returned HTTP {getattr(exc, 'code', '?')}: {exc.reason}",
            "failure_kind": kind,
            "http_status": int(getattr(exc, "code", 0) or 0),
        }
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError, ValueError) as exc:
        # No HTTP status reached us at all: DNS, TLS, timeout, socket. That is
        # an outage, never a quota -- the request never arrived to be counted.
        LOGGER.warning("news_provider_request_failed: %s", exc)
        return {
            "status": "provider_request_failed",
            "records": [],
            "reason": f"news provider request failed: {exc}",
            "failure_kind": FAILURE_UNAVAILABLE,
        }
    if str(payload.get("status", "ok")).lower() != "ok":
        # THE PATH THAT USED TO BE INVISIBLE. NewsAPI signals its rolling
        # limit as HTTP 200 with status=error in the body, so this branch
        # handled a quota day while reporting a generic failure.
        code = str(payload.get("code") or "")
        message = str(payload.get("message", "unknown error"))
        kind = classify_provider_error(code, message)
        reason = f"provider rejected the request: {message}"
        LOGGER.warning("news_provider_rejected_request: kind=%s code=%s %s", kind, code or "-", reason)
        return {
            "status": "provider_request_failed",
            "records": [],
            "reason": reason,
            "failure_kind": kind,
            "provider_code": code,
        }
    return {
        "status": "ok",
        "records": _normalize_provider_payload(payload),
        "reason": "",
        "failure_kind": FAILURE_NONE,
    }


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

    fetched = fetch_provider_articles(ticker, as_of_dt, timeout=timeout)
    if fetched["status"] != "ok":
        # `failure_kind` rides along on the LIVE-provider failure path only.
        # The no-key stub above is pinned byte-for-byte by contract test, so
        # it must never grow a field; a caller that sees no key already knows
        # the kind without being told.
        snapshot = _unavailable_snapshot(
            ticker,
            as_of_text,
            reason=f"News provider unavailable: {fetched['reason']}",
            source_id=NEWS_SOURCE_ID,
        )
        snapshot["failure_kind"] = fetched.get("failure_kind") or FAILURE_UNKNOWN
        return snapshot

    raw_records = fetched["records"]
    append_raw_records(
        source_id=NEWS_SOURCE_ID,
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
    # STEP 2: resolve each article to a ticker AT INGESTION, using E2's
    # resolver rather than this module's v1 relevance heuristic.
    #
    # WHY E2 AND NOT resolve_relevance(). MEASURED 2026-10-04, the v1
    # heuristic uppercases the text before looking for the ticker, which
    # destroys the only signal separating a symbol from a word. All five of
    # these scored relevance 1.0 - MAXIMUM confidence - on a false match:
    #
    #     ARM  "1968 Corvette with a new ARM rest"
    #     CAT  "Red Cat Holdings announces drone contract"
    #     KO   "Why KO is a dividend stalwart"
    #     BE   "This BE the way pirates talk"
    #     ALL  "ALL of the above applies to markets"
    #
    # E2 rejects CAT, KO and BE (short tickers must match by name) and is
    # case-sensitive on tickers, which is the rule v1 throws away. It is
    # also the resolver whose verdict already reaches Event.entity_
    # resolution, so using it here means ingestion and the event contract
    # agree about what an article is about instead of holding two opinions.
    #
    # The RESOLUTION IS RECORDED, not applied as a filter: an article that
    # fails to resolve keeps its place in the evidence with method="none",
    # because the rejection rate is itself data about coverage (E2's rule).
    from core.entity_resolution import resolve_article

    resolutions = {}
    for record, _published_dt in eligible:
        record_id = str(record.get("source_record_id", ""))
        try:
            resolutions[record_id] = resolve_article(record, ticker)
        except Exception:  # noqa: BLE001 - resolution must never break ingestion
            resolutions[record_id] = None

    enriched: list[dict] = []
    for record, published_dt in eligible:
        tone, derivation = resolve_tone(record)
        resolution = resolutions.get(str(record.get("source_record_id", "")))
        enriched.append({
            "resolved_ticker": (
                ticker if resolution is not None and resolution.matched else ""
            ),
            "entity_resolution_method": (
                str(resolution.method) if resolution is not None else ""
            ),
            "entity_resolution_confidence": (
                float(resolution.confidence) if resolution is not None else 0.0
            ),
            "source_id": NEWS_SOURCE_ID,
            # THE OUTLET, carried through enrichment. MEASURED 2026-10-04,
            # this dict rebuilt the article field by field and omitted
            # `source_name`, so the outlet captured by
            # _normalize_provider_payload died here — on all 1,637 articles
            # of the first news day, across 369 distinct outlets. Every
            # downstream Event then recorded "newsapi_news" as its source
            # and L4 had one source with no variation to learn from.
            "source_name": str(record.get("source_name") or ""),
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
            # THE OUTLET. This is the dict that reaches
            # events_from_news_snapshot, so it is the boundary that
            # actually decides whether an Event can name its reporter.
            # MEASURED 2026-10-04: the article was rebuilt field by field
            # TWICE on the way out (here and in `enriched`), and both
            # copies omitted the outlet.
            "source_name": article.get("source_name", ""),
            # STEP 2: the resolved ticker, and HOW it resolved. Carried so
            # an outcome can be joined back to the outlet that reported it
            # (open item 4). MEASURED 2026-09-21 and still true at 1,637
            # articles on 2026-10-04: 0 articles carried a ticker, so no
            # article joined to a price outcome and L4 could never measure
            # an outlet's record.
            "ticker": article.get("resolved_ticker", ""),
            "entity_resolution_method": article.get("entity_resolution_method", ""),
            "entity_resolution_confidence": article.get(
                "entity_resolution_confidence", 0.0
            ),
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