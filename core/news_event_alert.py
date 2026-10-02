"""A9 news event alert — "AVGO: Earnings Results", graded by realized history.

The operator's examples were corporate news: *Urgent: AVGO - Earnings Results*,
*Very High: NVDA - New AI Partnership*. A1-A6 detect statistical state changes
(regime flips, thesis reversals, confidence moves) and none of them can produce
those, because none of them reads a headline. This module does.

**IT CLASSIFIES WITH THE EXISTING TAXONOMY, NOT A NEW ONE.** ``news_adapter``
already carries a 10-category classifier and a tone lexicon that the scoring
engine uses, and ``data/event_memory.jsonl`` holds 2,921 memories keyed on those
same categories (1,969 earnings, 776 other, 42 product_launch, 37 litigation, …).
A second classifier here would produce categories that match no stored history,
so every event would be unrateable by construction. W5: one canonical
implementation.

**THE PRIORITY COMES FROM REALIZED OUTCOMES, NEVER FROM THE HEADLINE.** This is
the trap A3 documented and it applies with full force here, because a headline is
the most tempting thing in the system to grade directly. "Beats expectations by
8%" reads urgent; whether that KIND of event has ever moved the price is a
different question, and only the second one is measurable. So:

    classify the headline  ->  look up that type's realized |move| in memory
                           ->  let A3 grade it HIGH / LOW / UNKNOWN impact

A type with too few remembered outcomes returns **UNKNOWN_IMPACT**, not a guess.
MEASURED in A3: a median |move| estimate at n=3 spans a five-fold range across
resamples, so `guidance` (3 memories) genuinely cannot be rated while `earnings`
(1,969) can.

**TONE IS DIRECTION, NOT IMPORTANCE.** The lexicon says whether the article reads
positive or negative; it says nothing about how far the price moves. Using tone to
set priority would rank a cheerful press release above a quiet regulatory filing.
Tone is reported in the alert body and is deliberately absent from the grading.

**AN EMPTY NEWS DAY IS NOT A QUIET DAY.** When the provider is UNAVAILABLE — which
MEASURED is every weekday since 2026-09-21, because ``NEWSAPI_KEY`` is not set for
the scheduler — this returns ``NOT_EVALUATED``, never ``NO_EVENT``. The operator
must be able to tell "nothing happened" from "we could not look".
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    EVENT_IMPACT_HIGH,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_MIN_MEDIAN_MOVE,
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    NEWS_EVENT_MIN_RELEVANCE,
)
from core.event_impact_alert import realized_impact
from core.news_adapter import classify_event, resolve_relevance, resolve_tone

LOGGER = logging.getLogger("news_event_alert")

NEWS_EVENT_ALERT_VERSION = "news-event-alert-v1"

# The verdict when the provider could not be read at all. NOT the same as
# NO_EVENT, and the distinction is the whole point: absence of evidence is not
# evidence that nothing happened.
NEWS_EVENT_NOT_EVALUATED = "NOT_EVALUATED"

# SNAPSHOT STATUSES THAT MEAN "DO NOT CONCLUDE ANYTHING", with the sentence each
# one contributes to the alert's reason. Enumerated POSITIVELY so a status nobody
# listed here is not silently treated as readable.
#
# `INCOMPLETE` is deliberately ABSENT: it means the provider was asked and
# carried no on-entity article, which is a real negative answer -- "nothing
# happened" -- and reporting it as unevaluable would hide a measurement.
_UNREADABLE_STATUSES: dict[str, str] = {
    "UNAVAILABLE": "the news provider was unavailable",
    "INVALID": (
        "the provider payload violated the point-in-time policy, so it was "
        "rejected fail-closed"
    ),
    # MEASURED LIVE on NVDA: the adapter publishes NO sentiment for this status,
    # because averaging sources that disagree beyond the tone tolerance
    # manufactures a neutral reading no source supports. Grading an event from
    # such a day asserts something the credible evidence disputes.
    "CONTRADICTORY": (
        "credible sources disagree beyond the tone tolerance, so no aggregate "
        "was published"
    ),
}

# Human titles for the taxonomy, so a subject line reads like the operator's
# examples ("Earnings Results") rather than like a database key ("earnings").
_TITLES: dict[str, str] = {
    "earnings": "Earnings Results",
    "guidance": "Guidance Update",
    "litigation": "Legal Action",
    "regulation": "Regulatory Development",
    "product_launch": "Product Launch",
    "macro_shock": "Macro Development",
    "m_and_a": "Merger or Acquisition",
    "strategic_announcement": "Strategic Announcement",
    "management_commentary": "Management Commentary",
    "other": "Company News",
}


class NewsEventAlertError(ValueError):
    """Raised when a news event cannot be assessed honestly."""


def title_for(event_type: str, *, headline: str = "") -> str:
    """The short, direct description the subject line needs.

    Prefers the taxonomy's human title. The headline is NOT used as the title:
    provider headlines run to 120 characters and arrive in the subject truncated
    mid-word, and they are untrusted text in a position a reader trusts.
    """
    return _TITLES.get(str(event_type), _TITLES["other"])


def _articles(snapshot: Mapping[str, Any]) -> list[dict]:
    """The articles the adapter judged ELIGIBLE, in the adapter's own terms.

    MEASURED on a live snapshot: the payload key is ``articles``, and each row
    carries the adapter's own ``included_in_aggregation`` flag plus an
    ``exclusion_reason`` ("duplicate_headline", "zero_relevance",
    "zero_source_weight"). The ``credible`` list is an internal local in
    ``build_news_snapshot`` and never reaches the payload, so reading it finds
    nothing and silently falls through to the UNFILTERED list -- which would
    treat articles the adapter explicitly excluded as eligible evidence.
    """
    value = snapshot.get("articles")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping) and item.get("included_in_aggregation")
    ]


def leading_event(
    snapshot: Mapping[str, Any] | None,
    ticker: str,
    *,
    min_relevance: float = NEWS_EVENT_MIN_RELEVANCE,
) -> dict:
    """The one article that best represents today's news for this ticker.

    Returns ``{"found": bool, ...}``. Selection is by RELEVANCE, not by tone or
    recency: the most strongly-worded article is not the most important one, and
    the newest is often a syndicated copy of an earlier story.
    """
    if snapshot is None:
        return {
            "found": False,
            "readable": False,
            "reason": "no news snapshot was supplied",
        }
    if not isinstance(snapshot, Mapping):
        raise NewsEventAlertError("a news snapshot must be a mapping")

    # EVERY STATUS IS HANDLED EXPLICITLY, and the three that mean "do not
    # conclude anything" are distinguished from the one that means "nothing
    # happened". CAUGHT LIVE: NVDA returned CONTRADICTORY and the first version
    # graded an event from it, because only UNAVAILABLE was checked.
    status = str(snapshot.get("status") or "").upper()
    if status in _UNREADABLE_STATUSES:
        return {
            "found": False,
            "readable": False,
            "status": status,
            "reason": (
                f"{_UNREADABLE_STATUSES[status]} "
                f"({snapshot.get('reason') or 'no reason given'}); nothing can "
                f"be concluded about whether an event occurred"
            ),
        }

    articles = _articles(snapshot)
    if not articles:
        # READABLE but empty. A real answer: the provider responded and carried
        # no qualifying article, which is different from not being able to ask.
        return {
            "found": False,
            "readable": True,
            "reason": "the provider returned no point-in-time eligible article",
        }

    # THE ADAPTER'S OWN RELEVANCE IS USED, never recomputed. It stamps
    # `relevance` on every enriched article; recomputing it here from `ticker`
    # and `company_name` reads fields the enriched record does not carry, so it
    # returned 0.0 for every article and the floor was never really applied.
    # W5: one canonical implementation of one measurement.
    scored: list[tuple[float, dict]] = []
    for article in articles:
        relevance = article.get("relevance")
        if relevance is None:
            # Not stamped: fall back to the canonical resolver rather than
            # assuming either extreme.
            try:
                relevance = resolve_relevance(article, ticker)
            except Exception as exc:  # a malformed article must not lose the day
                LOGGER.warning("relevance failed for an article: %s", exc)
                continue
        try:
            relevance = float(relevance)
        except (TypeError, ValueError):
            continue
        if relevance < float(min_relevance):
            continue
        scored.append((relevance, article))

    if not scored:
        return {
            "found": False,
            "readable": True,
            "reason": (
                f"no article reached the {min_relevance} relevance floor for "
                f"{ticker}; a low-relevance headline is market noise rather "
                f"than an event about this holding"
            ),
        }

    relevance, article = max(scored, key=lambda pair: pair[0])
    headline = str(article.get("headline") or article.get("title") or "")
    body = str(article.get("summary") or article.get("description") or "")

    # THE ADAPTER'S OWN CATEGORY AND TONE, for the same W5 reason. It classifies
    # over the full text it fetched, which is more than the enriched record
    # keeps; reclassifying from the headline alone loses that context.
    event_type = article.get("category") or classify_event(f"{headline} {body}")
    tone = article.get("tone")
    derivation = article.get("tone_derivation")
    if tone is None:
        tone, derivation = resolve_tone(article)

    return {
        "found": True,
        "readable": True,
        "event_type": event_type,
        "headline": headline,
        "relevance": round(relevance, 4),
        "tone": tone,
        "tone_derivation": derivation,
        "published_time": article.get("published_time") or article.get("publishedAt"),
        "source": (
            article.get("source")
            or article.get("source_id")
            or article.get("source_name")
        ),
        "url": article.get("url"),
        "candidates": len(scored),
    }


def news_event_alert(
    snapshot: Mapping[str, Any] | None,
    ticker: str,
    *,
    memories: Iterable[Any] | None = None,
    as_of: str | None = None,
    horizon: str = "20d",
) -> dict:
    """Did a material news event occur, and how far does its KIND usually move?

    The verdict is A3's: HIGH_IMPACT / LOW_IMPACT / UNKNOWN_IMPACT / NO_EVENT, or
    NOT_EVALUATED when the provider could not be read. Severity follows the
    verdict; the headline never sets it.
    """
    if not str(ticker or "").strip():
        raise NewsEventAlertError("a ticker is required")

    found = leading_event(snapshot, ticker)

    base: dict[str, Any] = {
        "version": NEWS_EVENT_ALERT_VERSION,
        "alert": "news_event",
        "ticker": str(ticker).upper(),
        "as_of": as_of,
        "horizon": horizon,
        "blocks_trades": False,
    }

    # 1. COULD NOT LOOK. Reported as NOT_EVALUATED so it is never read as calm.
    if not found.get("readable"):
        return {
            **base,
            "verdict": NEWS_EVENT_NOT_EVALUATED,
            "severity": None,
            "title": "News Unavailable",
            "event_type": None,
            "reason": found.get("reason", "the news provider could not be read"),
            "measured": False,
            "detail": found,
        }

    # 2. LOOKED, NOTHING QUALIFIED. A real negative answer.
    if not found.get("found"):
        return {
            **base,
            "verdict": EVENT_IMPACT_NO_EVENT,
            "severity": None,
            "title": "No Qualifying Event",
            "event_type": None,
            "reason": found.get("reason", "no qualifying article"),
            "measured": True,
            "detail": found,
        }

    event_type = str(found["event_type"])

    # 3. GRADE THE TYPE BY ITS REMEMBERED OUTCOMES, not by the headline.
    impact = realized_impact(memories or [], event_type, horizon=horizon)

    if not impact.get("measured"):
        # UNRATEABLE IS A VERDICT. MEASURED in A3, a median |move| at n=3 spans a
        # five-fold range across resamples, so a type with too little history
        # genuinely cannot be sized -- and reporting it as low-impact would read
        # as reassurance the evidence does not support.
        return {
            **base,
            "verdict": EVENT_IMPACT_UNKNOWN,
            "severity": ALERT_SEVERITY_INFO,
            "title": title_for(event_type),
            "event_type": event_type,
            "reason": (
                f"a {event_type} event was detected for {ticker}, but "
                f"{impact.get('reason')}"
            ),
            "measured": False,
            "median_abs_move": None,
            "analogs": impact.get("analogs"),
            "tone": found.get("tone"),
            "headline": found.get("headline"),
            "detail": {**found, "impact": impact},
        }

    median = float(impact["median_move"])
    high = median >= float(EVENT_IMPACT_MIN_MEDIAN_MOVE)

    return {
        **base,
        "verdict": EVENT_IMPACT_HIGH if high else EVENT_IMPACT_LOW,
        # WARN only for a high-impact type. Tone does not enter this: a cheerful
        # press release and a grim one about the same event type move the price
        # by the same measured amount.
        "severity": ALERT_SEVERITY_WARN if high else ALERT_SEVERITY_INFO,
        "title": title_for(event_type),
        "event_type": event_type,
        "reason": (
            f"a {event_type} event was detected for {ticker}; across "
            f"{impact['analogs']} remembered outcomes this type's median "
            f"absolute {horizon} move is {median:.4f}"
            + (
                f", at or above the {EVENT_IMPACT_MIN_MEDIAN_MOVE} "
                f"high-impact threshold"
                if high
                else f", below the {EVENT_IMPACT_MIN_MEDIAN_MOVE} "
                f"high-impact threshold"
            )
        ),
        "measured": True,
        "median_abs_move": median,
        "p90_move": impact.get("p90_move"),
        "analogs": impact.get("analogs"),
        "tone": found.get("tone"),
        "tone_derivation": found.get("tone_derivation"),
        "headline": found.get("headline"),
        "detail": {**found, "impact": impact},
    }


def summary_for(alert: Mapping[str, Any]) -> dict:
    """The Who / What / When / Why block the operator's email template needs.

    Built from the alert's own measured fields. Nothing here is invented: when a
    fact was not measured the section says so rather than filling the gap.
    """
    if not isinstance(alert, Mapping):
        raise NewsEventAlertError("an alert must be a mapping")

    ticker = alert.get("ticker") or "the portfolio"
    raw_detail = alert.get("detail")
    detail: Mapping[str, Any] = (
        raw_detail if isinstance(raw_detail, Mapping) else {}
    )
    # The adapter stamps `source_id`; `source` is the provider-payload spelling.
    # Read both, because `leading_event` carries whichever the article had.
    source = detail.get("source") or detail.get("source_id")
    headline = alert.get("headline") or detail.get("headline")

    who = f"{ticker}"
    if source:
        who += f" — reported by {source}"

    event_type = alert.get("event_type")
    what = (
        f"{title_for(str(event_type))}: {headline}"
        if headline and event_type
        else str(alert.get("reason") or "A news event was detected.")
    )

    published = detail.get("published_time")
    when = (
        f"Published {published}; assessed for {alert.get('as_of') or 'today'}."
        if published
        else f"Assessed for {alert.get('as_of') or 'today'}."
    )

    why: list[str] = []
    if alert.get("measured") and alert.get("median_abs_move") is not None:
        why.append(
            f"This event type's median absolute {alert.get('horizon')} move is "
            f"{float(alert['median_abs_move']):.2%} across "
            f"{alert.get('analogs')} remembered outcomes."
        )
        if alert.get("p90_move") is not None:
            why.append(
                f"The 90th percentile move is "
                f"{float(alert['p90_move']):.2%}."
            )
    elif alert.get("verdict") == EVENT_IMPACT_UNKNOWN:
        why.append(
            f"Only {alert.get('analogs')} comparable outcome(s) are remembered, "
            f"too few to size this event type's typical move."
        )

    tone = alert.get("tone")
    if tone is not None:
        direction = "positive" if float(tone) > 0 else (
            "negative" if float(tone) < 0 else "neutral"
        )
        why.append(
            f"The article reads {direction} (tone {float(tone):+.2f}), which "
            f"indicates direction rather than size."
        )

    if not why:
        why.append(str(alert.get("reason") or "No measurement is available."))

    return {"who": who, "what": what, "when": when, "why": why}


def render_news_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable line, plus the reason — the house render convention."""
    return [
        f"  {str(alert.get('ticker') or '?'):8s} "
        f"{str(alert.get('event_type') or '—'):22s} "
        f"{str(alert.get('verdict')):16s} "
        f"[{alert.get('severity') or '—'}]",
        f"      {alert.get('reason')}",
    ]
