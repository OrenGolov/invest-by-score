"""B2 source → outcome join — attributing a price outcome to an OUTLET.

Open item 4 parked this as unbuildable: *"0 of 2650 stored articles carry a
ticker, so no article joins to a price outcome."*

**That reading was incomplete, and re-measuring changed the answer.** The article
records do carry a `ticker` field and it is indeed always `None` — but every raw
news envelope carries a `request_key` of the form `MSFT_2026-09-17`, naming the
ticker the fetch was made FOR. Measured across the tracked raw store:

    articles total                3,753
    with an article-level ticker      0
    with a request_key ticker     3,753
    distinct outlets                 58

So the join is buildable from data already tracked, with no re-ingestion. The
ticker is *inferred from the request*, which is weaker than an entity resolution
performed on the article text, and that weakness is recorded on every observation
rather than hidden: `attribution` is `"request_key"`, not `"entity_resolution"`.

**What this module does NOT do.** It does not claim the article CAUSED the move.
It reports the forward return that followed publication, which is an
event-associated return — the E5 distinction, kept deliberately. A "hit" means the
outlet's stated tone agreed with the direction the price then took, over a declared
horizon. That is a measurable property of an outlet's claims; it is not causation,
and the word `hit` is defined here so nothing downstream has to guess.

**Tone is required.** An article with no directional claim cannot be right or
wrong about direction, so it is DROPPED rather than counted as a miss — counting
it would penalise an outlet for the system's own missing classification, the same
rule `observation_records` already applies to a missing outlet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.config import (
    LABEL_HORIZON_SESSIONS,
    SOURCE_OUTCOME_ATTRIBUTION_REQUEST_KEY,
    SOURCE_OUTCOME_MIN_TONE_ABS,
    SOURCE_OUTCOME_VERSION,
)


class SourceOutcomeJoinError(ValueError):
    """Raised when a join request is structurally invalid."""


# How a ticker was attributed to an article. Recorded on every observation,
# because the two are not equally trustworthy and a consumer must be able to tell.
ATTRIBUTION_REQUEST_KEY = "request_key"        # inferred from the fetch
ATTRIBUTION_ENTITY_RESOLUTION = "entity_resolution"  # resolved from the text
ATTRIBUTIONS: tuple[str, ...] = (
    ATTRIBUTION_REQUEST_KEY,
    ATTRIBUTION_ENTITY_RESOLUTION,
)

# Why an article produced no observation. Counted and reported, so the size of
# the join's own losses is visible rather than inferred from a row count.
DROPPED_NO_OUTLET = "no_outlet"
DROPPED_NO_TICKER = "no_ticker"
DROPPED_NO_TONE = "no_directional_claim"
DROPPED_NO_PRICE = "no_price_outcome"
DROPPED_NO_PUBLISHED_TIME = "no_published_time"


def ticker_from_request_key(request_key: Any) -> str | None:
    """The ticker a fetch was made for, e.g. `MSFT_2026-09-17` -> `MSFT`.

    None when the key does not carry one. A key whose prefix is not a plausible
    ticker yields None rather than a guess: a wrong attribution is worse than a
    missing one, because it would credit an outlet for a move in another company.
    """
    text = str(request_key or "").strip()
    if not text or "_" not in text:
        return None
    candidate = text.split("_", 1)[0].strip().upper()
    if not candidate or not candidate.isalpha() or len(candidate) > 6:
        return None
    return candidate


def load_news_envelopes(directory: str | Path) -> list[dict[str, Any]]:
    """Read the raw news envelopes. Malformed lines raise — integrity is loud."""
    root = Path(directory)
    if not root.exists():
        return []
    envelopes: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw = line.strip()
            if not raw:
                continue
            try:
                envelopes.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise SourceOutcomeJoinError(
                    f"{path.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return envelopes


def _tone_direction(article: Mapping[str, Any]) -> tuple[int | None, str]:
    """(+1 / -1 / None, derivation) for an article's directional claim.

    A tone inside the neutral band is NOT a claim about direction, so it cannot
    be scored. Returning 0 would make every neutral article a miss against any
    non-zero move.

    **THE STORED TONE IS USUALLY ABSENT.** MEASURED, it is None for all 3,753
    stored articles: the raw store keeps the headline and no classified
    direction. `resolve_tone` derives one from the text, and its DERIVATION is
    returned so a consumer can tell a provider-supplied tone from a
    lexicon-derived one — they are not equally trustworthy, and an outlet's
    measured hit rate partly measures whichever produced it.
    """
    tone = (article or {}).get("tone")
    derivation = str((article or {}).get("tone_derivation") or "") or "stored"
    if tone is None:
        # Derive it rather than dropping the article: the classification is what
        # is missing, not the evidence.
        try:
            from core.news_adapter import resolve_tone

            tone, derivation = resolve_tone(
                {
                    "headline": (article or {}).get("headline"),
                    "summary": (article or {}).get("summary"),
                }
            )
        except Exception:
            return None, "unavailable"
    if tone is None:
        return None, derivation
    try:
        value = float(tone)
    except (TypeError, ValueError):
        return None, derivation
    if abs(value) < SOURCE_OUTCOME_MIN_TONE_ABS:
        return None, derivation
    return (1 if value > 0 else -1), derivation


def build_observations(
    envelopes: Sequence[Mapping[str, Any]] | None,
    forward_return: Any,
    *,
    horizon: str = "5d",
    sector_of: Any = None,
) -> dict[str, Any]:
    """Join articles to price outcomes, one observation per scorable article.

    `forward_return(ticker, published_time, horizon)` returns the return that
    FOLLOWED publication, or None when it cannot be computed. Supplied by the
    caller rather than imported, so this module owns the JOIN and not the price
    machinery (W5 — the event-study sprints own attribution).

    Every drop is counted by reason. A join that silently loses rows cannot be
    told from one that had nothing to join.
    """
    if horizon not in LABEL_HORIZON_SESSIONS:
        raise SourceOutcomeJoinError(
            f"{horizon!r} is not a declared label horizon "
            f"(known: {sorted(LABEL_HORIZON_SESSIONS)})"
        )
    if not callable(forward_return):
        raise SourceOutcomeJoinError("forward_return must be callable")

    observations: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}
    articles = 0

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for envelope in envelopes or []:
        ticker = ticker_from_request_key((envelope or {}).get("request_key"))
        for article in (envelope or {}).get("records") or []:
            articles += 1
            outlet = str((article or {}).get("source_name") or "").strip()
            if not outlet:
                drop(DROPPED_NO_OUTLET)
                continue
            # An article-level ticker is preferred when present, because it was
            # resolved from the text rather than inferred from the request.
            resolved = str((article or {}).get("ticker") or "").strip().upper()
            attribution = (
                ATTRIBUTION_ENTITY_RESOLUTION if resolved else ATTRIBUTION_REQUEST_KEY
            )
            subject = resolved or ticker
            if not subject:
                drop(DROPPED_NO_TICKER)
                continue
            published = str((article or {}).get("published_time") or "").strip()
            if not published:
                drop(DROPPED_NO_PUBLISHED_TIME)
                continue
            direction, tone_derivation = _tone_direction(article)
            if direction is None:
                drop(DROPPED_NO_TONE)
                continue
            outcome = forward_return(subject, published, horizon)
            if outcome is None:
                drop(DROPPED_NO_PRICE)
                continue
            realised = float(outcome)
            observations.append(
                {
                    "source": outlet,
                    # THE DEFINITION OF A HIT: the outlet's directional claim
                    # agreed with the direction the price then took. An
                    # event-ASSOCIATED agreement, never a causal one.
                    "hit": 1 if (realised > 0) == (direction > 0) else 0,
                    "ticker": subject,
                    "published_time": published,
                    "horizon": horizon,
                    "forward_return": round(realised, 8),
                    "tone_direction": direction,
                    # WHICH CLASSIFIER produced the claim. A lexicon tone and a
                    # provider tone are not equally trustworthy, and MEASURED the
                    # v1 lexicon agreed with the obvious reading on only 5 of 6
                    # hand probes.
                    "tone_derivation": tone_derivation,
                    "attribution": attribution,
                    "sector": (
                        sector_of(subject) if callable(sector_of) else None
                    ),
                    "event_type": str((article or {}).get("event_type") or "") or None,
                }
            )

    return {
        "version": SOURCE_OUTCOME_VERSION,
        "observations": observations,
        "articles": articles,
        "joined": len(observations),
        "dropped": dict(sorted(dropped.items())),
        "distinct_sources": len({o["source"] for o in observations}),
        "horizon": horizon,
        # The honest headline: what fraction of articles produced evidence.
        "join_rate": round(len(observations) / articles, 6) if articles else None,
        "attribution_mix": {
            name: sum(1 for o in observations if o["attribution"] == name)
            for name in ATTRIBUTIONS
        },
        # How the directional claims were obtained. A join resting entirely on a
        # derived tone is measuring the classifier as much as the outlets, and
        # that must be readable off the report.
        "tone_derivation_mix": {
            derivation: sum(
                1 for o in observations if o["tone_derivation"] == derivation
            )
            for derivation in sorted({o["tone_derivation"] for o in observations})
        },
    }


def join_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a join report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    observations = report.get("observations")
    if not isinstance(observations, Sequence):
        return ["the report carries no observations list"]

    articles = report.get("articles")
    joined = report.get("joined")
    if articles is not None and joined is not None and joined > articles:
        problems.append(
            f"{joined} observations from {articles} articles; a join cannot "
            f"produce more evidence than it consumed"
        )
    if articles and not report.get("dropped") and joined != articles:
        problems.append(
            "articles were lost without a recorded reason; a join that silently "
            "drops rows cannot be told from one that had nothing to join"
        )
    for observation in observations:
        if not isinstance(observation, Mapping):
            problems.append("an observation is not a mapping")
            continue
        if not str(observation.get("source") or "").strip():
            problems.append("an observation carries no outlet")
        if observation.get("hit") not in (0, 1):
            problems.append(
                f"{observation.get('source')}: hit is {observation.get('hit')!r}, "
                f"which is neither agreement nor disagreement"
            )
        if observation.get("attribution") not in ATTRIBUTIONS:
            problems.append(
                f"{observation.get('source')}: attribution "
                f"{observation.get('attribution')!r} is not declared — a "
                f"consumer cannot tell an inferred ticker from a resolved one"
            )
        if observation.get("forward_return") is None:
            problems.append(
                f"{observation.get('source')}: no forward return, so the hit "
                f"was scored against nothing"
            )
    return problems


def render_join(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines. Absent facts render as ABSENT, never as zero."""
    lines = [
        f"  articles consumed  {report.get('articles', 0)}",
        f"  observations       {report.get('joined', 0)}",
    ]
    rate = report.get("join_rate")
    lines.append(
        f"  join rate          {'ABSENT' if rate is None else f'{rate:.1%}'}"
    )
    lines.append(f"  distinct outlets   {report.get('distinct_sources', 0)}")
    mix = report.get("attribution_mix") or {}
    if mix:
        lines.append(
            "  attribution        "
            + ", ".join(f"{name}={count}" for name, count in sorted(mix.items()))
        )
    for reason, count in sorted((report.get("dropped") or {}).items()):
        lines.append(f"    dropped {reason:24s} {count}")
    return lines
