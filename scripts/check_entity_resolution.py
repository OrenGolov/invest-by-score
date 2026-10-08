"""CI drift gate for E2 entity resolution.

Proves, on every push, that bad resolution cannot silently enter training:

1. the curated registry is valid — no shared aliases, no empty names;
2. every match method resolves as expected, strongest to weakest;
3. a FAILED resolution is distinguishable from a correct exclusion (the
   exact defect E2 fixed);
4. ambiguity resolves to `ambiguous` with zero confidence, never to a guess;
5. short tickers ("V", "BE") do not match as bare words;
6. phrase matching respects word boundaries — no substring or scattered-word
   matches;
7. events carry their resolution, and strict mode drops unresolved ones.

Synthetic only; runs in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ENTITY_MATCH_ALIAS,
    ENTITY_MATCH_AMBIGUOUS,
    ENTITY_MATCH_EXECUTIVE,
    ENTITY_MATCH_LEGAL_NAME,
    ENTITY_MATCH_NONE,
    ENTITY_MATCH_TICKER,
    ENTITY_MIN_TRAINING_CONFIDENCE,
)
from core.entity_resolution import (  # noqa: E402
    EntityRecord,
    build_default_entity_registry,
    registry_problems,
    resolution_report,
    resolve_entity,
)
from core.event_contract import events_from_news_snapshot  # noqa: E402

_SNAPSHOT = {
    "ticker": "NVDA", "status": "OK", "source_id": "newsapi_news",
    "articles": [
        {"source_record_id": "r1", "published_time": "2026-01-05 10:00:00",
         "category": "earnings", "tone": 0.7, "relevance": 0.9,
         "source_weight": 0.8, "headline": "Nvidia beats earnings"},
        {"source_record_id": "r2", "published_time": "2026-01-05 11:00:00",
         "category": "other", "tone": 0.1, "relevance": 0.2,
         "source_weight": 0.5, "headline": "The company will expand"},
    ],
}


def main() -> int:
    failures: list[str] = []

    # 1. Registry validity, and that the validator itself works.
    problems = registry_problems(build_default_entity_registry())
    if problems:
        failures.append(f"the default entity registry is invalid: {problems[:3]}")
    colliding = {
        "AAA": EntityRecord("AAA", "Alpha Inc.", ("Apex",)),
        "BBB": EntityRecord("BBB", "Beta Inc.", ("Apex",)),
    }
    if not registry_problems(colliding):
        failures.append(
            "a shared alias was accepted — every mention of it would be ambiguous"
        )

    # 2. Match methods, strongest to weakest.
    for text, ticker, expected in (
        ("NVDA beats earnings", "NVDA", ENTITY_MATCH_TICKER),
        ("NVIDIA Corporation reports", "NVDA", ENTITY_MATCH_LEGAL_NAME),
        ("Nvidia beats earnings", "NVDA", ENTITY_MATCH_ALIAS),
        ("Jensen Huang announces a GPU", "NVDA", ENTITY_MATCH_EXECUTIVE),
    ):
        resolution = resolve_entity(text, ticker)
        if resolution.method != expected:
            failures.append(
                f"{text!r} resolved as {resolution.method!r}, expected {expected!r}"
            )
        if not resolution.is_training_eligible():
            failures.append(f"{text!r} ({expected}) is not training-eligible")

    executive = resolve_entity("Jensen Huang spoke", "NVDA")
    direct = resolve_entity("NVDA spoke", "NVDA")
    if executive.confidence >= direct.confidence:
        failures.append(
            "an executive mention is not weaker than a direct ticker match — a "
            "CEO is often quoted about the industry, not the company"
        )

    # 3. The defect E2 fixed: failed resolution vs correct exclusion.
    other_company = resolve_entity("Broadcom announces a deal", "NVDA")
    nothing = resolve_entity("The company said it will expand", "NVDA")
    for resolution, label in ((other_company, "other company"), (nothing, "no entity")):
        if resolution.method != ENTITY_MATCH_NONE:
            failures.append(f"{label} did not resolve to none")
        if resolution.is_training_eligible():
            failures.append(f"{label} was training-eligible")
    if other_company.reason == nothing.reason:
        failures.append(
            "a correct exclusion and a failed resolution give the same reason — "
            "coverage gaps would be invisible"
        )
    if "AVGO" not in other_company.candidates:
        failures.append("an off-entity match did not name the company it found")

    # 4. Ambiguity.
    ambiguous = resolve_entity("Nvidia and AMD both rallied", "NVDA")
    if ambiguous.method != ENTITY_MATCH_AMBIGUOUS:
        failures.append(f"two named entities resolved as {ambiguous.method!r}")
    if ambiguous.confidence != 0.0:
        failures.append("an ambiguous resolution carries non-zero confidence")
    if ambiguous.is_training_eligible():
        failures.append(
            "an ambiguous resolution was training-eligible — that is a guess"
        )
    if "AMD" not in ambiguous.candidates:
        failures.append("an ambiguous resolution did not name the other candidate")

    # 5. Short-ticker guard (the V/BE class of error).
    for text, ticker, should_match in (
        ("The V shape recovery continued", "V", False),
        ("Visa raised its outlook", "V", True),
        ("This will be a strong quarter", "BE", False),
        ("Bloom Energy raised guidance", "BE", True),
        ("NVDA rallied", "NVDA", True),
    ):
        matched = resolve_entity(text, ticker).matched
        if matched != should_match:
            failures.append(
                f"{text!r} against {ticker}: matched={matched}, expected {should_match}"
            )

    # 6. Phrase boundaries.
    for text, ticker, label in (
        ("Advanced sensors, micro lenses and other devices", "AMD", "scattered words"),
        ("Metaverse investment slowed", "META", "substring"),
    ):
        if resolve_entity(text, ticker).matched:
            failures.append(f"a {label} match was accepted for {ticker}")
    if not resolve_entity("nvidia beats", "NVDA").matched:
        failures.append("matching is not case-insensitive")

    # An unregistered ticker is refused, not guessed.
    unregistered = resolve_entity("ZZZZ soared", "ZZZZ")
    if unregistered.matched or "not in the entity registry" not in unregistered.reason:
        failures.append("an unregistered ticker was not refused with a clear reason")

    # 7. Event integration.
    events = events_from_news_snapshot(_SNAPSHOT)
    if len(events) != 2:
        failures.append(f"expected 2 events, got {len(events)}")
    for event in events:
        if not event.entity_resolution:
            failures.append("an event carries no entity resolution")
    resolved = [e.is_entity_resolved() for e in events]
    if resolved != [True, False]:
        failures.append(
            f"resolved/unresolved events are not distinguishable (got {resolved})"
        )
    strict = events_from_news_snapshot(_SNAPSHOT, require_resolved_entity=True)
    if len(strict) != 1:
        failures.append(f"strict mode kept {len(strict)} events, expected 1")
    if any(not e.is_entity_resolved() for e in strict):
        failures.append("strict mode admitted an unresolved event into training data")

    # Coverage reporting separates the two failure kinds.
    report = resolution_report([ambiguous, nothing, direct])
    if (
        report["by_method"][ENTITY_MATCH_AMBIGUOUS] != 1
        or report["by_method"][ENTITY_MATCH_NONE] != 1
    ):
        failures.append("the coverage report does not separate ambiguous from none")

    if failures:
        print("E2 entity-resolution gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E2 entity-resolution gate OK:")
    print("  registry valid; ticker > legal_name > alias > executive, each with a confidence.")
    print("  a failed resolution is distinguishable from a correct exclusion.")
    print("  ambiguity resolves to 'ambiguous' with zero confidence, never to a guess.")
    print("  short tickers need a name; phrase matching respects word boundaries.")
    print(
        f"  events carry their resolution; strict mode drops anything below "
        f"{ENTITY_MIN_TRAINING_CONFIDENCE}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
