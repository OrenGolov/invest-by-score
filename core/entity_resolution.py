"""Entity resolution (Sprint E2).

The binding rule:

    **Bad resolution must not silently enter training.**

The emphasis is on *silently*. Bad resolution is unavoidable — headlines are
ambiguous, companies share words, executives speak about their industry
rather than their company. What is avoidable is a bad resolution that looks
exactly like a good one by the time it reaches a training row.

The N1 relevance heuristic returned `0.0` for two very different situations:

    "Broadcom announces a deal"     -> 0.0 for NVDA   (correct exclusion)
    "Jensen Huang announces a GPU"  -> 0.0 for NVDA   (FAILED resolution)

The first is a right answer. The second is the resolver not knowing that
Jensen Huang runs NVIDIA. Collapsing them means coverage gaps are invisible:
you cannot tell "we have no news about this company" from "we have news we
could not attribute".

E2 separates them:

    resolve_entity(text, ticker)
        |
        +-- EntityResolution(matched, method, confidence, evidence)
        |
        +-- method=ticker/legal_name/alias/executive  -> resolved
        +-- method=ambiguous                          -> matched >1 entity
        +-- method=none                               -> nothing matched
                                                         (and WHY is recorded)

Design decisions worth stating:

- **A registry, not a regex.** Aliases and executives are curated data with
  provenance, so a wrong alias is a visible, fixable registry entry rather
  than a buried pattern.
- **Short tickers require a name.** A bare "V" or "BE" in a headline is not
  evidence about Visa or Bloom Energy — those are ordinary English words.
  Tickers at or below `ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH` must match by
  name or alias instead, which is exactly the `V`/`BE` class of error that
  bit the portfolio list.
- **A ticker is evidence only in caps.** Length alone does not separate a
  symbol from a word: "Cat" in "Red Cat Holdings" is not Caterpillar. Ticker
  matching is case-sensitive, so `CAT` matches and `Cat`/`cat` do not. Names,
  aliases and executives stay case-insensitive — those are prose.
- **Executives are weaker evidence than tickers.** A CEO is frequently
  quoted about the industry, a rival, or the economy. `executive` carries
  0.6 confidence, above the training floor but visibly below a direct match.
- **Ambiguity is a resolution, not an error.** An article naming two
  registered companies resolves to `ambiguous` with zero confidence, so it
  is recorded and excluded rather than arbitrarily assigned.
- **Rejections are recorded.** `is_training_eligible()` gates what reaches a
  training row; nothing is thrown away, because the rejection rate is itself
  data about coverage.

Pure and deterministic: no wall-clock, no network, no randomness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from core.config import (
    ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH,
    ENTITY_MATCH_ALIAS,
    ENTITY_MATCH_AMBIGUOUS,
    ENTITY_MATCH_CONFIDENCE,
    ENTITY_MATCH_EXECUTIVE,
    ENTITY_MATCH_LEGAL_NAME,
    ENTITY_MATCH_METHODS,
    ENTITY_MATCH_NONE,
    ENTITY_MATCH_TICKER,
    ENTITY_MIN_TRAINING_CONFIDENCE,
    ENTITY_RESOLVER_VERSION,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9&.']+")


class EntityResolutionError(ValueError):
    """Raised when the entity registry or a resolution request is invalid."""


@dataclass(frozen=True)
class EntityRecord:
    """One resolvable entity: a company, its aliases and its named officers."""

    ticker: str
    legal_name: str
    aliases: tuple[str, ...] = ()
    executives: tuple[str, ...] = ()

    def all_names(self) -> tuple[str, ...]:
        return (self.legal_name, *self.aliases)


@dataclass
class EntityResolution:
    """The outcome of resolving text against one ticker.

    Carries WHY as well as WHAT: a consumer can see which signal produced the
    match, and a failed resolution explains itself rather than looking like
    an ordinary exclusion.
    """

    ticker: str
    matched: bool
    method: str
    confidence: float
    matched_text: str = ""
    candidates: list[str] = field(default_factory=list)
    reason: str = ""
    resolver_version: str = ENTITY_RESOLVER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_training_eligible(self) -> bool:
        """Whether this resolution may back a training row.

        A rejected resolution is still recorded — the rejection rate is data
        about coverage, and discarding it would hide exactly the gap this
        module exists to expose.
        """
        return self.matched and self.confidence >= ENTITY_MIN_TRAINING_CONFIDENCE


def _cased_tokens(text: str) -> set[str]:
    """Tokens with original case preserved, so a ticker matches only as a symbol."""
    return set(_TOKEN_PATTERN.findall(str(text or "")))


def _phrase_present(phrase: str, text: str) -> bool:
    """Whether a multi-word phrase appears, matched on word boundaries.

    Token-subset matching would let "Advanced Micro Devices" match a headline
    that merely contains those three words scattered apart.
    """
    cleaned = str(phrase or "").strip()
    if not cleaned:
        return False
    # A trailing \b after a non-word character (the "." in "Apple Inc.")
    # requires a word character next, so such a name could never match even
    # itself — 24 of 74 registry entries were unmatchable. Anchor the
    # boundary only where the phrase actually begins/ends in a word char.
    body = r"\s+".join(re.escape(part) for part in cleaned.split())
    prefix = r"\b" if cleaned[0].isalnum() else ""
    suffix = r"\b" if cleaned[-1].isalnum() else ""
    return re.search(
        prefix + body + suffix, str(text or ""), flags=re.IGNORECASE
    ) is not None


def build_default_entity_registry() -> dict[str, EntityRecord]:
    """The curated entity registry.

    Deliberately small and explicit. Every entry is checkable data with a
    visible owner rather than a pattern buried in a matcher, so a wrong alias
    is a one-line fix and a missing company is an obvious gap.

    Executives are listed only where the person genuinely speaks for the
    company; a name listed here claims that a mention of them is evidence
    about the company, which is a real claim and should be made carefully.
    """
    records = (
        # --- Semiconductors and AI hardware ---------------------------------
        EntityRecord("NVDA", "NVIDIA Corporation", ("NVIDIA", "Nvidia"), ("Jensen Huang",)),
        EntityRecord("AMD", "Advanced Micro Devices", ("AMD",), ("Lisa Su",)),
        EntityRecord("INTC", "Intel Corporation", ("Intel",), ("Pat Gelsinger", "Lip-Bu Tan")),
        EntityRecord("AVGO", "Broadcom Inc.", ("Broadcom",), ("Hock Tan",)),
        EntityRecord("QCOM", "Qualcomm Incorporated", ("Qualcomm",), ("Cristiano Amon",)),
        EntityRecord("MRVL", "Marvell Technology", ("Marvell",), ("Matt Murphy",)),
        EntityRecord("ARM", "Arm Holdings", ("Arm Holdings",), ("Rene Haas",)),
        EntityRecord("ASML", "ASML Holding", ("ASML",), ("Christophe Fouquet",)),
        EntityRecord("KLAC", "KLA Corporation", ("KLA",), ("Rick Wallace",)),
        EntityRecord("LRCX", "Lam Research", ("Lam Research",), ("Tim Archer",)),
        EntityRecord("TER", "Teradyne Inc.", ("Teradyne",), ("Greg Smith",)),
        EntityRecord("ALAB", "Astera Labs", ("Astera Labs",), ("Jitendra Mohan",)),
        EntityRecord("SKHY", "SK hynix Inc.", ("SK hynix",), ()),
        EntityRecord("CBRS", "Cerebras Systems", ("Cerebras",), ("Andrew Feldman",)),

        # --- Mega-cap technology --------------------------------------------
        EntityRecord("AAPL", "Apple Inc.", ("Apple",), ("Tim Cook",)),
        EntityRecord("MSFT", "Microsoft Corporation", ("Microsoft",), ("Satya Nadella",)),
        EntityRecord("GOOGL", "Alphabet Inc.", ("Alphabet", "Google"), ("Sundar Pichai",)),
        EntityRecord("AMZN", "Amazon.com Inc.", ("Amazon",), ("Andy Jassy",)),
        EntityRecord("META", "Meta Platforms Inc.", ("Meta Platforms", "Facebook"), ("Mark Zuckerberg",)),
        EntityRecord("TSLA", "Tesla Inc.", ("Tesla",), ("Elon Musk",)),
        EntityRecord("ORCL", "Oracle Corporation", ("Oracle",), ("Safra Catz",)),
        EntityRecord("CSCO", "Cisco Systems", ("Cisco",), ("Chuck Robbins",)),
        EntityRecord("DELL", "Dell Technologies", ("Dell",), ("Michael Dell",)),
        EntityRecord("GLW", "Corning Incorporated", ("Corning",), ("Wendell Weeks",)),
        EntityRecord("NOK", "Nokia Corporation", ("Nokia",), ("Justin Hotard",)),

        # --- Software, data and security ------------------------------------
        EntityRecord("PLTR", "Palantir Technologies", ("Palantir",), ("Alex Karp",)),
        EntityRecord("NOW", "ServiceNow Inc.", ("ServiceNow",), ("Bill McDermott",)),
        EntityRecord("SNOW", "Snowflake Inc.", ("Snowflake",), ("Sridhar Ramaswamy",)),
        EntityRecord("DDOG", "Datadog Inc.", ("Datadog",), ("Olivier Pomel",)),
        EntityRecord("CRWD", "CrowdStrike Holdings", ("CrowdStrike",), ("George Kurtz",)),
        EntityRecord("PANW", "Palo Alto Networks", ("Palo Alto Networks",), ("Nikesh Arora",)),
        EntityRecord("FTNT", "Fortinet Inc.", ("Fortinet",), ("Ken Xie",)),
        EntityRecord("OKTA", "Okta Inc.", ("Okta",), ("Todd McKinnon",)),
        EntityRecord("ANET", "Arista Networks", ("Arista",), ("Jayshree Ullal",)),
        EntityRecord("CGNX", "Cognex Corporation", ("Cognex",), ("Robert Willett",)),
        EntityRecord("AXON", "Axon Enterprise", ("Axon",), ("Rick Smith",)),
        EntityRecord("WBD", "Warner Bros. Discovery", ("Warner Bros Discovery",), ("David Zaslav",)),

        # --- Fintech and consumer -------------------------------------------
        EntityRecord("V", "Visa Inc.", ("Visa",), ("Ryan McInerney",)),
        EntityRecord("SOFI", "SoFi Technologies", ("SoFi",), ("Anthony Noto",)),
        EntityRecord("HOOD", "Robinhood Markets", ("Robinhood",), ("Vlad Tenev",)),
        EntityRecord("NU", "Nu Holdings", ("Nubank",), ("David Velez",)),
        EntityRecord("UBER", "Uber Technologies", ("Uber",), ("Dara Khosrowshahi",)),
        EntityRecord("ABNB", "Airbnb Inc.", ("Airbnb",), ("Brian Chesky",)),
        EntityRecord("KO", "The Coca-Cola Company", ("Coca-Cola",), ("James Quincey",)),
        EntityRecord("CCEP", "Coca-Cola Europacific Partners", ("Coca-Cola Europacific",), ()),
        EntityRecord("ODFL", "Old Dominion Freight Line", ("Old Dominion",), ("Marty Freeman",)),

        # --- Healthcare and life sciences ------------------------------------
        EntityRecord("LLY", "Eli Lilly and Company", ("Eli Lilly",), ("David Ricks",)),
        EntityRecord("MRNA", "Moderna Inc.", ("Moderna",), ("Stephane Bancel",)),
        EntityRecord("VRTX", "Vertex Pharmaceuticals", ("Vertex Pharmaceuticals",), ("Reshma Kewalramani",)),
        EntityRecord("ISRG", "Intuitive Surgical", ("Intuitive Surgical",), ("Gary Guthart",)),

        # --- Energy, utilities and industrials -------------------------------
        EntityRecord("CEG", "Constellation Energy", ("Constellation Energy",), ("Joseph Dominguez",)),
        EntityRecord("VST", "Vistra Corp.", ("Vistra",), ("Jim Burke",)),
        EntityRecord("AEP", "American Electric Power", ("American Electric Power",), ("Bill Fehrman",)),
        EntityRecord("EXC", "Exelon Corporation", ("Exelon",), ("Calvin Butler",)),
        EntityRecord("BKR", "Baker Hughes Company", ("Baker Hughes",), ("Lorenzo Simonelli",)),
        EntityRecord("BE", "Bloom Energy Corporation", ("Bloom Energy",), ("KR Sridhar",)),
        EntityRecord("OKLO", "Oklo Inc.", ("Oklo",), ("Jacob DeWitte",)),
        EntityRecord("CAT", "Caterpillar Inc.", ("Caterpillar",), ("Joe Creed",)),
        EntityRecord("VRT", "Vertiv Holdings", ("Vertiv",), ("Giordano Albertazzi",)),
        EntityRecord("MP", "MP Materials", ("MP Materials",), ("James Litinsky",)),

        # --- Space, quantum and emerging -------------------------------------
        EntityRecord("RKLB", "Rocket Lab USA", ("Rocket Lab",), ("Peter Beck",)),
        EntityRecord("IONQ", "IonQ Inc.", ("IonQ",), ("Niccolo de Masi",)),
        EntityRecord("RGTI", "Rigetti Computing", ("Rigetti",), ("Subodh Kulkarni",)),
        EntityRecord("ONDS", "Ondas Holdings", ("Ondas",), ("Eric Brock",)),
        EntityRecord("OUST", "Ouster Inc.", ("Ouster",), ("Angus Pacala",)),
        EntityRecord("KEEL", "Keel Infrastructure Corp.", ("Keel Infrastructure",), ()),
        EntityRecord("ASTS", "AST SpaceMobile Inc.", ("AST SpaceMobile", "SpaceMobile"), ("Abel Avellan",)),
        EntityRecord("RCAT", "Red Cat Holdings", ("Red Cat", "Teal Drones"), ("Jeff Thompson",)),

        # --- Digital infrastructure -------------------------------------------
        EntityRecord("CRWV", "CoreWeave Inc.", ("CoreWeave",), ("Michael Intrator",)),
        EntityRecord("NBIS", "Nebius Group", ("Nebius",), ("Arkady Volozh",)),
        EntityRecord("IREN", "IREN Limited", ("Iris Energy",), ("Daniel Roberts",)),

        # --- Funds and ETFs ---------------------------------------------------
        # A fund has no officer who speaks for it, so `executives` stays empty:
        # listing one would claim that a person's statement is evidence about
        # the fund, which is not true.
        EntityRecord("VOO", "Vanguard S&P 500 ETF", ("Vanguard S&P 500",), ()),
        EntityRecord("SOXX", "iShares Semiconductor ETF", ("iShares Semiconductor",), ()),
        EntityRecord("CIBR", "First Trust NASDAQ Cybersecurity ETF", ("First Trust Cybersecurity",), ()),
        EntityRecord("NASA", "Tema Space Innovators ETF", ("Tema Space Innovators",), ()),
        EntityRecord("SPCX", "Space Exploration Technologies Corp.", ("SpaceX",), ()),
    )
    return {record.ticker: record for record in records}


def registry_problems(registry: dict[str, EntityRecord]) -> list[str]:
    """Validate the registry: no empty names, no alias collisions.

    An alias shared by two companies would make every article naming it
    ambiguous, which is a registry defect rather than a data property.
    """
    problems: list[str] = []
    seen_names: dict[str, str] = {}
    for ticker, record in sorted(registry.items()):
        if ticker != record.ticker:
            problems.append(f"{ticker}: key does not match record.ticker {record.ticker!r}")
        if not str(record.legal_name or "").strip():
            problems.append(f"{ticker}: legal_name is required")
        for name in record.all_names():
            key = str(name).strip().upper()
            if not key:
                problems.append(f"{ticker}: empty alias")
                continue
            if key in seen_names and seen_names[key] != ticker:
                problems.append(
                    f"alias {name!r} is claimed by both {seen_names[key]} and "
                    f"{ticker} — a shared alias makes every mention ambiguous"
                )
            seen_names[key] = ticker
        for person in record.executives:
            if not str(person or "").strip():
                problems.append(f"{ticker}: empty executive name")
    return problems


def _match_one(text: str, record: EntityRecord) -> tuple[str, str] | None:
    """Strongest match of `record` in `text`, or None. Returns (method, text)."""
    # A ticker is evidence only when written as a symbol, i.e. in caps. "Cat"
    # in "Red Cat Holdings" is the word, not Caterpillar; only "CAT" is CAT.
    if (
        len(record.ticker) > ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH
        and record.ticker in _cased_tokens(text)
    ):
        return ENTITY_MATCH_TICKER, record.ticker

    if _phrase_present(record.legal_name, text):
        return ENTITY_MATCH_LEGAL_NAME, record.legal_name
    for alias in record.aliases:
        if _phrase_present(alias, text):
            return ENTITY_MATCH_ALIAS, alias
    for person in record.executives:
        if _phrase_present(person, text):
            return ENTITY_MATCH_EXECUTIVE, person
    return None


def resolve_entity(
    text: str,
    ticker: str,
    registry: dict[str, EntityRecord] | None = None,
    provider_ticker: str | None = None,
) -> EntityResolution:
    """Resolve whether `text` is about `ticker`, and say how it decided.

    A provider-tagged ticker is trusted as a direct match: the provider
    asserted the association, and second-guessing it here would discard the
    strongest signal available.
    """
    active = registry if registry is not None else build_default_entity_registry()
    target = str(ticker).strip().upper()
    if not target:
        raise EntityResolutionError("a ticker is required to resolve against")

    if provider_ticker and str(provider_ticker).strip().upper() == target:
        return EntityResolution(
            ticker=target,
            matched=True,
            method=ENTITY_MATCH_TICKER,
            confidence=ENTITY_MATCH_CONFIDENCE[ENTITY_MATCH_TICKER],
            matched_text=target,
            candidates=[target],
            reason="the provider tagged this record with the ticker",
        )

    record = active.get(target)
    if record is None:
        return EntityResolution(
            ticker=target,
            matched=False,
            method=ENTITY_MATCH_NONE,
            confidence=0.0,
            reason=(
                f"{target} is not in the entity registry, so a mention of it "
                f"cannot be verified — add it rather than guessing"
            ),
        )

    # Which registered entities does this text mention at all? Ambiguity is a
    # property of the text, so every entity must be checked, not just the
    # target.
    matches: dict[str, tuple[str, str]] = {}
    for candidate_ticker, candidate in active.items():
        found = _match_one(text, candidate)
        if found is not None:
            matches[candidate_ticker] = found

    if target not in matches:
        return EntityResolution(
            ticker=target,
            matched=False,
            method=ENTITY_MATCH_NONE,
            confidence=0.0,
            candidates=sorted(matches),
            reason=(
                f"no ticker, name, alias or executive of {target} appears in the "
                f"text"
                + (f"; it mentions {sorted(matches)} instead" if matches else "")
            ),
        )

    if len(matches) > 1:
        others = sorted(set(matches) - {target})
        return EntityResolution(
            ticker=target,
            matched=False,
            method=ENTITY_MATCH_AMBIGUOUS,
            confidence=0.0,
            matched_text=matches[target][1],
            candidates=sorted(matches),
            reason=(
                f"the text also names {others} — attributing it to {target} "
                f"alone would be a guess"
            ),
        )

    method, matched_text = matches[target]
    return EntityResolution(
        ticker=target,
        matched=True,
        method=method,
        confidence=ENTITY_MATCH_CONFIDENCE[method],
        matched_text=matched_text,
        candidates=[target],
        reason=f"matched {matched_text!r} by {method}",
    )


def resolve_article(
    article: dict,
    ticker: str,
    registry: dict[str, EntityRecord] | None = None,
) -> EntityResolution:
    """Resolve an N1-shaped article against a ticker."""
    if not isinstance(article, dict):
        raise EntityResolutionError(
            f"article must be a dict, got {type(article).__name__}"
        )
    text = " ".join(
        str(article.get(field_name) or "")
        for field_name in ("headline", "summary", "company_name")
    )
    return resolve_entity(
        text, ticker, registry, provider_ticker=article.get("ticker")
    )


def resolution_report(resolutions: Iterable[EntityResolution]) -> dict[str, Any]:
    """Coverage summary over a batch — the visibility E2 exists to provide.

    The rejection rate is the number worth watching: a high `none` share
    means the registry is missing entries, while a high `ambiguous` share
    means the source material is genuinely hard to attribute. Those call for
    different fixes, so they are counted separately.
    """
    items = list(resolutions)
    by_method: dict[str, int] = {}
    for resolution in items:
        by_method[resolution.method] = by_method.get(resolution.method, 0) + 1
    eligible = [r for r in items if r.is_training_eligible()]
    return {
        "resolver_version": ENTITY_RESOLVER_VERSION,
        "total": len(items),
        "training_eligible": len(eligible),
        "rejected": len(items) - len(eligible),
        "by_method": {method: by_method.get(method, 0) for method in ENTITY_MATCH_METHODS},
        "eligibility_rate": round(len(eligible) / len(items), 4) if items else 0.0,
        "min_training_confidence": ENTITY_MIN_TRAINING_CONFIDENCE,
    }
