"""Configuration part 3: E1-E7 event intelligence.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_validation` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._validation import *  # noqa: F401,F403


# --- Canonical event object (E1) -------------------------------------------------
# Sprint E learns how events move the chart. That requires ONE event shape
# every downstream stage agrees on: the event study (E4), the attribution
# engine (E5) and event memory (E6) must all read the same object, or they
# will silently disagree about what an "event" is.
EVENT_SCHEMA_VERSION = "event-v1"

# Who or what produced the statement. An earnings release from the company
# and a broker note about the company are not the same kind of evidence, and
# E3 (influential-person intelligence) needs the distinction to exist from
# the start rather than being retrofitted.
ACTOR_TYPE_COMPANY = "company"
ACTOR_TYPE_EXECUTIVE = "executive"
ACTOR_TYPE_REGULATOR = "regulator"
ACTOR_TYPE_ANALYST = "analyst"
ACTOR_TYPE_INFLUENCER = "influencer"
ACTOR_TYPE_JOURNALIST = "journalist"
ACTOR_TYPE_UNKNOWN = "unknown"
ACTOR_TYPES = (
    ACTOR_TYPE_COMPANY, ACTOR_TYPE_EXECUTIVE, ACTOR_TYPE_REGULATOR,
    ACTOR_TYPE_ANALYST, ACTOR_TYPE_INFLUENCER, ACTOR_TYPE_JOURNALIST,
    ACTOR_TYPE_UNKNOWN,
)

# Direction is the CLAIMED polarity of the event, not a prediction and not a
# realised return. CONTRADICTORY is a first-class value: credible evidence
# pointing both ways must never average to neutral (master context section
# 15), and that rule has to survive into the event object.
EVENT_DIRECTION_POSITIVE = "positive"
EVENT_DIRECTION_NEGATIVE = "negative"
EVENT_DIRECTION_NEUTRAL = "neutral"
EVENT_DIRECTION_CONTRADICTORY = "contradictory"
EVENT_DIRECTIONS = (
    EVENT_DIRECTION_POSITIVE, EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_NEUTRAL, EVENT_DIRECTION_CONTRADICTORY,
)

# An event with no evidence is a rumour. Every event must cite at least this
# many source records, so nothing enters event memory unsourced.
EVENT_MIN_EVIDENCE = 1

# Magnitude is a bounded, unitless claim size — NOT an expected return.
# Keeping it unitless prevents it being mistaken for a forecast before the
# F-sprint exists to make real ones.
EVENT_MAGNITUDE_MIN = 0.0
EVENT_MAGNITUDE_MAX = 1.0


def _validate_event_config() -> None:
    """Import-time guard: the event vocabularies must stay coherent."""
    if ACTOR_TYPE_UNKNOWN not in ACTOR_TYPES:
        raise ValueError("an explicit 'unknown' actor type must exist — fail-closed")
    if EVENT_DIRECTION_CONTRADICTORY not in EVENT_DIRECTIONS:
        raise ValueError(
            "CONTRADICTORY must be a direction — contradictory evidence must "
            "never average to neutral"
        )
    if EVENT_MIN_EVIDENCE < 1:
        raise ValueError("an event must cite at least one source record")
    if EVENT_MAGNITUDE_MIN >= EVENT_MAGNITUDE_MAX:
        raise ValueError("the magnitude band must be non-empty")


_validate_event_config()


# --- Entity resolution (E2) ------------------------------------------------------
# "Bad resolution must not silently enter training."
#
# The N1 relevance heuristic returns 0.0 both for an article that is clearly
# about another company AND for one whose entity could not be resolved at
# all. Those are different failures: the first is a correct exclusion, the
# second is an UNKNOWN that must be visible. E2 separates them.
ENTITY_RESOLVER_VERSION = "entity-resolver-v1"

# How an entity mention was matched, strongest first. The method is recorded
# on every resolution so a downstream consumer can see WHY a match was made,
# not merely that it was.
ENTITY_MATCH_TICKER = "ticker"          # exact symbol, provider-tagged or in text
ENTITY_MATCH_LEGAL_NAME = "legal_name"  # full registered name
ENTITY_MATCH_ALIAS = "alias"            # a curated alias of the company
ENTITY_MATCH_EXECUTIVE = "executive"    # a named officer speaking for the company
ENTITY_MATCH_NONE = "none"              # nothing matched
ENTITY_MATCH_AMBIGUOUS = "ambiguous"    # matched more than one entity
ENTITY_MATCH_METHODS = (
    ENTITY_MATCH_TICKER, ENTITY_MATCH_LEGAL_NAME, ENTITY_MATCH_ALIAS,
    ENTITY_MATCH_EXECUTIVE, ENTITY_MATCH_NONE, ENTITY_MATCH_AMBIGUOUS,
)

# Confidence per match method. An executive mention is weaker evidence than a
# ticker: a CEO can be quoted about the industry rather than the company.
ENTITY_MATCH_CONFIDENCE = {
    ENTITY_MATCH_TICKER: 1.0,
    ENTITY_MATCH_LEGAL_NAME: 0.9,
    ENTITY_MATCH_ALIAS: 0.8,
    ENTITY_MATCH_EXECUTIVE: 0.6,
    ENTITY_MATCH_AMBIGUOUS: 0.0,
    ENTITY_MATCH_NONE: 0.0,
}

# Below this, a resolution is not usable as training evidence. It is still
# RECORDED — the rejection is data about coverage — but it cannot silently
# become a labelled example.
ENTITY_MIN_TRAINING_CONFIDENCE = 0.6

# Tickers short enough to collide with ordinary words. A bare "V" or "BE" in
# a headline is not evidence about Visa or Bloom Energy, so these require a
# name or alias match rather than a bare symbol token.
ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH = 2


def _validate_entity_config() -> None:
    """Import-time guard: the resolver vocabularies must stay coherent."""
    missing = set(ENTITY_MATCH_METHODS) - set(ENTITY_MATCH_CONFIDENCE)
    if missing:
        raise ValueError(f"match methods without a confidence: {sorted(missing)}")
    for method in (ENTITY_MATCH_NONE, ENTITY_MATCH_AMBIGUOUS):
        if ENTITY_MATCH_CONFIDENCE[method] != 0.0:
            raise ValueError(f"{method!r} must carry zero confidence")
    if not 0.0 < ENTITY_MIN_TRAINING_CONFIDENCE <= 1.0:
        raise ValueError("ENTITY_MIN_TRAINING_CONFIDENCE must be in (0, 1]")
    if ENTITY_AMBIGUOUS_TICKER_MAX_LENGTH < 1:
        raise ValueError("the ambiguous-ticker length guard must be positive")


_validate_entity_config()


# --- Influential person intelligence (E3) ----------------------------------------
# E3 tracks WHO speaks, and eventually learns actor x topic x company ->
# historical market response. The learning part belongs to E4/E6, which
# measure reactions; E3's job is to make the actor a first-class, versioned
# object so those measurements have something stable to attach to.
ACTOR_REGISTRY_VERSION = "actor-registry-v1"

# An actor's standing relative to the company they are speaking about.
# INSIDER binds them to the company (a CEO's guidance IS company guidance);
# EXTERNAL does not (an analyst's opinion is about the company, not from it).
# The distinction decides whether a statement can be treated as the company
# speaking.
ACTOR_STANDING_INSIDER = "insider"
ACTOR_STANDING_EXTERNAL = "external"
ACTOR_STANDING_UNKNOWN = "unknown"
ACTOR_STANDINGS = (
    ACTOR_STANDING_INSIDER, ACTOR_STANDING_EXTERNAL, ACTOR_STANDING_UNKNOWN,
)

# Baseline credibility by standing, before any measured track record exists.
# These are PRIORS, not findings: E3 has no market-impact history yet, and
# pretending otherwise would be the "arbitrary number presented as measured"
# failure M6 exists to prevent.
ACTOR_BASE_CREDIBILITY = {
    ACTOR_STANDING_INSIDER: 0.75,
    ACTOR_STANDING_EXTERNAL: 0.50,
    ACTOR_STANDING_UNKNOWN: 0.25,
}

# An actor needs this many observed statements before any per-actor
# historical impact may be reported. Below it, the actor is tracked but its
# impact is UNMEASURED — a two-statement "track record" is noise.
ACTOR_MIN_OBSERVATIONS = 20

# How much a topic specialisation can raise or lower relevance. A CEO
# speaking about their own product line is stronger evidence than the same
# CEO on macro policy; the adjustment is bounded so specialisation can
# inform a judgement without dominating it.
ACTOR_TOPIC_BONUS = 0.15
ACTOR_OFF_TOPIC_PENALTY = 0.20


def _validate_actor_config() -> None:
    """Import-time guard: the actor vocabularies must stay coherent."""
    missing = set(ACTOR_STANDINGS) - set(ACTOR_BASE_CREDIBILITY)
    if missing:
        raise ValueError(f"standings without a base credibility: {sorted(missing)}")
    if ACTOR_BASE_CREDIBILITY[ACTOR_STANDING_INSIDER] <= ACTOR_BASE_CREDIBILITY[
        ACTOR_STANDING_EXTERNAL
    ]:
        raise ValueError(
            "an insider must start more credible about their own company than an "
            "external commentator"
        )
    if ACTOR_MIN_OBSERVATIONS < 2:
        raise ValueError(
            "a track record needs more than one observation to be a record"
        )
    for name, value in (("ACTOR_TOPIC_BONUS", ACTOR_TOPIC_BONUS),
                        ("ACTOR_OFF_TOPIC_PENALTY", ACTOR_OFF_TOPIC_PENALTY)):
        if not 0.0 <= value <= 0.5:
            raise ValueError(f"{name} must be a bounded adjustment in [0, 0.5]")


_validate_actor_config()


# --- Event study engine (E4) -----------------------------------------------------
# For every event: pre-event baseline -> stock reaction -> benchmark reaction
# -> sector reaction -> abnormal return -> volatility response -> volume
# response, measured across the standard horizons.
EVENT_STUDY_VERSION = "event-study-v1"

# The pre-event estimation window. Everything the study calls "normal" comes
# from here, and it ENDS before the event so the baseline can never contain
# the reaction it is used to measure.
EVENT_STUDY_BASELINE_SESSIONS = 60

# A gap between the baseline window and the event. Information often leaks
# into prices before an announcement is published; including those sessions
# in "normal" would fold part of the reaction into the baseline and shrink
# the abnormal return toward zero.
EVENT_STUDY_BASELINE_GAP_SESSIONS = 2

# Minimum usable baseline sessions. Below this the baseline is too thin to
# describe normal behaviour, and the study refuses rather than reporting a
# number nobody should trust.
EVENT_STUDY_MIN_BASELINE_SESSIONS = 30

# Horizons, aligned with LABEL_HORIZON_SESSIONS so an event study and a
# training label always describe the same window. "intraday" is the event
# session itself (open to close), which the daily bars can support honestly.
EVENT_STUDY_HORIZONS = ("intraday", "1d", "5d", "20d", "60d")

# Abnormal return model. "market_adjusted" subtracts the benchmark return
# directly (beta = 1). A full market model would estimate beta from the
# baseline, which is the natural v2 — it is NOT done here, and the model name
# travels with every result so nobody mistakes one for the other.
EVENT_STUDY_MODEL_MARKET_ADJUSTED = "market_adjusted"
EVENT_STUDY_MODEL_MEAN_ADJUSTED = "mean_adjusted"
EVENT_STUDY_MODELS = (
    EVENT_STUDY_MODEL_MARKET_ADJUSTED, EVENT_STUDY_MODEL_MEAN_ADJUSTED,
)


def _validate_event_study_config() -> None:
    """Import-time guard: the study windows must be coherent."""
    if EVENT_STUDY_MIN_BASELINE_SESSIONS > EVENT_STUDY_BASELINE_SESSIONS:
        raise ValueError(
            "the minimum baseline cannot exceed the requested baseline window"
        )
    if EVENT_STUDY_MIN_BASELINE_SESSIONS < 2:
        raise ValueError("a baseline needs at least two sessions to have dispersion")
    if EVENT_STUDY_BASELINE_GAP_SESSIONS < 0:
        raise ValueError("the pre-event gap cannot be negative")
    for horizon in EVENT_STUDY_HORIZONS:
        if horizon != "intraday" and horizon not in LABEL_HORIZON_SESSIONS:
            raise ValueError(
                f"event-study horizon {horizon!r} is not a declared label horizon — "
                f"a study and a training label must describe the same window"
            )


_validate_event_study_config()


# --- Confounder / attribution engine (E5) ----------------------------------------
# "Never automatically claim causality — decompose observed movement into
# market component + sector component + stock-specific component +
# event-associated residual."
#
# E4 measures what happened. E5 asks how much of it was the market, the
# sector, or the company, and how much is left over near the event. The
# leftover is EVENT-ASSOCIATED, never event-caused: a residual is a question,
# not an answer.
EVENT_ATTRIBUTION_VERSION = "event-attribution-v1"

# The decomposition is additive by construction:
#
#     stock = market + sector_excess + stock_specific
#
# where sector_excess = sector - market. Using the RAW sector return would
# count market beta twice (a sector ETF already contains it) and hand the
# error to the residual, which is the one number nobody would notice was
# wrong.
ATTRIBUTION_COMPONENTS = (
    "market", "sector_excess", "stock_specific",
)

# How confidently the residual can be associated with the event at all.
# These are CONFOUNDING verdicts, not significance tests: they say whether
# other explanations have been ruled out enough to make the association
# interesting, never whether the event caused anything.
ATTRIBUTION_CONFOUNDED = "confounded"          # market/sector explains most of it
ATTRIBUTION_UNEXPLAINED = "unexplained"        # residual dominates, no attribution
ATTRIBUTION_EVENT_ASSOCIATED = "event_associated"
ATTRIBUTION_INCONCLUSIVE = "inconclusive"      # not enough context to judge
ATTRIBUTION_VERDICTS = (
    ATTRIBUTION_CONFOUNDED, ATTRIBUTION_UNEXPLAINED,
    ATTRIBUTION_EVENT_ASSOCIATED, ATTRIBUTION_INCONCLUSIVE,
)

# A residual must exceed this share of the total absolute movement before it
# is called event-associated at all. Below it, common factors explain the
# move and the event adds nothing worth recording.
ATTRIBUTION_MIN_RESIDUAL_SHARE = 0.5

# ...and must be at least this many baseline standard deviations. A residual
# that is large in share but tiny in magnitude is noise on a quiet day.
ATTRIBUTION_MIN_RESIDUAL_SIGMA = 1.5

# A window containing this many OTHER events cannot be attributed to any one
# of them. Overlapping events are the most common confounder in event
# studies and the easiest to forget.
ATTRIBUTION_MAX_OVERLAPPING_EVENTS = 0


def _validate_attribution_config() -> None:
    """Import-time guard: the attribution thresholds must be meaningful."""
    if not 0.0 < ATTRIBUTION_MIN_RESIDUAL_SHARE <= 1.0:
        raise ValueError("ATTRIBUTION_MIN_RESIDUAL_SHARE must be in (0, 1]")
    if ATTRIBUTION_MIN_RESIDUAL_SIGMA <= 0.0:
        raise ValueError(
            "ATTRIBUTION_MIN_RESIDUAL_SIGMA must be positive — a zero threshold "
            "would call every flicker event-associated"
        )
    if ATTRIBUTION_MAX_OVERLAPPING_EVENTS < 0:
        raise ValueError("the overlapping-event allowance cannot be negative")
    if ATTRIBUTION_EVENT_ASSOCIATED not in ATTRIBUTION_VERDICTS:
        raise ValueError("the event-associated verdict must be declared")


_validate_attribution_config()


# --- Event memory (E6) -----------------------------------------------------------
# "Store event, context, chart state, historical analogs, and 1D/5D/20D/60D
# response as institutional memory."
#
# This is the store the forecasting sprints retrieve from. Its value depends
# entirely on not remembering things that were never true, so a memory is
# written ONLY when its event, its study and its attribution are all
# complete — a half-recorded memory is worse than an absent one, because it
# looks like evidence.
EVENT_MEMORY_VERSION = "event-memory-v1"

# The chart-state fields captured at the moment of the event. Drawn from the
# shared snapshot surface so a remembered chart state and a live one are
# described in exactly the same terms — a memory that cannot be compared with
# the present is not institutional memory.
EVENT_MEMORY_CHART_FIELDS = (
    "close", "rsi", "volatility", "volume_ratio_20d", "atr_14",
    "trend_slope_60d", "trend_vs_20d_mean", "market_regime",
    "change_5d", "change_20d", "change_60d",
    "price_vs_ma_50", "price_vs_ma_200",
)

# The subset of the above that SIMILARITY is computed over. `close` and
# `atr_14` are LEVEL fields: they say how expensive the stock is, not what the
# chart is doing. Every other field is already scale-free (a ratio or a percent
# change), so including levels made analog retrieval a price-level filter.
#
# MEASURED, two IDENTICAL chart shapes at different price levels ($180 vs
# $420): similarity 0.846 with the level fields, 1.000 without. Over 5,466
# random CROSS-ticker pairs of real chart states the mean rose 0.356 -> 0.401,
# and discrimination IMPROVED rather than loosened: an OPPOSITE-shape pair fell
# 0.230 -> 0.090.
#
# The levels stay in EVENT_MEMORY_CHART_FIELDS because a memory should still
# RECORD what the stock cost at the time; they are simply not evidence of
# similarity. Recording and matching are different jobs.
EVENT_MEMORY_SIMILARITY_FIELDS = tuple(
    name for name in EVENT_MEMORY_CHART_FIELDS if name not in ("close", "atr_14")
)

# The response horizons a memory records. Aligned with the event study so a
# remembered response and a fresh measurement always describe the same window.
EVENT_MEMORY_RESPONSE_HORIZONS = ("1d", "5d", "20d", "60d")

# Comparison scale per chart field: roughly the magnitude at which a
# difference becomes meaningful. Similarity divides the gap by this rather
# than by the values themselves, because a purely relative measure collapses
# near zero — two nearly-flat slopes (+0.0012 vs -0.0009) are both "flat",
# yet relative difference calls them 0.0 similar and drags a 0.97 match below
# the retrieval bar. Fields absent here fall back to relative comparison.
EVENT_MEMORY_FIELD_SCALE = {
    "close": 20.0,              # retained for recording; not a similarity field
    "rsi": 20.0,                # 20 RSI points is a regime apart
    # MEASURED: the chart state carries ANNUALIZED volatility (0.28 = 28%),
    # not daily, so a 0.01 scale meant any 1pp gap scored ZERO — two ordinary
    # stocks at 20% and 25% vol were called completely dissimilar, and the
    # field contributed a mean of 0.029 across 4,000 real pairs. At 0.15 a
    # 15%/18% pair scores 0.80 and a genuinely different 28%/55% pair still
    # scores 0.00, which is the discrimination the field was meant to provide.
    "volatility": 0.15,
    "volume_ratio_20d": 0.5,
    "atr_14": 2.0,
    "trend_slope_60d": 0.25,
    "trend_vs_20d_mean": 0.03,
    "change_5d": 0.03,
    "change_20d": 0.08,
    "change_60d": 0.15,
    "price_vs_ma_50": 0.08,
    "price_vs_ma_200": 0.15,
}

# PROVENANCE — was this memory OBSERVED, or INFERRED from price behaviour?
#
# MEASURED: no configured source supplies dated historical events. The news
# adapter looks back NEWS_LOOKBACK_DAYS (7) and needs an API key; Alpha
# Vantage supplies at most the NEXT earnings date, one forward date per
# ticker. So a store that reaches back years can only be built by INFERENCE.
#
# A quarterly cadence filter dates earnings well — keeping the largest volume
# spike in each 63-session window yields 19-20 picks per 19 windows across 12
# tickers, matching the ~20 real earnings events in 5 years, at median gaps of
# 61-64 against a theoretical 63.
#
# THE DECIDING FACT: `fetch_fundamental_snapshot` returns earnings_date=None,
# and there is no historical earnings calendar anywhere in the system. An
# inferred date has nothing to be scored against. It can be FLAGGED; it cannot
# be VERIFIED. This field therefore records WHAT PRODUCED the memory, not a
# confidence — a confidence would imply a measurement that does not exist.
MEMORY_PROVENANCE_OBSERVED = "observed"    # a real, sourced event
MEMORY_PROVENANCE_INFERRED = "inferred"    # dated from price behaviour

EVENT_MEMORY_PROVENANCES: tuple[str, ...] = (
    MEMORY_PROVENANCE_OBSERVED,
    MEMORY_PROVENANCE_INFERRED,
)

# An unlabelled memory is indistinguishable from an observed one once written,
# so provenance is REQUIRED at the door rather than defaulted. The default on
# the dataclass exists only so older readers do not crash; `memory_problems`
# refuses a memory that does not state it.
EVENT_MEMORY_REQUIRE_PROVENANCE = True

# The methods an inferred memory may name. Declared as data so a reader can
# tell exactly what produced a date, and so a new method cannot appear without
# being written down here first.
EVENT_MEMORY_INFERENCE_METHODS: dict[str, str] = {
    "quarterly_volume_cadence": (
        "the largest 20-day-relative volume spike in each 63-session window, "
        "excluding quarterly triple-witching dates and requiring an opening "
        "gap, which together match the quarterly earnings cadence. "
        "MEASURED against PUBLISHED earnings dates for AAPL/MSFT/NVDA/JPM: "
        "PRECISION 0.65 (23 picks, 15 within +/-2 sessions of a real "
        "earnings date). A plain volume-cadence filter scores only 0.35 "
        "because 31.7% of its picks land on quarterly TRIPLE-WITCHING dates "
        "-- options expiry, not earnings -- which share the same quarterly "
        "high-volume signature. Tightening further trades away nearly all "
        "the volume (3 picks at 0.67), so ~0.65 is the ceiling this signal "
        "supports. ROUGHLY ONE IN THREE inferred events is therefore NOT the "
        "event it is labelled as; the price move is real, the attribution is "
        "not. This is why such memories are never laundered into observed "
        "ones and why F5 degrades a forecast that leans on them."
    ),
}

# The MEASURED precision of each inference method, against published ground
# truth. Declared as data because a reader of an inferred memory needs to know
# how often the label is wrong, and because a method whose precision was never
# measured must not be usable at all.
EVENT_MEMORY_INFERENCE_PRECISION: dict[str, float] = {
    "quarterly_volume_cadence": 0.65,
}

# An inference method below this precision is not worth recording: its
# memories would carry more mislabelling than signal. 0.65 clears it; the
# unfiltered 0.35 variant does not, which is why the witching and gap filters
# are part of the method rather than an option on it.
EVENT_MEMORY_MIN_INFERENCE_PRECISION = 0.50

# Analog retrieval. Similarity is computed over the chart-state fields above,
# and a match must clear this bar before it is offered as a comparable.
EVENT_MEMORY_MIN_SIMILARITY = 0.7

# No fewer than this many analogs makes a median response worth reporting.
# Below it, retrieval returns the matches but refuses to summarise them —
# a "typical response" from two examples is not typical of anything.
EVENT_MEMORY_MIN_ANALOGS = 5


def _validate_event_memory_config() -> None:
    """Import-time guard: the memory contract must stay coherent."""
    for horizon in EVENT_MEMORY_RESPONSE_HORIZONS:
        if horizon not in LABEL_HORIZON_SESSIONS:
            raise ValueError(
                f"memory horizon {horizon!r} is not a declared label horizon — a "
                f"remembered response and a fresh measurement must describe the "
                f"same window"
            )
    if not 0.0 < EVENT_MEMORY_MIN_SIMILARITY <= 1.0:
        raise ValueError("EVENT_MEMORY_MIN_SIMILARITY must be in (0, 1]")
    if EVENT_MEMORY_MIN_ANALOGS < 2:
        raise ValueError(
            "a typical response needs more than one example to be typical"
        )
    if not EVENT_MEMORY_CHART_FIELDS:
        raise ValueError("a memory without chart state cannot be compared to the present")
    for name, scale in EVENT_MEMORY_FIELD_SCALE.items():
        if scale <= 0:
            raise ValueError(
                f"EVENT_MEMORY_FIELD_SCALE[{name!r}] must be positive — a zero "
                f"scale would make every difference infinite"
            )
    if not EVENT_MEMORY_SIMILARITY_FIELDS:
        raise ValueError("similarity needs at least one field to compare")
    if len(set(EVENT_MEMORY_PROVENANCES)) != len(EVENT_MEMORY_PROVENANCES):
        raise ValueError("EVENT_MEMORY_PROVENANCES contains a duplicate")
    if MEMORY_PROVENANCE_OBSERVED not in EVENT_MEMORY_PROVENANCES:
        raise ValueError("a memory must be able to be observed")
    if MEMORY_PROVENANCE_INFERRED not in EVENT_MEMORY_PROVENANCES:
        raise ValueError(
            "the inferred provenance must stay declared — removing it would "
            "not delete the inferred memories already written, it would only "
            "stop them being recognisable as inferred"
        )
    if not EVENT_MEMORY_REQUIRE_PROVENANCE:
        raise ValueError(
            "provenance must be required at the door: an unlabelled memory is "
            "indistinguishable from an observed one once written"
        )
    if not EVENT_MEMORY_INFERENCE_METHODS:
        raise ValueError(
            "an inferred memory must be able to name the method that produced "
            "it, or a reader cannot tell what they are looking at"
        )
    for name, rationale in EVENT_MEMORY_INFERENCE_METHODS.items():
        if "MEASURED" not in rationale:
            raise ValueError(
                f"inference method {name!r} does not carry its measurement — "
                f"an inference method without evidence is a guess with a name"
            )
        if name not in EVENT_MEMORY_INFERENCE_PRECISION:
            raise ValueError(
                f"inference method {name!r} has no MEASURED precision — a "
                f"method whose error rate was never measured must not be "
                f"usable, because nobody can weigh what it produces"
            )
    for name, precision in EVENT_MEMORY_INFERENCE_PRECISION.items():
        if name not in EVENT_MEMORY_INFERENCE_METHODS:
            raise ValueError(f"precision declared for unknown method {name!r}")
        if not 0.0 < precision <= 1.0:
            raise ValueError(f"precision for {name!r} must lie in (0, 1]")
        if precision < EVENT_MEMORY_MIN_INFERENCE_PRECISION:
            raise ValueError(
                f"inference method {name!r} scores {precision}, below the "
                f"{EVENT_MEMORY_MIN_INFERENCE_PRECISION} floor — its memories "
                f"would carry more mislabelling than signal"
            )
    extra = set(EVENT_MEMORY_SIMILARITY_FIELDS) - set(EVENT_MEMORY_CHART_FIELDS)
    if extra:
        raise ValueError(
            f"EVENT_MEMORY_SIMILARITY_FIELDS must be a SUBSET of the recorded "
            f"chart fields; unknown: {sorted(extra)}"
        )
    # The level fields must stay OUT of similarity. MEASURED: including them
    # scores two identical chart shapes at 0.846 when the stocks trade at
    # different prices, and retrieval then returns near-duplicates of the same
    # ticker (84.5% of pairs clearing the 0.70 bar were the same ticker).
    for level_field in ("close", "atr_14"):
        if level_field in EVENT_MEMORY_SIMILARITY_FIELDS:
            raise ValueError(
                f"{level_field!r} is a PRICE-LEVEL field and must not be a "
                f"similarity field — it makes analog retrieval a price filter, "
                f"matching a $180 stock only to other $180 stocks"
            )


_validate_event_memory_config()


# --- Event revision / contradiction learning (E7) --------------------------------
# "Store event chains (initial claim -> correction -> confirmation ->
# reversal) — the evolution itself becomes training information."
#
# A story that was reported, corrected, then reversed is a DIFFERENT kind of
# evidence from one that was reported once and stood. Treating the four
# reports as four independent events would quadruple-count a single story and
# hide the fact that the market learned it was wrong.
EVENT_CHAIN_VERSION = "event-chain-v1"

# How a later event relates to an earlier one in the same chain.
CHAIN_LINK_INITIAL = "initial_claim"
CHAIN_LINK_CORRECTION = "correction"      # amends details, direction intact
CHAIN_LINK_CONFIRMATION = "confirmation"  # independent corroboration
CHAIN_LINK_REVERSAL = "reversal"          # the claim was wrong
CHAIN_LINK_DUPLICATE = "duplicate"        # same story, no new information
CHAIN_LINKS = (
    CHAIN_LINK_INITIAL, CHAIN_LINK_CORRECTION, CHAIN_LINK_CONFIRMATION,
    CHAIN_LINK_REVERSAL, CHAIN_LINK_DUPLICATE,
)

# A chain's settled state, which is what downstream learning should read.
CHAIN_STATUS_OPEN = "open"              # only an initial claim so far
CHAIN_STATUS_CONFIRMED = "confirmed"
CHAIN_STATUS_CORRECTED = "corrected"
CHAIN_STATUS_REVERSED = "reversed"
CHAIN_STATUS_CONTESTED = "contested"    # confirmed AND reversed — unresolved
CHAIN_STATUSES = (
    CHAIN_STATUS_OPEN, CHAIN_STATUS_CONFIRMED, CHAIN_STATUS_CORRECTED,
    CHAIN_STATUS_REVERSED, CHAIN_STATUS_CONTESTED,
)

# Events for the same entity and type within this many days are candidates
# for the same chain. Beyond it they are separate stories, not an evolution.
EVENT_CHAIN_WINDOW_DAYS = 14

# Reliability weight by settled state. A reversed claim is not merely
# uninformative — it is evidence the source got it wrong, so it weighs less
# than an unconfirmed one. These are PRIORS for chain handling, not measured
# source scores; L4 learns those from outcomes.
CHAIN_RELIABILITY_WEIGHT = {
    CHAIN_STATUS_CONFIRMED: 1.0,
    CHAIN_STATUS_OPEN: 0.7,
    CHAIN_STATUS_CORRECTED: 0.6,
    CHAIN_STATUS_CONTESTED: 0.3,
    CHAIN_STATUS_REVERSED: 0.1,
}


def _validate_event_chain_config() -> None:
    """Import-time guard: the chain vocabularies must stay coherent."""
    missing = set(CHAIN_STATUSES) - set(CHAIN_RELIABILITY_WEIGHT)
    if missing:
        raise ValueError(f"chain statuses without a weight: {sorted(missing)}")
    if CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_REVERSED] >= CHAIN_RELIABILITY_WEIGHT[
        CHAIN_STATUS_OPEN
    ]:
        raise ValueError(
            "a reversed claim must weigh less than an unconfirmed one — a "
            "reversal is evidence the source was wrong"
        )
    if CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_CONFIRMED] <= CHAIN_RELIABILITY_WEIGHT[
        CHAIN_STATUS_CONTESTED
    ]:
        raise ValueError("a confirmed claim must outweigh a contested one")
    if EVENT_CHAIN_WINDOW_DAYS < 1:
        raise ValueError("the chain window must be at least a day")


_validate_event_chain_config()


