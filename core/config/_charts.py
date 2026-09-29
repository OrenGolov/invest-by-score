"""Configuration part 4: C1-C7 chart and temporal intelligence.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_events` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._events import *  # noqa: F401,F403


# --- Multi-timeframe representation (C1) -----------------------------------------
# One `as_of`, five timeframes, each PIT-correct on its OWN bar clock.
#
# The defining hazard: a weekly/monthly bar is LABELLED at period start but only
# COMPLETE at period end. Yahoo labels the week of Sep 14 as `2026-09-14`; at
# as_of = Sep 15 that bar's Close is Friday's close — future data wearing a past
# timestamp. A naive `index <= as_of` filter keeps it, which is why each
# timeframe declares how long its bar takes to close and a bar is only eligible
# once `bar_open + period >= as_of` is FALSE. Providers also append a partial
# trailing bar (a `2026-09-17` row inside a monthly series); it fails the same
# test and is excluded and counted, never silently dropped.

TIMEFRAME_CONTRACT_VERSION = "timeframe-contract-v1"
TIMEFRAME_PIPELINE_VERSION = "timeframe-pipeline-v1"

TIMEFRAME_INTRADAY = "intraday"
TIMEFRAME_DAILY = "daily"
TIMEFRAME_WEEKLY = "weekly"
TIMEFRAME_MONTHLY = "monthly"
TIMEFRAME_YEARLY = "yearly"

# Ordered coarsest-last so a report reads from the fastest clock to the slowest.
TIMEFRAME_ORDER: tuple[str, ...] = (
    TIMEFRAME_INTRADAY,
    TIMEFRAME_DAILY,
    TIMEFRAME_WEEKLY,
    TIMEFRAME_MONTHLY,
    TIMEFRAME_YEARLY,
)

# provider interval + fetch range + how long one bar takes to close.
# `yearly` is derived by resampling monthly bars: the provider has no 1y
# interval, and inventing one would be a second price truth (W5).
TIMEFRAME_SPECS: dict[str, dict[str, object]] = {
    TIMEFRAME_INTRADAY: {"interval": "1h", "period": "1mo", "close_after": "1h", "resample": None},
    TIMEFRAME_DAILY: {"interval": "1d", "period": "1y", "close_after": "1D", "resample": None},
    TIMEFRAME_WEEKLY: {"interval": "1wk", "period": "2y", "close_after": "7D", "resample": None},
    TIMEFRAME_MONTHLY: {"interval": "1mo", "period": "10y", "close_after": "31D", "resample": None},
    TIMEFRAME_YEARLY: {"interval": "1mo", "period": "10y", "close_after": "366D", "resample": "YE"},
}

# Minimum eligible bars before a timeframe reports a trend rather than None.
# Below this the timeframe is INCOMPLETE: a slope fitted to two points is
# arithmetic, not evidence.
TIMEFRAME_MIN_BARS: dict[str, int] = {
    TIMEFRAME_INTRADAY: 8,
    TIMEFRAME_DAILY: 20,
    TIMEFRAME_WEEKLY: 8,
    TIMEFRAME_MONTHLY: 6,
    TIMEFRAME_YEARLY: 3,
}

# Trend label thresholds, as a fraction of the window's first close.
TIMEFRAME_TREND_FLAT_BAND = 0.02

TIMEFRAME_TREND_UP = "up"
TIMEFRAME_TREND_DOWN = "down"
TIMEFRAME_TREND_FLAT = "flat"
TIMEFRAME_TRENDS: tuple[str, ...] = (
    TIMEFRAME_TREND_UP,
    TIMEFRAME_TREND_DOWN,
    TIMEFRAME_TREND_FLAT,
)


def _validate_timeframe_config() -> None:
    """Import-time guard: the spec tables must agree on the timeframe set."""
    if set(TIMEFRAME_SPECS) != set(TIMEFRAME_ORDER):
        raise ValueError("TIMEFRAME_SPECS must cover exactly TIMEFRAME_ORDER")
    if set(TIMEFRAME_MIN_BARS) != set(TIMEFRAME_ORDER):
        raise ValueError("TIMEFRAME_MIN_BARS must cover exactly TIMEFRAME_ORDER")
    for name, spec in TIMEFRAME_SPECS.items():
        for field_name in ("interval", "period", "close_after"):
            if not spec.get(field_name):
                raise ValueError(f"timeframe {name!r} is missing {field_name!r}")
    if not 0.0 < TIMEFRAME_TREND_FLAT_BAND < 1.0:
        raise ValueError("the flat band must be a fraction strictly inside (0, 1)")
    if min(TIMEFRAME_MIN_BARS.values()) < 2:
        raise ValueError("a trend needs at least two bars")


_validate_timeframe_config()


# --- Chart feature expansion (C2) ------------------------------------------------
# The eight price/volume features C2 adds. Returns, momentum, volatility, ATR,
# volume surprise and slope are NOT here: they are already registered against
# `market_data_agent` (M1), and a second implementation would be exactly the
# split-brain scoring the master context forbids (W5, one canonical truth).
#
# Every window below is a LOOKBACK measured backwards from the last eligible
# bar. The producer never reads beyond that bar, so a feature cannot see past
# the as_of its caller already filtered to.

CHART_FEATURE_VERSION = "chart-feature-v1"
CHART_PIPELINE_VERSION = "chart-pipeline-v1"

# Acceleration: change in momentum. Second difference of price over two
# consecutive windows, so a decelerating rally is distinguishable from a
# stalling one.
CHART_ACCELERATION_WINDOW = 10

# A gap is measured open-vs-previous-close, in fractions of the previous close.
# Below this, ordinary overnight drift is not a gap.
CHART_GAP_MIN_FRACTION = 0.01

# Drawdown/recovery are measured over this trailing window.
CHART_DRAWDOWN_WINDOW = 60

# Support/resistance are the extremes of this window; distance is reported as
# a fraction of the last close.
CHART_LEVEL_WINDOW = 60

# Breakout: a close beyond the prior-window extreme by more than this margin.
# A "failed" breakout is one that broke out within the confirmation window and
# has since closed back inside the range — the distinction that makes the
# feature worth having.
CHART_BREAKOUT_MARGIN = 0.005
CHART_BREAKOUT_CONFIRM_SESSIONS = 5

# Volatility expansion/contraction: short realized vol against long realized
# vol. Above the expansion ratio is expanding, below the contraction ratio is
# contracting, between them is stable.
CHART_VOL_SHORT_WINDOW = 10
CHART_VOL_LONG_WINDOW = 60
CHART_VOL_EXPANSION_RATIO = 1.25
CHART_VOL_CONTRACTION_RATIO = 0.80

# Relative strength window. The benchmark frame is INJECTED — C2 never selects
# a benchmark (that is C3's contract) and never fabricates one.
CHART_RELATIVE_STRENGTH_WINDOW = 60

CHART_BREAKOUT_NONE = "none"
CHART_BREAKOUT_UP = "breakout_up"
CHART_BREAKOUT_DOWN = "breakout_down"
CHART_BREAKOUT_FAILED_UP = "failed_breakout_up"
CHART_BREAKOUT_FAILED_DOWN = "failed_breakout_down"
CHART_BREAKOUT_STATES: tuple[str, ...] = (
    CHART_BREAKOUT_NONE,
    CHART_BREAKOUT_UP,
    CHART_BREAKOUT_DOWN,
    CHART_BREAKOUT_FAILED_UP,
    CHART_BREAKOUT_FAILED_DOWN,
)

CHART_VOL_EXPANDING = "expanding"
CHART_VOL_CONTRACTING = "contracting"
CHART_VOL_STABLE = "stable"
CHART_VOL_STATES: tuple[str, ...] = (
    CHART_VOL_EXPANDING,
    CHART_VOL_CONTRACTING,
    CHART_VOL_STABLE,
)

# Feature name -> minimum eligible bars. Below this the feature is None, never
# a neutral zero: a drawdown computed over three bars is not a drawdown.
CHART_FEATURE_MIN_HISTORY: dict[str, int] = {
    "acceleration_10d": CHART_ACCELERATION_WINDOW * 2 + 1,
    "gap_pct": 2,
    "drawdown_60d": CHART_DRAWDOWN_WINDOW,
    "recovery_speed_60d": CHART_DRAWDOWN_WINDOW,
    "support_distance_60d": CHART_LEVEL_WINDOW,
    "resistance_distance_60d": CHART_LEVEL_WINDOW,
    "breakout_state_60d": CHART_LEVEL_WINDOW + CHART_BREAKOUT_CONFIRM_SESSIONS,
    "volatility_regime_ratio": CHART_VOL_LONG_WINDOW + 1,
    "relative_strength_60d": CHART_RELATIVE_STRENGTH_WINDOW + 1,
}


def _validate_chart_feature_config() -> None:
    """Import-time guard: windows must be usable and thresholds ordered."""
    for name, window in (
        ("acceleration", CHART_ACCELERATION_WINDOW),
        ("drawdown", CHART_DRAWDOWN_WINDOW),
        ("level", CHART_LEVEL_WINDOW),
        ("vol_short", CHART_VOL_SHORT_WINDOW),
        ("vol_long", CHART_VOL_LONG_WINDOW),
        ("relative_strength", CHART_RELATIVE_STRENGTH_WINDOW),
    ):
        if window < 2:
            raise ValueError(f"chart window {name!r} must span at least two bars")
    if CHART_VOL_SHORT_WINDOW >= CHART_VOL_LONG_WINDOW:
        raise ValueError("the short vol window must be shorter than the long one")
    if CHART_VOL_CONTRACTION_RATIO >= CHART_VOL_EXPANSION_RATIO:
        raise ValueError("contraction must sit below expansion")
    if not 0.0 < CHART_GAP_MIN_FRACTION < 1.0:
        raise ValueError("the gap floor must be a fraction inside (0, 1)")
    if CHART_BREAKOUT_MARGIN < 0.0:
        raise ValueError("the breakout margin cannot be negative")


_validate_chart_feature_config()


# --- Market context features (C3) ------------------------------------------------
# A stock cannot be interpreted in isolation: "+4% while the S&P did +4%" and
# "+4% while the S&P did -2%" are different facts, and only the second is
# relative strength. C3 supplies the benchmark/sector frames that C2's
# `relative_strength_60d` has been waiting on.
#
# VIX and the 10Y are deliberately NOT price series here. Both are already
# registered macro series (FRED, vintage-aware, publication-time gated) from
# N3, and re-fetching them as Yahoo bars would create a second source of truth
# for the same quantity — the split-brain the master context forbids (W5). The
# context block REFERENCES the macro snapshot instead.

CONTEXT_CONTRACT_VERSION = "context-contract-v1"
CONTEXT_PIPELINE_VERSION = "context-pipeline-v1"

# Broad market proxies, fetched as price bars. ETFs rather than raw indices:
# an index level has no volume and no tradable history, while the ETF shares
# the same provider contract, split handling and cache path as every other
# ticker in the system.
CONTEXT_INDEX_PROXIES: dict[str, str] = {
    "sp500": "SPY",
    "nasdaq100": "QQQ",
    "russell2000": "IWM",
    "usd": "UUP",
}

# The default benchmark when a caller does not name one. SPY, because the
# ensemble's equity view is US large-cap; a caller may override per request.
CONTEXT_DEFAULT_BENCHMARK = "sp500"

# GICS sector -> sector ETF. The industry benchmark C3 supplies is the sector
# ETF; a finer industry classification needs a real classification service and
# is explicitly NOT invented here.
CONTEXT_SECTOR_ETFS: dict[str, str] = {
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# Macro series the context block references rather than re-fetching. The key is
# the context field, the value is the macro registry key it reads.
CONTEXT_MACRO_REFERENCES: dict[str, str] = {
    "vix": "vix",
    "yield_10y": "10y_yield",
}

# Window for context returns, matching C2's relative-strength window so the
# two are directly comparable.
CONTEXT_RETURN_WINDOW = 60

CONTEXT_SOURCE_ID = "yahoo_finance_chart"


def _validate_context_config() -> None:
    """Import-time guard: the context tables must be complete and consistent."""
    if CONTEXT_DEFAULT_BENCHMARK not in CONTEXT_INDEX_PROXIES:
        raise ValueError(
            f"the default benchmark {CONTEXT_DEFAULT_BENCHMARK!r} must be one of "
            f"{sorted(CONTEXT_INDEX_PROXIES)}"
        )
    if not CONTEXT_SECTOR_ETFS:
        raise ValueError("at least one sector ETF must be declared")
    duplicates = [
        etf for etf in set(CONTEXT_SECTOR_ETFS.values())
        if list(CONTEXT_SECTOR_ETFS.values()).count(etf) > 1
    ]
    if duplicates:
        raise ValueError(f"a sector ETF is claimed by two sectors: {sorted(duplicates)}")
    if CONTEXT_RETURN_WINDOW < 2:
        raise ValueError("the context return window must span at least two bars")


_validate_context_config()


# --- Deterministic chart structure (C4) ------------------------------------------
# C4 answers "what is this chart DOING" as a composed, reproducible description
# rather than a pattern name. Every label below is derived from measurable
# primitives with declared thresholds: the same bars always produce the same
# structure, and a reader can check the arithmetic. No screenshot
# interpretation, no "head and shoulders", no vague pattern vocabulary.
#
# Swing structure uses FRACTAL pivots: a bar is a swing high when its High is
# the maximum of the window centred on it. That centring is the PIT hazard —
# a pivot cannot be confirmed until CHART_SWING_FRACTAL_K bars AFTER it exist,
# so scanning to the final bar would let future bars decide a past label. The
# scan therefore stops k bars short of the end, and the unconfirmed tail is
# reported rather than silently trimmed.

STRUCTURE_CONTRACT_VERSION = "structure-contract-v1"
STRUCTURE_PIPELINE_VERSION = "structure-pipeline-v1"

# Fractal half-width. 3 gives ~26 swings per trading year on a liquid name:
# enough structure to read, few enough that noise is not promoted to a swing.
CHART_SWING_FRACTAL_K = 3

# Swings considered when labelling structure. Two highs and two lows are the
# minimum that can express "higher high AND higher low".
CHART_SWING_LOOKBACK = 4

# A swing must differ from its predecessor by this fraction to count as higher
# or lower. Without it, a one-cent difference becomes a structural break.
CHART_SWING_MIN_CHANGE = 0.005

# Consolidation: the range of the window as a fraction of its mean close.
# Inside this, price is going sideways rather than trending.
CHART_CONSOLIDATION_WINDOW = 20
CHART_CONSOLIDATION_MAX_RANGE = 0.06

# Reversal: a directional flip between the prior window and the recent one,
# where both moves clear the threshold. A drift that changes sign is not a
# reversal.
CHART_REVERSAL_WINDOW = 10
CHART_REVERSAL_MIN_MOVE = 0.03

# Swing structure labels.
STRUCTURE_SWING_UPTREND = "higher_high_higher_low"
STRUCTURE_SWING_DOWNTREND = "lower_high_lower_low"
STRUCTURE_SWING_EXPANDING = "broadening"          # higher high AND lower low
STRUCTURE_SWING_CONTRACTING = "narrowing"         # lower high AND higher low
STRUCTURE_SWING_MIXED = "mixed"
STRUCTURE_SWING_LABELS: tuple[str, ...] = (
    STRUCTURE_SWING_UPTREND,
    STRUCTURE_SWING_DOWNTREND,
    STRUCTURE_SWING_EXPANDING,
    STRUCTURE_SWING_CONTRACTING,
    STRUCTURE_SWING_MIXED,
)

# Phase: the single-word summary of the structure, resolved by explicit
# precedence (see core.chart_structure.resolve_phase). Precedence is declared
# here so it is data, not buried control flow.
STRUCTURE_PHASE_BREAKOUT = "breakout"
STRUCTURE_PHASE_FAILED_BREAKOUT = "failed_breakout"
STRUCTURE_PHASE_REVERSAL = "reversal"
STRUCTURE_PHASE_CONSOLIDATION = "consolidation"
STRUCTURE_PHASE_TRENDING = "trending"
STRUCTURE_PHASE_UNDEFINED = "undefined"
STRUCTURE_PHASES: tuple[str, ...] = (
    STRUCTURE_PHASE_BREAKOUT,
    STRUCTURE_PHASE_FAILED_BREAKOUT,
    STRUCTURE_PHASE_REVERSAL,
    STRUCTURE_PHASE_CONSOLIDATION,
    STRUCTURE_PHASE_TRENDING,
    STRUCTURE_PHASE_UNDEFINED,
)

# Minimum eligible bars before a structure is described at all.
CHART_STRUCTURE_MIN_BARS = 70


def _validate_structure_config() -> None:
    """Import-time guard: the structure vocabulary must be coherent."""
    if CHART_SWING_FRACTAL_K < 1:
        raise ValueError("the fractal half-width must be at least one bar")
    if CHART_SWING_LOOKBACK < 2:
        raise ValueError("labelling higher-high AND higher-low needs two swings a side")
    if not 0.0 < CHART_SWING_MIN_CHANGE < 1.0:
        raise ValueError("the swing threshold must be a fraction inside (0, 1)")
    if not 0.0 < CHART_CONSOLIDATION_MAX_RANGE < 1.0:
        raise ValueError("the consolidation band must be a fraction inside (0, 1)")
    if CHART_REVERSAL_WINDOW < 2 or CHART_CONSOLIDATION_WINDOW < 2:
        raise ValueError("structure windows must span at least two bars")
    if CHART_STRUCTURE_MIN_BARS < CHART_CONSOLIDATION_WINDOW + CHART_REVERSAL_WINDOW:
        raise ValueError("the minimum history must cover every structure window")
    if STRUCTURE_PHASE_UNDEFINED not in STRUCTURE_PHASES:
        raise ValueError("an undefined phase must remain expressible")


_validate_structure_config()


# --- Temporal sequence dataset (C5) ----------------------------------------------
# A sequence is T-LOOKBACK .. T0: one step per session, each carrying the state
# that was knowable at that step. C6 sequence models consume these.
#
# C5 is NOT a second training-row generator. M2 remains the only door into a
# supervised dataset (`core.training_dataset`); C5 builds a separate, versioned,
# hashed SEQUENCE artifact and takes its labels from the same V1 label builder
# M2 uses, so the two can never disagree about an outcome.
#
# The honesty rule that shapes the schema: price, volume, the C2 feature surface
# and the C4 structure are recomputed AT EVERY STEP from the same PIT-filtered
# frame, so they genuinely vary across the window. Market, sector, macro,
# sentiment, event and fundamental state are attached ONCE at T0 and labelled
# `as_of_t0`. Back-projecting today's macro reading across sixty past steps
# would be exactly the revised-data-in-history failure the master context
# forbids, and pretending otherwise would be worse than not carrying them.

SEQUENCE_CONTRACT_VERSION = "sequence-contract-v1"
SEQUENCE_PIPELINE_VERSION = "sequence-pipeline-v1"
SEQUENCE_SCHEMA_VERSION = "sequence-schema-v1"

# T-60 .. T0 inclusive == 61 steps.
SEQUENCE_LOOKBACK_STEPS = 60

# A step's own feature computation needs history behind it: C2's widest window
# is 61 bars and C4 needs CHART_STRUCTURE_MIN_BARS. The frame handed to the
# builder must cover the lookback PLUS that warm-up, or early steps would be
# silently thinner than late ones.
SEQUENCE_STEP_WARMUP_BARS = 70

# Channels carried per step, recomputed at each one.
SEQUENCE_STEP_CHANNELS: tuple[str, ...] = (
    "price",
    "volume",
    "technical",
    "structure",
)

# Channels attached once at T0. Labelled, never back-projected.
SEQUENCE_T0_CHANNELS: tuple[str, ...] = (
    "market",
    "sector",
    "macro",
    "sentiment",
    "events",
    "fundamentals",
)

SEQUENCE_CONTEXT_SCOPE = "as_of_t0"

SEQUENCE_STATUS_OK = "OK"
SEQUENCE_STATUS_INCOMPLETE = "INCOMPLETE"
SEQUENCE_STATUS_UNAVAILABLE = "UNAVAILABLE"

# A sequence with gaps is not a sequence. Below this share of populated steps
# the artifact is INCOMPLETE and carries no steps at all, because a model
# trained on a ragged window learns the raggedness.
SEQUENCE_MIN_STEP_COVERAGE = 0.95


def _validate_sequence_config() -> None:
    """Import-time guard: the sequence contract must be self-consistent."""
    if SEQUENCE_LOOKBACK_STEPS < 2:
        raise ValueError("a sequence must span at least two steps")
    if SEQUENCE_STEP_WARMUP_BARS < CHART_STRUCTURE_MIN_BARS:
        raise ValueError(
            "the per-step warm-up must cover the structure minimum, or early "
            "steps would be thinner than late ones"
        )
    if set(SEQUENCE_STEP_CHANNELS) & set(SEQUENCE_T0_CHANNELS):
        raise ValueError("a channel cannot be both per-step and as-of-T0")
    if not 0.0 < SEQUENCE_MIN_STEP_COVERAGE <= 1.0:
        raise ValueError("step coverage must be a fraction inside (0, 1]")
    if SEQUENCE_CONTEXT_SCOPE != "as_of_t0":
        raise ValueError("the context scope label must stay explicit")


_validate_sequence_config()


# --- Sequence model research (C6) ------------------------------------------------
# C6's own precondition: "evaluate only after baseline validation". The master
# context is blunter — do not build a complex neural network before proving
# simple baselines, and promote nothing without beating the incumbent OOS.
#
# So this block declares the ARCHITECTURES C6 may eventually evaluate and the
# gate every one of them must pass. It deliberately declares no framework: a
# deep-learning dependency is a real decision, and adding one before a
# candidate can be judged would put the tooling ahead of the evidence.
#
# The promotion bar is NOT redefined here. A sequence candidate is judged by
# `core.baseline_suite.compare_runs` against the measured incumbent, on the
# same margin and fold-consistency rules every other candidate faces. A second
#, friendlier bar for the fashionable model is exactly how complexity sneaks
# past its evidence.

SEQUENCE_RESEARCH_VERSION = "sequence-research-v1"

SEQUENCE_ARCH_TCN = "temporal_convolution"
SEQUENCE_ARCH_LSTM = "lstm"
SEQUENCE_ARCH_GRU = "gru"
SEQUENCE_ARCH_TRANSFORMER = "time_series_transformer"
SEQUENCE_ARCH_FUSION = "temporal_fusion"

SEQUENCE_ARCHITECTURES: tuple[str, ...] = (
    SEQUENCE_ARCH_TCN,
    SEQUENCE_ARCH_LSTM,
    SEQUENCE_ARCH_GRU,
    SEQUENCE_ARCH_TRANSFORMER,
    SEQUENCE_ARCH_FUSION,
)

# Every architecture is unimplemented until it has a framework AND a measured
# incumbent to beat. Status is data, so `research_readiness` can report the
# truth rather than a module-level optimism.
SEQUENCE_ARCH_STATUS_PROPOSED = "proposed"
SEQUENCE_ARCH_STATUS_IMPLEMENTED = "implemented"

SEQUENCE_ARCH_REGISTRY: dict[str, dict[str, str]] = {
    SEQUENCE_ARCH_TCN: {
        "status": SEQUENCE_ARCH_STATUS_PROPOSED,
        "rationale": (
            "Dilated causal convolutions read a fixed window cheaply and cannot "
            "see forward by construction, which makes them the least dangerous "
            "first sequence model on PIT data."
        ),
    },
    SEQUENCE_ARCH_LSTM: {
        "status": SEQUENCE_ARCH_STATUS_PROPOSED,
        "rationale": "Recurrent memory over the 61-step window; the classic baseline for ordered data.",
    },
    SEQUENCE_ARCH_GRU: {
        "status": SEQUENCE_ARCH_STATUS_PROPOSED,
        "rationale": "Fewer parameters than LSTM; often equal on short windows, so it tests whether the extra gate earns its place.",
    },
    SEQUENCE_ARCH_TRANSFORMER: {
        "status": SEQUENCE_ARCH_STATUS_PROPOSED,
        "rationale": (
            "Attention over the window. Needs a causal mask or it reads the "
            "future outright — the single largest leakage risk in this list."
        ),
    },
    SEQUENCE_ARCH_FUSION: {
        "status": SEQUENCE_ARCH_STATUS_PROPOSED,
        "rationale": (
            "Combines the per-step channels with the as-of-T0 context C5 "
            "separates; the only architecture here that uses that split directly."
        ),
    },
}

# The metric a sequence candidate is judged on, matching the baseline suite so
# the comparison is like-for-like.
SEQUENCE_PRIMARY_METRIC = "directional_accuracy"

# Minimum completed trials carrying the primary metric before C6 may claim a
# baseline exists. One run is an anecdote.
SEQUENCE_MIN_BASELINE_TRIALS = 1


def _validate_sequence_research_config() -> None:
    """Import-time guard: the architecture registry must stay honest."""
    if set(SEQUENCE_ARCH_REGISTRY) != set(SEQUENCE_ARCHITECTURES):
        raise ValueError("SEQUENCE_ARCH_REGISTRY must cover exactly SEQUENCE_ARCHITECTURES")
    for name, entry in SEQUENCE_ARCH_REGISTRY.items():
        if entry.get("status") not in (
            SEQUENCE_ARCH_STATUS_PROPOSED,
            SEQUENCE_ARCH_STATUS_IMPLEMENTED,
        ):
            raise ValueError(f"architecture {name!r} carries an unknown status")
        if not entry.get("rationale"):
            raise ValueError(f"architecture {name!r} must justify its presence")
    if SEQUENCE_MIN_BASELINE_TRIALS < 1:
        raise ValueError("at least one completed baseline trial is required")


_validate_sequence_research_config()


# --- Chart reaction memory (C7) --------------------------------------------------
# For a significant event, retain what the chart looked like BEFORE it and how it
# reacted after, so a later setup can be matched against history.
#
# C7 composes rather than duplicates. E4 (core.event_study) already measures the
# intraday/1d/5d/20d/60d reaction against a pre-event baseline, and E6
# (core.event_memory) already stores an event with a flat chart snapshot and
# retrieves analogs. What C7 adds is the part neither has: the 1h reaction, and a
# STRUCTURAL before-picture from C4 rather than a bag of numbers.
#
# The 1h honesty rule. Providers serve roughly one month of hourly bars, while a
# 60d reaction needs sixty sessions AFTER the event — so for any event old enough
# to have a 60d reaction, hourly data does not exist. 1h is therefore OPTIONAL and
# explicitly UNAVAILABLE when absent. Interpolating it from daily bars would
# invent a reaction that was never observed, which is worse than not having one.

REACTION_MEMORY_VERSION = "reaction-memory-v1"
REACTION_MEMORY_SCHEMA_VERSION = "reaction-memory-schema-v1"

# Reaction horizons, coarsest last. "1h" is the C7 addition; the rest are E4's,
# named identically so a reader can line them up.
REACTION_HORIZON_1H = "1h"
REACTION_HORIZONS: tuple[str, ...] = ("1h", "intraday", "1d", "5d", "20d", "60d")

# Horizons that need intraday bars, and are therefore allowed to be UNAVAILABLE
# on an older event without the memory being considered incomplete.
REACTION_INTRADAY_HORIZONS: tuple[str, ...] = (REACTION_HORIZON_1H,)

# How far after the event an hourly entry bar may sit. Without this bound, an
# event that PREDATES the hourly window matches its first bar -- `index >=
# target` is true for every bar -- and silently measures an unrelated hour
# months later as if it were the reaction.
REACTION_HOURLY_MAX_ENTRY_GAP_HOURS = 24

# Sessions of chart history captured before the event. Enough for C4 to describe
# a structure rather than guess at one.
REACTION_PRE_EVENT_SESSIONS = 70

REACTION_STATUS_OK = "OK"
REACTION_STATUS_INCOMPLETE = "INCOMPLETE"
REACTION_STATUS_UNAVAILABLE = "UNAVAILABLE"

# A memory missing any DAILY horizon is INCOMPLETE: those are the horizons every
# historical event can supply, so their absence means something went wrong.
REACTION_REQUIRED_HORIZONS: tuple[str, ...] = ("1d", "5d", "20d")

# Retrieval. A match is only useful if the before-picture is comparable, so
# similarity keys on the structural phase first and the numeric state second.
REACTION_MIN_SIMILARITY = 0.7
REACTION_PHASE_MATCH_WEIGHT = 0.4
REACTION_NUMERIC_MATCH_WEIGHT = 0.6

# Fewer analogs than this and a median response is an anecdote, not a base rate.
REACTION_MIN_ANALOGS = 3


def _validate_reaction_memory_config() -> None:
    """Import-time guard: the reaction vocabulary must stay coherent."""
    for horizon in REACTION_REQUIRED_HORIZONS:
        if horizon not in REACTION_HORIZONS:
            raise ValueError(f"required horizon {horizon!r} is not a declared reaction horizon")
    for horizon in REACTION_INTRADAY_HORIZONS:
        if horizon not in REACTION_HORIZONS:
            raise ValueError(f"intraday horizon {horizon!r} is not a declared reaction horizon")
    if set(REACTION_REQUIRED_HORIZONS) & set(REACTION_INTRADAY_HORIZONS):
        raise ValueError(
            "an intraday horizon cannot also be required — its data does not "
            "exist for older events"
        )
    if REACTION_PRE_EVENT_SESSIONS < CHART_STRUCTURE_MIN_BARS:
        raise ValueError(
            "the pre-event window must cover the structure minimum, or the "
            "before-picture cannot be described"
        )
    if not 0.0 < REACTION_MIN_SIMILARITY <= 1.0:
        raise ValueError("the similarity bar must be a fraction inside (0, 1]")
    weight_total = REACTION_PHASE_MATCH_WEIGHT + REACTION_NUMERIC_MATCH_WEIGHT
    if abs(weight_total - 1.0) > 1e-9:
        raise ValueError(f"similarity weights must sum to 1.0, got {weight_total}")
    if REACTION_MIN_ANALOGS < 2:
        raise ValueError("a base rate needs more than one observation")


_validate_reaction_memory_config()


