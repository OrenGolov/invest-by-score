"""Configuration part: Sprint L1-L8 continuous learning and institutional memory.

Split out of the single 9,783-line `core/config.py` in C1, and then again
because the forecasting part alone was 3,631 lines — leaving it would have
moved the navigability problem rather than solved it. The text is UNCHANGED.

CHAINED from `_forecasting`: 88 constants are read across part boundaries, so
the parts reproduce ONE flat namespace in the original order.
"""

from core.config._forecasting import *  # noqa: F401,F403


# Sprint L1 - Outcome closure
# ---------------------------------------------------------------------------
# forecast -> horizon expires -> actual outcome -> error -> calibration.
#
# The sprint goal is "controlled learning, not uncontrolled self-modification",
# and closure is where that control lives: a forecast is scored ONCE, against
# the outcome that actually happened, by a measure that fits the claim it made.
#
# L1 CANNOT ASSUME A FORECAST EXISTS TO CLOSE. MEASURED: nothing persists a
# ForecastSnapshot - F8 builds one on demand and it is discarded, and the
# 3,301-row decision audit holds ZERO forecast-shaped rows. It stores SCORES,
# which have no horizon and therefore cannot expire. So L1's first job is the
# FORECAST LEDGER.
#
# THE MATURITY QUESTION IS ALREADY SOLVED. `build_outcome_labels` and
# `horizon_readiness` (V1/F2) answer "has this expired?" and "what actually
# happened?". MEASURED: NVDA at 2024-06-15 has all six horizons matured, while
# 2026-09-10 has 1d and 5d matured and the rest pending. L1 COMPOSES them.
FORECAST_LEDGER_VERSION = "forecast-ledger-v1"
OUTCOME_CLOSURE_VERSION = "outcome-closure-v1"
CLOSURE_CONTRACT_VERSION = "closure-contract-v1"

# The append-only ledger. One row per (ticker, as_of, horizon, target) written
# WHEN THE FORECAST IS MADE, so a later outcome has something to close against.
FORECAST_LEDGER_PATH = "data/forecast_ledger.jsonl"

# Closure states. A forecast moves OPEN -> MATURED -> CLOSED, and a horizon
# whose window elapsed without usable data lands in EXPIRED_NO_DATA rather
# than being silently dropped: "we could not score it" and "it never existed"
# are different facts.
CLOSURE_OPEN = "OPEN"                      # window has not elapsed
CLOSURE_MATURED = "MATURED"                # window elapsed, not yet scored
CLOSURE_CLOSED = "CLOSED"                  # scored against a real outcome
CLOSURE_EXPIRED_NO_DATA = "EXPIRED_NO_DATA"  # elapsed, but no usable outcome

CLOSURE_STATES: tuple[str, ...] = (
    CLOSURE_OPEN,
    CLOSURE_MATURED,
    CLOSURE_CLOSED,
    CLOSURE_EXPIRED_NO_DATA,
)

# States from which a forecast may still be closed. A CLOSED forecast is
# terminal: re-closing it would double-count one observation and silently
# re-weight every metric derived from the ledger.
CLOSURE_CLOSEABLE_STATES: tuple[str, ...] = (CLOSURE_MATURED,)

# ERROR IS SCORED PER CLAIM TIER, never by one universal function. F4/F5 emit
# four kinds of claim and one measure cannot serve them:
#
#   POINT        a number   -> Brier / log loss
#   INTERVAL     a range    -> COVERAGE: did the outcome fall inside?
#   DIRECTIONAL  a side     -> direction hit or miss
#   INSUFFICIENT nothing    -> NOT_SCORED
#
# MEASURED, Brier on an interval is undefined: a [0.39, 0.73] interval around
# a true rate of 0.55 contains the realised value in 93.8% of 2,000 draws.
# Coverage is the quantity that interval claimed, so coverage is what is
# scored.
SCORE_METHOD_BRIER = "brier"
SCORE_METHOD_COVERAGE = "coverage"
SCORE_METHOD_DIRECTION = "direction"
SCORE_METHOD_NOT_SCORED = "not_scored"

CLOSURE_SCORE_METHODS: dict[str, str] = {
    "POINT": SCORE_METHOD_BRIER,
    "INTERVAL": SCORE_METHOD_COVERAGE,
    "DIRECTIONAL": SCORE_METHOD_DIRECTION,
    "INSUFFICIENT": SCORE_METHOD_NOT_SCORED,
}

# A refusal is NOT_SCORED, and that is load-bearing. Scoring it as zero error
# would reward silence; scoring it as maximum error would punish honesty.
# Neither is a measurement of anything.
CLOSURE_SCORE_REFUSALS = False

# THE SCOREBOARD IS GAMEABLE BY REFUSING, and it was measured rather than
# argued. Over 200 forecasts from a genuinely skilled forecaster (skill 0.58),
# refusing the hardest cases - those nearest 0.5 - improves the average error:
#
#   refuse   0%:  Brier 0.2206 over 200 scored
#   refuse  50%:  Brier 0.2043 over 100 scored
#   refuse  90%:  Brier 0.1379 over  19 scored
#   refuse  99%:  Brier 0.0198 over   2 scored    <- 11x "better"
#
# This system is DESIGNED to refuse often (F4 tiers, F5 stages, F7
# confidence), so the hazard is live. Every report therefore publishes the
# refusal and pending counts beside the error: an error figure must never be
# readable without the coverage it came from.
CLOSURE_REPORT_COVERAGE = True

# Below this share of scored forecasts, an error figure is reported as
# UNRELIABLE rather than as a headline. Not a round number: it is the point at
# which the measured gaming curve above has already halved the apparent error.
CLOSURE_MIN_SCORED_SHARE = 0.50

# Minimum closed forecasts before a calibration evaluation is attempted.
# Aligned with F4's measured POINT floor, so one sample-size policy governs
# both the forecast and its evaluation.
CLOSURE_MIN_FOR_CALIBRATION = 40


def _validate_closure_config() -> None:
    """Import-time guard for the L1 contract."""
    if len(set(CLOSURE_STATES)) != len(CLOSURE_STATES):
        raise ValueError("CLOSURE_STATES contains a duplicate")
    for state in (CLOSURE_OPEN, CLOSURE_MATURED, CLOSURE_CLOSED,
                  CLOSURE_EXPIRED_NO_DATA):
        if state not in CLOSURE_STATES:
            raise ValueError(f"closure state {state!r} is not declared")
    if CLOSURE_EXPIRED_NO_DATA not in CLOSURE_STATES:
        raise ValueError(
            "EXPIRED_NO_DATA must stay declared: 'we could not score it' and "
            "'it never existed' are different facts"
        )
    if CLOSURE_CLOSED in CLOSURE_CLOSEABLE_STATES:
        raise ValueError(
            "a CLOSED forecast must not be closeable again - re-closing "
            "double-counts one observation and silently re-weights every "
            "metric derived from the ledger"
        )
    if not CLOSURE_CLOSEABLE_STATES:
        raise ValueError("no state can be closed, so nothing can ever be scored")
    unknown = set(CLOSURE_CLOSEABLE_STATES) - set(CLOSURE_STATES)
    if unknown:
        raise ValueError(f"closeable states must be declared: {sorted(unknown)}")

    if set(CLOSURE_SCORE_METHODS) != {
        "POINT", "INTERVAL", "DIRECTIONAL", "INSUFFICIENT"
    }:
        raise ValueError(
            "CLOSURE_SCORE_METHODS must cover exactly the four F4 claim tiers"
        )
    if CLOSURE_SCORE_METHODS["INTERVAL"] != SCORE_METHOD_COVERAGE:
        raise ValueError(
            "an INTERVAL claim is scored by COVERAGE: Brier on a range is "
            "undefined, and coverage is the quantity the interval claimed"
        )
    if CLOSURE_SCORE_METHODS["INSUFFICIENT"] != SCORE_METHOD_NOT_SCORED:
        raise ValueError(
            "a refusal is NOT_SCORED: zero error would reward silence and "
            "maximum error would punish honesty"
        )
    if CLOSURE_SCORE_REFUSALS:
        raise ValueError(
            "refusals must not be scored - see CLOSURE_SCORE_METHODS"
        )
    if not CLOSURE_REPORT_COVERAGE:
        raise ValueError(
            "coverage must travel with every error figure. MEASURED: refusing "
            "the hardest 99% of forecasts improves Brier from 0.2206 to "
            "0.0198, so an error read without its refusal rate is meaningless"
        )
    if not 0.0 < CLOSURE_MIN_SCORED_SHARE <= 1.0:
        raise ValueError("the minimum scored share must lie in (0, 1]")
    if CLOSURE_MIN_FOR_CALIBRATION < 2:
        raise ValueError("calibration needs more than one observation")
    if not FORECAST_LEDGER_PATH.startswith("data/"):
        raise ValueError("the forecast ledger belongs under data/")


_validate_closure_config()


# ---------------------------------------------------------------------------
# Sprint L2 - Forecast performance ledger
# ---------------------------------------------------------------------------
# Performance measured by ticker, sector, regime, horizon, event type, source,
# model, confidence bucket and volatility regime.
#
# ONLY TWO OF THE NINE EXIST ON A LEDGER ROW TODAY (ticker, horizon); two more
# are derivable (sector from the ticker, a confidence bucket from the value);
# and FOUR are missing entirely: regime, event type, source and volatility
# regime.
#
# THE MISSING FOUR ARE POINT-IN-TIME FACTS AND CANNOT BE RECOVERED LATER.
# regime and volatility regime can be recomputed from bars, but the forecast
# was MADE under a specific classifier reading - recomputing later risks a
# different answer if the classifier version moved, silently re-attributing
# past performance. Event type depends on the event set at as_of, and news is
# gone after NEWS_LOOKBACK_DAYS. Source - the observed/inferred mix of the
# analogs - depends on a retrieval against a store that grows.
#
# MEASURED, comparing retrieval against half the store versus the full store
# across 8 probes: 7 of 8 returned a DIFFERENT analog count (14->19, 14->20,
# 16->24, 0->1, 3->4, 9->16, 0->1). So "what evidence did this forecast rest
# on?" is answerable only if it was written down when the forecast was made.
# L2 captures these AT RECORD TIME.
#
# THE DECIDING MEASUREMENT: nine dimensions cut the data to nothing.
#
#   ticker 77 | sector 11 | regime 5 | horizon 4 | event_type 10
#   source 2  | model 1   | confidence_bucket 4  | volatility_regime 3
#
#   full cross-product : 4,065,600 cells
#   forecasts per year :        77,616
#   average per cell   :        0.0191
#
# A full cross-tab is EMPTY ALMOST EVERYWHERE. So a breakdown is MARGINAL -
# one dimension at a time - and every cell faces the same F4 floors the
# forecasts themselves face. A performance number from 3 forecasts is the F4
# stress cell wearing an analytics hat.
PERFORMANCE_LEDGER_VERSION = "performance-ledger-v1"
PERFORMANCE_CONTRACT_VERSION = "performance-contract-v1"

# The nine dimensions, in reading order.
PERF_TICKER = "ticker"
PERF_SECTOR = "sector"
PERF_REGIME = "regime"
PERF_HORIZON = "horizon"
PERF_EVENT_TYPE = "event_type"
PERF_SOURCE = "source"
PERF_MODEL = "model"
PERF_CONFIDENCE_BUCKET = "confidence_bucket"
PERF_VOLATILITY_REGIME = "volatility_regime"

PERFORMANCE_DIMENSIONS: tuple[str, ...] = (
    PERF_TICKER,
    PERF_SECTOR,
    PERF_REGIME,
    PERF_HORIZON,
    PERF_EVENT_TYPE,
    PERF_SOURCE,
    PERF_MODEL,
    PERF_CONFIDENCE_BUCKET,
    PERF_VOLATILITY_REGIME,
)

# Dimensions that MUST be written onto the ledger row when the forecast is
# made, because they cannot be reconstructed afterwards. Declared as data so a
# future reader can see which facts are perishable and why.
PERFORMANCE_CAPTURED_AT_RECORD: tuple[str, ...] = (
    PERF_REGIME,
    PERF_EVENT_TYPE,
    PERF_SOURCE,
    PERF_VOLATILITY_REGIME,
)

# Dimensions derived from what the row already carries. Deriving is safe here
# because the inputs are immutable: a ticker's sector mapping and a recorded
# confidence value do not change retroactively.
PERFORMANCE_DERIVED: tuple[str, ...] = (PERF_SECTOR, PERF_CONFIDENCE_BUCKET)

# Breakdowns are MARGINAL: one dimension at a time. A full cross-product is
# 4,065,600 cells against 77,616 forecasts a year. Setting this True would
# produce a table that is empty almost everywhere while looking thorough.
PERFORMANCE_CROSS_TABULATE = False

PERFORMANCE_SPARSITY_EVIDENCE = (
    "MEASURED: the nine dimensions cross-multiply to 4,065,600 cells against "
    "77,616 forecasts a year - 0.0191 per cell. F4's floors are 8 for an "
    "interval and 40 for a point estimate, so a full cross-tab is empty "
    "almost everywhere while appearing thorough."
)

# Confidence buckets, from F7's reading bands. Reused rather than re-cut, so a
# bucket boundary means the same thing in the forecast and in its post-mortem.
PERFORMANCE_CONFIDENCE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("NONE", 0.0),
    ("LOW", 0.25),
    ("MODERATE", 0.50),
    ("HIGH", 0.75),
)

# Volatility bands for the volatility-regime dimension, on annualized vol.
# Boundaries drawn from the E6 similarity scale's working range rather than
# invented: 0.15 and 0.35 separate calm, normal and turbulent.
PERFORMANCE_VOLATILITY_BANDS: tuple[tuple[str, float], ...] = (
    ("CALM", 0.0),
    ("NORMAL", 0.15),
    ("TURBULENT", 0.35),
)

# A cell below this many SCORED forecasts reports its count and refuses a
# metric. Aligned with F4's INTERVAL floor so one sample-size policy governs
# the forecast and its evaluation alike.
PERFORMANCE_MIN_CELL = 8

# Every breakdown carries L1's coverage rule. Refusing remains the cheapest
# way to look accurate, and a per-cell error is no more readable without its
# refusal count than a global one is.
PERFORMANCE_REPORT_COVERAGE = True


def _validate_performance_config() -> None:
    """Import-time guard for the L2 contract."""
    if len(set(PERFORMANCE_DIMENSIONS)) != len(PERFORMANCE_DIMENSIONS):
        raise ValueError("PERFORMANCE_DIMENSIONS contains a duplicate")
    if len(PERFORMANCE_DIMENSIONS) != 9:
        raise ValueError(
            "the sprint names nine dimensions; dropping one silently narrows "
            "what performance is measured across"
        )
    for group, label in (
        (PERFORMANCE_CAPTURED_AT_RECORD, "PERFORMANCE_CAPTURED_AT_RECORD"),
        (PERFORMANCE_DERIVED, "PERFORMANCE_DERIVED"),
    ):
        unknown = set(group) - set(PERFORMANCE_DIMENSIONS)
        if unknown:
            raise ValueError(f"{label} names undeclared dimensions: {sorted(unknown)}")
    overlap = set(PERFORMANCE_CAPTURED_AT_RECORD) & set(PERFORMANCE_DERIVED)
    if overlap:
        raise ValueError(
            f"a dimension cannot be both captured and derived: {sorted(overlap)}"
        )
    for name in (PERF_REGIME, PERF_EVENT_TYPE, PERF_SOURCE, PERF_VOLATILITY_REGIME):
        if name not in PERFORMANCE_CAPTURED_AT_RECORD:
            raise ValueError(
                f"{name!r} must be captured at record time: it is a "
                f"point-in-time fact, and MEASURED, 7 of 8 retrieval probes "
                f"returned a different analog set as the store grew"
            )

    if PERFORMANCE_CROSS_TABULATE:
        raise ValueError(
            "breakdowns must stay MARGINAL. " + PERFORMANCE_SPARSITY_EVIDENCE
        )
    if "MEASURED" not in PERFORMANCE_SPARSITY_EVIDENCE:
        raise ValueError(
            "the sparsity evidence must carry its measurement, or a reader "
            "cannot tell a finding from a preference"
        )

    for bands, label in (
        (PERFORMANCE_CONFIDENCE_BUCKETS, "PERFORMANCE_CONFIDENCE_BUCKETS"),
        (PERFORMANCE_VOLATILITY_BANDS, "PERFORMANCE_VOLATILITY_BANDS"),
    ):
        values = [value for _name, value in bands]
        if values != sorted(values):
            raise ValueError(f"{label} must ascend")
        if values[0] != 0.0:
            raise ValueError(f"{label} must start at 0.0 so every value lands")
        if len({name for name, _ in bands}) != len(bands):
            raise ValueError(f"{label} contains a duplicate name")

    if PERFORMANCE_MIN_CELL < 2:
        raise ValueError("a cell needs more than one observation to mean anything")
    if not PERFORMANCE_REPORT_COVERAGE:
        raise ValueError(
            "coverage must travel with every per-cell error, for the same "
            "reason it travels with the global one: refusing is the cheapest "
            "way to look accurate"
        )


_validate_performance_config()


# ---------------------------------------------------------------------------
# Sprint L3 - Error memory
# ---------------------------------------------------------------------------
# "Store forecast, actual, error, context; identify systematic errors."
#
# THE FIRST CLAUSE IS ALREADY DONE. A closed L1 ledger row carries the
# forecast (value, claim), the actual (forward_return, direction_up), the
# error (brier, absolute_error) and the context (regime, event_type,
# volatility, observed_share). L3 does not duplicate that store (W5); it reads
# it. The real work is the second clause.
#
# A SYSTEMATIC ERROR IS A BIAS, NOT A LARGE ERROR. MEASURED over 200
# forecasts:
#
#   case                          brier    bias      t
#   noisy but unbiased           0.2466  -0.0459   -1.3
#   systematically over-bullish  0.3300  +0.2850   +8.1
#
# Brier barely separates them; the SIGNED BIAS does. Systematic means a
# direction that persists, not an error that is big.
#
# THE WORST CELL IN A TABLE IS NOT A SYSTEMATIC ERROR. L2 reports roughly 36
# cells. MEASURED with NO real effect anywhere, every cell drawn from
# identical skill: the true mean Brier is 0.2722 while the WORST cell per
# report averages 0.3712 - it looks 36% worse than the truth, every time.
# Pointing at it would manufacture a finding on every run.
#
# THE DECIDING MEASUREMENT: SCANNING MANY CELLS MANUFACTURES FINDINGS. With no
# systematic error present anywhere:
#
#   t>1.96 (p<0.05)    2.77 false alarms/report, 94% of reports flag something
#   t>2.58 (p<0.01)    1.15 false alarms/report, 68% flag something
#   t>3.29 (p<0.001)   0.39 false alarms/report, 34% flag something
#
# At the conventional p<0.05 a PERFECTLY CLEAN system reports ~2.8 systematic
# errors on 94% of runs, which trains an operator to ignore the alert.
ERROR_MEMORY_VERSION = "error-memory-v1"
ERROR_MEMORY_CONTRACT_VERSION = "error-memory-contract-v1"

# The t-statistic a bias must clear to be called systematic. Strict BECAUSE
# many cells are scanned - it is a multiple-comparison threshold, not a
# statement about any single cell.
ERROR_MEMORY_T_THRESHOLD = 3.29

# Minimum scored forecasts in a cell before bias is even tested.
#
# THRESHOLD AND FLOOR WERE CHOSEN TOGETHER, and a first pass got this wrong.
# MEASURED, detection of a real 0.25 bias:
#
#       n   t>1.96   t>2.58   t>3.29
#      12      40%      19%       7%
#      30      80%      56%      31%
#     100     100%      99%      95%
#
# Detection FALLS as the threshold rises, so no threshold rescues a thin cell:
# the sample floor buys detection, the threshold buys silence. At n=30 with
# t>3.29 the pair gives 0.26 false alarms per report (78% of clean reports
# silent) and 31% detection, rising to 95% by n=100 as the ledger fills.
ERROR_MEMORY_MIN_SAMPLES = 30

# Bias smaller than this is not worth reporting even when it clears the
# t-test: with enough observations a trivial bias becomes "significant"
# without becoming important. 0.05 on a probability is one twentieth of the
# scale.
ERROR_MEMORY_MIN_BIAS = 0.05

# Detection power is REPORTED, not assumed. MEASURED at the shipped pair, a
# real 0.25 bias is found ~31% of the time at n=30. So the absence of a
# finding is NOT evidence of no bias, and every report says so.
ERROR_MEMORY_REPORT_POWER = True

ERROR_MEMORY_POWER_EVIDENCE = (
    "MEASURED at t>3.29 with a floor of 30: a real 0.25 bias is detected 31% "
    "of the time at n=30, 58% at n=50 and 95% at n=100. The absence of a "
    "finding is therefore not evidence of no bias - it is most often a "
    "statement about sample size."
)

# Findings, weakest first. The order IS the precedence: a cell reports the
# strongest verdict its evidence supports.
ERROR_VERDICT_NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"   # below the sample floor
ERROR_VERDICT_NO_BIAS_DETECTED = "NO_BIAS_DETECTED"  # tested, nothing found
ERROR_VERDICT_NEGLIGIBLE = "NEGLIGIBLE"              # significant but tiny
ERROR_VERDICT_SYSTEMATIC = "SYSTEMATIC"              # persistent and material

ERROR_MEMORY_VERDICTS: tuple[str, ...] = (
    ERROR_VERDICT_NOT_ENOUGH_DATA,
    ERROR_VERDICT_NO_BIAS_DETECTED,
    ERROR_VERDICT_NEGLIGIBLE,
    ERROR_VERDICT_SYSTEMATIC,
)

# The direction a systematic bias runs in. Named rather than signed, because
# "+0.25" means nothing without knowing which way is which.
ERROR_DIRECTION_OVERCONFIDENT = "OVER_PREDICTS"   # forecast above the outcome
ERROR_DIRECTION_UNDERCONFIDENT = "UNDER_PREDICTS"  # forecast below the outcome

# Every finding carries the number of cells that were scanned to produce it.
# 36 cells at p<0.05 yields 2.8 false alarms per report; a reader cannot weigh
# a finding without knowing how many chances it had to appear.
ERROR_MEMORY_REPORT_COMPARISONS = True


def _validate_error_memory_config() -> None:
    """Import-time guard for the L3 contract."""
    if len(set(ERROR_MEMORY_VERDICTS)) != len(ERROR_MEMORY_VERDICTS):
        raise ValueError("ERROR_MEMORY_VERDICTS contains a duplicate")
    if ERROR_MEMORY_VERDICTS[0] != ERROR_VERDICT_NOT_ENOUGH_DATA:
        raise ValueError(
            "NOT_ENOUGH_DATA must be the weakest verdict - it is where every "
            "thin cell starts"
        )
    if ERROR_MEMORY_VERDICTS[-1] != ERROR_VERDICT_SYSTEMATIC:
        raise ValueError("SYSTEMATIC must be the strongest verdict")
    if ERROR_VERDICT_NOT_ENOUGH_DATA == ERROR_VERDICT_NO_BIAS_DETECTED:
        raise ValueError(
            "'we could not test' and 'we tested and found nothing' are "
            "different facts and must stay distinct"
        )

    if ERROR_MEMORY_T_THRESHOLD < 2.58:
        raise ValueError(
            f"a t-threshold of {ERROR_MEMORY_T_THRESHOLD} is too loose for a "
            f"scan of ~36 cells. MEASURED with no real effect anywhere: "
            f"t>1.96 produces 2.77 false alarms per report and flags something "
            f"on 94% of runs, which trains an operator to ignore the alert"
        )
    if ERROR_MEMORY_MIN_SAMPLES < 30:
        raise ValueError(
            f"a floor of {ERROR_MEMORY_MIN_SAMPLES} cannot support a bias "
            f"test. MEASURED, detection of a real 0.25 bias at t>3.29 is 7% "
            f"at n=12 and 31% at n=30 - below 30 the test is close to blind"
        )
    if not 0.0 < ERROR_MEMORY_MIN_BIAS < 1.0:
        raise ValueError("the minimum reportable bias must lie inside (0, 1)")
    if not ERROR_MEMORY_REPORT_POWER:
        raise ValueError(
            "detection power must be reported. " + ERROR_MEMORY_POWER_EVIDENCE
        )
    if "MEASURED" not in ERROR_MEMORY_POWER_EVIDENCE:
        raise ValueError(
            "the power evidence must carry its measurement, or a reader "
            "cannot tell a finding from a preference"
        )
    if not ERROR_MEMORY_REPORT_COMPARISONS:
        raise ValueError(
            "the comparison count must travel with every finding: 36 cells at "
            "p<0.05 yields 2.8 false alarms per report, and a reader cannot "
            "weigh a finding without knowing how many chances it had"
        )


_validate_error_memory_config()


# ---------------------------------------------------------------------------
# Sprint L - Live chart state (the blocker before scheduling)
# ---------------------------------------------------------------------------
# `scripts/run_forecasts.py` derived each ticker's chart state from that
# ticker's MOST RECENT EVENT MEMORY. That is correct for a backfill, where the
# memory IS the point in time being forecast from. For a LIVE daily run it is
# wrong: the forecast gets anchored to whenever that ticker last happened to
# produce a memory.
#
# MEASURED across the 73 tickers carrying a memory, against 2026-09-21:
#
#     staleness   min 3 days, median 144, mean 166, max 535
#
# So the median live forecast would have been conditioned on a chart roughly
# five months old, and VOO's on one from 2025-04-04.
#
# THE DECIDING MEASUREMENT IS NOT THE AGE, IT IS WHAT THE AGE RETRIEVES. The
# chart state is the retrieval key for analogs, so a stale key does not shift
# the forecast slightly - it looks up a different history:
#
#   tkr    memory as_of   close then   close now   regime then  regime now
#   VOO    2025-04-04         465.52      701.78   bearish      bullish
#   CIBR   2025-04-07          57.71       99.87   bearish      bullish
#   CAT    2025-10-29         585.49      808.99   bullish      bullish
#   AAPL   2026-09-18         337.00      336.13   bullish      bullish
#
# The retrieved analog SETS overlap by a mean Jaccard of 0.205 - four fifths of
# the evidence differs - and the stale key called the regime BEARISH for VOO
# and CIBR while both are in fact bullish. VOO retrieved 22 analogs from its
# April-2025 chart and 6 from today's. A forecast built that way is not a
# stale-but-reasonable forecast; it answers a question about a different market.
#
# So a live run MUST build today's state from today's bars. The builder is
# canonical in `core.chart_features` (W5): the backfill calls it at a historical
# position, the live run calls it at the last bar, and there is exactly one
# implementation of what a chart state IS.
LIVE_CHART_STATE_VERSION = "live-chart-state-v1"

# Bars required before a chart state can be built at all. The state carries a
# 200-session moving average, so below this the longest feature is computed
# from a window that does not exist and `price_vs_ma_200` silently becomes a
# mean over whatever happens to be present.
LIVE_CHART_MIN_HISTORY = 210

# How stale a chart state may be, in calendar days, before a LIVE run refuses
# to forecast from it. One week: long enough to survive a holiday week or a
# fetch outage, short enough that the retrieval key still describes the chart
# the forecast is about. MEASURED, the memory-derived path exceeded this for 71
# of 73 tickers.
LIVE_CHART_MAX_STALENESS_DAYS = 7

# A live run REFUSES rather than falling back to the stale memory state. A
# fallback is the failure this block exists to prevent: it would restore
# exactly the behaviour above while reporting success, and the refusal is
# recorded in the ledger where L1 can count it.
LIVE_CHART_FALLBACK_TO_MEMORY = False


def _validate_live_chart_config() -> None:
    """Import-time guard for the live chart-state contract."""
    if LIVE_CHART_MIN_HISTORY < 210:
        raise ValueError(
            f"a chart state needs at least 210 bars, not "
            f"{LIVE_CHART_MIN_HISTORY}: it carries a 200-session moving "
            f"average, and a shorter window turns price_vs_ma_200 into a mean "
            f"over whatever bars happen to exist"
        )
    if LIVE_CHART_MAX_STALENESS_DAYS < 1:
        raise ValueError("the staleness bound must allow at least one day")
    if LIVE_CHART_MAX_STALENESS_DAYS > 30:
        raise ValueError(
            f"a staleness bound of {LIVE_CHART_MAX_STALENESS_DAYS} days is not "
            f"a live run. MEASURED, chart states a median 144 days old "
            f"retrieved analog sets overlapping the live ones by a Jaccard of "
            f"0.205, and called the regime bearish for VOO and CIBR while both "
            f"were bullish"
        )
    if LIVE_CHART_FALLBACK_TO_MEMORY:
        raise ValueError(
            "a live run must not fall back to the most recent memory's chart "
            "state: that is the stale-anchor bug itself, and falling back "
            "would reinstate it while reporting success. Refuse instead - the "
            "refusal is recorded and counted"
        )


_validate_live_chart_config()


# ---------------------------------------------------------------------------
# Sprint L4 - Source reliability learning
# ---------------------------------------------------------------------------
# "Source quality may depend on source x event type x sector x horizon - one
# global source score is not assumed sufficient."
#
# THE TASK STATES A HYPOTHESIS, SO THE FIRST JOB IS TO TEST IT RATHER THAN
# IMPLEMENT IT. Two things were measured before any scorer was written.
#
# FIRST: WHAT EXISTS TODAY IS NOT LEARNED AT ALL. `fetch_data.SOURCE_REGISTRY`
# carries ONE asserted `base_confidence` per DOMAIN - news 0.75, fundamentals
# 0.90, market data 0.80 - with no measurement behind any of them, and nothing
# ever updates them from outcomes. The news schema already carries a
# `source_quality` field: MEASURED over 2650 stored articles it is populated
# ZERO times. So the gap is real and it is the whole task.
#
# SECOND, AND DECISIVE: CONDITIONING IS NOT FREE. Estimating a rate per cell
# costs variance, and MEASURED against a simulated source whose TRUE accuracy
# really does vary by cell, per-cell estimation is FIVE TIMES WORSE than a
# single global rate when evidence is thin:
#
#   true quality varies by cell (sd=0.10)   global MSE   per-cell MSE
#     5 observations per cell                  0.01224        0.05935
#    25 observations per cell                  0.01000        0.00950
#   100 observations per cell                  0.00936        0.00234
#
# Per-cell conditioning only starts to pay from ~25 observations per cell, and
# when quality does NOT in fact vary it is worse everywhere. So "one global
# score is not sufficient" and "condition on everything" are both wrong, and
# the honest answer is that THE EVIDENCE DECIDES, PER CELL.
#
# THE SHIPPED ESTIMATOR IS SHRINKAGE: the rate of a cell is pulled toward the
# global rate by k pseudo-counts, so it IS the global score when a cell is
# empty and becomes the score of that cell as evidence accumulates. MEASURED,
# it never loses badly in either world:
#
#   quality REALLY varies      global    per-cell    shrunk
#     5 per cell              0.01224     0.05935   0.01064
#    25 per cell              0.01000     0.00950   0.00505
#   100 per cell              0.00936     0.00234   0.00192
#
# WHAT THIS CANNOT DO YET, AND WHY IT SHIPS ANYWAY. The scheme the task
# describes is 50 sources x 9 event types x 11 sectors x 4 horizons = 19,800
# cells, which at the ~380 outcomes needed to separate a 0.65 source from a
# 0.55 one would require ~7.5 MILLION source-linked outcomes. MEASURED today:
# ZERO. Not few - zero, structurally, because no outcome in the system is
# joined to an outlet (see SOURCE_RELIABILITY_JOIN_GAP below). Shrinkage is
# exactly the estimator that degrades to the global score under that
# condition, which is why it can ship before the join exists.
SOURCE_RELIABILITY_VERSION = "source-reliability-v1"

# Pseudo-counts pulling a cell toward the global rate.
#
# CHOSEN BY WORST CASE, NOT BY MEAN. MEASURED across quality-variation regimes
# x sample sizes, k=5 minimised the worst case overall but ONLY because of a
# regime with sd=0.20 - sources ranging from 0.2 to 0.95 accuracy - which no
# outlet population plausibly shows. Over the plausible range (sd <= 0.10):
#
#     k      worst-case MSE
#     5             0.01503
#    15             0.01079
#    20             0.01013   <- minimum
#    25             0.01075
#    40             0.01054
#
# 15-30 is a flat basin, so this is not a knife edge.
SOURCE_RELIABILITY_SHRINKAGE_K = 20

# Observations in a cell before its OWN rate is reported as distinguishable
# from the global one. MEASURED, per-cell estimation first beats global at ~25
# observations per cell; below that the shrunk estimate is still used but the
# cell is reported as GLOBAL-backed, because a reader must not read a
# shrinkage-dominated number as a learned source property.
SOURCE_RELIABILITY_MIN_CELL = 25

# The dimensions a source score MAY condition on, in the order of the task.
# Listed rather than assumed: a dimension is only USED where a cell has the
# evidence for it, and the estimator falls back along this order.
SOURCE_RELIABILITY_DIMENSIONS: tuple[str, ...] = (
    "source", "event_type", "sector", "horizon",
)

# How a reported score was actually backed, weakest to strongest. The order IS
# the precedence.
SOURCE_BACKING_NONE = "NO_EVIDENCE"      # nothing observed anywhere
SOURCE_BACKING_PRIOR = "REGISTRY_PRIOR"  # the asserted base_confidence only
SOURCE_BACKING_GLOBAL = "GLOBAL"         # the overall rate of that source
SOURCE_BACKING_CELL = "CONDITIONAL"      # this cell has its own evidence

SOURCE_RELIABILITY_BACKINGS: tuple[str, ...] = (
    SOURCE_BACKING_NONE,
    SOURCE_BACKING_PRIOR,
    SOURCE_BACKING_GLOBAL,
    SOURCE_BACKING_CELL,
)

# THE JOIN GAP, RECORDED AS DATA. No outcome in the system is currently
# attributable to an OUTLET. MEASURED 2026-09-21:
#   - 2650 stored articles carry 50 distinct `source_name` values
#   - 0 of them carry a ticker, so no article joins to a price outcome
#   - 2084 event memories carry NO source name at all (1954 are inferred from
#     volume cadence and have no source by construction)
#   - `Event.source` is set to the PROVIDER ("newsapi_news"), not the outlet
# Until an outlet reaches an outcome, every cell is NO_EVIDENCE and every score
# is the registry prior. That is reported, never silently rendered as 0.0.
SOURCE_RELIABILITY_JOIN_GAP = (
    "MEASURED 2026-09-21: 0 of 2650 stored articles carry a ticker and 0 of "
    "2084 event memories carry an outlet, so no source-linked outcome exists "
    "yet. Event.source records the provider, not the outlet. Scores are "
    "registry priors until the join is built."
)

# A source is never scored 0.0 for absence of evidence. An unmeasured outlet
# and an outlet measured to be useless are different facts, and collapsing
# them would silently discard every new source the moment it appeared.
SOURCE_RELIABILITY_ABSENT_IS_ZERO = False


def _validate_source_reliability_config() -> None:
    """Import-time guard for the L4 contract."""
    if SOURCE_RELIABILITY_SHRINKAGE_K < 1:
        raise ValueError(
            "shrinkage k must be at least 1: k=0 is per-cell estimation, "
            "MEASURED five times worse than a global rate at 5 observations "
            "per cell even when quality genuinely varies"
        )
    if SOURCE_RELIABILITY_SHRINKAGE_K > 100:
        raise ValueError(
            f"a k of {SOURCE_RELIABILITY_SHRINKAGE_K} pins every cell to the "
            f"global rate, which is the 'one global score' the task refuses to "
            f"assume is sufficient"
        )
    if SOURCE_RELIABILITY_MIN_CELL < 25:
        raise ValueError(
            f"a cell floor of {SOURCE_RELIABILITY_MIN_CELL} reports a "
            f"shrinkage-dominated number as a learned source property. "
            f"MEASURED, per-cell estimation first beats global at ~25 "
            f"observations per cell"
        )
    if SOURCE_RELIABILITY_DIMENSIONS[0] != "source":
        raise ValueError(
            "the source itself must be the first dimension - every other one "
            "conditions it"
        )
    if len(set(SOURCE_RELIABILITY_DIMENSIONS)) != len(SOURCE_RELIABILITY_DIMENSIONS):
        raise ValueError("SOURCE_RELIABILITY_DIMENSIONS contains a duplicate")
    if SOURCE_RELIABILITY_BACKINGS[0] != SOURCE_BACKING_NONE:
        raise ValueError("NO_EVIDENCE must be the weakest backing")
    if SOURCE_RELIABILITY_BACKINGS[-1] != SOURCE_BACKING_CELL:
        raise ValueError("CONDITIONAL must be the strongest backing")
    if SOURCE_BACKING_PRIOR == SOURCE_BACKING_GLOBAL:
        raise ValueError(
            "an asserted registry prior and a measured global rate are "
            "different facts and must stay distinct"
        )
    if SOURCE_RELIABILITY_ABSENT_IS_ZERO:
        raise ValueError(
            "an unmeasured source must not score 0.0: that is the same number "
            "as a source measured to be useless, and it would discard every "
            "new outlet the moment it appeared. " + SOURCE_RELIABILITY_JOIN_GAP
        )
    if "MEASURED" not in SOURCE_RELIABILITY_JOIN_GAP:
        raise ValueError(
            "the join gap must carry its measurement, or a reader cannot tell "
            "a missing capability from a finding"
        )


_validate_source_reliability_config()

# The noise band a conditioning claim must clear, as a multiplier on the mean
# standard error across cells.
#
# THE BAND TRACKS THE TAIL OF THE NULL, NOT ITS MEAN. A first version used a
# flat 2.0, which sits near the MEAN of the null range, so MEASURED it claimed
# "quality depends on event_type" on 7 of 30 runs with NO effect present.
#
# MEASURED, the multiplier placing the band at the 95th percentile of the null
# range grows with the number of cells - more cells, more chances for a wide
# spread by luck, the same multiple-comparison effect L3 measured:
#
#     cells    multiplier needed
#       2                   1.90
#       3                   2.35
#       5                   2.70
#       9                   3.11
#
# Fitted c = 1.4 + 0.8*ln(cells).
CONDITIONING_NOISE_INTERCEPT = 1.4
CONDITIONING_NOISE_LOG_SLOPE = 0.8


def _validate_conditioning_noise_config() -> None:
    if CONDITIONING_NOISE_LOG_SLOPE <= 0:
        raise ValueError(
            "the noise band must GROW with the number of cells: more cells "
            "means more chances for a wide spread by luck, and a flat band "
            "MEASURED 7 false conditioning claims in 30 clean runs"
        )
    if CONDITIONING_NOISE_INTERCEPT < 1.0:
        raise ValueError(
            "a band below one standard error is inside the noise by "
            "construction"
        )


_validate_conditioning_noise_config()


# ---------------------------------------------------------------------------
# Sprint L5 - Controlled incremental learning
# ---------------------------------------------------------------------------
# "New data -> candidate update -> shadow evaluation -> drift testing -> OOS
# validation -> promotion gate -> human approval -> new champion. Never
# auto-replace the production champion daily."
#
# MOST OF THIS CHAIN ALREADY EXISTS, and L5 must not rebuild it (W5):
#
#   new data            scripts/daily_collect.py            L-sprint
#   candidate update    core.training.train_baseline        M-sprint
#   shadow evaluation   M7 roles, SHADOW_MIN_OBSERVATIONS   M-sprint
#   drift testing       M5 PROMO_CHECK_DRIFT (PSI)          M-sprint
#   OOS validation      M5 PROMO_CHECK_OOS                  M-sprint
#   promotion gate      core.promotion.evaluate_promotion   M-sprint
#   human approval      M5 PROMO_CHECK_APPROVAL             M-sprint
#   new champion        registry.crown_champion             M7
#
# TWO THINGS WERE MISSING, and they are what L5 adds.
#
# MISSING 1: THE LAST SENTENCE WAS UNENFORCED. "Never auto-replace the
# production champion daily" had NO implementation anywhere - MEASURED, the
# words cooldown/last_promoted/min_days/interval/elapsed/cadence appear ZERO
# times across core/promotion.py and core/model_registry.py. `crown_champion`
# checks role, status and approval but never compares `changed_at` to the
# outgoing champion's tenure. DEMONSTRATED: four champions crowned inside
# FIFTEEN MINUTES, every crowning accepted.
#
# WHY THAT MATTERS, MEASURED. Two models with IDENTICAL true skill: the
# challenger looks better ~48% of the time at ANY sample size (43.6% at n=20,
# 49.5% at n=1000). A rule of "promote whatever is better on the evidence so
# far" therefore churns the champion roughly every other evaluation FOREVER,
# on pure noise. Meanwhile a REAL 3-point edge is detected only 64% of the
# time at n=100 - so evidence, not a threshold, is what separates them.
#
# THE COST OF CHURN, SIMULATED over three years with a candidate appearing
# daily and a genuinely better (+4pt) model arriving on day 360:
#
#     min days   noise churn/3yr   mean adopt delay   median
#            0             433.5                1.9      1.0
#            7             127.2                7.8      5.5
#           14              70.0               16.7     12.0
#           30              34.8               35.6     19.0
#           60              18.0               47.7     11.0
#           90              12.0               64.5      8.0
#          180               6.0              150.2      6.0
#
# Churn collapses 433 -> 35 (92%) by 30 days and then FLATTENS, while the
# delay in adopting a real winner climbs steeply past 60. 30 days is the knee:
# both costs are acceptable and neither dominates.
CHAMPION_TENURE_VERSION = "champion-tenure-v1"

# The minimum days a champion must serve before it may be replaced.
#
# This is the implementation of "never auto-replace the production champion
# daily" - the clause had no code behind it before L5.
CHAMPION_MIN_TENURE_DAYS = 30

# A tenure bound must never block a SAFETY withdrawal. Retiring a champion
# that is failing is not the churn this guard exists to prevent, and a guard
# that trapped a broken model in production would be worse than no guard.
CHAMPION_TENURE_ALLOWS_ROLLBACK = True

# The bound is on REPLACEMENT, measured from the outgoing champion's crowning.
# Measuring from the candidate's readiness instead would let a queue of
# candidates replace the champion in sequence, each "ready" on a different day.
CHAMPION_TENURE_MEASURED_FROM = "incumbent_crowned_at"

# L5 STAGES, in order. The chain is declared as data so the orchestrator
# cannot silently skip one: every stage must reach a verdict, and a stage that
# could not run blocks exactly as a failure does (inherited from M5's
# NOT_EVALUATED).
L5_STAGE_DATA = "new_data"
L5_STAGE_CANDIDATE = "candidate_update"
L5_STAGE_SHADOW = "shadow_evaluation"
L5_STAGE_DRIFT = "drift_testing"
L5_STAGE_OOS = "oos_validation"
L5_STAGE_GATE = "promotion_gate"
L5_STAGE_APPROVAL = "human_approval"
L5_STAGE_TENURE = "champion_tenure"
L5_STAGE_CHAMPION = "new_champion"

L5_STAGES: tuple[str, ...] = (
    L5_STAGE_DATA,
    L5_STAGE_CANDIDATE,
    L5_STAGE_SHADOW,
    L5_STAGE_DRIFT,
    L5_STAGE_OOS,
    L5_STAGE_GATE,
    L5_STAGE_APPROVAL,
    L5_STAGE_TENURE,
    L5_STAGE_CHAMPION,
)

# Stages L5 OWNS rather than delegates. Everything else is called through to
# its existing owner (W5), and the orchestrator holds no second copy of the
# logic.
L5_OWNED_STAGES: tuple[str, ...] = (L5_STAGE_TENURE,)

# MISSING 2: nothing ran the chain end to end, so no single call could answer
# "may this candidate become champion today, and if not, which stage stopped
# it". Each stage was individually correct and collectively unsequenced.
L5_PIPELINE_VERSION = "controlled-learning-v1"

# Promotion is never automatic, whatever the evidence says. This is a separate
# statement from the approval CHECK: the check verifies an approver was
# recorded, while this forbids the pipeline from supplying one itself.
L5_AUTO_PROMOTE = False


def _validate_controlled_learning_config() -> None:
    """Import-time guard for the L5 contract."""
    if CHAMPION_MIN_TENURE_DAYS < 2:
        raise ValueError(
            f"a tenure bound of {CHAMPION_MIN_TENURE_DAYS} day(s) permits "
            f"daily replacement, which is exactly what the sprint forbids. "
            f"MEASURED, two models of IDENTICAL skill trade places ~48% of "
            f"evaluations at any sample size, so an unbounded rule churns the "
            f"champion on pure noise - 433 replacements per three years"
        )
    if CHAMPION_MIN_TENURE_DAYS > 180:
        raise ValueError(
            f"a tenure bound of {CHAMPION_MIN_TENURE_DAYS} days delays a "
            f"genuinely better model beyond usefulness. MEASURED, mean "
            f"adoption delay for a real +4pt model rises to 150 days at a "
            f"180-day bound while churn barely improves on 30 days"
        )
    if not CHAMPION_TENURE_ALLOWS_ROLLBACK:
        raise ValueError(
            "the tenure bound must never block a safety withdrawal: trapping "
            "a failing champion in production would be worse than no guard"
        )
    if CHAMPION_TENURE_MEASURED_FROM != "incumbent_crowned_at":
        raise ValueError(
            "tenure must be measured from the OUTGOING champion's crowning; "
            "measuring from candidate readiness lets a queue of candidates "
            "replace the champion in sequence, each ready on a different day"
        )

    if len(set(L5_STAGES)) != len(L5_STAGES):
        raise ValueError("L5_STAGES contains a duplicate")
    if L5_STAGES[0] != L5_STAGE_DATA:
        raise ValueError("the chain begins with new data")
    if L5_STAGES[-1] != L5_STAGE_CHAMPION:
        raise ValueError("the chain ends with a new champion")
    if L5_STAGES.index(L5_STAGE_TENURE) >= L5_STAGES.index(L5_STAGE_CHAMPION):
        raise ValueError(
            "the tenure check must run BEFORE a champion is crowned - after "
            "is not a guard, it is a report"
        )
    if L5_STAGES.index(L5_STAGE_APPROVAL) >= L5_STAGES.index(L5_STAGE_CHAMPION):
        raise ValueError("human approval must precede the new champion")
    if L5_STAGES.index(L5_STAGE_SHADOW) >= L5_STAGES.index(L5_STAGE_OOS):
        raise ValueError(
            "shadow evaluation precedes OOS validation - a model that has not "
            "run in shadow has no out-of-sample record to validate"
        )
    unknown = set(L5_OWNED_STAGES) - set(L5_STAGES)
    if unknown:
        raise ValueError(f"L5_OWNED_STAGES names unknown stages: {sorted(unknown)}")
    if L5_STAGE_GATE in L5_OWNED_STAGES:
        raise ValueError(
            "L5 must DELEGATE the promotion gate to core.promotion (W5), not "
            "own a second copy of it"
        )
    if L5_AUTO_PROMOTE:
        raise ValueError(
            "promotion must never be automatic: the sprint requires human "
            "approval, and a pipeline that supplies its own approver has "
            "removed the only stage a machine cannot satisfy"
        )


_validate_controlled_learning_config()


# ---------------------------------------------------------------------------
# Sprint L6 - Concept drift detection
# ---------------------------------------------------------------------------
# "Feature distribution drift, relationship drift, calibration drift,
# event-response drift."
#
# THE TASK NAMES FOUR THINGS, AND THEY REALLY ARE FOUR. MEASURED over 600
# observations per scenario, each breaking exactly ONE thing, scored by the
# feature-distribution detector alone:
#
#   scenario                 feature PSI   caught?
#   1 feature drift               1.2197   YES
#   2 relationship drift          0.0360   no
#   3 calibration drift           0.0133   no
#   4 event-response drift        0.0274   no
#
# Feature PSI catches ONE of the four. The other three are invisible to it
# because THE FEATURES DID NOT MOVE - the world did. Verified each needs its
# own detector:
#
#   relationship: corr(x,y) +0.368 -> -0.315 while the features are identical
#   calibration : ranking intact (corr 0.448 -> 0.430) while the calibration
#                 gap goes 0.000 -> +0.251, so a relationship detector is blind
#   response    : mean event response +0.0404 -> +0.0001 with features,
#                 relationship AND calibration all unchanged
#
# FOUR SEPARATE QUESTIONS. No single detector answers them, which is why this
# block defines four and refuses to collapse them.
#
# M5's `score_drift_psi` IS NOT REUSABLE HERE, and that is a measurement, not
# a preference. It bins on a fixed [0, 10] score scale and REFUSES anything
# outside it. MEASURED on a feature living in [-0.3, 0.3] shifted by 3.2
# standard deviations - an enormous, unmistakable drift:
#
#   PSI with fixed [0,10] bins : 0.0000   <- every value lands in one bin
#   PSI with quantile bins     : 6.9450
#
# So L6 bins by the REFERENCE QUANTILES. The PSI formula is shared; only the
# binning differs, and the binning is the whole difference between seeing a
# 3.2-sigma shift and reporting zero.
DRIFT_DETECTION_VERSION = "concept-drift-v1"

# The four drift types, in the order the task names them. Declared as data so
# a scan cannot silently cover three and report clean.
DRIFT_FEATURE = "feature_distribution"
DRIFT_RELATIONSHIP = "relationship"
DRIFT_CALIBRATION = "calibration"
DRIFT_RESPONSE = "event_response"

DRIFT_TYPES: tuple[str, ...] = (
    DRIFT_FEATURE,
    DRIFT_RELATIONSHIP,
    DRIFT_CALIBRATION,
    DRIFT_RESPONSE,
)

# Quantile bins for the feature detector. Ten is the industry convention and
# is what every PSI threshold in circulation was calibrated against; changing
# it silently rescales the thresholds below.
DRIFT_PSI_BINS = 10

# PSI thresholds. Deliberately the same numbers M5 uses (PROMOTION_PSI_WARN /
# PROMOTION_PSI_FAIL) because they are the same statistic - what L6 changes is
# the BINNING and the WINDOW FLOOR, not the scale.
DRIFT_PSI_WARN = 0.10
DRIFT_PSI_ALERT = 0.25

# THE WINDOW FLOOR IS LOAD-BEARING, AND THE CONVENTIONAL THRESHOLD IS UNSAFE
# WITHOUT IT. PSI between two IDENTICAL distributions is not zero - it is a
# random quantity that grows as the window shrinks. MEASURED with no drift
# whatsoever:
#
#   window   mean PSI     p95     p99   share exceeding 0.25
#       50     0.5251  1.1898  1.6852                  77.0%
#      100     0.1954  0.3719  0.5123                  20.5%
#      250     0.0736  0.1317  0.1681                   0.0%
#      500     0.0355  0.0685  0.0799                   0.0%
#     1000     0.0182  0.0335  0.0431                   0.0%
#
# At a window of 50 the conventional "PSI > 0.25 means significant shift" rule
# fires on 77% of CLEAN comparisons.
#
# ACROSS THE 13-FEATURE chart_state SURFACE, a report flagging if ANY feature
# exceeds the threshold:
#
#   window   thr 0.25   thr 0.10
#      100      96.0%     100.0%
#      250       0.7%      95.3%
#      500       0.0%       4.7%
#     1000       0.0%       0.0%
#
# A FIRST READING OF THE DETECTION SWEEP WAS WRONG and is recorded because the
# correction is the point. Detection of a real 0.5-sd shift appeared to FALL
# with window size (86.7% at 100, 51.0% at 1000), which would have argued for
# a small window. It is an artefact: at small windows PSI is INFLATED BY
# NOISE, so crossing 0.25 is not detection - it is the same noise that
# produces 96% false alarms. The signal converges to its true value while the
# noise shrinks:
#
#   window   mean PSI (0.5sd shift)   mean PSI (clean)   clean p95   separated?
#      100                   0.4428             0.2104      0.3755          no
#      250                   0.3260             0.0758      0.1387         YES
#      500                   0.2732             0.0362      0.0681         YES
#     1000                   0.2573             0.0177      0.0365         YES
#
# SEPARATION - signal clear of the clean p95 - is the honest measure, and it
# first holds at 250.
DRIFT_MIN_WINDOW = 250

# Relationship drift: the correlation between a feature and the outcome. A
# change of this size is reported. MEASURED, a sign flip moves it 0.683
# (+0.368 -> -0.315); ordinary resampling noise at the window floor is well
# under 0.15.
DRIFT_CORRELATION_SHIFT = 0.15

# Calibration drift: mean predicted probability minus mean observed rate. A
# model whose RANKING is intact can still be badly miscalibrated - MEASURED,
# inflating probabilities by 1.6x moved the gap 0.000 -> +0.251 while the
# correlation barely moved (0.448 -> 0.430), so the relationship detector is
# blind to it by construction.
DRIFT_CALIBRATION_GAP = 0.10

# Event-response drift: the mean absolute response to an event, compared as a
# LOG2 FACTOR rather than a percentage change.
#
# A PERCENTAGE RATIO IS THE WRONG SCALE, and my own gate caught it. A decline
# is bounded at -100% while an increase is unbounded, so a symmetric rule on
# |ratio| is not symmetric at all. MEASURED under the first version, with WARN
# at 50% and ALERT at twice that:
#
#   total collapse   -100.0%   ALERT      <- only an EXACT zero reaches it
#   99% weaker        -99.1%   WARN       <- an event type that stopped
#                                            moving price entirely
#   3x stronger      +180.9%   ALERT
#
# An event that has stopped working could never raise an alert. A log ratio is
# symmetric - halving is -1, doubling is +1 - so the bound becomes a FACTOR
# that reads the same in both directions:
#
#   25% weaker   log2 -0.42   STABLE
#   halved       log2 -1.00   WARN
#   quartered    log2 -2.00   ALERT
#   99% gone     log2 -6.64   ALERT
#   doubled      log2 +1.00   WARN
#   5x stronger  log2 +2.32   ALERT
DRIFT_RESPONSE_LOG2_WARN = 1.0    # a factor of 2 in either direction
DRIFT_RESPONSE_LOG2_ALERT = 2.0   # a factor of 4 in either direction

# A drift verdict is never silently a pass. These mirror M5's three states
# because the same rule applies: a detector that could not run has found
# nothing, not found the system clean.
DRIFT_STABLE = "STABLE"
DRIFT_WARN = "WARN"
DRIFT_ALERT = "ALERT"
DRIFT_NOT_EVALUATED = "NOT_EVALUATED"

DRIFT_VERDICTS: tuple[str, ...] = (
    DRIFT_NOT_EVALUATED,
    DRIFT_STABLE,
    DRIFT_WARN,
    DRIFT_ALERT,
)

# Drift is DETECTED and REPORTED, never acted on automatically. Retraining on
# a drift alert without the L5 chain would be the uncontrolled self-modifying
# system Sprint L exists to prevent.
DRIFT_TRIGGERS_RETRAIN = False


def _validate_drift_config() -> None:
    """Import-time guard for the L6 contract."""
    if len(set(DRIFT_TYPES)) != len(DRIFT_TYPES):
        raise ValueError("DRIFT_TYPES contains a duplicate")
    if len(DRIFT_TYPES) != 4:
        raise ValueError(
            f"the sprint names four drift types and {len(DRIFT_TYPES)} are "
            f"declared. MEASURED, the feature detector catches ONE of the "
            f"four: relationship, calibration and response drift are all "
            f"invisible to it because the features do not move"
        )
    if DRIFT_FEATURE not in DRIFT_TYPES or DRIFT_RESPONSE not in DRIFT_TYPES:
        raise ValueError("every named drift type must be declared")

    if DRIFT_MIN_WINDOW < 250:
        raise ValueError(
            f"a window floor of {DRIFT_MIN_WINDOW} cannot support a PSI "
            f"threshold. MEASURED with NO drift present, PSI exceeds 0.25 on "
            f"77% of comparisons at window 50 and 20.5% at window 100, and a "
            f"13-feature scan flags something on 96% of clean reports at 100"
        )
    if not 0.0 < DRIFT_PSI_WARN < DRIFT_PSI_ALERT:
        raise ValueError("PSI thresholds must ascend and be positive")
    if DRIFT_PSI_BINS < 5:
        raise ValueError(
            "fewer than five bins cannot resolve a distribution shift; ten is "
            "the convention every circulating PSI threshold was set against"
        )
    if not 0.0 < DRIFT_CORRELATION_SHIFT < 2.0:
        raise ValueError(
            "a correlation shift bound must lie inside the possible range of "
            "a correlation change"
        )
    if not 0.0 < DRIFT_CALIBRATION_GAP < 1.0:
        raise ValueError("the calibration gap bound must lie inside (0, 1)")
    if not 0.0 < DRIFT_RESPONSE_LOG2_WARN < DRIFT_RESPONSE_LOG2_ALERT:
        raise ValueError("response log2 bounds must ascend and be positive")

    if DRIFT_VERDICTS[0] != DRIFT_NOT_EVALUATED:
        raise ValueError(
            "NOT_EVALUATED must be the weakest verdict - a detector that "
            "could not run has found nothing, not found the system clean"
        )
    if DRIFT_VERDICTS[-1] != DRIFT_ALERT:
        raise ValueError("ALERT must be the strongest verdict")
    if DRIFT_STABLE == DRIFT_NOT_EVALUATED:
        raise ValueError(
            "'stable' and 'could not be measured' are different facts and "
            "must stay distinct"
        )
    if DRIFT_TRIGGERS_RETRAIN:
        raise ValueError(
            "drift must never trigger a retrain by itself: a model that "
            "replaces itself on an alert is the uncontrolled self-modifying "
            "system Sprint L exists to prevent. Drift is evidence FOR the L5 "
            "chain, not a substitute for it"
        )


_validate_drift_config()


# ---------------------------------------------------------------------------
# Sprint L7 - Regime-specific learning
# ---------------------------------------------------------------------------
# "Evaluate separate models per regime (bullish/bearish/range/risk-off/stress)
# ONLY IF OOS evidence supports specialization."
#
# The last clause is the whole task. Splitting the training data five ways is
# not free, and the default answer must be NO.
#
# THE DECIDING MEASUREMENT: SPECIALIZATION LOSES ON THIN DATA EVEN WHEN THE
# SKILL GENUINELY DIFFERS BY REGIME. Brier on held-out data:
#
#   skill is the SAME in every regime        pooled   specialized   winner
#      100 observations (20 per regime)     0.25114       0.26521   POOLED
#     1000 observations (200 per regime)    0.24788       0.24906   POOLED
#    10000 observations                     0.24753       0.24763   POOLED
#
#   skill GENUINELY DIFFERS by regime        pooled   specialized   winner
#      100 observations (20 per regime)     0.24913       0.25808   POOLED
#      300 observations (60 per regime)     0.24504       0.24371   SPECIALIZED
#     1000 observations                     0.24559       0.24315   SPECIALIZED
#    10000 observations                     0.24498       0.24175   SPECIALIZED
#
# When the skill does NOT differ, pooling wins at every sample size. When it
# DOES differ, specialization still loses until ~300 observations. So the task
# is right to make OOS evidence the condition, and the system must be able to
# answer NO.
#
# THE RARE-REGIME PROBLEM. The five regimes are not equally common - stress is
# roughly 5% of observations, so at n=1000 overall it holds ~52:
#
#   total n   bullish  bearish  range  risk_off  stress
#      1000       378      163    295       112      52
#     10000      3990     1457   3012      1025     516
#
# And a rate estimated from ~50 observations carries a typical error of 0.056,
# COMPARABLE TO THE 0.08 DIFFERENCE BEING DETECTED - the estimate IS the
# noise. A stress-specific model is therefore refused until the evidence
# exists, no matter how much total data the system holds.
REGIME_SPECIALIZATION_VERSION = "regime-specialization-v1"

# The regimes a model MAY specialize on. Taken from the governed five-state
# classifier (REGIME_LABELS) rather than redefined here - a second regime
# vocabulary would be the split-brain W5 forbids.
REGIME_SPECIALIZATION_LABELS: tuple[str, ...] = REGIME_LABELS

# THE DEFAULT IS POOLED. Specialization is adopted per regime, on evidence,
# and never assumed.
REGIME_SPECIALIZATION_DEFAULT_POOLED = True

# Minimum observations in a regime, in BOTH the training and the evaluation
# window, before its specialization can even be considered. MEASURED, a rate
# from 50 observations has a typical error of 0.056 while the effect being
# detected is 0.08.
REGIME_SPECIALIZATION_MIN_CELL = 60

# The relative Brier improvement a specialized model must show over the pooled
# one. A BARE WIN IS NOT EVIDENCE: MEASURED with NO real difference anywhere,
# specialization still "wins OOS" on 14-23% of single comparisons.
#
#   rule                          false adopt        true adopt
#   bare OOS win                    14 - 23%          30 - 100%
#   win by >= 0.5% relative          0 -  1%          75 - 100%
#   win by >= 2.0% relative                0%           0 - 10%
#
# 0.5% is the point where false adoption collapses while real specialization
# is still found; 2% is past the knee and rejects everything.
REGIME_SPECIALIZATION_MARGIN = 0.005

# Folds the comparison is repeated over, and how many it must win. Single
# comparisons are noisy even with a margin - MEASURED, a regime with NO real
# difference still adopted on 17-30% of single-fold runs.
#
#   folds/need     real effect found     worst noise regime
#      1 of 1          85 - 96%               19 - 26%
#      3 of 5          68 - 95%                    15%
#      4 of 5          36 - 68%                 2 -  4%
#      5 of 5          10 - 33%                     0%
#
# 4 of 5 is the balance: noise falls to 2-4% while real specialization is
# still adopted. Demanding all five rejects a real effect two thirds of the
# time, which is not caution but blindness.
REGIME_SPECIALIZATION_FOLDS = 5
REGIME_SPECIALIZATION_FOLDS_REQUIRED = 4

# Verdicts, weakest to strongest. The order IS the precedence.
REGIME_SPEC_NOT_EVALUATED = "NOT_EVALUATED"   # no comparison was possible
REGIME_SPEC_INSUFFICIENT = "INSUFFICIENT_DATA"  # below the cell floor
REGIME_SPEC_POOLED = "POOLED"                 # tested, pooling is not beaten
REGIME_SPEC_SPECIALIZED = "SPECIALIZED"       # earned its own model

REGIME_SPECIALIZATION_VERDICTS: tuple[str, ...] = (
    REGIME_SPEC_NOT_EVALUATED,
    REGIME_SPEC_INSUFFICIENT,
    REGIME_SPEC_POOLED,
    REGIME_SPEC_SPECIALIZED,
)

# A regime that fails to earn specialization FALLS BACK to the pooled model,
# it is never left unserved. The alternative - no model for stress because
# stress is rare - would remove coverage exactly when it matters most.
REGIME_SPECIALIZATION_FALLBACK_POOLED = True

# Specialization is decided PER REGIME, not all-or-nothing. MEASURED, a common
# regime with a real difference (bullish, 63% adoption at n=1000) should not
# be held hostage to a rare one (stress, 1% at the same size).
REGIME_SPECIALIZATION_PER_REGIME = True


def _validate_regime_specialization_config() -> None:
    """Import-time guard for the L7 contract."""
    if set(REGIME_SPECIALIZATION_LABELS) != set(REGIME_LABELS):
        raise ValueError(
            "L7 must specialize on the GOVERNED regime labels, not a second "
            "vocabulary of its own - that is the split-brain W5 forbids"
        )
    if len(REGIME_SPECIALIZATION_LABELS) != 5:
        raise ValueError(
            f"the sprint names five regimes and "
            f"{len(REGIME_SPECIALIZATION_LABELS)} are declared"
        )
    if not REGIME_SPECIALIZATION_DEFAULT_POOLED:
        raise ValueError(
            "the default must be POOLED. MEASURED, when skill does NOT differ "
            "by regime, pooling wins at every sample size, and even when it "
            "DOES differ specialization loses until ~300 observations"
        )
    if REGIME_SPECIALIZATION_MIN_CELL < 60:
        raise ValueError(
            f"a cell floor of {REGIME_SPECIALIZATION_MIN_CELL} cannot support "
            f"a per-regime rate. MEASURED, an estimate from 50 observations "
            f"carries a typical error of 0.056 while the effect being detected "
            f"is 0.08 - the estimate is the noise"
        )
    if REGIME_SPECIALIZATION_MARGIN <= 0.0:
        raise ValueError(
            "a bare OOS win is not evidence. MEASURED with no real difference "
            "anywhere, specialization still wins 14-23% of single comparisons"
        )
    if REGIME_SPECIALIZATION_MARGIN > 0.02:
        raise ValueError(
            f"a margin of {REGIME_SPECIALIZATION_MARGIN} rejects real "
            f"specialization. MEASURED at 2%, a genuine per-regime difference "
            f"is adopted 0-10% of the time"
        )
    if REGIME_SPECIALIZATION_FOLDS < 3:
        raise ValueError(
            "fewer than three folds cannot show a result repeats; MEASURED, a "
            "single fold adopts a noise regime on 17-30% of runs"
        )
    if not 1 <= REGIME_SPECIALIZATION_FOLDS_REQUIRED <= REGIME_SPECIALIZATION_FOLDS:
        raise ValueError("the required folds must lie within the folds run")
    if REGIME_SPECIALIZATION_FOLDS_REQUIRED * 2 <= REGIME_SPECIALIZATION_FOLDS:
        raise ValueError(
            f"requiring {REGIME_SPECIALIZATION_FOLDS_REQUIRED} of "
            f"{REGIME_SPECIALIZATION_FOLDS} folds is a minority, which adopts "
            f"on noise: a result that fails most of its own folds has not "
            f"repeated"
        )
    if REGIME_SPECIALIZATION_VERDICTS[0] != REGIME_SPEC_NOT_EVALUATED:
        raise ValueError("NOT_EVALUATED must be the weakest verdict")
    if REGIME_SPECIALIZATION_VERDICTS[-1] != REGIME_SPEC_SPECIALIZED:
        raise ValueError("SPECIALIZED must be the strongest verdict")
    if REGIME_SPEC_INSUFFICIENT == REGIME_SPEC_POOLED:
        raise ValueError(
            "'too little data to test' and 'tested, pooling wins' are "
            "different facts and must stay distinct"
        )
    if not REGIME_SPECIALIZATION_FALLBACK_POOLED:
        raise ValueError(
            "a regime that does not earn specialization must fall back to the "
            "pooled model, never be left unserved: no model for stress "
            "because stress is rare removes coverage exactly when it matters"
        )
    if not REGIME_SPECIALIZATION_PER_REGIME:
        raise ValueError(
            "specialization is decided PER REGIME. MEASURED, a common regime "
            "with a real difference adopts at 63% while a rare one adopts at "
            "1% on the same data - an all-or-nothing rule holds the first "
            "hostage to the second"
        )


_validate_regime_specialization_config()


# ---------------------------------------------------------------------------
# Sprint L8 - Champion evolution
# ---------------------------------------------------------------------------
# "Replacement requires credible OOS improvement, acceptable calibration, no
# unacceptable false-positive degradation, acceptable risk, regime robustness,
# reproducibility, governance approval. Historical forecasts remain
# immutable."
#
# SEVEN CONDITIONS AND AN INVARIANT. Most have an owner already, and L8 must
# SEQUENCE them rather than rebuild them (W5):
#
#   credible OOS improvement    M5 PROMO_CHECK_OOS          + L8 credibility
#   acceptable calibration      L6 calibration_drift        delegated
#   no FP degradation           NEW - M5 measures VETO rate, not precision
#   acceptable risk             NEW - nothing measured risk at promotion
#   regime robustness           L7 per-regime evaluation    delegated
#   reproducibility             M8 REPRODUCIBILITY          delegated
#   governance approval         M5 PROMO_CHECK_APPROVAL     delegated
#   historical immutability     M5 PROMOTION_HISTORY_IMMUTABLE
#
# THREE GAPS WERE REAL.
#
# GAP 1: "CREDIBLE" WAS UNQUALIFIED. M5 requires the candidate to beat the
# incumbent out of sample, but not by how much relative to noise. MEASURED,
# two models of IDENTICAL skill (0.55):
#
#     n     mean gap   p90 gap   p99 gap
#   100      -0.0009   +0.0900   +0.1700
#   250      -0.0014   +0.0560   +0.1040
#   500      +0.0006   +0.0420   +0.0720
#  1000      -0.0004   +0.0280   +0.0500
#  2000      -0.0003   +0.0195   +0.0365
#
# At n=250 a challenger with NO real edge beats the champion by 5.6 points on
# 10% of comparisons and by 10.4 points on 1%. "It improved out of sample" is
# therefore not credible evidence on its own - the improvement must be large
# relative to what luck produces at that sample size.
#
# GAP 2: RISK WAS NEVER MEASURED AT PROMOTION. A model can improve accuracy
# and destroy the account. MEASURED with the hit rate held FIXED and only the
# SIZE of losing moves changed:
#
#   model                              hit rate   mean ret    max DD
#   champion                              0.580   +0.00160   -0.1244
#   challenger: same acc, big losses      0.580   -0.00681   -3.4496
#   challenger: better acc + big losses   0.620   -0.00520   -2.6626
#
# The second has the SAME hit rate and a 28x worse drawdown; the third has a
# BETTER hit rate and still loses money. An accuracy-only gate promotes both.
#
# GAP 3: FALSE POSITIVES ARE NOT THE VETO RATE. M5's regression check watches
# how often the policy vetoes; L8 needs how often an ACTED-ON call was wrong.
# MEASURED:
#
#   model                             overall acc   BUY false-positive rate
#   champion                                0.594                     0.349
#   challenger: same acc, worse BUYs        0.597                     0.551
#   challenger: fewer but better BUYs       0.591                     0.204
#
# The second matches the champion on overall accuracy while every acted-on
# call got worse. Overall accuracy cannot see it.
CHAMPION_EVOLUTION_VERSION = "champion-evolution-v1"

# The seven conditions, in the order the sprint names them. Declared as data so
# a replacement cannot silently satisfy six and proceed.
EVO_OOS = "credible_oos_improvement"
EVO_CALIBRATION = "acceptable_calibration"
EVO_FALSE_POSITIVE = "no_false_positive_degradation"
EVO_RISK = "acceptable_risk"
EVO_REGIME = "regime_robustness"
EVO_REPRODUCIBILITY = "reproducibility"
EVO_GOVERNANCE = "governance_approval"

CHAMPION_EVOLUTION_CONDITIONS: tuple[str, ...] = (
    EVO_OOS,
    EVO_CALIBRATION,
    EVO_FALSE_POSITIVE,
    EVO_RISK,
    EVO_REGIME,
    EVO_REPRODUCIBILITY,
    EVO_GOVERNANCE,
)

# EVERY condition is required. There is no "mostly acceptable" replacement:
# the sprint lists them with "and", not "or".
CHAMPION_EVOLUTION_REQUIRED: tuple[str, ...] = CHAMPION_EVOLUTION_CONDITIONS

# Conditions L8 OWNS rather than delegates - the three real gaps.
CHAMPION_EVOLUTION_OWNED: tuple[str, ...] = (
    EVO_OOS,          # the CREDIBILITY test on top of M5's bare comparison
    EVO_FALSE_POSITIVE,
    EVO_RISK,
)

# CREDIBILITY. The OOS improvement must exceed this multiple of its own
# standard error, so the bar scales with the evidence rather than being a
# fixed number of points. 2.58 is the p<0.01 bar; MEASURED, the p99 of the
# null gap is +0.1040 at n=250 and +0.0365 at n=2000, so a fixed percentage
# bar would be far too lax at small n and far too strict at large n.
CHAMPION_EVOLUTION_CREDIBILITY_SIGMA = 2.58

# ...and a floor on the evidence itself. MEASURED, detection of a REAL 3-point
# edge at the p<0.01 bar is 6.3% at n=250 and 16.7% at n=1000 - below this
# there is no point running the test, because it cannot find anything.
CHAMPION_EVOLUTION_MIN_OOS_SAMPLE = 500

# RISK. Maximum tolerated worsening of the drawdown, as a fraction of the
# incumbent's. MEASURED, a challenger with the SAME hit rate carried a 28x
# worse drawdown, so the bound must be a multiple rather than an absolute.
CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO = 1.25

# ...and of the left tail (5th percentile of returns). A model can hold its
# drawdown while making its bad days much worse.
CHAMPION_EVOLUTION_MAX_TAIL_RATIO = 1.25

# FALSE POSITIVES. How much the acted-on false-positive rate may rise in
# ABSOLUTE terms. MEASURED, a challenger matching the champion on overall
# accuracy moved it 0.349 -> 0.551, a rise of 0.202.
CHAMPION_EVOLUTION_MAX_FP_INCREASE = 0.05

# Minimum acted-on calls before the false-positive comparison means anything.
# A precision estimated from a handful of BUYs is noise.
CHAMPION_EVOLUTION_MIN_ACTED_CALLS = 100

# Outcomes. NOT_EVALUATED blocks exactly as FAIL does, inherited from M5:
# absence of evidence is not evidence of safety.
EVO_PASS = "PASS"
EVO_FAIL = "FAIL"
EVO_NOT_EVALUATED = "NOT_EVALUATED"

CHAMPION_EVOLUTION_OUTCOMES: tuple[str, ...] = (EVO_PASS, EVO_FAIL, EVO_NOT_EVALUATED)

# HISTORICAL FORECASTS REMAIN IMMUTABLE. A promotion never rewrites what a
# past forecast said or which model made it. This is the one clause that is
# not a threshold: it is an invariant, and a system that edits its own history
# can report any track record it likes.
CHAMPION_EVOLUTION_HISTORY_IMMUTABLE = True


def _validate_champion_evolution_config() -> None:
    """Import-time guard for the L8 contract."""
    if len(set(CHAMPION_EVOLUTION_CONDITIONS)) != len(CHAMPION_EVOLUTION_CONDITIONS):
        raise ValueError("CHAMPION_EVOLUTION_CONDITIONS contains a duplicate")
    if len(CHAMPION_EVOLUTION_CONDITIONS) != 7:
        raise ValueError(
            f"the sprint names seven conditions and "
            f"{len(CHAMPION_EVOLUTION_CONDITIONS)} are declared"
        )
    if set(CHAMPION_EVOLUTION_REQUIRED) != set(CHAMPION_EVOLUTION_CONDITIONS):
        raise ValueError(
            "every condition is required - the sprint lists them with 'and', "
            "and a replacement that satisfies six is not a replacement"
        )
    unknown = set(CHAMPION_EVOLUTION_OWNED) - set(CHAMPION_EVOLUTION_CONDITIONS)
    if unknown:
        raise ValueError(f"CHAMPION_EVOLUTION_OWNED names unknown conditions: {sorted(unknown)}")
    if EVO_GOVERNANCE in CHAMPION_EVOLUTION_OWNED:
        raise ValueError(
            "governance approval is M5's check - a second copy would let L8 "
            "approve a promotion M5 would refuse"
        )
    if EVO_RISK not in CHAMPION_EVOLUTION_OWNED:
        raise ValueError(
            "risk must be owned here: MEASURED, nothing in the promotion path "
            "measured risk at all, and a challenger with the SAME hit rate "
            "carried a 28x worse drawdown"
        )

    if CHAMPION_EVOLUTION_CREDIBILITY_SIGMA < 1.96:
        raise ValueError(
            f"a credibility bar of {CHAMPION_EVOLUTION_CREDIBILITY_SIGMA} sigma "
            f"is too lax. MEASURED, two models of IDENTICAL skill differ by "
            f"5.6 points on 10% of comparisons at n=250 - an unqualified "
            f"'it improved out of sample' is not credible evidence"
        )
    if CHAMPION_EVOLUTION_MIN_OOS_SAMPLE < 500:
        raise ValueError(
            f"an OOS floor of {CHAMPION_EVOLUTION_MIN_OOS_SAMPLE} cannot "
            f"support a credibility test. MEASURED, detection of a real "
            f"3-point edge at the p<0.01 bar is 6.3% at n=250"
        )
    if CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO <= 1.0:
        raise ValueError(
            "the drawdown bound must allow at least parity, or no candidate "
            "could ever be promoted"
        )
    if CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO > 2.0:
        raise ValueError(
            f"a drawdown bound of {CHAMPION_EVOLUTION_MAX_DRAWDOWN_RATIO}x "
            f"permits doubling the worst loss in exchange for accuracy. "
            f"MEASURED, a challenger with the SAME hit rate carried a 28x "
            f"worse drawdown while losing money"
        )
    if not 1.0 < CHAMPION_EVOLUTION_MAX_TAIL_RATIO <= 2.0:
        raise ValueError("the tail bound must allow parity and stay under 2x")
    if not 0.0 < CHAMPION_EVOLUTION_MAX_FP_INCREASE <= 0.10:
        raise ValueError(
            f"a false-positive allowance of {CHAMPION_EVOLUTION_MAX_FP_INCREASE} "
            f"is not a bound. MEASURED, a challenger matching the champion on "
            f"OVERALL accuracy moved the acted-on false-positive rate "
            f"0.349 -> 0.551"
        )
    if CHAMPION_EVOLUTION_MIN_ACTED_CALLS < 100:
        raise ValueError(
            "a precision estimated from fewer than 100 acted-on calls is noise"
        )
    if EVO_NOT_EVALUATED == EVO_PASS:
        raise ValueError(
            "'could not be evaluated' must never equal PASS - absence of "
            "evidence is not evidence of safety"
        )
    if not CHAMPION_EVOLUTION_HISTORY_IMMUTABLE:
        raise ValueError(
            "historical forecasts must remain immutable. A system that can "
            "edit what a past forecast said, or which model made it, can "
            "report any track record it likes - and every measurement in this "
            "repository would become unverifiable"
        )


_validate_champion_evolution_config()


# ---------------------------------------------------------------------------
