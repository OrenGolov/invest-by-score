MAX_SCORE = 10.0
MIN_SCORE = 0.0
DEFAULT_ACTION = "ANALYSIS_ONLY"
DEFAULT_CONFIDENCE = 0.5

MARKET_FEATURE_VERSION = "market-feature-v1"
# v3: the embedded news term was removed from the current-time view (Sprint N1);
# news enters the published score exclusively through its own ensemble line.
CURRENT_SCORE_VERSION = "current-score-v3"
LONG_TERM_SCORE_VERSION = "long-term-score-v2"
NEWS_CONTRACT_VERSION = "news-contract-v1"
SENTIMENT_CONTRACT_VERSION = "sentiment-contract-v1"
# N2 anti-proxying rule: a news-derived sentiment design is permitted only as
# an explicit, labeled feature (`derivation: "derived_from_news"`); its
# confidence MUST be scaled by this factor relative to the news evidence it
# consumed, so the dependency is reflected in every downstream confidence read.
SENTIMENT_DERIVED_CONFIDENCE_SCALE = 0.5

# --- Ensemble wiring (W1) ------------------------------------------------------
# The published score is the weighted product of agent contributions, not an
# independent hand-built blend. Both weight sets share an identical key set,
# must each sum to 1.0 (validated at import time), and intentionally differ
# per horizon: business quality matters more to the structural view than to
# the tactical one. Agents without a live implementation hold an explicit 0.0
# weight — presence in the dict is the contract; absence fails at import.
# v3: macroeconomic is born wired (N3) with a dedicated weight in BOTH
# horizons; while its status is not OK (no FRED key, failed/empty fetch,
# partial coverage) the weight renormalizes across eligible agents.
ENSEMBLE_VERSION = "ensemble-v3"

ENSEMBLE_WEIGHTS_CURRENT = {
    "market_data": 0.0,          # informational only: feeds confidence/gates
    "technical_analysis": 0.70,  # current-time technical view
    "fundamental_analysis": 0.10,
    "news_intelligence": 0.10,   # N1: live whenever the news contract reads OK
    "sentiment": 0.0,            # N2 typed placeholder; no legitimate provider yet
    "macroeconomic": 0.10,       # N3: live whenever the macro contract reads OK
    "market_regime": 0.0,        # not implemented; regime gates via risk policy
}

ENSEMBLE_WEIGHTS_LONG = {
    "market_data": 0.0,
    "technical_analysis": 0.70,  # long-term structural technical view
    "fundamental_analysis": 0.20,
    "news_intelligence": 0.0,    # tactical-only: news never enters the structural view
    "sentiment": 0.0,
    "macroeconomic": 0.10,       # macro state is horizon-agnostic evidence
    "market_regime": 0.0,
}


def _validate_ensemble_weights(name: str, weights: dict[str, float]) -> None:
    """Import-time guard: complete key set, non-negative, summing to 1.0."""
    if not weights:
        raise ValueError(f"{name} must not be empty")
    if set(weights) != set(ENSEMBLE_WEIGHTS_CURRENT):
        raise ValueError(
            f"{name} keys must match ENSEMBLE_WEIGHTS_CURRENT exactly: "
            f"missing={sorted(set(ENSEMBLE_WEIGHTS_CURRENT) - set(weights))} "
            f"extra={sorted(set(weights) - set(ENSEMBLE_WEIGHTS_CURRENT))}"
        )
    negative = {agent: weight for agent, weight in weights.items() if float(weight) < 0.0}
    if negative:
        raise ValueError(f"{name} weights must be non-negative, got {negative}")
    total = float(sum(weights.values()))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1.0 (tolerance 1e-9), got {total!r}")


_validate_ensemble_weights("ENSEMBLE_WEIGHTS_CURRENT", ENSEMBLE_WEIGHTS_CURRENT)
_validate_ensemble_weights("ENSEMBLE_WEIGHTS_LONG", ENSEMBLE_WEIGHTS_LONG)

# --- Narrative vs fundamental attribution (N5) ----------------------------------
# Decomposes the blended score into three governed buckets so the system can
# state whether a thesis is supported by business reality, market narrative,
# or both. The evaluator lives in core/score_engine.py::build_attribution and
# only classifies/aggregates the ensemble breakdown's per-agent contributions
# — the ensemble remains the single source of contribution math, so the
# attribution can never contradict the published score.
#
# Bucket semantics (versioned): operational = fundamental + technical
# (the long-horizon technical view anchors the bucket; its current-horizon
# component is tactical but still technical evidence); narrative = news +
# sentiment ONLY — with news at zero weight the bucket reads exactly 0.0
# (no phantom narrative, the N5 acceptance criterion); macro_shock = macro
# + regime. market_data is an informational zero-weight line and sits
# outside the buckets by design.
ATTRIBUTION_VERSION = "score-attribution-v1"

ATTRIBUTION_BUCKETS = {
    "operational": ("fundamental_analysis", "technical_analysis"),
    "narrative": ("news_intelligence", "sentiment"),
    "macro_shock": ("macroeconomic", "market_regime"),
}

# Zero-weight informational lines that stay outside every bucket.
ATTRIBUTION_INFORMATIONAL_LINES = ("market_data",)

# A bucket "supports" the thesis above this many score points, "opposes"
# below the negated threshold, and is "neutral" in between (0-10 scale).
ATTRIBUTION_SUPPORT_THRESHOLD = 0.25


def _validate_attribution_buckets() -> None:
    """Import-time guard: the buckets partition the ensemble lines exactly."""
    assigned = [
        agent
        for members in ATTRIBUTION_BUCKETS.values()
        for agent in members
    ]
    if len(assigned) != len(set(assigned)):
        raise ValueError("ATTRIBUTION_BUCKETS: an agent line appears in more than one bucket")
    covered = set(assigned) | set(ATTRIBUTION_INFORMATIONAL_LINES)
    ensemble_lines = set(ENSEMBLE_WEIGHTS_CURRENT)
    if covered != ensemble_lines:
        raise ValueError(
            "ATTRIBUTION_BUCKETS must partition the ensemble lines exactly: "
            f"missing={sorted(ensemble_lines - covered)}, unknown={sorted(covered - ensemble_lines)}"
        )
    if not 0.0 < ATTRIBUTION_SUPPORT_THRESHOLD <= MAX_SCORE:
        raise ValueError(
            f"ATTRIBUTION_SUPPORT_THRESHOLD must be within (0, {MAX_SCORE}], "
            f"got {ATTRIBUTION_SUPPORT_THRESHOLD!r}"
        )


_validate_attribution_buckets()

# --- Risk policy (W2) -----------------------------------------------------------
# Single source of truth for every governance threshold. The evaluator lives in
# core/risk_policy.py and is the only consumer; nothing else may hard-code these
# limits. severity "veto" blocks PAPER posture; "warning" is visible but does
# not block. Missing/None inputs evaluate to triggered rules — fail-closed.
RISK_POLICY_VERSION = "risk-policy-v2"

# --- Audit policy (W3) ----------------------------------------------------------
# The auditor independently verifies that a decision is provable: evidence
# sufficiency, hash integrity, determinism, calibration sanity, and ensemble
# consistency. Its evaluator lives in core/audit_policy.py; a failed veto-
# severity check appends the "auditor_veto" reason and blocks PAPER posture.
AUDIT_POLICY_VERSION = "audit-policy-v1"

RISK_POLICY_V2 = {
    "data_quality_below_threshold": {
        "severity": "veto",
        "minimum_market_data_quality": 60.0,
    },
    "market_source_confidence_below_threshold": {
        "severity": "veto",
        "minimum_market_source_confidence": 0.7,
    },
    "future_dated_market_data": {
        "severity": "veto",
    },
    "future_dated_fundamental_payload": {
        "severity": "veto",
    },
    "fundamental_source_confidence_below_threshold": {
        "severity": "veto",
        "minimum_fundamental_source_confidence": 0.7,
    },
    "score_below_threshold": {
        "severity": "veto",
        "minimum_score": 5.5,
    },
    # N4 governance coupling: the STRESS regime forces NO_TRADE. A missing or
    # unknown regime label also triggers (fail-closed) — a decision without
    # market-risk context must not trade.
    "market_regime_stress": {
        "severity": "veto",
    },
    "analysis_only_mode": {
        "severity": "veto",
    },
    "confidence_below_minimum": {
        "severity": "veto",
        "minimum_confidence": 0.35,
    },
    "confidence_penalty_budget_exceeded": {
        "severity": "warning",
        "maximum_total_penalty": 0.15,
    },
    "freshness_degraded": {
        "severity": "warning",
        "minimum_freshness_factor": 0.5,
    },
    "volatility_regime_elevated": {
        "severity": "warning",
        "minimum_volatility_regime_factor": 0.3,
    },
}


def _validate_risk_policy() -> None:
    """Import-time guard: every rule is a non-empty spec with valid severity."""
    if not RISK_POLICY_V2:
        raise ValueError("RISK_POLICY_V2 must contain at least one rule")
    for rule_id, spec in RISK_POLICY_V2.items():
        if not isinstance(spec, dict) or not spec:
            raise ValueError(f"RISK_POLICY_V2 rule {rule_id!r} must be a non-empty spec")
        if spec.get("severity") not in {"veto", "warning"}:
            raise ValueError(
                f"RISK_POLICY_V2 rule {rule_id!r} severity must be 'veto' or 'warning', "
                f"got {spec.get('severity')!r}"
            )


_validate_risk_policy()

# Evidence-based confidence model (see core.score_engine._compute_confidence).
# Each factor produces a value in [0, 1]; the confidence is the weighted sum
# minus explicit risk penalties, clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CAP].
# Weights sum to 1.0 so the baseline stays interpretable as a percentage.
CONFIDENCE_VERSION = "evidence-confidence-v2"

CONFIDENCE_WEIGHT_DATA_QUALITY = 0.25
CONFIDENCE_WEIGHT_SOURCE_RELIABILITY = 0.20
CONFIDENCE_WEIGHT_SIGNAL_AGREEMENT = 0.20
CONFIDENCE_WEIGHT_FRESHNESS = 0.15
CONFIDENCE_WEIGHT_HISTORY_COVERAGE = 0.10
CONFIDENCE_WEIGHT_VOLATILITY_REGIME = 0.10

# Calendar-age gap between the newest bar used and as_of before freshness decays,
# and how many calendar days after that grace window reach zero freshness credit.
CONFIDENCE_FRESHNESS_GRACE_DAYS = 4
CONFIDENCE_FRESHNESS_DECAY_DAYS = 26

# Minimum number of valid daily bars for full long-window (200d MA) coverage.
CONFIDENCE_FULL_COVERAGE_BARS = 230

# Daily-return standard deviation treated as fully calm versus fully chaotic.
CONFIDENCE_VOL_CALM_DAILY_STD = 0.015
CONFIDENCE_VOL_CHAOTIC_DAILY_STD = 0.060

# Dead-zone half-width used when reading factor directions so tiny moves do not
# flip a signal between bullish/bearish arbitrarily.
CONFIDENCE_SIGNAL_DEADZONE_RATIO = 0.002

# Named penalties applied once per condition, replacing the previous flat -0.20
# per category. Scaled to reflect how much each condition undermines the result.
RISK_FLAG_CONFIDENCE_PENALTIES = {
    "Weak momentum": 0.10,
    "Low volume": 0.08,
    "Downtrend": 0.12,
    "RSI extreme": 0.06,
}
FUNDAMENTAL_SOURCE_PENALTY = 0.10
GOVERNANCE_RISK_GATE_PENALTY = 0.05

CONFIDENCE_FLOOR = 0.10
CONFIDENCE_CAP = 0.95

# --- News intelligence adapter (N1) ---------------------------------------------
# Pipeline: NEWS -> PIT FILTER -> ENTITY RESOLUTION -> EVENT CLASSIFICATION ->
# SOURCE QUALITY -> RELEVANCE/NOVELTY -> DIRECTION/MAGNITUDE -> CONTRADICTION
# DETECTION -> EVIDENCE-BACKED OUTPUT. The evaluator lives in
# core/news_adapter.py; thresholds and versions live here so governance reads
# them from exactly one place. The no-key path stays the explicit UNAVAILABLE
# contract (a missing provider is a status, never a neutral score).
NEWS_CLASSIFIER_VERSION = "news-classifier-v2"
NEWS_TONE_LEXICON_VERSION = "news-tone-lexicon-v1"
NEWS_AGGREGATOR_VERSION = "news-aggregator-v1"
NEWS_PIPELINE_VERSION = "news-pipeline-v1"

NEWS_PROVIDER_API_KEY_ENV = "NEWS_PROVIDER_API_KEY"
NEWS_PROVIDER_URL = "https://newsapi.org/v2/everything"
NEWS_PROVIDER_TIMEOUT_SECONDS = 10.0

# Query window: articles with published_time <= as_of, window end at as_of,
# start at as_of minus this many calendar days.
NEWS_LOOKBACK_DAYS = 7

# Exponential recency decay half-life, in days. v1 approximates the 3-trading-
# day half-life with calendar days (deterministic; no market calendar needed),
# mirroring the calendar-day convention of CONFIDENCE_FRESHNESS_*.
NEWS_RECENCY_HALF_LIFE_DAYS = 3.0

# Contradiction v1: a same-day, same-category cluster of credible articles whose
# positive/negative mean tones are opposite-sign with |delta| strictly above
# this threshold yields status CONTRADICTORY (never a neutral average).
NEWS_CONTRADICTION_TONE_DELTA = 0.6
NEWS_CONTRADICTION_CONFIDENCE_FLOOR = 0.10

# Registry base confidence for the news domain (SOURCE_REGISTRY["news"] mirrors
# this value; the adapter never imports fetch_data to avoid coupling).
NEWS_BASE_SOURCE_CONFIDENCE = 0.75

# Maximum provider articles considered per request (also the provider page size).
NEWS_MAX_ARTICLES = 50

# Aggregated sentiment [-1, 1] maps onto the ensemble contribution line as
# base + span * sentiment, i.e. a 0-10 score like every other agent line.
NEWS_SCORE_BASE = 5.0
NEWS_SCORE_SPAN = 5.0

# --- Macroeconomic agent (N3) ---------------------------------------------------
# Vintage-aware economic data with PIT filtering by published_time (first-release
# semantics). Every series carries provenance: source, publication lag, frequency,
# transformation, and sector-specific sensitivities. Missing series degrades
# confidence (INCOMPLETE), never silently zero-fills. Revisions append to
# raw_store (W6); eligibility gates on published_time <= as_of.

MACRO_CONTRACT_VERSION = "macro-contract-v1"
# v2: VINTAGE SELECTION stage (N3) — first-release retention + revision history
# via the ALFRED realtime feed; values are selected as known at as_of.
# v3: VIX + 30Y yield join the risk-regime signal set (7 series). The regime
# score is the MEAN of available signals, so widening the set changes every
# historical score — hence a version bump rather than a silent addition.
MACRO_ADAPTER_VERSION = "macro-adapter-v3"

# Provider configuration (FRED as v1; BLS/CENSUS extensible but not in v1).
MACRO_PROVIDER_API_KEY_ENV = "FRED_API_KEY"
MACRO_PROVIDER_TIMEOUT_SECONDS = 10.0

# Fetch parameters: how far back to fetch (enough history for trend/shock detection).
# FRED returns all available history; we cache and use the last N periods.
MACRO_LOOKBACK_PERIODS = 120  # ~10 years of monthly (or equivalent weekly/daily)

# Confidence degradation for missing or INCOMPLETE series.
MACRO_MISSING_SERIES_PENALTY = 0.15

# Macro score range and centering (risk-on/off tilt, 0-10 scale).
MACRO_SCORE_BASE = 5.0
MACRO_SCORE_SPAN = 5.0

# Risk-on/off regime thresholds (0.5 = neutral, >0.5 = risk-on tilt).
MACRO_RISKOFF_THRESHOLD = 0.3
MACRO_RISKON_THRESHOLD = 0.7

# VIX bands (index points). VIX is the one INVERTED series in the set: a high
# reading is risk-OFF. ~20 is the long-run average, ~30 the stress threshold.
MACRO_VIX_ELEVATED = 20.0
MACRO_VIX_STRESS = 30.0
MACRO_VIX_CALM = 14.0

# 30Y yield bands (percent), mirroring the 10Y level signal one notch higher
# to reflect the term premium at the long end.
MACRO_YIELD_30Y_ELEVATED = 4.0
MACRO_YIELD_30Y_NORMAL = 3.0
MACRO_YIELD_30Y_DEPRESSED = 2.0

# --- Market regime agent (N4) ----------------------------------------------------
# Five-state governance classification: bullish, bearish, range, risk_off, stress.
# The classifier lives in core/regime_agent.py and is the ONLY source of the
# governed regime label; agents/market_data_agent.py keeps its legacy 3-state
# `market_regime` display heuristic (ungoverned, not the decision regime).
# Governance coupling: STRESS forces NO_TRADE through the W2 policy rule
# `market_regime_stress` (severity veto, evaluated in core/risk_policy.py, the
# only evaluator); RISK_OFF dampens the momentum coefficients of the
# current-time technical view by REGIME_RISKOFF_MOMENTUM_DAMPING (applied in
# core/score_engine.py and mirrored in the scoring breakdown so the
# explanation can never contradict the number).

REGIME_CONTRACT_VERSION = "regime-contract-v1"
REGIME_PIPELINE_VERSION = "regime-pipeline-v1"
REGIME_CLASSIFIER_VERSION = "regime-classifier-v1"

REGIME_LABELS = ("bullish", "bearish", "range", "risk_off", "stress")
REGIME_STRESS_LABEL = "stress"
REGIME_RISKOFF_LABEL = "risk_off"
REGIME_RANGE_LABEL = "range"

# Range v1: flat zone — both MA distances strictly inside +/- this band.
REGIME_RANGE_FLAT_THRESHOLD = 0.02

# Realized volatility: daily-return standard deviation (ddof=0) over this many
# sessions, matching the market-data agent's `volatility` feature convention.
REGIME_REALIZED_VOL_WINDOW_SESSIONS = 30

# Volatility is compared against its own trailing history: prior observations
# (excluding the current session) over a 1y session lookback.
REGIME_VOL_PERCENTILE_LOOKBACK_SESSIONS = 252
REGIME_STRESS_VOL_PERCENTILE = 0.95
REGIME_RISKOFF_VOL_PERCENTILE = 0.80

# Stress requires a second, independent condition: depth below the rolling
# 60-session high, strictly above this fraction.
REGIME_DRAWDOWN_HIGH_WINDOW_SESSIONS = 60
REGIME_STRESS_DRAWDOWN_THRESHOLD = 0.15

# Risk-off trend branch: MA50 < MA200 with 20d AND 60d momentum both negative.
# Transition risk counts regime flips across this many trailing sessions.
REGIME_TRANSITION_WINDOW_SESSIONS = 20

# Strict minimum eligible history: the 30d realized vol needs 31 closes and
# the trailing 1y percentile needs 252 prior vol observations, so the first
# fully classified session is index 282 (0-based). MA200 and change_60d sit
# inside that span. Derived constant — do not hand-edit.
REGIME_REQUIRED_SESSIONS = (
    REGIME_REALIZED_VOL_WINDOW_SESSIONS + REGIME_VOL_PERCENTILE_LOOKBACK_SESSIONS + 1
)

# Calendar-day coverage the provider fetch must reach behind as_of
# (~283 sessions plus a holiday buffer). Selects the fetch period
# deterministically in core/regime_agent.py.
REGIME_CALENDAR_COVERAGE_DAYS = 430

# probability_proxy scales: how deep past the deciding boundary, normalized
# to [0, 1] and clipped. Trend depth uses MA-distance fractions; the risk_off
# momentum branch uses trailing-return fractions.
REGIME_TREND_MARGIN_SCALE = 0.10
REGIME_MOMENTUM_MARGIN_SCALE = 0.05

# RISK_OFF governance coupling: momentum coefficients of the current-time
# technical view are multiplied by this factor (1.0 = no dampening). The
# long-term structural view is regime-agnostic in v1; its own volatility drag
# already discounts regime risk there.
REGIME_RISKOFF_MOMENTUM_DAMPING = 0.5

# --- Outcome labels (V1) ---------------------------------------------------------
# Leakage-safe labels for every persisted decision, computed strictly from
# bars in (as_of, as_of + h] where h is counted in TRADING SESSIONS (no
# calendar ambiguity). The evaluator lives in core/labels.py; thresholds and
# versions live here so governance reads them from exactly one place.
#
# Boundary rule: a horizon's label is null until the horizon has fully
# elapsed RELATIVE TO THE DATA'S LATEST BAR (never wall-clock) — partially
# elapsed horizons are never emitted, so no partial-window leakage can enter
# training or evaluation. Labels use the same adjusted close series as the
# scoring path (single price truth); label thresholds are versioned via
# OUTCOME_LABEL_VERSION so changing them never rewrites history.

# v2: adverse_excursion is floored at 0.0. It answers "worst DRAWDOWN within
# the horizon", and a drawdown is a loss from entry — but on a gap-up every low
# sits above the entry close, so an unfloored min() returned the smallest GAIN
# and the realized label violated its own F1 bound (-1.0, 0.0). Measured: 9 of
# 300 real 20d labels. The honest answer on a gap-up is zero drawdown, not a
# positive one.
OUTCOME_LABEL_VERSION = "outcome-label-v2"

# Horizon name -> trading-session count after the as_of entry bar.
# F2 horizons. 120d and 252d are the long end the forecasting engine needs; a
# 252-session window is roughly a trading year, so a label at that horizon is
# only scorable once a full year of forward bars exists.
LABEL_HORIZON_SESSIONS = {
    "1d": 1, "5d": 5, "20d": 20, "60d": 60, "120d": 120, "252d": 252,
}

# label_20d_up is True when forward_return_20d is STRICTLY greater than this
# threshold (0.0 = any positive forward return).
OUTCOME_LABEL_UP_THRESHOLD = 0.0

# Calendar-day coverage the provider fetch must reach AHEAD of as_of, so a
# matured horizon is actually visible in the fetched frame.
#
# DERIVED, not hardcoded. It was 130 days, sized for the old 60-session maximum.
# Adding a 252-session horizon (~365 calendar days) without moving this would
# have left the longest labels silently unfetchable: the horizon would report
# pending forever because the future bars were never requested. Roughly 1.45
# calendar days per trading session, plus a holiday buffer.
LABEL_CALENDAR_DAYS_PER_SESSION = 1.45
LABEL_CALENDAR_BUFFER_DAYS = 45
LABEL_CALENDAR_COVERAGE_DAYS = int(
    max(LABEL_HORIZON_SESSIONS.values()) * LABEL_CALENDAR_DAYS_PER_SESSION
) + LABEL_CALENDAR_BUFFER_DAYS


def _validate_label_config() -> None:
    """Import-time guard: horizons are positive sessions, threshold sane."""
    if not LABEL_HORIZON_SESSIONS:
        raise ValueError("LABEL_HORIZON_SESSIONS must not be empty")
    for name, sessions in LABEL_HORIZON_SESSIONS.items():
        if not isinstance(sessions, int) or sessions < 1:
            raise ValueError(f"LABEL_HORIZON_SESSIONS[{name!r}] must be a positive int, got {sessions!r}")
    # The fetch must reach past the LONGEST horizon, or its labels can never
    # mature no matter how much time passes.
    longest_sessions = max(LABEL_HORIZON_SESSIONS.values())
    if LABEL_CALENDAR_COVERAGE_DAYS < longest_sessions:
        raise ValueError(
            f"LABEL_CALENDAR_COVERAGE_DAYS ({LABEL_CALENDAR_COVERAGE_DAYS}) is below "
            f"the longest horizon ({longest_sessions} sessions) — those labels could "
            f"never mature because the future bars are never fetched"
        )
    if OUTCOME_LABEL_UP_THRESHOLD < 0.0:
        raise ValueError("OUTCOME_LABEL_UP_THRESHOLD must be non-negative")


_validate_label_config()

# --- Walk-forward backtest (V2) --------------------------------------------------
# Validation infrastructure only — nothing here is a production trading path.
# Folding: train [t0, t1] -> embargo (>= max label horizon; derived below)
# -> validation -> advance; the final frozen configuration is evaluated once
# on a never-touched tail holdout. The embargo protects both features (PIT
# eligibility, already enforced) and labels (V1 boundary rule).

# DERIVED from the horizon set, not hardcoded. The embargo must cover the
# LONGEST label horizon or a validation fold can see bars that shaped a training
# row's outcome. It was 60, sized for the old 60-session maximum; F2's 252d
# horizon moves it to 252, and the fold/holdout scale with it so a fold still
# contains enough sessions to be worth validating on.
#
# The cost is real and worth stating: a fold now spans embargo + train +
# validation, so a walk-forward run needs roughly 630 sessions (~2.5 years) per
# fold. That is the price of a one-year forecast horizon, not a bug.
BACKTEST_EMBARGO_SESSIONS = max(LABEL_HORIZON_SESSIONS.values())
BACKTEST_FOLD_SESSIONS = max(120, BACKTEST_EMBARGO_SESSIONS)
BACKTEST_HOLDOUT_SESSIONS = max(60, BACKTEST_EMBARGO_SESSIONS // 2)

# Versioned harness strategy: decisions at bar t act at bar t+1 open.
# Enter long when the score is at/above BACKTEST_ENTER_SCORE and the posture
# is not NO_TRADE; exit when below BACKTEST_EXIT_SCORE; hold in between.
BACKTEST_STRATEGY_VERSION = "backtest-strategy-v1"
BACKTEST_ENTER_SCORE = 6.5
BACKTEST_EXIT_SCORE = 4.5

BACKTEST_INITIAL_CAPITAL = 100_000.0
BACKTEST_TRADE_NOTIONAL = 50_000.0
BACKTEST_AVG_DOLLAR_VOLUME_WINDOW = 20


# --- Historical universe ledger (V6) ---------------------------------------------
# Survivorship-safe membership: a universe is a set of membership intervals,
# not a list of today's tickers. The ledger lives in core/universe.py; the
# version lives here so manifests and the shared-contract verifier read it
# from exactly one place (same discipline as every other version constant).
UNIVERSE_VERSION = "universe-ledger-v1"

# --- Backtest framing (V8) ---------------------------------------------------------
# A backtest is evidence about historical behavior under explicit assumptions —
# NOT proof the future behaves the same way. The framing contract lives in
# core/framing.py; the version lives here so manifests and the shared-contract
# verifier read it from exactly one place.
FRAMING_VERSION = "backtest-framing-v1"

# --- Canonical feature registry (M1) -------------------------------------------
# Every feature that enters a production model must be registered in
# core/feature_registry.py with complete metadata. The version lives here so
# manifests and the shared-contract verifier read it from exactly one place.
FEATURE_REGISTRY_VERSION = "feature-registry-v1"
# Per-domain feature-definition versions: a domain's factors evolve on their
# own cadence, so each carries its own version constant (read from exactly
# one place, same discipline as MARKET_FEATURE_VERSION).
FUNDAMENTAL_FEATURE_VERSION = "fundamental-feature-v1"


def _validate_backtest_config() -> None:
    """Import-time guard: the embargo must cover the longest label horizon."""
    max_horizon = max(LABEL_HORIZON_SESSIONS.values())
    if BACKTEST_EMBARGO_SESSIONS < max_horizon:
        raise ValueError(
            f"BACKTEST_EMBARGO_SESSIONS ({BACKTEST_EMBARGO_SESSIONS}) must be >= the max "
            f"label horizon ({max_horizon}) — walk-forward folds would leak label information."
        )
    if BACKTEST_ENTER_SCORE <= BACKTEST_EXIT_SCORE:
        raise ValueError("BACKTEST_ENTER_SCORE must be above BACKTEST_EXIT_SCORE")
    if BACKTEST_FOLD_SESSIONS < 1 or BACKTEST_HOLDOUT_SESSIONS < 1:
        raise ValueError("fold and holdout session counts must be positive")


_validate_backtest_config()


# --- Paper-trading order engine (V6) --------------------------------------------
# Simulation only. The paper engine turns accepted decisions into order
# intents and simulated fills so later sprints (L1 outcome closure) have
# real trade evidence to close against.
#
# Execution-mode constants. LIVE_APPROVED exists ONLY as a schema value with
# no construction path anywhere in the codebase — the live branch raises
# unconditionally. It is deliberately not a config flag away from working:
# turning it on requires writing new code, which is the point.
PAPER_ENGINE_VERSION = "paper-engine-v1"

EXECUTION_MODE_PAPER = "PAPER_ONLY"
EXECUTION_MODE_LIVE_DISABLED = "LIVE_DISABLED"
EXECUTION_MODE_LIVE_APPROVED = "LIVE_APPROVED"

# The permanent default. Nothing in the codebase assigns any other value.
EXECUTION_MODE = EXECUTION_MODE_LIVE_DISABLED

# Only a decision in this posture may produce an order intent. ANALYSIS_ONLY
# and NO_TRADE decisions are logged as rejections instead (the paper log must
# show why nothing happened, per the V6 acceptance criteria).
PAPER_TRADABLE_MODE = "PAPER"

# Fixed notional per paper order. A sizing model is Sprint R's job
# (correlation-aware sizing); until then every intent is the same size so
# that fill evidence is not confounded by an unvalidated sizing rule.
PAPER_ORDER_NOTIONAL = 10_000.0

PAPER_ORDER_SCHEMA_VERSION = "paper-order-v1"


def _validate_paper_config() -> None:
    """Import-time guard: the engine must be unable to reach a live posture."""
    if EXECUTION_MODE != EXECUTION_MODE_LIVE_DISABLED:
        raise ValueError(
            f"EXECUTION_MODE must be {EXECUTION_MODE_LIVE_DISABLED!r}; the paper "
            f"engine has no live execution path (got {EXECUTION_MODE!r})"
        )
    if PAPER_ORDER_NOTIONAL <= 0:
        raise ValueError("PAPER_ORDER_NOTIONAL must be positive")


_validate_paper_config()


# --- Official training dataset builder (M2) --------------------------------------
# The ONLY sanctioned dataset generator. Every training row is
# prediction_time -> information available at prediction_time -> features ->
# future outcome, and the whole dataset carries a deterministic hash so a
# trial can name exactly the data it trained on.
TRAINING_DATASET_VERSION = "training-dataset-v1"

# A row is emitted only when its label horizon has fully matured. An
# unmatured horizon is not a zero and not a neutral — it is an excluded row,
# recorded in the build report so coverage is never silently thinned.
TRAINING_DATASET_SCHEMA_VERSION = "training-row-v1"

# The horizon a row's supervised target is drawn from. 20d is the only
# horizon carrying a directional label (label_up) and an adverse-excursion
# measure in the V1 label contract, so it is the default target.
TRAINING_DEFAULT_TARGET_HORIZON = "20d"

# Fail-closed row admission. A row must carry every declared feature with a
# usable value; a missing feature excludes the row rather than imputing a
# neutral (master context section 32: never silently replace missing values
# with neutral evidence).
TRAINING_REQUIRE_COMPLETE_FEATURES = True


def _validate_training_dataset_config() -> None:
    """Import-time guard: the default target horizon must be a real horizon."""
    if TRAINING_DEFAULT_TARGET_HORIZON not in LABEL_HORIZON_SESSIONS:
        raise ValueError(
            f"TRAINING_DEFAULT_TARGET_HORIZON {TRAINING_DEFAULT_TARGET_HORIZON!r} is not "
            f"a declared label horizon (known: {sorted(LABEL_HORIZON_SESSIONS)})"
        )


_validate_training_dataset_config()


# --- Model registry and artifact tracking (M2) -----------------------------------
# A model referenced by a live decision must be `approved`. The registry is
# the single source of truth for which model versions exist, what they were
# trained on, and whether governance has cleared them for the score path.
MODEL_REGISTRY_VERSION = "model-registry-v1"

# Lifecycle. A model is born CANDIDATE and can only reach APPROVED through a
# promotion that carries an out-of-sample comparison against the incumbent
# plus a human approver. RETIRED is terminal for the score path but never
# deletes the artifact — historical decisions must stay explainable.
MODEL_STATUS_CANDIDATE = "candidate"
MODEL_STATUS_APPROVED = "approved"
MODEL_STATUS_RETIRED = "retired"
MODEL_STATUSES = (MODEL_STATUS_CANDIDATE, MODEL_STATUS_APPROVED, MODEL_STATUS_RETIRED)

# Only these statuses may back a live decision.
MODEL_LIVE_ELIGIBLE_STATUSES = (MODEL_STATUS_APPROVED,)

# The deterministic-scorer entries that back today's score path. These are
# rule-based scorers, not trained artifacts, so they carry no dataset hash —
# but they are registered and approved like anything else, because the
# binding rule is about GOVERNANCE not about whether gradient descent was
# involved. The orchestrator resolves these through the registry rather than
# hardcoding the strings.
MODEL_REGISTRY_SEED_APPROVER = "governance_baseline"


# --- Research trial registry (M3) ------------------------------------------------
# Every research experiment is recorded BEFORE its metrics are known. The
# registry exists to protect against uncontrolled experimentation and
# cherry-picking: an unrecorded trial is not evidence, and a hypothesis
# written after seeing the result is not a hypothesis.
TRIAL_REGISTRY_VERSION = "trial-registry-v1"

# Lifecycle. A trial is REGISTERED (hypothesis + full configuration, no
# metrics yet), then exactly once COMPLETED (metrics attached) or ABANDONED
# (recorded with a reason). A completed trial is immutable.
TRIAL_STATUS_REGISTERED = "registered"
TRIAL_STATUS_COMPLETED = "completed"
TRIAL_STATUS_ABANDONED = "abandoned"
TRIAL_STATUSES = (
    TRIAL_STATUS_REGISTERED, TRIAL_STATUS_COMPLETED, TRIAL_STATUS_ABANDONED,
)

# Recognised validation schemes. "none" is deliberately absent: a trial with
# no validation scheme is not a trial.
TRIAL_VALIDATION_SCHEMES = (
    "walk_forward", "walk_forward_embargo", "expanding_window", "holdout",
)

# The pre-registered primary metric must be declared at registration time and
# cannot be changed afterwards. Changing which metric counts after seeing the
# results is the definition of cherry-picking.
TRIAL_METRIC_LOCK_ENFORCED = True


# --- Offline training pipeline (board M3) ----------------------------------------
# The only sanctioned trainer. It consumes M2-DS datasets, splits them with
# the V2 walk-forward harness (never sklearn's own splitters, which are not
# time-safe), and records everything needed to reproduce the artifact.
TRAINING_PIPELINE_VERSION = "training-pipeline-v1"

# Fixed seed for every estimator that accepts one. Recorded in the run
# manifest; changing it is a new run, not a re-run.
TRAINING_DEFAULT_SEED = 42

# The baseline families M3 trains. Sequence/temporal models are explicitly
# out of scope until these survive V2 validation (board M3, and master
# context section 32: do not build a complex network before proving simple
# baselines).
TRAINING_BASELINE_FAMILIES = ("linear", "tree", "boosting")

# Artifacts are hashed over their PARAMETERS, not the pickle bytes: joblib
# output embeds library versions and is not byte-stable across environments,
# which would make the M8 reproducibility guarantee untestable in CI.
TRAINING_ARTIFACT_HASH_BASIS = "fitted_parameters"


# --- Baseline model suite (M4) ---------------------------------------------------
# "No advanced model is promoted without beating the incumbent out-of-sample."
# The incumbent is the best SIMPLE baseline, not the best model: a trained
# model must clear historical_mean / momentum / mean_reversion before it is
# interesting at all.
BASELINE_SUITE_VERSION = "baseline-suite-v1"

# A candidate must beat the incumbent by at least this much on the primary
# metric. Beating it by 0.0001 is noise, and promoting on noise is how a
# backtest-overfit model reaches production.
BASELINE_PROMOTION_MARGIN = 0.02

# ...and must win at least this share of paired folds. A single lucky fold is
# not an edge; the win has to be consistent across the walk-forward windows.
BASELINE_MIN_WINNING_FOLD_RATIO = 0.6


def _validate_baseline_suite_config() -> None:
    """Import-time guard: the promotion bars must be meaningful."""
    if BASELINE_PROMOTION_MARGIN <= 0:
        raise ValueError("BASELINE_PROMOTION_MARGIN must be positive — a zero margin promotes noise")
    if not 0.5 < BASELINE_MIN_WINNING_FOLD_RATIO <= 1.0:
        raise ValueError(
            "BASELINE_MIN_WINNING_FOLD_RATIO must be above 0.5 (a majority) and at most 1.0"
        )


_validate_baseline_suite_config()


# --- Model artifact registry extension (M5) --------------------------------------
# M5 completes the M2 entry with the provenance a TRAINED artifact needs:
# code commit, artifact hash, calibration version, seed, hyperparameters, and
# the forecast contract (target, horizon, universe) the model was fitted for.
MODEL_ARTIFACT_VERSION = "model-artifact-v1"

# What a model predicts. Declared per entry so two models over the same
# features are never confused for one another — an expected-return model and
# a direction classifier are not interchangeable.
FORECAST_TARGET_RETURN = "expected_return"
FORECAST_TARGET_DIRECTION = "probability_up"
FORECAST_TARGET_VOLATILITY = "expected_volatility"
# F1 completes the set: a return distribution, the downside a position would
# have sat through, and performance against a benchmark. Declared here rather
# than in a new table so the M5 registry validator keeps ONE vocabulary.
FORECAST_TARGET_DISTRIBUTION = "return_distribution"
FORECAST_TARGET_DOWNSIDE = "adverse_excursion"
FORECAST_TARGET_RELATIVE = "probability_outperform"
FORECAST_TARGETS = (
    FORECAST_TARGET_RETURN,
    FORECAST_TARGET_DIRECTION,
    FORECAST_TARGET_VOLATILITY,
    FORECAST_TARGET_DISTRIBUTION,
    FORECAST_TARGET_DOWNSIDE,
    FORECAST_TARGET_RELATIVE,
)

# A trained model must name its calibration state. "uncalibrated" is an
# explicit, honest value — never a missing field — and M6 will refuse to
# expose an uncalibrated probability as if it were calibrated.
CALIBRATION_UNCALIBRATED = "uncalibrated"


# --- Calibration and uncertainty (M6 / board M4) ---------------------------------
# "Never expose arbitrary probability numbers as if they were calibrated."
# A raw model score is not a probability. It becomes one only by being mapped
# through a calibration fitted on VALIDATION folds, and that map is versioned
# and travels with the artifact.
CALIBRATION_VERSION = "calibration-v1"

# Isotonic preferred; Platt (logistic) fallback for small folds, because
# isotonic regression overfits badly on thin data.
CALIBRATION_METHOD_ISOTONIC = "isotonic"
CALIBRATION_METHOD_PLATT = "platt"
CALIBRATION_METHODS = (CALIBRATION_METHOD_ISOTONIC, CALIBRATION_METHOD_PLATT)

# Below this many out-of-sample observations, isotonic is not trustworthy and
# the fitter falls back to Platt. Recorded in the artifact so a reader knows
# which method actually ran and why.
CALIBRATION_MIN_ISOTONIC_SAMPLES = 100

# A calibration fitted on fewer than this many observations is refused
# outright — there is no honest probability to expose.
CALIBRATION_MIN_SAMPLES = 30

# Reliability-curve bin count. Ten deciles is the conventional default and
# keeps per-bin counts meaningful at our sample sizes.
CALIBRATION_RELIABILITY_BINS = 10

# Prediction intervals are reported at this level. 0.80 (10th-90th) rather
# than 0.95: at fold counts this small, a 95% band derived from fold
# dispersion would imply precision the data cannot support.
PREDICTION_INTERVAL_LEVEL = 0.80


def _validate_calibration_config() -> None:
    """Import-time guard: the calibration thresholds must be coherent."""
    if CALIBRATION_MIN_SAMPLES < 2:
        raise ValueError("CALIBRATION_MIN_SAMPLES must allow at least two observations")
    if CALIBRATION_MIN_ISOTONIC_SAMPLES < CALIBRATION_MIN_SAMPLES:
        raise ValueError(
            "CALIBRATION_MIN_ISOTONIC_SAMPLES must be >= CALIBRATION_MIN_SAMPLES — "
            "the isotonic threshold cannot sit below the refusal threshold"
        )
    if not 0.0 < PREDICTION_INTERVAL_LEVEL < 1.0:
        raise ValueError("PREDICTION_INTERVAL_LEVEL must be strictly between 0 and 1")
    if CALIBRATION_RELIABILITY_BINS < 2:
        raise ValueError("CALIBRATION_RELIABILITY_BINS must be at least 2")


_validate_calibration_config()


# --- Champion / challenger (M7) --------------------------------------------------
# One Champion per forecast contract, any number of Challengers, and every
# new model begins in SHADOW. A model's ROLE is declared, never inferred:
# before M7 the "incumbent" was whichever approved version sorted last by
# string, which made the most important question in the registry an accident
# of naming.
CHAMPION_CHALLENGER_VERSION = "champion-challenger-v1"

# SHADOW: runs alongside production, predictions recorded, never consulted.
#   Every model starts here — there is no path that registers a model
#   directly into a serving role.
# CHALLENGER: a shadow model that has earned evaluation against the champion.
# CHAMPION: the single model serving a forecast contract.
MODEL_ROLE_SHADOW = "shadow"
MODEL_ROLE_CHALLENGER = "challenger"
MODEL_ROLE_CHAMPION = "champion"
MODEL_ROLES = (MODEL_ROLE_SHADOW, MODEL_ROLE_CHALLENGER, MODEL_ROLE_CHAMPION)

# Only a champion may back a live decision. A challenger is measured, not
# trusted; a shadow model is not even consulted.
MODEL_SERVING_ROLES = (MODEL_ROLE_CHAMPION,)

# The role every newly registered model takes, with no way to override it at
# registration time.
MODEL_DEFAULT_ROLE = MODEL_ROLE_SHADOW

# A shadow model must accumulate this many recorded out-of-sample
# observations before it may be promoted to challenger. Promotion from
# shadow is not a formality — an unmeasured model has earned nothing.
SHADOW_MIN_OBSERVATIONS = 100


def _validate_champion_challenger_config() -> None:
    """Import-time guard: the role model must stay coherent."""
    if MODEL_DEFAULT_ROLE != MODEL_ROLE_SHADOW:
        raise ValueError(
            f"MODEL_DEFAULT_ROLE must be {MODEL_ROLE_SHADOW!r} — every new model "
            f"begins in shadow (M7), got {MODEL_DEFAULT_ROLE!r}"
        )
    if set(MODEL_SERVING_ROLES) - set(MODEL_ROLES):
        raise ValueError("MODEL_SERVING_ROLES must be a subset of MODEL_ROLES")
    if MODEL_ROLE_SHADOW in MODEL_SERVING_ROLES:
        raise ValueError("a shadow model must never be a serving role")
    if MODEL_ROLE_CHALLENGER in MODEL_SERVING_ROLES:
        raise ValueError(
            "a challenger is measured, not trusted — it must never serve live"
        )
    if SHADOW_MIN_OBSERVATIONS < 1:
        raise ValueError("SHADOW_MIN_OBSERVATIONS must be positive")


_validate_champion_challenger_config()


# --- ML reproducibility gate (M8) ------------------------------------------------
# "Same data, features, seed, code, configuration must produce identical model
# artifacts/predictions WITHIN EXPLICITLY DEFINED REPRODUCIBILITY GUARANTEES."
#
# The last clause is the important one. A blanket "bit-identical forever"
# claim would be false the moment numpy changes a summation order, so this
# states exactly what is guaranteed and what is not.
REPRODUCIBILITY_VERSION = "reproducibility-v1"

# GUARANTEED — same process, same environment, same inputs:
#   * identical fitted parameters, so an identical artifact hash;
#   * identical predictions, metrics and run hash;
#   * identical results across separate processes on the same machine.
#
# NOT GUARANTEED — and deliberately so:
#   * bit-identical artifacts across different library versions. BLAS
#     summation order and estimator internals change between releases; the
#     environment block records the versions so a divergence can be explained
#     rather than merely detected.
#   * bit-identical artifacts across platforms, for the same reason.
#   * stability across a code change that alters a formula. That is what the
#     *_VERSION constants exist to make visible.
REPRODUCIBILITY_GUARANTEES = (
    "same_process_identical_parameters",
    "cross_process_identical_parameters",
    "identical_predictions",
    "identical_metrics",
    "identical_run_hash",
)

REPRODUCIBILITY_NON_GUARANTEES = (
    "cross_library_version_bitwise",
    "cross_platform_bitwise",
    "stability_across_formula_changes",
)

# Numeric tolerance when comparing predictions. Zero: within one environment
# the same inputs must produce the SAME floats, not merely close ones. A
# tolerance here would hide exactly the nondeterminism this gate exists to
# catch.
REPRODUCIBILITY_TOLERANCE = 0.0

# The environment fields recorded with every run, so a cross-environment
# divergence can be attributed instead of guessed at.
REPRODUCIBILITY_ENVIRONMENT_KEYS = ("python", "numpy", "sklearn", "platform")


def _validate_reproducibility_config() -> None:
    """Import-time guard: the guarantees must not contradict themselves."""
    overlap = set(REPRODUCIBILITY_GUARANTEES) & set(REPRODUCIBILITY_NON_GUARANTEES)
    if overlap:
        raise ValueError(
            f"a property cannot be both guaranteed and not guaranteed: {sorted(overlap)}"
        )
    if REPRODUCIBILITY_TOLERANCE != 0.0:
        raise ValueError(
            "REPRODUCIBILITY_TOLERANCE must be exactly 0.0 — a tolerance would "
            "hide the nondeterminism this gate exists to catch"
        )
    if not REPRODUCIBILITY_ENVIRONMENT_KEYS:
        raise ValueError("at least one environment key must be recorded")


_validate_reproducibility_config()


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


# --- Forecast targets (F1) -------------------------------------------------------
# What the forecasting engine may be asked to predict, and what each answer
# MEANS. A target is not just a name: it declares the kind of quantity it is,
# whether it is a probability (and therefore must pass through a fitted M6
# calibration before anyone sees it), its bounds, and the realized label field
# it is scored against.
#
# The label link is the part that keeps F1 honest. Every target below resolves
# to a field the V1 label builder already produces, so a forecast can always be
# compared to what actually happened. A target with no realized counterpart
# cannot be validated, and an unvalidatable forecast is an opinion.

FORECAST_TARGET_CONTRACT_VERSION = "forecast-target-v1"

# Kinds. A probability is not a return and must not be rendered like one.
FORECAST_KIND_PROBABILITY = "probability"
FORECAST_KIND_RETURN = "return"
FORECAST_KIND_VOLATILITY = "volatility"
FORECAST_KIND_DISTRIBUTION = "distribution"
FORECAST_KINDS: tuple[str, ...] = (
    FORECAST_KIND_PROBABILITY,
    FORECAST_KIND_RETURN,
    FORECAST_KIND_VOLATILITY,
    FORECAST_KIND_DISTRIBUTION,
)

# Per-target contract. `requires_calibration` is the load-bearing field: a
# target marked True may only be emitted through `calibration.calibrated_
# probability`, which refuses without a fitted map. That is what makes "never
# expose arbitrary probability numbers as if they were calibrated" enforceable
# rather than aspirational.
#
# `requires_benchmark` marks the one target that cannot be computed from the
# stock alone -- C3 supplies the benchmark, and without one the target is
# UNAVAILABLE rather than silently scored against nothing.
FORECAST_TARGET_CONTRACTS: dict[str, dict[str, object]] = {
    FORECAST_TARGET_DIRECTION: {
        "kind": FORECAST_KIND_PROBABILITY,
        "unit": "probability",
        "bounds": (0.0, 1.0),
        "requires_calibration": True,
        "requires_benchmark": False,
        "label_field": "label_up",
        "question": "P(return > 0) over the horizon",
    },
    FORECAST_TARGET_RETURN: {
        "kind": FORECAST_KIND_RETURN,
        "unit": "ratio",
        "bounds": (-1.0, None),
        "requires_calibration": False,
        "requires_benchmark": False,
        "label_field": "forward_return",
        "question": "E(return) over the horizon",
    },
    FORECAST_TARGET_DISTRIBUTION: {
        "kind": FORECAST_KIND_DISTRIBUTION,
        "unit": "probability",
        "bounds": (0.0, 1.0),
        "requires_calibration": True,
        "requires_benchmark": False,
        "label_field": "forward_return",
        "question": "P(return within a stated interval) over the horizon",
    },
    FORECAST_TARGET_DOWNSIDE: {
        "kind": FORECAST_KIND_RETURN,
        "unit": "ratio",
        # An adverse excursion is the worst drawdown INSIDE the window, so it
        # is never positive: a forecast claiming otherwise is malformed.
        "bounds": (-1.0, 0.0),
        "requires_calibration": False,
        "requires_benchmark": False,
        "label_field": "adverse_excursion",
        "question": "expected worst drawdown within the horizon",
    },
    FORECAST_TARGET_VOLATILITY: {
        "kind": FORECAST_KIND_VOLATILITY,
        "unit": "ratio",
        # Volatility is a dispersion, so it cannot be negative.
        "bounds": (0.0, None),
        "requires_calibration": False,
        "requires_benchmark": False,
        "label_field": "realized_vol",
        "question": "expected realized volatility over the horizon",
    },
    FORECAST_TARGET_RELATIVE: {
        "kind": FORECAST_KIND_PROBABILITY,
        "unit": "probability",
        "bounds": (0.0, 1.0),
        "requires_calibration": True,
        "requires_benchmark": True,
        # NOT SCORABLE TODAY, and saying so is the point. This target needs the
        # stock return MINUS the benchmark's over the same window, and the V1
        # label set carries no benchmark return at all. Pointing `label_field`
        # at `forward_return` made the target look scorable while silently
        # scoring it against the raw stock return — identical to
        # expected_return, and measuring nothing about outperformance.
        # `label_field` stays declared so the contract shape is uniform;
        # `label_unavailable` is what a consumer must read.
        "label_field": "forward_return",
        "label_unavailable": (
            "the V1 label set carries no benchmark return, so "
            "P(stock > benchmark) has no realized counterpart and cannot be "
            "scored; a relative-return label is owned by a later sprint"
        ),
        "question": "P(stock return > benchmark return) over the horizon",
    },
}


def _validate_forecast_target_contracts() -> None:
    """Import-time guard: every declared target must carry a full contract."""
    if set(FORECAST_TARGET_CONTRACTS) != set(FORECAST_TARGETS):
        raise ValueError(
            "FORECAST_TARGET_CONTRACTS must cover exactly FORECAST_TARGETS: "
            f"missing={sorted(set(FORECAST_TARGETS) - set(FORECAST_TARGET_CONTRACTS))} "
            f"extra={sorted(set(FORECAST_TARGET_CONTRACTS) - set(FORECAST_TARGETS))}"
        )
    for name, contract in FORECAST_TARGET_CONTRACTS.items():
        if contract["kind"] not in FORECAST_KINDS:
            raise ValueError(f"target {name!r} declares an unknown kind")
        if not contract.get("label_field"):
            raise ValueError(
                f"target {name!r} names no realized label field — a forecast that "
                f"cannot be compared to an outcome is an opinion"
            )
        if not contract.get("question"):
            raise ValueError(f"target {name!r} does not state the question it answers")
        low, high = contract["bounds"]
        if low is not None and high is not None and low >= high:
            raise ValueError(f"target {name!r} declares empty bounds")
        # A probability that skips calibration is exactly the failure M6 exists
        # to prevent, so the two fields may never disagree.
        is_probability = contract["kind"] in (
            FORECAST_KIND_PROBABILITY, FORECAST_KIND_DISTRIBUTION,
        )
        if is_probability and not contract["requires_calibration"]:
            raise ValueError(
                f"target {name!r} is a probability but does not require calibration"
            )
        if not is_probability and contract["requires_calibration"]:
            raise ValueError(
                f"target {name!r} is not a probability but requires calibration"
            )


_validate_forecast_target_contracts()


# --- Forecast horizons (F2) ------------------------------------------------------
# The horizon set the forecasting engine answers over. F2's requirement is
# "at minimum 1D, 5D, 20D, 60D, 120D, 252D", and every one of them resolves to
# a V1 label horizon so a forecast at that horizon can be scored against a
# realized outcome.
#
# The horizons ARE the label horizons, deliberately. A forecast horizon with no
# matching label could never be validated, so rather than keeping two tables in
# sync, F2 reads LABEL_HORIZON_SESSIONS and pins the F2 minimum against it.

FORECAST_HORIZON_CONTRACT_VERSION = "forecast-horizon-v1"

# Ordered shortest-first, so a joint forecast (F3) always reads in time order.
FORECAST_HORIZONS: tuple[str, ...] = tuple(
    sorted(LABEL_HORIZON_SESSIONS, key=lambda name: LABEL_HORIZON_SESSIONS[name])
)

# The set F2 requires. Pinned separately from FORECAST_HORIZONS so dropping one
# is an import-time failure rather than a silently narrower product.
FORECAST_REQUIRED_HORIZONS: tuple[str, ...] = ("1d", "5d", "20d", "60d", "120d", "252d")

# Roughly how many calendar days a horizon spans, for reporting when an outcome
# becomes knowable. Derived, so it cannot drift from the session counts.
FORECAST_HORIZON_CALENDAR_DAYS: dict[str, int] = {
    name: int(sessions * LABEL_CALENDAR_DAYS_PER_SESSION)
    for name, sessions in LABEL_HORIZON_SESSIONS.items()
}

# Horizons beyond this are LONG: they need materially more history to validate,
# and a thin dataset shows up there first. Reported so a caller can see which
# part of a joint forecast rests on less evidence.
FORECAST_LONG_HORIZON_SESSIONS = 100


def _validate_forecast_horizons() -> None:
    """Import-time guard: the F2 minimum must actually be met."""
    missing = [h for h in FORECAST_REQUIRED_HORIZONS if h not in FORECAST_HORIZONS]
    if missing:
        raise ValueError(
            f"F2 requires horizons {missing} which are not declared — a forecast "
            f"product narrower than its contract is a silent downgrade"
        )
    for name in FORECAST_HORIZONS:
        if name not in LABEL_HORIZON_SESSIONS:
            raise ValueError(
                f"forecast horizon {name!r} has no label horizon — a forecast that "
                f"cannot be scored against an outcome is an opinion"
            )
    sessions = [LABEL_HORIZON_SESSIONS[name] for name in FORECAST_HORIZONS]
    if sessions != sorted(sessions):
        raise ValueError("FORECAST_HORIZONS must be ordered shortest-first")


_validate_forecast_horizons()


# --- Joint forecast (F3) ---------------------------------------------------------
# One object reporting every target across every horizon together, with the
# uncertainty that belongs to each cell. F3 DEFINES and VALIDATES the joint
# contract; it fits no models, because none exist yet (the measured incumbent
# is a momentum baseline at 0.575 directional accuracy).
#
# The shape rule that carries the design: a cell carries a `value` key IF AND
# ONLY IF its status is OK. Not `value: None` — the key is ABSENT. The
# dashboard's existing idiom is `Number(x ?? 0)`, so a null P(up) would coalesce
# to 0.0% and render as CERTAIN DOWN — the most dangerous possible misreading,
# produced by defensive-looking code. With the key absent the failure is loud at
# the boundary instead of plausible on the screen.

JOINT_FORECAST_CONTRACT_VERSION = "joint-forecast-v1"
JOINT_FORECAST_PIPELINE_VERSION = "joint-forecast-pipeline-v1"

# Snapshot-level status.
JOINT_STATUS_OK = "OK"
JOINT_STATUS_PARTIAL = "PARTIAL"
JOINT_STATUS_NO_MODEL = "NO_MODEL"
JOINT_STATUS_UNAVAILABLE = "UNAVAILABLE"

# Cell-level status, in STRICT precedence order — first match wins. Precedence
# is declared as data so the reason a cell is empty is never ambiguous, and so
# a reader can see which explanation outranks which.
CELL_STATUS_UNAVAILABLE = "UNAVAILABLE"          # no label set at all
CELL_STATUS_PENDING = "PENDING"                  # the horizon's window has not closed
CELL_STATUS_LABEL_UNBACKED = "LABEL_UNBACKED"    # no realized counterpart at this horizon
CELL_STATUS_NEEDS_BENCHMARK = "NEEDS_BENCHMARK"  # relative target, no benchmark supplied
CELL_STATUS_DEGENERATE_INTERVAL = "DEGENERATE_INTERVAL"  # zero-width uncertainty
CELL_STATUS_UNCALIBRATED = "UNCALIBRATED"        # probability with no fitted map
CELL_STATUS_NO_MODEL = "NO_MODEL"                # nothing to produce a value
CELL_STATUS_OK = "OK"

JOINT_CELL_PRECEDENCE: tuple[str, ...] = (
    CELL_STATUS_UNAVAILABLE,
    CELL_STATUS_PENDING,
    CELL_STATUS_LABEL_UNBACKED,
    CELL_STATUS_NEEDS_BENCHMARK,
    CELL_STATUS_DEGENERATE_INTERVAL,
    CELL_STATUS_UNCALIBRATED,
    CELL_STATUS_NO_MODEL,
    CELL_STATUS_OK,
)

# An interval narrower than this is not uncertainty, it is arithmetic. MEASURED:
# `prediction_interval([0.55, 0.55, 0.55])` returns width 0.0 at folds=3, so a
# fold-COUNT floor passes its own check while publishing perfect certainty.
# Width is the guard; the count is decorative.
JOINT_MIN_INTERVAL_WIDTH = 1e-9

# Coherence verdicts.
JOINT_COHERENCE_OK = "OK"
JOINT_COHERENCE_FLAGGED = "FLAGGED"
JOINT_COHERENCE_VIOLATED = "VIOLATED"
JOINT_COHERENCE_NOT_EVALUATED = "NOT_EVALUATED"

# Coherence rules F3 deliberately does NOT enforce, with the measurement that
# rejected each. Declared as DATA so the reasoning is auditable and so a future
# reader cannot re-add one believing it was merely overlooked.
JOINT_REJECTED_RULES: dict[str, str] = {
    "monotonic_return": (
        "returns must grow with horizon — REJECTED: real NVDA labels at "
        "2024-06-15 run -0.7%, -10.4%, -4.2%, -11.4%, +10.0%, +10.3% across "
        "1d..252d. A stock falling for 60 sessions and recovering by 120 is "
        "ordinary, not incoherent."
    ),
    "direction_matches_return": (
        "sign(E[return]) must agree with P(up) > 0.5 — REJECTED: a skewed "
        "payoff (75% of +2%, 25% of -9%) gives P(up)=0.73 with E[return]="
        "-0.0075. Both are correct simultaneously."
    ),
    "volatility_grows_with_horizon": (
        "realized volatility must grow with horizon — REJECTED: 1d volatility "
        "is structurally None (labels.py needs >= 2 sessions for a dispersion), "
        "so the rule cannot even be evaluated at the short end."
    ),
    "adverse_excursion_below_return": (
        "adverse_excursion <= min(0, expected_return) — REJECTED: a gap-up "
        "produces a POSITIVE adverse excursion (entry 100, lows 101/103/105 "
        "-> +0.01), and 9 of 300 real 20d labels violate it. The realized "
        "label is ground truth; a rule that calls ground truth malformed is "
        "the wrong rule."
    ),
}


def _validate_joint_forecast_config() -> None:
    """Import-time guard: the cell vocabulary must be total and ordered."""
    declared = {
        CELL_STATUS_UNAVAILABLE, CELL_STATUS_PENDING, CELL_STATUS_LABEL_UNBACKED,
        CELL_STATUS_NEEDS_BENCHMARK, CELL_STATUS_DEGENERATE_INTERVAL,
        CELL_STATUS_UNCALIBRATED, CELL_STATUS_NO_MODEL, CELL_STATUS_OK,
    }
    if set(JOINT_CELL_PRECEDENCE) != declared:
        raise ValueError("JOINT_CELL_PRECEDENCE must cover exactly the declared cell statuses")
    if len(JOINT_CELL_PRECEDENCE) != len(declared):
        raise ValueError("JOINT_CELL_PRECEDENCE contains a duplicate")
    if JOINT_CELL_PRECEDENCE[-1] != CELL_STATUS_OK:
        raise ValueError("OK must be the LAST precedence entry — every refusal outranks it")
    if JOINT_MIN_INTERVAL_WIDTH <= 0.0:
        raise ValueError("the minimum interval width must be strictly positive")
    if not JOINT_REJECTED_RULES:
        raise ValueError(
            "the rejected-rule register must not be empty — the reasoning for "
            "NOT enforcing a rule is as load-bearing as the rules that are"
        )


_validate_joint_forecast_config()


# ---------------------------------------------------------------------------
# Sprint F4 - Conditional forecasting
# ---------------------------------------------------------------------------
# F4 answers "P(+5% in 20D | bullish) vs P(+5% in 20D | stress)". The hard part
# is not computing a conditional rate; it is that conditioning SHREDS the
# sample. 6 targets x 6 horizons x 5 regimes = 180 cells drawn from one history.
#
# MEASURED, 5y SPY, 191 PIT-correct sessions:
#     bullish 143 | risk_off 32 | range 9 | bearish 5 | stress 2
# The roadmap's own example condition (stress) has TWO observations, and reads
# P(up)=1.00, P(+5%)=1.00, mean +13.64% - the most confident-looking and least
# trustworthy cell in the grid.
#
# WHY NOT ONE GLOBAL RULE. A single N floor was measured against the real
# slices and both directions fail:
#   - a floor at N>=30 silences range, bearish AND stress - 3 of 5 regimes.
#     The grid then cannot answer the question F4 exists to answer, which is a
#     loss of ACCURACY, not a conservative default.
#   - no floor at all publishes "stress: rose 100% of the time, +13.6%" from
#     N=2, which is a loss of RELIABILITY.
# So the estimator and the STRENGTH OF CLAIM are selected per cell, from the
# evidence that cell actually holds. Every cell says the strongest true thing
# it can support, and none says more.
#
# THE TIERS, and the measurement that placed each boundary (40k binomial draws
# per N, true rate 0.20; see scripts/check_conditional_forecast.py):
#
#   POINT (N >= 40): a point estimate is publishable.
#       |error| > 0.15 in 1.6% of draws at N=40, vs 3.7% at N=30 and 22.7% at
#       N=10. Mean Wilson width 0.238.
#   INTERVAL (N >= 8): no point estimate; the interval IS the answer.
#       Wilson coverage holds at every N measured (0.92-1.00 down to N=2), so
#       an interval here is HONEST - it is merely wide, and a wide interval is
#       self-limiting in a way a wrong point estimate is not. N=8 is where mean
#       width first drops below CONDITIONAL_MAX_INTERVAL_WIDTH.
#   DIRECTIONAL (N >= 20 AND the interval excludes the base rate): the cell may
#       only say "higher/lower than unconditional", never a number.
#       Detection of a LARGE 15pt effect: 24% at N=10, 40% at N=20, 49% at
#       N=30, 69% at N=40. Below N=20 a directional claim is a coin flip
#       dressed as a finding, so it is not offered.
#   INSUFFICIENT (everything else): the cell reports N and refuses.
#
# NOTE the deliberate inversion: DIRECTIONAL demands MORE evidence than
# INTERVAL, though it says less. An interval that is too wide advertises its
# own weakness; a directional claim that failed to detect an effect looks
# exactly like one that detected nothing. Ordering these by apparent
# "strength of wording" would have been backwards.
CONDITIONAL_FORECAST_VERSION = "conditional-forecast-v1"
CONDITIONAL_CONTRACT_VERSION = "conditional-contract-v1"

# Claim strengths, weakest-first. The ORDER is the precedence used when
# several tiers qualify: the LAST qualifying tier wins, so a cell always makes
# the strongest claim its evidence supports.
CONDITIONAL_CLAIM_INSUFFICIENT = "INSUFFICIENT"
CONDITIONAL_CLAIM_DIRECTIONAL = "DIRECTIONAL"
CONDITIONAL_CLAIM_INTERVAL = "INTERVAL"
CONDITIONAL_CLAIM_POINT = "POINT"

CONDITIONAL_CLAIM_PRECEDENCE: tuple[str, ...] = (
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_DIRECTIONAL,
    CONDITIONAL_CLAIM_INTERVAL,
    CONDITIONAL_CLAIM_POINT,
)

# Sample floors per tier. MEASURED above - not round numbers.
CONDITIONAL_MIN_SAMPLES_POINT = 40
CONDITIONAL_MIN_SAMPLES_DIRECTIONAL = 20
CONDITIONAL_MIN_SAMPLES_INTERVAL = 8

# An interval wider than this spans so much of [0,1] that it constrains
# nothing. MEASURED: mean Wilson width is 0.474 at N=8 and 0.554 at N=5, so
# this admits N>=8 and excludes N<=5 on width alone at a realistic base rate.
CONDITIONAL_MAX_INTERVAL_WIDTH = 0.50

# A width cap CANNOT replace the sample floor, and this is why it is not
# allowed to. MEASURED: a unanimous 5/5 gives Wilson width 0.434 - NARROWER
# than the well-sampled 9-observation range cell at 0.525. Width rewards
# unanimity, and unanimity is precisely what a tiny sample manufactures. Both
# guards apply; neither is sufficient alone.
CONDITIONAL_WIDTH_REQUIRES_FLOOR = True

# Wilson score interval, two-sided 95%. NOT core.calibration.prediction_interval:
# MEASURED, that function returns width 0.00 at EVERY N on identical
# observations because it measures spread ACROSS CV FOLDS. A conditional rate
# needs uncertainty FROM SAMPLE SIZE. Same-looking output, different quantity -
# composing it here would be a W5 violation in reverse (reusing the wrong
# existing thing rather than duplicating the right one).
CONDITIONAL_INTERVAL_METHOD = "wilson_score"
CONDITIONAL_INTERVAL_Z = 1.959964
CONDITIONAL_INTERVAL_LEVEL = 0.95

# Conditions F4 slices on. The regime is the governed five-state label from N4
# (core.regime_agent), never the ungoverned display heuristic.
CONDITIONAL_CONDITION_REGIME = "market_regime"
CONDITIONAL_CONDITIONS: tuple[str, ...] = (CONDITIONAL_CONDITION_REGIME,)

# Cell statuses. These mirror F3 vocabulary for the obstacles F3 already
# names, and add only what CONDITIONING introduces.
COND_STATUS_UNAVAILABLE = "UNAVAILABLE"        # no label set at all
COND_STATUS_PENDING = "PENDING"                # the horizon window has not closed
COND_STATUS_LABEL_UNBACKED = "LABEL_UNBACKED"  # no realized counterpart at this horizon
COND_STATUS_NO_CONDITION = "NO_CONDITION"      # the condition could not be resolved PIT
COND_STATUS_EMPTY_SLICE = "EMPTY_SLICE"        # the condition never occurred in history
COND_STATUS_INSUFFICIENT = "INSUFFICIENT"      # the slice is real but too thin to claim
COND_STATUS_OK = "OK"

CONDITIONAL_CELL_PRECEDENCE: tuple[str, ...] = (
    COND_STATUS_UNAVAILABLE,
    COND_STATUS_PENDING,
    COND_STATUS_LABEL_UNBACKED,
    COND_STATUS_NO_CONDITION,
    COND_STATUS_EMPTY_SLICE,
    COND_STATUS_INSUFFICIENT,
    COND_STATUS_OK,
)

# EMPTY_SLICE is NOT INSUFFICIENT. "this regime never occurred in the sampled
# history" and "it occurred 3 times" are different facts with different fixes
# (widen the window vs wait for data), and collapsing them hides which one
# applies.
CONDITIONAL_EMPTY_SLICE_N = 0

# Multiple testing is real: the full grid is 6 targets x 6 horizons x 5
# regimes = 180 cells drawn from ONE history. At a 5% false-signal rate that
# is ~9 spurious DIRECTIONAL findings per run. Directional claims therefore
# carry the count of comparisons they were drawn from, so a reader can weigh
# them; F4 reports the exposure rather than silently applying a correction
# that would also suppress true findings in a 5-cell regime row.
CONDITIONAL_REPORT_MULTIPLICITY = True

# The STRESS -> NO_TRADE coupling (W2 veto market_regime_stress) is
# GOVERNANCE and is unaffected by anything F4 publishes. A conditional cell
# reading "stress: P(up) high" must never be read as permission to trade: the
# veto is evaluated in core/risk_policy.py and F4 is a reporting surface.
CONDITIONAL_STRESS_VETO_UNAFFECTED = True


def _validate_conditional_forecast_config() -> None:
    """Import-time guard for the F4 tier machinery."""
    declared = {
        COND_STATUS_UNAVAILABLE, COND_STATUS_PENDING, COND_STATUS_LABEL_UNBACKED,
        COND_STATUS_NO_CONDITION, COND_STATUS_EMPTY_SLICE,
        COND_STATUS_INSUFFICIENT, COND_STATUS_OK,
    }
    if set(CONDITIONAL_CELL_PRECEDENCE) != declared:
        raise ValueError(
            "CONDITIONAL_CELL_PRECEDENCE must cover exactly the declared statuses"
        )
    if len(CONDITIONAL_CELL_PRECEDENCE) != len(declared):
        raise ValueError("CONDITIONAL_CELL_PRECEDENCE contains a duplicate")
    if CONDITIONAL_CELL_PRECEDENCE[-1] != COND_STATUS_OK:
        raise ValueError("OK must be LAST - every refusal outranks it")

    tiers = {
        CONDITIONAL_CLAIM_INSUFFICIENT, CONDITIONAL_CLAIM_DIRECTIONAL,
        CONDITIONAL_CLAIM_INTERVAL, CONDITIONAL_CLAIM_POINT,
    }
    if set(CONDITIONAL_CLAIM_PRECEDENCE) != tiers:
        raise ValueError("CONDITIONAL_CLAIM_PRECEDENCE must cover exactly the declared tiers")
    if len(CONDITIONAL_CLAIM_PRECEDENCE) != len(tiers):
        raise ValueError("CONDITIONAL_CLAIM_PRECEDENCE contains a duplicate")
    if CONDITIONAL_CLAIM_PRECEDENCE[0] != CONDITIONAL_CLAIM_INSUFFICIENT:
        raise ValueError(
            "INSUFFICIENT must be FIRST - it is the fallback every cell starts from"
        )
    if CONDITIONAL_CLAIM_PRECEDENCE[-1] != CONDITIONAL_CLAIM_POINT:
        raise ValueError("POINT must be LAST - it is the strongest claim")

    # The floors must be strictly ordered POINT > DIRECTIONAL > INTERVAL. The
    # middle term is the counter-intuitive one and is asserted deliberately:
    # a DIRECTIONAL claim needs MORE evidence than an INTERVAL despite saying
    # less, because a too-wide interval advertises its own weakness while a
    # directional claim that failed to detect looks like one that found
    # nothing. Re-sorting these by apparent wording strength would invert it.
    if not (
        CONDITIONAL_MIN_SAMPLES_POINT
        > CONDITIONAL_MIN_SAMPLES_DIRECTIONAL
        > CONDITIONAL_MIN_SAMPLES_INTERVAL
        > CONDITIONAL_EMPTY_SLICE_N
    ):
        raise ValueError(
            "sample floors must run POINT > DIRECTIONAL > INTERVAL > 0; "
            "DIRECTIONAL above INTERVAL is intentional (see the comment above)"
        )
    if not 0.0 < CONDITIONAL_MAX_INTERVAL_WIDTH < 1.0:
        raise ValueError("the max interval width must lie strictly inside (0, 1)")
    if not CONDITIONAL_WIDTH_REQUIRES_FLOOR:
        raise ValueError(
            "a width cap must NOT be allowed to stand alone: a unanimous 5/5 "
            "gives Wilson width 0.434, narrower than a 9-observation cell at "
            "0.525, so width alone admits the thinnest samples"
        )
    if CONDITIONAL_INTERVAL_METHOD != "wilson_score":
        raise ValueError(
            "conditional rates use the Wilson score interval; "
            "core.calibration.prediction_interval measures fold spread and "
            "returns width 0.00 at every N here"
        )
    if CONDITIONAL_INTERVAL_Z <= 0.0:
        raise ValueError("the interval z must be strictly positive")
    if CONDITIONAL_CONDITION_REGIME not in CONDITIONAL_CONDITIONS:
        raise ValueError("the regime condition must be registered in CONDITIONAL_CONDITIONS")


_validate_conditional_forecast_config()


# ---------------------------------------------------------------------------
# Sprint F5 - Event-conditioned forecast
# ---------------------------------------------------------------------------
# The pipeline the roadmap names:
#
#   new event -> event representation -> historical event matches
#             -> current chart -> market regime -> forecast
#
# F5 is a PIPELINE WITH PER-STAGE STATUS, not a function that returns a number.
# Every stage can fail independently, and when one does the forecast must name
# WHICH stage failed. "no forecast" is not a useful answer; "retrieval found 1
# analog and needs 5" is.
#
# WHY THE STAGES ARE THE CONTRACT. MEASURED on a fresh clone: the event memory
# store (data/event_memory.jsonl) does not exist and holds zero memories, so
# stages 3-6 are unreachable. That is not a defect to paper over inside F5 --
# it is the honest terminal state of a system that has not yet observed
# anything. F5 says so, at the stage where it became true.
#
# COMPOSITION, NOT A THIRD RETRIEVAL (W5). Two analog-retrieval systems already
# exist:
#   E6 core.event_memory.find_analogs  - chart numbers + EVENT TYPE, floor 5
#   C7 core.reaction_memory.find_reaction_analogs - C4 structural PHASE, floor 3
# F5 builds neither. It calls E6 (the only one that filters by event type,
# which is what "event-conditioned" means) and declares that in
# CONDITIONAL_EVENT_RETRIEVAL. A third similarity metric would be the
# split-brain the master context forbids.
#
# THE CLAIM STRENGTH IS F4'S, NOT A SECOND ANSWER. An analog set is exactly a
# conditional slice: N observations of what followed a comparable setup. F4
# already decides what a slice of size N can support, measured. F5 calls
# core.forecast_conditional.select_claim rather than inventing a second
# sample-size policy that would inevitably drift from the first.
#
# REGIME IS ALREADY INSIDE RETRIEVAL. `market_regime` is one of
# EVENT_MEMORY_SIMILARITY_FIELDS, so the roadmap's "market regime" stage is
# partly upstream of the match. F5 therefore reports regime AGREEMENT of the
# retrieved set rather than re-filtering on it, because filtering again would
# weight the same evidence twice and shrink an already thin set for nothing.
EVENT_FORECAST_VERSION = "event-forecast-v1"
EVENT_FORECAST_CONTRACT_VERSION = "event-forecast-contract-v1"

# The declared stage order. This IS the pipeline: stages run in this sequence
# and the first failure stops it, so a reader always learns the most upstream
# obstacle rather than a downstream symptom of it.
EVENT_FORECAST_STAGE_EVENT = "event"                  # a valid, PIT-eligible event
EVENT_FORECAST_STAGE_REPRESENTATION = "representation"  # event -> comparable form
EVENT_FORECAST_STAGE_MATCHES = "matches"              # historical analogs
EVENT_FORECAST_STAGE_CHART = "chart"                  # the current chart state
EVENT_FORECAST_STAGE_REGIME = "regime"                # market regime at as_of
EVENT_FORECAST_STAGE_FORECAST = "forecast"            # the conditioned claim

EVENT_FORECAST_STAGES: tuple[str, ...] = (
    EVENT_FORECAST_STAGE_EVENT,
    EVENT_FORECAST_STAGE_REPRESENTATION,
    EVENT_FORECAST_STAGE_MATCHES,
    EVENT_FORECAST_STAGE_CHART,
    EVENT_FORECAST_STAGE_REGIME,
    EVENT_FORECAST_STAGE_FORECAST,
)

# Per-stage outcomes.
EVENT_STAGE_OK = "OK"
EVENT_STAGE_BLOCKED = "BLOCKED"        # an upstream stage failed; not attempted
EVENT_STAGE_FAILED = "FAILED"          # attempted, could not complete
EVENT_STAGE_DEGRADED = "DEGRADED"      # completed, but with a caveat that travels

EVENT_STAGE_STATUSES: tuple[str, ...] = (
    EVENT_STAGE_OK,
    EVENT_STAGE_DEGRADED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_BLOCKED,
)

# Overall pipeline verdicts.
EVENT_FORECAST_OK = "OK"
EVENT_FORECAST_REFUSED = "REFUSED"

# Which retrieval F5 composes. Declared as DATA so the choice is auditable and
# so a future reader cannot add a second one believing none was chosen.
EVENT_FORECAST_RETRIEVAL = "event_memory.find_analogs"

# Analogs drawn from ONE ticker are not independent observations. MEASURED
# before the E6 similarity fix: 84.5% of every pair clearing the 0.70 bar was
# the same ticker -- adjacent sessions of one stock, which is one situation
# counted many times. Above this share the forecast is DEGRADED and says so,
# because a base rate from a single name is a fact about that name.
EVENT_FORECAST_MAX_SAME_TICKER_SHARE = 0.60

# How many INDEPENDENT observations one ticker may contribute. Analogs from a
# single name are adjacent sessions of one situation, resampled; counting them
# as independent is the pseudo-replication that made the F4 stress cell read
# P(up)=1.00 from two observations. MEASURED before the E6 similarity fix,
# 84.5% of every pair clearing the retrieval bar was the same ticker.
#
# The value is deliberately crude -- distinct tickers, not a correlation model
# -- because a precise-looking adjustment would imply a precision retrieval
# cannot support. It is a floor on honesty, not an estimate. At 5, an
# all-one-ticker set can never reach the POINT tier (floor 40) however many
# rows it contains.
EVENT_FORECAST_EFFECTIVE_PER_TICKER = 5

# The fewest distinct tickers a POINT claim must rest on. Derived, not chosen:
# with the per-ticker cap above, reaching the point floor requires at least
# this many names. Declared so the relationship is asserted rather than
# implied, because a later edit to either number alone would break it silently.
EVENT_FORECAST_MIN_DISTINCT_FOR_POINT = 8

# An analog set built mostly from INFERRED events is weaker evidence than one
# built from observed ones: the price move is real, but which event produced
# it -- or whether one did at all -- was never sourced. MEASURED: an inferred
# date cannot be validated, because fetch_fundamental_snapshot returns
# earnings_date=None and no historical earnings calendar exists anywhere in
# the system. Below this share of OBSERVED analogs the forecast is DEGRADED
# and says so, the same treatment a confounded set receives.
EVENT_FORECAST_MIN_OBSERVED_SHARE = 0.50

# Reported alongside every forecast. Retrieval yield is the binding constraint
# on this whole sprint, so it is never silently absorbed.
# MEASURED after the E6 fix, 1,608 real chart states, bar 0.70, type matched:
#   share reaching 5 analogs 0.247 | reaching 3 0.373 | ZERO analogs 0.393
EVENT_FORECAST_REPORT_YIELD = True

# Regime agreement of the retrieved set, reported not enforced. See the header:
# market_regime is already a similarity field, so re-filtering would weight it
# twice and shrink a thin set for no new evidence.
EVENT_FORECAST_REPORT_REGIME_AGREEMENT = True

# The horizons a conditioned forecast is offered at. Constrained to the
# intersection of what a memory RECORDS (E6) and what the forecast layer
# declares (F2), because a horizon outside either cannot be both retrieved and
# expressed.
EVENT_FORECAST_HORIZONS: tuple[str, ...] = tuple(
    horizon for horizon in EVENT_MEMORY_RESPONSE_HORIZONS
    if horizon in FORECAST_HORIZONS
)

# An attribution-aware floor. E5 already distinguishes an event-ASSOCIATED
# response from a confounded one; a base rate built mostly from confounded
# analogs describes the market, not the event. Reported as DEGRADED below this.
EVENT_FORECAST_MIN_ASSOCIATED_SHARE = 0.50


def _validate_event_forecast_config() -> None:
    """Import-time guard for the F5 pipeline contract."""
    if len(set(EVENT_FORECAST_STAGES)) != len(EVENT_FORECAST_STAGES):
        raise ValueError("EVENT_FORECAST_STAGES contains a duplicate")
    if EVENT_FORECAST_STAGES[0] != EVENT_FORECAST_STAGE_EVENT:
        raise ValueError(
            "the pipeline must begin at the event -- every later stage is "
            "conditioned on one existing"
        )
    if EVENT_FORECAST_STAGES[-1] != EVENT_FORECAST_STAGE_FORECAST:
        raise ValueError(
            "the forecast must be the LAST stage; anything after it would be "
            "reasoning that the published claim did not account for"
        )
    if EVENT_FORECAST_STAGE_MATCHES not in EVENT_FORECAST_STAGES:
        raise ValueError("retrieval is not optional in an event-CONDITIONED forecast")
    # Retrieval must precede the forecast, or the claim would not rest on the
    # analogs at all.
    if EVENT_FORECAST_STAGES.index(
        EVENT_FORECAST_STAGE_MATCHES
    ) >= EVENT_FORECAST_STAGES.index(EVENT_FORECAST_STAGE_FORECAST):
        raise ValueError("matches must be retrieved BEFORE the forecast is made")

    if len(set(EVENT_STAGE_STATUSES)) != len(EVENT_STAGE_STATUSES):
        raise ValueError("EVENT_STAGE_STATUSES contains a duplicate")
    if EVENT_STAGE_OK not in EVENT_STAGE_STATUSES:
        raise ValueError("a stage must be able to succeed")

    if EVENT_FORECAST_RETRIEVAL != "event_memory.find_analogs":
        raise ValueError(
            "F5 composes E6 retrieval; it must not introduce a third analog "
            "metric beside E6 and C7 (W5: one canonical implementation)"
        )
    if not 0.0 < EVENT_FORECAST_MAX_SAME_TICKER_SHARE <= 1.0:
        raise ValueError("the same-ticker share must lie in (0, 1]")
    if EVENT_FORECAST_EFFECTIVE_PER_TICKER < 1:
        raise ValueError("a ticker must contribute at least one observation")
    # One ticker must not approach the POINT tier, not merely fall short of
    # it. A value just under the floor (39 against a floor of 40) passes a
    # naive >= check while letting a single stock's own history carry almost
    # the entire claim. The cap is a MULTIPLE below the floor, so a point
    # estimate always rests on several distinct names.
    if (
        EVENT_FORECAST_EFFECTIVE_PER_TICKER * EVENT_FORECAST_MIN_DISTINCT_FOR_POINT
        > CONDITIONAL_MIN_SAMPLES_POINT
    ):
        raise ValueError(
            f"one ticker may contribute {EVENT_FORECAST_EFFECTIVE_PER_TICKER} "
            f"observations, so fewer than "
            f"{EVENT_FORECAST_MIN_DISTINCT_FOR_POINT} distinct tickers could "
            f"reach the POINT floor of {CONDITIONAL_MIN_SAMPLES_POINT} — a "
            f"point estimate must rest on several distinct names, not one "
            f"stock's own history"
        )
    if not 0.0 <= EVENT_FORECAST_MIN_ASSOCIATED_SHARE <= 1.0:
        raise ValueError("the associated share must lie in [0, 1]")
    if not 0.0 <= EVENT_FORECAST_MIN_OBSERVED_SHARE <= 1.0:
        raise ValueError("the observed share must lie in [0, 1]")
    if not EVENT_FORECAST_HORIZONS:
        raise ValueError(
            "no horizon is both recorded by a memory and declared by the "
            "forecast layer -- an event-conditioned forecast could never be "
            "expressed at any horizon"
        )
    for horizon in EVENT_FORECAST_HORIZONS:
        if horizon not in EVENT_MEMORY_RESPONSE_HORIZONS:
            raise ValueError(f"{horizon!r} is not recorded by a memory")
        if horizon not in FORECAST_HORIZONS:
            raise ValueError(f"{horizon!r} is not a declared forecast horizon")
    if not EVENT_FORECAST_REPORT_YIELD:
        raise ValueError(
            "retrieval yield is the binding constraint on this sprint and must "
            "travel with every forecast; MEASURED, 39.3% of events retrieve "
            "ZERO analogs even after the E6 similarity fix"
        )


_validate_event_forecast_config()


# ---------------------------------------------------------------------------
# Daily collection - the capture step continuous learning depends on
# ---------------------------------------------------------------------------
# The system learns from what it recorded, and nothing recorded on a schedule.
# MEASURED over three weeks of the W6 ledger: 2-3 of 15 business days are
# MISSING, because collection happened only when somebody ran a command.
#
# Most inputs lose nothing by waiting - price bars, macro series, regime
# labels and inferred event memories are all rebuildable on demand. MEASURED,
# hourly bars reach 2 YEARS back, not the ~1 month the C7 docstring assumed.
#
# NEWS IS DIFFERENT. NEWS_LOOKBACK_DAYS is 7: a day of articles not captured
# within a week is gone permanently, and with it that day's sentiment and any
# OBSERVED event memory it would have produced. F5 currently retrieves only
# INFERRED analogs (precision 0.65) precisely because no observed ones exist.
#
# So the argument for scheduling is IRREVERSIBILITY, not convenience.
COLLECT_VERSION = "daily-collect-v1"

# The sources one run exercises, in order. Each already appends to the W6
# ledger inside its own fetch, so this list names TRIGGERS, not writers --
# a second ledger path would be the split-brain W5 forbids.
COLLECT_SOURCES: tuple[str, ...] = (
    "prices", "fundamentals", "macro", "news", "events",
)

# The sources whose data cannot be recovered later. A run that loses one of
# these FAILS, while a run that loses a rebuildable source does not: the exit
# code has to mean "something irreplaceable was lost", or a scheduler cannot
# act on it.
COLLECT_PERISHABLE_SOURCES: tuple[str, ...] = ("news", "events")

# Seconds between provider calls. Providers rate-limit, and a run that gets
# throttled halfway captures half a day.
COLLECT_THROTTLE_SECONDS = 0.2

# A ceiling per run so one invocation cannot hang for hours on a large
# universe. 200 comfortably covers the current 77-ticker portfolio.
COLLECT_MAX_TICKERS_PER_RUN = 200

# One JSON line per run. This is what makes a SILENT outage visible: a
# scheduled task that stops firing leaves no error anywhere, but it also
# leaves no line here, and the coverage gate reads exactly that.
COLLECT_REPORT_PATH = "data/collection_report.jsonl"

# -- Quota-aware news collection ------------------------------------------
#
# MEASURED: the NewsAPI free tier allows 100 requests/day and the collector
# makes ONE call per ticker, so 77 tickers plus any ad-hoc testing exceeds it.
# Today's runs returned HTTP 429 (quota exhausted) for every ticker.
#
# THE COST IS PER CALL, NOT PER BYTE. An ETF costs exactly the same one call as
# any stock, so excluding funds saves 3-4 calls out of 77 -- it is not a
# bandwidth measure. Funds are excluded for a different and better reason
# below.
#
# FUNDS ARE EXCLUDED BECAUSE THEIR NEWS IS NOT A COMPANY EVENT. MEASURED, C3
# already refuses VOO/SOXX/CIBR/NASA a sector ("a fund has no single sector,
# and a sector ETF is not a benchmark for itself"), and E6 retrieval matches
# on event_type against a company's chart state. A fund-level headline is
# market commentary, which is precisely what E5 attribution calls CONFOUNDED.
# Spending a scarce call on one buys an observed memory that describes the
# market rather than the holding.
#
# They keep their INFERRED memories from price history -- those are real price
# moves and cost no quota.
#
# The exclusion is DERIVED, not a hardcoded list: any ticker C3 cannot give a
# sector is skipped, so adding a fund to the portfolio needs no edit here.
# The provider's documented daily request ceiling on the free tier. Declared
# so the gate can refuse a batch that exceeds it, rather than discovering the
# limit again through a day of HTTP 429s.
NEWS_PROVIDER_DAILY_LIMIT = 100

COLLECT_NEWS_SKIP_SECTORLESS = True

# Sector-less tickers that ARE worth their call anyway, by explicit decision.
#
# SOXX (semiconductors) and CIBR (cyber security) are NARROW THEMATIC funds:
# their news is about one industry, so a headline is closer to a sector event
# than to broad market commentary. The portfolio also holds many of their
# constituents, so what moves them tends to move real holdings.
#
# This is an operator's judgement, not a measurement, and it is recorded as
# such. VOO stays skipped because it tracks the whole S&P 500 — its "news" is
# the market itself, which is what E5 attribution already calls confounded.
# NASA stays skipped because the repo carries no metadata saying what it is:
# skipping an unknown costs one call a day, while tracking one on a guess
# feeds the store memories nobody can interpret.
#
# Their memories remain honestly labelled: E5 will mark a fund reaction
# CONFOUNDED if that is what it measures, and F5 reports the observed share,
# so tracking them adds evidence without weakening any claim built on it.
COLLECT_NEWS_TRACK_ANYWAY: tuple[str, ...] = ("SOXX", "CIBR")

# Tickers per news run. MEASURED against the 100/day ceiling: 40 covers all 73
# sector-mapped holdings in two runs and leaves 60 calls spare for retries and
# ad-hoc work -- the margin whose absence produced today's 429.
COLLECT_NEWS_BATCH_SIZE = 40

# Where the rotation cursor lives, so consecutive runs advance rather than
# re-fetching the same head of the list. MEASURED: a sequential cursor covers
# 73/73 in 2 runs with a visit spread of 1 over 20 runs, where a date-derived
# stride left a spread of 2.
COLLECT_NEWS_CURSOR_PATH = "data/collect_cursor.json"

# How many business days back the coverage gate checks. Long enough to catch
# a scheduler that died last week, short enough that the historical gaps
# already in the ledger do not fail every future run.
COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS = 10

# Business days that may be missing inside that window before the gate fails.
# NOT zero: a provider outage or a market holiday this table does not know
# about should not break the build. Two is one bad day plus one surprise.
COLLECT_COVERAGE_MAX_MISSING = 2


def _validate_collect_config() -> None:
    """Import-time guard for the collection contract."""
    if len(set(COLLECT_SOURCES)) != len(COLLECT_SOURCES):
        raise ValueError("COLLECT_SOURCES contains a duplicate")
    if not COLLECT_SOURCES:
        raise ValueError("a collector with no sources collects nothing")
    unknown = set(COLLECT_PERISHABLE_SOURCES) - set(COLLECT_SOURCES)
    if unknown:
        raise ValueError(
            f"perishable sources must be collected sources: {sorted(unknown)}"
        )
    if "news" not in COLLECT_PERISHABLE_SOURCES:
        raise ValueError(
            "news MUST stay perishable: NEWS_LOOKBACK_DAYS is "
            f"{NEWS_LOOKBACK_DAYS}, so an uncaptured day is gone permanently "
            "and no OBSERVED event memory for it can ever exist"
        )
    if COLLECT_THROTTLE_SECONDS < 0:
        raise ValueError("the throttle must not be negative")
    if COLLECT_MAX_TICKERS_PER_RUN < 1:
        raise ValueError("a run must cover at least one ticker")
    if COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS < 1:
        raise ValueError("the coverage window must span at least one day")
    if COLLECT_COVERAGE_MAX_MISSING < 0:
        raise ValueError("the missing-day allowance must not be negative")
    if COLLECT_NEWS_BATCH_SIZE < 1:
        raise ValueError("a news run must cover at least one ticker")
    if COLLECT_NEWS_BATCH_SIZE > 100:
        raise ValueError(
            f"a batch of {COLLECT_NEWS_BATCH_SIZE} exceeds the provider's "
            f"100-request daily ceiling on its own, before any retry or "
            f"ad-hoc call"
        )
    if not COLLECT_NEWS_CURSOR_PATH.startswith("data/"):
        raise ValueError("the rotation cursor belongs under data/")
    if len(set(COLLECT_NEWS_TRACK_ANYWAY)) != len(COLLECT_NEWS_TRACK_ANYWAY):
        raise ValueError("COLLECT_NEWS_TRACK_ANYWAY contains a duplicate")
    if not COLLECT_NEWS_SKIP_SECTORLESS and COLLECT_NEWS_TRACK_ANYWAY:
        raise ValueError(
            "COLLECT_NEWS_TRACK_ANYWAY names exceptions to a skip that is "
            "disabled — one of the two settings is not doing what its name says"
        )
    if COLLECT_COVERAGE_MAX_MISSING >= COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS:
        raise ValueError(
            "the allowance must be smaller than the window, or a scheduler "
            "that never runs at all would still pass the coverage gate"
        )


_validate_collect_config()


# ---------------------------------------------------------------------------
# Sprint F6 - Forecast decomposition
# ---------------------------------------------------------------------------
# "Break the forecast into Technical / Fundamental / News-Event / Macro /
# Regime / Sentiment / Historical-analog contributions - interpretable,
# without implying false causal certainty."
#
# THE WORD "CONTRIBUTION" IS THE TRAP, and it was measured rather than argued.
# F5's value is a base rate over analogs retrieved by chart similarity,
# filtered by event type, within a regime. Those filters are NOT independent:
# one historical day can satisfy all three. Simulated over 4,000 observations
# with a realistic regime/chart correlation:
#
#     all sessions           P(up) 0.546
#     bullish regime         P(up) 0.611   "contribution" +0.064
#     uptrend chart          P(up) 0.614   "contribution" +0.067
#     bullish AND uptrend    P(up) 0.606   SUM +0.131, ACTUAL +0.060
#
# The parts overlap and double-count by more than 2x. An additive
# decomposition would be arithmetically WRONG, and calling the parts
# "contributions" would imply each factor independently caused its share.
#
# So F6 reports WHAT EVIDENCE ENTERED and HOW MUCH IT NARROWED THE CLAIM,
# never "this component contributed +0.064".
#
# NOT A SECOND ENSEMBLE BREAKDOWN (W5). W1 already decomposes the SCORE across
# these same seven agent names, and score_engine renders per-term values for
# the dashboard. F6 decomposes the FORECAST, which is a different object
# produced by different machinery, and it says so in its own output.
#
# THERE ARE NO COEFFICIENTS TO ATTRIBUTE. MEASURED: build_joint_forecast
# returns NO_MODEL - "no trained forecasting model is registered". So F6
# cannot be feature attribution, SHAP, or a weight table. What produces a
# forecast value today is F4 (a regime-conditioned base rate) and F5 (an
# analog base rate), both EMPIRICAL SLICES of history.
FORECAST_DECOMPOSITION_VERSION = "forecast-decomposition-v1"
DECOMPOSITION_CONTRACT_VERSION = "decomposition-contract-v1"

# The seven components, in reading order. Named to match the sprint brief so
# a reader can map the output onto the request.
DECOMP_TECHNICAL = "technical"
DECOMP_FUNDAMENTAL = "fundamental"
DECOMP_NEWS_EVENT = "news_event"
DECOMP_MACRO = "macro"
DECOMP_REGIME = "regime"
DECOMP_SENTIMENT = "sentiment"
DECOMP_HISTORICAL_ANALOG = "historical_analog"

DECOMPOSITION_COMPONENTS: tuple[str, ...] = (
    DECOMP_TECHNICAL,
    DECOMP_FUNDAMENTAL,
    DECOMP_NEWS_EVENT,
    DECOMP_MACRO,
    DECOMP_REGIME,
    DECOMP_SENTIMENT,
    DECOMP_HISTORICAL_ANALOG,
)

# Which components can supply FORECAST evidence today. MEASURED:
#   technical         YES  the chart state IS F5's retrieval key
#   news_event        YES  F5 over 2,084 memories (130 observed)
#   regime            YES  F4 conditions on it; F5 reports agreement
#   historical_analog YES  F4 slices and F5 analogs produce the value
#   fundamental       NO   captured daily, but no forecast consumes it
#   macro             NO   snapshot UNAVAILABLE (no FRED key), unconsumed
#   sentiment         NO   ensemble weight 0.0, typed placeholder, no provider
#
# Declared as DATA so the distinction is auditable, and so wiring one later is
# a deliberate edit here rather than a silent behaviour change.
DECOMPOSITION_WIRED_COMPONENTS: tuple[str, ...] = (
    DECOMP_TECHNICAL,
    DECOMP_NEWS_EVENT,
    DECOMP_REGIME,
    DECOMP_HISTORICAL_ANALOG,
)

# Component states. NOT_WIRED is deliberately distinct from a measured zero:
# reporting an unwired component as 0.0 would say it was measured and found
# irrelevant, when it was never measured at all.
DECOMP_STATUS_PRESENT = "PRESENT"        # supplied evidence to this forecast
DECOMP_STATUS_ABSENT = "ABSENT"          # wired, but had nothing to say here
DECOMP_STATUS_NOT_WIRED = "NOT_WIRED"    # no forecast path consumes it yet

DECOMPOSITION_STATUSES: tuple[str, ...] = (
    DECOMP_STATUS_PRESENT,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
)

# What a component is allowed to report about its effect. There is no
# "contribution" member, and that absence is the point.
DECOMP_EFFECT_NARROWED = "NARROWED"          # it cut the analog set / slice
DECOMP_EFFECT_NO_EFFECT = "NO_EFFECT"        # it matched everything available
DECOMP_EFFECT_UNMEASURED = "UNMEASURED"      # present, but its effect is not isolable

DECOMPOSITION_EFFECTS: tuple[str, ...] = (
    DECOMP_EFFECT_NARROWED,
    DECOMP_EFFECT_NO_EFFECT,
    DECOMP_EFFECT_UNMEASURED,
)

# Marginal slice rates are reported per component, each with its OWN sample
# size and interval, and explicitly flagged non-additive. Setting this True
# would be the single edit that turns F6 back into a false causal story.
DECOMPOSITION_ADDITIVE = False

# The overlap measurement above, kept as data so the reason additive
# decomposition is refused travels with the code.
DECOMPOSITION_OVERLAP_EVIDENCE = (
    "MEASURED over 4,000 observations with a realistic regime/chart "
    "correlation: regime alone +0.064, chart alone +0.067, sum +0.131, "
    "ACTUAL joint effect +0.060. The parts overlap and double-count by more "
    "than 2x, so contributions cannot be added and must not be presented as "
    "though they could."
)

# Every decomposition carries this. E6 and F5 already carry the same sentence
# shape; a decomposition is the surface where a reader is MOST likely to read
# causation into association, so it is stated at the top level, not a footnote.
DECOMPOSITION_DISCLAIMER = (
    "historical association under comparable conditions - these components "
    "describe WHAT EVIDENCE ENTERED the forecast, not what caused the "
    "outcome. The effects are marginal slice rates, they overlap, and they "
    "do not sum to the forecast."
)


def _validate_decomposition_config() -> None:
    """Import-time guard for the F6 contract."""
    if len(set(DECOMPOSITION_COMPONENTS)) != len(DECOMPOSITION_COMPONENTS):
        raise ValueError("DECOMPOSITION_COMPONENTS contains a duplicate")
    if len(DECOMPOSITION_COMPONENTS) != 7:
        raise ValueError(
            "the sprint names seven components; dropping one would silently "
            "narrow what the decomposition claims to cover"
        )
    unknown = set(DECOMPOSITION_WIRED_COMPONENTS) - set(DECOMPOSITION_COMPONENTS)
    if unknown:
        raise ValueError(f"wired components must be declared ones: {sorted(unknown)}")
    if not DECOMPOSITION_WIRED_COMPONENTS:
        raise ValueError(
            "no component is wired - a decomposition of nothing is not a "
            "decomposition"
        )
    if len(set(DECOMPOSITION_STATUSES)) != len(DECOMPOSITION_STATUSES):
        raise ValueError("DECOMPOSITION_STATUSES contains a duplicate")
    if DECOMP_STATUS_NOT_WIRED not in DECOMPOSITION_STATUSES:
        raise ValueError(
            "NOT_WIRED must stay distinct from a measured zero: reporting an "
            "unwired component as 0.0 claims it was measured and found "
            "irrelevant, which is a different fact"
        )
    if len(set(DECOMPOSITION_EFFECTS)) != len(DECOMPOSITION_EFFECTS):
        raise ValueError("DECOMPOSITION_EFFECTS contains a duplicate")
    for forbidden in ("CONTRIBUTION", "CONTRIBUTED", "CAUSED"):
        if forbidden in DECOMPOSITION_EFFECTS:
            raise ValueError(
                f"{forbidden!r} is not an effect this system can report: the "
                f"components overlap and do not sum to the forecast"
            )
    if DECOMPOSITION_ADDITIVE:
        raise ValueError(
            "the decomposition must NOT be additive. " + DECOMPOSITION_OVERLAP_EVIDENCE
        )
    if "MEASURED" not in DECOMPOSITION_OVERLAP_EVIDENCE:
        raise ValueError(
            "the overlap evidence must carry its measurement, or a future "
            "reader cannot tell a finding from an assertion"
        )
    for phrase in ("not what caused", "do not sum"):
        if phrase not in DECOMPOSITION_DISCLAIMER:
            raise ValueError(
                f"the disclaimer must keep saying {phrase!r}: a decomposition "
                f"is where a reader is most likely to read causation into "
                f"association"
            )


_validate_decomposition_config()


# ---------------------------------------------------------------------------
# Sprint F7 - Forecast confidence
# ---------------------------------------------------------------------------
# "Accounts for sample size, calibration, model agreement, feature
# completeness, source quality, regime similarity, event similarity, model
# drift, uncertainty. Confidence is not the same thing as P(up)."
#
# CONFIDENCE IS NOT P(up), and it is not a rhetorical caveat - it is MEASURED:
#
#     case                        P(up)      N      interval   width
#     coin flip, huge sample      0.500   1000   [0.47,0.53]   0.062
#     near-certain, tiny sample   1.000      2   [0.34,1.00]   0.658
#
# P(up)=0.500 from 1,000 observations is a HIGH-confidence statement.
# P(up)=1.000 from two is a NEAR-ZERO-confidence one. They are orthogonal, and
# a dashboard that renders P(up) as a confidence bar inverts the meaning
# exactly when it matters most.
#
# THIS IS NOT core.score_engine._compute_confidence (W5). That produces a
# weighted sum over six factors for the SCORE. F7 is confidence in the
# FORECAST: different object, different inputs, and - measured below - a
# different aggregation. Both exist; neither may be read as the other, so
# every payload names the object it describes.
#
# THE AGGREGATION IS A LIMITING FACTOR, NOT AN AVERAGE. MEASURED, applying the
# score's weighted-sum approach to forecast factors:
#
#     case                          weighted sum   MIN factor
#     everything strong                    0.950        0.950
#     N=2 fatal, rest strong               0.717        0.020
#     all-inferred sources                 0.807        0.000
#     uniformly mediocre                   0.550        0.550
#
# A forecast built on TWO observations scores 0.717 - HIGHER than a uniformly
# mediocre one at 0.550. That is exactly backwards: N=2 is the F4 stress cell,
# the least trustworthy state in the system, and averaging hides it behind
# five strong factors.
#
# Pure MIN is too blunt in the other direction: "good N, weak everything else"
# scores 0.300, identical to a single weak factor, ignoring the other five.
# So the weakest factor CAPS the confidence, and the weighted sum may only
# lower it further. That also matches how the system already reasons - F3, F4
# and F5 each refuse on the single most fundamental obstacle rather than
# averaging obstacles together.
FORECAST_CONFIDENCE_VERSION = "forecast-confidence-v1"
FORECAST_CONFIDENCE_CONTRACT_VERSION = "forecast-confidence-contract-v1"

# The nine factors the sprint names, in reading order.
FCONF_SAMPLE_SIZE = "sample_size"
FCONF_CALIBRATION = "calibration"
FCONF_MODEL_AGREEMENT = "model_agreement"
FCONF_FEATURE_COMPLETENESS = "feature_completeness"
FCONF_SOURCE_QUALITY = "source_quality"
FCONF_REGIME_SIMILARITY = "regime_similarity"
FCONF_EVENT_SIMILARITY = "event_similarity"
FCONF_MODEL_DRIFT = "model_drift"
FCONF_UNCERTAINTY = "uncertainty"

FORECAST_CONFIDENCE_FACTORS: tuple[str, ...] = (
    FCONF_SAMPLE_SIZE,
    FCONF_CALIBRATION,
    FCONF_MODEL_AGREEMENT,
    FCONF_FEATURE_COMPLETENESS,
    FCONF_SOURCE_QUALITY,
    FCONF_REGIME_SIMILARITY,
    FCONF_EVENT_SIMILARITY,
    FCONF_MODEL_DRIFT,
    FCONF_UNCERTAINTY,
)

# Which factors can be MEASURED today. MEASURED:
#   sample_size          YES  F4/F5 samples, scored through select_claim
#   feature_completeness YES  chart_state fields present vs expected
#   source_quality       YES  E6 provenance; inferred precision is 0.65
#   regime_similarity    YES  F5 already computes the agreement share
#   event_similarity     YES  E6 chart_similarity is retrievable per analog
#   uncertainty          YES  Wilson interval width
#   calibration          NO   no fitted M6 map exists
#   model_agreement      NO   NO_MODEL: one baseline, nothing to agree with
#   model_drift          NO   nothing is deployed, so nothing can drift
#
# The three absent ones all depend on a TRAINED MODEL that does not exist.
# Scoring them 1.0 ("nothing wrong") or 0.0 ("everything wrong") would both be
# false; UNMEASURABLE is a third state and is reported as such.
FORECAST_CONFIDENCE_MEASURABLE: tuple[str, ...] = (
    FCONF_SAMPLE_SIZE,
    FCONF_FEATURE_COMPLETENESS,
    FCONF_SOURCE_QUALITY,
    FCONF_REGIME_SIMILARITY,
    FCONF_EVENT_SIMILARITY,
    FCONF_UNCERTAINTY,
)

# Weights over the MEASURABLE factors only. These do NOT produce the
# confidence on their own - they only lower the cap set by the weakest factor.
# sample_size carries the most because it is the factor that was measured to
# be fatal when ignored.
FORECAST_CONFIDENCE_WEIGHTS: dict[str, float] = {
    FCONF_SAMPLE_SIZE: 0.30,
    FCONF_UNCERTAINTY: 0.20,
    FCONF_SOURCE_QUALITY: 0.15,
    FCONF_EVENT_SIMILARITY: 0.15,
    FCONF_REGIME_SIMILARITY: 0.10,
    FCONF_FEATURE_COMPLETENESS: 0.10,
}

# Factor states. UNMEASURABLE is deliberately distinct from a measured zero,
# for the same reason F6's NOT_WIRED is: a number claims a measurement that
# never happened.
FCONF_STATUS_MEASURED = "MEASURED"
FCONF_STATUS_UNMEASURABLE = "UNMEASURABLE"
FCONF_STATUS_UNAVAILABLE = "UNAVAILABLE"   # measurable in principle, absent here

FORECAST_CONFIDENCE_STATUSES: tuple[str, ...] = (
    FCONF_STATUS_MEASURED,
    FCONF_STATUS_UNMEASURABLE,
    FCONF_STATUS_UNAVAILABLE,
)

# The aggregation. Changing this to an average is the single edit that
# reintroduces the measured defect above, so the validator refuses it.
FORECAST_CONFIDENCE_AGGREGATION = "limiting_factor"

FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE = (
    "MEASURED: a weighted sum rates an N=2 forecast at 0.717, HIGHER than a "
    "uniformly mediocre one at 0.550. N=2 is the F4 stress cell - the least "
    "trustworthy state in the system - so averaging inverts the ranking "
    "exactly where it matters. Pure MIN is too blunt in the other direction "
    "(good N with everything else weak scores 0.300, ignoring five factors), "
    "so the weakest factor CAPS the confidence and the weighted sum may only "
    "lower it."
)

# Confidence bands, for reading. Boundaries are NOT round guesses: they are
# the points where the sample-size factor crosses F4's measured claim tiers,
# so a band change means a claim-tier change rather than a cosmetic one.
FCONF_BAND_NONE = "NONE"        # nothing here supports a claim
FCONF_BAND_LOW = "LOW"
FCONF_BAND_MODERATE = "MODERATE"
FCONF_BAND_HIGH = "HIGH"

FORECAST_CONFIDENCE_BANDS: tuple[tuple[str, float], ...] = (
    (FCONF_BAND_NONE, 0.0),
    (FCONF_BAND_LOW, 0.25),
    (FCONF_BAND_MODERATE, 0.50),
    (FCONF_BAND_HIGH, 0.75),
)

# A confidence value is never published without the factor that bound it.
# A bare scalar is what lets a reader treat it as P(up).
FORECAST_CONFIDENCE_REQUIRE_BINDING_FACTOR = True

# Stated on every payload.
FORECAST_CONFIDENCE_DISCLAIMER = (
    "confidence describes HOW RELIABLE this forecast is, not how bullish. It "
    "is not P(up): a probability of 0.50 from 1,000 observations is a "
    "high-confidence statement, and a probability of 1.00 from 2 observations "
    "is a near-zero-confidence one."
)


def _validate_forecast_confidence_config() -> None:
    """Import-time guard for the F7 contract."""
    if len(set(FORECAST_CONFIDENCE_FACTORS)) != len(FORECAST_CONFIDENCE_FACTORS):
        raise ValueError("FORECAST_CONFIDENCE_FACTORS contains a duplicate")
    if len(FORECAST_CONFIDENCE_FACTORS) != 9:
        raise ValueError(
            "the sprint names nine factors; dropping one would silently narrow "
            "what the confidence claims to account for"
        )
    unknown = set(FORECAST_CONFIDENCE_MEASURABLE) - set(FORECAST_CONFIDENCE_FACTORS)
    if unknown:
        raise ValueError(f"measurable factors must be declared ones: {sorted(unknown)}")
    if not FORECAST_CONFIDENCE_MEASURABLE:
        raise ValueError(
            "no factor is measurable - a confidence built from nothing is not "
            "a confidence"
        )
    if set(FORECAST_CONFIDENCE_WEIGHTS) != set(FORECAST_CONFIDENCE_MEASURABLE):
        raise ValueError(
            "weights must cover exactly the MEASURABLE factors: an unmeasurable "
            "factor with a weight would contribute a number to a measurement "
            "that never happened"
        )
    total = float(sum(FORECAST_CONFIDENCE_WEIGHTS.values()))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"FORECAST_CONFIDENCE_WEIGHTS must sum to 1.0, got {total!r}"
        )
    for name, weight in FORECAST_CONFIDENCE_WEIGHTS.items():
        if weight <= 0.0:
            raise ValueError(
                f"weight for {name!r} must be positive - a zero weight is a "
                f"factor that is declared but cannot matter"
            )
    if FORECAST_CONFIDENCE_WEIGHTS[FCONF_SAMPLE_SIZE] < max(
        FORECAST_CONFIDENCE_WEIGHTS.values()
    ):
        raise ValueError(
            "sample_size must carry the largest weight: it is the factor "
            "MEASURED to be fatal when averaged away (N=2 scoring 0.717)"
        )

    if len(set(FORECAST_CONFIDENCE_STATUSES)) != len(FORECAST_CONFIDENCE_STATUSES):
        raise ValueError("FORECAST_CONFIDENCE_STATUSES contains a duplicate")
    if FCONF_STATUS_UNMEASURABLE not in FORECAST_CONFIDENCE_STATUSES:
        raise ValueError(
            "UNMEASURABLE must stay distinct from a measured zero: calibration, "
            "model agreement and drift all need a trained model that does not "
            "exist, and scoring them 0.0 or 1.0 would both be false"
        )

    if FORECAST_CONFIDENCE_AGGREGATION != "limiting_factor":
        raise ValueError(
            "confidence must aggregate by LIMITING FACTOR. "
            + FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE
        )
    if "MEASURED" not in FORECAST_CONFIDENCE_AGGREGATION_EVIDENCE:
        raise ValueError(
            "the aggregation evidence must carry its measurement, or a reader "
            "cannot tell a finding from a preference"
        )
    if not FORECAST_CONFIDENCE_REQUIRE_BINDING_FACTOR:
        raise ValueError(
            "a confidence value must always travel with the factor that bound "
            "it; a bare scalar is what lets a reader treat it as P(up)"
        )

    bands = [value for _name, value in FORECAST_CONFIDENCE_BANDS]
    if bands != sorted(bands):
        raise ValueError("FORECAST_CONFIDENCE_BANDS must ascend")
    if bands[0] != 0.0:
        raise ValueError("the lowest band must start at 0.0 so every value lands")
    if "not P(up)" not in FORECAST_CONFIDENCE_DISCLAIMER:
        raise ValueError(
            "the disclaimer must keep saying confidence is not P(up) - that "
            "conflation is the specific failure this sprint exists to prevent"
        )


_validate_forecast_confidence_config()


# ---------------------------------------------------------------------------
# Sprint F8 - The versioned ForecastSnapshot API contract
# ---------------------------------------------------------------------------
# One object that carries everything F1-F7 produced, versioned so a consumer
# can depend on its shape and a stored snapshot can be replayed.
#
# MEASURED, a forecast is deterministic - two runs of the same request produce
# the identical digest (6deb5e9815782497 both times). That is what makes
# versioning worth doing: a snapshot is an identity that can be compared and
# replayed, not merely a record that was stored.
#
# THREE OF THE SIXTEEN REQUESTED FIELDS HAVE NO HONEST PRODUCER TODAY:
#
#   model_versions      NO_MODEL - no trained forecasting model is registered
#   expected_return     F3 emits zero values for the same reason
#   model_contributions a NAMING CONFLICT with a measurement F6 already made
#
# A field with no producer is ABSENT WITH A REASON, never null and never zero.
# `expected_return: None` would coalesce to 0.0 under the dashboard's
# `Number(x ?? 0)` idiom and render as "flat" - the identical hazard F3, F4, F5
# and F6 each guard against, arriving through the API surface instead.
#
# THE model_contributions CONFLICT. F6 MEASURED that contributions cannot be
# reported: over 4,000 observations with a realistic regime/chart correlation,
# regime alone +0.064, chart alone +0.067, sum +0.131, ACTUAL joint effect
# +0.060. The parts overlap and double-count by more than 2x, and F6's
# validator bars CONTRIBUTION from its vocabulary. So this field carries F6's
# decomposition UNCHANGED and the contract states that the name is not to be
# read literally. Renaming F6's output to match the field would undo a
# measurement; leaving the field out would break the requested contract.
FORECAST_SNAPSHOT_VERSION = "forecast-snapshot-v1"
FORECAST_SNAPSHOT_CONTRACT_VERSION = "forecast-snapshot-contract-v1"

# The declared field order. This IS the contract: a consumer may rely on every
# name being present, and on absence being explicit rather than missing.
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "ticker",
    "as_of",
    "forecast_version",
    "model_versions",
    "horizon",
    "expected_return",
    "probability_up",
    "prediction_interval",
    "confidence",
    "regime",
    "event_context",
    "feature_digest",
    "evidence",
    "model_contributions",
    "warnings",
)

# Fields that cannot be produced today, each with the reason. Declared as DATA
# so a reader learns WHY a field is empty without reading code, and so wiring
# one later is a deliberate edit here.
SNAPSHOT_UNAVAILABLE_FIELDS: dict[str, str] = {
    "model_versions": (
        "no trained forecasting model is registered - build_joint_forecast "
        "reports NO_MODEL, and the measured incumbent is a rule-based "
        "baseline. An empty dict here would claim a model ran and declared "
        "nothing; ABSENT says no model ran at all"
    ),
    "expected_return": (
        "producing a return forecast needs a trained model, and none exists. "
        "A value of 0.0 would render as 'flat' in any consumer that coalesces "
        "nulls, which is a confident claim about a quantity nobody computed"
    ),
}

# Per-field status. ABSENT is deliberately distinct from a present-but-empty
# value, for the same reason F6's NOT_WIRED and F7's UNMEASURABLE are.
SNAPSHOT_STATUS_PRESENT = "PRESENT"
SNAPSHOT_STATUS_ABSENT = "ABSENT"        # no producer exists yet
SNAPSHOT_STATUS_REFUSED = "REFUSED"      # a producer exists and declined here

SNAPSHOT_STATUSES: tuple[str, ...] = (
    SNAPSHOT_STATUS_PRESENT,
    SNAPSHOT_STATUS_ABSENT,
    SNAPSHOT_STATUS_REFUSED,
)

# Fields whose absence makes the whole snapshot meaningless. A snapshot
# missing one of these is not a degraded forecast, it is not a forecast.
SNAPSHOT_REQUIRED_FIELDS: tuple[str, ...] = (
    "ticker", "as_of", "forecast_version", "horizon",
)

# The sub-objects that carry their own contract version, so a reader can tell
# which sprint produced which part of the snapshot and replay against it.
SNAPSHOT_VERSIONED_PARTS: tuple[str, ...] = (
    "forecast_version", "confidence", "event_context", "model_contributions",
)

# `model_contributions` is F6's decomposition verbatim. Stated as data so the
# reason the name is not literal travels with the contract.
SNAPSHOT_CONTRIBUTIONS_NOTE = (
    "this field carries the F6 forecast decomposition UNCHANGED. Despite the "
    "field name, the components are NOT contributions and do NOT sum to the "
    "forecast: MEASURED, regime alone +0.064 and chart alone +0.067 sum to "
    "+0.131 against an ACTUAL joint effect of +0.060, because the filters "
    "overlap. The decomposition reports what evidence entered, not what "
    "caused the outcome."
)

# Warnings a snapshot always surfaces rather than leaving in a sub-object.
# A consumer that renders only the headline must still see these.
SNAPSHOT_WARNING_NO_MODEL = "no_trained_model"
SNAPSHOT_WARNING_ABSENT_FIELDS = "fields_absent"
SNAPSHOT_WARNING_LOW_CONFIDENCE = "low_confidence"
SNAPSHOT_WARNING_INFERRED_EVIDENCE = "inferred_evidence"
SNAPSHOT_WARNING_THIN_SAMPLE = "thin_sample"

SNAPSHOT_WARNINGS: tuple[str, ...] = (
    SNAPSHOT_WARNING_NO_MODEL,
    SNAPSHOT_WARNING_ABSENT_FIELDS,
    SNAPSHOT_WARNING_LOW_CONFIDENCE,
    SNAPSHOT_WARNING_INFERRED_EVIDENCE,
    SNAPSHOT_WARNING_THIN_SAMPLE,
)

# Below this confidence the snapshot carries a low_confidence warning. Set to
# F7's MODERATE band floor so the warning means "below moderate", a boundary
# that already has a measured meaning rather than a new invented one.
SNAPSHOT_LOW_CONFIDENCE_THRESHOLD = 0.50

# Below this observed share the snapshot warns that its evidence was dated by
# inference. Matches EVENT_FORECAST_MIN_OBSERVED_SHARE so one number governs.
SNAPSHOT_MIN_OBSERVED_SHARE = 0.50


def _validate_forecast_snapshot_config() -> None:
    """Import-time guard for the F8 contract."""
    if len(set(SNAPSHOT_FIELDS)) != len(SNAPSHOT_FIELDS):
        raise ValueError("SNAPSHOT_FIELDS contains a duplicate")
    if len(SNAPSHOT_FIELDS) != 15:
        raise ValueError(
            f"the contract declares 15 fields, got {len(SNAPSHOT_FIELDS)} - "
            f"adding or dropping one changes what consumers may rely on"
        )
    unknown = set(SNAPSHOT_UNAVAILABLE_FIELDS) - set(SNAPSHOT_FIELDS)
    if unknown:
        raise ValueError(
            f"unavailable fields must be declared contract fields: {sorted(unknown)}"
        )
    for name, reason in SNAPSHOT_UNAVAILABLE_FIELDS.items():
        if not reason:
            raise ValueError(
                f"{name!r} is declared unavailable with no reason - a consumer "
                f"cannot tell a gap from an oversight"
            )
    missing_required = set(SNAPSHOT_REQUIRED_FIELDS) - set(SNAPSHOT_FIELDS)
    if missing_required:
        raise ValueError(f"required fields must be declared: {sorted(missing_required)}")
    overlap = set(SNAPSHOT_REQUIRED_FIELDS) & set(SNAPSHOT_UNAVAILABLE_FIELDS)
    if overlap:
        raise ValueError(
            f"a field cannot be both REQUIRED and permanently unavailable: "
            f"{sorted(overlap)}"
        )
    missing_parts = set(SNAPSHOT_VERSIONED_PARTS) - set(SNAPSHOT_FIELDS)
    if missing_parts:
        raise ValueError(f"versioned parts must be fields: {sorted(missing_parts)}")

    if len(set(SNAPSHOT_STATUSES)) != len(SNAPSHOT_STATUSES):
        raise ValueError("SNAPSHOT_STATUSES contains a duplicate")
    if SNAPSHOT_STATUS_ABSENT not in SNAPSHOT_STATUSES:
        raise ValueError(
            "ABSENT must stay distinct from a present-but-empty value: "
            "expected_return = 0.0 would render as 'flat' in a consumer that "
            "coalesces nulls, which is a confident claim about a quantity "
            "nobody computed"
        )
    if len(set(SNAPSHOT_WARNINGS)) != len(SNAPSHOT_WARNINGS):
        raise ValueError("SNAPSHOT_WARNINGS contains a duplicate")
    if SNAPSHOT_WARNING_NO_MODEL not in SNAPSHOT_WARNINGS:
        raise ValueError(
            "the no-model warning must stay declared while NO_MODEL is the "
            "system's actual state"
        )
    if not 0.0 < SNAPSHOT_LOW_CONFIDENCE_THRESHOLD < 1.0:
        raise ValueError("the low-confidence threshold must lie inside (0, 1)")
    if not 0.0 <= SNAPSHOT_MIN_OBSERVED_SHARE <= 1.0:
        raise ValueError("the observed-share threshold must lie in [0, 1]")

    for phrase in ("NOT contributions", "do NOT sum", "MEASURED"):
        if phrase not in SNAPSHOT_CONTRIBUTIONS_NOTE:
            raise ValueError(
                f"the contributions note must keep saying {phrase!r}: the "
                f"field name invites exactly the additive reading F6 measured "
                f"to be wrong"
            )


_validate_forecast_snapshot_config()


# ---------------------------------------------------------------------------
# Sprint M5 - Promotion gates and drift hooks
# ---------------------------------------------------------------------------
# The checklist a candidate model must pass before it can serve. Sprint L (the
# learning loop) depends on this: "promotion requires out-of-sample
# comparison, drift checks, reproducibility, and a release gate" is its stated
# precondition, so the loop cannot close without it.
#
# MEASURED, the gap is NARROWER than the sprint doc implies.
# `ModelRegistry.promote()` already refuses a candidate without an OOS
# comparison and without a named human approver. Three of the five required
# checks are enforced NOWHERE:
#
#   [x] OOS beats the incumbent on the pre-registered metric  (promote())
#   [x] human approval recorded                               (promote())
#   [ ] no veto-rate / false-positive regression
#   [ ] drift check clean
#   [ ] manifest complete
#
# So M5 adds the three missing checks and the script that SEQUENCES all five,
# rather than reimplementing the two that already work (W5).
#
# EVERY CHECK IS A REFUSAL, NOT A SCORE. A promotion checklist that produces a
# number invites "close enough". Each check returns PASS, FAIL or a third
# state - NOT_EVALUATED - and any FAIL stops the promotion. NOT_EVALUATED is
# deliberately distinct from PASS: a drift check that could not run has not
# found the candidate clean, it has found nothing.
PROMOTION_GATE_VERSION = "promotion-gate-v1"
PROMOTION_CONTRACT_VERSION = "promotion-contract-v1"

# The checks, in evaluation order. Ordered cheapest-and-most-fundamental
# first, so a candidate missing its manifest is rejected before anyone spends
# time on drift.
PROMO_CHECK_MANIFEST = "manifest_complete"
PROMO_CHECK_OOS = "oos_beats_incumbent"
PROMO_CHECK_REGRESSION = "no_quality_regression"
PROMO_CHECK_DRIFT = "drift_clean"
PROMO_CHECK_APPROVAL = "human_approval"

PROMOTION_CHECKS: tuple[str, ...] = (
    PROMO_CHECK_MANIFEST,
    PROMO_CHECK_OOS,
    PROMO_CHECK_REGRESSION,
    PROMO_CHECK_DRIFT,
    PROMO_CHECK_APPROVAL,
)

# Check outcomes. NOT_EVALUATED is the load-bearing third state.
PROMO_PASS = "PASS"
PROMO_FAIL = "FAIL"
PROMO_NOT_EVALUATED = "NOT_EVALUATED"

PROMOTION_OUTCOMES: tuple[str, ...] = (PROMO_PASS, PROMO_FAIL, PROMO_NOT_EVALUATED)

# Checks that must reach PASS. A check left NOT_EVALUATED blocks promotion
# exactly as a FAIL does - absence of evidence is not evidence of safety.
PROMOTION_REQUIRED_CHECKS: tuple[str, ...] = PROMOTION_CHECKS

# Manifest fields a candidate must carry before anything else is considered.
# Drawn from ModelEntry: these are the fields without which a promotion could
# not be reproduced or audited afterwards.
PROMOTION_REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "model_version",
    "family",
    "feature_set_version",
    "training_data_cutoff",
    "dataset_hash",
    "code_commit",
    "seed",
    "artifact_hash",
    "hyperparameters",
    "forecast_target",
    "horizon",
)

# Population Stability Index thresholds. Industry-conventional and MEASURED to
# discriminate here: over 400 synthetic scores, an unshifted distribution
# scores PSI 0.013 while a +1.7-point shift scores 2.854.
PROMOTION_PSI_WARN = 0.10
PROMOTION_PSI_FAIL = 0.25

# How much the candidate's veto rate may exceed the incumbent's before the
# regression check fails. A candidate that vetoes far more is not safer, it is
# less useful; one that vetoes far less may have lost a guard.
PROMOTION_MAX_VETO_RATE_INCREASE = 0.10

# ...and how far it may fall. A large DROP in veto rate is also a regression:
# the W2 policy rules exist to refuse bad decisions, and a candidate that
# stops refusing them has not improved, it has stopped checking.
PROMOTION_MAX_VETO_RATE_DECREASE = 0.10

# Minimum decisions before the regression check can say anything. Below this
# the check reports NOT_EVALUATED rather than a verdict on noise.
PROMOTION_MIN_DECISIONS_FOR_REGRESSION = 30

# Historical predictions are immutable: promotion never rewrites the model
# version recorded against a past decision. The append-only audit log is what
# guarantees it, and the gate verifies the guarantee rather than assuming it.
PROMOTION_HISTORY_IMMUTABLE = True


def _validate_promotion_config() -> None:
    """Import-time guard for the M5 contract."""
    if len(set(PROMOTION_CHECKS)) != len(PROMOTION_CHECKS):
        raise ValueError("PROMOTION_CHECKS contains a duplicate")
    if len(PROMOTION_CHECKS) != 5:
        raise ValueError(
            "the checklist has five checks; dropping one silently widens what "
            "may be promoted"
        )
    if PROMOTION_CHECKS[0] != PROMO_CHECK_MANIFEST:
        raise ValueError(
            "the manifest check must run FIRST - a candidate that cannot be "
            "reproduced should be rejected before anyone evaluates its metrics"
        )
    unknown = set(PROMOTION_REQUIRED_CHECKS) - set(PROMOTION_CHECKS)
    if unknown:
        raise ValueError(f"required checks must be declared: {sorted(unknown)}")
    if set(PROMOTION_REQUIRED_CHECKS) != set(PROMOTION_CHECKS):
        raise ValueError(
            "every check is required: an optional promotion gate is not a gate"
        )

    if len(set(PROMOTION_OUTCOMES)) != len(PROMOTION_OUTCOMES):
        raise ValueError("PROMOTION_OUTCOMES contains a duplicate")
    if PROMO_NOT_EVALUATED not in PROMOTION_OUTCOMES:
        raise ValueError(
            "NOT_EVALUATED must stay declared: a check that could not run has "
            "not found the candidate clean, and collapsing it into PASS would "
            "promote on absent evidence"
        )

    if not PROMOTION_REQUIRED_MANIFEST_FIELDS:
        raise ValueError("a promotion with no manifest requirements is unauditable")
    for field in ("model_version", "dataset_hash", "code_commit", "seed"):
        if field not in PROMOTION_REQUIRED_MANIFEST_FIELDS:
            raise ValueError(
                f"{field!r} must stay required: without it a promoted model "
                f"cannot be reproduced from its record"
            )

    if not 0.0 < PROMOTION_PSI_WARN < PROMOTION_PSI_FAIL:
        raise ValueError("PSI thresholds must ascend and be positive")
    if not 0.0 < PROMOTION_MAX_VETO_RATE_INCREASE <= 1.0:
        raise ValueError("the veto-rate increase tolerance must lie in (0, 1]")
    if not 0.0 < PROMOTION_MAX_VETO_RATE_DECREASE <= 1.0:
        raise ValueError(
            "a veto-rate DECREASE tolerance must exist and be positive: a "
            "candidate that stops refusing bad decisions has not improved"
        )
    if PROMOTION_MIN_DECISIONS_FOR_REGRESSION < 1:
        raise ValueError("the regression check needs at least one decision")
    if not PROMOTION_HISTORY_IMMUTABLE:
        raise ValueError(
            "historical predictions are immutable; promotion must never "
            "rewrite the model version recorded against a past decision"
        )


_validate_promotion_config()


# ---------------------------------------------------------------------------
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
