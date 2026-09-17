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
MACRO_ADAPTER_VERSION = "macro-adapter-v2"

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
# Risk-on/off regime thresholds (0.5 = neutral, >0.5 = risk-on tilt).
MACRO_RISKOFF_THRESHOLD = 0.3
MACRO_RISKON_THRESHOLD = 0.7

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

OUTCOME_LABEL_VERSION = "outcome-label-v1"

# Horizon name -> trading-session count after the as_of entry bar.
LABEL_HORIZON_SESSIONS = {"1d": 1, "5d": 5, "20d": 20, "60d": 60}

# label_20d_up is True when forward_return_20d is STRICTLY greater than this
# threshold (0.0 = any positive forward return).
OUTCOME_LABEL_UP_THRESHOLD = 0.0

# Calendar-day coverage the provider fetch must reach AHEAD of as_of (~60
# sessions of future window plus a holiday buffer) so matured horizons are
# actually visible in the fetched frame.
LABEL_CALENDAR_COVERAGE_DAYS = 130


def _validate_label_config() -> None:
    """Import-time guard: horizons are positive sessions, threshold sane."""
    if not LABEL_HORIZON_SESSIONS:
        raise ValueError("LABEL_HORIZON_SESSIONS must not be empty")
    for name, sessions in LABEL_HORIZON_SESSIONS.items():
        if not isinstance(sessions, int) or sessions < 1:
            raise ValueError(f"LABEL_HORIZON_SESSIONS[{name!r}] must be a positive int, got {sessions!r}")
    if OUTCOME_LABEL_UP_THRESHOLD < 0.0:
        raise ValueError("OUTCOME_LABEL_UP_THRESHOLD must be non-negative")


_validate_label_config()

# --- Walk-forward backtest (V2) --------------------------------------------------
# Validation infrastructure only — nothing here is a production trading path.
# Folding: train [t0, t1] -> embargo (>= max label horizon, i.e. 60 sessions)
# -> validation -> advance; the final frozen configuration is evaluated once
# on a never-touched tail holdout. The embargo protects both features (PIT
# eligibility, already enforced) and labels (V1 boundary rule).

BACKTEST_EMBARGO_SESSIONS = 60
BACKTEST_FOLD_SESSIONS = 120
BACKTEST_HOLDOUT_SESSIONS = 60

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
FORECAST_TARGETS = (
    FORECAST_TARGET_RETURN, FORECAST_TARGET_DIRECTION, FORECAST_TARGET_VOLATILITY,
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
