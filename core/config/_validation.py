"""Configuration part 2: V1-V8 labels/backtest and M1-M8 the ML layer.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_base` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._base import *  # noqa: F401,F403


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


