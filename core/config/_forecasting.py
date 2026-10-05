"""Configuration part: F1-F8 forecasting and the M5 promotion hooks.

Split out of the single 9,783-line `core/config.py` in C1, and then again
because the forecasting part alone was 3,631 lines — leaving it would have
moved the navigability problem rather than solved it. The text is UNCHANGED.

CHAINED from `_charts`: 88 constants are read across part boundaries, so
the parts reproduce ONE flat namespace in the original order.
"""

from core.config._charts import *  # noqa: F401,F403


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
#
# RAISED from 0.2 to 1.1 when COLLECT_NEWS_BATCH_SIZE went to 75. The two are
# one decision: Finnhub allows 60 calls per MINUTE, and at 0.2s a 75-ticker
# sweep issues 300/min -- five times the limit, so it would truncate on 429
# partway through and trade a KNOWN rotation for an unpredictable one.
#
#     75 tickers x 1.1s = 82.5s  ->  ~55 calls/min, under 60 with room
#
# 1.1 rather than 1.0 because the limit is enforced on a rolling window and the
# request itself takes non-zero time; a rate computed to land exactly on the
# boundary lands over it whenever the network is fast.
COLLECT_THROTTLE_SECONDS = 1.1

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

# Tickers per news run. RAISED from 40 to 75 once Finnhub became the leading
# provider, because the cap was a NewsAPI artifact.
#
# MEASURED: NewsAPI's free tier allows 100 requests/DAY, so 40 covered the 73
# sector-mapped holdings in two runs and left a margin for ad-hoc work. Finnhub
# allows 60 calls per MINUTE -- the daily ceiling is gone, and rotating cost
# coverage for nothing: half the portfolio was dark on any given day, and the
# alert run can only grade what was captured.
#
# THIS CONSTANT CANNOT MOVE ALONE. See COLLECT_THROTTLE_SECONDS below: at the
# previous 0.2s throttle a 75-ticker run would issue 300 calls/min, five times
# Finnhub's limit, and truncate on 429 partway through -- trading a known
# rotation for an unpredictable one, which is strictly worse. The two are
# validated against each other.
COLLECT_NEWS_BATCH_SIZE = 75

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
    # THE BATCH AND THE THROTTLE ARE ONE DECISION, so they are validated against
    # each other rather than separately.
    #
    # Finnhub's free tier allows 60 calls per MINUTE. A batch raised without
    # slowing the throttle issues them faster than that and truncates on 429
    # partway through the sweep -- which trades a KNOWN rotation for an
    # unpredictable one, strictly worse than the rotation it replaced. MEASURED:
    # 75 tickers at the former 0.2s throttle is 300 calls/min, five times over.
    if COLLECT_THROTTLE_SECONDS > 0:
        calls_per_minute = 60.0 / COLLECT_THROTTLE_SECONDS
        if calls_per_minute > FINNHUB_RATE_LIMIT_PER_MINUTE:
            raise ValueError(
                f"a {COLLECT_THROTTLE_SECONDS}s throttle issues "
                f"{calls_per_minute:.0f} calls/min against Finnhub's "
                f"{FINNHUB_RATE_LIMIT_PER_MINUTE}/min limit; a sweep at this "
                f"rate truncates on 429 partway through, which is worse than "
                f"the rotation it replaced"
            )
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
