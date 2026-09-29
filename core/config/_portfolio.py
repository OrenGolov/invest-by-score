"""Configuration part 6: R3-R7 portfolio risk and D1-D4 the dashboard.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_sizing` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._sizing import *  # noqa: F401,F403


# --- R3: sector concentration ----------------------------------------------------
# R1 established that weight alone is gameable: satisfying a 10% per-name cap
# by splitting 40% NVDA across four correlated semiconductors raised portfolio
# volatility. R3 is that finding one level up.
#
# MEASURED on the tracked portfolio: 38 of 73 sector-mapped holdings are
# Information Technology - 52.1% of the book. Every one of those 38 names sits
# near 1.3% by equal weight, so the entire concentration respects a 10%
# per-name cap while being a single sector bet.
SECTOR_CONCENTRATION_VERSION = "sector-concentration-v1"

# Sector shares are reported on BOTH scales, for the same reason R1 refuses a
# weight-only exposure report: they answer different questions and can
# disagree. Weight says how much capital sits in a sector; risk says how much
# of the portfolio's variance it drives.
SECTOR_BASIS_WEIGHT = "weight"
SECTOR_BASIS_RISK = "risk_contribution"
SECTOR_BASES: tuple[str, ...] = (SECTOR_BASIS_WEIGHT, SECTOR_BASIS_RISK)

# A sector's share of portfolio VALUE above which it is flagged for review.
# MEASURED, the tracked portfolio's IT share is 52.1% - more than double this
# - while no single holding breaches R1's 25% per-name review threshold.
SECTOR_WEIGHT_REVIEW = 0.25

# A sector's share of portfolio VARIANCE above which it is flagged. Set above
# the weight threshold because sector risk concentrates faster than sector
# weight: correlated names inside one sector contribute jointly.
SECTOR_RISK_REVIEW = 0.40

# Herfindahl-Hirschman index over sector shares, and the effective sector
# count it implies (1/HHI). MEASURED, the tracked portfolio scores HHI 0.3113
# across 10 sectors present - an effective diversification of 3.21 sectors.
# Holding ten sectors is not the same as being diversified across ten, and the
# count is what makes that visible.
SECTOR_HHI_REVIEW = 0.25

# Holdings whose sector cannot be established are reported as a distinct
# UNCLASSIFIED bucket and never folded into a sector. MEASURED, 4 of 77
# tracked holdings are funds (VOO, SOXX, CIBR, NASA): a fund has no single
# sector, and assigning one would invent concentration that is not there --
# or hide concentration that is.
SECTOR_UNCLASSIFIED = "UNCLASSIFIED"

# Unclassified share above which the sector picture itself is untrustworthy.
# Below the threshold the measured sectors still describe most of the book;
# above it, a "top sector 30%" claim is really "30% of the part we could
# classify", which is a different and weaker statement.
SECTOR_UNCLASSIFIED_REVIEW = 0.20

# R3 DESCRIBES concentration; it does not block trades. R7 is where a
# portfolio-level NO_TRADE is decided, with its own evidence. A measurement
# that silently blocked trades would be a policy wearing a measurement's
# clothes -- the same rule R1 states and for the same reason.
SECTOR_BLOCKS_TRADES = False


def _validate_sector_concentration_config() -> None:
    """Import-time guard for the R3 contract."""
    if SECTOR_BASIS_WEIGHT not in SECTOR_BASES or SECTOR_BASIS_RISK not in SECTOR_BASES:
        raise ValueError(
            "sector concentration is reported on both weight and risk: "
            "MEASURED, 38 IT names at ~1.3% each respect every per-name weight "
            "cap while forming a 52.1% single-sector bet"
        )
    if SECTOR_RISK_REVIEW <= SECTOR_WEIGHT_REVIEW:
        raise ValueError(
            "the risk threshold sits above the weight threshold: correlated "
            "names inside one sector contribute jointly, so sector variance "
            "concentrates faster than sector weight"
        )
    for name, value in (
        ("SECTOR_WEIGHT_REVIEW", SECTOR_WEIGHT_REVIEW),
        ("SECTOR_RISK_REVIEW", SECTOR_RISK_REVIEW),
        ("SECTOR_HHI_REVIEW", SECTOR_HHI_REVIEW),
        ("SECTOR_UNCLASSIFIED_REVIEW", SECTOR_UNCLASSIFIED_REVIEW),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be a share in (0, 1], got {value!r}")
    if not SECTOR_UNCLASSIFIED:
        raise ValueError(
            "unclassified holdings need their own bucket: a fund has no single "
            "sector, and folding it into one invents or hides concentration"
        )
    if SECTOR_BLOCKS_TRADES:
        raise ValueError(
            "R3 describes concentration, it does not enforce it. The "
            "portfolio-level NO_TRADE is R7's decision, with its own evidence"
        )


_validate_sector_concentration_config()


# --- R4: forecast-adjusted risk --------------------------------------------------
# R2 sized a position from correlation alone: how much of this can the
# portfolio absorb? That question never asks whether the forecast motivating
# the trade is worth acting on. A 51% P(up) from two observations and a 51%
# from eight hundred produce the identical size.
#
# THE DECIDING MEASUREMENT: scaling size by confidence helps in one direction
# and HURTS in the other. On a candidate correlated with what is held and more
# volatile, shrinking by confidence walked portfolio volatility from +19.64%
# down to +0.98%. On an uncorrelated candidate the SAME scaling walked it from
# -19.23% to -0.99% -- it removed a benefit. Shrinking a low-confidence
# diversifier makes the portfolio worse, not safer.
#
# MEASURED, the ordering is not even close: a HIGH-confidence concentrator at
# 20% raised volatility +19.78%, while a LOW-confidence diversifier at 5%
# LOWERED it -4.82%. So confidence ADJUSTS R2's answer; it never replaces the
# correlation analysis, and it is never the only input.
FORECAST_RISK_VERSION = "forecast-adjusted-risk-v1"

# Confidence scales the size DOWNWARD only. It is a haircut on conviction, not
# a multiplier: a confident forecast earns R2's size, never more than it.
# MEASURED, allowing >1.0 on the TWIN candidate pushed volatility past +19.64%
# for no reason other than the model liking its own forecast.
FORECAST_RISK_MAX_MULTIPLIER = 1.0

# The floor below which a forecast buys no position at all. A confidence-
# scaled size decays smoothly toward zero, which silently produces positions
# too small to be real -- R2 already refuses a trade under SIZING_MIN_WEIGHT.
# Below this band the honest answer is NO_TRADE, not a token position.
FORECAST_RISK_MIN_CONFIDENCE = 0.25

# Direction matters more than confidence, which is why the haircut is applied
# ONLY to candidates that increase portfolio risk. MEASURED, applying it to a
# risk-REDUCING candidate degraded the portfolio monotonically at every
# confidence level tested (-19.23%, -14.54%, -9.76%, -4.91%, -0.99%).
FORECAST_RISK_HAIRCUT_INCREASERS_ONLY = True

# An UNMEASURABLE confidence factor is not a low one. MEASURED, calibration,
# model_agreement and model_drift are all UNMEASURABLE today because no model
# is registered; F7 excludes them from the weighting rather than scoring them
# 0.0. R4 inherits that: it never converts "not measured" into a haircut,
# because a forecast is not less reliable for living in a system that has not
# yet trained a model.
FORECAST_RISK_UNMEASURABLE_IS_NOT_LOW = True

# Verdicts. Kept distinct for the reason R2 keeps REFUSED and NOT_EVALUATED
# apart: "the forecast is too weak to act on" and "no confidence could be
# computed" are different answers, and collapsing them hides the second.
FRISK_NOT_EVALUATED = "NOT_EVALUATED"  # no confidence or no R2 size to adjust
FRISK_NO_TRADE = "NO_TRADE"            # confidence below the floor
FRISK_REDUCED = "REDUCED"              # haircut applied to a risk increaser
FRISK_UNCHANGED = "UNCHANGED"          # risk reducer, or full confidence
FRISK_VERDICTS: tuple[str, ...] = (
    FRISK_NOT_EVALUATED,
    FRISK_NO_TRADE,
    FRISK_REDUCED,
    FRISK_UNCHANGED,
)

# R4 adjusts a proposed size. It does not decide whether to trade: that is R7,
# with the whole portfolio in view. The same rule as R1 and R3.
FORECAST_RISK_BLOCKS_TRADES = False


def _validate_forecast_risk_config() -> None:
    """Import-time guard for the R4 contract."""
    if FORECAST_RISK_MAX_MULTIPLIER != 1.0:
        raise ValueError(
            "confidence is a haircut, not a multiplier: a confident forecast "
            "earns R2's size and never more than it"
        )
    if not 0.0 < FORECAST_RISK_MIN_CONFIDENCE < 1.0:
        raise ValueError(
            f"FORECAST_RISK_MIN_CONFIDENCE must be a share in (0, 1), got "
            f"{FORECAST_RISK_MIN_CONFIDENCE!r}"
        )
    if not FORECAST_RISK_HAIRCUT_INCREASERS_ONLY:
        raise ValueError(
            "the haircut applies only to risk INCREASERS: MEASURED, scaling a "
            "risk-reducing candidate by confidence degraded the portfolio at "
            "every level tested (-19.23% through -0.99%)"
        )
    if not FORECAST_RISK_UNMEASURABLE_IS_NOT_LOW:
        raise ValueError(
            "an UNMEASURABLE confidence factor is not a low one; F7 excludes "
            "it from the weighting rather than scoring it 0.0"
        )
    if FRISK_NO_TRADE == FRISK_NOT_EVALUATED:
        raise ValueError(
            "'too weak to act on' and 'no confidence could be computed' are "
            "different answers; collapsing them hides the second"
        )
    if len(set(FRISK_VERDICTS)) != len(FRISK_VERDICTS):
        raise ValueError("duplicate R4 verdict")
    if FORECAST_RISK_BLOCKS_TRADES:
        raise ValueError(
            "R4 adjusts a size; whether to trade at all is R7's decision"
        )


_validate_forecast_risk_config()


# --- R5: expected portfolio impact -----------------------------------------------
# R2 answered "how much of this can the portfolio absorb?" and R4 answered "how
# much has the forecast earned?". Neither answers what the portfolio LOOKS LIKE
# afterwards, which is the question a decision actually rests on.
#
# THE DECIDING MEASUREMENT: volatility and concentration disagree about half
# the time. Across 20 seeded markets x 2 candidates, 21 of 40 trades (52%)
# moved portfolio volatility and risk concentration in OPPOSITE directions. On
# the reference market a correlated candidate raised volatility +5.00% while
# IMPROVING risk concentration (risk-HHI 0.5000 -> 0.3736), and a diversifier
# cut volatility -20.56% while leaving concentration untouched (0.5005).
# A single risk number cannot answer both questions, so R5 reports both and
# never collapses them into one verdict.
EXPECTED_IMPACT_VERSION = "expected-impact-v1"

# WHAT R5 REFUSES TO ESTIMATE. There is no trained forecasting model:
# build_joint_forecast reports UNAVAILABLE, and SNAPSHOT_UNAVAILABLE_FIELDS
# already records that expected_return "needs a trained model, and none
# exists". So R5 projects RISK, which is measurable from the covariance, and
# never RETURN, which is not. An expected-return impact would be a confident
# claim about a quantity nobody computed - the exact hazard F3/F4/F5/F6 and the
# snapshot contract each guard against.
IMPACT_PROJECTS_RETURN = False
IMPACT_RETURN_REASON = (
    "projecting an expected return needs a trained model and none is "
    "registered; build_joint_forecast reports UNAVAILABLE. A 0.0 impact would "
    "render as 'flat' in any consumer that coalesces nulls, which is a "
    "confident claim about a quantity nobody computed"
)

# The dimensions R5 projects. Each is measurable from the covariance alone.
IMPACT_DIMENSION_VOLATILITY = "volatility"
IMPACT_DIMENSION_CONCENTRATION = "risk_concentration"
IMPACT_DIMENSION_LARGEST = "largest_risk_share"
IMPACT_DIMENSIONS: tuple[str, ...] = (
    IMPACT_DIMENSION_VOLATILITY,
    IMPACT_DIMENSION_CONCENTRATION,
    IMPACT_DIMENSION_LARGEST,
)

# A relative move in portfolio volatility worth reporting as material. Set at
# the R2 risk budget so the two agree about what "a lot" means: a trade R2
# sized to the edge of its budget is exactly a material one here.
IMPACT_MATERIAL_VOLATILITY = SIZING_RISK_BUDGET

# A move in risk-HHI worth reporting. MEASURED, the reference correlated trade
# moved risk-HHI by 0.1264 (0.5000 -> 0.3736) while the diversifier moved it
# 0.0005 - three orders of magnitude apart. A threshold between them separates
# a trade that reshapes who drives risk from one that does not.
IMPACT_MATERIAL_CONCENTRATION = 0.05

# Directions. Kept distinct from verdicts: R5 states which way each dimension
# moved and by how much. It does not rule, because MEASURED the dimensions
# disagree 52% of the time and a single verdict would have to silently pick a
# winner. R7 weighs them with the whole portfolio in view.
IMPACT_IMPROVES = "IMPROVES"
IMPACT_WORSENS = "WORSENS"
IMPACT_NEUTRAL = "NEUTRAL"
IMPACT_NOT_EVALUATED = "NOT_EVALUATED"
IMPACT_DIRECTIONS: tuple[str, ...] = (
    IMPACT_IMPROVES,
    IMPACT_WORSENS,
    IMPACT_NEUTRAL,
    IMPACT_NOT_EVALUATED,
)

# R5 projects; it does not decide. Same rule as R1, R3 and R4.
IMPACT_BLOCKS_TRADES = False


def _validate_expected_impact_config() -> None:
    """Import-time guard for the R5 contract."""
    if IMPACT_PROJECTS_RETURN:
        raise ValueError(
            "R5 must not project an expected return: no trained model exists, "
            "and a fabricated 0.0 renders as 'flat' to any consumer that "
            "coalesces nulls"
        )
    if not IMPACT_RETURN_REASON.strip():
        raise ValueError("the refusal to project return must carry its reason")
    if len(set(IMPACT_DIMENSIONS)) != len(IMPACT_DIMENSIONS):
        raise ValueError("duplicate R5 impact dimension")
    if IMPACT_DIMENSION_VOLATILITY not in IMPACT_DIMENSIONS:
        raise ValueError("volatility is not among the projected dimensions")
    if IMPACT_DIMENSION_CONCENTRATION not in IMPACT_DIMENSIONS:
        raise ValueError(
            "concentration is not projected: MEASURED, volatility and "
            "concentration moved in opposite directions on 21 of 40 trades, "
            "so volatility alone answers only half the question"
        )
    if not 0.0 < IMPACT_MATERIAL_VOLATILITY < 1.0:
        raise ValueError(
            f"IMPACT_MATERIAL_VOLATILITY must be a share in (0, 1), got "
            f"{IMPACT_MATERIAL_VOLATILITY!r}"
        )
    if not 0.0 < IMPACT_MATERIAL_CONCENTRATION < 1.0:
        raise ValueError(
            f"IMPACT_MATERIAL_CONCENTRATION must be in (0, 1), got "
            f"{IMPACT_MATERIAL_CONCENTRATION!r}"
        )
    if len(set(IMPACT_DIRECTIONS)) != len(IMPACT_DIRECTIONS):
        raise ValueError("duplicate R5 direction")
    if IMPACT_NOT_EVALUATED == IMPACT_NEUTRAL:
        raise ValueError(
            "'no impact could be measured' and 'the impact was negligible' "
            "are different answers; collapsing them hides the first"
        )
    if IMPACT_BLOCKS_TRADES:
        raise ValueError("R5 projects an impact; R7 decides whether to trade")


_validate_expected_impact_config()


# --- R6: stress scenarios --------------------------------------------------------
# R1-R5 all rest on ONE covariance estimated over a long, mostly-calm window.
# That covariance is the thing a stressed market breaks, so every answer built
# on it is a calm-market answer unless it is re-asked under stress.
#
# THE DECIDING MEASUREMENT, on 76 real tickers over 1,170 common sessions:
# correlation does not merely rise under stress, it roughly DOUBLES. Average
# pairwise correlation measured about the full-period mean:
#
#     stress definition   calm     stress   ratio
#     worst  5% of days   0.2421   0.6066   x2.51
#     worst 10% of days   0.2263   0.5534   x2.45
#     worst 15% of days   0.2214   0.4958   x2.24
#     worst 20% of days   0.2199   0.4591   x2.09
#
# On an equal-weight portfolio of those 76 names, portfolio volatility under
# stress is 2.30x calm, and the diversification ratio falls 38.1% (2.080 ->
# 1.288). Diversification weakens most in exactly the conditions it is held
# for, so a calm-window covariance overstates every hedge in the book.
STRESS_SCENARIO_VERSION = "stress-scenario-v1"

# THE TRAP THIS MODULE EXISTS TO AVOID, and it is not hypothetical: it is the
# first result this sprint produced. Estimating the stress covariance by
# demeaning WITHIN the stress subset reported average correlation FALLING
# under stress (0.2166 -> 0.1492, -31.1%) and the diversification ratio
# IMPROVING (2.128 -> 2.789). MEASURED, the stress decile has a mean market
# return of -3.15% against +0.14% over all sessions; subtracting the subset
# mean removes the crash itself, so every name looks as though it barely
# moved and co-movement collapses. The stress covariance is therefore measured
# about the FULL-PERIOD mean, which keeps the crash in the deviations.
STRESS_DEMEAN_FULL_PERIOD = True
STRESS_DEMEAN_REASON = (
    "demeaning within the stress subset subtracts the crash being measured: "
    "MEASURED, the worst decile averages -3.15% against +0.14% overall, and "
    "subset demeaning reported correlation FALLING 31.1% under stress and "
    "diversification IMPROVING - both artifacts of the estimator"
)

# Which sessions count as stressed. Defined by the worst market-wide returns
# rather than by a volatility filter, because a decision needs to know what
# happens when everything falls together, not when quotes are merely noisy.
STRESS_WORST_SHARE = 0.10

# The floor of sessions a stress estimate needs. Below this the covariance is
# estimated from too few observations to be an estimate; MEASURED, the 10%
# decile of 1,170 sessions is 117 days, comfortably above this floor.
STRESS_MIN_SESSIONS = 30

# Scenarios are DECLARED here as data, so a reader learns what was tested
# without reading code, and adding one is a deliberate edit. Each names a
# shock in the units of a daily return.
STRESS_SCENARIO_HISTORICAL = "historical_worst_decile"
STRESS_SCENARIO_CORRELATION_SHOCK = "correlation_to_one"
STRESS_SCENARIO_VOLATILITY_SHOCK = "volatility_x2"
STRESS_SCENARIOS: tuple[str, ...] = (
    STRESS_SCENARIO_HISTORICAL,
    STRESS_SCENARIO_CORRELATION_SHOCK,
    STRESS_SCENARIO_VOLATILITY_SHOCK,
)

# The correlation assumed in the correlation_to_one scenario. Not literally
# 1.0: a perfectly correlated matrix is singular and portfolio variance stops
# being well conditioned. MEASURED, the worst 5% decile already reaches 0.607
# average pairwise correlation, so 0.95 is a stress beyond the observed record
# without being a degenerate matrix.
STRESS_CORRELATION_LEVEL = 0.95

# The multiplier in the volatility_x2 scenario. MEASURED, portfolio volatility
# under the historical stress decile was 2.30x calm, so a 2x per-name shock is
# calibrated to the observed record rather than chosen for roundness.
STRESS_VOLATILITY_MULTIPLIER = 2.0

# A loss of diversification worth reporting. MEASURED, the historical decile
# costs 38.1% of the diversification ratio; a threshold well below that flags
# the observed case while leaving room for milder ones to pass quietly.
STRESS_MATERIAL_DEGRADATION = 0.15

# R6 measures what stress does. It does not refuse trades: that is R7, which
# weighs this against everything else. Same rule as R1, R3, R4 and R5.
STRESS_BLOCKS_TRADES = False


def _validate_stress_scenario_config() -> None:
    """Import-time guard for the R6 contract."""
    if not STRESS_DEMEAN_FULL_PERIOD:
        raise ValueError(
            "the stress covariance must be measured about the FULL-PERIOD "
            "mean: MEASURED, subset demeaning reported correlation FALLING "
            "31.1% under stress and diversification IMPROVING, because it "
            "subtracts away the -3.15% crash it is trying to measure"
        )
    if not STRESS_DEMEAN_REASON.strip():
        raise ValueError("the demeaning choice must carry its measured reason")
    if not 0.0 < STRESS_WORST_SHARE < 0.5:
        raise ValueError(
            f"STRESS_WORST_SHARE must be a minority of sessions, got "
            f"{STRESS_WORST_SHARE!r}"
        )
    if STRESS_MIN_SESSIONS < 2:
        raise ValueError("a covariance needs at least two observations")
    if len(set(STRESS_SCENARIOS)) != len(STRESS_SCENARIOS):
        raise ValueError("duplicate stress scenario")
    if STRESS_SCENARIO_HISTORICAL not in STRESS_SCENARIOS:
        raise ValueError(
            "the historical scenario is mandatory: a hypothetical shock that "
            "is never checked against the observed record is an assumption"
        )
    if not 0.0 < STRESS_CORRELATION_LEVEL < 1.0:
        raise ValueError(
            f"STRESS_CORRELATION_LEVEL must lie in (0, 1) - a literal 1.0 is "
            f"singular - got {STRESS_CORRELATION_LEVEL!r}"
        )
    if STRESS_VOLATILITY_MULTIPLIER <= 1.0:
        raise ValueError(
            f"a volatility shock must raise volatility, got "
            f"{STRESS_VOLATILITY_MULTIPLIER!r}"
        )
    if not 0.0 < STRESS_MATERIAL_DEGRADATION < 1.0:
        raise ValueError(
            f"STRESS_MATERIAL_DEGRADATION must be a share in (0, 1), got "
            f"{STRESS_MATERIAL_DEGRADATION!r}"
        )
    if STRESS_BLOCKS_TRADES:
        raise ValueError(
            "R6 measures what stress does; whether to trade is R7's decision"
        )


_validate_stress_scenario_config()


# --- R7: portfolio-level NO_TRADE ------------------------------------------------
# R1, R3, R4, R5 and R6 each deliberately refused to block a trade and deferred
# to R7. This is where that deferral comes due.
#
# W2 (RISK_POLICY_V2) already vetoes on EVIDENCE quality, and R7 does not
# duplicate it: every W2 rule reads one ticker's inputs - data quality, source
# confidence, point-in-time validity, that ticker's score and regime. None of
# them can see the portfolio. R7 asks the question W2 structurally cannot:
# given everything already held, can the book absorb this?
#
# THE DECIDING MEASUREMENT: a set of individually-correct trades can be
# collectively impossible. On a diversified 4-name book with 6 candidates in
# one correlated cluster, R2 sized each candidate against the ORIGINAL
# portfolio and approved all six at 20% - every one marked diversifying=True,
# every one REDUCING volatility in isolation (-0.28% to -1.45%). The approved
# set sums to 120% OF THE BOOK.
#
#     total executed   portfolio vol   cluster share of RISK
#              24%          -0.67%              27%
#              48%          +7.88%              60%
#              72%         +23.76%              85%
#              96%         +44.57%              99%
#             120%      INFEASIBLE - exceeds 100% of the book
#
# Every trade was individually right and the set is unexecutable. Nothing
# below R7 can see this, because each module sizes ONE candidate against the
# CURRENT book and never against the other candidates.
PORTFOLIO_DECISION_VERSION = "portfolio-decision-v1"

# R7 inherits W2's fail-closed construction: a missing input TRIGGERS the rule
# that reads it rather than passing it. A portfolio check that silently passes
# when its evidence is absent is worse than no check, because it produces a
# confident approval from nothing.
PORTFOLIO_FAIL_CLOSED = True

# The total of all proposed sizes above which the set is refused outright.
# MEASURED, the approved set summed to 120% of the book; anything at or above
# 100% cannot be executed at all. The budget sits below that because a book
# fully consumed by new positions has sold everything it already held, which
# is a different decision than the one being asked.
PORTFOLIO_MAX_TOTAL_SIZE = 0.40

# Portfolio volatility increase, across the whole accepted set, above which
# the set is refused. Set at R2's per-trade budget: a set of trades must not
# do collectively what no single trade was allowed to do.
PORTFOLIO_MAX_VOLATILITY_INCREASE = SIZING_RISK_BUDGET

# Severities. Reused from W2's vocabulary rather than redefined, so a veto
# means the same thing at both levels.
PORTFOLIO_SEVERITY_VETO = "veto"
PORTFOLIO_SEVERITY_WARN = "warn"

# Verdicts. NO_TRADE and NOT_EVALUATED stay distinct for the reason R2 and R4
# keep REFUSED and NOT_EVALUATED apart: "the portfolio refuses this" and "the
# portfolio could not be assessed" are different answers, and collapsing them
# turns an absence of evidence into a decision.
PORTFOLIO_PROCEED = "PROCEED"
PORTFOLIO_REDUCED = "REDUCED"
PORTFOLIO_NO_TRADE = "NO_TRADE"
PORTFOLIO_NOT_EVALUATED = "NOT_EVALUATED"
PORTFOLIO_VERDICTS: tuple[str, ...] = (
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_REDUCED,
    PORTFOLIO_PROCEED,
)

# The portfolio-level rules. Declared as DATA so a reader learns what the book
# is checked against without reading code, and adding one is a deliberate
# edit here. Each names the sprint whose measurement it enforces.
PORTFOLIO_RULES: dict[str, dict] = {
    "total_size_exceeds_budget": {
        "severity": PORTFOLIO_SEVERITY_VETO,
        "source": "R7",
        "detail": (
            "MEASURED, six individually-approved trades summed to 120% of the "
            "book. Each was sized against the ORIGINAL portfolio and never "
            "against the others"
        ),
    },
    "set_volatility_exceeds_budget": {
        "severity": PORTFOLIO_SEVERITY_VETO,
        "source": "R2/R5",
        "detail": (
            "MEASURED, a set whose members each REDUCED volatility in "
            "isolation raised it +44.57% when executed together"
        ),
    },
    "sector_concentration_breach": {
        "severity": PORTFOLIO_SEVERITY_WARN,
        "source": "R3",
        "detail": (
            "MEASURED, 38 IT names at ~1.3% each formed a 52.1% single-sector "
            "bet that cleared every per-name cap"
        ),
    },
    "stress_diversification_collapse": {
        "severity": PORTFOLIO_SEVERITY_WARN,
        "source": "R6",
        "detail": (
            "MEASURED on 76 real tickers, correlation roughly doubles under "
            "the worst decile and diversification falls 38.1%"
        ),
    },
    "no_candidate_survived": {
        "severity": PORTFOLIO_SEVERITY_VETO,
        "source": "R4",
        "detail": (
            "every candidate was already refused upstream; proceeding would "
            "execute nothing while reporting approval"
        ),
    },
}

# THIS is the module that blocks trades, and the only one. Every other Sprint
# R module carries *_BLOCKS_TRADES = False and says so in its report.
PORTFOLIO_BLOCKS_TRADES = True


def _validate_portfolio_decision_config() -> None:
    """Import-time guard for the R7 contract."""
    if not PORTFOLIO_FAIL_CLOSED:
        raise ValueError(
            "R7 must be fail-closed like W2: a portfolio check that passes "
            "when its evidence is absent produces a confident approval from "
            "nothing"
        )
    if not PORTFOLIO_BLOCKS_TRADES:
        raise ValueError(
            "R7 is the module that blocks trades; R1, R3, R4, R5 and R6 each "
            "deferred here precisely so one place owns the refusal"
        )
    if not 0.0 < PORTFOLIO_MAX_TOTAL_SIZE < 1.0:
        raise ValueError(
            f"PORTFOLIO_MAX_TOTAL_SIZE must leave the book intact, got "
            f"{PORTFOLIO_MAX_TOTAL_SIZE!r}"
        )
    if PORTFOLIO_MAX_VOLATILITY_INCREASE != SIZING_RISK_BUDGET:
        raise ValueError(
            "the set-level volatility budget must equal R2's per-trade "
            "budget: a set of trades must not do collectively what no single "
            "trade was allowed to do"
        )
    if PORTFOLIO_NO_TRADE == PORTFOLIO_NOT_EVALUATED:
        raise ValueError(
            "'the portfolio refuses this' and 'the portfolio could not be "
            "assessed' are different answers; collapsing them turns an "
            "absence of evidence into a decision"
        )
    if len(set(PORTFOLIO_VERDICTS)) != len(PORTFOLIO_VERDICTS):
        raise ValueError("duplicate R7 verdict")
    if PORTFOLIO_SEVERITY_VETO != "veto":
        raise ValueError(
            "the veto severity must match W2's vocabulary so a veto means the "
            "same thing at both levels"
        )
    for rule_id, spec in PORTFOLIO_RULES.items():
        if spec.get("severity") not in (
            PORTFOLIO_SEVERITY_VETO,
            PORTFOLIO_SEVERITY_WARN,
        ):
            raise ValueError(f"{rule_id}: unknown severity")
        if not str(spec.get("detail") or "").strip():
            raise ValueError(
                f"{rule_id}: a portfolio rule must carry the measurement it "
                f"enforces, not merely a name"
            )
        if not str(spec.get("source") or "").strip():
            raise ValueError(f"{rule_id}: no sprint named as the rule's source")
    if not any(
        spec["severity"] == PORTFOLIO_SEVERITY_VETO
        for spec in PORTFOLIO_RULES.values()
    ):
        raise ValueError(
            "no rule can veto: a decision module that cannot refuse is not a "
            "decision module"
        )


_validate_portfolio_decision_config()


# --- D1: the combined research view ----------------------------------------------
# The sprint goal is "a research workstation, not a black box". The failure
# mode of a dashboard is not that it shows too little; it is that it renders
# an empty system as a confident one.
#
# THE DECIDING MEASUREMENT, taken on live data before any view was built. A
# forecast snapshot for NVDA at 2026-09-22, at every horizon the roadmap asks
# for, reports:
#
#     horizon   PRESENT   ABSENT   REFUSED
#     1d              5        2         8
#     5d              5        2         8
#     20d             5        2         8
#     60d             5        2         8
#
# The five PRESENT fields are ticker, as_of, forecast_version, horizon and
# warnings - metadata only. Every field a reader would actually act on
# (probability_up, confidence, regime, event_context, evidence) is REFUSED,
# and expected_return is ABSENT because no trained model exists.
#
# So D1's first duty is arithmetic honesty: a view over 5 of 15 fields must
# say so, in the header, before anything else. A dashboard that rendered these
# as blanks would look like a working system with a quiet day.
RESEARCH_VIEW_VERSION = "research-view-v1"

# The horizons the research view shows, exactly as the roadmap names them.
# A SUBSET of FORECAST_HORIZONS, which also carries 120d and 252d; the view
# does not invent horizons the forecast machinery cannot produce, and it does
# not silently drop the ones it was asked for.
RESEARCH_VIEW_HORIZONS: tuple[str, ...] = ("1d", "5d", "20d", "60d")

# The panels the view shows together. "Together" is the requirement: a score
# read without its risk status, or a forecast read without its confidence, is
# the black box the sprint exists to replace.
RESEARCH_PANEL_QUALITY = "investment_quality"
RESEARCH_PANEL_FORECAST = "multi_horizon_forecast"
RESEARCH_PANEL_CONFIDENCE = "confidence"
RESEARCH_PANEL_REGIME = "regime"
RESEARCH_PANEL_RISK = "risk_status"
RESEARCH_PANELS: tuple[str, ...] = (
    RESEARCH_PANEL_QUALITY,
    RESEARCH_PANEL_FORECAST,
    RESEARCH_PANEL_CONFIDENCE,
    RESEARCH_PANEL_REGIME,
    RESEARCH_PANEL_RISK,
)

# Cell states. Deliberately the snapshot's own vocabulary rather than a second
# one: PRESENT/ABSENT/REFUSED already distinguish "measured", "no producer
# exists" and "a producer declined here", and a view that collapsed them would
# undo the distinction F3-F7 were built to preserve.
VIEW_CELL_PRESENT = "PRESENT"
VIEW_CELL_ABSENT = "ABSENT"
VIEW_CELL_REFUSED = "REFUSED"
VIEW_CELL_STATES: tuple[str, ...] = (
    VIEW_CELL_PRESENT,
    VIEW_CELL_ABSENT,
    VIEW_CELL_REFUSED,
)

# THE RETURN COLUMN IS SHOWN, AND IT IS ALWAYS ABSENT. The roadmap asks for
# "return% and probability" per horizon. MEASURED, expected_return has no
# honest producer: build_joint_forecast reports NO_MODEL, and
# SNAPSHOT_UNAVAILABLE_FIELDS records that a 0.0 "would render as 'flat' in
# any consumer that coalesces nulls". Omitting the column would hide that the
# field was requested and refused; filling it would fabricate the quantity.
# So the column exists, every cell reads ABSENT, and the reason travels with
# it.
RESEARCH_SHOW_RETURN_COLUMN = True
RESEARCH_RETURN_IS_ABSENT = True

# A view whose fields are mostly unavailable must say so before its contents.
# This is the share of PRESENT fields below which the header leads with the
# gap rather than the data. MEASURED, the live snapshot sits at 5/15 = 0.333,
# far below it, so today every real view leads with its own emptiness.
RESEARCH_COVERAGE_WARN = 0.60

# A research view renders; it decides nothing. The refusal is R7's and the
# veto is W2's, and the view reports both rather than forming a third opinion.
RESEARCH_VIEW_BLOCKS_TRADES = False


def _validate_research_view_config() -> None:
    """Import-time guard for the D1 contract."""
    if not RESEARCH_VIEW_HORIZONS:
        raise ValueError("the research view must show at least one horizon")
    for horizon in RESEARCH_VIEW_HORIZONS:
        if horizon not in FORECAST_HORIZONS:
            raise ValueError(
                f"{horizon!r} is not a forecast horizon; the view must not "
                f"invent horizons the forecast machinery cannot produce"
            )
    if len(set(RESEARCH_VIEW_HORIZONS)) != len(RESEARCH_VIEW_HORIZONS):
        raise ValueError("duplicate research horizon")
    if len(set(RESEARCH_PANELS)) != len(RESEARCH_PANELS):
        raise ValueError("duplicate research panel")
    for required in (
        RESEARCH_PANEL_QUALITY,
        RESEARCH_PANEL_FORECAST,
        RESEARCH_PANEL_CONFIDENCE,
        RESEARCH_PANEL_REGIME,
        RESEARCH_PANEL_RISK,
    ):
        if required not in RESEARCH_PANELS:
            raise ValueError(
                f"{required!r} is missing: the sprint requires score, "
                f"forecast, confidence, regime and risk status shown TOGETHER"
            )
    if set(VIEW_CELL_STATES) != set(SNAPSHOT_STATUSES):
        raise ValueError(
            "the view must reuse the snapshot's PRESENT/ABSENT/REFUSED "
            "vocabulary; a second set of states would collapse the "
            "distinction between 'no producer' and 'a producer declined'"
        )
    if not RESEARCH_SHOW_RETURN_COLUMN:
        raise ValueError(
            "the return column is shown so its refusal is visible; omitting "
            "it hides that the field was requested and could not be produced"
        )
    if not RESEARCH_RETURN_IS_ABSENT:
        raise ValueError(
            "expected_return has no producer: MEASURED, build_joint_forecast "
            "reports NO_MODEL and a 0.0 would render as 'flat'"
        )
    if not 0.0 < RESEARCH_COVERAGE_WARN <= 1.0:
        raise ValueError(
            f"RESEARCH_COVERAGE_WARN must be a share in (0, 1], got "
            f"{RESEARCH_COVERAGE_WARN!r}"
        )
    if RESEARCH_VIEW_BLOCKS_TRADES:
        raise ValueError(
            "a research view renders; the refusal is R7's and the veto is "
            "W2's, and the view reports both rather than forming a third"
        )


_validate_research_view_config()


# --- D2: the WHY breakdown -------------------------------------------------------
# The roadmap asks for "Technical / Fundamental / News / Macro / Regime /
# Sentiment contributions". Two measured facts stand between that wording and
# an honest panel, and D2 is built around both.
#
# FIRST: THEY ARE NOT CONTRIBUTIONS. F6 MEASURED, over 4,000 observations with
# a realistic regime/chart correlation:
#
#     regime alone   +0.064
#     chart alone    +0.067
#     sum            +0.131
#     ACTUAL joint   +0.060
#
# The parts overlap and double-count by more than 2x. A panel showing six
# numbers that sum to the forecast would be arithmetically wrong AND would
# imply each factor independently CAUSED its share. DECOMPOSITION_ADDITIVE is
# False and F6's validator refuses to let it become True; D2 inherits that
# refusal rather than re-deciding it.
#
# SECOND: HALF OF WHAT THE ROADMAP NAMES CANNOT BE MEASURED. Of F6's seven
# components only four are wired, and the three that are NOT_WIRED -
# fundamental, macro, sentiment - are exactly three of the six the roadmap
# asks for. MEASURED on a live decomposition: 1 PRESENT, 3 ABSENT, 3
# NOT_WIRED. Rendering those three as 0.0 would say they were measured and
# found irrelevant; they were never measured, which is a different fact.
WHY_PANEL_VERSION = "why-panel-v1"

# The roadmap's six names, mapped onto F6's components. Declared as DATA so
# the correspondence is auditable and so a reader sees that "News" and
# "news_event" are the same thing rather than two systems.
WHY_ROADMAP_NAMES: dict[str, str] = {
    DECOMP_TECHNICAL: "Technical",
    DECOMP_FUNDAMENTAL: "Fundamental",
    DECOMP_NEWS_EVENT: "News",
    DECOMP_MACRO: "Macro",
    DECOMP_REGIME: "Regime",
    DECOMP_SENTIMENT: "Sentiment",
    # F6 carries a seventh the roadmap does not name. It is SHOWN anyway:
    # MEASURED, historical_analog is the component that actually produced the
    # forecast value, so omitting it would hide the only wired evidence that
    # spoke.
    DECOMP_HISTORICAL_ANALOG: "Historical analog",
}

# The panel shows every F6 component, not only the six the roadmap names. A
# panel that dropped historical_analog would answer "why?" without the part
# that did the work.
WHY_SHOW_ALL_COMPONENTS = True

# D2 NEVER PRESENTS A CONTRIBUTION NUMBER. It reports, per component, whether
# it supplied evidence and how much it NARROWED the claim - F6's own
# vocabulary. The word "contribution" is barred from the panel's own fields
# for the reason F6 bars it from its vocabulary: it asserts additivity and
# causation that were measured to be false.
WHY_REPORTS_CONTRIBUTIONS = False
WHY_BARRED_TERMS: tuple[str, ...] = ("contribution", "contributions", "contributed")

# The panel's own statuses are F6's, reused rather than redefined, so
# NOT_WIRED cannot quietly become ABSENT on the way to the screen. They mean
# different things: ABSENT is "this component could have spoken and did not",
# NOT_WIRED is "this component cannot speak at all yet".
WHY_STATUSES: tuple[str, ...] = DECOMPOSITION_STATUSES

# A WHY panel in which nothing was measured must say so before its rows.
# MEASURED, a live decomposition has exactly ONE component PRESENT out of
# seven, so today that headline fires on every real panel.
WHY_MIN_PRESENT = 2

# D2 explains; it decides nothing, and it recomputes nothing. Every value is
# F6's, so the panel cannot disagree with the decomposition it displays.
WHY_PANEL_BLOCKS_TRADES = False


def _validate_why_panel_config() -> None:
    """Import-time guard for the D2 contract."""
    if WHY_REPORTS_CONTRIBUTIONS:
        raise ValueError(
            "D2 must not report contributions: MEASURED, regime +0.064 and "
            "chart +0.067 sum to +0.131 against an ACTUAL joint effect of "
            "+0.060 - the parts overlap and double-count by more than 2x"
        )
    if DECOMPOSITION_ADDITIVE:
        raise ValueError(
            "the WHY panel rests on F6's non-additivity; if the "
            "decomposition ever became additive this panel must be "
            "re-derived rather than inherited"
        )
    if set(WHY_ROADMAP_NAMES) != set(DECOMPOSITION_COMPONENTS):
        missing = set(DECOMPOSITION_COMPONENTS) - set(WHY_ROADMAP_NAMES)
        extra = set(WHY_ROADMAP_NAMES) - set(DECOMPOSITION_COMPONENTS)
        raise ValueError(
            f"the WHY panel must name every F6 component exactly once; "
            f"missing {sorted(missing)}, unknown {sorted(extra)}"
        )
    if len(set(WHY_ROADMAP_NAMES.values())) != len(WHY_ROADMAP_NAMES):
        raise ValueError("two components share a display name")
    for component, label in WHY_ROADMAP_NAMES.items():
        if not str(label or "").strip():
            raise ValueError(f"{component}: no display name")
    if not WHY_SHOW_ALL_COMPONENTS:
        raise ValueError(
            "the panel shows every component: MEASURED, historical_analog is "
            "the one that produced the forecast value, and the roadmap does "
            "not name it"
        )
    if tuple(WHY_STATUSES) != tuple(DECOMPOSITION_STATUSES):
        raise ValueError(
            "the panel must reuse F6's statuses; a second vocabulary lets "
            "NOT_WIRED become ABSENT on the way to the screen"
        )
    if DECOMP_STATUS_NOT_WIRED not in WHY_STATUSES:
        raise ValueError(
            "NOT_WIRED must survive to the panel: reporting an unwired "
            "component as 0.0 claims it was measured and found irrelevant"
        )
    if not WHY_BARRED_TERMS:
        raise ValueError("the barred-term list cannot be empty")
    if WHY_MIN_PRESENT < 1:
        raise ValueError("WHY_MIN_PRESENT must be at least 1")
    if WHY_PANEL_BLOCKS_TRADES:
        raise ValueError("D2 explains; it decides nothing")


_validate_why_panel_config()


# --- D3: the event context panel -------------------------------------------------
# The roadmap asks for "recent event, classification, historical analog count,
# median historical response, current setup similarity". Every one of those has
# a producer - E4/E6 supply them - so D3 is mostly assembly. What it must NOT
# do is quote the median response as though it were a clean number.
#
# THE DECIDING MEASUREMENT: a third of comparable setups are not comparable for
# the reason the panel implies. E5 classifies each remembered response as
# event_associated or confounded, and MEASURED across 40 seeded analog sets,
# excluding the confounded ones shifts the median historical response by
# +1.36pp (median shift), up to 5.71pp, and by more than one percentage point
# in 24 of 40 cases.
#
# So "median historical response: +2.7%" is a composite of the event's own
# association AND whatever else moved those names. The panel reports the
# median WITH the event-associated share beside it, always, and never one
# without the other.
EVENT_CONTEXT_VERSION = "event-context-v1"

# The five fields the roadmap names, declared as data so a reader sees the
# correspondence without reading code.
EVENT_CONTEXT_FIELDS: tuple[str, ...] = (
    "recent_event",
    "classification",
    "analog_count",
    "median_response",
    "setup_similarity",
)

# THE MEDIAN NEVER TRAVELS ALONE. MEASURED, dropping confounded analogs moves
# it by more than a percentage point in 24 of 40 cases, so a median quoted
# without its association share overstates what the event itself explains.
EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE = True

# Below this share of event-associated analogs, the median is reported with an
# explicit caution. MEASURED, the reference set sits at 0.667 - a third of the
# evidence is attributed elsewhere - which is exactly the case that needs
# saying out loud.
EVENT_CONTEXT_ASSOCIATION_CAUTION = 0.75

# Analog statuses, reused from E6 rather than redefined. "insufficient_analogs"
# is not "no effect": MEASURED, E6 refuses to summarise below
# EVENT_MEMORY_MIN_ANALOGS because a typical response from that few examples is
# not typical of anything.
EVENT_CONTEXT_STATUS_MEASURED = "measured"
EVENT_CONTEXT_STATUS_INSUFFICIENT = "insufficient_analogs"
EVENT_CONTEXT_STATUS_NO_EVENT = "no_event"
EVENT_CONTEXT_STATUSES: tuple[str, ...] = (
    EVENT_CONTEXT_STATUS_MEASURED,
    EVENT_CONTEXT_STATUS_INSUFFICIENT,
    EVENT_CONTEXT_STATUS_NO_EVENT,
)

# An INFERRED event is real evidence about a real price move; what is uncertain
# is WHICH event produced it. The panel states provenance per event rather than
# letting an inferred event read like an observed one - E6 defaults provenance
# to "" precisely so an unlabelled memory cannot launder itself into evidence.
EVENT_CONTEXT_SHOW_PROVENANCE = True

# Association is not causation, and the panel says so in its own payload
# rather than relying on a reader to remember it.
EVENT_CONTEXT_DISCLAIMER = (
    "historical association under comparable conditions - not a forecast, and "
    "not evidence that these events caused these moves. MEASURED, excluding "
    "E5-confounded analogs shifts the median response by more than a "
    "percentage point in 24 of 40 cases"
)

# D3 describes what happened before; it decides nothing.
EVENT_CONTEXT_BLOCKS_TRADES = False


def _validate_event_context_config() -> None:
    """Import-time guard for the D3 contract."""
    if len(set(EVENT_CONTEXT_FIELDS)) != len(EVENT_CONTEXT_FIELDS):
        raise ValueError("duplicate event context field")
    for required in ("analog_count", "median_response", "setup_similarity"):
        if required not in EVENT_CONTEXT_FIELDS:
            raise ValueError(f"the roadmap requires {required!r}")
    if not EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE:
        raise ValueError(
            "the median response must travel with its event-associated "
            "share: MEASURED, excluding confounded analogs shifts it by more "
            "than a percentage point in 24 of 40 cases"
        )
    if not 0.0 < EVENT_CONTEXT_ASSOCIATION_CAUTION <= 1.0:
        raise ValueError(
            f"EVENT_CONTEXT_ASSOCIATION_CAUTION must be a share in (0, 1], "
            f"got {EVENT_CONTEXT_ASSOCIATION_CAUTION!r}"
        )
    if len(set(EVENT_CONTEXT_STATUSES)) != len(EVENT_CONTEXT_STATUSES):
        raise ValueError("duplicate event context status")
    if EVENT_CONTEXT_STATUS_INSUFFICIENT == EVENT_CONTEXT_STATUS_NO_EVENT:
        raise ValueError(
            "'too few analogs to summarise' and 'no event to analyse' are "
            "different answers; collapsing them hides which one applies"
        )
    if not EVENT_CONTEXT_SHOW_PROVENANCE:
        raise ValueError(
            "provenance is shown per event: an INFERRED event is evidence "
            "about a real move, but WHICH event produced it is uncertain"
        )
    if "not a forecast" not in EVENT_CONTEXT_DISCLAIMER:
        raise ValueError(
            "the disclaimer must state that this is association, not a "
            "forecast and not causation"
        )
    if EVENT_CONTEXT_BLOCKS_TRADES:
        raise ValueError("D3 describes what happened before; it decides nothing")


_validate_event_context_config()


# --- D4: the provenance surface --------------------------------------------------
# "The UI must expose provenance, not hide it." Exposing it requires that it
# exist per displayed value, and MEASURED, it does not.
#
# THE DECIDING MEASUREMENT: of the 5 PRESENT fields in a live forecast
# snapshot, ZERO carry any identification of where the value came from. The
# snapshot as a whole is digest-addressable (snapshot_digest) and its versions
# are pinned (forecast_versions), but no individual value says which source,
# which payload or which calculation produced it. A reader looking at
# "P(up) 0.56" has no way to reach the bytes behind it.
#
# The substrate exists: MEASURED, the raw ledger holds 6,641 payloads across
# 2,758 distinct SHA-256 digests, each carrying source_id, request_key,
# ingested_time and schema_version. What was missing is the LINK from a
# displayed value to one of them.
PROVENANCE_SURFACE_VERSION = "provenance-surface-v1"

# The four things a provenance record must answer. Fewer than four and the
# value cannot be re-derived: WHICH source, WHICH payload, WHEN it arrived,
# and under WHICH calculation version it was interpreted.
PROV_FIELD_SOURCE = "source_id"
PROV_FIELD_PAYLOAD = "payload_sha256"
PROV_FIELD_INGESTED = "ingested_time"
PROV_FIELD_VERSION = "calculation_version"
PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    PROV_FIELD_SOURCE,
    PROV_FIELD_PAYLOAD,
    PROV_FIELD_INGESTED,
    PROV_FIELD_VERSION,
)

# Origin kinds. A value that was OBSERVED from a source and one DERIVED by
# calculation are different claims, and a reader must be able to tell which
# is which without inspecting the pipeline. COMPOSED marks a value assembled
# from other traced values - it is traceable only as far as its parts.
PROV_ORIGIN_OBSERVED = "OBSERVED"
PROV_ORIGIN_DERIVED = "DERIVED"
PROV_ORIGIN_COMPOSED = "COMPOSED"
PROV_ORIGIN_UNTRACED = "UNTRACED"
PROVENANCE_ORIGINS: tuple[str, ...] = (
    PROV_ORIGIN_OBSERVED,
    PROV_ORIGIN_DERIVED,
    PROV_ORIGIN_COMPOSED,
    PROV_ORIGIN_UNTRACED,
)

# UNTRACED IS REPORTED, NEVER HIDDEN. This is the whole point of the sprint
# bullet. A value with no provenance is shown AS untraced rather than shown
# without the question being asked - MEASURED, that is currently every
# PRESENT value in the snapshot, and a surface that quietly omitted them
# would report perfect provenance over an empty set.
PROVENANCE_REPORT_UNTRACED = True

# The share of displayed values that must be traceable before the surface
# stops leading with its own incompleteness. MEASURED, the live snapshot sits
# at 0.00, so today every real surface leads with the gap.
PROVENANCE_COVERAGE_WARN = 0.80

# A refusal carries provenance too: WHY a value is absent is itself evidence,
# and a reader who cannot see why cannot judge whether to wait for it.
PROVENANCE_TRACE_REFUSALS = True

# D4 exposes; it decides nothing and it recomputes nothing.
PROVENANCE_BLOCKS_TRADES = False


def _validate_provenance_surface_config() -> None:
    """Import-time guard for the D4 contract."""
    if len(set(PROVENANCE_REQUIRED_FIELDS)) != len(PROVENANCE_REQUIRED_FIELDS):
        raise ValueError("duplicate provenance field")
    for required in (PROV_FIELD_SOURCE, PROV_FIELD_PAYLOAD):
        if required not in PROVENANCE_REQUIRED_FIELDS:
            raise ValueError(
                f"{required!r} is required: without it a displayed value "
                f"cannot be traced to the bytes behind it"
            )
    if len(set(PROVENANCE_ORIGINS)) != len(PROVENANCE_ORIGINS):
        raise ValueError("duplicate provenance origin")
    if PROV_ORIGIN_OBSERVED == PROV_ORIGIN_DERIVED:
        raise ValueError(
            "'read from a source' and 'computed from other values' are "
            "different claims and must not share a label"
        )
    if PROV_ORIGIN_UNTRACED not in PROVENANCE_ORIGINS:
        raise ValueError(
            "UNTRACED must be expressible: MEASURED, 0 of 5 PRESENT snapshot "
            "fields carry provenance today, and a surface that cannot say so "
            "would report perfect provenance over an empty set"
        )
    if not PROVENANCE_REPORT_UNTRACED:
        raise ValueError(
            "the sprint requires provenance be EXPOSED, not hidden; an "
            "untraced value is the case that most needs exposing"
        )
    if not PROVENANCE_TRACE_REFUSALS:
        raise ValueError(
            "a refusal carries provenance too: why a value is absent is "
            "itself evidence a reader needs"
        )
    if not 0.0 < PROVENANCE_COVERAGE_WARN <= 1.0:
        raise ValueError(
            f"PROVENANCE_COVERAGE_WARN must be a share in (0, 1], got "
            f"{PROVENANCE_COVERAGE_WARN!r}"
        )
    if PROVENANCE_BLOCKS_TRADES:
        raise ValueError("D4 exposes provenance; it decides nothing")


_validate_provenance_surface_config()


