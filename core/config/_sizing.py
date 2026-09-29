"""Configuration part: R1-R2 position exposure and correlation-aware sizing.

Split out of the single 9,783-line `core/config.py` in C1, and then again
because the forecasting part alone was 3,631 lines — leaving it would have
moved the navigability problem rather than solved it. The text is UNCHANGED.

CHAINED from `_learning`: 88 constants are read across part boundaries, so
the parts reproduce ONE flat namespace in the original order.
"""

from core.config._learning import *  # noqa: F401,F403


# Sprint R1 - Position exposure
# ---------------------------------------------------------------------------
# Sprint R moves from "is this ticker attractive?" to "does acting on this
# forecast improve the CURRENT portfolio without violating risk constraints?"
# R1 is the foundation: knowing what is actually held, and what holding it
# EXPOSES the portfolio to.
#
# THERE WAS NO PORTFOLIO STATE AT ALL. `fetch_data.PORTFOLIO_TICKERS` is a
# 77-name WATCHLIST - a list of symbols with no share counts, no cost basis
# and no weights. The backtest engine tracks shares internally but nothing
# persists live holdings. So the question Sprint R asks could not previously
# be asked: there was nothing to improve.
#
# THE DECIDING MEASUREMENT: POSITION WEIGHT IS NOT EXPOSURE. Two portfolios
# built on REAL daily returns (498 aligned sessions to 2026-09-18):
#
#   A: 40% NVDA, 20% MSFT, 20% GOOGL, 20% CAT          max weight 40%
#   B: 10% each NVDA/AMD/AVGO/SOXX, then 20/20/20      max weight 10%
#
#   portfolio      daily vol   max weight   share of variance in semis
#   A (40% NVDA)      1.741%          40%                       58.6%
#   B (4x10% semis)   1.760%          10%                       56.9%
#
# B looks FOUR TIMES more diversified by weight and is very slightly MORE
# volatile, carrying the same bet. The watchlist's real mean pairwise
# correlation is 0.484, with SOXX/AMD at 0.78.
#
# WORSE, A WEIGHT CAP IS GAMEABLE IN THE WRONG DIRECTION. "Comply with a 10%
# cap" was satisfied by splitting 40% NVDA across four correlated semis, which
# RAISED volatility 1.741% -> 1.760%. A rule that can be satisfied by making
# risk worse is not merely incomplete.
#
# AND NEITHER NAIVE MEASURE ORDERS PORTFOLIOS CORRECTLY:
#
#   portfolio         vol   max weight   effective bets   max risk share
#   all VOO        1.006%         100%             1.00           100.0%
#   diversified    1.449%          30%             3.66            32.3%
#   40% NVDA       1.741%          40%             2.49            58.6%
#   4x10% semis    1.760%          10%             6.80            17.9%
#
# 100% VOO has the LOWEST volatility of the four while every concentration
# measure ranks it worst. A single diversified fund is not a concentrated
# position, and a weight-based rule cannot tell the difference. So R1 reports
# RISK CONTRIBUTION alongside weight, and never weight alone.
#
# MARGINAL EXPOSURE IS THE QUESTION SPRINT R ASKS. Against a held portfolio of
# 30% NVDA / 20% AMD / 25% MSFT / 25% CAT (vol 1.984%), the same 5% purchase:
#
#   add 5% SOXX   -> 2.000%  (+0.016%)  ADDS risk
#   add 5% MSFT   -> 1.936%  (-0.048%)  REDUCES risk
#   add 5% VOO    -> 1.927%  (-0.057%)  REDUCES risk
#
# The same trade helps or hurts depending ENTIRELY on what is already held,
# and no per-ticker forecast can answer that.
POSITION_EXPOSURE_VERSION = "position-exposure-v1"

# A held position needs these to be a position rather than a mention. Without
# a quantity there is no exposure to compute; without an as_of the holding is
# not point-in-time and could silently describe a different day.
POSITION_REQUIRED_FIELDS: tuple[str, ...] = ("ticker", "quantity", "as_of")

# Sessions of aligned return history required before a covariance-based
# exposure is reported. Below this the estimate is dominated by sampling
# error and would give a confident-looking number for a relationship that has
# not been observed.
EXPOSURE_MIN_SESSIONS = 120

# Returns MUST be aligned by DATE, never by position. MEASURED, slicing the
# last N rows of each series instead put MSFT (501 rows ending 2026-09-18)
# out of step with NVDA and VOO (500 rows ending 2026-09-21) and reported
# MSFT's correlation with EVERYTHING as ~0.00, including with VOO - which the
# date-aligned data puts at 0.53. A position-offset join does not fail loudly;
# it silently reports independence.
EXPOSURE_ALIGN_BY_DATE = True

# Exposure is reported on both scales, and a weight-only report is refused.
EXPOSURE_BASIS_WEIGHT = "weight"              # share of portfolio value
EXPOSURE_BASIS_RISK = "risk_contribution"     # share of portfolio variance

EXPOSURE_BASES: tuple[str, ...] = (EXPOSURE_BASIS_WEIGHT, EXPOSURE_BASIS_RISK)

# A single position's share of portfolio VALUE above which it is flagged for
# review. This is a reporting threshold, not a limit: MEASURED, 100% VOO
# breaches it while being the least volatile portfolio tested.
EXPOSURE_WEIGHT_REVIEW = 0.25

# A single position's share of portfolio VARIANCE above which it is flagged.
# MEASURED, the 40%-NVDA portfolio puts 58.6% of its variance in one name
# while the 4x10% semi basket puts 56.9% in the same bet at a tenth of the
# per-name weight.
EXPOSURE_RISK_REVIEW = 0.40

# Exposure is DESCRIBED, never enforced here. R1 reports what is held and what
# it exposes the portfolio to; whether a trade is permitted is a later task
# with its own evidence. A measurement that silently blocked trades would be a
# policy wearing a measurement's clothes.
EXPOSURE_BLOCKS_TRADES = False


def _validate_position_exposure_config() -> None:
    """Import-time guard for the R1 contract."""
    if "quantity" not in POSITION_REQUIRED_FIELDS:
        raise ValueError(
            "a position without a quantity is a watchlist entry, not a "
            "holding - PORTFOLIO_TICKERS was exactly that, and it is why the "
            "Sprint R question could not previously be asked"
        )
    if "as_of" not in POSITION_REQUIRED_FIELDS:
        raise ValueError(
            "a holding must carry its as_of or it is not point-in-time and "
            "may silently describe a different day"
        )
    if EXPOSURE_MIN_SESSIONS < 120:
        raise ValueError(
            f"{EXPOSURE_MIN_SESSIONS} sessions cannot support a covariance "
            f"estimate across a portfolio; the result would be a "
            f"confident-looking number for a relationship never observed"
        )
    if not EXPOSURE_ALIGN_BY_DATE:
        raise ValueError(
            "returns must be aligned by DATE. MEASURED, a position-offset "
            "join reported MSFT's correlation with every other holding as "
            "~0.00 - including 0.00 against VOO, which is really 0.53 - "
            "because MSFT's series ended three days earlier. It does not fail "
            "loudly; it silently reports independence"
        )
    if EXPOSURE_BASIS_RISK not in EXPOSURE_BASES:
        raise ValueError(
            "risk contribution must be reported. MEASURED, 100% VOO is the "
            "LEAST volatile portfolio tested (1.006%) while carrying the "
            "worst score on every weight-based concentration measure"
        )
    if EXPOSURE_BASIS_WEIGHT not in EXPOSURE_BASES:
        raise ValueError("weight must still be reported alongside risk")
    if not 0.0 < EXPOSURE_WEIGHT_REVIEW < 1.0:
        raise ValueError("the weight review threshold must lie inside (0, 1)")
    if not 0.0 < EXPOSURE_RISK_REVIEW < 1.0:
        raise ValueError("the risk review threshold must lie inside (0, 1)")
    if EXPOSURE_BLOCKS_TRADES:
        raise ValueError(
            "R1 DESCRIBES exposure, it does not enforce limits. A weight cap "
            "enforced alone is gameable in the wrong direction: MEASURED, "
            "satisfying a 10% cap by splitting 40% NVDA across four "
            "correlated semiconductors RAISED portfolio volatility "
            "1.741% -> 1.760%"
        )


_validate_position_exposure_config()


# ---------------------------------------------------------------------------
# Sprint R2 - Correlation-aware sizing
# ---------------------------------------------------------------------------
# R1 established that weight is not exposure. R2 asks the next question: HOW
# BIG should a position be, given what is already held?
#
# EQUAL-WEIGHT SIZING IGNORES CORRELATION, and the cost is large. MEASURED on
# 498 date-aligned sessions, a 40% budget spent four ways beside 60% VOO:
#
#   4 correlated semis (NVDA/AMD/AVGO/SOXX)   vol 1.587%
#   4 diverse names (NVDA/MSFT/JPM/XOM)       vol 1.054%
#
# The SAME total weight carries 50.5% more risk.
#
# A NEGATIVE RESULT THAT SHAPES THE WHOLE TASK: REWEIGHTING INSIDE A
# CORRELATED BASKET BARELY HELPS. Three sizing rules on the same four semis:
#
#   rule                   NVDA    AMD   AVGO   SOXX       vol
#   equal weight          0.100  0.100  0.100  0.100    1.587%
#   inverse volatility    0.111  0.078  0.094  0.116    1.570%
#   equal risk contrib    0.114  0.081  0.098  0.106    1.572%
#
# All three land within 1% of each other. The 50.5% excess came from the
# basket's COMPOSITION, not its internal weights:
#
#   allocation of the 40% budget          vol      vs equal-weight semis
#   4 semis, equal weight              1.587%                      +0.0%
#   4 semis, inverse-vol weighted      1.568%                      -1.2%
#   3 semis + 1 diversifier            1.429%                     -10.0%
#   2 semis + 2 diversifiers           1.184%                     -25.4%
#   1 semi  + 3 diversifiers           1.054%                     -33.6%
#
# Reweighting buys ~1%; changing what is held buys 10-34%. SIZING CANNOT FIX
# SELECTION, and a sizing module that implied otherwise would be selling a
# false remedy.
#
# SO CORRELATION-AWARE SIZING SIZES AGAINST THE WHOLE PORTFOLIO. MEASURED
# against a held portfolio of 25% NVDA / 15% AMD / 30% MSFT / 30% VOO
# (vol 1.690%), adding a FIXED 10%:
#
#   add 10% AVGO  -> +3.07% portfolio vol
#   add 10% SOXX  -> +2.75%
#   add 10% VOO   -> -4.89%
#   add 10% XOM   -> -9.24%
#
# The same nominal size means something different for every ticker. Sizing to
# a RISK BUDGET - the largest position keeping the volatility increase inside
# a bound - makes "position size" a comparable unit for the first time.
CORRELATION_SIZING_VERSION = "correlation-sizing-v1"

# The default risk budget: how much a single new position may raise portfolio
# volatility, as a RELATIVE increase. Relative, because an absolute bound
# means something different for a 1% portfolio and a 3% one.
SIZING_RISK_BUDGET = 0.05

# The largest position the sizer will ever propose, whatever the risk budget
# allows. MEASURED, an uncorrelated name (XOM) could take 50%+ of the
# portfolio inside a 5% volatility budget because it REDUCES volatility - the
# risk budget alone does not bound concentration, and R1 measured that weight
# and risk are different scales. This is the weight scale's bound.
SIZING_MAX_WEIGHT = 0.20

# The smallest position worth proposing. Below this the trade is dominated by
# costs and the portfolio effect is indistinguishable from noise.
SIZING_MIN_WEIGHT = 0.005

# Search bounds for the size solver, and its tolerance. The solver bisects on
# volatility, which is monotone in position size ONLY over the range where the
# position is diversifying or neutral; the cap above keeps the answer inside
# the region where that holds.
SIZING_SEARCH_TOLERANCE = 1e-4

# Sessions of aligned history required before a correlation-aware size is
# proposed. Inherited from R1: below this the covariance is sampling error
# wearing a number's clothes.
SIZING_MIN_SESSIONS = EXPOSURE_MIN_SESSIONS

# Verdicts, weakest to strongest. The order IS the precedence.
SIZING_NOT_EVALUATED = "NOT_EVALUATED"   # no covariance could be built
SIZING_REFUSED = "REFUSED"               # even the minimum breaches the budget
SIZING_CAPPED = "CAPPED"                 # the weight cap bound, not the risk
SIZING_SIZED = "SIZED"                   # the risk budget bound

SIZING_VERDICTS: tuple[str, ...] = (
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
    SIZING_CAPPED,
    SIZING_SIZED,
)

# A size is a PROPOSAL, never an order. R2 answers "how big could this be
# without breaching the risk budget", which is not the same as "buy this".
SIZING_IS_ADVISORY = True

# The sizer must report what BOUND it. A size that does not say whether the
# risk budget or the weight cap was the binding constraint cannot be acted on
# intelligently - one says "the portfolio cannot absorb more of this", the
# other says "policy stops here".
SIZING_REPORT_BINDING_CONSTRAINT = True


def _validate_correlation_sizing_config() -> None:
    """Import-time guard for the R2 contract."""
    if not 0.0 < SIZING_RISK_BUDGET < 1.0:
        raise ValueError("the risk budget is a relative increase inside (0, 1)")
    if SIZING_RISK_BUDGET > 0.25:
        raise ValueError(
            f"a risk budget of {SIZING_RISK_BUDGET:.0%} lets one position move "
            f"portfolio volatility by a quarter; MEASURED, a fixed 10% in the "
            f"most correlated available name moved it 3.07%"
        )
    if not 0.0 < SIZING_MAX_WEIGHT <= 0.50:
        raise ValueError("the weight cap must lie inside (0, 0.5]")
    if not 0.0 < SIZING_MIN_WEIGHT < SIZING_MAX_WEIGHT:
        raise ValueError("the minimum size must be positive and below the cap")
    if SIZING_MIN_SESSIONS < 120:
        raise ValueError(
            "a correlation-aware size needs the same history a covariance "
            "needs; below 120 sessions the estimate is sampling error"
        )
    if SIZING_VERDICTS[0] != SIZING_NOT_EVALUATED:
        raise ValueError("NOT_EVALUATED must be the weakest verdict")
    if SIZING_VERDICTS[-1] != SIZING_SIZED:
        raise ValueError("SIZED must be the strongest verdict")
    if SIZING_REFUSED == SIZING_NOT_EVALUATED:
        raise ValueError(
            "'no size fits the budget' and 'no size could be computed' are "
            "different facts and must stay distinct"
        )
    if not SIZING_IS_ADVISORY:
        raise ValueError(
            "a size is a proposal, never an order: R2 answers how big a "
            "position COULD be inside the risk budget, which is not the same "
            "as a decision to buy it"
        )
    if not SIZING_REPORT_BINDING_CONSTRAINT:
        raise ValueError(
            "the sizer must say what bound it. 'The portfolio cannot absorb "
            "more of this' and 'policy stops here' are different answers, and "
            "MEASURED an uncorrelated name reached 50%+ on the risk budget "
            "alone while a correlated one stopped at 14%"
        )


_validate_correlation_sizing_config()


