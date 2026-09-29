"""Configuration part 8: X1-X10 the release gate and Sprint B.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_alerts` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._alerts import *  # noqa: F401,F403


# --- X1: out-of-sample forecast validation ---------------------------------------
# "Demonstrate performance on unseen data." MEASURED against the 8 training runs
# on disk, the honest answer is that NO ESTIMATOR DEMONSTRATES ANYTHING.
#
# THE DECIDING MEASUREMENT. Every run is a single fold of 120 validation
# observations at the 20d horizon, trained to 2025-04-16 and validated from
# 2025-07-21 - a 96-day embargo, so PIT ordering is clean and the numbers below
# are not leakage artefacts. They are simply negative.
#
#     estimator            rmse      95% bootstrap CI     dir_acc
#     historical_mean   0.12164   [0.09741, 0.14757]      0.5500   <- BASELINE
#     random_forest     0.13192   [0.10390, 0.16231]      0.4583
#     gradient_boosting 0.14657   [0.11887, 0.17134]      0.4750
#     ridge             0.15769   [0.13414, 0.18321]      0.5000
#     momentum          0.17255   [0.15174, 0.19451]      0.5750
#     elastic_net       0.17850   [0.15316, 0.20415]      0.5333
#     mean_reversion    0.19152   [0.16585, 0.22028]      0.4250
#     logistic          0.19178   [0.16519, 0.22092]      0.4750
#
# ZERO OF SEVEN LEARNED ESTIMATORS BEAT THE NO-FEATURE BASELINE. historical_mean
# carries no features at all and posts the best RMSE. Four estimators are
# SIGNIFICANTLY WORSE - their bootstrap intervals exclude the baseline entirely.
# The three that overlap are merely indistinguishable from predicting the mean.
#
# DIRECTIONAL ACCURACY IS ENTIRELY NOISE. At n=120 the 95% sampling band around
# a coin flip is 0.5 +/- 0.0895, i.e. [0.4105, 0.5895]. ALL EIGHT observed
# accuracies fall inside it. A 10,000-shuffle permutation test on the best of
# them, momentum at 0.5750, returns p=0.0625 - failing at alpha=0.05 BEFORE any
# correction for having tested eight estimators. With eight tests the chance of
# at least one spurious winner at 0.05 is 33.7%.
#
# SO X1 REPORTS NOT_APPROVED, AND THAT IS THE CORRECT ENGINEERING OUTCOME. The
# temptation this gate exists to refuse is picking momentum because 0.5750 is
# the largest number in the column. A gate that approves the best of eight
# noise draws is not a gate; it is a random number generator with a rubber stamp.
OOS_VALIDATION_VERSION = "oos-validation-v1"

# The baseline every candidate must BEAT, not merely match. Named here as data
# so the comparison cannot quietly drift to an easier opponent.
OOS_BASELINE_ESTIMATOR = "historical_mean"

# A CANDIDATE MUST BEAT THE BASELINE, not tie it. MEASURED, the three estimators
# whose intervals overlap the baseline are indistinguishable from predicting the
# mean, and shipping one would claim an edge the data does not show.
OOS_REQUIRES_BEATING_BASELINE = True

# Significance is required, not just a better point estimate. MEASURED, the
# whole directional column sits inside the sampling band, so point estimates
# alone would approve pure noise.
OOS_REQUIRES_SIGNIFICANCE = True
OOS_ALPHA = 0.05

# Minimum validation observations. At n=120 the directional sampling band is
# +/-0.0895 - wider than any effect observed - so 120 is demonstrably too few to
# resolve the question, and a gate that accepted it would be certifying noise.
# 120 is therefore the FLOOR for even attempting the test, never evidence of
# sufficiency.
OOS_MIN_OBSERVATIONS = 120

# FOLDS. MEASURED, every run on disk has exactly ONE fold, so its metric has no
# dispersion and nothing distinguishes a real edge from one lucky split. More
# than one fold is required before a result is believable.
OOS_MIN_FOLDS = 2

# The embargo between train and validation, in days. MEASURED at 96 days on the
# runs on disk, which is why those numbers are negative rather than leaked. A
# zero embargo at a 20d horizon would let the validation window overlap labels
# the model already saw.
OOS_MIN_EMBARGO_DAYS = 20

# THE GATE NEVER INVENTS A RESULT FOR A MISSING RUN. An absent estimator is
# ABSENT, never a zero score, because a zero would rank it last rather than
# unranked and could make a real candidate look good by comparison.
OOS_COERCES_MISSING = False

# Verdicts.
OOS_APPROVED = "APPROVED"
OOS_NOT_APPROVED = "NOT_APPROVED"
OOS_NOT_EVALUATED = "NOT_EVALUATED"   # nothing to evaluate
OOS_VERDICTS: tuple[str, ...] = (
    OOS_NOT_EVALUATED,
    OOS_NOT_APPROVED,
    OOS_APPROVED,
)

# X1 reports; it does not promote. The registry decides promotion.
OOS_BLOCKS_TRADES = False


def _validate_oos_validation_config() -> None:
    """Import-time guard for the X1 contract."""
    if not OOS_REQUIRES_BEATING_BASELINE:
        raise ValueError(
            "a candidate must BEAT the baseline, not tie it: MEASURED, the "
            "estimators whose bootstrap intervals overlap historical_mean are "
            "indistinguishable from predicting the mean"
        )
    if not OOS_REQUIRES_SIGNIFICANCE:
        raise ValueError(
            "a better point estimate is not evidence: MEASURED, all eight "
            "directional accuracies fall inside the 0.5 +/- 0.0895 sampling "
            "band at n=120, so point estimates alone would approve pure noise"
        )
    if not 0.0 < OOS_ALPHA < 1.0:
        raise ValueError(f"OOS_ALPHA must lie in (0, 1), got {OOS_ALPHA!r}")
    if OOS_ALPHA > 0.05:
        raise ValueError(
            f"an alpha of {OOS_ALPHA} is looser than the 0.05 the best "
            f"estimator already fails at (momentum, p=0.0625)"
        )
    if OOS_MIN_OBSERVATIONS < 120:
        raise ValueError(
            "120 observations is already too few to resolve a directional "
            "edge; a lower floor certifies noise"
        )
    if OOS_MIN_FOLDS < 2:
        raise ValueError(
            "a single fold has no dispersion: MEASURED, every run on disk has "
            "exactly one fold, so nothing distinguishes a real edge from one "
            "lucky split"
        )
    if OOS_MIN_EMBARGO_DAYS <= 0:
        raise ValueError(
            "a zero embargo at a 20d horizon lets the validation window "
            "overlap labels the model already saw"
        )
    if OOS_COERCES_MISSING:
        raise ValueError(
            "an absent estimator must stay ABSENT: a zero score would rank it "
            "last rather than unranked, and could make a real candidate look "
            "good by comparison"
        )
    if len(set(OOS_VERDICTS)) != len(OOS_VERDICTS):
        raise ValueError("duplicate OOS verdict")
    if OOS_NOT_APPROVED == OOS_NOT_EVALUATED:
        raise ValueError(
            "'we tested and it failed' and 'there was nothing to test' are "
            "different answers; collapsing them hides which one happened"
        )
    if not OOS_BASELINE_ESTIMATOR:
        raise ValueError("the baseline estimator must be named")
    if OOS_BLOCKS_TRADES:
        raise ValueError("X1 reports; the registry promotes")


_validate_oos_validation_config()


# --- X2: calibration gate --------------------------------------------------------
# "Probabilities must be demonstrably calibrated." MEASURED, the number the
# system currently reports is 0.0000 for every estimator, and that number is
# WORTHLESS - not optimistic, but structurally incapable of being anything else.
#
# THE DECIDING MEASUREMENT. Fitting the isotonic map on 120 observations and
# scoring it on the SAME 120 gives a perfect ECE every time. Refitting on the
# first 60 and scoring the held-out 60 gives the honest number:
#
#     estimator          ECE in-sample   ECE holdout   MCE holdout
#     elastic_net              0.0000        0.1894        0.1920
#     ridge                    0.0000        0.1689        0.1689
#     logistic                 0.0000        0.1669        0.1669
#     mean_reversion           0.0000        0.1669        0.1669
#     momentum                 0.0000        0.1669        0.1669
#     historical_mean          0.0000        0.1667        0.1667
#     random_forest            0.0000        0.1665        0.1665
#     gradient_boosting        0.0000        0.1660        0.1660
#
# WHY IN-SAMPLE ECE IS EXACTLY ZERO, not merely small. Isotonic regression on
# this data collapses to TWO knots. Every observation is mapped to the base rate
# of its own group, so each reliability bin reproduces its own observed
# frequency by construction. An in-sample ECE cannot detect miscalibration; it
# can only report the arithmetic identity it was built from.
#
# WHAT THE 0.1667 ACTUALLY IS, and it is not a fitting artefact: BASE-RATE
# DRIFT. The training half is 63.3% up-days and the holdout half 46.7%. The map
# learned a base rate that had ALREADY CHANGED by the time it was applied. That
# is a live point-in-time failure - exactly what a calibration gate exists to
# catch - and the system currently reports it as perfect.
#
# AND IT IS REAL, NOT SAMPLING NOISE. Simulating a perfectly calibrated constant
# predictor at n=60 gives a median ECE of 0.0500 and a 95th percentile of
# 0.1167. The observed 0.1667 sits above that, so it is a genuine error rather
# than the noise a small holdout always produces.
CALIBRATION_GATE_VERSION = "calibration-gate-v1"

# IN-SAMPLE CALIBRATION IS NEVER EVIDENCE. The single most important rule here.
CALIBRATION_GATE_REQUIRES_HOLDOUT = True

# The ECE bar. REUSED from L6's measured drift threshold rather than invented,
# so "miscalibrated" means one thing system wide and does not drift between the
# gate that admits a model and the detector that retires it.
CALIBRATION_GATE_MAX_ECE = DRIFT_CALIBRATION_GAP

# The worst-bin bar. ECE is count-weighted, so a badly wrong region carrying few
# observations can hide inside an acceptable average. MCE cannot hide it. Set at
# twice the ECE bar: a single bin may be worse than the average, but not
# unboundedly so.
CALIBRATION_GATE_MAX_MCE = 2.0 * DRIFT_CALIBRATION_GAP

# THE NOISE FLOOR, MEASURED rather than assumed. A perfectly calibrated constant
# predictor at n=60 still posts a median ECE of 0.0500 and a p95 of 0.1167, so
# an ECE below the floor is NOT evidence of good calibration - it is evidence
# the holdout is too small to tell. A gate that read it as a pass would approve
# anything on a thin sample.
# The anchor is the base-rate-0.5 simulation (0.1333), which is the CONSERVATIVE
# of the two measurements: the observed-base-rate run gave 0.1167, and using the
# smaller number would understate how much noise a thin holdout produces.
CALIBRATION_GATE_NOISE_FLOOR_P95 = 0.1333
CALIBRATION_GATE_NOISE_FLOOR_N = 60

# Minimum holdout observations. At n=60 the noise floor (0.1167) is already
# larger than the ECE bar (0.10), so a 60-observation holdout CANNOT
# distinguish a calibrated model from an uncalibrated one.
#
# 200 is MEASURED as the smallest round size whose p95 floor falls below the
# 0.10 bar (3,000 simulations per size at base rate 0.5):
#
#     n      median ECE   p95 ECE
#     60         0.0500    0.1333
#     100        0.0400    0.1000
#     150        0.0267    0.0800
#     200        0.0250    0.0700   <- first size that can decide
#     500        0.0160    0.0440
CALIBRATION_GATE_MIN_HOLDOUT = 200

# A MISSING PROBABILITY IS NEVER 0.5. "The model declined to predict" and "the
# model predicted a coin flip" are different claims, and averaging the first
# into the second manufactures calibration evidence from silence.
CALIBRATION_GATE_COERCES_MISSING = False

# Verdicts.
CALIBRATION_GATE_APPROVED = "APPROVED"
CALIBRATION_GATE_NOT_APPROVED = "NOT_APPROVED"
CALIBRATION_GATE_NOT_EVALUATED = "NOT_EVALUATED"   # no honest test was possible
CALIBRATION_GATE_VERDICTS: tuple[str, ...] = (
    CALIBRATION_GATE_NOT_EVALUATED,
    CALIBRATION_GATE_NOT_APPROVED,
    CALIBRATION_GATE_APPROVED,
)

# X2 reports; the registry promotes.
CALIBRATION_GATE_BLOCKS_TRADES = False


def _validate_calibration_gate_config() -> None:
    """Import-time guard for the X2 contract."""
    if not CALIBRATION_GATE_REQUIRES_HOLDOUT:
        raise ValueError(
            "in-sample calibration is never evidence: MEASURED, fitting and "
            "scoring isotonic on the same 120 observations gives ECE 0.0000 "
            "for every estimator, because each bin reproduces its own base "
            "rate by construction"
        )
    if not 0.0 < CALIBRATION_GATE_MAX_ECE < 1.0:
        raise ValueError(
            f"CALIBRATION_GATE_MAX_ECE must lie in (0, 1), got "
            f"{CALIBRATION_GATE_MAX_ECE!r}"
        )
    if CALIBRATION_GATE_MAX_ECE != DRIFT_CALIBRATION_GAP:
        raise ValueError(
            "the ECE bar must be L6's measured drift threshold: a gate that "
            "admits a model looser than the detector that retires it would "
            "approve something already known to be drifting"
        )
    if CALIBRATION_GATE_MAX_MCE <= CALIBRATION_GATE_MAX_ECE:
        raise ValueError(
            "the worst-bin bar must exceed the average bar, or MCE adds "
            "nothing that ECE did not already say"
        )
    if CALIBRATION_GATE_MIN_HOLDOUT <= CALIBRATION_GATE_NOISE_FLOOR_N:
        raise ValueError(
            f"a {CALIBRATION_GATE_MIN_HOLDOUT}-observation holdout cannot "
            f"decide anything: MEASURED, the p95 noise floor at n="
            f"{CALIBRATION_GATE_NOISE_FLOOR_N} is "
            f"{CALIBRATION_GATE_NOISE_FLOOR_P95}, already larger than the "
            f"{CALIBRATION_GATE_MAX_ECE} bar"
        )
    if CALIBRATION_GATE_NOISE_FLOOR_P95 <= CALIBRATION_GATE_MAX_ECE:
        raise ValueError(
            "the measured noise floor is below the bar, which would make the "
            "floor irrelevant; it is recorded because it is NOT"
        )
    if CALIBRATION_GATE_COERCES_MISSING:
        raise ValueError(
            "a missing probability must never become 0.5: 'the model declined "
            "to predict' and 'the model predicted a coin flip' are different "
            "claims, and merging them manufactures calibration evidence from "
            "silence"
        )
    if len(set(CALIBRATION_GATE_VERDICTS)) != len(CALIBRATION_GATE_VERDICTS):
        raise ValueError("duplicate calibration verdict")
    if CALIBRATION_GATE_NOT_APPROVED == CALIBRATION_GATE_NOT_EVALUATED:
        raise ValueError(
            "'measurably miscalibrated' and 'no honest test was possible' are "
            "different answers; collapsing them hides which one happened"
        )
    if CALIBRATION_GATE_BLOCKS_TRADES:
        raise ValueError("X2 reports; the registry promotes")


_validate_calibration_gate_config()


# --- X3: regime robustness -------------------------------------------------------
# "No single regime should explain the entire edge." MEASURED, this gate CANNOT
# RUN on the data the repository ships, and the reason is structural rather
# than a shortage of history.
#
# THE BLOCKING MEASUREMENT: THE FOLDS CARRY NO REGIME AND NO WAY TO RECOVER ONE.
# A persisted fold holds exactly this:
#
#     actuals, predictions, fold_id, metrics,
#     train_end_time, train_rows, validation_rows, validation_start_time
#
# There is no ticker and no per-observation timestamp - only a FOLD-LEVEL
# window. TrainingRow DOES carry ticker and prediction_time (training_dataset.py
# builds and sorts on them), and the dataset record on disk is a MANIFEST: it
# describes 406 rows with hashes and versions but stores no row. The identity
# exists at build time and is DROPPED when folds are persisted, because
# _fit_predict works on bare arrays.
#
# So a regime label cannot be joined onto a validation observation, and X3 must
# report NOT_EVALUATED rather than invent an attribution. The named next action
# is concrete: persist ticker and prediction_time alongside each fold's
# predictions, which is a change to the training pipeline and not to this gate.
#
# WHAT THE GATE WILL MEASURE ONCE THAT LANDS, demonstrated here on seeded data
# rather than asserted. Four regimes of 100 observations each, one carrying a
# real edge and three pure noise:
#
#     pooled directional accuracy          0.6800
#       bullish                            0.9700
#       range                              0.6400
#       bearish                            0.5600
#       risk_off                           0.5500
#     with the carrying regime removed     0.5833
#
# A pooled 0.6800 that looks like an edge collapses to 0.5833 when ONE regime
# is dropped. That is the failure this gate exists to catch.
#
# WHY "IS EVERY REGIME ABOVE A COIN FLIP" IS THE WRONG TEST. MEASURED, the
# answer is 4 OF 4 for BOTH the concentrated book above and a book with a
# modest edge spread evenly - the statistic does not separate them at all.
#
# THE STATISTIC THAT DOES: LEAVE ONE REGIME OUT. Measured on the same pair:
#
#                         concentrated   broad
#     worst LOO drop            0.0967  0.0225
#     per-regime spread         0.4200  0.1200
#
# THE THRESHOLD IS MEASURED, NOT CHOSEN. Simulating 400 books whose edge is
# GENUINELY UNIFORM across regimes, the worst leave-one-out drop has a median
# of 0.0150, a p95 of 0.0300, and a MAXIMUM of 0.0525. A 0.05 bar therefore
# sits at the top of what uniformity produces by chance. At cell sizes 60, 100
# and 200 it flagged the concentrated book 100% of the time and the uniform
# book 0% of the time.
REGIME_ROBUSTNESS_VERSION = "regime-robustness-v1"

# The regimes an edge must survive, reused from N4's vocabulary so "regime"
# means one thing system wide.
REGIME_ROBUSTNESS_LABELS: tuple[str, ...] = REGIME_LABELS

# The leave-one-out drop above which an edge is judged CONCENTRATED. MEASURED:
# the maximum a uniform edge produced in 400 trials was 0.0525, with a p95 of
# 0.0300, so 0.05 is the top of the null rather than a round number.
REGIME_ROBUSTNESS_MAX_DROP = 0.05

# PER-REGIME SPREAD IS RECORDED BUT NEVER DECIDES. MEASURED, a uniform edge
# produces a spread with a p95 of 0.1800 and a max of 0.2700 purely from
# per-cell sampling noise, so a spread test would fail robust models constantly.
REGIME_ROBUSTNESS_SPREAD_DECIDES = False

# Minimum observations in a regime cell before that regime's number is used.
# MEASURED, the sampling band on directional accuracy at n=100 is +/-0.0980 -
# wider than the 0.05 bar itself - so a thinner cell cannot support a
# leave-one-out claim in either direction. Reused from L7's measured floor so
# the two agree on what a usable cell is.
REGIME_ROBUSTNESS_MIN_CELL = REGIME_SPECIALIZATION_MIN_CELL

# Minimum regimes that must be represented. With fewer than two, "leave one
# out" leaves nothing to compare against.
REGIME_ROBUSTNESS_MIN_REGIMES = 2

# AN UNLABELLED OBSERVATION IS NEVER ASSIGNED A REGIME. This is the rule that
# makes the gate honest today: guessing a label from the fold window would
# manufacture the very attribution the gate is supposed to test.
REGIME_ROBUSTNESS_INFERS_LABELS = False

# Verdicts.
REGIME_ROBUST = "ROBUST"                       # no single regime carries it
REGIME_CONCENTRATED = "CONCENTRATED"           # one regime explains the edge
REGIME_ROBUSTNESS_NOT_EVALUATED = "NOT_EVALUATED"  # no regime labels exist
REGIME_ROBUSTNESS_VERDICTS: tuple[str, ...] = (
    REGIME_ROBUSTNESS_NOT_EVALUATED,
    REGIME_CONCENTRATED,
    REGIME_ROBUST,
)

# X3 reports; the registry promotes.
REGIME_ROBUSTNESS_BLOCKS_TRADES = False


def _validate_regime_robustness_config() -> None:
    """Import-time guard for the X3 contract."""
    if set(REGIME_ROBUSTNESS_LABELS) != set(REGIME_LABELS):
        raise ValueError(
            "X3 must test the same regimes N4 classifies, or an edge could "
            "hide in a regime nobody checked"
        )
    if not 0.0 < REGIME_ROBUSTNESS_MAX_DROP < 1.0:
        raise ValueError(
            f"REGIME_ROBUSTNESS_MAX_DROP must lie in (0, 1), got "
            f"{REGIME_ROBUSTNESS_MAX_DROP!r}"
        )
    if REGIME_ROBUSTNESS_MAX_DROP < 0.03:
        raise ValueError(
            f"a drop bar of {REGIME_ROBUSTNESS_MAX_DROP} sits below the 0.0300 "
            f"p95 that a UNIFORM edge produces by chance, so genuinely robust "
            f"models would be flagged as concentrated"
        )
    if REGIME_ROBUSTNESS_MAX_DROP > 0.0967:
        raise ValueError(
            f"a drop bar of {REGIME_ROBUSTNESS_MAX_DROP} is above the 0.0967 "
            f"drop MEASURED on a book where one regime carried the entire "
            f"edge, so that book would pass"
        )
    if REGIME_ROBUSTNESS_SPREAD_DECIDES:
        raise ValueError(
            "per-regime spread must not decide: MEASURED, a uniform edge "
            "produces a spread with a p95 of 0.1800 and a max of 0.2700 from "
            "sampling noise alone"
        )
    if REGIME_ROBUSTNESS_MIN_CELL != REGIME_SPECIALIZATION_MIN_CELL:
        raise ValueError(
            "X3 and L7 must agree on what a usable regime cell is, or one "
            "would trust a cell the other rejects"
        )
    if REGIME_ROBUSTNESS_MIN_REGIMES < 2:
        raise ValueError(
            "leave-one-out needs at least two regimes, or there is nothing "
            "left to compare against"
        )
    if REGIME_ROBUSTNESS_INFERS_LABELS:
        raise ValueError(
            "an unlabelled observation must never be assigned a regime: "
            "guessing from the fold window would manufacture the very "
            "attribution this gate exists to test"
        )
    if len(set(REGIME_ROBUSTNESS_VERDICTS)) != len(REGIME_ROBUSTNESS_VERDICTS):
        raise ValueError("duplicate regime robustness verdict")
    if REGIME_CONCENTRATED == REGIME_ROBUSTNESS_NOT_EVALUATED:
        raise ValueError(
            "'one regime carries the edge' and 'we could not tell' are "
            "different answers; collapsing them hides that the test never ran"
        )
    if REGIME_ROBUSTNESS_BLOCKS_TRADES:
        raise ValueError("X3 reports; the registry promotes")


_validate_regime_robustness_config()


# --- X4: event robustness --------------------------------------------------------
# "No single viral event/source should explain the apparent edge." MEASURED,
# this gate cannot run today, and its blocker is DIFFERENT from X3's.
#
# X3's blocker is a DROPPED JOIN KEY: the regime data exists, but folds discard
# the ticker and timestamp needed to attach it. X4's blocker is that THE EVENT
# DATA DOES NOT EXIST AT ALL:
#
#     event memories on disk                     0
#     raw store directories      alpha_vantage_overview, yahoo_finance_chart
#     news store                                 none
#     COLLECT_NEWS_CURSOR_PATH                   does not exist
#
# The raw store holds price and fundamentals only. No news has ever been
# ingested, so there is no event to attribute an edge to. X4 reports
# NOT_EVALUATED and names the two distinct fixes - ingest news, AND carry the
# event id and source onto each validation observation - because doing only the
# first still leaves nothing joinable.
#
# WHAT THE GATE MEASURES ONCE THAT LANDS, demonstrated on seeded books.
#
# THE NAIVE TEST IS WRONG: A RAW DROP THRESHOLD FALSE-FLAGS. Removing a viral
# event carrying the edge drops accuracy 0.7850 -> 0.4750 (a fall of 0.3100),
# but removing an ORDINARY event from a small book still drops it 0.1100. A
# fixed bar cannot tell a carried edge from a book with few events, because
# MEASURED, the null drop scales with the removed item's SHARE of the book:
#
#     events   each share   p95 null drop   p95/share
#     3            33.3%          0.0667        0.200
#     4            25.0%          0.0488        0.195
#     5            20.0%          0.0367        0.183
#     8            12.5%          0.0232        0.186
#     10           10.0%          0.0204        0.204
#     20            5.0%          0.0114        0.228
#
# THE RATIO IS STABLE ACROSS A 6.7x RANGE OF EVENT COUNTS. So the test is
# SCALE-FREE: compare the drop to the item's share, never to a fixed number.
#
# THE BAR IS MEASURED, AND IT SITS IN A REAL GAP - BUT ONLY ONCE THE
# THREE-ITEM CASE IS EXCLUDED. An early 480-book sweep put the uniform maximum
# at 0.308; widening it to 1,800 books found 0.350, which TOUCHES a 0.35 bar.
# The whole tail came from THREE-item books, where removing one item deletes a
# third of the evidence and the surviving two-thirds decide the answer:
#
#     min items   uniform books   p99     max
#     3 or more           1,800   0.267   0.350   <- touches the bar
#     4 or more           1,500   0.263   0.322
#     5 or more           1,200   0.263   0.319
#
# So the floor is FOUR items, not three, and with it the gap is real: the
# uniform maximum is 0.322 and the minimum over 480 genuinely-carried books is
# 0.356. The 0.35 bar lies between them. The p99 barely moves (0.267 -> 0.263),
# which is the tell that this is a degenerate-case tail rather than a shift in
# the statistic.
#
# BOTH AXES ARE TESTED, because a source can carry an edge that no single event
# does. MEASURED on a book where one source supplies every third event, the
# worst EVENT ratio is 0.340 - under the bar - while the worst SOURCE ratio is
# 0.425. Testing events alone would miss that carrier entirely.
EVENT_ROBUSTNESS_VERSION = "event-robustness-v1"

# The axes an edge must survive. Declared as DATA so a reader sees what is
# tested, and so dropping one is a deliberate edit here.
EVENT_ROBUSTNESS_AXIS_EVENT = "event"
EVENT_ROBUSTNESS_AXIS_SOURCE = "source"
EVENT_ROBUSTNESS_AXES: tuple[str, ...] = (
    EVENT_ROBUSTNESS_AXIS_EVENT,
    EVENT_ROBUSTNESS_AXIS_SOURCE,
)

# The drop-to-share ratio above which an item is judged to CARRY the edge.
# MEASURED: the uniform null reached at most 0.308 over 480 books and a genuine
# carrier fell no lower than 0.360, so 0.35 sits inside that gap rather than
# being a round number.
EVENT_ROBUSTNESS_MAX_RATIO = 0.35

# A RAW DROP IS NEVER THE TEST. MEASURED, an ordinary event in a small book
# drops accuracy 0.1100 - larger than many real effects - purely because it is
# a fifth of the observations.
EVENT_ROBUSTNESS_USES_RAW_DROP = False

# Minimum distinct items on an axis before it can be judged. MEASURED, this is
# FOUR rather than three: with one item leaving it out leaves nothing, with two
# every removal halves the book, and with THREE the null tail reaches 0.350 -
# touching the bar - because removing one item deletes a third of the evidence.
# At four or more the uniform maximum falls to 0.322.
EVENT_ROBUSTNESS_MIN_ITEMS = 4

# Minimum observations attributed to an item before its own number is used.
# Reused from A3's analog floor so "too few examples to characterise an event"
# means one thing across the system.
EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS = EVENT_MEMORY_MIN_ANALOGS

# AN UNATTRIBUTED OBSERVATION IS NEVER ASSIGNED AN EVENT OR A SOURCE. The rule
# that makes the blocker honest: bucketing everything under one synthetic id
# would leave nothing to leave out, and the gate would return ROBUST having
# tested nothing.
EVENT_ROBUSTNESS_INFERS_ATTRIBUTION = False

# Verdicts.
EVENT_ROBUST = "ROBUST"                        # no single item carries it
EVENT_CARRIED = "CARRIED"                      # one event or source explains it
EVENT_ROBUSTNESS_NOT_EVALUATED = "NOT_EVALUATED"  # no event attribution exists
EVENT_ROBUSTNESS_VERDICTS: tuple[str, ...] = (
    EVENT_ROBUSTNESS_NOT_EVALUATED,
    EVENT_CARRIED,
    EVENT_ROBUST,
)

# X4 reports; the registry promotes.
EVENT_ROBUSTNESS_BLOCKS_TRADES = False


def _validate_event_robustness_config() -> None:
    """Import-time guard for the X4 contract."""
    if len(set(EVENT_ROBUSTNESS_AXES)) != len(EVENT_ROBUSTNESS_AXES):
        raise ValueError("duplicate event robustness axis")
    for required in (EVENT_ROBUSTNESS_AXIS_EVENT, EVENT_ROBUSTNESS_AXIS_SOURCE):
        if required not in EVENT_ROBUSTNESS_AXES:
            raise ValueError(
                f"the roadmap names {required!r} and it is not tested: "
                f"MEASURED, a source supplying every third event scores 0.425 "
                f"on the source axis and only 0.340 on the event axis, so "
                f"testing one axis misses carriers visible on the other"
            )
    if EVENT_ROBUSTNESS_USES_RAW_DROP:
        raise ValueError(
            "a raw drop is not the test: MEASURED, an ordinary event in a "
            "small book drops accuracy 0.1100 purely because it is a fifth of "
            "the observations, so a fixed bar false-flags honest books"
        )
    if not 0.0 < EVENT_ROBUSTNESS_MAX_RATIO < 1.0:
        raise ValueError(
            f"EVENT_ROBUSTNESS_MAX_RATIO must lie in (0, 1), got "
            f"{EVENT_ROBUSTNESS_MAX_RATIO!r}"
        )
    if EVENT_ROBUSTNESS_MAX_RATIO <= 0.322:
        raise ValueError(
            f"a ratio bar of {EVENT_ROBUSTNESS_MAX_RATIO} sits at or below the "
            f"0.322 maximum a UNIFORM edge produced over 1,500 four-item-plus "
            f"books, so honest books would be flagged as carried"
        )
    if EVENT_ROBUSTNESS_MAX_RATIO >= 0.356:
        raise ValueError(
            f"a ratio bar of {EVENT_ROBUSTNESS_MAX_RATIO} sits at or above the "
            f"0.356 minimum a GENUINE carrier produced over 480 books, so a "
            f"real carrier would pass"
        )
    if EVENT_ROBUSTNESS_MIN_ITEMS < 4:
        raise ValueError(
            f"{EVENT_ROBUSTNESS_MIN_ITEMS} items cannot be judged: with one "
            f"there is nothing to leave out, with two every removal halves the "
            f"book, and MEASURED, with three the uniform null reaches 0.350 "
            f"and touches the {EVENT_ROBUSTNESS_MAX_RATIO} bar"
        )
    if EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS != EVENT_MEMORY_MIN_ANALOGS:
        raise ValueError(
            "X4 and E6 must agree on how few examples are too few to "
            "characterise an event"
        )
    if EVENT_ROBUSTNESS_INFERS_ATTRIBUTION:
        raise ValueError(
            "an unattributed observation must never be assigned an event or a "
            "source: bucketing everything under one synthetic id would leave "
            "nothing to leave out, and the gate would return ROBUST having "
            "tested nothing"
        )
    if len(set(EVENT_ROBUSTNESS_VERDICTS)) != len(EVENT_ROBUSTNESS_VERDICTS):
        raise ValueError("duplicate event robustness verdict")
    if EVENT_CARRIED == EVENT_ROBUSTNESS_NOT_EVALUATED:
        raise ValueError(
            "'one event carries the edge' and 'there is no event data at all' "
            "are different answers; collapsing them hides that the test never "
            "ran"
        )
    if EVENT_ROBUSTNESS_BLOCKS_TRADES:
        raise ValueError("X4 reports; the registry promotes")


_validate_event_robustness_config()


# --- X5: feature ablation --------------------------------------------------------
# "Test removing news, technicals, macro, sentiment, fundamentals to prove which
# sources add incremental value." MEASURED, FOUR OF THE FIVE NAMED GROUPS HAVE
# NOTHING TO REMOVE - and unlike X3 and X4, the fifth is genuinely testable, so
# X5 is PARTIALLY evaluable rather than blocked outright.
#
# THE DECIDING MEASUREMENT. Every feature the training pipeline can produce:
#
#     change_1d   change_5d   change_20d  change_60d
#     ma_50       ma_100      ma_150      ma_200
#     price_vs_ma_50  price_vs_ma_100  price_vs_ma_150  price_vs_ma_200
#     rsi         trend_vs_20d_mean      volatility     volume_ratio_20d
#
# All sixteen are price and volume derivatives. Counting them by the groups the
# roadmap names:
#
#     group           features   ablatable?
#     technicals            16   YES
#     news                   0   nothing to remove
#     macro                  0   nothing to remove
#     sentiment              0   nothing to remove
#     fundamentals           0   nothing to remove
#
# WHY, STRUCTURALLY. The dataset builder defaults to
# CURRENT_SCORE_FEATURES | LONG_TERM_SCORE_FEATURES, both hardcoded tuples in
# score_engine.py holding only price and volume derivatives. The feature
# REGISTRY declares eight domains - market, fundamental, news, sentiment, macro,
# regime, technical, event - but data/feature_registry.jsonl DOES NOT EXIST, so
# zero features are registered and no non-technical feature has a producer a
# model could consume. An ablation over a group that was never present measures
# nothing, and reporting "removing news did not change performance" would be a
# lie told with a straight face.
#
# THE ONE ABLATION THE SHIPPED DATA DOES SUPPORT, and it is a real result.
# Removing ALL technical features leaves a model with no features at all, which
# is exactly the historical_mean baseline:
#
#     with all 16 technical features (best learned)   rmse 0.13192   dir 0.4583
#     with NO features at all (historical_mean)       rmse 0.12164   dir 0.5500
#
#     incremental value of all 16 technical features  +0.01028 rmse - WORSE
#
# So the one group that CAN be ablated has NEGATIVE incremental value: the
# features actively hurt. That is X1's finding re-expressed as an ablation, and
# it is the answer X5 was asked for, for the only group it can ask about.
FEATURE_ABLATION_VERSION = "feature-ablation-v1"

# The groups the roadmap names, declared as DATA so a reader sees what must be
# tested and so dropping one is a deliberate edit here.
ABLATION_GROUP_TECHNICAL = "technicals"
ABLATION_GROUP_NEWS = "news"
ABLATION_GROUP_MACRO = "macro"
ABLATION_GROUP_SENTIMENT = "sentiment"
ABLATION_GROUP_FUNDAMENTAL = "fundamentals"
ABLATION_GROUPS: tuple[str, ...] = (
    ABLATION_GROUP_TECHNICAL,
    ABLATION_GROUP_NEWS,
    ABLATION_GROUP_MACRO,
    ABLATION_GROUP_SENTIMENT,
    ABLATION_GROUP_FUNDAMENTAL,
)

# AN ABSENT GROUP IS NOT A GROUP THAT ADDED NOTHING. The rule this whole task
# turns on. MEASURED, four of the five groups have zero features, and reporting
# them as "no incremental value" would present an untested group as a tested
# one - the exact confusion A6 drew between NOT_EVALUATED and NOT_MET.
ABLATION_ABSENT_MEANS_NO_VALUE = False

# Minimum features a group must contribute before it can be ablated at all.
# One is enough to remove; zero is not.
ABLATION_MIN_GROUP_FEATURES = 1

# The metric ablations are compared on, and its direction. RMSE, lower better,
# matching X1 so "better" means one thing across the release gate.
ABLATION_METRIC = "rmse"
ABLATION_HIGHER_IS_BETTER = False

# THE MARGIN A GROUP MUST CLEAR TO COUNT AS ADDING VALUE. Reused from L7's
# measured specialization margin rather than invented, so "this change is real"
# means the same thing whether a regime model or a feature group is being
# judged.
ABLATION_MIN_IMPROVEMENT = REGIME_SPECIALIZATION_MARGIN

# A group whose removal IMPROVES the metric is reported as HARMFUL, not merely
# as adding nothing. MEASURED, the technical group is exactly this case at
# +0.01028 rmse, and calling that "no value" would understate it.
ABLATION_REPORTS_HARMFUL = True

# Verdicts, per group.
ABLATION_ADDS_VALUE = "ADDS_VALUE"
ABLATION_NO_VALUE = "NO_VALUE"           # present, removed, nothing changed
ABLATION_HARMFUL = "HARMFUL"             # removing it IMPROVED the metric
ABLATION_ABSENT = "ABSENT"               # no producer exists; never tested
ABLATION_NOT_EVALUATED = "NOT_EVALUATED"  # present but the test could not run
ABLATION_VERDICTS: tuple[str, ...] = (
    ABLATION_NOT_EVALUATED,
    ABLATION_ABSENT,
    ABLATION_HARMFUL,
    ABLATION_NO_VALUE,
    ABLATION_ADDS_VALUE,
)

# X5 reports; the registry promotes.
ABLATION_BLOCKS_TRADES = False


def _validate_feature_ablation_config() -> None:
    """Import-time guard for the X5 contract."""
    if len(set(ABLATION_GROUPS)) != len(ABLATION_GROUPS):
        raise ValueError("duplicate ablation group")
    for required in (
        ABLATION_GROUP_TECHNICAL,
        ABLATION_GROUP_NEWS,
        ABLATION_GROUP_MACRO,
        ABLATION_GROUP_SENTIMENT,
        ABLATION_GROUP_FUNDAMENTAL,
    ):
        if required not in ABLATION_GROUPS:
            raise ValueError(
                f"the roadmap names {required!r} and it is not tested; a group "
                f"nobody ablates is a source whose value is never proved"
            )
    if ABLATION_ABSENT_MEANS_NO_VALUE:
        raise ValueError(
            "an absent group must never be reported as adding no value: "
            "MEASURED, four of the five named groups have ZERO features, and "
            "saying 'removing news changed nothing' about a group that was "
            "never present presents an untested source as a tested one"
        )
    if ABLATION_MIN_GROUP_FEATURES < 1:
        raise ValueError(
            "a group needs at least one feature to be ablated; removing "
            "nothing is not an experiment"
        )
    if ABLATION_HIGHER_IS_BETTER:
        raise ValueError(
            f"{ABLATION_METRIC} is an error metric: lower is better, and "
            f"inverting it would report harmful features as valuable"
        )
    if ABLATION_MIN_IMPROVEMENT != REGIME_SPECIALIZATION_MARGIN:
        raise ValueError(
            "the improvement margin must be L7's measured one, or 'this "
            "change is real' would mean different things in different gates"
        )
    if ABLATION_MIN_IMPROVEMENT <= 0.0:
        raise ValueError(
            "a zero margin makes any rounding difference look like value"
        )
    if not ABLATION_REPORTS_HARMFUL:
        raise ValueError(
            "a group whose removal IMPROVES the metric must be reported as "
            "HARMFUL: MEASURED, the technical group is +0.01028 rmse, and "
            "calling that 'no value' understates it"
        )
    if len(set(ABLATION_VERDICTS)) != len(ABLATION_VERDICTS):
        raise ValueError("duplicate ablation verdict")
    if ABLATION_ABSENT == ABLATION_NO_VALUE:
        raise ValueError(
            "'this source has no producer' and 'we removed it and nothing "
            "changed' are different answers; collapsing them is how an "
            "untested source gets presented as a tested one"
        )
    if ABLATION_HARMFUL == ABLATION_NO_VALUE:
        raise ValueError(
            "'removing it helped' and 'removing it changed nothing' are "
            "different findings"
        )
    if ABLATION_BLOCKS_TRADES:
        raise ValueError("X5 reports; the registry promotes")


_validate_feature_ablation_config()


# --- X6: temporal robustness -----------------------------------------------------
# "Evaluate across 1D/5D/20D/60D/120D." MEASURED, FOUR OF THE FIVE HORIZONS HAVE
# NO TRAINED MODEL AT ALL - and this blocker is different in kind from X3's and
# X4's, because NOTHING NEEDS BUILDING.
#
# THE DECIDING MEASUREMENT: EVERY TRAINED RUN IS 20d.
#
#     horizons in data/training_runs.jsonl    {'20d': 8}
#     horizons X6 requires                    1d, 5d, 20d, 60d, 120d
#     horizons with a model                   1 of 5
#
# X3's blocker is a dropped join key and X4's is data that was never ingested.
# X6's is neither: LABEL_HORIZON_SESSIONS already maps all five horizons
# (1/5/20/60/120 sessions), TrainingRun already carries target_horizon, and
# train_baseline already accepts one. The runs were simply never produced. The
# named next action is to train the four missing horizons, not to build
# machinery.
#
# WHY A SINGLE HORIZON CANNOT ESTABLISH ROBUSTNESS, and this is the number that
# decides the module. Simulating 1,500 books per arm at n=120, where the "no
# edge" arm is pure noise:
#
#     require   false-positive   detection
#     1 of 5            24.80%     100.00%
#     2 of 5             2.53%     100.00%
#     3 of 5             0.13%      99.93%
#     4 of 5             0.00%      98.73%
#     5 of 5             0.00%      84.93%
#
# A MODEL EVALUATED AT ONE HORIZON CLEARS THE BAND 24.8% OF THE TIME ON PURE
# NOISE. That is the cost of the current state: nearly one in four noise models
# would look temporally validated, because "it worked at 20d" is exactly the
# single-horizon claim the table's first row prices.
#
# THREE OF FIVE IS WHERE THE FALSE-POSITIVE RATE COLLAPSES - 2.53% to 0.13% -
# while detection is essentially untouched (100.00% to 99.93%). Requiring four
# is safer on paper but costs real detection (98.73%) for no measured gain, and
# requiring all five costs 15 points. Across 1,500 null trials the maximum
# number of horizons a noise model cleared was 3, so the bar sits exactly at
# the top of the null.
TEMPORAL_ROBUSTNESS_VERSION = "temporal-robustness-v1"

# The horizons X6 names. A SUBSET of F2's six - 252d is excluded deliberately,
# because at 120 observations a 252-session label consumes more history than a
# fold holds.
TEMPORAL_HORIZONS: tuple[str, ...] = ("1d", "5d", "20d", "60d", "120d")

# How many must show an edge before the edge is called temporally robust.
# MEASURED as the point where the false-positive rate collapses and the null
# tops out.
TEMPORAL_MIN_AGREEING = 3

# EVERY REQUIRED HORIZON MUST BE EVALUATED, present or not. A horizon with no
# model is MISSING, never "did not show an edge" - the A6 distinction again.
TEMPORAL_ABSENT_MEANS_NO_EDGE = False

# Minimum observations before a horizon's own result is used. Reused from X1's
# measured floor so "too few to resolve a directional edge" means one thing
# across the release gate.
TEMPORAL_MIN_OBSERVATIONS = OOS_MIN_OBSERVATIONS

# THE EDGE TEST AT EACH HORIZON IS X1'S. Reused rather than restated, so a
# horizon cannot be called robust on a looser standard than X1 applies to the
# whole model.
TEMPORAL_ALPHA = OOS_ALPHA

# WHETHER AGREEMENT MUST SPAN THE RANGE: NO, AND THIS IS A CLAIM I TESTED AND
# DROPPED. It is intuitive that three ADJACENT horizons agreeing says less than
# three spread across the range, because their labels overlap. MEASURED over
# 2,000 null trials per arm, it is not true at this sample size:
#
#     adjacent (overlapping labels)   both clear the band   0.50%
#     spread   (disjoint labels)      both clear the band   0.45%
#
# A five-hundredths-of-a-percent difference is noise. Requiring spread would be
# an unmeasured threshold dressed as a safeguard, so X6 counts agreeing
# horizons without weighting where they sit. The short/long split is recorded
# for the REPORT, so a reader can see the shape of the agreement, but it does
# not gate.
TEMPORAL_REQUIRES_SPREAD = False
TEMPORAL_SHORT_HORIZONS: tuple[str, ...] = ("1d", "5d")
TEMPORAL_LONG_HORIZONS: tuple[str, ...] = ("60d", "120d")

# Verdicts.
TEMPORAL_ROBUST = "ROBUST"                      # enough horizons agree
TEMPORAL_FRAGILE = "FRAGILE"                    # evaluated, too few agree
TEMPORAL_NOT_EVALUATED = "NOT_EVALUATED"        # horizons are missing
TEMPORAL_VERDICTS: tuple[str, ...] = (
    TEMPORAL_NOT_EVALUATED,
    TEMPORAL_FRAGILE,
    TEMPORAL_ROBUST,
)

# X6 reports; the registry promotes.
TEMPORAL_BLOCKS_TRADES = False


def _validate_temporal_robustness_config() -> None:
    """Import-time guard for the X6 contract."""
    if len(set(TEMPORAL_HORIZONS)) != len(TEMPORAL_HORIZONS):
        raise ValueError("duplicate temporal horizon")
    for required in ("1d", "5d", "20d", "60d", "120d"):
        if required not in TEMPORAL_HORIZONS:
            raise ValueError(
                f"the roadmap names {required} and it is not evaluated; a "
                f"horizon nobody tests is a horizon whose edge is unproven"
            )
    for horizon in TEMPORAL_HORIZONS:
        if horizon not in LABEL_HORIZON_SESSIONS:
            raise ValueError(
                f"{horizon} has no label definition, so it cannot be trained "
                f"or evaluated"
            )
    if TEMPORAL_MIN_AGREEING < 3:
        raise ValueError(
            f"requiring {TEMPORAL_MIN_AGREEING} horizon(s) is too loose: "
            f"MEASURED, a pure-noise model clears one horizon 24.80% of the "
            f"time and two 2.53% of the time, against 0.13% at three"
        )
    if TEMPORAL_MIN_AGREEING > len(TEMPORAL_HORIZONS):
        raise ValueError(
            "more agreeing horizons are required than exist, so nothing can "
            "ever pass"
        )
    if TEMPORAL_ABSENT_MEANS_NO_EDGE:
        raise ValueError(
            "a horizon with no model must never be read as 'no edge there': "
            "MEASURED, four of the five horizons have no trained run at all, "
            "and calling that a negative result reports an untested horizon "
            "as a tested one"
        )
    if TEMPORAL_MIN_OBSERVATIONS != OOS_MIN_OBSERVATIONS:
        raise ValueError(
            "X6 and X1 must agree on how few observations are too few, or a "
            "horizon could be called robust on a sample X1 rejects"
        )
    if TEMPORAL_ALPHA != OOS_ALPHA:
        raise ValueError(
            "X6 must test each horizon at X1's alpha, or a horizon could pass "
            "on a looser standard than the whole model is held to"
        )
    if TEMPORAL_REQUIRES_SPREAD:
        raise ValueError(
            "requiring agreement to span the range is an UNMEASURED "
            "threshold: MEASURED over 2,000 null trials per arm, adjacent "
            "horizons cleared the band together 0.50% of the time and spread "
            "ones 0.45% - no difference at this sample size"
        )
    if set(TEMPORAL_SHORT_HORIZONS) & set(TEMPORAL_LONG_HORIZONS):
        raise ValueError("a horizon cannot be both short and long")
    for horizon in (*TEMPORAL_SHORT_HORIZONS, *TEMPORAL_LONG_HORIZONS):
        if horizon not in TEMPORAL_HORIZONS:
            raise ValueError(f"{horizon} is not an evaluated horizon")
    if len(set(TEMPORAL_VERDICTS)) != len(TEMPORAL_VERDICTS):
        raise ValueError("duplicate temporal verdict")
    if TEMPORAL_FRAGILE == TEMPORAL_NOT_EVALUATED:
        raise ValueError(
            "'we tested every horizon and too few agreed' and 'most horizons "
            "have no model' are different answers; collapsing them hides that "
            "the sweep never happened"
        )
    if TEMPORAL_BLOCKS_TRADES:
        raise ValueError("X6 reports; the registry promotes")


_validate_temporal_robustness_config()


# --- X7: multiple-testing protection ---------------------------------------------
# "Guard against selection bias (bootstrap/permutation, Reality Check/SPA-style
# methods, Probability of Backtest Overfitting, Deflated Sharpe where
# applicable)." MEASURED, the first problem is not WHICH correction to apply -
# it is that THE SYSTEM DOES NOT KNOW HOW MANY TESTS IT RAN.
#
# THE DECIDING MEASUREMENT: THE TRIAL REGISTRY UNDERCOUNTS BY 8x.
#
#     rows in data/research_trials.jsonl               2
#     DISTINCT trial ids among them                    1  (registered, then completed)
#     estimators actually trained                      8
#     estimators with a registered trial               1
#
# M3 exists precisely to stop uncontrolled experimentation, and seven of the
# eight runs bypassed it. Any correction computed from the registry would use
# n=1 when the truth is n=8:
#
#     n=1   bonferroni alpha 0.0500   family-wise risk  5.0%
#     n=8   bonferroni alpha 0.0063   family-wise risk 33.7%
#
# THE STATED RISK WOULD BE WRONG BY 6.7x. The conclusion happens to survive -
# X1's best permutation p of 0.0625 fails at both 0.0500 and 0.0063 - but a gate
# that reports 5% when the answer is 33.7% is reporting a number it did not
# measure. So X7 corrects on the OBSERVED family, never on the registry, and
# reports the discrepancy as its own finding.
#
# WHY NOT BONFERRONI ALONE. It assumes INDEPENDENT tests. Estimators trained on
# the SAME data and the SAME folds are correlated, so it over-corrects.
# MEASURED over 1,500 families of 8 at n=120:
#
#                            FWER at raw 0.05   at bonferroni 0.00625
#     independent estimators           35.6%                    6.00%
#     correlated (same data)           18.6%                    3.73%
#
# Against a 5% target, Bonferroni lands at 3.73% on the realistic (correlated)
# case: CONSERVATIVE, not wrong. It is kept as a floor because it needs no
# resampling and cannot be gamed.
#
# THE MAX-STATISTIC PERMUTATION TEST is what handles correlation properly,
# because it resamples the ACTUAL family rather than assuming its structure -
# shuffle the outcomes, recompute the BEST of the family, and build the null
# from those maxima. MEASURED over 200 families each:
#
#                            max-stat permutation FWER (target 5%)
#     independent estimators                              3.00%
#     correlated (same data)                              5.50%
#
# It tracks the target in both regimes where Bonferroni does not. So X7 requires
# BOTH: the permutation test decides, and Bonferroni is the floor that a result
# must also clear.
MULTIPLE_TESTING_VERSION = "multiple-testing-v1"

# The family-wise error rate the correction targets. Reused from X1's alpha so
# "significant" means one thing across the release gate.
MT_TARGET_FWER = OOS_ALPHA

# THE FAMILY SIZE IS COUNTED FROM WHAT WAS RUN, NEVER FROM THE REGISTRY.
# MEASURED, the registry holds 1 distinct trial against 8 trained estimators, so
# trusting it would understate the correction by 8x.
MT_COUNTS_OBSERVED_RUNS = True

# AND THE DISCREPANCY IS REPORTED, not silently repaired. A registry that
# undercounts is itself a governance finding: it means experiments are being run
# outside the mechanism built to track them.
MT_REPORTS_REGISTRY_GAP = True

# Both corrections are required. The permutation test DECIDES because it handles
# correlation; Bonferroni is a floor that needs no resampling and cannot be
# gamed by a badly-seeded shuffle.
MT_METHOD_BONFERRONI = "bonferroni"
MT_METHOD_PERMUTATION = "max_statistic_permutation"
MT_METHODS: tuple[str, ...] = (MT_METHOD_BONFERRONI, MT_METHOD_PERMUTATION)
MT_DECIDING_METHOD = MT_METHOD_PERMUTATION
MT_REQUIRES_ALL_METHODS = True

# Permutation count. A CLAIM I CHECKED AND CORRECTED: I assumed fewer shuffles
# would be too coarse to resolve 0.05. MEASURED, that is false - 100 shuffles
# already resolve it (smallest non-zero p 0.01), and the FWER is IDENTICAL at
# 100, 200 and 400 shuffles (2.67% in all three).
#
# The real reason for more shuffles is the PRECISION of the reported p-value,
# not the pass/fail decision: at 100 shuffles a p-value is quantised to 0.01
# steps, so a reported 0.02 could be anything from 0.015 to 0.025. 400 gives
# 0.0025 steps, which is fine enough that the number in the report means what
# it says.
MT_PERMUTATIONS = 400
MT_MIN_PERMUTATIONS = 100

# A FAMILY OF ONE STILL NEEDS NO CORRECTION, and saying so is honest rather than
# vacuous - but a family of one that arose because seven runs went unregistered
# is a different thing, which is why the gap is reported separately.
MT_MIN_FAMILY_FOR_CORRECTION = 2

# Verdicts.
MT_SURVIVES = "SURVIVES_CORRECTION"
MT_FAILS = "FAILS_CORRECTION"
MT_NOT_EVALUATED = "NOT_EVALUATED"
MT_VERDICTS: tuple[str, ...] = (
    MT_NOT_EVALUATED,
    MT_FAILS,
    MT_SURVIVES,
)

# X7 reports; the registry promotes.
MT_BLOCKS_TRADES = False


def _validate_multiple_testing_config() -> None:
    """Import-time guard for the X7 contract."""
    if MT_TARGET_FWER != OOS_ALPHA:
        raise ValueError(
            "the family-wise target must be X1's alpha, or a result could "
            "survive correction at a looser standard than it was tested at"
        )
    if not 0.0 < MT_TARGET_FWER < 1.0:
        raise ValueError(
            f"MT_TARGET_FWER must lie in (0, 1), got {MT_TARGET_FWER!r}"
        )
    if not MT_COUNTS_OBSERVED_RUNS:
        raise ValueError(
            "the family size must be counted from the runs that were actually "
            "executed: MEASURED, the trial registry holds 1 distinct trial "
            "against 8 trained estimators, so trusting it understates the "
            "correction by 8x and reports 5% risk where the answer is 33.7%"
        )
    if not MT_REPORTS_REGISTRY_GAP:
        raise ValueError(
            "a registry that undercounts the search is itself a governance "
            "finding and must be reported, not silently repaired"
        )
    if len(set(MT_METHODS)) != len(MT_METHODS):
        raise ValueError("duplicate correction method")
    if MT_DECIDING_METHOD not in MT_METHODS:
        raise ValueError("the deciding method must be one of the applied ones")
    if MT_DECIDING_METHOD != MT_METHOD_PERMUTATION:
        raise ValueError(
            "the permutation test must decide: MEASURED, Bonferroni lands at "
            "3.73% against a 5% target on correlated estimators because it "
            "assumes independence, while the max-statistic permutation test "
            "tracks the target at 5.50%"
        )
    if not MT_REQUIRES_ALL_METHODS:
        raise ValueError(
            "both corrections are required: the permutation test handles "
            "correlation, and Bonferroni is a floor that needs no resampling "
            "and cannot be gamed by a badly-seeded shuffle"
        )
    if MT_PERMUTATIONS < MT_MIN_PERMUTATIONS:
        raise ValueError(
            f"{MT_PERMUTATIONS} shuffles is below the {MT_MIN_PERMUTATIONS} "
            f"floor"
        )
    if MT_MIN_PERMUTATIONS * MT_TARGET_FWER < 1:
        raise ValueError(
            f"at {MT_MIN_PERMUTATIONS} shuffles the smallest non-zero p-value "
            f"is {1 / MT_MIN_PERMUTATIONS}, which cannot express a result at "
            f"{MT_TARGET_FWER}"
        )
    if MT_MIN_FAMILY_FOR_CORRECTION < 2:
        raise ValueError(
            "a family of one has no multiplicity to correct for"
        )
    if len(set(MT_VERDICTS)) != len(MT_VERDICTS):
        raise ValueError("duplicate multiple-testing verdict")
    if MT_FAILS == MT_NOT_EVALUATED:
        raise ValueError(
            "'it was corrected and did not survive' and 'no correction could "
            "be computed' are different answers"
        )
    if MT_BLOCKS_TRADES:
        raise ValueError("X7 reports; the registry promotes")


_validate_multiple_testing_config()


# --- X8: sealed holdout ----------------------------------------------------------
# "Maintain an untouched final period. Once opened, do not use it for further
# model selection." MEASURED, the final period IS untouched - and that is an
# accident of arithmetic, not a seal. Nothing records where the seal is, nothing
# records whether it was opened, and the run that respected it CANNOT BE
# REPRODUCED.
#
# THE FIRST MEASUREMENT: THE SEAL IS NOT RECORDED ANYWHERE.
# `build_walk_forward_folds` computes `holdout: [start, end]` and
# `train_baseline` DISCARDS IT. Of the four facts needed to verify a seal, the
# run ledger carries none:
#
#     run records holdout bounds           NO
#     run records dataset row count        NO
#     run records fold geometry            NO
#     fold records absolute row indices    NO
#
# Reconstructing the seal for the 8 shipped runs took the row count from a
# SECOND file (`training_datasets.jsonl`: 406 rows) plus GUESSING the geometry
# from `train_rows`. It resolves to fold=120, embargo=60, holdout=60, which
# reproduces the ledger's single fold exactly:
#
#     dataset rows                    406
#     fold 0   train rows   0..119    validation rows 180..299
#     HOLDOUT               rows 346..405   (60 rows, 14.8% of the data)
#     gap between validation and holdout      46 rows
#
# So the tail was never read. But a seal that can only be recovered by guessing
# the geometry is not auditable, and "we think nothing touched it" is not the
# claim X8 is supposed to support.
#
# THE SECOND MEASUREMENT, AND THE REASON X8 CANNOT PASS: THE SHIPPED RUNS ARE
# UNREPRODUCIBLE. F2 (34bd464, 2026-09-19) added the 252-session horizon. The
# ledger was written 2026-09-18, under a max horizon of 60.
# `build_walk_forward_folds` requires `embargo >= max label horizon`, so the
# geometry that MADE the ledger is now ILLEGAL:
#
#     ledger geometry (406 rows, fold=120, embargo=60, holdout=60)
#         -> REJECTED: embargo (60) must be >= the max label horizon (252)
#     scripts/train.py shipped defaults (fold=80, embargo=60, holdout=60)
#         -> REJECTED for the same reason
#
# `scripts/train.py` cannot regenerate its own ledger, and its defaults were
# never updated when F2 moved the horizon. An immutable holdout whose run cannot
# be re-executed is a record of a measurement nobody can check.
#
# THE THIRD MEASUREMENT: NO LEGAL GEOMETRY FITS THE DATA AT ALL. With the
# embargo pinned at >= 252, every candidate exceeds the 406 rows available:
#
#     fold= 60  embargo=252  holdout= 60  -> needs  432 rows   short by  26
#     fold= 60  embargo=252  holdout=126  -> needs  498 rows   short by  92
#     fold=120  embargo=252  holdout= 60  -> needs  552 rows   short by 146
#     fold=252  embargo=252  holdout=126  -> needs  882 rows   short by 476
#
# The last line is the CURRENT DEFAULT geometry. The 252-session horizon makes a
# walk-forward fit WITH a sealed holdout impossible on the present dataset. The
# honest X8 verdict is therefore NOT_EVALUATED with a named shortfall, not
# SEALED. Reporting SEALED because the tail happens to be untouched would credit
# the system for a property it cannot demonstrate and cannot re-establish.
SEALED_HOLDOUT_VERSION = "sealed-holdout-v1"

# THE SEAL MUST BE RECORDED, not inferred. A holdout whose bounds live only in a
# discarded local variable cannot be audited: MEASURED, recovering it for the
# shipped runs required a second file and a guessed geometry.
HOLDOUT_RECORDS_BOUNDS = True

# ...and the geometry that produced it, because the bounds alone do not say
# whether they were legal. MEASURED, the shipped geometry is illegal TODAY and
# nothing on disk revealed which geometry was used.
HOLDOUT_RECORDS_GEOMETRY = True

# OPENING THE HOLDOUT IS A ONE-WAY EVENT. Once read, the period stops being a
# clean estimate of out-of-sample performance, so the count of openings is part
# of the record. Zero openings is the only state in which a holdout result may
# be quoted as unseen.
HOLDOUT_MAX_OPENINGS = 1

# An opening must name WHO opened it and WHY. "Once opened, do not use it for
# further model selection" is unenforceable if the opening is anonymous.
HOLDOUT_OPENING_REQUIRES_REASON = True

# A holdout read more than once, or read and then used for selection, is
# BURNED: the number it produces is no longer an unseen estimate. This state
# exists so the system can say so rather than quietly continuing to quote it.
HOLDOUT_BURNED_IS_TERMINAL = True

# The embargo between the last validation row and the holdout must still cover
# the longest label horizon, or a label inside the holdout was computed from
# prices the final fold trained on. MEASURED, the shipped gap is 46 rows
# against a 252-session horizon - the seal leaks by 206 sessions even though the
# tail itself was never read.
HOLDOUT_REQUIRES_EMBARGO = True

# Verdicts, weakest to strongest. The order IS the precedence.
HOLDOUT_NOT_EVALUATED = "NOT_EVALUATED"   # the seal cannot be established
HOLDOUT_BURNED = "BURNED"                 # opened twice, or used for selection
HOLDOUT_OPENED = "OPENED"                 # read once, legitimately, now spent
HOLDOUT_SEALED = "SEALED"                 # recorded, embargoed, never read
HOLDOUT_VERDICTS: tuple[str, ...] = (
    HOLDOUT_NOT_EVALUATED,
    HOLDOUT_BURNED,
    HOLDOUT_OPENED,
    HOLDOUT_SEALED,
)

# X8 reports; the registry promotes. Consistent with X1-X7.
HOLDOUT_BLOCKS_TRADES = False


def _validate_sealed_holdout_config() -> None:
    """Import-time guard for the X8 contract."""
    if not HOLDOUT_RECORDS_BOUNDS:
        raise ValueError(
            "the holdout bounds must be recorded: MEASURED, the 8 shipped runs "
            "carry none of the four facts needed to verify the seal, and "
            "recovering it required a second file plus a guessed geometry"
        )
    if not HOLDOUT_RECORDS_GEOMETRY:
        raise ValueError(
            "the geometry must be recorded alongside the bounds: MEASURED, the "
            "geometry that produced the ledger is ILLEGAL today (embargo 60 "
            "against a 252-session horizon) and nothing on disk revealed it"
        )
    if HOLDOUT_MAX_OPENINGS != 1:
        raise ValueError(
            f"a sealed holdout may be opened exactly once, got "
            f"{HOLDOUT_MAX_OPENINGS}; a period read twice is no longer an "
            f"unseen estimate of anything"
        )
    if not HOLDOUT_OPENING_REQUIRES_REASON:
        raise ValueError(
            "'once opened, do not use it for further model selection' is "
            "unenforceable if the opening is anonymous"
        )
    if not HOLDOUT_BURNED_IS_TERMINAL:
        raise ValueError(
            "a burned holdout cannot be un-burned: the number it produces has "
            "already informed selection and is no longer out-of-sample"
        )
    if not HOLDOUT_REQUIRES_EMBARGO:
        raise ValueError(
            "the holdout needs an embargo covering the longest label horizon, "
            "or a label inside it was computed from prices the last fold "
            "trained on: MEASURED, the shipped gap is 46 rows against 252"
        )
    if len(set(HOLDOUT_VERDICTS)) != len(HOLDOUT_VERDICTS):
        raise ValueError("duplicate holdout verdict")
    if HOLDOUT_VERDICTS[0] != HOLDOUT_NOT_EVALUATED:
        raise ValueError("NOT_EVALUATED must be the weakest verdict")
    if HOLDOUT_VERDICTS[-1] != HOLDOUT_SEALED:
        raise ValueError("SEALED must be the strongest verdict")
    if HOLDOUT_BURNED == HOLDOUT_NOT_EVALUATED:
        raise ValueError(
            "'the seal was broken' and 'the seal could not be established' are "
            "different facts: the first is a spent holdout, the second is an "
            "unverifiable one, and MEASURED the shipped runs are the second"
        )
    if HOLDOUT_SEALED == HOLDOUT_OPENED:
        raise ValueError(
            "a holdout that has been read once is spent, not sealed"
        )
    if HOLDOUT_BLOCKS_TRADES:
        raise ValueError("X8 reports; the registry promotes")


_validate_sealed_holdout_config()


# --- X9: release snapshot --------------------------------------------------------
# "Freeze code, data, features, models, weights, calibration, configuration,
# validation results." MEASURED, every component is present and deterministically
# hashable - except the one the whole snapshot rests on. THE RECORDED CODE
# IDENTITY CANNOT TELL A FROZEN TREE FROM A THAWED ONE.
#
# THE DECIDING MEASUREMENT. `_code_commit()` (core/backtest/manifest.py) returns
# `git rev-parse --short HEAD`. Appending a line to `core/config.py` and asking
# again:
#
#     commit before the edit    ea29e33
#     commit after the edit     ea29e33
#     identical                 True      <- the code changed, the identity did not
#     tree dirty                True
#
# A snapshot recording `ea29e33` is claiming a reproducibility it cannot deliver:
# the code that ran was not the code at that commit, and nothing in the record
# says so. Every other component is fine - 1050 config constants, 40 registered
# features, 8 training runs with artifact hashes, versioned ensemble weights and
# the calibration modules all hash deterministically and repeatably - so the code
# digest is the single point where a freeze silently stops being a freeze.
#
# AN OBVIOUS FIX THAT DOES NOT WORK, CHECKED BEFORE USING IT. `git ls-files -s`
# looks like a cheap tracked-content digest (339 files, 18 ms). It reports the
# STAGED blob hashes, so an UNSTAGED edit leaves it unchanged:
#
#     tracked-content digest              cd1aa1f3...
#     after an unstaged edit              cd1aa1f3...   changed: False
#
# It is dirt-blind in exactly the same way as the commit hash, and would have
# shipped the same defect wearing a digest's clothes.
#
# WHAT DOES WORK is hashing the WORKTREE contents via `git hash-object`:
#
#     clean tree              e7b84424...
#     after an edit           d5a911af...   changed: True
#     after reverting         e7b84424...   equals clean: True
#
# Stable when clean, moves on any edit, and returns to exactly the prior value on
# revert. Cost 0.4 s over 339 files, which a release snapshot can afford because
# it is taken once per release, not per forecast.
#
# SO A SNAPSHOT RECORDS BOTH: the commit for provenance (where this came from)
# and the worktree digest for integrity (whether it is still that). They answer
# different questions, and MEASURED the commit alone answers neither reliably.
RELEASE_SNAPSHOT_VERSION = "release-snapshot-v1"

# The components a release snapshot must freeze. Named as data so the gate and
# the builder cannot drift apart about what "complete" means, and so a component
# added later cannot be silently omitted.
RELEASE_COMPONENTS: tuple[str, ...] = (
    "code",
    "configuration",
    "features",
    "models",
    "weights",
    "calibration",
    "data",
    "validation",
)

# THE WORKTREE DIGEST IS REQUIRED, not just the commit. MEASURED, the commit hash
# is identical before and after an edit, so a snapshot carrying only a commit
# cannot detect that the frozen tree has thawed.
RELEASE_REQUIRES_WORKTREE_DIGEST = True

# ...and the commit is still recorded, because the digest says WHETHER the tree
# changed while the commit says WHERE it came from. Neither substitutes for the
# other.
RELEASE_RECORDS_COMMIT = True

# A snapshot taken from a DIRTY tree is not a release. It may still be recorded -
# refusing to describe reality would be worse - but it is marked, because its
# digest matches no commit anyone else can check out.
RELEASE_DIRTY_IS_NOT_RELEASABLE = True

# Every component must state PRESENT or ABSENT explicitly. A component omitted
# from the manifest is indistinguishable from one that was checked and found
# missing, and the second is a release blocker while the first is a bug in the
# snapshot builder.
RELEASE_REQUIRES_EXPLICIT_ABSENCE = True

# Verdicts, weakest to strongest. The order IS the precedence.
RELEASE_NOT_EVALUATED = "NOT_EVALUATED"   # the snapshot could not be taken
RELEASE_INCOMPLETE = "INCOMPLETE"         # a required component is absent
RELEASE_DIRTY = "DIRTY"                   # complete, but the tree is not frozen
RELEASE_FROZEN = "FROZEN"                 # complete, clean, digested
RELEASE_SNAPSHOT_VERDICTS: tuple[str, ...] = (
    RELEASE_NOT_EVALUATED,
    RELEASE_INCOMPLETE,
    RELEASE_DIRTY,
    RELEASE_FROZEN,
)

# X9 reports; the registry promotes. Consistent with X1-X8.
RELEASE_BLOCKS_TRADES = False


def _validate_release_snapshot_config() -> None:
    """Import-time guard for the X9 contract."""
    if not RELEASE_REQUIRES_WORKTREE_DIGEST:
        raise ValueError(
            "a release snapshot must record a worktree digest: MEASURED, "
            "`git rev-parse HEAD` returns the SAME commit before and after an "
            "edit, so a snapshot carrying only a commit cannot detect that the "
            "tree it claims to freeze has changed"
        )
    if not RELEASE_RECORDS_COMMIT:
        raise ValueError(
            "the commit is still required: the digest says WHETHER the tree "
            "changed, the commit says WHERE it came from, and neither answers "
            "the other's question"
        )
    if not RELEASE_DIRTY_IS_NOT_RELEASABLE:
        raise ValueError(
            "a snapshot taken from a dirty tree is not releasable: its digest "
            "matches no commit anyone else can check out"
        )
    if not RELEASE_REQUIRES_EXPLICIT_ABSENCE:
        raise ValueError(
            "every component must state PRESENT or ABSENT: an omitted component "
            "is indistinguishable from one found missing, and only the second "
            "is a release blocker"
        )
    if len(set(RELEASE_COMPONENTS)) != len(RELEASE_COMPONENTS):
        raise ValueError("duplicate snapshot component")
    for required in ("code", "configuration", "models", "validation"):
        if required not in RELEASE_COMPONENTS:
            raise ValueError(
                f"{required!r} is part of what X9 freezes and cannot be dropped "
                f"from the component list"
            )
    if len(set(RELEASE_SNAPSHOT_VERDICTS)) != len(RELEASE_SNAPSHOT_VERDICTS):
        raise ValueError("duplicate release-snapshot verdict")
    if RELEASE_SNAPSHOT_VERDICTS[0] != RELEASE_NOT_EVALUATED:
        raise ValueError("NOT_EVALUATED must be the weakest verdict")
    if RELEASE_SNAPSHOT_VERDICTS[-1] != RELEASE_FROZEN:
        raise ValueError("FROZEN must be the strongest verdict")
    if RELEASE_DIRTY == RELEASE_FROZEN:
        raise ValueError(
            "'complete but unfrozen' and 'frozen' are different states: "
            "MEASURED, the commit hash cannot tell them apart, which is the "
            "defect X9 exists to close"
        )
    if RELEASE_INCOMPLETE == RELEASE_NOT_EVALUATED:
        raise ValueError(
            "'a component is missing' and 'no snapshot could be taken' are "
            "different facts with different fixes"
        )
    if RELEASE_BLOCKS_TRADES:
        raise ValueError("X9 reports; the registry promotes")


_validate_release_snapshot_config()


# --- X10: honest gate report -----------------------------------------------------
# "If evidence is insufficient, report FORECASTING RELEASE: NOT APPROVED with the
# reason and the next action. 'Gate not met' is a valid, successful engineering
# outcome."
#
# THE DECIDING MEASUREMENT. Running X1-X9 against the shipped data:
#
#     X1  OOS validation        NOT_APPROVED
#     X2  calibration           NOT_EVALUATED    IN_SAMPLE_ONLY
#     X3  regime robustness     NOT_EVALUATED    NO_REGIME_LABELS
#     X4  event robustness      NOT_EVALUATED    NO_EVENT_ATTRIBUTION
#     X5  feature ablation      NOT_EVALUATED
#     X6  temporal robustness   NOT_EVALUATED    HORIZONS_MISSING
#     X7  multiple testing      FAILS_CORRECTION
#     X8  sealed holdout        NOT_EVALUATED    NO_RECORDED_BOUNDS
#     X9  release snapshot      FROZEN
#
# SIX OF NINE GATES COULD NOT RUN AT ALL. One pass-like verdict exists (X9
# FROZEN), and it says only that the tree is frozen - not that anything in it
# works. So the release verdict is NOT_APPROVED, and X10's job is to say so in a
# way that cannot be misread as "nearly there".
#
# THE TRAP X10 EXISTS TO CLOSE. Only TWO gates carry an explicit failure verdict.
# A reading of "no FAIL means ship it" would approve SEVEN OF NINE:
#
#     gates with an explicit failure verdict            2 of 9
#     gates a 'no FAIL = pass' rule would approve       7 of 9   (78%)
#     gates that could not run at all                   6 of 9
#
# 78% GREEN ON A SYSTEM WHERE TWO THIRDS OF THE EVIDENCE DOES NOT EXIST. This is
# the NOT_EVALUATED-as-pass error the whole project treats as load-bearing, and at
# the release gate it is the most expensive place to make it. So X10 counts
# NOT_EVALUATED as a BLOCKER, never as a pass, and requires every gate to reach a
# genuine pass before approval.
#
# THE SECOND FINDING: FIVE OF THE SIX BLOCKERS ARE ONE DEFECT. The training fold
# is a thin record. MEASURED, it carries exactly these keys:
#
#     actuals, fold_id, metrics, predictions, train_end_time, train_rows,
#     validation_rows, validation_start_time
#
# and none of what the gates need:
#
#     regime labels (X3)              'regimes'      absent
#     event ids (X4)                  'events'       absent
#     sources (X4)                    'sources'      absent
#     absolute fold indices (X8)      'validation'   absent
#     per-feature-set scores (X5)     'ablation'     absent
#
# X6 is the same defect at run level: all 8 runs train the SINGLE horizon '20d',
# so there is no horizon spread to compare. Only X2's blocker is different in
# kind - it needs a holdout split, which X8 measured is not currently
# constructible on 406 rows.
#
# So X10 reports the RICHER FOLD RECORD as the common prerequisite AND each gate's
# own blocker. A report naming one fix would understate the work; one naming six
# independent fixes would overstate it.
#
# X10 IS A REPORT, NOT A PROMOTION. It states what the evidence supports. Nothing
# here blocks a trade, because nothing in this system trades - the execution path
# does not exist, which is itself part of the honest answer.
HONEST_GATE_VERSION = "honest-gate-v1"

# The gates a forecasting release must clear, in roadmap order. Named as data so a
# gate cannot be quietly dropped from the release criteria: MEASURED, the CI gate
# list fell eight behind when it was maintained by hand (see the workflow note).
RELEASE_GATES: tuple[str, ...] = (
    "X1_oos_validation",
    "X2_calibration",
    "X3_regime_robustness",
    "X4_event_robustness",
    "X5_feature_ablation",
    "X6_temporal_robustness",
    "X7_multiple_testing",
    "X8_sealed_holdout",
    "X9_release_snapshot",
)

# NOT_EVALUATED IS A BLOCKER, NOT A PASS. The single most important line in X10.
# MEASURED, treating it as a pass would approve 7 of 9 gates on a system where 6
# could not run.
GATE_UNEVALUATED_BLOCKS_RELEASE = True

# Every gate must reach a genuine pass. No quorum, no weighting, no "most of them
# are fine": a release gate that can be satisfied by a majority is not a gate.
GATE_REQUIRES_ALL = True

# A NOT_APPROVED report must name, per blocking gate, WHY it blocks and WHAT WOULD
# SETTLE IT. "Not approved" without a next action is a verdict nobody can act on,
# and X10's whole point is that a negative result is a useful one.
GATE_REQUIRES_NEXT_ACTION = True

# ...and the report must state the COMMON prerequisite where one exists. MEASURED,
# five of six blockers are the same thin-fold defect, so six independent action
# items would overstate the work by 5x.
GATE_REPORTS_COMMON_BLOCKER = True

# "Gate not met" is a SUCCESSFUL outcome of the reporting machinery. The script
# exits 0 having correctly reported NOT_APPROVED; it exits non-zero only when the
# report itself is malformed. Conflating "the release is not approved" with "the
# check crashed" would make an honest negative indistinguishable from a bug.
GATE_NOT_MET_IS_A_VALID_OUTCOME = True

# Verdicts. Only two, deliberately: a release is approved or it is not, and every
# gradation ("nearly", "provisionally") is a place for a negative to be read as a
# positive.
RELEASE_APPROVED = "APPROVED"
RELEASE_NOT_APPROVED = "NOT_APPROVED"
HONEST_GATE_VERDICTS: tuple[str, ...] = (
    RELEASE_NOT_APPROVED,
    RELEASE_APPROVED,
)

# Per-gate states in the report. PASSED / BLOCKED / UNEVALUATED are distinct
# because they need different work: BLOCKED means the gate ran and said no,
# UNEVALUATED means it could not run, and the second is the shipped majority.
GATE_PASSED = "PASSED"
GATE_BLOCKED = "BLOCKED"
GATE_UNEVALUATED = "UNEVALUATED"
GATE_STATES: tuple[str, ...] = (GATE_UNEVALUATED, GATE_BLOCKED, GATE_PASSED)

# X10 reports; the registry promotes. Consistent with X1-X9.
HONEST_GATE_BLOCKS_TRADES = False


def _validate_honest_gate_config() -> None:
    """Import-time guard for the X10 contract."""
    if not GATE_UNEVALUATED_BLOCKS_RELEASE:
        raise ValueError(
            "NOT_EVALUATED must block a release: MEASURED, only 2 of 9 gates "
            "carry an explicit failure verdict, so counting the unevaluated as "
            "passes would approve 7 of 9 on a system where 6 gates could not "
            "run at all. Absence of evidence is not evidence of safety"
        )
    if not GATE_REQUIRES_ALL:
        raise ValueError(
            "every release gate must pass: a gate satisfiable by a majority of "
            "gates is not a gate"
        )
    if not GATE_REQUIRES_NEXT_ACTION:
        raise ValueError(
            "a NOT_APPROVED report must name what would settle each blocker; "
            "'not approved' with no next action is a verdict nobody can act on"
        )
    if not GATE_REPORTS_COMMON_BLOCKER:
        raise ValueError(
            "the report must name a common prerequisite where one exists: "
            "MEASURED, five of six blockers are the same thin-fold defect, so "
            "six independent action items would overstate the work by 5x"
        )
    if not GATE_NOT_MET_IS_A_VALID_OUTCOME:
        raise ValueError(
            "'gate not met' is a successful engineering outcome: conflating it "
            "with a crash makes an honest negative indistinguishable from a bug"
        )
    if len(set(RELEASE_GATES)) != len(RELEASE_GATES):
        raise ValueError("duplicate release gate")
    if len(RELEASE_GATES) < 9:
        raise ValueError(
            f"only {len(RELEASE_GATES)} release gates are declared; X1-X9 all "
            f"gate a forecasting release and a hand-maintained list is exactly "
            f"what fell eight behind in CI"
        )
    if len(set(HONEST_GATE_VERDICTS)) != len(HONEST_GATE_VERDICTS):
        raise ValueError("duplicate honest-gate verdict")
    if HONEST_GATE_VERDICTS[0] != RELEASE_NOT_APPROVED:
        raise ValueError(
            "NOT_APPROVED must be the weakest verdict, and the default: a "
            "release gate that defaults to approved is not fail-closed"
        )
    if RELEASE_APPROVED == RELEASE_NOT_APPROVED:
        raise ValueError("the two release verdicts must stay distinct")
    if len(set(GATE_STATES)) != len(GATE_STATES):
        raise ValueError("duplicate gate state")
    if GATE_STATES[0] != GATE_UNEVALUATED:
        raise ValueError(
            "UNEVALUATED must be the weakest gate state: it is the shipped "
            "majority and must never outrank a genuine pass"
        )
    if GATE_UNEVALUATED == GATE_BLOCKED:
        raise ValueError(
            "'the gate ran and said no' and 'the gate could not run' are "
            "different facts needing different work, and MEASURED six of nine "
            "gates are the second"
        )
    if GATE_PASSED in (GATE_BLOCKED, GATE_UNEVALUATED):
        raise ValueError("a pass must be distinguishable from a non-pass")
    if HONEST_GATE_BLOCKS_TRADES:
        raise ValueError("X10 reports; the registry promotes")


_validate_honest_gate_config()


# --- B1: the paper engine is wired -----------------------------------------------
# MEASURED, `core/paper_engine.py` was 259 tested lines imported by its own test
# and NOTHING ELSE, while the system published a governance mode meaning
# "paper-trading ready". The modes already agreed - the orchestrator emits
# mode == "PAPER" and the engine gates on exactly that - so the only thing missing
# was the call.
#
# A flag rather than an unconditional call, because a paper intent is a SIDE
# EFFECT of a research decision: an operator replaying history to inspect a score
# should be able to do so without appending to the order log.
PAPER_ENGINE_WIRED = True


def _validate_paper_engine_wiring() -> None:
    """Import-time guard for the B1 contract."""
    if PAPER_TRADABLE_MODE != "PAPER":
        raise ValueError(
            f"the paper engine gates on mode {PAPER_TRADABLE_MODE!r} while the "
            f"orchestrator emits 'PAPER'; a mismatch here means the engine is "
            f"wired but can never accept an intent"
        )
    if not PAPER_ENGINE_WIRED:
        raise ValueError(
            "the paper engine must stay wired: an unwired engine makes the "
            "published PAPER posture a claim nothing can act on, which is the "
            "defect B1 closed"
        )


_validate_paper_engine_wiring()


# --- B2: the source -> outcome join ----------------------------------------------
# Open item 4 parked this as unbuildable: "0 of 2650 stored articles carry a
# ticker, so no article joins to a price outcome."
#
# THAT READING WAS INCOMPLETE. Re-measuring the tracked raw store:
#
#     articles total                3,753
#     with an article-level ticker       0   <- the parked measurement, correct
#     with a request_key ticker     3,753   <- and every one is attributable
#     distinct outlets                 58
#
# Every raw news envelope carries a `request_key` of the form `MSFT_2026-09-17`
# naming the ticker the fetch was made FOR. So the join is buildable from data
# already tracked, with no re-ingestion.
#
# The ticker is INFERRED FROM THE REQUEST, which is weaker than resolving the
# entity from the article text. That weakness is recorded on every observation
# (`attribution`) rather than hidden, so a consumer can tell the two apart and
# discount accordingly.
SOURCE_OUTCOME_VERSION = "source-outcome-join-v1"

# The smallest absolute tone that counts as a DIRECTIONAL CLAIM.
#
# An article inside this band makes no claim about direction, so it cannot be
# right or wrong about one and is DROPPED rather than scored. Scoring it would
# make every neutral article a miss against any non-zero move, which penalises an
# outlet for the system's own missing classification - the same rule
# `observation_records` already applies to a missing outlet.
SOURCE_OUTCOME_MIN_TONE_ABS = 0.05

# The attribution recorded when a ticker came from the fetch rather than from the
# article text. Named in config so the weaker provenance cannot be silently
# relabelled as the stronger one.
SOURCE_OUTCOME_ATTRIBUTION_REQUEST_KEY = "request_key"

# A HIT IS AGREEMENT, NOT CAUSATION. The join reports the forward return that
# FOLLOWED publication - an event-associated return (E5) - and a hit means the
# outlet's stated tone agreed with the direction the price then took. Nothing here
# claims the article moved the price.
SOURCE_OUTCOME_IS_ASSOCIATION_ONLY = True


def _validate_source_outcome_config() -> None:
    """Import-time guard for the B2 contract."""
    if not 0.0 < SOURCE_OUTCOME_MIN_TONE_ABS < 1.0:
        raise ValueError(
            "the neutral tone band must lie inside (0, 1): it separates an "
            "article that made a directional claim from one that did not"
        )
    if not SOURCE_OUTCOME_IS_ASSOCIATION_ONLY:
        raise ValueError(
            "the join reports an event-ASSOCIATED return, never a causal one: "
            "an article that preceded a move did not thereby cause it, and E5 "
            "owns the attribution methodology that could say otherwise"
        )
    if SOURCE_OUTCOME_ATTRIBUTION_REQUEST_KEY == "entity_resolution":
        raise ValueError(
            "a ticker inferred from the fetch must not be labelled as resolved "
            "from the article text; the two have different trustworthiness and "
            "a consumer must be able to tell them apart"
        )


_validate_source_outcome_config()


# --- B4: cluster exposure --------------------------------------------------------
# R1 flags exposure per POSITION. A CLUSTER of correlated holdings is a different
# question, and open item 5 parked it because "defining a cluster needs its own
# evidence - by sector, by correlation clustering, or by factor loading - and each
# choice is a measurement, not a preference."
#
# THE MEASUREMENT, on 3,771 sessions of real returns. A book of 10% each in
# NVDA/AMD/AVGO/SOXX plus 20% each in MSFT/GOOGL/CAT raises ZERO R1 flags:
#
#     AMD 16.3%  GOOGL 16.1%  MSFT 15.8%  CAT 15.5%  NVDA 14.1%  AVGO/SOXX 11.1%
#
# while the four semiconductors TOGETHER hold 52.6% of portfolio variance at a
# mean pairwise correlation of 0.62. Sector labels do not catch it either: SOXX is
# an ETF, and the correlation binding these four is not a label.
#
# WHY AVERAGE LINKAGE. Both rules swept across 13 thresholds:
#
#     thr    single   average
#     0.450   100.0%    84.5%
#     0.500   100.0%    52.6%   <- average finds exactly the four semis
#     0.550   100.0%    36.3%
#     0.600    52.6%    36.3%   <- single finds them only here
#     0.650    36.3%    36.3%
#
# A FIRST READING OF THIS WAS WRONG. I took average linkage to be the more STABLE
# of the two, but the spreads are nearly identical (63.7% vs 62.3%). The real
# discriminator is DEGENERACY: single linkage merges on ONE qualifying pair, and
# SOXX correlates 0.55-0.76 with everything while MSFT-GOOGL is 0.59, so it chains
# the semis to the non-semis and reports the WHOLE BOOK as one cluster at 5 of 13
# thresholds. A cluster containing every holding cannot distinguish a concentrated
# book from a diversified one. Average linkage degenerated at 0 of 13.
CLUSTER_EXPOSURE_VERSION = "cluster-exposure-v1"

# Average linkage: a merge requires the MEAN cross-correlation to clear the bar,
# not a single qualifying pair.
CLUSTER_EXPOSURE_LINKAGE = "average"

# The correlation a group must average to count as one bet. 0.50 under average
# linkage recovers exactly the four semis on the measured book; 0.55 splits AMD
# out and 0.45 pulls in a fifth name at 84.5%.
CLUSTER_EXPOSURE_MIN_CORRELATION = 0.50

# A CLUSTER OF ONE IS A POSITION, and R1 already flags those. Reporting it here
# would double-count the same concentration under a second name.
CLUSTER_EXPOSURE_MIN_MEMBERS = 2

# The share of portfolio variance in one cluster that warrants review. Set at the
# same level R1 uses for a single position, because the point of B4 is that a
# basket acting as one bet deserves the same scrutiny as one holding of that size.
CLUSTER_EXPOSURE_RISK_REVIEW = EXPOSURE_RISK_REVIEW

# Verdicts, weakest to strongest. The order IS the precedence.
CLUSTER_NOT_EVALUATED = "NOT_EVALUATED"   # no cluster could be located
CLUSTER_CONCENTRATED = "CONCENTRATED"     # one cluster is above the review level
CLUSTER_DIFFUSE = "DIFFUSE"               # no cluster carries that much risk
CLUSTER_VERDICTS: tuple[str, ...] = (
    CLUSTER_NOT_EVALUATED,
    CLUSTER_CONCENTRATED,
    CLUSTER_DIFFUSE,
)

# B4 reports; the risk gate blocks. Consistent with R1.
CLUSTER_EXPOSURE_BLOCKS_TRADES = False


def _validate_cluster_exposure_config() -> None:
    """Import-time guard for the B4 contract."""
    if CLUSTER_EXPOSURE_LINKAGE != "average":
        raise ValueError(
            "average linkage is required: MEASURED, single linkage reports the "
            "WHOLE BOOK as one cluster at 5 of 13 swept thresholds because a "
            "broadly-correlated ETF chains unrelated holdings together, and a "
            "cluster containing every holding cannot distinguish a concentrated "
            "book from a diversified one"
        )
    if not -1.0 < CLUSTER_EXPOSURE_MIN_CORRELATION < 1.0:
        raise ValueError("the correlation threshold must lie inside (-1, 1)")
    if CLUSTER_EXPOSURE_MIN_MEMBERS < 2:
        raise ValueError(
            "a cluster of one is a POSITION and R1 already flags it; reporting "
            "it here would double-count the same concentration"
        )
    if not 0.0 < CLUSTER_EXPOSURE_RISK_REVIEW < 1.0:
        raise ValueError("the risk review level is a fraction inside (0, 1)")
    if len(set(CLUSTER_VERDICTS)) != len(CLUSTER_VERDICTS):
        raise ValueError("duplicate cluster verdict")
    if CLUSTER_VERDICTS[0] != CLUSTER_NOT_EVALUATED:
        raise ValueError(
            "NOT_EVALUATED must be the weakest verdict: a portfolio whose "
            "covariance is unavailable has an UNKNOWN cluster exposure, and "
            "reporting 0% would make the least-understood book look the safest"
        )
    if CLUSTER_CONCENTRATED == CLUSTER_DIFFUSE:
        raise ValueError("the two measured verdicts must stay distinct")
    if CLUSTER_EXPOSURE_BLOCKS_TRADES:
        raise ValueError("B4 reports; the risk gate blocks")


_validate_cluster_exposure_config()
