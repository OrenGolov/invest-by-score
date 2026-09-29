"""Configuration part 7: A1-A7 the alert family.

Split out of the single 9,783-line `core/config.py` in C1. The text is
UNCHANGED — the comments are 38% of the file and carry the measurement
that justifies each rule, which is this project's best documentation.

CHAINED from `_portfolio` rather than standing alone: MEASURED, 88
constants are read across part boundaries, so the parts must reproduce
ONE flat namespace in the original order. The star import is what keeps
`from core.config import ANYTHING` working unchanged.
"""

from core.config._portfolio import *  # noqa: F401,F403


# --- A1: forecast change alert ---------------------------------------------------
# "Alert when the forecast materially changes." Two obvious implementations
# both fire on things that are not changes, and both were measured rather than
# argued.
#
# THE FIRST TRAP - DIGEST DIFFING. A snapshot is digest-addressable, so the
# tempting test is "did the digest change?". MEASURED on four consecutive days
# of an identical, entirely-unavailable forecast: FOUR DISTINCT DIGESTS. The
# digest covers as_of, so it changes every day by construction. A digest-diff
# alert fires on 100% of days while the forecast never changes at all.
#
# THE SECOND TRAP - COALESCING NULLS. MEASURED: comparing values through
# `float(x or 0)` turns a forecast APPEARING (REFUSED -> 0.56) into a +0.56
# move, and a forecast DISAPPEARING (0.56 -> REFUSED) into a -0.56 crash.
# Nothing fell. The availability changed, which is a different event and one a
# reader acts on differently.
#
# So A1 compares AVAILABILITY first and MAGNITUDE only between two values that
# both exist.
FORECAST_ALERT_VERSION = "forecast-change-alert-v1"

# Change kinds. APPEARED and DISAPPEARED are first-class rather than being
# folded into a magnitude, because that is precisely the fold that produces a
# phantom crash to zero.
ALERT_CHANGE_NONE = "NONE"
ALERT_CHANGE_MOVED = "MOVED"              # both present, magnitude crossed
ALERT_CHANGE_APPEARED = "APPEARED"        # was unavailable, now measured
ALERT_CHANGE_DISAPPEARED = "DISAPPEARED"  # was measured, now unavailable
ALERT_CHANGE_NOT_EVALUATED = "NOT_EVALUATED"  # no prior to compare against
FORECAST_CHANGE_KINDS: tuple[str, ...] = (
    ALERT_CHANGE_NOT_EVALUATED,
    ALERT_CHANGE_NONE,
    ALERT_CHANGE_MOVED,
    ALERT_CHANGE_APPEARED,
    ALERT_CHANGE_DISAPPEARED,
)

# The P(up) move that counts as material. DERIVED from sampling noise rather
# than chosen: at F4's point-estimate floor of 40 observations, the 95% band on
# a base rate near 0.5 is +/-0.155. A threshold below that alerts on
# resampling. At n=100 the band is +/-0.098, so 0.10 is the point where a move
# starts to mean something for a realistically-sized cell.
FORECAST_ALERT_MIN_PROBABILITY_MOVE = 0.10

# Availability changes are ALWAYS material. A forecast arriving or vanishing is
# news about the system regardless of magnitude, and it cannot be compared on
# magnitude at all - which is the trap above.
FORECAST_ALERT_AVAILABILITY_IS_MATERIAL = True

# A1 NEVER COMPARES DIGESTS. MEASURED, that fires every day on an unchanged
# forecast, because the digest covers as_of by design.
FORECAST_ALERT_USES_DIGEST = False

# Severities, reused from W2's vocabulary so severity means one thing system
# wide.
ALERT_SEVERITY_INFO = "info"
ALERT_SEVERITY_WARN = "warn"
ALERT_SEVERITIES: tuple[str, ...] = (ALERT_SEVERITY_INFO, ALERT_SEVERITY_WARN)

# An alert reports; it does not trade and it does not override governance. A7
# adds suppression and the risk/governance gate on top of this.
FORECAST_ALERT_BLOCKS_TRADES = False


def _validate_forecast_alert_config() -> None:
    """Import-time guard for the A1 contract."""
    if FORECAST_ALERT_USES_DIGEST:
        raise ValueError(
            "A1 must not compare snapshot digests: MEASURED, four "
            "consecutive days of an identical unavailable forecast produced "
            "four distinct digests, because the digest covers as_of"
        )
    if not FORECAST_ALERT_AVAILABILITY_IS_MATERIAL:
        raise ValueError(
            "a forecast appearing or disappearing is always material: "
            "MEASURED, folding it into a magnitude turns a vanished forecast "
            "into a -0.56 crash that never happened"
        )
    if not 0.0 < FORECAST_ALERT_MIN_PROBABILITY_MOVE < 1.0:
        raise ValueError(
            f"FORECAST_ALERT_MIN_PROBABILITY_MOVE must lie in (0, 1), got "
            f"{FORECAST_ALERT_MIN_PROBABILITY_MOVE!r}"
        )
    if FORECAST_ALERT_MIN_PROBABILITY_MOVE < 0.05:
        raise ValueError(
            f"a {FORECAST_ALERT_MIN_PROBABILITY_MOVE} threshold sits inside "
            f"sampling noise: at 100 observations the 95% band on a base rate "
            f"near 0.5 is +/-0.098, so the alert would fire on resampling"
        )
    if len(set(FORECAST_CHANGE_KINDS)) != len(FORECAST_CHANGE_KINDS):
        raise ValueError("duplicate forecast change kind")
    for required in (ALERT_CHANGE_APPEARED, ALERT_CHANGE_DISAPPEARED):
        if required not in FORECAST_CHANGE_KINDS:
            raise ValueError(
                f"{required!r} must be expressible: an availability change "
                f"folded into a magnitude is a phantom move"
            )
    if ALERT_CHANGE_NONE == ALERT_CHANGE_NOT_EVALUATED:
        raise ValueError(
            "'the forecast did not change' and 'there was nothing to compare "
            "against' are different answers"
        )
    if len(set(ALERT_SEVERITIES)) != len(ALERT_SEVERITIES):
        raise ValueError("duplicate alert severity")
    if FORECAST_ALERT_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_forecast_alert_config()


# --- A2: confidence change alert -------------------------------------------------
# "Alert when confidence changes materially." A2 is NOT A1 with a different
# field, and the difference was measured rather than assumed.
#
# CONFIDENCE MOVES WHEN THE FORECAST DOES NOT. MEASURED: holding P(up) fixed at
# 0.56 and dropping feature completeness from 10/10 to 6/10 moved confidence
# -0.0146 while the forecast value did not move at all. A1 would never fire on
# this, and it should not - the forecast is unchanged. What changed is how much
# it can be trusted.
#
# THE DECIDING MEASUREMENT: A MAGNITUDE-ONLY ALERT IS BLIND HALF THE TIME.
# F7's confidence is bounded by its WEAKEST factor and always reports which one
# bound it. Over 3,000 sampled assessment pairs, 1,170 moved confidence by less
# than 0.10 - below any sane threshold - and in 600 OF THOSE 1,170 (51%) the
# BINDING FACTOR changed completely. A concrete instance: confidence 0.5559 ->
# 0.4640, a -0.0919 move no threshold would fire on, while the binding factor
# went from event_similarity to regime_similarity. The forecast became
# uncertain for an entirely different reason and a magnitude-only alert says
# nothing.
#
# So A2 watches THREE things: the magnitude, the BAND, and the BINDING FACTOR.
CONFIDENCE_ALERT_VERSION = "confidence-change-alert-v1"

# The raw confidence move that counts as material on its own.
CONFIDENCE_ALERT_MIN_MOVE = 0.10

# A BAND CROSSING IS MATERIAL AT ANY MAGNITUDE. MEASURED, the same 0.10 delta
# crosses a band at 0.20->0.30, 0.45->0.55 and 0.70->0.80, but not at
# 0.05->0.15 or 0.85->0.95. The bands are what a reader acts on, so a crossing
# is news even when the number barely moved.
CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL = True

# A CHANGE OF BINDING FACTOR IS MATERIAL AT ANY MAGNITUDE. This is the 51%
# case. "We are unsure because the sample is small" and "we are unsure because
# no comparable regime exists" are different problems with different remedies,
# and a reader who is told only the number cannot tell them apart.
CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL = True

# Change kinds. Availability is first-class for A1's reason: a confidence
# APPEARING or DISAPPEARING is not a move of its own magnitude.
CONF_CHANGE_NONE = "NONE"
CONF_CHANGE_MOVED = "MOVED"                  # magnitude crossed the threshold
CONF_CHANGE_BAND = "BAND_CHANGED"            # crossed a band boundary
CONF_CHANGE_BINDING = "BINDING_CHANGED"      # a different factor now binds
CONF_CHANGE_APPEARED = "APPEARED"
CONF_CHANGE_DISAPPEARED = "DISAPPEARED"
CONF_CHANGE_NOT_EVALUATED = "NOT_EVALUATED"
CONFIDENCE_CHANGE_KINDS: tuple[str, ...] = (
    CONF_CHANGE_NOT_EVALUATED,
    CONF_CHANGE_NONE,
    CONF_CHANGE_MOVED,
    CONF_CHANGE_BAND,
    CONF_CHANGE_BINDING,
    CONF_CHANGE_APPEARED,
    CONF_CHANGE_DISAPPEARED,
)

# UNMEASURABLE IS NOT LOW, inherited from F7 and R4. MEASURED, calibration,
# model_agreement and model_drift are all UNMEASURABLE because no model is
# registered. A factor becoming unmeasurable is not a confidence drop, and a
# factor becoming measurable is not a rise - the SET of things being measured
# changed, which is its own event.
CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT = True

# A2 reports; A7 suppresses and gates. Same rule as A1.
CONFIDENCE_ALERT_BLOCKS_TRADES = False


def _validate_confidence_alert_config() -> None:
    """Import-time guard for the A2 contract."""
    if not 0.0 < CONFIDENCE_ALERT_MIN_MOVE < 1.0:
        raise ValueError(
            f"CONFIDENCE_ALERT_MIN_MOVE must lie in (0, 1), got "
            f"{CONFIDENCE_ALERT_MIN_MOVE!r}"
        )
    if not CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL:
        raise ValueError(
            "a band crossing is material at any magnitude: MEASURED, the same "
            "0.10 delta crosses a band at 0.45->0.55 and does not at "
            "0.85->0.95, and the band is what a reader acts on"
        )
    if not CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL:
        raise ValueError(
            "a change of binding factor is material at any magnitude: "
            "MEASURED, of 1,170 assessment pairs moving confidence by less "
            "than 0.10, 600 (51%) changed which factor bound it. A "
            "magnitude-only alert is silent on all of them"
        )
    if not CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT:
        raise ValueError(
            "a factor becoming UNMEASURABLE is not a confidence drop; F7 "
            "excludes unmeasurable factors rather than scoring them 0.0"
        )
    if len(set(CONFIDENCE_CHANGE_KINDS)) != len(CONFIDENCE_CHANGE_KINDS):
        raise ValueError("duplicate confidence change kind")
    for required in (CONF_CHANGE_BAND, CONF_CHANGE_BINDING):
        if required not in CONFIDENCE_CHANGE_KINDS:
            raise ValueError(
                f"{required!r} must be expressible, or A2 is A1 with a "
                f"different field and is blind to 51% of real changes"
            )
    for required in (CONF_CHANGE_APPEARED, CONF_CHANGE_DISAPPEARED):
        if required not in CONFIDENCE_CHANGE_KINDS:
            raise ValueError(
                f"{required!r} must be expressible: an availability change "
                f"folded into a magnitude is a phantom move"
            )
    if CONF_CHANGE_NONE == CONF_CHANGE_NOT_EVALUATED:
        raise ValueError(
            "'confidence did not change' and 'there was nothing to compare' "
            "are different answers"
        )
    if CONFIDENCE_ALERT_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_confidence_alert_config()


# --- A3: high-impact event alert -------------------------------------------------
# "Alert when a high-impact event is detected." The obvious implementation
# reads the event's own `magnitude` field and fires above a threshold. That is
# exactly what the E-sprint contract forbids, and the reason is structural
# rather than stylistic.
#
# THE FIRST TRAP - ALERTING ON AN UNVALIDATED CLAIM. `magnitude` is documented
# in core/event_contract.py as "a bounded, unitless claim size - NOT an
# expected return", precisely so an unvalidated number is not treated as a
# forecast. MEASURED, it is worse than unvalidated: `magnitude` appears NOWHERE
# in core/event_memory.py or core/event_study.py, and is not among
# EventMemory's fields. It is never carried into memory, so it has never been
# compared against a single realized outcome. An alert keyed on it would fire
# on an assertion nobody has ever checked.
#
# WHAT IS MEASURABLE: THE EVENT TYPE'S REALIZED HISTORY. E6 stores the abnormal
# return per horizon and E5's attribution verdict. MEASURED on 400 seeded
# memories across four event types, the median absolute 20d move differs by
# more than 12x between them:
#
#     event_type        n    median |move|   p90 |move|
#     earnings_beat    94        0.0497        0.0962
#     guidance_cut    114        0.0470        0.1056
#     analyst_note    102        0.0085        0.0183
#     minor_pr         90        0.0040        0.0088
#
# So "high impact" means "this KIND of event has historically moved this name",
# measured from stored outcomes, not claimed in the payload.
EVENT_IMPACT_ALERT_VERSION = "event-impact-alert-v1"

# A3 NEVER READS THE CLAIMED MAGNITUDE. An import-time guard enforces it.
EVENT_IMPACT_USES_CLAIMED_MAGNITUDE = False
EVENT_IMPACT_MAGNITUDE_REASON = (
    "magnitude is a bounded, unitless CLAIM SIZE, not an expected return, and "
    "MEASURED it appears in neither event_memory nor event_study and is not "
    "an EventMemory field - it has never been compared against a realized "
    "outcome, so an alert keyed on it fires on an unchecked assertion"
)

# The horizon whose realized response defines impact. 20d matches the horizon
# E5's attribution verdict is recorded against, so impact and attribution
# describe the same window rather than two.
EVENT_IMPACT_HORIZON = "20d"

# The median absolute abnormal return above which an event type counts as
# high-impact. MEASURED, this separates earnings_beat (0.0497) and
# guidance_cut (0.0470) from analyst_note (0.0085) and minor_pr (0.0040) with
# room on both sides - it is not tuned to the boundary of either group.
EVENT_IMPACT_MIN_MEDIAN_MOVE = 0.025

# The analog floor, REUSED from E6 rather than redefined. MEASURED, a median
# |move| estimate at n=3 spans 0.0128-0.0646 across resamples - a five-fold
# range around a true value of 0.0337 - and tightens to roughly 1.9x at n=20.
# A second floor here would drift from the one E6 already enforces.
EVENT_IMPACT_MIN_ANALOGS = EVENT_MEMORY_MIN_ANALOGS

# IMPACT IS NOT DIRECTION. A large move is high-impact whether it was up or
# down, which is why the measure is the median ABSOLUTE return. A reader needs
# to know a name is about to move before knowing which way, and folding sign
# into the magnitude would let a symmetric history cancel itself to zero.
EVENT_IMPACT_USES_ABSOLUTE_MOVE = True

# Verdicts.
EVENT_IMPACT_HIGH = "HIGH_IMPACT"
EVENT_IMPACT_LOW = "LOW_IMPACT"
EVENT_IMPACT_UNKNOWN = "UNKNOWN_IMPACT"   # too few analogs to say
EVENT_IMPACT_NO_EVENT = "NO_EVENT"
EVENT_IMPACT_VERDICTS: tuple[str, ...] = (
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_HIGH,
)

# UNKNOWN IS NOT LOW. An event type with too little history is not a quiet one;
# it is one nobody can rate. Reporting it as LOW_IMPACT would silence exactly
# the events the system has never seen before, which are the ones most worth
# a human look.
EVENT_IMPACT_UNKNOWN_IS_NOT_LOW = True

# A3 reports; A7 suppresses and gates.
EVENT_IMPACT_BLOCKS_TRADES = False


def _validate_event_impact_alert_config() -> None:
    """Import-time guard for the A3 contract."""
    if EVENT_IMPACT_USES_CLAIMED_MAGNITUDE:
        raise ValueError(
            "A3 must not read the claimed magnitude: it is a unitless claim "
            "size that MEASURED appears in neither event_memory nor "
            "event_study, so it has never been checked against an outcome"
        )
    if not EVENT_IMPACT_MAGNITUDE_REASON.strip():
        raise ValueError("the refusal to read magnitude must carry its reason")
    if not EVENT_IMPACT_USES_ABSOLUTE_MOVE:
        raise ValueError(
            "impact is measured on the ABSOLUTE move: a symmetric history "
            "would otherwise cancel itself to zero and read as harmless"
        )
    if not 0.0 < EVENT_IMPACT_MIN_MEDIAN_MOVE < 1.0:
        raise ValueError(
            f"EVENT_IMPACT_MIN_MEDIAN_MOVE must lie in (0, 1), got "
            f"{EVENT_IMPACT_MIN_MEDIAN_MOVE!r}"
        )
    if EVENT_IMPACT_MIN_ANALOGS != EVENT_MEMORY_MIN_ANALOGS:
        raise ValueError(
            "the analog floor must be E6's: a second floor drifts from the "
            "one E6 already enforces on the same memories"
        )
    if EVENT_IMPACT_MIN_ANALOGS < 3:
        raise ValueError(
            f"MEASURED, a median |move| estimate at n=3 spans a five-fold "
            f"range across resamples; {EVENT_IMPACT_MIN_ANALOGS} is not a "
            f"floor"
        )
    if not EVENT_IMPACT_HORIZON.strip():
        raise ValueError("an impact horizon is required")
    if len(set(EVENT_IMPACT_VERDICTS)) != len(EVENT_IMPACT_VERDICTS):
        raise ValueError("duplicate event impact verdict")
    if EVENT_IMPACT_UNKNOWN == EVENT_IMPACT_LOW:
        raise ValueError(
            "'too little history to rate' and 'historically quiet' are "
            "different answers; collapsing them silences the events the "
            "system has never seen"
        )
    if not EVENT_IMPACT_UNKNOWN_IS_NOT_LOW:
        raise ValueError(
            "an unrated event type must not be reported as low-impact"
        )
    if EVENT_IMPACT_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_event_impact_alert_config()


# --- A4: regime change alert -----------------------------------------------------
# "Alert when the regime changes. E.g. BULLISH -> RISK_OFF." The obvious
# implementation compares today's label to yesterday's and fires on any
# difference. MEASURED on real data, that alert is mostly noise.
#
# THE DECIDING MEASUREMENT, over 74 tickers and 74,600 labelled sessions from
# the local 5-year ingest: the regime classifier flips 4,923 times - 16.6 times
# per 252 sessions per ticker - and 26% OF THOSE RUNS LAST A SINGLE SESSION.
# 48% last three sessions or fewer, and the median run is 4 sessions. A
# label-difference alert fires roughly seventeen times a year per name, a
# quarter of them on a state that reverses the next day.
#
# CONFIRMATION IS WHAT MAKES THE ALERT MEAN SOMETHING. Requiring the new label
# to persist before firing, and asking whether the alert still held 5 sessions
# later:
#
#     confirming sessions   alerts   per 252d   still held 5d later
#                       1    4,923       16.6                   42%
#                       2    3,016       10.2                   53%
#                       3    2,272        7.6                   61%
#                       5    1,628        5.5                   71%
#
# At confirm=1 the MAJORITY of alerts - 58% - do not survive a week. Three
# confirming sessions more than halves the alert rate and lifts durability to
# 61%, which is the knee: going to 5 buys 10 more points of durability for
# another 28% fewer alerts, and delays every genuine regime change by two more
# sessions.
REGIME_ALERT_VERSION = "regime-change-alert-v1"

# Sessions the new label must hold before the change is reported. MEASURED,
# this is the knee of the durability curve above.
REGIME_ALERT_CONFIRM_SESSIONS = 3

# A4 NEVER FIRES ON A SINGLE-SESSION DIFFERENCE. MEASURED, 26% of regime runs
# last exactly one session, so an unconfirmed alert is a quarter noise by
# construction.
REGIME_ALERT_REQUIRES_CONFIRMATION = True

# Transitions INTO these labels are escalated regardless of confirmation
# progress being complete, because the cost of a late warning is asymmetric: a
# missed stress onset is worse than a false one, and the roadmap names
# BULLISH -> RISK_OFF as the example transition. They still require
# confirmation - the escalation is in SEVERITY, not in skipping the evidence.
REGIME_ALERT_ESCALATED_LABELS: tuple[str, ...] = (
    REGIME_RISKOFF_LABEL,
    REGIME_STRESS_LABEL,
)

# Change kinds. Availability is first-class for A1's reason: a regime becoming
# computable is not a transition from some previous state.
REGIME_CHANGE_NONE = "NONE"
REGIME_CHANGE_CONFIRMED = "CONFIRMED"      # held for the required sessions
REGIME_CHANGE_PENDING = "PENDING"          # differs, not yet confirmed
REGIME_CHANGE_APPEARED = "APPEARED"        # regime became computable
REGIME_CHANGE_DISAPPEARED = "DISAPPEARED"  # regime stopped being computable
REGIME_CHANGE_NOT_EVALUATED = "NOT_EVALUATED"
REGIME_CHANGE_KINDS: tuple[str, ...] = (
    REGIME_CHANGE_NOT_EVALUATED,
    REGIME_CHANGE_NONE,
    REGIME_CHANGE_PENDING,
    REGIME_CHANGE_CONFIRMED,
    REGIME_CHANGE_APPEARED,
    REGIME_CHANGE_DISAPPEARED,
)

# PENDING IS REPORTED BUT DOES NOT FIRE. A change that is accumulating
# evidence is worth showing on a dashboard and is NOT worth waking someone
# for; collapsing it into NONE would hide a transition in progress, and
# collapsing it into CONFIRMED is the noise this module exists to remove.
REGIME_ALERT_PENDING_FIRES = False

# A regime that cannot be computed is not a regime of "no regime". The
# classifier returns computable=False when the frame is too short or missing
# columns, and that is an absence of measurement rather than a calm market.
REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME = True

# A4 reports; A7 suppresses and gates.
REGIME_ALERT_BLOCKS_TRADES = False


def _validate_regime_alert_config() -> None:
    """Import-time guard for the A4 contract."""
    if not REGIME_ALERT_REQUIRES_CONFIRMATION:
        raise ValueError(
            "A4 must require confirmation: MEASURED, 26% of regime runs last "
            "a single session and 58% of unconfirmed alerts do not survive "
            "five sessions"
        )
    if REGIME_ALERT_CONFIRM_SESSIONS < 2:
        raise ValueError(
            f"{REGIME_ALERT_CONFIRM_SESSIONS} confirming session(s) is not "
            f"confirmation: MEASURED, a single-session rule fires 16.6 times "
            f"per year per ticker and 58% of those alerts reverse within a "
            f"week"
        )
    if REGIME_ALERT_CONFIRM_SESSIONS > 10:
        raise ValueError(
            f"{REGIME_ALERT_CONFIRM_SESSIONS} confirming sessions delays "
            f"every genuine regime change past the point of usefulness; the "
            f"median regime run is 4 sessions"
        )
    for label in REGIME_ALERT_ESCALATED_LABELS:
        if label not in REGIME_LABELS:
            raise ValueError(
                f"{label!r} is escalated but is not a regime label; the "
                f"escalation set must name states the classifier can emit"
            )
    if REGIME_RISKOFF_LABEL not in REGIME_ALERT_ESCALATED_LABELS:
        raise ValueError(
            "risk_off must be escalated: the roadmap names BULLISH -> "
            "RISK_OFF as the example transition, and a missed stress onset "
            "costs more than a false one"
        )
    if len(set(REGIME_CHANGE_KINDS)) != len(REGIME_CHANGE_KINDS):
        raise ValueError("duplicate regime change kind")
    if REGIME_CHANGE_PENDING == REGIME_CHANGE_NONE:
        raise ValueError(
            "'a transition is accumulating evidence' and 'nothing changed' "
            "are different answers; collapsing them hides a change in progress"
        )
    if REGIME_CHANGE_PENDING == REGIME_CHANGE_CONFIRMED:
        raise ValueError(
            "collapsing PENDING into CONFIRMED reinstates the single-session "
            "alert this module exists to remove"
        )
    if REGIME_ALERT_PENDING_FIRES:
        raise ValueError(
            "a PENDING change must not fire: it is exactly the unconfirmed "
            "state MEASURED to reverse 58% of the time"
        )
    if not REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME:
        raise ValueError(
            "an uncomputable regime is an absence of measurement, not a calm "
            "market"
        )
    if REGIME_ALERT_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_regime_alert_config()


# --- A5: thesis break alert ------------------------------------------------------
# "Alert when evidence materially contradicts the existing thesis." The obvious
# implementation watches the SCORE and fires when it falls. MEASURED, that
# alert is silent on most real thesis breaks.
#
# THE DECIDING MEASUREMENT: the score is a SUM, and a sum hides a reversal.
# W1 attributes every score to three evidence buckets - operational
# (fundamental + technical), narrative (news + sentiment) and macro_shock
# (macroeconomic + market regime). Over 20,000 sampled bucket pairs, 1,306
# moved the total score by less than the 0.25 support threshold, and in 765 OF
# THOSE 1,306 (59%) A BUCKET REVERSED SIGN - it went from supporting the case
# to opposing it, or the reverse.
#
# The concrete case, constructed from the same arithmetic:
#
#     bucket          before          after
#     operational     +1.40 supports  -1.20 opposes
#     narrative       -0.10 neutral   +2.50 supports
#     macro_shock     +0.20 neutral   +0.20 neutral
#     SCORE           +1.50           +1.50   delta +0.00
#
# A score-only alert sees a +0.00 move and says nothing, while the thesis has
# inverted: the case WAS carried by operational evidence and is now carried
# purely by narrative. That is a different investment with the same number.
THESIS_ALERT_VERSION = "thesis-break-alert-v1"

# A5 watches the BUCKETS, not the score. The score is reported for context and
# is never the trigger on its own.
THESIS_ALERT_WATCHES_BUCKETS = True

# The support threshold, REUSED from W1's attribution rather than redefined.
# A second threshold would let A5 call a bucket "supporting" that the score
# engine calls neutral, and the two would disagree about the same number.
THESIS_SUPPORT_THRESHOLD = ATTRIBUTION_SUPPORT_THRESHOLD

# Break kinds, ordered by how much they contradict the thesis.
THESIS_BREAK_NONE = "NONE"
THESIS_BREAK_REVERSAL = "REVERSAL"        # a bucket crossed supports <-> opposes
THESIS_BREAK_CARRIER = "CARRIER_CHANGED"  # a different bucket now carries it
THESIS_BREAK_WITHDRAWN = "SUPPORT_WITHDRAWN"  # the carrier went neutral
THESIS_BREAK_APPEARED = "APPEARED"
THESIS_BREAK_DISAPPEARED = "DISAPPEARED"
THESIS_BREAK_NOT_EVALUATED = "NOT_EVALUATED"
THESIS_BREAK_KINDS: tuple[str, ...] = (
    THESIS_BREAK_NOT_EVALUATED,
    THESIS_BREAK_NONE,
    THESIS_BREAK_WITHDRAWN,
    THESIS_BREAK_CARRIER,
    THESIS_BREAK_REVERSAL,
    THESIS_BREAK_APPEARED,
    THESIS_BREAK_DISAPPEARED,
)

# A REVERSAL IS THE STRONGEST BREAK and fires regardless of the score move.
# MEASURED, 59% of flat-score periods contain one.
THESIS_REVERSAL_IS_MATERIAL = True

# A BUCKET AT 0.0 IS NOT A BUCKET THAT OPPOSES. MEASURED, score_engine itself
# distinguishes three reasons a bucket totals exactly 0.0: "status OK but no
# usable sentiment score", "no eligible narrative sources", and "net-zero
# contribution". Two of the three are an ABSENCE of evidence, and reading
# absence as contradiction would fire a thesis break every time a news feed
# went quiet.
THESIS_ZERO_IS_NOT_OPPOSITION = True

# A5 reports; A7 suppresses and gates.
THESIS_ALERT_BLOCKS_TRADES = False


def _validate_thesis_alert_config() -> None:
    """Import-time guard for the A5 contract."""
    if not THESIS_ALERT_WATCHES_BUCKETS:
        raise ValueError(
            "A5 must watch the evidence buckets, not the score: MEASURED, "
            "59% of flat-score periods contain a bucket reversal, and a "
            "score-only alert is silent on all of them"
        )
    if THESIS_SUPPORT_THRESHOLD != ATTRIBUTION_SUPPORT_THRESHOLD:
        raise ValueError(
            "the support threshold must be W1's: a second threshold lets A5 "
            "call a bucket supporting that the score engine calls neutral"
        )
    if not 0.0 < THESIS_SUPPORT_THRESHOLD < 10.0:
        raise ValueError(
            f"THESIS_SUPPORT_THRESHOLD {THESIS_SUPPORT_THRESHOLD!r} is not a "
            f"score-scale threshold"
        )
    if not THESIS_REVERSAL_IS_MATERIAL:
        raise ValueError(
            "a bucket crossing from supporting to opposing is the definition "
            "of evidence contradicting the thesis"
        )
    if not THESIS_ZERO_IS_NOT_OPPOSITION:
        raise ValueError(
            "a bucket at 0.0 is not opposition: MEASURED, two of the three "
            "reasons score_engine gives for a 0.0 bucket are an ABSENCE of "
            "evidence, and reading absence as contradiction fires a thesis "
            "break every time a feed goes quiet"
        )
    if len(set(THESIS_BREAK_KINDS)) != len(THESIS_BREAK_KINDS):
        raise ValueError("duplicate thesis break kind")
    for required in (THESIS_BREAK_REVERSAL, THESIS_BREAK_CARRIER, THESIS_BREAK_WITHDRAWN):
        if required not in THESIS_BREAK_KINDS:
            raise ValueError(
                f"{required!r} must be expressible, or A5 collapses distinct "
                f"kinds of contradiction into one verdict"
            )
    if THESIS_BREAK_NONE == THESIS_BREAK_NOT_EVALUATED:
        raise ValueError(
            "'the thesis held' and 'there was nothing to compare against' are "
            "different answers"
        )
    if THESIS_ALERT_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_thesis_alert_config()


# --- A6: forecast threshold alert ------------------------------------------------
# "E.g. 20D expected return + P(up) + confidence crossing a defined threshold,
# with no veto active." Three conditions and a gate. MEASURED, one of the three
# has no producer, and the two obvious ways to handle that are both wrong.
#
# THE DECIDING MEASUREMENT: A6 CANNOT FIRE TODAY, AT ANY HORIZON. Across 1d,
# 5d, 20d and 60d, ZERO of the twelve condition inputs are PRESENT -
# expected_return is ABSENT at every horizon because no trained model exists,
# and probability_up and confidence are REFUSED because nothing supplied them.
#
#     horizon  expected_return  probability_up  confidence
#     1d       ABSENT           REFUSED         REFUSED
#     5d       ABSENT           REFUSED         REFUSED
#     20d      ABSENT           REFUSED         REFUSED
#     60d      ABSENT           REFUSED         REFUSED
#
# THE TWO WRONG ANSWERS, both measured rather than argued:
#
#   Coalesce the missing value to 0.0. Then 0.0 >= the return threshold is
#   False and the alert NEVER FIRES - dead, and silent about being dead.
#
#   Skip the condition that cannot be evaluated. Then P(up) and confidence
#   alone can fire, reporting a "threshold crossing" on two of three
#   conditions, silently weakening the rule it claims to enforce.
#
# So an unevaluable condition produces NOT_EVALUATED, which is neither fired
# nor quiet. W2's fail-closed rule does not transfer: blocking on missing
# evidence is right for a VETO, but an ALERT that fires on absent data is pure
# noise.
FORECAST_THRESHOLD_ALERT_VERSION = "forecast-threshold-alert-v1"

# The horizon the roadmap names.
FTHRESHOLD_HORIZON = "20d"

# The three conditions, declared as DATA so a reader sees what must hold
# without reading code, and so adding one is a deliberate edit here.
FTHRESHOLD_CONDITION_RETURN = "expected_return"
FTHRESHOLD_CONDITION_PROBABILITY = "probability_up"
FTHRESHOLD_CONDITION_CONFIDENCE = "confidence"
FTHRESHOLD_CONDITIONS: tuple[str, ...] = (
    FTHRESHOLD_CONDITION_RETURN,
    FTHRESHOLD_CONDITION_PROBABILITY,
    FTHRESHOLD_CONDITION_CONFIDENCE,
)

# The thresholds each condition must clear.
#
# expected_return: unreachable today and stated anyway, so the contract is
# visible rather than quietly dropped. 3% over 20 sessions is roughly 40%
# annualised - a deliberately demanding bar for a system with no trained model.
FTHRESHOLD_MIN_RETURN = 0.03
# probability_up: DERIVED, not chosen. At 100 observations the 95% band on a
# base rate near 0.5 is +/-0.098 (A1's measurement), so 0.60 is the first
# tenth that clears a coin flip by more than sampling noise.
FTHRESHOLD_MIN_PROBABILITY = 0.60
# confidence: F7's MODERATE band floor, READ FROM F7's own band table rather
# than restated. A bar below MODERATE would act on a forecast F7 itself
# describes as weak, and a hardcoded 0.5 would drift if F7 retuned its bands.
FTHRESHOLD_MIN_CONFIDENCE = dict(FORECAST_CONFIDENCE_BANDS)[FCONF_BAND_MODERATE]

# EVERY CONDITION MUST BE EVALUABLE FOR THE ALERT TO HAVE A VERDICT. Skipping
# one turns "all three held" into "the ones we could check held", which is a
# weaker claim wearing the stronger one's name.
FTHRESHOLD_REQUIRES_ALL_CONDITIONS = True

# A MISSING INPUT IS NEVER COERCED. MEASURED, coalescing to 0.0 makes the
# alert permanently and silently dead.
FTHRESHOLD_COERCES_MISSING = False

# A VETO SUPPRESSES THE ALERT, and its absence is not assumed. An unknown veto
# state is NOT "no veto active" - that is the one place A6 does inherit W2's
# fail-closed instinct, because claiming governance passed when nobody asked
# is how a blocked trade gets recommended.
FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES = True

# Verdicts.
FTHRESHOLD_FIRED = "FIRED"
FTHRESHOLD_NOT_MET = "NOT_MET"           # every condition evaluable, some failed
FTHRESHOLD_VETOED = "VETOED"             # conditions met but governance blocks
FTHRESHOLD_NOT_EVALUATED = "NOT_EVALUATED"  # a condition could not be tested
FTHRESHOLD_VERDICTS: tuple[str, ...] = (
    FTHRESHOLD_NOT_EVALUATED,
    FTHRESHOLD_VETOED,
    FTHRESHOLD_NOT_MET,
    FTHRESHOLD_FIRED,
)

# A6 reports; A7 suppresses and gates.
FTHRESHOLD_BLOCKS_TRADES = False


def _validate_forecast_threshold_config() -> None:
    """Import-time guard for the A6 contract."""
    if FTHRESHOLD_COERCES_MISSING:
        raise ValueError(
            "a missing condition input must never be coerced: MEASURED, "
            "coalescing expected_return to 0.0 makes the alert permanently "
            "and silently dead, because 0.0 never clears the return threshold"
        )
    if not FTHRESHOLD_REQUIRES_ALL_CONDITIONS:
        raise ValueError(
            "skipping an unevaluable condition turns 'all three held' into "
            "'the ones we could check held', which is a weaker claim wearing "
            "the stronger one's name"
        )
    if len(set(FTHRESHOLD_CONDITIONS)) != len(FTHRESHOLD_CONDITIONS):
        raise ValueError("duplicate threshold condition")
    for required in (
        FTHRESHOLD_CONDITION_RETURN,
        FTHRESHOLD_CONDITION_PROBABILITY,
        FTHRESHOLD_CONDITION_CONFIDENCE,
    ):
        if required not in FTHRESHOLD_CONDITIONS:
            raise ValueError(
                f"the roadmap names {required!r} as a condition and it is "
                f"missing"
            )
    if not 0.0 < FTHRESHOLD_MIN_PROBABILITY < 1.0:
        raise ValueError(
            f"FTHRESHOLD_MIN_PROBABILITY must lie in (0, 1), got "
            f"{FTHRESHOLD_MIN_PROBABILITY!r}"
        )
    if FTHRESHOLD_MIN_PROBABILITY <= 0.5:
        raise ValueError(
            f"a probability bar of {FTHRESHOLD_MIN_PROBABILITY} does not "
            f"clear a coin flip; DERIVED, the 95% sampling band at 100 "
            f"observations is +/-0.098 around 0.5"
        )
    if not 0.0 < FTHRESHOLD_MIN_CONFIDENCE <= 1.0:
        raise ValueError(
            f"FTHRESHOLD_MIN_CONFIDENCE must lie in (0, 1], got "
            f"{FTHRESHOLD_MIN_CONFIDENCE!r}"
        )
    if FTHRESHOLD_MIN_CONFIDENCE != dict(FORECAST_CONFIDENCE_BANDS)[FCONF_BAND_MODERATE]:
        raise ValueError(
            "the confidence bar must be F7's MODERATE floor: a lower bar acts "
            "on a forecast F7 itself describes as weak"
        )
    if FTHRESHOLD_MIN_RETURN <= 0.0:
        raise ValueError(
            "a non-positive return bar would fire on a forecast of nothing"
        )
    if not FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES:
        raise ValueError(
            "an unknown veto state is not 'no veto active': claiming "
            "governance passed when nobody asked is how a blocked trade gets "
            "recommended"
        )
    if len(set(FTHRESHOLD_VERDICTS)) != len(FTHRESHOLD_VERDICTS):
        raise ValueError("duplicate threshold verdict")
    if FTHRESHOLD_NOT_EVALUATED == FTHRESHOLD_NOT_MET:
        raise ValueError(
            "'a condition could not be tested' and 'the conditions were "
            "tested and failed' are different answers; collapsing them hides "
            "that the alert is dead"
        )
    if FTHRESHOLD_BLOCKS_TRADES:
        raise ValueError("an alert reports; it does not trade")


_validate_forecast_threshold_config()


# --- A7: alert suppression -------------------------------------------------------
# "Prevent repeated/spam alerts. Alerts must respect risk and governance."
# Two requirements that pull in OPPOSITE directions, and the measurement says
# the naive reading of the first one breaks the system.
#
# THE DECIDING MEASUREMENT, over 8 tickers walked forward one session at a time
# exactly as a daily run would execute them - 4,622 classified sessions:
#
#     sessions in an alerting state      4,622
#     distinct episodes                    308
#     repeat days                        4,314  (93.3%)
#     alerts per episode, unsuppressed   15.01
#
# So spam is REAL and suppression is genuinely needed. But the obvious rule -
# "suppress while the label is unchanged" - is measurably wrong:
#
#     same-label consecutive pairs               4,314
#     of those, |d probability| >= 0.10            697  (16.2%)
#     of those, CROSSED a 0.5/0.6/0.75 band        491  (11.4%)
#     median |d probability| within a run       0.0033
#     max |d probability| within a run          1.0000
#
# A LABEL THAT DID NOT CHANGE IS NOT A STATE THAT DID NOT CHANGE. Eleven
# percent of the days a label-only rule would silence crossed a decision
# boundary, and the largest move under an unchanged label was a FULL reversal
# of conviction from 1.00 to 0.00. Deduplicating on the label alone hides
# exactly the days worth reading.
#
# Hence suppression keys on the DECISION-RELEVANT STATE, not on the label and
# not on the whole payload. Keying on the whole payload suppresses nothing at
# all, because as_of changes every day by construction - the same trap A1
# measured for digests (FORECAST_ALERT_USES_DIGEST).
ALERT_SUPPRESSION_VERSION = "alert-suppression-v1"

# WHAT THE SUPPRESSION KEY COVERS. Declared as DATA so a reader sees the
# identity rule without reading code, and so widening it is a deliberate edit
# here rather than a quiet drift.
SUPPRESS_KEY_FIELDS: tuple[str, ...] = ("alert", "ticker", "horizon", "state")

# FIELDS THAT MUST NEVER ENTER THE KEY. MEASURED, as_of advances every session
# by construction, so any key containing it is unique every day and suppresses
# nothing - the alert storm survives untouched while the code claims to
# deduplicate.
SUPPRESS_KEY_FORBIDDEN: tuple[str, ...] = ("as_of", "timestamp", "digest", "note")

# ESCALATION ALWAYS BREAKS SUPPRESSION. A repeat that got WORSE is new
# information, and the whole point of the measurement above is that 11.4% of
# repeats moved across a decision boundary under an unchanged label.
SUPPRESS_ESCALATION_BREAKS = True

# DE-ESCALATION DOES NOT. Going from warn to info is the situation improving;
# re-notifying on relief is how a channel trains its reader to ignore it.
SUPPRESS_DEESCALATION_BREAKS = False

# THE COOLDOWN. MEASURED, an episode runs 15.01 sessions on average, so a
# window shorter than that re-fires inside a single unchanged episode and
# reintroduces the spam this module exists to remove. 15 sessions is that
# measured mean, used directly rather than rounded to a habitual number.
SUPPRESS_COOLDOWN_SESSIONS = 15

# A SUPPRESSED ALERT IS RECORDED, NEVER DISCARDED. "Nothing fired" and "it
# fired and we chose not to show it" are different facts, and only one of them
# can be audited after a loss.
SUPPRESS_RECORDS_SUPPRESSED = True

# GOVERNANCE GATING. An alert that survives suppression still must not be
# delivered as actionable while the portfolio refuses to trade.
#
# THE UNKNOWN STATE IS NOT THE CLEAR STATE. This is the one place A7 inherits
# W2 fail-closed reasoning wholesale, for the reason A6 gave: claiming
# governance passed when nobody asked is how a blocked trade gets recommended.
SUPPRESS_UNKNOWN_GOVERNANCE_GATES = True

# GATING IS NOT DELETION. A vetoed alert is still DELIVERED, marked
# non-actionable, because "the market moved against you and the book is frozen"
# is precisely the alert a reader most needs. Silencing alerts during a veto
# would blind the operator exactly when governance says conditions are worst.
SUPPRESS_VETO_SILENCES = False

# Dispositions.
SUPPRESS_DELIVER = "DELIVER"                  # new, or escalated past the last
SUPPRESS_SUPPRESSED = "SUPPRESSED"            # a repeat inside the cooldown
SUPPRESS_GATED = "GATED"                      # delivered, but not actionable
SUPPRESS_NOT_EVALUATED = "NOT_EVALUATED"      # the alert own state is unknown
SUPPRESS_DISPOSITIONS: tuple[str, ...] = (
    SUPPRESS_NOT_EVALUATED,
    SUPPRESS_SUPPRESSED,
    SUPPRESS_GATED,
    SUPPRESS_DELIVER,
)

# A7 gates DELIVERY; it does not trade.
SUPPRESS_BLOCKS_TRADES = False


def _validate_alert_suppression_config() -> None:
    """Import-time guard for the A7 contract."""
    if not SUPPRESS_KEY_FIELDS:
        raise ValueError("a suppression key with no fields suppresses everything")
    if len(set(SUPPRESS_KEY_FIELDS)) != len(SUPPRESS_KEY_FIELDS):
        raise ValueError("duplicate suppression key field")
    if "state" not in SUPPRESS_KEY_FIELDS:
        raise ValueError(
            "the key must cover the decision-relevant STATE: MEASURED, 11.4% "
            "of same-label repeats crossed a 0.5/0.6/0.75 band and the largest "
            "move under an unchanged label was a full 1.00 reversal, so a "
            "label-only key hides exactly the days worth reading"
        )
    for forbidden in SUPPRESS_KEY_FORBIDDEN:
        if forbidden in SUPPRESS_KEY_FIELDS:
            raise ValueError(
                f"{forbidden!r} must not enter the suppression key: it advances "
                f"every session by construction, so the key is unique every day "
                f"and suppresses nothing"
            )
    if not SUPPRESS_ESCALATION_BREAKS:
        raise ValueError(
            "escalation must break suppression: a repeat that got worse is new "
            "information, and 697 of 4,314 measured repeats moved the "
            "probability by 0.10 or more under an unchanged label"
        )
    if SUPPRESS_DEESCALATION_BREAKS:
        raise ValueError(
            "de-escalation must not break suppression: re-notifying on relief "
            "is how a channel trains its reader to ignore it"
        )
    if SUPPRESS_COOLDOWN_SESSIONS < 1:
        raise ValueError(
            "a cooldown below one session suppresses nothing, because two "
            "alerts never share a session"
        )
    if SUPPRESS_COOLDOWN_SESSIONS < 15:
        raise ValueError(
            f"a cooldown of {SUPPRESS_COOLDOWN_SESSIONS} sessions re-fires "
            f"inside a single unchanged episode: MEASURED, an episode runs "
            f"15.01 sessions on average"
        )
    if not SUPPRESS_RECORDS_SUPPRESSED:
        raise ValueError(
            "a suppressed alert must be recorded: 'nothing fired' and 'it "
            "fired and we chose not to show it' are different facts, and only "
            "one of them can be audited after a loss"
        )
    if not SUPPRESS_UNKNOWN_GOVERNANCE_GATES:
        raise ValueError(
            "an unknown governance state is not a clear one: claiming "
            "governance passed when nobody asked is how a blocked trade gets "
            "recommended"
        )
    if SUPPRESS_VETO_SILENCES:
        raise ValueError(
            "a veto must not silence an alert: 'the market moved against you "
            "and the book is frozen' is precisely the alert a reader most "
            "needs, and silencing it blinds the operator when governance says "
            "conditions are worst"
        )
    if len(set(SUPPRESS_DISPOSITIONS)) != len(SUPPRESS_DISPOSITIONS):
        raise ValueError("duplicate suppression disposition")
    if SUPPRESS_SUPPRESSED == SUPPRESS_GATED:
        raise ValueError(
            "'we chose not to show this' and 'governance blocks acting on "
            "this' are different answers; collapsing them loses the reason"
        )
    if SUPPRESS_BLOCKS_TRADES:
        raise ValueError("an alert gate governs delivery; it does not trade")


_validate_alert_suppression_config()


