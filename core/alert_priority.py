"""A8 priority and action — the five bands the operator reads, derived not guessed.

The operator specified five priorities (Urgent / Very High / High / Medium / Low)
and five actions (Buy / Sell / Hold / Watch / Review). The detectors A1-A6 speak
a two-value severity, ``info`` and ``warn``. This module is the bridge, and both
halves of it are constrained by measurements made elsewhere in the system.

**WHY SEVERITY ALONE CANNOT PRODUCE FIVE BANDS.** MEASURED, five core modules
import ``ALERT_SEVERITY_INFO``/``ALERT_SEVERITY_WARN`` and W2 shares the same
vocabulary with the audit surface and the risk report. A straight map gives two
of the five bands, so ``Urgent`` could never occur and the operator's top band
would be decorative. Severity answers *how bad*; it does not answer *how soon*.

**WHAT ANSWERS "HOW SOON" IS ALREADY MEASURED.** Two facts per alert:

* **Availability changes** (``APPEARED``/``DISAPPEARED``) are not magnitudes at
  all. A1 MEASURED that folding them into a magnitude turns a vanished forecast
  into a -0.56 crash that never happened. A forecast or regime that stopped
  being computable is a statement about the *system*, and the reader must act on
  it today, so it outranks any in-band move.
* **Confirmation.** A4 MEASURED that an unconfirmed regime flip reverses the next
  session 26% of the time, and that confirmation lifts the "still held 5 days
  later" rate from 42% to 61%. ``PENDING`` is therefore genuinely less urgent
  than ``CONFIRMED`` — not a different severity, a different *certainty*.

So priority is a function of (severity, kind, confirmation), and the detectors
keep their two-value severity untouched.

**THE THIRD STATE SURVIVES THE MAPPING.** A detector that could not reach a
verdict reports ``NOT_EVALUATED`` with ``severity=None``. That is not a quiet
alert and it is not a Low one — it means a detector is BLIND, which is the thing
a reader most needs to know and the thing a Low band would bury at the bottom of
the feed. It maps to ``None``, and the dashboard shows it in its own section.

**ACTION: WHY NOTHING HERE EVER RETURNS BUY OR SELL.** The operator's vocabulary
is preserved in config, but this module cannot emit a directional call, for two
reasons that are measurements rather than caution:

* X10's honest gate reports the current release **NOT APPROVED**, with 2 of 9
  gates passing, and README rule 5 disables real-capital execution until they do.
* X1 MEASURED four of five estimators scoring *below a coin flip* (0.400-0.427);
  only ``historical_mean`` beat chance at 0.720.

A ``Buy`` emitted from an alert would be a directional recommendation from a
system that has measured itself unable to make one, and it would route around the
governance W2 and A7 exist to enforce. ``recommended_action`` therefore returns
``Watch`` or ``Review``, and ``directional_action_permitted`` states in one place
what would have to become true first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.config import (
    ALERT_ACTION_DEFAULT,
    ALERT_ACTION_REVIEW,
    ALERT_ACTION_WATCH,
    ALERT_ACTIONS,
    ALERT_DIRECTIONAL_ACTIONS,
    ALERT_PRIORITIES,
    ALERT_PRIORITY_HIGH,
    ALERT_PRIORITY_LOW,
    ALERT_PRIORITY_MEDIUM,
    ALERT_PRIORITY_RANK,
    ALERT_PRIORITY_URGENT,
    ALERT_PRIORITY_VERY_HIGH,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
)


class AlertPriorityError(ValueError):
    """Raised when a priority or action cannot be derived honestly."""


# Kinds that report an AVAILABILITY change rather than a magnitude. Listed
# explicitly per module vocabulary, never by substring: matching on "APPEARED"
# would also catch "DISAPPEARED", and the two mean opposite things.
_APPEARED_KINDS: frozenset[str] = frozenset({"APPEARED"})
_DISAPPEARED_KINDS: frozenset[str] = frozenset({"DISAPPEARED"})

# Kinds that say the detector reached no verdict. These carry severity None by
# construction in every A1-A6 module.
_UNEVALUATED_KINDS: frozenset[str] = frozenset({"NOT_EVALUATED"})

# Kinds that mean "nothing to tell the reader". Mirrors A7's _QUIET_* sets; kept
# as its own name so a change here is a deliberate edit rather than an import
# that drifts with another module's purpose.
_QUIET_STATES: frozenset[str] = frozenset(
    {"NONE", "UNCHANGED", "NO_CHANGE", "NOT_MET", "NO_IMPACT", "NO_EVENT"}
)

# A regime or thesis change that differs but has not yet persisted. A4 MEASURED
# 26% of single-session runs reverse the next day.
_UNCONFIRMED_STATES: frozenset[str] = frozenset({"PENDING"})

# Verdicts from A3 that already carry their own impact grading.
_HIGH_IMPACT = "HIGH_IMPACT"
_LOW_IMPACT = "LOW_IMPACT"
_UNKNOWN_IMPACT = "UNKNOWN_IMPACT"


def state_of(alert: Mapping[str, Any]) -> str:
    """The decision-relevant state string, read from the key its module uses.

    A1/A2 carry ``kind``; A3/A6 carry ``verdict``; A4/A5 carry ``kind``. Read
    EXPLICITLY rather than by looking for anything truthy, because an
    unrecognised shape must raise rather than silently grade as quiet. This
    mirrors ``alert_suppression.fired_state`` deliberately: the two modules ask
    different questions of the same field and must not disagree about which
    field it is.
    """
    if not isinstance(alert, Mapping):
        raise AlertPriorityError("an alert must be a mapping")
    for field in ("kind", "verdict"):
        if field in alert:
            value = alert[field]
            if value is None:
                return ""
            if not isinstance(value, str):
                raise AlertPriorityError(
                    f"{field!r} must be a string, got {value!r}; coercing it "
                    f"would grade an unreadable state as a confident one"
                )
            return value
    raise AlertPriorityError(
        "the alert carries neither 'kind' nor 'verdict', so its state cannot be "
        "read; grading it anyway would assign a priority on no evidence"
    )


def priority_for(alert: Mapping[str, Any]) -> tuple[str | None, str]:
    """The operator-facing priority for one alert, and why it got it.

    Returns ``(priority, reason)``. ``priority`` is None when the alert reports
    no verdict — a blind detector is not a Low-priority finding, and burying it
    at the bottom of the feed is the inverse of what it means.
    """
    state = state_of(alert)
    severity = alert.get("severity")

    if severity is not None and severity not in (
        ALERT_SEVERITY_INFO,
        ALERT_SEVERITY_WARN,
    ):
        raise AlertPriorityError(
            f"unknown severity {severity!r}; grading it would silently order it "
            f"against the measured vocabulary"
        )

    # 1. NO VERDICT. Checked FIRST, because a NOT_EVALUATED alert also has no
    #    severity, and every rule below would otherwise read that absence as
    #    "mild" rather than as "unknown".
    if state in _UNEVALUATED_KINDS or state.endswith("NOT_EVALUATED"):
        return None, (
            f"the detector reported {state!r}, which is not a verdict on "
            f"whether anything fired; it carries no priority because inferring "
            f"a band from an absent severity claims evidence the detector "
            f"explicitly refused to give"
        )

    # 2. QUIET. Nothing fired, so there is nothing to rank.
    if state in _QUIET_STATES:
        return None, f"the alert did not fire (state {state!r})"

    # 3. A FIRED STATE WITH NO SEVERITY IS UNGRADABLE, and this is checked
    #    BEFORE any band is assigned.
    #
    #    CAUGHT BY ENUMERATION, not by a failing test: sweeping all six detector
    #    vocabularies against every severity showed `DISAPPEARED/none` reaching
    #    Urgent and `LOW_IMPACT/none` reaching Low, because the availability and
    #    impact rules below never looked at severity. No current detector emits
    #    those pairs -- all six set a severity alongside these states -- so this
    #    was not a live defect. It was a fail-OPEN shape: the next detector to
    #    omit severity would have been handed the top band silently, and that is
    #    precisely how a grader starts inventing evidence.
    #
    #    PENDING is the one deliberate exception. A4 reports it with no severity
    #    by design -- an unconfirmed flip has no adverseness yet -- and its band
    #    comes from the confirmation measurement rather than from severity.
    if severity is None and state not in _UNCONFIRMED_STATES:
        return None, (
            f"the alert reports state {state!r}, which is neither quiet nor "
            f"unevaluated, yet carries no severity; it cannot be graded, and "
            f"assigning a band would invent the missing evidence"
        )

    # 4. AVAILABILITY CHANGES OUTRANK MAGNITUDES. A1 MEASURED that folding these
    #    into a magnitude invents a move that never happened. Something the
    #    system could measure yesterday and cannot measure today is a failure of
    #    the evidence base, and the reader must know today.
    if state in _DISAPPEARED_KINDS:
        return ALERT_PRIORITY_URGENT, (
            "a measurement that existed yesterday cannot be made today, so the "
            "evidence base itself changed; MEASURED, folding an availability "
            "change into a magnitude reports a -0.56 crash that never happened"
        )
    if state in _APPEARED_KINDS:
        return ALERT_PRIORITY_HIGH, (
            "a measurement that could not be made before is now available, "
            "which is material by construction and cannot be compared on "
            "magnitude at all"
        )

    # 5. A3's own impact grading, which is measured from realized outcomes rather
    #    than from the event's unvalidated `magnitude` claim.
    if state == _HIGH_IMPACT:
        return ALERT_PRIORITY_VERY_HIGH, (
            "the event type's realized history shows a high median absolute "
            "move at this horizon; MEASURED across 400 analogs the 20d move "
            "differs by more than 12x between event types"
        )
    if state == _LOW_IMPACT:
        return ALERT_PRIORITY_LOW, (
            "the event type's realized history shows a small median move, so "
            "the event is recorded rather than escalated"
        )
    if state == _UNKNOWN_IMPACT:
        return ALERT_PRIORITY_MEDIUM, (
            "too few analogs to grade the impact; the event is real but its "
            "history cannot size it, and silence would read as 'no impact'"
        )

    # 6. UNCONFIRMED CHANGES ARE ONE BAND LOWER. A4 MEASURED that 26% of
    #    single-session regime runs reverse the next day, and that confirmation
    #    lifts the 5-day persistence rate from 42% to 61%.
    if state in _UNCONFIRMED_STATES:
        return ALERT_PRIORITY_MEDIUM, (
            "the state differs but has not persisted for the confirming "
            "sessions; MEASURED, 26% of single-session runs reverse the next "
            "day, so this is reported without being escalated"
        )

    # 7. A CONFIRMED CHANGE, graded by severity. This is the ordinary path.
    if severity == ALERT_SEVERITY_WARN:
        return ALERT_PRIORITY_VERY_HIGH, (
            f"a confirmed change ({state!r}) at warn severity: the move is both "
            f"adverse and persistent"
        )
    if severity == ALERT_SEVERITY_INFO:
        return ALERT_PRIORITY_HIGH, (
            f"a confirmed change ({state!r}) at info severity: persistent, and "
            f"not adverse"
        )

    # 8. AN UNRECOGNISED FIRED STATE. Rule 3 already refused the severity-less
    #    case, so reaching here means a detector emitted a state this grader has
    #    never seen alongside a valid severity. It is NOT graded: a state nobody
    #    mapped has no measured band, and guessing one from severity alone would
    #    reintroduce the two-band collapse this module exists to avoid.
    return None, (
        f"state {state!r} is not one this grader maps, so it has no measured "
        f"band; grading it from severity alone would collapse the five bands "
        f"back to two"
    )


def directional_action_permitted(
    *,
    governance_clear: bool | None,
    release_approved: bool | None,
) -> tuple[bool, str]:
    """Whether a Buy or Sell may be recommended at all, and why not.

    Both inputs must be explicitly True. None is NOT True: claiming a release was
    approved because nobody asked is how an unvalidated model reaches capital.
    """
    if release_approved is not True:
        return False, (
            "the release is not approved: X10's honest gate reports NOT "
            "APPROVED with 2 of 9 gates passing, and README rule 5 disables "
            "real-capital execution until they pass"
            if release_approved is False
            else "nobody asked whether the release is approved, and an unknown "
            "approval state is not an approval"
        )
    if governance_clear is not True:
        return False, (
            "governance does not permit acting"
            if governance_clear is False
            else "the governance state is unknown, and claiming it passed when "
            "nobody asked is how a blocked trade gets recommended"
        )
    return True, "the release is approved and governance is clear"


def recommended_action(
    alert: Mapping[str, Any],
    *,
    priority: str | None = None,
    governance_clear: bool | None = None,
    release_approved: bool | None = None,
) -> tuple[str, str]:
    """What the reader should do about this alert, and why.

    Never returns a directional action under the current measured state of the
    system. ``Watch`` means "keep looking at this"; ``Review`` means "something
    about the evidence needs a human".
    """
    state = state_of(alert)
    permitted, governance_reason = directional_action_permitted(
        governance_clear=governance_clear, release_approved=release_approved
    )

    if priority is None:
        priority, _ = priority_for(alert)

    # An ungraded alert always needs a human: either a detector is blind or the
    # alert contradicts itself, and neither is something to merely watch.
    if priority is None:
        return ALERT_ACTION_REVIEW, (
            f"the alert carries no priority (state {state!r}), so a person has "
            f"to establish what it means before anything can be watched"
        )

    # An availability change is a defect in the evidence base, not a market
    # move, so it is reviewed rather than watched however severe it looks.
    if state in _DISAPPEARED_KINDS or state in _APPEARED_KINDS:
        return ALERT_ACTION_REVIEW, (
            "the set of measurable evidence changed, which is a question about "
            "the pipeline rather than about the position"
        )

    if not permitted:
        # THE IMPORTANT CASE, and the reason this function exists. Everything
        # reaching here is a real, graded market finding, and the honest answer
        # is "watch it" rather than a direction the system cannot support.
        return ALERT_ACTION_WATCH, (
            f"a graded market change, reported for monitoring rather than as a "
            f"direction: {governance_reason}"
        )

    # Reachable only once a release is approved AND governance is clear. Even
    # then this module does not invent a direction -- the forecast does -- so it
    # defers to Review rather than guessing one here.
    return ALERT_ACTION_DEFAULT, (
        "a directional action is permitted, but the direction must come from a "
        "promoted forecast rather than from the alert grader"
    )


def priority_rank(priority: str | None) -> int | None:
    """Rank a priority, or None when there is none to rank.

    None is a real answer: an ungraded alert has no band, and inventing 0 would
    order it below Low rather than as incomparable.
    """
    if priority is None:
        return None
    if priority not in ALERT_PRIORITY_RANK:
        raise AlertPriorityError(
            f"unknown priority {priority!r}; the known priorities are "
            f"{list(ALERT_PRIORITIES)}"
        )
    return ALERT_PRIORITY_RANK[priority]


def grade(
    alert: Mapping[str, Any],
    *,
    governance_clear: bool | None = None,
    release_approved: bool | None = None,
) -> dict:
    """Priority and action for one alert, with the reasoning for both."""
    priority, priority_reason = priority_for(alert)
    action, action_reason = recommended_action(
        alert,
        priority=priority,
        governance_clear=governance_clear,
        release_approved=release_approved,
    )
    if action not in ALERT_ACTIONS:
        raise AlertPriorityError(f"produced unknown action {action!r}")
    if action in ALERT_DIRECTIONAL_ACTIONS:
        raise AlertPriorityError(
            f"the grader produced the directional action {action!r}: MEASURED, "
            f"four of five estimators score below chance and X10 reports the "
            f"release NOT APPROVED, so no alert may assert a direction"
        )
    return {
        "priority": priority,
        "priority_rank": priority_rank(priority),
        "priority_reason": priority_reason,
        "action": action,
        "action_reason": action_reason,
        "state": state_of(alert),
        "severity": alert.get("severity"),
    }
