"""A7 alert suppression — the last gate between a finding and a reader.

"Prevent repeated/spam alerts. Alerts must respect risk and governance." Two
requirements that pull in opposite directions, and MEASURED, the naive reading
of the first one breaks the second.

**Spam is real.** Walking 8 tickers forward one session at a time, exactly as a
daily run would execute them, 4,622 classified sessions collapse to 308 distinct
episodes. 93.3% of alert-days are repeats, and an unsuppressed channel emits
15.01 alerts per episode. Nobody reads the fifteenth.

**But deduplicating on the label is measurably wrong.** Of the 4,314 same-label
consecutive pairs a label-only rule would silence, 697 (16.2%) moved the
probability by 0.10 or more, 491 (11.4%) crossed a 0.5/0.6/0.75 decision band,
and the largest move under an unchanged label was a FULL reversal of conviction
from 1.00 to 0.00. A label that did not change is not a state that did not
change, and the label-only rule hides exactly the days worth reading.

So the suppression key covers the DECISION-RELEVANT STATE. Two other keys are
tempting and both fail:

* **The whole payload.** Suppresses nothing, because ``as_of`` advances every
  session by construction, so the key is unique every day. This is the same trap
  A1 measured for digests.
* **The label alone.** Suppresses the 11.4% measured above.

**Escalation always breaks suppression; de-escalation never does.** A repeat that
got worse is new information. A repeat that got better is relief, and
re-notifying on relief is how a channel trains its reader to ignore it.

**Gating is not deletion.** An alert blocked by governance is still DELIVERED,
marked non-actionable. "The market moved against you and the book is frozen" is
precisely the alert a reader most needs; silencing it would blind the operator
exactly when governance says conditions are worst. What gating removes is the
*call to act*, not the *fact*.

**An unknown governance state is not a clear one.** The one place A7 inherits
W2 fail-closed reasoning wholesale, for the reason A6 gave: claiming governance
passed when nobody asked is how a blocked trade gets recommended.

**A suppressed alert is recorded, never discarded.** "Nothing fired" and "it
fired and we chose not to show it" are different facts, and only one of them can
be audited after a loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.config import (
    ALERT_SEVERITIES,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    ALERT_SUPPRESSION_VERSION,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    SUPPRESS_BLOCKS_TRADES,
    SUPPRESS_COOLDOWN_SESSIONS,
    SUPPRESS_DEESCALATION_BREAKS,
    SUPPRESS_DELIVER,
    SUPPRESS_DISPOSITIONS,
    SUPPRESS_ESCALATION_BREAKS,
    SUPPRESS_GATED,
    SUPPRESS_KEY_FIELDS,
    SUPPRESS_KEY_FORBIDDEN,
    SUPPRESS_NOT_EVALUATED,
    SUPPRESS_RECORDS_SUPPRESSED,
    SUPPRESS_SUPPRESSED,
    SUPPRESS_UNKNOWN_GOVERNANCE_GATES,
    SUPPRESS_VETO_SILENCES,
)


class AlertSuppressionError(ValueError):
    """Raised when an alert cannot be dispositioned without guessing."""


# Severity order, so "escalated" is a comparison rather than an opinion.
_SEVERITY_RANK: dict[str, int] = {
    ALERT_SEVERITY_INFO: 1,
    ALERT_SEVERITY_WARN: 2,
}


def severity_rank(severity: Any) -> int | None:
    """Rank a severity, or None when there is none to rank.

    None is a real answer here: an alert that did not fire has no severity, and
    inventing 0 for it would make "no alert" compare as less severe than info
    rather than as incomparable.
    """
    if severity is None:
        return None
    if not isinstance(severity, str) or severity not in _SEVERITY_RANK:
        raise AlertSuppressionError(
            f"unknown severity {severity!r}; the known severities are "
            f"{list(ALERT_SEVERITIES)}, and ranking an unknown one would "
            f"silently order it against them"
        )
    return _SEVERITY_RANK[severity]


def fired_state(alert: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Whether this alert fired, read from the key its own module uses.

    A1 and A2 carry ``fired``; A3 and A6 carry ``verdict``; A4 and A5 carry
    ``kind``. Each is read EXPLICITLY. A generic "look for anything truthy"
    reader would silently treat an unrecognised shape as quiet, which is the
    failure mode this whole sprint exists to prevent.

    Returns ``(fired, state)`` where ``fired`` is None when the alert does not
    say, and ``state`` is the decision-relevant state string for the key.
    """
    if not isinstance(alert, Mapping):
        raise AlertSuppressionError("an alert must be a mapping")

    if "fired" in alert:
        fired = alert["fired"]
        if fired is not None and not isinstance(fired, bool):
            raise AlertSuppressionError(
                f"'fired' must be a bool or None, got {fired!r}; coercing it "
                f"would turn an unreadable answer into a confident one"
            )
        kind = alert.get("kind")
        return fired, str(kind) if kind is not None else ""

    if "verdict" in alert:
        verdict = alert["verdict"]
        if verdict is None:
            return None, ""
        if not isinstance(verdict, str):
            raise AlertSuppressionError(
                f"'verdict' must be a string, got {verdict!r}"
            )
        if verdict.endswith("NOT_EVALUATED"):
            return None, verdict
        return verdict not in _QUIET_VERDICTS, verdict

    if "kind" in alert:
        kind = alert["kind"]
        if kind is None:
            return None, ""
        if not isinstance(kind, str):
            raise AlertSuppressionError(f"'kind' must be a string, got {kind!r}")
        if kind.endswith("NOT_EVALUATED"):
            return None, kind
        return kind not in _QUIET_KINDS, kind

    raise AlertSuppressionError(
        "the alert carries none of 'fired', 'verdict' or 'kind', so whether it "
        "fired cannot be read; treating it as quiet would silently drop it"
    )


# States that mean "nothing to tell the reader". Listed explicitly so adding an
# alert kind is a deliberate edit rather than a default.
_QUIET_VERDICTS: frozenset[str] = frozenset({"NOT_MET", "NONE", "NO_IMPACT"})
_QUIET_KINDS: frozenset[str] = frozenset({"NONE", "UNCHANGED", "NO_CHANGE"})


def suppression_key(alert: Mapping[str, Any]) -> tuple[str, ...]:
    """The identity an alert is deduplicated on.

    Covers the decision-relevant STATE, never ``as_of``. MEASURED, a key
    containing ``as_of`` is unique every session and suppresses nothing.
    """
    _, state = fired_state(alert)
    parts: list[str] = []
    for field in SUPPRESS_KEY_FIELDS:
        if field in SUPPRESS_KEY_FORBIDDEN:
            raise AlertSuppressionError(
                f"{field!r} must never enter the suppression key: it advances "
                f"every session by construction, so the key would be unique "
                f"every day and suppress nothing"
            )
        if field == "state":
            parts.append(state)
            continue
        value = alert.get(field)
        parts.append("" if value is None else str(value))
    return tuple(parts)


def governance_state(
    portfolio: Mapping[str, Any] | None,
    risk: Mapping[str, Any] | None,
) -> tuple[bool | None, str]:
    """Whether governance permits acting on an alert, and why.

    Returns ``(clear, reason)`` where ``clear`` is None when nobody asked. None
    is NOT True: claiming governance passed when nobody asked is how a blocked
    trade gets recommended.
    """
    if portfolio is None and risk is None:
        return None, (
            "no portfolio decision and no risk report were supplied, so "
            "whether governance permits acting is unknown, not clear"
        )

    if risk is not None:
        if not isinstance(risk, Mapping):
            raise AlertSuppressionError("a risk report must be a mapping")
        if "veto" not in risk:
            return None, (
                "the risk report carries no 'veto' field, so whether a veto is "
                "active is unknown, not absent"
            )
        if risk["veto"]:
            ids = risk.get("veto_rule_ids") or []
            return False, (
                f"risk policy vetoes: {', '.join(map(str, ids)) or 'unnamed rule'}"
            )

    if portfolio is not None:
        if not isinstance(portfolio, Mapping):
            raise AlertSuppressionError("a portfolio decision must be a mapping")
        verdict = portfolio.get("verdict")
        if verdict is None:
            return None, (
                "the portfolio decision carries no verdict, so whether the "
                "book accepts a trade is unknown, not clear"
            )
        if verdict == PORTFOLIO_NOT_EVALUATED:
            return None, (
                "the portfolio decision is NOT_EVALUATED, which is not the "
                "same as PROCEED"
            )
        if verdict == PORTFOLIO_NO_TRADE:
            ids = portfolio.get("veto_rule_ids") or []
            return False, (
                f"the portfolio refuses new risk: "
                f"{', '.join(map(str, ids)) or 'unnamed rule'}"
            )
        if verdict != PORTFOLIO_PROCEED:
            raise AlertSuppressionError(
                f"unknown portfolio verdict {verdict!r}; treating it as clear "
                f"would let an unrecognised refusal through"
            )

    return True, "governance permits acting on this alert"


def disposition(
    alert: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    sessions_since: int | None = None,
    portfolio: Mapping[str, Any] | None = None,
    risk: Mapping[str, Any] | None = None,
) -> dict:
    """Decide whether one alert reaches the reader, and as what.

    ``previous`` is the last alert delivered under the same suppression key, or
    None when there was none. ``sessions_since`` is how many sessions have
    passed since it was delivered; None means unknown, which does NOT suppress,
    because suppressing on an unknown age hides an alert on a guess.
    """
    fired, state = fired_state(alert)
    key = suppression_key(alert)
    clear, governance_reason = governance_state(portfolio, risk)

    severity = alert.get("severity")
    rank = severity_rank(severity)

    detail: dict[str, Any] = {
        "key": list(key),
        "state": state,
        "fired": fired,
        "severity": severity,
        "governance_clear": clear,
        "governance_reason": governance_reason,
        "sessions_since": sessions_since,
        "cooldown_sessions": SUPPRESS_COOLDOWN_SESSIONS,
        "escalated": False,
        "previous_severity": None,
    }

    if fired is None:
        return _record(
            SUPPRESS_NOT_EVALUATED,
            alert,
            reason=(
                f"the alert reports state {state!r}, which is not a verdict on "
                f"whether anything fired; it is neither delivered nor "
                f"suppressed, because inferring calm from an unreadable answer "
                f"is how a live alert goes missing"
            ),
            **detail,
        )

    if not fired:
        return _record(
            SUPPRESS_SUPPRESSED,
            alert,
            reason=f"the alert did not fire (state {state!r})",
            **detail,
        )

    if previous is not None:
        if not isinstance(previous, Mapping):
            raise AlertSuppressionError("a previous alert must be a mapping")
        previous_key = suppression_key(previous)
        if previous_key != key:
            raise AlertSuppressionError(
                f"the previous alert has key {list(previous_key)}, not "
                f"{list(key)}; comparing across keys would suppress a "
                f"different alert than the one that fired"
            )
        previous_rank = severity_rank(previous.get("severity"))
        detail["previous_severity"] = previous.get("severity")

        escalated = (
            rank is not None
            and previous_rank is not None
            and rank > previous_rank
        )
        detail["escalated"] = escalated

        if escalated and SUPPRESS_ESCALATION_BREAKS:
            return _gate(
                alert,
                clear,
                reason=(
                    f"severity rose from {previous.get('severity')!r} to "
                    f"{severity!r}; a repeat that got worse is new information"
                ),
                **detail,
            )

        de_escalated = (
            rank is not None
            and previous_rank is not None
            and rank < previous_rank
        )
        if de_escalated and not SUPPRESS_DEESCALATION_BREAKS:
            return _record(
                SUPPRESS_SUPPRESSED,
                alert,
                reason=(
                    f"severity fell from {previous.get('severity')!r} to "
                    f"{severity!r}; re-notifying on relief is how a channel "
                    f"trains its reader to ignore it"
                ),
                **detail,
            )

        if sessions_since is None:
            return _gate(
                alert,
                clear,
                reason=(
                    "a previous alert exists under this key but its age is "
                    "unknown; suppressing on an unknown age would hide an "
                    "alert on a guess"
                ),
                **detail,
            )

        if not isinstance(sessions_since, int) or isinstance(sessions_since, bool):
            raise AlertSuppressionError(
                f"sessions_since must be an int or None, got {sessions_since!r}"
            )
        if sessions_since < 0:
            raise AlertSuppressionError(
                f"sessions_since must not be negative, got {sessions_since}"
            )

        if sessions_since < SUPPRESS_COOLDOWN_SESSIONS:
            return _record(
                SUPPRESS_SUPPRESSED,
                alert,
                reason=(
                    f"an identical state was delivered {sessions_since} "
                    f"session(s) ago, inside the {SUPPRESS_COOLDOWN_SESSIONS}-"
                    f"session cooldown; MEASURED, an episode runs 15.01 "
                    f"sessions and re-firing inside one is the spam this gate "
                    f"exists to remove"
                ),
                **detail,
            )

        return _gate(
            alert,
            clear,
            reason=(
                f"the last delivery under this key was {sessions_since} "
                f"session(s) ago, past the {SUPPRESS_COOLDOWN_SESSIONS}-session "
                f"cooldown"
            ),
            **detail,
        )

    return _gate(
        alert,
        clear,
        reason="no previous alert under this key; this state is new",
        **detail,
    )


def _gate(
    alert: Mapping[str, Any],
    clear: bool | None,
    *,
    reason: str,
    **detail,
) -> dict:
    """Apply governance to an alert that survived suppression.

    Gating never silences. A vetoed alert still reaches the reader, marked
    non-actionable, because the alert a reader most needs is the one that fires
    while the book is frozen.
    """
    governance_reason = detail.get("governance_reason", "")
    if clear is True:
        return _record(SUPPRESS_DELIVER, alert, reason=reason, actionable=True, **detail)

    if clear is False:
        return _record(
            SUPPRESS_GATED,
            alert,
            reason=f"{reason}; delivered but not actionable because {governance_reason}",
            actionable=False,
            **detail,
        )

    if SUPPRESS_UNKNOWN_GOVERNANCE_GATES:
        return _record(
            SUPPRESS_GATED,
            alert,
            reason=(
                f"{reason}; delivered but not actionable because "
                f"{governance_reason} - an unknown governance state is not a "
                f"clear one"
            ),
            actionable=False,
            **detail,
        )

    raise AlertSuppressionError(  # pragma: no cover - config forbids reaching here
        "an unknown governance state must gate"
    )


def _record(verdict: str, alert: Mapping[str, Any], **detail) -> dict:
    """One A7 answer, always carrying the alert it dispositioned."""
    if verdict not in SUPPRESS_DISPOSITIONS:
        raise AlertSuppressionError(f"unknown disposition {verdict!r}")
    payload = {
        "version": ALERT_SUPPRESSION_VERSION,
        "gate": "alert_suppression",
        "disposition": verdict,
        "alert": alert.get("alert"),
        "ticker": alert.get("ticker"),
        "records_suppressed": SUPPRESS_RECORDS_SUPPRESSED,
        "veto_silences": SUPPRESS_VETO_SILENCES,
        "blocks_trades": SUPPRESS_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.setdefault("actionable", False)
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED over 4,622 sessions walked forward one at a time: 93.3% of "
    "alert-days are repeats and an unsuppressed channel emits 15.01 alerts per "
    "episode, so suppression is needed. But 11.4% of the same-label repeats a "
    "label-only rule would silence CROSSED a decision band, and the largest "
    "move under an unchanged label was a full 1.00 reversal - so the key "
    "covers the decision-relevant state, never the label alone and never "
    "as_of. Gating is not deletion: a vetoed alert is still delivered, marked "
    "non-actionable."
)


def suppression_problems(record: Mapping[str, Any]) -> list[str]:
    """Contract check on a disposition record. Empty means clean."""
    problems: list[str] = []
    if not isinstance(record, Mapping):
        return ["record is not a mapping"]

    if record.get("version") != ALERT_SUPPRESSION_VERSION:
        problems.append(
            f"version is {record.get('version')!r}, expected "
            f"{ALERT_SUPPRESSION_VERSION!r}"
        )

    verdict = record.get("disposition")
    if verdict not in SUPPRESS_DISPOSITIONS:
        problems.append(f"unknown disposition {verdict!r}")

    key = record.get("key")
    if not isinstance(key, list) or not key:
        problems.append("the record carries no suppression key")
    elif len(key) != len(SUPPRESS_KEY_FIELDS):
        problems.append(
            f"the key has {len(key)} part(s), expected "
            f"{len(SUPPRESS_KEY_FIELDS)}"
        )

    if record.get("blocks_trades"):
        problems.append("an alert gate governs delivery; it does not trade")

    if record.get("veto_silences"):
        problems.append(
            "a veto must not silence an alert: the alert a reader most needs "
            "is the one that fires while the book is frozen"
        )

    if not record.get("records_suppressed"):
        problems.append(
            "a suppressed alert must be recorded, or 'nothing fired' and 'we "
            "chose not to show it' become indistinguishable"
        )

    actionable = record.get("actionable")
    if verdict == SUPPRESS_GATED and actionable:
        problems.append(
            "a GATED alert must not be actionable; that is what gating means"
        )
    if verdict == SUPPRESS_DELIVER and not actionable:
        problems.append(
            "a DELIVER alert must be actionable, or it should have been GATED "
            "with the reason"
        )
    if verdict == SUPPRESS_DELIVER and record.get("governance_clear") is not True:
        problems.append(
            "an alert may only be DELIVERED as actionable when governance is "
            "affirmatively clear; unknown is not clear"
        )
    if verdict == SUPPRESS_SUPPRESSED and actionable:
        problems.append("a SUPPRESSED alert cannot be actionable")

    if verdict == SUPPRESS_NOT_EVALUATED and record.get("fired") is not None:
        problems.append(
            "NOT_EVALUATED means the alert never said whether it fired; a "
            "record carrying a fired verdict is a different answer"
        )

    if record.get("escalated") and verdict == SUPPRESS_SUPPRESSED:
        problems.append(
            "an escalated alert must never be suppressed: a repeat that got "
            "worse is new information, and MEASURED, 697 of 4,314 repeats "
            "moved the probability by 0.10 or more under an unchanged label"
        )

    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every disposition must carry a reason")

    return problems


def render_suppression(record: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one disposition."""
    lines = [
        f"Alert gate: {record.get('alert')} / {record.get('ticker')}",
        f"  disposition : {record.get('disposition')}"
        f"  (actionable: {record.get('actionable')})",
        f"  key         : {' | '.join(map(str, record.get('key') or []))}",
        f"  severity    : {_shown(record.get('severity'))}"
        f"  (was {_shown(record.get('previous_severity'))},"
        f" escalated: {record.get('escalated')})",
        f"  cooldown    : {_shown(record.get('sessions_since'))} of"
        f" {record.get('cooldown_sessions')} session(s) elapsed",
        f"  governance  : {_governance(record.get('governance_clear'))}"
        f" - {record.get('governance_reason')}",
        f"  reason      : {record.get('reason')}",
    ]
    return lines


def _governance(clear: Any) -> str:
    """Governance shown as three states, never two."""
    if clear is True:
        return "CLEAR"
    if clear is False:
        return "BLOCKED"
    return "UNKNOWN"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
