"""A7 governance gate — alert suppression must suppress spam without hiding news.

Exits 1 when any of these fails. Each scenario is built from a SEEDED generator,
never from ingested data, so the gate runs on a fresh clone.

The structural effect this gate pins, reproduced from a seeded episode series
rather than quoted: a label-only suppression key silences days on which the
decision-relevant state changed. On the measured book that was 11.4% of
same-label repeats, with the largest move a full 1.00 reversal of conviction.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.alert_suppression import (  # noqa: E402
    AlertSuppressionError,
    disposition,
    fired_state,
    governance_state,
    render_suppression,
    severity_rank,
    suppression_key,
    suppression_problems,
)
from core.config import (  # noqa: E402
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    SUPPRESS_COOLDOWN_SESSIONS,
    SUPPRESS_DELIVER,
    SUPPRESS_GATED,
    SUPPRESS_KEY_FIELDS,
    SUPPRESS_KEY_FORBIDDEN,
    SUPPRESS_NOT_EVALUATED,
    SUPPRESS_SUPPRESSED,
)

CLEAR = {"verdict": PORTFOLIO_PROCEED}
NO_VETO = {"veto": False}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def alert(**over):
    base = {
        "alert": "regime_change",
        "ticker": "AAPL",
        "horizon": "20d",
        "kind": "CONFIRMED",
        "severity": ALERT_SEVERITY_WARN,
    }
    base.update(over)
    return base


# --- 1. Suppression actually suppresses ------------------------------------------
# MEASURED on the book: 93.3% of alert-days are repeats, 15.01 alerts per
# episode unsuppressed. A gate that cannot show suppression happening would pass
# a module that never suppresses.

episode = [alert(as_of=f"2026-01-{day:02d}") for day in range(1, 16)]
delivered = 0
previous = None
since = 0
for day, item in enumerate(episode):
    record = disposition(
        item,
        previous=previous,
        sessions_since=since if previous is not None else None,
        portfolio=CLEAR,
        risk=NO_VETO,
    )
    if record["disposition"] == SUPPRESS_DELIVER:
        delivered += 1
        previous = item
        since = 0
    else:
        since += 1

check(
    delivered == 1,
    f"an unchanged 15-session episode delivered {delivered} alerts, expected 1; "
    f"MEASURED, an unsuppressed channel emits 15.01 alerts per episode and "
    f"this is the spam the gate exists to remove",
)


# --- 2. But it does NOT suppress a changed state ----------------------------------
# The 11.4% measurement, reproduced structurally: when the decision-relevant
# state changes, the key must change even though the alert name and ticker did
# not.

check(
    suppression_key(alert(kind="CONFIRMED")) != suppression_key(alert(kind="ESCALATED")),
    "a changed decision state produced the same suppression key: MEASURED, "
    "11.4% of same-label repeats crossed a decision band and the largest move "
    "under an unchanged label was a full 1.00 reversal",
)

changed = disposition(
    alert(kind="ESCALATED"),
    previous=None,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    changed["disposition"] == SUPPRESS_DELIVER,
    f"a newly-changed state was {changed['disposition']}, expected DELIVER",
)


# --- 3. as_of must never enter the key -------------------------------------------
# MEASURED, as_of advances every session by construction, so a key containing it
# is unique every day and suppresses NOTHING while claiming to deduplicate.

check(
    suppression_key(alert(as_of="2026-09-22")) == suppression_key(alert(as_of="2026-09-23")),
    "as_of changed the suppression key: a key containing as_of is unique every "
    "session, so the alert storm survives untouched",
)
for forbidden in SUPPRESS_KEY_FORBIDDEN:
    check(
        forbidden not in SUPPRESS_KEY_FIELDS,
        f"{forbidden!r} is in the suppression key and must not be",
    )


# --- 4. Escalation breaks suppression, de-escalation does not ---------------------

escalated = disposition(
    alert(severity=ALERT_SEVERITY_WARN),
    previous=alert(severity=ALERT_SEVERITY_INFO),
    sessions_since=1,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    escalated["disposition"] == SUPPRESS_DELIVER and escalated["escalated"],
    f"an escalated repeat was {escalated['disposition']} one session into the "
    f"cooldown; a repeat that got WORSE is new information",
)

# The de-escalation probe must sit PAST the cooldown. Inside it, the cooldown
# already suppresses and the de-escalation rule never decides anything - a
# probe there passes whether the rule exists or not.
relieved = disposition(
    alert(severity=ALERT_SEVERITY_INFO),
    previous=alert(severity=ALERT_SEVERITY_WARN),
    sessions_since=SUPPRESS_COOLDOWN_SESSIONS + 5,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    relieved["disposition"] == SUPPRESS_SUPPRESSED,
    f"a de-escalated repeat {SUPPRESS_COOLDOWN_SESSIONS + 5} sessions on was "
    f"{relieved['disposition']}, expected SUPPRESSED; re-notifying on relief "
    f"is how a channel trains its reader to ignore it",
)

# And the same age with severity UNCHANGED must deliver, or the probe above
# would pass on a module that simply suppressed everything old.
unchanged_old = disposition(
    alert(severity=ALERT_SEVERITY_WARN),
    previous=alert(severity=ALERT_SEVERITY_WARN),
    sessions_since=SUPPRESS_COOLDOWN_SESSIONS + 5,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    unchanged_old["disposition"] == SUPPRESS_DELIVER,
    f"an unchanged repeat past the cooldown was {unchanged_old['disposition']}, "
    f"expected DELIVER; without this the de-escalation probe proves nothing",
)


# --- 4b. An ABSENT severity is incomparable, never rank 0 -------------------------
# The shape rule, applied to ordering. If an absent severity ranks 0 then a
# severity APPEARING where there was none compares as 1 > 0 and reads as
# escalation, breaking the cooldown on an alert that did not get worse.

check(
    severity_rank(None) is None,
    "an absent severity was ranked as a number; ranking it 0 makes 'no "
    "severity' compare as less severe than info rather than incomparable",
)

appeared = disposition(
    alert(severity=ALERT_SEVERITY_INFO),
    previous=alert(severity=None),
    sessions_since=1,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    not appeared["escalated"],
    "a severity appearing where there was none was reported as ESCALATION; "
    "the situation did not get worse, the measurement did",
)
check(
    appeared["disposition"] == SUPPRESS_SUPPRESSED,
    f"a severity appearing where there was none was {appeared['disposition']} "
    f"one session into the cooldown, expected SUPPRESSED",
)

vanished = disposition(
    alert(severity=None),
    previous=alert(severity=ALERT_SEVERITY_WARN),
    sessions_since=1,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    not vanished["escalated"],
    "a severity vanishing was reported as escalation",
)


# --- 5. Governance gates, and gating is NOT deletion ------------------------------

vetoed = disposition(alert(), portfolio={"verdict": PORTFOLIO_NO_TRADE}, risk=NO_VETO)
check(
    vetoed["disposition"] == SUPPRESS_GATED,
    f"a vetoed alert was {vetoed['disposition']}, expected GATED",
)
check(
    not vetoed["actionable"],
    "a vetoed alert was marked actionable; that is what gating prevents",
)
check(
    vetoed["disposition"] != SUPPRESS_SUPPRESSED,
    "a veto SILENCED an alert: 'the market moved against you and the book is "
    "frozen' is precisely the alert a reader most needs, and silencing it "
    "blinds the operator when governance says conditions are worst",
)

risk_vetoed = disposition(alert(), portfolio=CLEAR, risk={"veto": True, "veto_rule_ids": ["dd"]})
check(
    risk_vetoed["disposition"] == SUPPRESS_GATED,
    f"a risk-policy veto produced {risk_vetoed['disposition']}, expected GATED",
)


# --- 6. An unknown governance state is not a clear one ----------------------------
# The one place A7 inherits W2 fail-closed reasoning wholesale.

for label, portfolio, risk in (
    ("nothing supplied", None, None),
    ("risk report with no veto field", CLEAR, {"policy_version": "v1"}),
    ("portfolio NOT_EVALUATED", {"verdict": PORTFOLIO_NOT_EVALUATED}, NO_VETO),
    ("portfolio with no verdict", {"reason": "x"}, NO_VETO),
):
    record = disposition(alert(), portfolio=portfolio, risk=risk)
    check(
        record["disposition"] == SUPPRESS_GATED,
        f"{label}: disposition was {record['disposition']}, expected GATED - "
        f"claiming governance passed when nobody asked is how a blocked trade "
        f"gets recommended",
    )
    check(
        not record["actionable"],
        f"{label}: the alert was marked actionable on an unknown governance state",
    )

check(
    governance_state(None, None)[0] is None,
    "an absent governance state reported a boolean; unknown is a third answer",
)


# --- 7. Escalation does not override governance -----------------------------------
# Two rules that could be confused: escalation breaks SUPPRESSION, never the
# governance gate.

both = disposition(
    alert(severity=ALERT_SEVERITY_WARN),
    previous=alert(severity=ALERT_SEVERITY_INFO),
    sessions_since=1,
    portfolio={"verdict": PORTFOLIO_NO_TRADE},
    risk=NO_VETO,
)
check(
    both["disposition"] == SUPPRESS_GATED and not both["actionable"],
    f"an escalated alert under a portfolio veto was {both['disposition']} "
    f"(actionable={both['actionable']}); escalation breaks suppression, not "
    f"governance",
)


# --- 8. NOT_EVALUATED stays distinct ----------------------------------------------

unevaluated = disposition({"alert": "x", "ticker": "AAPL", "verdict": "NOT_EVALUATED"})
check(
    unevaluated["disposition"] == SUPPRESS_NOT_EVALUATED,
    f"an unevaluated alert was {unevaluated['disposition']}; 'the alert never "
    f"said' and 'the alert said no' are different answers",
)
check(
    unevaluated["disposition"] != SUPPRESS_SUPPRESSED,
    "NOT_EVALUATED collapsed into SUPPRESSED, hiding that the alert is blind",
)


# --- 9. An unreadable alert raises rather than being treated as quiet -------------

try:
    fired_state({"alert": "mystery", "ticker": "AAPL"})
except AlertSuppressionError:
    pass
else:
    failures.append(
        "an alert carrying none of fired/verdict/kind was read without error; "
        "treating an unrecognised shape as quiet silently drops it"
    )

try:
    severity_rank("catastrophic")
except AlertSuppressionError:
    pass
else:
    failures.append("an unknown severity was ranked instead of raising")

try:
    governance_state({"verdict": "MAYBE"}, NO_VETO)
except AlertSuppressionError:
    pass
else:
    failures.append(
        "an unrecognised portfolio verdict was accepted; it could let a "
        "refusal through as clear"
    )


# --- 10. Suppressing on an unknown age would hide an alert on a guess -------------

unknown_age = disposition(
    alert(), previous=alert(), sessions_since=None, portfolio=CLEAR, risk=NO_VETO
)
check(
    unknown_age["disposition"] == SUPPRESS_DELIVER,
    f"an alert with an unknown age was {unknown_age['disposition']}; "
    f"suppressing on an unknown age hides an alert on a guess",
)


# --- 11. The cooldown boundary is exact -------------------------------------------

inside = disposition(
    alert(),
    previous=alert(),
    sessions_since=SUPPRESS_COOLDOWN_SESSIONS - 1,
    portfolio=CLEAR,
    risk=NO_VETO,
)
outside = disposition(
    alert(),
    previous=alert(),
    sessions_since=SUPPRESS_COOLDOWN_SESSIONS,
    portfolio=CLEAR,
    risk=NO_VETO,
)
check(
    inside["disposition"] == SUPPRESS_SUPPRESSED,
    f"one session inside the cooldown produced {inside['disposition']}",
)
check(
    outside["disposition"] == SUPPRESS_DELIVER,
    f"exactly at the cooldown boundary produced {outside['disposition']}",
)


# --- 12. Every record passes its own contract check --------------------------------

seeded = random.Random(20260923)
for _ in range(200):
    severity = seeded.choice([ALERT_SEVERITY_INFO, ALERT_SEVERITY_WARN, None])
    prev_severity = seeded.choice([ALERT_SEVERITY_INFO, ALERT_SEVERITY_WARN, None])
    has_previous = seeded.choice([True, False])
    kind = seeded.choice(["CONFIRMED", "ESCALATED", "NONE"])
    record = disposition(
        alert(severity=severity, kind=kind),
        previous=alert(severity=prev_severity, kind=kind) if has_previous else None,
        sessions_since=seeded.choice([None, 0, 1, 7, 15, 40]) if has_previous else None,
        portfolio=seeded.choice(
            [CLEAR, {"verdict": PORTFOLIO_NO_TRADE}, {"verdict": PORTFOLIO_NOT_EVALUATED}, None]
        ),
        risk=seeded.choice([NO_VETO, {"veto": True, "veto_rule_ids": ["dd"]}, None]),
    )
    problems = suppression_problems(record)
    if problems:
        failures.append(f"a generated record failed its own contract check: {problems}")
        break
    lines = render_suppression(record)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append("render_suppression did not return lines")
        break


# --- 13. Nothing renders an absent value as a number -------------------------------

blank = disposition(alert(severity=None, kind="NONE"))
text = "\n".join(render_suppression(blank))
check(
    "ABSENT" in text,
    "an absent severity was not rendered as ABSENT; the shape rule says a "
    "value exists IFF it was measured",
)


if failures:
    print("A7 ALERT SUPPRESSION GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("A7 alert suppression gate: OK")
print(f"  unchanged 15-session episode      delivered {delivered} alert(s), not 15")
print(f"  cooldown                          {SUPPRESS_COOLDOWN_SESSIONS} sessions")
print("  escalation breaks suppression     yes; de-escalation does not")
print("  veto                              GATES, never silences")
print("  unknown governance                gates, because unknown is not clear")
sys.exit(0)
