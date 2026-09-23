"""Gate: a label difference is not a regime change.

"Alert when the regime changes. E.g. BULLISH -> RISK_OFF." The obvious
implementation compares today's label to yesterday's and fires on any
difference. MEASURED on real data, that alert is mostly noise.

THE DECIDING MEASUREMENT, over 74 tickers and 74,600 labelled sessions from
the local 5-year ingest: the classifier flips 4,923 times - 16.6 per 252
sessions per ticker - and 26% OF THOSE RUNS LAST A SINGLE SESSION. A second,
slower expanding-window pass over 8 tickers agreed independently: 16.6 flips
per 252 sessions, 25% one-day runs.

    confirming sessions   alerts   per 252d   still held 5d later
                      1    4,923       16.6                   42%
                      2    3,016       10.2                   53%
                      3    2,272        7.6                   61%
                      5    1,628        5.5                   71%

At confirm=1 the MAJORITY of alerts - 58% - do not survive a week.

The real-data figures are quoted, not recomputed: the ingest is gitignored and
absent on every clone. The STRUCTURAL effect is recomputed here from a seeded
label sequence carrying the same flicker proportion.

Verified to FAIL when any of these is reinjected:
  - confirmation dropped, so a one-session flicker fires
  - PENDING collapsed into CONFIRMED, or made to fire
  - PENDING collapsed into NONE, hiding a transition in progress
  - an escalated label skipping confirmation
  - an escalated transition firing at info rather than warn
  - an uncomputable regime treated as a calm one
  - a confirmation run counted through an interruption
  - the alert claiming to block trades
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    REGIME_ALERT_BLOCKS_TRADES,
    REGIME_ALERT_CONFIRM_SESSIONS,
    REGIME_ALERT_ESCALATED_LABELS,
    REGIME_ALERT_PENDING_FIRES,
    REGIME_ALERT_REQUIRES_CONFIRMATION,
    REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME,
    REGIME_CHANGE_APPEARED,
    REGIME_CHANGE_CONFIRMED,
    REGIME_CHANGE_DISAPPEARED,
    REGIME_CHANGE_NONE,
    REGIME_CHANGE_NOT_EVALUATED,
    REGIME_CHANGE_PENDING,
    REGIME_LABELS,
    REGIME_RISKOFF_LABEL,
    REGIME_STRESS_LABEL,
)
from core.regime_alert import (  # noqa: E402
    RegimeAlertError,
    confirmation_run,
    regime_alert_problems,
    regime_change_alert,
    render_regime_alert,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def classification(label, recent=None, computable=True, ticker="NVDA"):
    return {
        "computable": computable,
        "label": label,
        "recent_labels": list(recent or []),
        "ticker": ticker,
    }


def flickering_sequence(seed=5, sessions=3000):
    """A label series carrying the measured flicker proportion."""
    rng = random.Random(seed)
    labels = list(REGIME_LABELS)
    sequence = []
    current = labels[0]
    while len(sequence) < sessions:
        run = rng.choices([1, 2, 3, 5, 8, 13], weights=[26, 12, 10, 20, 18, 14])[0]
        sequence.extend([current] * run)
        current = rng.choice([entry for entry in labels if entry != current])
    return sequence[:sessions]


def main() -> int:
    # 1. THE FLICKER STRUCTURE, RECOMPUTED from the seeded sequence.
    sequence = flickering_sequence()
    runs = []
    current, length = sequence[0], 1
    for label in sequence[1:]:
        if label == current:
            length += 1
        else:
            runs.append(length)
            current, length = label, 1
    runs.append(length)
    one_day = sum(1 for run in runs if run == 1) / len(runs)
    check(
        one_day > 0.15,
        f"only {one_day:.0%} of runs last a single session; A4's confirmation "
        f"rule rests on flicker being common, and must be re-derived rather "
        f"than kept if that has vanished",
    )

    # 2. CONFIRMATION CUTS THE ALERT RATE.
    def alerts(confirm):
        fired = 0
        last = sequence[0]
        pending, run = None, 0
        for label in sequence:
            if label != last:
                if pending == label:
                    run += 1
                else:
                    pending, run = label, 1
                if run >= confirm:
                    fired += 1
                    last, pending, run = label, None, 0
            else:
                pending, run = None, 0
        return fired

    unconfirmed = alerts(1)
    confirmed_count = alerts(REGIME_ALERT_CONFIRM_SESSIONS)
    check(
        confirmed_count < unconfirmed * 0.75,
        f"{REGIME_ALERT_CONFIRM_SESSIONS} confirming sessions cut alerts only "
        f"from {unconfirmed} to {confirmed_count}; the rule buys nothing",
    )
    check(
        REGIME_ALERT_REQUIRES_CONFIRMATION and REGIME_ALERT_CONFIRM_SESSIONS >= 2,
        f"the alert requires {REGIME_ALERT_CONFIRM_SESSIONS} confirming "
        f"session(s); a single-session rule is a quarter noise by "
        f"construction",
    )

    # 3. A ONE-SESSION FLICKER DOES NOT FIRE.
    flicker = regime_change_alert(
        classification("bullish"),
        classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
    )
    check(
        flicker["kind"] == REGIME_CHANGE_PENDING and not flicker["fired"],
        f"a one-session flicker produced {flicker['kind']!r} "
        f"fired={flicker['fired']}",
    )
    check(
        not REGIME_ALERT_PENDING_FIRES,
        "PENDING is configured to fire, which reinstates the unconfirmed "
        "alert MEASURED to reverse 58% of the time",
    )

    # 4. A CONFIRMED CHANGE FIRES, and reports its run.
    confirmed = regime_change_alert(
        classification("bullish"),
        classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
    )
    check(
        confirmed["kind"] == REGIME_CHANGE_CONFIRMED and confirmed["fired"],
        f"a confirmed change produced {confirmed['kind']!r} "
        f"fired={confirmed['fired']}",
    )
    check(
        confirmed["run"] >= REGIME_ALERT_CONFIRM_SESSIONS,
        f"a CONFIRMED change reports only {confirmed['run']} session(s)",
    )

    # 5. THE CONFIRMATION RUN COUNTS BACKWARDS AND STOPS AT AN INTERRUPTION.
    check(
        confirmation_run(["a", "b", "b", "b"], "b") == 3,
        "a trailing run of three was not counted as three",
    )
    check(
        confirmation_run(["b", "b", "a", "b"], "b") == 1,
        "an interrupted run was counted as persistence; that interruption is "
        "exactly the flicker being filtered",
    )
    check(
        confirmation_run(None, "bullish") == 0,
        "a missing history produced a confirmation run",
    )
    # Without history, a difference cannot be confirmed.
    unverifiable = regime_change_alert(
        classification("bullish"), classification("risk_off")
    )
    check(
        unverifiable["kind"] == REGIME_CHANGE_PENDING,
        f"a change with no label history produced {unverifiable['kind']!r}; "
        f"an unverifiable claim of persistence is not evidence",
    )

    # 6. ESCALATION IS IN SEVERITY, NOT IN SKIPPING EVIDENCE.
    check(
        REGIME_RISKOFF_LABEL in REGIME_ALERT_ESCALATED_LABELS
        and REGIME_STRESS_LABEL in REGIME_ALERT_ESCALATED_LABELS,
        f"the escalated set {REGIME_ALERT_ESCALATED_LABELS} omits risk_off or "
        f"stress",
    )
    check(
        confirmed["escalated"] and confirmed["severity"] == ALERT_SEVERITY_WARN,
        f"a confirmed transition into risk_off fired at "
        f"{confirmed['severity']!r} without escalation",
    )
    benign = regime_change_alert(
        classification("risk_off"),
        classification("bullish", ["risk_off"] * 17 + ["bullish"] * 3),
    )
    check(
        not benign["escalated"] and benign["severity"] == ALERT_SEVERITY_INFO,
        f"a confirmed transition into bullish escalated to "
        f"{benign['severity']!r}",
    )
    unconfirmed_stress = regime_change_alert(
        classification("bullish"),
        classification("stress", ["bullish"] * 19 + ["stress"]),
    )
    check(
        unconfirmed_stress["kind"] == REGIME_CHANGE_PENDING,
        f"an escalated label skipped confirmation, producing "
        f"{unconfirmed_stress['kind']!r}; the escalation is in severity, not "
        f"in the evidence required",
    )

    # 7. AN UNCOMPUTABLE REGIME IS NOT A CALM ONE.
    check(
        REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME,
        "an uncomputable regime is being treated as a regime",
    )
    appeared = regime_change_alert(
        classification(None, computable=False),
        classification("bullish", ["bullish"] * 3),
    )
    check(
        appeared["kind"] == REGIME_CHANGE_APPEARED and appeared["previous"] is None,
        f"a regime becoming computable produced {appeared['kind']!r}",
    )
    disappeared = regime_change_alert(
        classification("bullish"), classification(None, computable=False)
    )
    check(
        disappeared["kind"] == REGIME_CHANGE_DISAPPEARED
        and disappeared["current"] is None,
        f"a regime going away produced {disappeared['kind']!r}",
    )
    check(
        disappeared["severity"] == ALERT_SEVERITY_WARN,
        "a vanished regime measurement did not warn",
    )
    both_gone = regime_change_alert(
        classification(None, computable=False), classification(None, computable=False)
    )
    check(
        both_gone["kind"] == REGIME_CHANGE_NOT_EVALUATED and not both_gone["fired"],
        f"two uncomputable regimes produced {both_gone['kind']!r}",
    )
    # A STALE LABEL ON AN UNCOMPUTABLE CLASSIFICATION IS THE REAL HAZARD.
    # MEASURED: without this case, removing the computable guard entirely
    # still passed the gate, because every other uncomputable scenario
    # already carries label=None and the guard never decides. A classifier
    # that reports computable=False while leaving its previous label in place
    # would otherwise be read as a live regime and fire CONFIRMED.
    stale = regime_change_alert(
        classification("bullish"),
        {
            "computable": False,
            "label": "risk_off",
            "recent_labels": ["risk_off"] * 3,
            "ticker": "NVDA",
        },
    )
    check(
        stale["kind"] == REGIME_CHANGE_DISAPPEARED,
        f"an uncomputable classification carrying a stale 'risk_off' label "
        f"produced {stale['kind']!r}; a label that survived its own "
        f"computability must not be read as a live regime",
    )
    check(
        stale["current"] is None,
        f"a stale label {stale['current']!r} was reported as the current "
        f"regime although the classification is not computable",
    )
    check(
        regime_alert_problems(stale) == [],
        f"the stale-label alert reported problems: {regime_alert_problems(stale)}",
    )

    # 8. AN UNCHANGED REGIME IS QUIET, and a first observation is not a change.
    same = regime_change_alert(
        classification("bullish"), classification("bullish", ["bullish"] * 20)
    )
    check(
        same["kind"] == REGIME_CHANGE_NONE and not same["fired"],
        f"an unchanged regime produced {same['kind']!r} fired={same['fired']}",
    )
    first = regime_change_alert(None, classification("bullish"))
    check(
        first["kind"] == REGIME_CHANGE_NOT_EVALUATED and not first["fired"],
        f"a first observation produced {first['kind']!r}",
    )

    # 9. EVERY ALERT PRODUCED IS CONTRACT-CLEAN.
    for alert in (
        flicker,
        confirmed,
        benign,
        unconfirmed_stress,
        appeared,
        disappeared,
        both_gone,
        same,
        first,
        unverifiable,
    ):
        check(
            regime_alert_problems(alert) == [],
            f"{alert['kind']}: a well-formed alert reported problems: "
            f"{regime_alert_problems(alert)}",
        )

    # 10. MALFORMED INPUT IS REFUSED.
    for bad_call, label in (
        (
            lambda: regime_change_alert(
                classification("bullish"), classification("euphoric")
            ),
            "an unknown regime label",
        ),
        (
            lambda: regime_change_alert("bullish", classification("bullish")),
            "a non-mapping classification",
        ),
        (
            lambda: regime_change_alert(classification("bullish"), None),
            "no current classification",
        ),
        (
            lambda: regime_change_alert(
                classification("bullish"),
                classification("risk_off"),
                confirm_sessions=0,
            ),
            "a zero confirmation window",
        ),
    ):
        try:
            bad_call()
            FAILURES.append(f"{label} was accepted")
        except RegimeAlertError:
            pass

    # 11. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not REGIME_ALERT_BLOCKS_TRADES and not confirmed["blocks_trades"],
        "the alert claims to block trades",
    )

    # 12. THE RENDER CARRIES THE TRANSITION AND THE RUN.
    text = "\n".join(render_regime_alert(confirmed))
    check("bullish" in text and "risk_off" in text, "the transition is not rendered")
    check(
        f"{confirmed['run']}/{REGIME_ALERT_CONFIRM_SESSIONS}" in text,
        "the confirmation run is not rendered",
    )
    check("!!" in text, "an escalated change is not marked in the render")
    check("—" in render_regime_alert(disappeared)[0], "a vanished regime rendered a label")
    for line in render_regime_alert(confirmed):
        check(len(line) < 250, f"a rendered line is {len(line)} characters long")

    # 13. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda a: a.update({"requires_confirmation": False}),
            "an alert not requiring confirmation",
        ),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"kind": "PROBABLY"}), "an unknown kind"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"severity": ALERT_SEVERITY_INFO}), "an escalated info alert"),
        (
            lambda a: a.update({"kind": REGIME_CHANGE_NONE, "fired": False, "severity": None}),
            "a silenced transition",
        ),
        (lambda a: a.update({"run": 1}), "a CONFIRMED after one session"),
    ]
    for mutation, label in mutations:
        broken = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        mutation(broken)
        check(regime_alert_problems(broken) != [], f"{label} passed the contract check")

    broken = regime_change_alert(
        classification("bullish"),
        classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
    )
    broken["fired"] = True
    broken["severity"] = ALERT_SEVERITY_WARN
    check(
        regime_alert_problems(broken) != [],
        "a PENDING change that fired passed the contract check",
    )
    broken = regime_change_alert(
        classification("bullish"),
        classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
    )
    broken["kind"] = REGIME_CHANGE_PENDING
    broken["fired"] = False
    broken["severity"] = None
    check(
        regime_alert_problems(broken) != [],
        "a confirmed run reported as PENDING passed the contract check",
    )

    if FAILURES:
        print("REGIME CHANGE ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("REGIME CHANGE ALERT GATE: PASS")
    print(f"  {one_day:.0%} of regime runs last a single session (recomputed)")
    print(f"  {REGIME_ALERT_CONFIRM_SESSIONS} confirming sessions cut alerts "
          f"{unconfirmed} -> {confirmed_count} on the seeded sequence")
    print("  a one-session flicker is PENDING and does not fire")
    print("  risk_off/stress escalate in SEVERITY, never skipping confirmation")
    print("  an uncomputable regime is an availability change, not a calm market")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
