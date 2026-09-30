"""Gate: impact is measured from outcomes, never from the event's own claim.

"Alert when a high-impact event is detected." The obvious implementation reads
the event's `magnitude` field and fires above a threshold.

THE TRAP: magnitude is documented in core/event_contract.py as "a bounded,
unitless claim size - NOT an expected return". MEASURED, it is worse than
unvalidated - it appears in neither core/event_memory.py nor
core/event_study.py and is not an EventMemory field, so it has never been
compared against a single realized outcome. An alert keyed on it fires on an
assertion nobody has checked.

WHAT IS MEASURABLE, recomputed here: E6 stores the abnormal return per horizon.
Across seeded memories the median absolute 20d move differs by more than 4x
between event types, which is what makes "this KIND of event moves this name"
a real statement.

UNKNOWN IS NOT LOW. MEASURED, a median |move| estimate at n=3 spans a
five-fold range across resamples, so an event type below E6's analog floor
cannot be rated - and reporting it as LOW_IMPACT would silence exactly the
events the system has never seen before.

Verified to FAIL when any of these is reinjected:
  - the claimed magnitude consulted, or smuggled into the payload
  - a loud claim firing on a historically quiet event type
  - the absolute move replaced by the signed one, so symmetry cancels to zero
  - an unrated event type reported as LOW_IMPACT or silenced
  - a verdict rated below the analog floor
  - a HIGH_IMPACT below the threshold, or a LOW_IMPACT above it
  - the alert claiming to block trades
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    EVENT_IMPACT_BLOCKS_TRADES,
    EVENT_IMPACT_HIGH,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_MIN_ANALOGS,
    EVENT_IMPACT_MIN_MEDIAN_MOVE,
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    EVENT_IMPACT_UNKNOWN_IS_NOT_LOW,
    EVENT_IMPACT_USES_ABSOLUTE_MOVE,
    EVENT_IMPACT_USES_CLAIMED_MAGNITUDE,
    EVENT_MEMORY_MIN_ANALOGS,
)
from core.event_impact_alert import (  # noqa: E402
    EventImpactAlertError,
    event_impact_alert,
    impact_alert_problems,
    realized_impact,
    render_impact_alert,
)
from core.event_memory import EventMemory  # noqa: E402

FAILURES: list[str] = []

TYPES = {
    "earnings_beat": 0.055,
    "guidance_cut": 0.070,
    "analyst_note": 0.012,
    "minor_pr": 0.006,
}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def memories(seed=23, count=400):
    rng = random.Random(seed)
    built = []
    for index in range(count):
        event_type = rng.choice(list(TYPES))
        built.append(
            EventMemory(
                event_id=f"e{index}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type=event_type,
                direction="positive",
                response={"20d": {"abnormal_return": rng.gauss(0, TYPES[event_type])}},
                attribution={
                    "20d": "event_associated" if rng.random() < 0.7 else "confounded"
                },
                provenance="observed",
            )
        )
    return built


def event(event_type, **extra):
    payload = {
        "event_type": event_type,
        "ticker": "NVDA",
        "published_time": "2026-09-23",
    }
    payload.update(extra)
    return payload


def main() -> int:
    pool = memories()

    # 1. MAGNITUDE IS NOT A MEMORY FIELD, recomputed against the real class.
    sample = EventMemory(
        event_id="x",
        ticker="NVDA",
        published_time="2025-01-10",
        event_type="earnings_beat",
        direction="positive",
    )
    check(
        "magnitude" not in sample.to_dict(),
        "magnitude is now an EventMemory field; A3's refusal to read it rests "
        "on it never having been compared against an outcome, and must be "
        "re-derived rather than kept if that has changed",
    )
    check(
        not EVENT_IMPACT_USES_CLAIMED_MAGNITUDE,
        "the alert is configured to read the claimed magnitude",
    )

    # 2. THE CLAIM CANNOT OVERRIDE THE HISTORY, in either direction.
    loud_claim = event_impact_alert(event("minor_pr", magnitude=0.99), pool)
    check(
        loud_claim["verdict"] == EVENT_IMPACT_LOW and not loud_claim["fired"],
        f"a 0.99 claimed magnitude on a historically quiet type produced "
        f"{loud_claim['verdict']!r} fired={loud_claim['fired']}",
    )
    quiet_claim = event_impact_alert(event("guidance_cut", magnitude=0.01), pool)
    check(
        quiet_claim["verdict"] == EVENT_IMPACT_HIGH and quiet_claim["fired"],
        f"a 0.01 claimed magnitude suppressed a historically loud type: "
        f"{quiet_claim['verdict']!r}",
    )

    # 3. THE DECIDING MEASUREMENT, RECOMPUTED. Event type is informative.
    medians = {}
    for event_type in TYPES:
        impact = realized_impact(pool, event_type)
        check(impact["measured"], f"{event_type}: not measurable from the pool")
        medians[event_type] = impact["median_move"]
    spread = max(medians.values()) / min(medians.values())
    check(
        spread > 4.0,
        f"event types span only {spread:.1f}x in realized impact ({medians}); "
        f"A3 rests on type being informative and must be re-derived rather "
        f"than kept if that has vanished",
    )

    # 4. THE THRESHOLD SEPARATES THE GROUPS WITH ROOM ON BOTH SIDES.
    loud = min(medians[t] for t in ("earnings_beat", "guidance_cut"))
    quiet = max(medians[t] for t in ("analyst_note", "minor_pr"))
    check(
        quiet < EVENT_IMPACT_MIN_MEDIAN_MOVE < loud,
        f"the {EVENT_IMPACT_MIN_MEDIAN_MOVE} threshold does not sit between "
        f"the quiet group ({quiet:.4f}) and the loud one ({loud:.4f})",
    )
    for event_type in ("earnings_beat", "guidance_cut"):
        alert = event_impact_alert(event(event_type), pool)
        check(
            alert["verdict"] == EVENT_IMPACT_HIGH and alert["fired"],
            f"{event_type} produced {alert['verdict']!r} fired={alert['fired']}",
        )
        check(
            alert["severity"] == ALERT_SEVERITY_WARN,
            f"{event_type} fired at severity {alert['severity']!r}",
        )
    for event_type in ("analyst_note", "minor_pr"):
        alert = event_impact_alert(event(event_type), pool)
        check(
            alert["verdict"] == EVENT_IMPACT_LOW and not alert["fired"],
            f"{event_type} produced {alert['verdict']!r} fired={alert['fired']} "
            f"— the alert fires on quiet events and therefore means nothing",
        )
        check(alert["severity"] is None, f"{event_type}: quiet alert with severity")

    # 5. IMPACT IS NOT DIRECTION. A symmetric history must not cancel to zero.
    check(EVENT_IMPACT_USES_ABSOLUTE_MOVE, "impact is not measured on the absolute move")
    symmetric = [
        EventMemory(
            event_id=f"s{index}",
            ticker="NVDA",
            published_time="2025-01-10",
            event_type="volatile_event",
            direction="positive",
            response={"20d": {"abnormal_return": 0.06 if index % 2 == 0 else -0.06}},
            attribution={"20d": "event_associated"},
            provenance="observed",
        )
        for index in range(20)
    ]
    symmetric_impact = realized_impact(symmetric, "volatile_event")
    check(
        abs(symmetric_impact["median_move"] - 0.06) < 1e-6,
        f"a history of +/-6% moves reported a median of "
        f"{symmetric_impact['median_move']}; the signed mean would be ~0 and "
        f"read as harmless",
    )
    symmetric_alert = event_impact_alert(event("volatile_event"), symmetric)
    check(
        symmetric_alert["verdict"] == EVENT_IMPACT_HIGH,
        f"a symmetric +/-6% history produced {symmetric_alert['verdict']!r}",
    )

    # 6. UNKNOWN IS NOT LOW, and the floor is E6's.
    check(
        EVENT_IMPACT_MIN_ANALOGS == EVENT_MEMORY_MIN_ANALOGS,
        f"the analog floor {EVENT_IMPACT_MIN_ANALOGS} differs from E6's "
        f"{EVENT_MEMORY_MIN_ANALOGS}; a second floor drifts from the one E6 "
        f"already enforces",
    )
    unseen = event_impact_alert(event("never_seen_before"), pool)
    check(
        unseen["verdict"] == EVENT_IMPACT_UNKNOWN,
        f"an unseen event type produced {unseen['verdict']!r}",
    )
    check(
        EVENT_IMPACT_UNKNOWN_IS_NOT_LOW and unseen["fired"],
        "an event type the system has never seen was silenced — those are "
        "the ones most worth a human look",
    )
    check(
        unseen["severity"] == ALERT_SEVERITY_INFO,
        f"an unrated type fired at severity {unseen['severity']!r}",
    )
    for field in ("median_move", "p90_move", "event_associated_share"):
        check(
            unseen["impact"][field] is None,
            f"an unrated event type reported {field}="
            f"{unseen['impact'][field]!r}",
        )

    # The floor must bite exactly where E6 puts it.
    def rare(count):
        return [
            EventMemory(
                event_id=f"f{i}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type="rare_event",
                direction="positive",
                response={"20d": {"abnormal_return": 0.08}},
                attribution={"20d": "event_associated"},
                provenance="observed",
            )
            for i in range(count)
        ]

    below = event_impact_alert(event("rare_event"), rare(EVENT_IMPACT_MIN_ANALOGS - 1))
    check(
        below["verdict"] == EVENT_IMPACT_UNKNOWN,
        f"{EVENT_IMPACT_MIN_ANALOGS - 1} analogs produced "
        f"{below['verdict']!r} rather than UNKNOWN",
    )
    at_floor = event_impact_alert(event("rare_event"), rare(EVENT_IMPACT_MIN_ANALOGS))
    check(
        at_floor["verdict"] == EVENT_IMPACT_HIGH,
        f"{EVENT_IMPACT_MIN_ANALOGS} analogs produced {at_floor['verdict']!r} "
        f"rather than a rating",
    )

    # 7. NO EVENT IS ITS OWN ANSWER.
    none_alert = event_impact_alert(None, pool)
    check(
        none_alert["verdict"] == EVENT_IMPACT_NO_EVENT and not none_alert["fired"],
        f"no event produced {none_alert['verdict']!r}",
    )
    check(none_alert["impact"] is None, "NO_EVENT carried an impact measurement")
    untyped = event_impact_alert({"ticker": "NVDA"}, pool)
    check(
        untyped["verdict"] == EVENT_IMPACT_NO_EVENT,
        f"an untyped event produced {untyped['verdict']!r}",
    )

    # 8. EVERY ALERT PRODUCED IS CONTRACT-CLEAN.
    for alert in (loud_claim, quiet_claim, unseen, below, at_floor, none_alert, untyped):
        check(
            impact_alert_problems(alert) == [],
            f"{alert['verdict']}: a well-formed alert reported problems: "
            f"{impact_alert_problems(alert)}",
        )

    # 9. MALFORMED INPUT IS REFUSED.
    try:
        realized_impact(pool, "")
        FAILURES.append("an empty event type was accepted")
    except EventImpactAlertError:
        pass
    try:
        event_impact_alert("earnings", pool)
        FAILURES.append("a non-mapping event was accepted")
    except EventImpactAlertError:
        pass
    for bad in (0.0, 1.0, -0.1):
        try:
            event_impact_alert(event("earnings_beat"), pool, threshold=bad)
            FAILURES.append(f"a threshold of {bad} was accepted")
        except EventImpactAlertError:
            pass

    # 10. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not EVENT_IMPACT_BLOCKS_TRADES and not quiet_claim["blocks_trades"],
        "the alert claims to block trades",
    )

    # 11. THE RENDER KEEPS ABSENCES ABSENT.
    text = "\n".join(render_impact_alert(quiet_claim))
    check(EVENT_IMPACT_HIGH in text, "the verdict does not reach the render")
    check("FIRED" in text, "a fired alert is not marked in the render")
    unseen_line = render_impact_alert(unseen)[0]
    check(
        "median —" in unseen_line,
        f"an unrated type rendered as {unseen_line!r} rather than as absent",
    )
    check("0.00%" not in unseen_line, "an unrated type rendered a 0.00% median")
    for line in render_impact_alert(quiet_claim):
        check(len(line) < 250, f"a rendered line is {len(line)} characters long")

    # 12. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda a: a.update({"uses_claimed_magnitude": True}),
            "an alert reading the claimed magnitude",
        ),
        (lambda a: a.update({"claimed_magnitude": 0.9}), "a smuggled magnitude"),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"verdict": "PROBABLY"}), "an unknown verdict"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"magnitude_reason": ""}), "a missing magnitude reason"),
        (
            lambda a: a["impact"].update({"median_move": -0.05}),
            "a negative median move",
        ),
        (lambda a: a.update({"fired": False, "severity": None}), "a silenced HIGH_IMPACT"),
    ]
    for mutation, label in mutations:
        broken = event_impact_alert(event("guidance_cut"), pool)
        mutation(broken)
        check(impact_alert_problems(broken) != [], f"{label} passed the contract check")

    broken = event_impact_alert(event("minor_pr"), pool)
    broken["verdict"] = EVENT_IMPACT_HIGH
    broken["fired"] = True
    broken["severity"] = ALERT_SEVERITY_WARN
    check(
        impact_alert_problems(broken) != [],
        "a HIGH_IMPACT below the threshold passed the contract check",
    )
    broken = event_impact_alert(event("never_seen_before"), pool)
    broken["impact"]["median_move"] = 0.0
    check(
        impact_alert_problems(broken) != [],
        "an UNKNOWN carrying a 0.0 median passed the contract check",
    )
    broken = event_impact_alert(event("never_seen_before"), pool)
    broken["fired"] = False
    broken["severity"] = None
    check(
        impact_alert_problems(broken) != [],
        "a silenced unrated event type passed the contract check",
    )

    if FAILURES:
        print("EVENT IMPACT ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("EVENT IMPACT ALERT GATE: PASS")
    print("  magnitude is not an EventMemory field, so it is never read "
          "(recomputed)")
    print(f"  event types span {spread:.1f}x in realized impact; the "
          f"{EVENT_IMPACT_MIN_MEDIAN_MOVE:.1%} threshold sits between "
          f"{quiet:.2%} and {loud:.2%}")
    print("  a +/-6% symmetric history is HIGH_IMPACT, not a cancelled zero")
    print(f"  UNKNOWN is not LOW: below {EVENT_IMPACT_MIN_ANALOGS} analogs an "
          f"event fires for a human look")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
