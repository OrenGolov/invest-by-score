"""Gate: the thesis break alert watches the evidence, not the score.

"Alert when evidence materially contradicts the existing thesis." The obvious
implementation watches the SCORE and fires when it falls. MEASURED, that alert
is silent on most real thesis breaks.

THE DECIDING MEASUREMENT, recomputed here: W1 attributes every score to three
evidence buckets - operational, narrative and macro_shock. Over 20,000 sampled
bucket pairs, those moving the total score by less than the 0.25 support
threshold contain a BUCKET REVERSAL about 59% of the time - a bucket that went
from supporting the case to opposing it, or the reverse.

The concrete case, from the same arithmetic:

    bucket          before          after
    operational     +1.40 supports  -1.20 opposes
    narrative       -0.10 neutral   +2.50 supports
    macro_shock     +0.20 neutral   +0.20 neutral
    SCORE           +1.50           +1.50   delta +0.00

A score-only alert sees +0.00 and says nothing while the thesis inverts from
operational to narrative. That is a different investment with the same number.

A BUCKET AT 0.0 IS NOT A BUCKET THAT OPPOSES. score_engine itself gives three
reasons a bucket totals exactly 0.0, and two are an ABSENCE of evidence.
Reading absence as contradiction fires a thesis break every time a feed goes
quiet.

Verified to FAIL when any of these is reinjected:
  - the score used as the trigger instead of the buckets
  - a reversal reported as NONE
  - a reversal firing at info rather than warn
  - unmeasured evidence given a stance
  - a carrier change collapsed into no change
  - a reversal list that does not match the buckets that reversed
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
    ATTRIBUTION_BUCKETS,
    ATTRIBUTION_SUPPORT_THRESHOLD,
    THESIS_ALERT_BLOCKS_TRADES,
    THESIS_ALERT_WATCHES_BUCKETS,
    THESIS_BREAK_APPEARED,
    THESIS_BREAK_CARRIER,
    THESIS_BREAK_DISAPPEARED,
    THESIS_BREAK_NONE,
    THESIS_BREAK_NOT_EVALUATED,
    THESIS_BREAK_REVERSAL,
    THESIS_BREAK_WITHDRAWN,
    THESIS_REVERSAL_IS_MATERIAL,
    THESIS_SUPPORT_THRESHOLD,
    THESIS_ZERO_IS_NOT_OPPOSITION,
)
from core.thesis_alert import (  # noqa: E402
    STANCE_NEUTRAL,
    STANCE_OPPOSES,
    STANCE_SUPPORTS,
    STANCE_UNMEASURED,
    ThesisAlertError,
    render_thesis_alert,
    stance_of,
    thesis_alert_problems,
    thesis_break_alert,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def attribution(operational, narrative, macro, *, measured=None, ticker="NVDA"):
    measured = measured or {}
    buckets = {}
    for name, total in (
        ("operational", operational),
        ("narrative", narrative),
        ("macro_shock", macro),
    ):
        entry = {"total": total}
        if name in measured:
            entry["measured"] = measured[name]
        buckets[name] = entry
    return {
        "buckets": buckets,
        "score": sum(v for v in (operational, narrative, macro) if v is not None),
        "ticker": ticker,
    }


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED.
    rng = random.Random(11)
    flat = 0
    reversed_count = 0
    for _ in range(20000):
        before = [rng.uniform(-2, 2) for _ in range(3)]
        after = [rng.uniform(-2, 2) for _ in range(3)]
        if abs(sum(after) - sum(before)) >= ATTRIBUTION_SUPPORT_THRESHOLD:
            continue
        flat += 1
        if any(
            {stance_of(a), stance_of(b)} == {STANCE_SUPPORTS, STANCE_OPPOSES}
            for a, b in zip(before, after)
        ):
            reversed_count += 1
    check(flat > 200, f"only {flat} flat-score pairs sampled; too few to measure")
    share = reversed_count / flat if flat else 0.0
    check(
        share > 0.30,
        f"only {share:.0%} of flat-score periods contained a bucket reversal; "
        f"A5's design rests on that being common and must be re-derived "
        f"rather than kept if it has vanished",
    )

    # 2. THE CONCRETE CASE FIRES ON A ZERO SCORE MOVE.
    inverted = thesis_break_alert(
        attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
    )
    check(
        abs(inverted["score_delta"]) < 1e-9,
        f"the reference pair moves the score {inverted['score_delta']}; the "
        f"gate needs a move a score-only alert would ignore",
    )
    check(
        inverted["kind"] == THESIS_BREAK_REVERSAL and inverted["fired"],
        f"a thesis inverting at an identical score produced "
        f"{inverted['kind']!r} fired={inverted['fired']}",
    )
    check(
        inverted["severity"] == ALERT_SEVERITY_WARN,
        f"a reversal fired at {inverted['severity']!r} rather than warn",
    )
    check(
        inverted["reversals"] == ["operational"],
        f"the reversal names {inverted['reversals']} rather than operational",
    )
    check(
        inverted["previous_carrier"] == "operational"
        and inverted["current_carrier"] == "narrative",
        f"the carrier went {inverted['previous_carrier']} -> "
        f"{inverted['current_carrier']}, not operational -> narrative",
    )
    check(
        THESIS_ALERT_WATCHES_BUCKETS and inverted["watches_buckets"],
        "the alert does not declare that it watches the buckets",
    )
    check(THESIS_REVERSAL_IS_MATERIAL, "a reversal is not configured as material")

    # 3. THE THRESHOLD IS W1's, not a second one.
    check(
        THESIS_SUPPORT_THRESHOLD == ATTRIBUTION_SUPPORT_THRESHOLD,
        f"the support threshold {THESIS_SUPPORT_THRESHOLD} differs from W1's "
        f"{ATTRIBUTION_SUPPORT_THRESHOLD}; the two would disagree about the "
        f"same number",
    )
    check(
        stance_of(THESIS_SUPPORT_THRESHOLD) == STANCE_NEUTRAL,
        "a bucket exactly at the threshold was called supporting",
    )

    # 4. ABSENCE IS NOT CONTRADICTION.
    check(THESIS_ZERO_IS_NOT_OPPOSITION, "a 0.0 bucket is being read as opposition")
    check(
        stance_of(0.0, measured=False) == STANCE_UNMEASURED,
        "unmeasured evidence was given a stance",
    )
    check(
        stance_of(0.0) == STANCE_NEUTRAL,
        "a genuine 0.0 measurement was reported as unmeasured",
    )
    quiet_feed = thesis_break_alert(
        attribution(1.40, 0.90, 0.20),
        attribution(1.40, 0.0, 0.20, measured={"narrative": False}),
    )
    check(
        quiet_feed["kind"] != THESIS_BREAK_REVERSAL,
        f"a narrative feed going quiet produced {quiet_feed['kind']!r}; "
        f"absence of evidence was read as the narrative opposing the thesis",
    )
    check(
        quiet_feed["after"]["narrative"]["stance"] == STANCE_UNMEASURED,
        "a quiet feed was given a stance",
    )

    # 5. THE OTHER BREAK KINDS ARE REACHABLE AND DISTINCT.
    carrier = thesis_break_alert(
        attribution(1.40, 0.10, 0.20), attribution(0.10, 1.40, 0.20)
    )
    check(
        carrier["kind"] == THESIS_BREAK_CARRIER and carrier["fired"],
        f"a carrier change produced {carrier['kind']!r}",
    )
    withdrawn = thesis_break_alert(
        attribution(2.00, 0.90, 0.20), attribution(2.00, 0.10, 0.20)
    )
    check(
        withdrawn["kind"] == THESIS_BREAK_WITHDRAWN and withdrawn["fired"],
        f"a pure withdrawal produced {withdrawn['kind']!r}",
    )
    check(
        withdrawn["severity"] == ALERT_SEVERITY_INFO,
        f"a withdrawal fired at {withdrawn['severity']!r}; it is weaker than "
        f"a reversal",
    )
    unchanged = thesis_break_alert(
        attribution(1.40, 0.10, 0.20), attribution(1.45, 0.10, 0.20)
    )
    check(
        unchanged["kind"] == THESIS_BREAK_NONE and not unchanged["fired"],
        f"an unchanged thesis produced {unchanged['kind']!r} "
        f"fired={unchanged['fired']} — the alert fires on nothing and "
        f"therefore means nothing",
    )
    check(unchanged["severity"] is None, "a quiet alert carried a severity")

    # 6. AVAILABILITY IS NOT A BREAK.
    appeared = thesis_break_alert(
        {"buckets": {}, "score": None}, attribution(1.40, 0.10, 0.20)
    )
    check(
        appeared["kind"] == THESIS_BREAK_APPEARED and appeared["before"] is None,
        f"a thesis becoming attributable produced {appeared['kind']!r}",
    )
    disappeared = thesis_break_alert(
        attribution(1.40, 0.10, 0.20), {"buckets": {}, "score": None}
    )
    check(
        disappeared["kind"] == THESIS_BREAK_DISAPPEARED
        and disappeared["after"] is None,
        f"a thesis going away produced {disappeared['kind']!r}",
    )
    check(
        disappeared["severity"] == ALERT_SEVERITY_WARN,
        "a vanished attribution did not warn",
    )
    first = thesis_break_alert(None, attribution(1.40, 0.10, 0.20))
    check(
        first["kind"] == THESIS_BREAK_NOT_EVALUATED and not first["fired"],
        f"a first observation produced {first['kind']!r}",
    )

    # 7. EVERY ALERT PRODUCED IS CONTRACT-CLEAN.
    for alert in (
        inverted,
        quiet_feed,
        carrier,
        withdrawn,
        unchanged,
        appeared,
        disappeared,
        first,
    ):
        check(
            thesis_alert_problems(alert) == [],
            f"{alert['kind']}: a well-formed alert reported problems: "
            f"{thesis_alert_problems(alert)}",
        )

    # 8. MALFORMED INPUT IS REFUSED.
    for bad_call, label in (
        (
            lambda: thesis_break_alert(attribution(1.0, 0.0, 0.0), None),
            "no current attribution",
        ),
        (
            lambda: thesis_break_alert("bullish", attribution(1.0, 0.0, 0.0)),
            "a non-mapping attribution",
        ),
        (
            lambda: thesis_break_alert(
                attribution(1.0, 0.0, 0.0),
                attribution(1.0, 0.0, 0.0),
                threshold=0.0,
            ),
            "a zero threshold",
        ),
    ):
        try:
            bad_call()
            FAILURES.append(f"{label} was accepted")
        except ThesisAlertError:
            pass

    # 9. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not THESIS_ALERT_BLOCKS_TRADES and not inverted["blocks_trades"],
        "the alert claims to block trades",
    )

    # 10. THE RENDER SHOWS EVERY BUCKET AND BOTH SIDES.
    text = "\n".join(render_thesis_alert(inverted))
    for name in ATTRIBUTION_BUCKETS:
        check(name in text, f"{name} does not reach the render")
    check("FIRED" in text, "a fired alert is not marked in the render")
    check(
        "—" in "\n".join(render_thesis_alert(disappeared)),
        "a vanished attribution rendered a value",
    )
    for line in render_thesis_alert(inverted):
        check(len(line) < 250, f"a rendered line is {len(line)} characters long")

    # 11. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda a: a.update({"watches_buckets": False}),
            "an alert not watching the buckets",
        ),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"kind": "PROBABLY"}), "an unknown kind"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"severity": ALERT_SEVERITY_INFO}), "a reversal at info"),
        (
            lambda a: a.update(
                {"kind": THESIS_BREAK_NONE, "fired": False, "severity": None}
            ),
            "a silenced reversal",
        ),
        (lambda a: a.update({"reversals": []}), "a REVERSAL naming no bucket"),
        (lambda a: a.update({"withdrawn": ["astrology"]}), "an unknown bucket"),
    ]
    for mutation, label in mutations:
        broken = thesis_break_alert(
            attribution(1.40, -0.10, 0.20), attribution(-1.20, 2.50, 0.20)
        )
        mutation(broken)
        check(thesis_alert_problems(broken) != [], f"{label} passed the contract check")

    broken = thesis_break_alert(
        attribution(1.40, 0.90, 0.20),
        attribution(1.40, 0.0, 0.20, measured={"narrative": False}),
    )
    broken["after"]["narrative"]["stance"] = STANCE_OPPOSES
    check(
        thesis_alert_problems(broken) != [],
        "unmeasured evidence given a stance passed the contract check",
    )
    broken = thesis_break_alert(
        attribution(1.40, 0.10, 0.20), attribution(0.10, 1.40, 0.20)
    )
    broken["current_carrier"] = broken["previous_carrier"]
    check(
        thesis_alert_problems(broken) != [],
        "a CARRIER_CHANGED with the same carrier passed the contract check",
    )

    if FAILURES:
        print("THESIS BREAK ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("THESIS BREAK ALERT GATE: PASS")
    print(f"  {reversed_count}/{flat} ({share:.0%}) flat-score periods hide a "
          f"bucket reversal (recomputed)")
    print("  a thesis inverting at +0.00 score fires REVERSAL at warn")
    print("  a quiet feed is UNMEASURED, never opposition")
    print("  REVERSAL / CARRIER_CHANGED / SUPPORT_WITHDRAWN stay distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
