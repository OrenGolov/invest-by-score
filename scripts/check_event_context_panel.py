"""Gate: the median historical response never travels without its association share.

The roadmap asks for "recent event, classification, historical analog count,
median historical response, current setup similarity". Every one has a producer
in E4/E6, so D3 is mostly assembly. What it must not do is quote the median as
though it were a clean number.

THE DECIDING MEASUREMENT, recomputed here: a third of comparable setups are not
comparable for the reason the panel implies. E5 classifies each remembered
response as event_associated or confounded, and excluding the confounded ones
shifts the median historical response by +1.36pp (median shift), up to 5.71pp,
and by more than one percentage point in 24 of 40 seeded analog sets.

So "median historical response: +2.7%" is a composite of the event's own
association AND whatever else moved those names.

INSUFFICIENT IS NOT ZERO. E6 refuses to summarise below EVENT_MEMORY_MIN_ANALOGS
and emits no median_response key at all. MEASURED, the live memory store on a
clean clone holds 0 rows, so this is the normal case rather than an edge one.

Verified to FAIL when any of these is reinjected:
  - a median reported with no event-associated share
  - a low association share with no caution
  - a refused summary carrying a 0.0 median
  - "no event" and "too few analogs" collapsed into one status
  - an unlabelled event treated as OBSERVED
  - analogs below the retrieval bar presented as comparable setups
  - the causation disclaimer stripped
  - the panel claiming to block trades
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    EVENT_CONTEXT_ASSOCIATION_CAUTION,
    EVENT_CONTEXT_BLOCKS_TRADES,
    EVENT_CONTEXT_FIELDS,
    EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE,
    EVENT_CONTEXT_SHOW_PROVENANCE,
    EVENT_CONTEXT_STATUS_INSUFFICIENT,
    EVENT_CONTEXT_STATUS_MEASURED,
    EVENT_CONTEXT_STATUS_NO_EVENT,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    MEMORY_PROVENANCE_INFERRED,
)
from core.event_context_panel import (  # noqa: E402
    EventContextError,
    build_event_context,
    context_problems,
    render_context,
)
from core.event_memory import (  # noqa: E402
    EventMemory,
    analog_summary,
    find_analogs,
)

FAILURES: list[str] = []
CURRENT = {"trend": 0.62, "momentum": 0.51, "volatility": 0.30}
EVENT = {
    "event_id": "e-now",
    "event_type": "earnings_beat",
    "direction": "positive",
    "published_time": "2026-09-19",
    "provenance": "observed",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def memories(seed=7, count=15):
    """Analogs whose confounded members moved differently from the rest."""
    rng = random.Random(seed)
    built = []
    for index in range(count):
        chart = {
            "trend": 0.6 + rng.gauss(0, 0.03),
            "momentum": 0.5 + rng.gauss(0, 0.03),
            "volatility": 0.3 + rng.gauss(0, 0.02),
        }
        associated = index % 3 != 0
        response = rng.gauss(0.03, 0.04) if associated else rng.gauss(-0.02, 0.04)
        built.append(
            EventMemory(
                event_id=f"e{index}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type="earnings_beat",
                direction="positive",
                chart_state=chart,
                response={"20d": {"abnormal_return": response}},
                attribution={"20d": "event_associated" if associated else "confounded"},
                provenance="observed",
            )
        )
    return built


def summary(seed=7):
    return analog_summary(
        find_analogs(CURRENT, "earnings_beat", memories=memories(seed)), "20d"
    )


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED.
    shifts = []
    for seed in range(1, 41):
        pool = memories(seed=seed)
        analogs = find_analogs(CURRENT, "earnings_beat", memories=pool)
        associated = [a for a in analogs if a["memory"].is_event_associated("20d")]
        if len(analogs) < EVENT_MEMORY_MIN_ANALOGS:
            continue
        if len(associated) < EVENT_MEMORY_MIN_ANALOGS:
            continue
        shifts.append(
            analog_summary(associated, "20d")["median_response"]
            - analog_summary(analogs, "20d")["median_response"]
        )
    check(
        len(shifts) >= 20,
        f"only {len(shifts)} seeds produced comparable analog sets; the "
        f"measurement needs a population",
    )
    material = sum(1 for s in shifts if abs(s) > 0.01)
    check(
        material > len(shifts) * 0.3,
        f"only {material} of {len(shifts)} analog sets shifted by more than a "
        f"percentage point when E5-confounded analogs were excluded. D3's "
        f"caution rests on that shift being material; if it has vanished the "
        f"design must be re-derived rather than kept",
    )

    # 2. THE MEDIAN NEVER TRAVELS ALONE.
    check(
        EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE,
        "the panel does not require an association share alongside the median",
    )
    panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
    check(
        context_problems(panel) == [],
        f"a well-formed panel reported problems: {context_problems(panel)}",
    )
    analogs = panel["analogs"]
    check(
        analogs["median_response"] is not None,
        "the reference set produced no median; the gate needs one under test",
    )
    check(
        analogs["event_associated_share"] is not None,
        "a median was reported with no event-associated share",
    )
    check(
        float(analogs["event_associated_share"]) < EVENT_CONTEXT_ASSOCIATION_CAUTION,
        f"the reference set is {analogs['event_associated_share']} associated, "
        f"at or above the {EVENT_CONTEXT_ASSOCIATION_CAUTION} caution "
        f"threshold; the gate needs a set that triggers the caution",
    )
    check(
        str(analogs.get("caution") or "").strip() != "",
        "a third of the evidence was attributed elsewhere and the panel "
        "carried no caution — the median reads as the event's own effect",
    )
    check(
        "attributed to the event itself" in panel["headline"],
        "the headline quotes a median without naming its association share",
    )

    # A median supplied without a share must be REFUSED at construction.
    raw = dict(summary())
    raw.pop("event_associated_share")
    try:
        build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=raw)
        FAILURES.append(
            "a median with no event-associated share was accepted; MEASURED, "
            "excluding confounded analogs shifts it by more than a percentage "
            "point in 24 of 40 cases"
        )
    except EventContextError:
        pass

    # 3. THE CAUTION CAN STAY QUIET. A warning on every panel says nothing.
    clean = dict(summary())
    clean["event_associated_share"] = 0.95
    clean_panel = build_event_context(
        "NVDA", "2026-09-22", event=EVENT, analogs=clean
    )
    check(
        "caution" not in clean_panel["analogs"],
        "a 95%-associated analog set still carried a caution — the threshold "
        "fires on everything and therefore means nothing",
    )
    check(
        context_problems(clean_panel) == [],
        f"a clean panel reported problems: {context_problems(clean_panel)}",
    )

    # 4. INSUFFICIENT IS NOT ZERO. This is the live case: the memory store is
    #    gitignored and holds 0 rows on any clone.
    empty = build_event_context(
        "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
    )
    check(
        empty["analogs"]["status"] == EVENT_CONTEXT_STATUS_INSUFFICIENT,
        f"an empty store produced {empty['analogs']['status']!r}",
    )
    for field in ("median_response", "setup_similarity", "event_associated_share"):
        check(
            empty["analogs"][field] is None,
            f"a refused summary carried {field}="
            f"{empty['analogs'][field]!r}; 0.0 there reads as 'no historical "
            f"move'",
        )
    check(
        str(empty["analogs"].get("reason") or "").strip() != "",
        "a refused summary carried no reason",
    )
    check(
        context_problems(empty) == [],
        f"an honestly-refused panel reported problems: {context_problems(empty)}",
    )

    # 5. NO EVENT IS NOT TOO FEW ANALOGS.
    check(
        EVENT_CONTEXT_STATUS_NO_EVENT != EVENT_CONTEXT_STATUS_INSUFFICIENT,
        "'nothing happened' and 'we could not learn from what happened' share "
        "a status",
    )
    none_panel = build_event_context("NVDA", "2026-09-22")
    check(
        none_panel["event"]["status"] == EVENT_CONTEXT_STATUS_NO_EVENT,
        "a panel with no event did not say so",
    )
    for field in ("event_id", "event_type", "direction"):
        check(
            none_panel["event"][field] is None,
            f"a no_event block carried {field}={none_panel['event'][field]!r}",
        )
    check(
        "no recent event" in none_panel["headline"],
        "the no-event headline does not say there was no event",
    )

    # 6. PROVENANCE PER EVENT. An INFERRED event must not read as OBSERVED.
    check(EVENT_CONTEXT_SHOW_PROVENANCE, "provenance is not shown")
    inferred = build_event_context(
        "NVDA",
        "2026-09-22",
        event=dict(EVENT, provenance=MEMORY_PROVENANCE_INFERRED),
    )
    check(
        inferred["event"]["inferred"] is True,
        "an inferred event was not marked as inferred",
    )
    check(
        "INFERRED" in str(inferred["event"].get("provenance_note") or ""),
        "an inferred event carries no note explaining what is uncertain",
    )
    unlabelled = dict(EVENT)
    unlabelled.pop("provenance")
    unlabelled_panel = build_event_context("NVDA", "2026-09-22", event=unlabelled)
    check(
        unlabelled_panel["event"]["provenance"] is None,
        "an unlabelled event was given a provenance",
    )
    check(
        "not treated as observed"
        in str(unlabelled_panel["event"].get("provenance_note") or ""),
        "an unlabelled event was laundered into an observed one",
    )

    # 7. THE ROADMAP'S FIVE FIELDS ALL APPEAR.
    for field in EVENT_CONTEXT_FIELDS:
        check(
            field in ("recent_event", "classification")
            or field in panel["analogs"],
            f"the roadmap names {field!r} and the panel does not carry it",
        )
    check(
        panel["event"]["event_type"] is not None,
        "the recent event has no classification",
    )

    # 8. ASSOCIATION IS NOT CAUSATION, stated in the payload.
    check(
        "not a forecast" in panel["disclaimer"]
        and "caused" in panel["disclaimer"],
        "the panel does not state that this is association, not a forecast "
        "and not causation",
    )

    # 9. THE PANEL DECIDES NOTHING.
    check(
        not EVENT_CONTEXT_BLOCKS_TRADES and not panel["blocks_trades"],
        "the event context panel claims to block trades",
    )

    # 10. THE RENDER KEEPS THE REFUSALS AND STAYS READABLE.
    lines = render_context(panel)
    check(lines[0].strip() == panel["headline"], "the headline is not rendered first")
    check("!!" in "\n".join(lines), "the caution does not reach the render")
    check(
        "observed" in "\n".join(render_context(none_panel) + lines),
        "provenance does not reach the render",
    )
    empty_line = [l for l in render_context(empty) if "analogs" in l][0]
    check(
        "median —" in empty_line,
        f"an absent median rendered as {empty_line!r} rather than as absent",
    )
    check(
        "0.00%" not in empty_line,
        "an absent median rendered as 0.00%",
    )
    for line in lines:
        check(len(line) < 200, f"a rendered line is {len(line)} characters long")

    # 11. MALFORMED INPUT IS REFUSED.
    for bad_kwargs, label in (
        ({"event": "earnings"}, "a non-mapping event"),
        ({"analogs": "lots"}, "a non-mapping summary"),
        ({"analogs": {"status": "probably_fine"}}, "an unknown analog status"),
    ):
        try:
            build_event_context("NVDA", "2026-09-22", **bad_kwargs)
            FAILURES.append(f"{label} was accepted")
        except EventContextError:
            pass
    try:
        build_event_context("", "2026-09-22")
        FAILURES.append("a panel with no ticker was accepted")
    except EventContextError:
        pass

    # 12. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda p: p["analogs"].pop("caution"),
            "a low association share with no caution",
        ),
        (
            lambda p: p["analogs"].update(
                {"setup_similarity": EVENT_MEMORY_MIN_SIMILARITY - 0.2}
            ),
            "analogs below the retrieval bar",
        ),
        (lambda p: p["analogs"].update({"analog_count": 2}), "a summary below the floor"),
        (lambda p: p.update({"disclaimer": "historical analogs"}), "a stripped disclaimer"),
        (lambda p: p.update({"headline": ""}), "a silenced headline"),
        (lambda p: p.update({"blocks_trades": True}), "a panel that blocks trades"),
        (lambda p: p["event"].pop("provenance"), "an event with no provenance"),
    ]
    for mutation, label in mutations:
        broken = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=summary()
        )
        mutation(broken)
        check(context_problems(broken) != [], f"{label} passed the contract check")

    broken = build_event_context(
        "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
    )
    broken["analogs"]["median_response"] = 0.0
    check(
        context_problems(broken) != [],
        "a refused summary carrying a 0.0 median passed the contract check",
    )
    broken = build_event_context("NVDA", "2026-09-22")
    broken["event"]["event_type"] = "earnings_beat"
    check(
        context_problems(broken) != [],
        "a no_event block carrying an event type passed the contract check",
    )

    if FAILURES:
        print("EVENT CONTEXT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("EVENT CONTEXT GATE: PASS")
    print(f"  excluding confounded analogs moved the median >1pp in "
          f"{material}/{len(shifts)} sets (recomputed)")
    print(f"  median {analogs['median_response']:+.4f} always travels with its "
          f"{analogs['event_associated_share']:.0%} association share")
    print(f"  insufficient analogs carry NO median (live store holds 0 rows)")
    print("  no_event, insufficient_analogs and measured stay distinct")
    print("  provenance per event: INFERRED never reads as OBSERVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
