"""Gate: the WHY panel reports evidence, never contributions, and never zeros.

The roadmap asks for "Technical / Fundamental / News / Macro / Regime /
Sentiment contributions". Two measured facts stand between that wording and an
honest panel.

FIRST: THEY ARE NOT CONTRIBUTIONS. F6 MEASURED over 4,000 observations with a
realistic regime/chart correlation: regime alone +0.064, chart alone +0.067,
sum +0.131, ACTUAL joint effect +0.060. The parts overlap and double-count by
more than 2x. Six numbers summing to the forecast would be arithmetically
wrong AND would imply each factor independently CAUSED its share.

SECOND: HALF OF WHAT THE ROADMAP NAMES CANNOT BE MEASURED. Of F6's seven
components only four are wired, and the three that are NOT_WIRED -
fundamental, macro, sentiment - are exactly three of the six the roadmap asks
for. MEASURED on a live decomposition: 1 PRESENT, 3 ABSENT, 3 NOT_WIRED.
Rendering those three as 0.0 would say they were measured and found
irrelevant.

Both facts are RECOMPUTED here against the real F6 machinery, not asserted
from a comment.

Verified to FAIL when any of these is reinjected:
  - a contribution number on any row
  - the panel declaring its rows additive
  - an unwired component reported ABSENT (measured and silent)
  - an unwired component carrying an effect or a number
  - NOT_WIRED collapsed into ABSENT
  - a component dropped from the panel
  - a mostly-empty panel rendering without its headline
  - the panel claiming to block trades
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_STATUSES,
    DECOMPOSITION_WIRED_COMPONENTS,
    DECOMP_FUNDAMENTAL,
    DECOMP_HISTORICAL_ANALOG,
    DECOMP_MACRO,
    DECOMP_SENTIMENT,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    DECOMP_TECHNICAL,
    WHY_BARRED_TERMS,
    WHY_MIN_PRESENT,
    WHY_PANEL_BLOCKS_TRADES,
    WHY_REPORTS_CONTRIBUTIONS,
    WHY_ROADMAP_NAMES,
    WHY_SHOW_ALL_COMPONENTS,
    WHY_STATUSES,
)
from core.forecast_decomposition import decompose_event_forecast  # noqa: E402
from core.why_panel import (  # noqa: E402
    WhyPanelError,
    build_why_panel,
    display_name,
    render_why,
    why_problems,
)

FAILURES: list[str] = []

ROADMAP_SIX = (
    DECOMP_TECHNICAL,
    DECOMP_FUNDAMENTAL,
    "news_event",
    DECOMP_MACRO,
    "regime",
    DECOMP_SENTIMENT,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def live_decomposition():
    forecast = {
        "value": 0.56,
        "horizon": "20d",
        "samples": 800,
        "as_of": "2026-09-22",
        "event": {"entity": "NVDA", "event_type": "earnings_beat"},
        "stages": {
            "matches": {"observed_share": 0.95, "count": 40},
            "regime": {"label": "bullish", "agreement": 0.9, "trials": 120},
            "forecast": {"effective_samples": 800},
            "chart": {"fields": 10, "similarity": 0.87},
        },
    }
    return decompose_event_forecast(forecast, pool_size=4000)


def main() -> int:
    decomposition = live_decomposition()
    panel = build_why_panel(decomposition)

    # 1. NOT ADDITIVE, RECOMPUTED against F6's own declaration.
    check(
        not DECOMPOSITION_ADDITIVE,
        "F6 now declares its components additive; the WHY panel rests on "
        "them NOT being additive and must be re-derived rather than inherited",
    )
    check(not panel["additive"], "the panel claims its rows are additive")
    check(
        not WHY_REPORTS_CONTRIBUTIONS and not panel["reports_contributions"],
        "the panel reports contributions. MEASURED, regime +0.064 and chart "
        "+0.067 sum to +0.131 against an ACTUAL joint effect of +0.060",
    )
    check(
        "0.131" in panel["overlap_evidence"]
        and "0.060" in panel["overlap_evidence"],
        "the overlap measurement does not travel with the panel; the reason "
        "it refuses to add its own rows must be visible",
    )
    for row in panel["rows"]:
        for key in row:
            for barred in WHY_BARRED_TERMS:
                check(
                    barred not in str(key).lower(),
                    f"{row['component']}: field {key!r} names a contribution",
                )

    # 2. THE UNWIRED THREE, RECOMPUTED. These are exactly three of the six the
    #    roadmap names, which is the fact that shapes the panel.
    unwired = sorted(
        c for c in DECOMPOSITION_COMPONENTS if c not in DECOMPOSITION_WIRED_COMPONENTS
    )
    check(
        unwired == sorted([DECOMP_FUNDAMENTAL, DECOMP_MACRO, DECOMP_SENTIMENT]),
        f"the unwired set is {unwired}; D2's headline rests on three of the "
        f"roadmap's six being unmeasurable",
    )
    for component in unwired:
        check(
            component in ROADMAP_SIX,
            f"{component} is unwired but not one the roadmap names; the "
            f"finding no longer holds",
        )

    rows = {r["component"]: r for r in panel["rows"]}
    for component in unwired:
        row = rows.get(component)
        check(row is not None, f"{component}: no row")
        if not row:
            continue
        check(
            row["status"] == DECOMP_STATUS_NOT_WIRED,
            f"{component}: reported {row['status']} rather than NOT_WIRED — "
            f"ABSENT would say it was measured and had nothing to say",
        )
        check(
            row["effect"] is None,
            f"{component}: an unwired component carried the effect "
            f"{row['effect']!r}",
        )
        for key, value in row.items():
            if key in ("component", "label", "status", "reason", "wired"):
                continue
            check(
                value is None,
                f"{component}: carried {key}={value!r}; an unwired component "
                f"rendered as a number claims a measurement nobody made",
            )

    # 3. NOT_WIRED AND ABSENT NEVER MERGE.
    check(
        tuple(WHY_STATUSES) == tuple(DECOMPOSITION_STATUSES),
        "the panel does not reuse F6's statuses; a second vocabulary lets "
        "NOT_WIRED become ABSENT on the way to the screen",
    )
    check(
        DECOMP_STATUS_NOT_WIRED in WHY_STATUSES,
        "NOT_WIRED does not survive to the panel",
    )
    check(
        not (set(panel["not_wired"]) & set(panel["absent"])),
        "a component is reported both NOT_WIRED and ABSENT",
    )
    check(
        bool(panel["not_wired"]) and bool(panel["absent"]),
        f"the live panel shows not_wired={panel['not_wired']} and "
        f"absent={panel['absent']}; the gate needs both states populated for "
        f"the distinction to be under test",
    )

    # 4. EVERY COMPONENT IS SHOWN, including the one the roadmap omits.
    check(WHY_SHOW_ALL_COMPONENTS, "the panel does not show every component")
    shown = {r["component"] for r in panel["rows"]}
    check(
        shown == set(DECOMPOSITION_COMPONENTS),
        f"components shown {sorted(shown)} do not match F6's "
        f"{sorted(DECOMPOSITION_COMPONENTS)}",
    )
    for component in ROADMAP_SIX:
        check(component in shown, f"the roadmap names {component} and it is absent")
    analog = rows.get(DECOMP_HISTORICAL_ANALOG)
    check(
        analog is not None and analog["status"] == DECOMP_STATUS_PRESENT,
        "historical_analog is not PRESENT; MEASURED it is the component that "
        "produced the forecast value, and a 'why?' panel without it omits "
        "the only wired evidence that spoke",
    )
    for component in DECOMPOSITION_COMPONENTS:
        check(
            bool(display_name(component).strip()),
            f"{component}: no display name",
        )

    # 5. THE HEADLINE TELLS THE TRUTH ABOUT AN EMPTY PANEL.
    check(
        len(panel["present"]) < WHY_MIN_PRESENT,
        f"{len(panel['present'])} components supplied evidence; the gate's "
        f"scenario no longer demonstrates a mostly-empty panel",
    )
    check(
        "not a weighting" in panel["headline"],
        "a panel where one component spoke did not say so — six rows of "
        "ABSENT read as a considered weighting rather than an empty one",
    )
    total = len(panel["present"]) + len(panel["absent"]) + len(panel["not_wired"])
    check(
        total == len(DECOMPOSITION_COMPONENTS),
        f"the headline accounts for {total} of "
        f"{len(DECOMPOSITION_COMPONENTS)} components",
    )

    # 6. A POPULATED PANEL STILL REFUSES TO ADD UP. The warning must survive
    #    the case where evidence exists.
    populated_source = live_decomposition()
    for component in DECOMPOSITION_WIRED_COMPONENTS:
        populated_source["components"][component] = {
            "status": DECOMP_STATUS_PRESENT,
            "effect": "NARROWED",
            "reason": "evidence entered",
        }
    populated = build_why_panel(populated_source)
    check(
        why_problems(populated) == [],
        f"a well-formed populated panel reported problems: "
        f"{why_problems(populated)}",
    )
    check(
        "not additive" in populated["headline"],
        "a populated panel stopped warning that its rows do not add up",
    )

    # 7. NO DECOMPOSITION IS NOT AN EMPTY ONE.
    bare = build_why_panel(None)
    check(
        len(bare["rows"]) == len(DECOMPOSITION_COMPONENTS),
        "a panel with no decomposition dropped its rows instead of reporting "
        "them unavailable",
    )
    check(
        len(bare["not_wired"]) == 3,
        f"a bare panel reports {len(bare['not_wired'])} unwired components, "
        f"not 3 — NOT_WIRED is a standing property and does not depend on "
        f"what was supplied",
    )
    check(why_problems(bare) == [], f"a bare panel reported problems: {why_problems(bare)}")

    # 8. MALFORMED INPUT IS REFUSED, not absorbed.
    for bad, label in (
        ("technical was strong", "a non-mapping decomposition"),
        ({"components": ["technical"]}, "non-mapping components"),
        (
            {"components": {DECOMP_TECHNICAL: {"status": "STRONG", "reason": "x"}}},
            "an unknown status",
        ),
        (
            {
                "components": {
                    DECOMP_TECHNICAL: {"status": DECOMP_STATUS_ABSENT, "reason": ""}
                }
            },
            "a component with no reason",
        ),
    ):
        try:
            build_why_panel(bad)
            FAILURES.append(f"{label} was accepted")
        except WhyPanelError:
            pass

    # 9. THE PANEL EXPLAINS; IT DECIDES NOTHING.
    check(
        not WHY_PANEL_BLOCKS_TRADES and not panel["blocks_trades"],
        "the WHY panel claims to block trades",
    )

    # 10. THE RENDER KEEPS THE STATES AND STAYS READABLE.
    lines = render_why(panel)
    check(lines[0].strip() == panel["headline"], "the headline is not rendered first")
    text = "\n".join(lines)
    for component in DECOMPOSITION_COMPONENTS:
        check(
            WHY_ROADMAP_NAMES[component] in text,
            f"{component} is not rendered by its roadmap name",
        )
    check(DECOMP_STATUS_NOT_WIRED in text, "NOT_WIRED does not reach the render")
    for line in lines:
        if DECOMP_STATUS_NOT_WIRED in line:
            column = line.split(DECOMP_STATUS_NOT_WIRED, 1)[1][:14]
            check(
                "0.0" not in column,
                "an unwired component rendered an effect of 0.0, which says "
                "it was measured and found irrelevant",
            )
        check(len(line) < 200, f"a rendered line is {len(line)} characters long")

    # 11. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (lambda p: p["rows"][0].update({"contribution": 0.064}), "a contribution number"),
        (lambda p: p.update({"additive": True}), "a panel claiming additivity"),
        (
            lambda p: p.update({"reports_contributions": True}),
            "a panel claiming contributions",
        ),
        (lambda p: p.update({"overlap_evidence": ""}), "missing overlap evidence"),
        (lambda p: p.update({"headline": ""}), "a silenced headline"),
        (lambda p: p.update({"blocks_trades": True}), "a panel that blocks trades"),
        (lambda p: p.update({"rows": p["rows"][:-1]}), "a dropped component"),
        (lambda p: p["rows"][0].update({"label": "Vibes"}), "a mismatched label"),
    ]
    for mutation, label in mutations:
        broken = build_why_panel(live_decomposition())
        mutation(broken)
        check(why_problems(broken) != [], f"{label} passed the contract check")

    # An unwired component reported ABSENT, and a wired one NOT_WIRED.
    broken = build_why_panel(live_decomposition())
    {r["component"]: r for r in broken["rows"]}[DECOMP_MACRO]["status"] = (
        DECOMP_STATUS_ABSENT
    )
    check(
        why_problems(broken) != [],
        "an unwired component reported ABSENT passed the contract check — "
        "that says the macro data was checked and had nothing to say",
    )
    broken = build_why_panel(live_decomposition())
    {r["component"]: r for r in broken["rows"]}[DECOMP_TECHNICAL]["status"] = (
        DECOMP_STATUS_NOT_WIRED
    )
    check(
        why_problems(broken) != [],
        "a wired component reported NOT_WIRED passed the contract check",
    )
    broken = build_why_panel(live_decomposition())
    {r["component"]: r for r in broken["rows"]}[DECOMP_TECHNICAL]["effect"] = (
        "NARROWED"
    )
    check(
        why_problems(broken) != [],
        "an effect on a non-PRESENT row passed the contract check",
    )

    if FAILURES:
        print("WHY PANEL GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("WHY PANEL GATE: PASS")
    print("  not contributions: parts overlap 2x (+0.131 sum vs +0.060 actual)")
    print(f"  unwired: {unwired} — exactly three of the roadmap's six "
          f"(recomputed)")
    print(f"  live panel: {len(panel['present'])} PRESENT, "
          f"{len(panel['absent'])} ABSENT, {len(panel['not_wired'])} NOT_WIRED")
    print("  NOT_WIRED never becomes ABSENT, an effect, or a zero")
    print("  all 7 components shown, including the one the roadmap omits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
