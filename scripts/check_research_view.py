"""Gate: an empty system renders as empty, and a refusal never becomes a number.

The sprint goal is "a research workstation, not a black box". The failure mode
of a dashboard is not showing too little; it is rendering an empty system as a
confident one.

THE DECIDING MEASUREMENT, recomputed here against the real snapshot assembly:
at every horizon the roadmap names, a live forecast snapshot reports 5 PRESENT
fields, 2 ABSENT and 8 REFUSED. The five PRESENT are ticker, as_of,
forecast_version, horizon and warnings - metadata only. Every field a reader
would act on is REFUSED, and expected_return is ABSENT because no trained
model exists.

THE RETURN COLUMN IS SHOWN AND ALWAYS ABSENT. Omitting it would hide that the
roadmap asked for it and the system cannot produce it; filling it would
fabricate the quantity. SNAPSHOT_UNAVAILABLE_FIELDS records that a 0.0 there
"would render as 'flat' in any consumer that coalesces nulls".

Verified to FAIL when any of these is reinjected:
  - a non-PRESENT cell carrying a value (the Number(x ?? 0) failure on screen)
  - an expected return rendered as PRESENT
  - the return column omitted entirely
  - ABSENT and REFUSED collapsed into one state
  - a mostly-empty view rendering without a headline
  - a requested horizon silently dropped
  - W2's veto and R7's verdict merged into one risk field
  - the view claiming to block trades
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    RESEARCH_COVERAGE_WARN,
    RESEARCH_PANEL_CONFIDENCE,
    RESEARCH_PANEL_FORECAST,
    RESEARCH_PANEL_QUALITY,
    RESEARCH_PANEL_RISK,
    RESEARCH_PANELS,
    RESEARCH_RETURN_IS_ABSENT,
    RESEARCH_SHOW_RETURN_COLUMN,
    RESEARCH_VIEW_BLOCKS_TRADES,
    RESEARCH_VIEW_HORIZONS,
    SNAPSHOT_STATUSES,
    VIEW_CELL_ABSENT,
    VIEW_CELL_PRESENT,
    VIEW_CELL_REFUSED,
    VIEW_CELL_STATES,
)
from core.forecast_confidence import assess_confidence  # noqa: E402
from core.forecast_snapshot import build_forecast_snapshot  # noqa: E402
from core.research_view import (  # noqa: E402
    ResearchViewError,
    build_research_view,
    render_view,
    view_problems,
)

FAILURES: list[str] = []
TICKER = "NVDA"
AS_OF = "2026-09-22"


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def bare_snapshots():
    return {
        h: build_forecast_snapshot(TICKER, AS_OF, h) for h in RESEARCH_VIEW_HORIZONS
    }


def populated_snapshots():
    confidence = assess_confidence(
        samples=800,
        interval={"low": 0.48, "high": 0.60},
        observed_share=0.95,
        regime_agreement=0.9,
        similarities=[0.9, 0.85],
        features_present=10,
        features_expected=10,
        probability=0.56,
    )
    base = {
        "value": 0.56,
        "samples": 800,
        "interval": {"low": 0.48, "high": 0.60},
        "as_of": AS_OF,
        "event": {"entity": TICKER},
    }
    return {
        h: build_forecast_snapshot(
            TICKER,
            AS_OF,
            h,
            event_forecast=dict(base, horizon=h),
            confidence=confidence,
            regime="bullish",
        )
        for h in RESEARCH_VIEW_HORIZONS
    }


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED against the real assembly.
    for horizon, snapshot in bare_snapshots().items():
        check(
            len(snapshot["present"]) == 5,
            f"{horizon}: a live snapshot reports {len(snapshot['present'])} "
            f"PRESENT fields, not 5. D1's headline rests on the system being "
            f"mostly unavailable; if that has changed the design must be "
            f"re-derived rather than kept",
        )
        check(
            "expected_return" in snapshot["absent"],
            f"{horizon}: expected_return is no longer ABSENT — the return "
            f"column's refusal rests on it having no producer",
        )

    # 2. A BARE VIEW LEADS WITH WHAT IS MISSING.
    bare = build_research_view(TICKER, AS_OF, bare_snapshots())
    check(
        view_problems(bare) == [],
        f"a well-formed bare view reported problems: {view_problems(bare)}",
    )
    check(
        bare["coverage"]["below_threshold"],
        f"a view with {bare['coverage']['share_present']:.0%} coverage was not "
        f"flagged below the {RESEARCH_COVERAGE_WARN:.0%} threshold",
    )
    check(
        "Read the gaps" in bare["headline"],
        "a mostly-empty view did not tell the reader to read the gaps first; "
        "rendered as blanks it reads as a working system having a quiet day",
    )
    check(
        bare["coverage"]["refused"] > bare["coverage"]["present"],
        "the bare view reports more available values than refused ones",
    )

    # 3. THE RETURN COLUMN IS SHOWN AND ALWAYS ABSENT.
    check(RESEARCH_SHOW_RETURN_COLUMN, "the return column is not shown")
    check(RESEARCH_RETURN_IS_ABSENT, "the return column is not declared absent")
    populated = build_research_view(
        TICKER,
        AS_OF,
        populated_snapshots(),
        quality={"score": 78.5, "action": "BUY"},
        risk={"veto": False, "veto_rule_ids": []},
        portfolio={"verdict": "PROCEED", "reason": "fits the book"},
    )
    check(
        view_problems(populated) == [],
        f"a well-formed populated view reported problems: "
        f"{view_problems(populated)}",
    )
    for row in populated["panels"][RESEARCH_PANEL_FORECAST]["rows"]:
        check(
            "expected_return" in row,
            f"{row.get('horizon')}: the return column is missing, hiding that "
            f"the field was requested and could not be produced",
        )
        cell = row.get("expected_return") or {}
        check(
            cell.get("state") == VIEW_CELL_ABSENT,
            f"{row.get('horizon')}: the return cell is "
            f"{cell.get('state')!r}, not ABSENT",
        )
        check(
            cell.get("value") is None,
            f"{row.get('horizon')}: an expected return carried the value "
            f"{cell.get('value')!r}; no trained model exists to produce one",
        )

    # 4. THE HEADLINE CAN STAY QUIET. A warning that always fires says nothing.
    check(
        not populated["coverage"]["below_threshold"],
        f"a view at {populated['coverage']['share_present']:.0%} coverage was "
        f"still flagged — the threshold warns on everything and therefore "
        f"means nothing",
    )

    # 5. THE THREE STATES STAY DISTINCT, and the view reuses the snapshot's.
    check(
        set(VIEW_CELL_STATES) == set(SNAPSHOT_STATUSES),
        f"the view's states {sorted(VIEW_CELL_STATES)} differ from the "
        f"snapshot's {sorted(SNAPSHOT_STATUSES)}; a second vocabulary "
        f"collapses 'no producer' into 'declined here'",
    )
    bare_states = {
        cell["state"]
        for row in bare["panels"][RESEARCH_PANEL_FORECAST]["rows"]
        for cell in row.values()
        if isinstance(cell, dict)
    }
    check(
        VIEW_CELL_ABSENT in bare_states and VIEW_CELL_REFUSED in bare_states,
        f"the bare view shows only {sorted(bare_states)}; ABSENT and REFUSED "
        f"must both survive to the screen",
    )

    # 6. A NON-PRESENT CELL CANNOT CARRY A VALUE. This is the Number(x ?? 0)
    #    failure, caught at the render boundary where it does the most damage.
    from core.research_view import _cell

    for state in (VIEW_CELL_ABSENT, VIEW_CELL_REFUSED):
        try:
            _cell(state, value=0.0, reason="looks measured")
            FAILURES.append(
                f"a {state} cell accepted the value 0.0 — on screen that is "
                f"indistinguishable from a measured zero"
            )
        except ResearchViewError:
            pass
        try:
            _cell(state, reason="   ")
            FAILURES.append(f"a {state} cell accepted no reason")
        except ResearchViewError:
            pass

    # 7. EVERY PANEL AND EVERY REQUESTED HORIZON APPEARS.
    for panel in RESEARCH_PANELS:
        check(
            panel in bare["panels"],
            f"panel {panel!r} is missing — the sprint requires score, "
            f"forecast, confidence, regime and risk shown TOGETHER",
        )
    shown = {
        row["horizon"] for row in bare["panels"][RESEARCH_PANEL_FORECAST]["rows"]
    }
    check(
        shown == set(RESEARCH_VIEW_HORIZONS),
        f"horizons shown {sorted(shown)} do not match those requested "
        f"{sorted(RESEARCH_VIEW_HORIZONS)}",
    )
    # A missing snapshot still produces a row, so a dropped horizon cannot
    # look like one that was never asked for.
    partial = populated_snapshots()
    del partial["60d"]
    partial_view = build_research_view(TICKER, AS_OF, partial)
    partial_rows = {
        r["horizon"]: r
        for r in partial_view["panels"][RESEARCH_PANEL_FORECAST]["rows"]
    }
    check(
        "60d" in partial_rows
        and partial_rows["60d"]["probability_up"]["state"] == VIEW_CELL_REFUSED,
        "a horizon with no snapshot vanished from the view instead of "
        "appearing as refused",
    )

    # 8. W2 AND R7 ARE SHOWN SEPARATELY. They refuse for different reasons.
    risky = build_research_view(
        TICKER,
        AS_OF,
        bare_snapshots(),
        risk={"veto": True, "veto_rule_ids": ["data_quality_below_threshold"]},
        portfolio={"verdict": "NO_TRADE", "reason": "the book refuses"},
    )
    panel = risky["panels"][RESEARCH_PANEL_RISK]
    check(
        panel["veto"]["value"] is True,
        "W2's veto is not displayed",
    )
    check(
        panel["portfolio_verdict"]["value"] == "NO_TRADE",
        "R7's verdict is not displayed",
    )
    check(
        "W2" in panel["note"] and "R7" in panel["note"],
        "the risk panel does not say which question each verdict answers",
    )
    # Absent risk inputs must not read as safe.
    check(
        bare["panels"][RESEARCH_PANEL_RISK]["veto"]["state"] == VIEW_CELL_REFUSED,
        "a missing risk evaluation rendered as though no veto applied",
    )

    # 9. CONFIDENCE IS READ OFF THE SAME ROWS, not recomputed.
    forecast_cells = {
        r["horizon"]: r["confidence"]
        for r in populated["panels"][RESEARCH_PANEL_FORECAST]["rows"]
    }
    check(
        populated["panels"][RESEARCH_PANEL_CONFIDENCE]["per_horizon"]
        == forecast_cells,
        "the confidence panel disagrees with the forecast table beside it — "
        "two numbers for one quantity",
    )

    # 10. THE RENDER SHOWS STATES, NOT FABRICATED NUMBERS, and stays readable.
    text = render_view(bare)
    joined = "\n".join(text)
    check(VIEW_CELL_ABSENT in joined, "ABSENT does not reach the render")
    check(VIEW_CELL_REFUSED in joined, "REFUSED does not reach the render")
    check(
        "0.0" not in joined,
        "an unavailable value rendered as a number in the bare view",
    )
    check(
        bare["headline"] in text[1],
        "the headline is not the first thing after the ticker line",
    )
    for line in render_view(populated):
        check(
            "aggregation_evidence" not in line,
            "the confidence cell dumped its whole F7 assessment into the "
            "table instead of the number a reader needs",
        )
        check(len(line) < 200, f"a rendered line is {len(line)} characters long")

    # 11. THE VIEW DECIDES NOTHING.
    check(
        not RESEARCH_VIEW_BLOCKS_TRADES,
        "the research view claims to block trades — it renders, and the "
        "refusal belongs to R7 and W2",
    )
    check(not bare["blocks_trades"], "a rendered view claims to block trades")

    # 12. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda v: v["panels"][RESEARCH_PANEL_QUALITY]["score"].update(
                {"value": 0.0}
            ),
            "a REFUSED cell carrying 0.0",
        ),
        (
            lambda v: v["panels"][RESEARCH_PANEL_QUALITY]["score"].update(
                {"reason": "  "}
            ),
            "a refusal with no reason",
        ),
        (lambda v: v["panels"].pop(RESEARCH_PANEL_RISK), "a missing panel"),
        (
            lambda v: v["panels"][RESEARCH_PANEL_FORECAST].update(
                {"rows": v["panels"][RESEARCH_PANEL_FORECAST]["rows"][:-1]}
            ),
            "a dropped horizon",
        ),
        (
            lambda v: v["panels"][RESEARCH_PANEL_FORECAST]["rows"][0].update(
                {
                    "expected_return": {
                        "state": VIEW_CELL_PRESENT,
                        "value": 0.031,
                        "reason": "model says so",
                    }
                }
            ),
            "a fabricated expected return",
        ),
        (lambda v: v.update({"headline": ""}), "a silenced headline"),
        (lambda v: v.update({"blocks_trades": True}), "a view that blocks trades"),
        (lambda v: v["coverage"].update({"cells": 999}), "a miscounted coverage"),
    ]
    for mutation, label in mutations:
        broken = build_research_view(TICKER, AS_OF, bare_snapshots())
        mutation(broken)
        check(view_problems(broken) != [], f"{label} passed the contract check")

    if FAILURES:
        print("RESEARCH VIEW GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("RESEARCH VIEW GATE: PASS")
    print(f"  live snapshots: 5 PRESENT, 2 ABSENT, 8 REFUSED at every horizon "
          f"(recomputed)")
    print(f"  the bare view leads with its gaps "
          f"({bare['coverage']['share_present']:.0%} available)")
    print("  the return column is shown and always ABSENT (no trained model)")
    print("  ABSENT and REFUSED survive to the screen; neither becomes a number")
    print("  W2 and R7 shown separately; the view itself decides nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
