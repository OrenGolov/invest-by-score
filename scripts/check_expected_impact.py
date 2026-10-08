"""Gate: impact is projected on risk only, on every dimension, ruling on none.

R2 sized a position and R4 adjusted it by confidence. Neither answers what the
portfolio LOOKS LIKE afterwards, which is the question a decision rests on.

THE DECIDING MEASUREMENT: volatility and concentration disagree about half the
time. Across 20 seeded markets x 2 candidates, 21 of 40 trades (52%) moved
portfolio volatility and risk concentration in OPPOSITE directions. On the
reference market a correlated candidate raised volatility +5.00% while
IMPROVING risk concentration (risk-HHI 0.5000 -> 0.3736), and a diversifier
cut volatility -20.56% while leaving concentration untouched (0.5005).

That measurement is RECOMPUTED here, not asserted from a comment. If the
dimensions ever stop disagreeing, reporting both loses its evidence and this
gate fails rather than letting a stale claim justify the design.

R5 PROJECTS RISK, NEVER RETURN. build_joint_forecast reports UNAVAILABLE and
SNAPSHOT_UNAVAILABLE_FIELDS already records that expected_return "needs a
trained model, and none exists". A 0.0 return impact would render as "flat"
under the Number(x ?? 0) idiom - the hazard F3, F4, F5, F6 and the snapshot
contract each guard against.

Verified to FAIL when any of these is reinjected:
  - an expected return projected from no model
  - the concentration dimension dropped, leaving volatility alone
  - a disagreement between dimensions silenced
  - before and after measured on different covariances
  - a direction contradicting the sign of its own change
  - a trade with no size projected as "no impact"
  - a NOT_EVALUATED dimension carrying a change
  - R5 declared to block trades
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    IMPACT_BLOCKS_TRADES,
    IMPACT_DIMENSION_CONCENTRATION,
    IMPACT_DIMENSION_LARGEST,
    IMPACT_DIMENSION_VOLATILITY,
    IMPACT_DIMENSIONS,
    IMPACT_DIRECTIONS,
    IMPACT_IMPROVES,
    IMPACT_MATERIAL_CONCENTRATION,
    IMPACT_MATERIAL_VOLATILITY,
    IMPACT_NEUTRAL,
    IMPACT_NOT_EVALUATED,
    IMPACT_PROJECTS_RETURN,
    IMPACT_WORSENS,
    SIZING_RISK_BUDGET,
)
from core.correlation_sizing import size_position  # noqa: E402
from core.expected_impact import (  # noqa: E402
    blend,
    concentration_of,
    impact_problems,
    impact_report,
    project_impact,
    render_impact,
)
from core.position_exposure import (  # noqa: E402
    aligned_returns,
    covariance,
    portfolio_variance,
    risk_contributions,
)

FAILURES: list[str] = []
HELD = {"HELD_A": 0.5, "HELD_B": 0.5}
SKEWED = {"BIG": 0.9, "SMALL": 0.1}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def reference(seed=11, sessions=500):
    rng = random.Random(seed)
    series = {"HELD_A": {}, "HELD_B": {}, "TWIN": {}, "DIVR": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        base = rng.gauss(0, 0.008)
        shock = rng.gauss(0, 0.018)
        series["HELD_A"][date] = base + shock + rng.gauss(0, 0.003)
        series["HELD_B"][date] = base + shock + rng.gauss(0, 0.003)
        series["TWIN"][date] = 1.4 * (base + shock)
        series["DIVR"][date] = rng.gauss(0, 0.009)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


def opposing(seed=5, sessions=600):
    """BIG dominates risk; CAND is correlated with it AND more volatile."""
    rng = random.Random(seed)
    series = {"BIG": {}, "SMALL": {}, "CAND": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        big = rng.gauss(0, 0.030)
        series["BIG"][date] = big
        series["SMALL"][date] = rng.gauss(0, 0.004)
        series["CAND"][date] = 1.3 * big + rng.gauss(0, 0.004)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


def doubling_down(seed=9, sessions=600):
    """Three independent names, one already the largest risk driver.

    Adding to BIG must WORSEN concentration. Without a case in this
    direction, a direction function that always returned IMPROVES would pass
    every other check here — MEASURED, it did, until this scenario was added.
    """
    rng = random.Random(seed)
    series = {"BIG": {}, "A": {}, "B": {}}
    for index in range(sessions):
        date = f"d{index:04d}"
        series["BIG"][date] = rng.gauss(0, 0.025)
        series["A"][date] = rng.gauss(0, 0.010)
        series["B"][date] = rng.gauss(0, 0.010)
    tickers, matrix = aligned_returns(series)
    return tickers, covariance(matrix)


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED.
    def hhi(shares):
        return sum(v * v for v in shares.values())

    disagreements = 0
    total = 0
    for seed in range(1, 21):
        tickers, cov = reference(seed=seed)
        base_vol = math.sqrt(portfolio_variance(HELD, tickers, cov))
        base_hhi = hhi(risk_contributions(HELD, tickers, cov))
        for ticker, size in (("TWIN", 0.1278), ("DIVR", 0.20)):
            after = blend(HELD, ticker, size)
            vol = math.sqrt(portfolio_variance(after, tickers, cov))
            concentration = hhi(risk_contributions(after, tickers, cov))
            total += 1
            if (vol > base_vol) != (concentration > base_hhi):
                disagreements += 1
    check(
        disagreements > total * 0.25,
        f"only {disagreements} of {total} trades moved volatility and "
        f"concentration in opposite directions. Reporting both dimensions "
        f"rests on that disagreement; if it has vanished the design must be "
        f"re-derived rather than kept",
    )

    # 2. A TRADE CAN RAISE VOLATILITY WHILE IMPROVING CONCENTRATION, and the
    #    report must say so rather than leaving the reader to spot it.
    tickers, cov = opposing()
    projection = project_impact(SKEWED, "CAND", 0.25, tickers, cov)
    volatility = projection["dimensions"][IMPACT_DIMENSION_VOLATILITY]
    concentration = projection["dimensions"][IMPACT_DIMENSION_CONCENTRATION]
    check(
        volatility["direction"] == IMPACT_WORSENS,
        f"the constructed opposing trade reported volatility "
        f"{volatility['direction']}, not WORSENS — the gate's own scenario no "
        f"longer demonstrates a disagreement",
    )
    check(
        concentration["direction"] == IMPACT_IMPROVES,
        f"the constructed opposing trade reported concentration "
        f"{concentration['direction']}, not IMPROVES",
    )
    check(
        projection["disagreement"]["disagree"],
        "the dimensions pointed opposite ways and the report did not say so "
        "— a reader seeing one direction would act on half the evidence",
    )
    check(
        IMPACT_DIMENSION_VOLATILITY in projection["disagreement"]["worsening"]
        and IMPACT_DIMENSION_CONCENTRATION
        in projection["disagreement"]["improving"],
        "the disagreement does not name which dimension went which way",
    )

    # 3. RISK ONLY, NEVER RETURN.
    check(
        not IMPACT_PROJECTS_RETURN,
        "R5 is configured to project an expected return, which no trained "
        "model can produce",
    )
    check(
        not projection["projects_return"],
        "a projection claims to project an expected return",
    )
    check(
        str(projection.get("return_reason") or "").strip() != "",
        "the refusal to project a return carries no reason",
    )
    for key in projection:
        check(
            "expected_return" not in str(key),
            f"an expected_return field appeared as {key!r}; a 0.0 there "
            f"renders as 'flat' to any consumer that coalesces nulls",
        )
    check(
        "expected_return" not in projection["dimensions"],
        "expected_return appeared among the projected dimensions",
    )

    # 4. EVERY DIMENSION IS REPORTED.
    check(
        IMPACT_DIMENSION_CONCENTRATION in IMPACT_DIMENSIONS,
        "concentration is not among the projected dimensions; volatility "
        "alone answers half the question",
    )
    for name in IMPACT_DIMENSIONS:
        detail = projection["dimensions"].get(name)
        check(detail is not None, f"dimension {name!r} was not reported")
        if detail:
            check(
                detail.get("direction") in IMPACT_DIRECTIONS,
                f"{name}: unknown direction {detail.get('direction')!r}",
            )
            check(
                str(detail.get("reason") or "").strip() != "",
                f"{name}: direction with no reason",
            )

    # 5. BEFORE AND AFTER ARE MEASURED ON ONE COVARIANCE. A difference must be
    #    attributable to the trade, not to a change of estimation window.
    ref_tickers, ref_cov = reference()
    ref = project_impact(HELD, "TWIN", 0.10, ref_tickers, ref_cov)
    direct_before = math.sqrt(portfolio_variance(HELD, ref_tickers, ref_cov))
    direct_after = math.sqrt(
        portfolio_variance(blend(HELD, "TWIN", 0.10), ref_tickers, ref_cov)
    )
    check(
        abs(ref["before"]["volatility"] - direct_before) < 1e-8,
        f"the reported before-volatility {ref['before']['volatility']} does "
        f"not match a direct measurement {direct_before}",
    )
    check(
        abs(ref["after"]["volatility"] - direct_after) < 1e-8,
        f"the reported after-volatility {ref['after']['volatility']} does not "
        f"match the blended portfolio {direct_after}",
    )

    # 6. A ZERO-SIZE TRADE CHANGES NOTHING, and a trade with NO size is not a
    #    zero-size trade.
    zero = project_impact(HELD, "TWIN", 0.0, ref_tickers, ref_cov)
    check(
        abs(zero["before"]["volatility"] - zero["after"]["volatility"]) < 1e-8,
        "a zero-size trade changed the portfolio",
    )
    unsized = impact_report(
        HELD, {"TWIN": {"verdict": "NO_TRADE", "adjusted_size": None}},
        ref_tickers, ref_cov,
    )
    unsized_projection = unsized["projections"]["TWIN"]
    check(
        unsized_projection["size"] is None,
        "a trade with no size was given one",
    )
    for name in IMPACT_DIMENSIONS:
        check(
            unsized_projection["dimensions"][name]["direction"]
            == IMPACT_NOT_EVALUATED,
            f"{name}: a trade with no size was projected as having no impact, "
            f"which is a different claim",
        )
    check(
        impact_problems(unsized) == [],
        f"an honestly-unsized report reported problems: "
        f"{impact_problems(unsized)}",
    )

    # 7. THE MATERIALITY THRESHOLDS AGREE WITH R2, and can stay quiet.
    check(
        IMPACT_MATERIAL_VOLATILITY == SIZING_RISK_BUDGET,
        f"the volatility materiality threshold {IMPACT_MATERIAL_VOLATILITY} "
        f"disagrees with R2's risk budget {SIZING_RISK_BUDGET}; the two "
        f"modules would mean different things by 'a lot'",
    )
    tiny = project_impact(HELD, "TWIN", 0.001, ref_tickers, ref_cov)
    check(
        tiny["dimensions"][IMPACT_DIMENSION_VOLATILITY]["direction"]
        == IMPACT_NEUTRAL,
        "a 0.1% position produced a material volatility direction — the "
        "threshold flags everything and therefore means nothing",
    )

    # 7b. EVERY DIRECTION MUST BE REACHABLE IN BOTH SENSES. Doubling down on
    #     the largest risk driver must WORSEN concentration. MEASURED, without
    #     this scenario a _direction that always returned IMPROVES passed the
    #     whole gate: every other case here happens to move DOWN, so the sign
    #     was never contradicted.
    dd_tickers, dd_cov = doubling_down()
    dd = project_impact(
        {"BIG": 0.34, "A": 0.33, "B": 0.33}, "BIG", 0.30, dd_tickers, dd_cov
    )
    dd_concentration = dd["dimensions"][IMPACT_DIMENSION_CONCENTRATION]
    dd_largest = dd["dimensions"][IMPACT_DIMENSION_LARGEST]
    check(
        dd_concentration["direction"] == IMPACT_WORSENS
        and dd_concentration["change"] > 0,
        f"doubling down on the largest risk driver reported concentration "
        f"{dd_concentration['direction']} with a change of "
        f"{dd_concentration['change']} — a direction that cannot report "
        f"WORSENS is not reporting a direction at all",
    )
    check(
        dd_largest["direction"] == IMPACT_WORSENS and dd_largest["change"] > 0,
        f"doubling down reported the largest risk share as "
        f"{dd_largest['direction']}",
    )
    check(
        impact_problems(dd) == [],
        f"a well-formed worsening projection reported problems: "
        f"{impact_problems(dd)}",
    )

    # 8. HHI OVER AN EMPTY SET IS NONE, NOT ZERO.
    check(
        concentration_of({}) is None,
        "an HHI of 0.0 was reported over an empty set, which reads as "
        "perfect diversification rather than unknown",
    )

    # 9. THE REAL SIZER FEEDS IT. Using R2's own output so the two cannot
    #    drift apart.
    proposals = {
        t: size_position(HELD, t, ref_tickers, ref_cov) for t in ("TWIN", "DIVR")
    }
    report = impact_report(HELD, proposals, ref_tickers, ref_cov)
    check(
        impact_problems(report) == [],
        f"a well-formed report reported problems: {impact_problems(report)}",
    )
    check(
        report["candidates"] == 2,
        f"{report['candidates']} candidates projected, expected 2",
    )
    # R4's adjusted size takes precedence over R2's raw size.
    adjusted = impact_report(
        HELD, {"TWIN": {"size": 0.20, "adjusted_size": 0.05}}, ref_tickers, ref_cov
    )
    check(
        abs(adjusted["projections"]["TWIN"]["size"] - 0.05) < 1e-6,
        "R4's adjusted size was ignored in favour of R2's raw size",
    )

    # 10. DESCRIBED, NOT ENFORCED.
    check(
        not IMPACT_BLOCKS_TRADES,
        "R5 claims to block trades — it projects an impact, and whether to "
        "trade is R7's decision",
    )

    # 11. A DISAGREEMENT IS VISIBLE IN THE RENDER.
    opposing_report = impact_report(SKEWED, {"CAND": {"size": 0.25}}, tickers, cov)
    text = "\n".join(render_impact(opposing_report))
    check(
        "worsens" in text.lower(),
        "a disagreeing trade rendered without naming what it worsens",
    )

    # 12. THE CONTRACT CHECK CAN FAIL.
    def fresh():
        return impact_report(HELD, {"TWIN": {"size": 0.10}}, ref_tickers, ref_cov)

    mutations = [
        (
            lambda r: r["projections"]["TWIN"]["dimensions"].pop(
                IMPACT_DIMENSION_CONCENTRATION
            ),
            "a report omitting concentration",
        ),
        (
            lambda r: r["projections"]["TWIN"]["dimensions"][
                IMPACT_DIMENSION_VOLATILITY
            ].update({"direction": "BETTER_ISH"}),
            "an unknown direction",
        ),
        (
            lambda r: r["projections"]["TWIN"]["dimensions"][
                IMPACT_DIMENSION_CONCENTRATION
            ].update({"direction": IMPACT_IMPROVES, "change": 0.5}),
            "IMPROVES with a positive change",
        ),
        (
            lambda r: r["projections"]["TWIN"]["dimensions"][
                IMPACT_DIMENSION_LARGEST
            ].update({"direction": IMPACT_NOT_EVALUATED, "change": 0.1}),
            "a NOT_EVALUATED carrying a change",
        ),
        (
            lambda r: r["projections"]["TWIN"]["dimensions"][
                IMPACT_DIMENSION_VOLATILITY
            ].update({"change": None}),
            "a direction with no change",
        ),
        (
            lambda r: r["projections"]["TWIN"]["dimensions"][
                IMPACT_DIMENSION_VOLATILITY
            ].update({"reason": "  "}),
            "a dimension with no reason",
        ),
        (lambda r: r.update({"blocks_trades": True}), "a report that blocks trades"),
        (lambda r: r.update({"projects_return": True}), "a report projecting return"),
        (lambda r: r.update({"return_reason": ""}), "a missing return reason"),
    ]
    for mutation, label in mutations:
        broken = fresh()
        mutation(broken)
        check(impact_problems(broken) != [], f"{label} passed the contract check")

    # A SILENCED DISAGREEMENT must be caught.
    broken = impact_report(SKEWED, {"CAND": {"size": 0.25}}, tickers, cov)
    broken["projections"]["CAND"]["disagreement"]["disagree"] = False
    check(
        impact_problems(broken) != [],
        "a silenced disagreement passed the contract check",
    )

    if FAILURES:
        print("EXPECTED IMPACT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("EXPECTED IMPACT GATE: PASS")
    print(f"  volatility and concentration disagreed on {disagreements}/{total} "
          f"trades (recomputed)")
    print("  a trade can raise volatility while improving concentration, and "
          "the report says so")
    print("  risk is projected, return is REFUSED (no trained model exists)")
    print(f"  materiality: volatility {IMPACT_MATERIAL_VOLATILITY} (= R2's "
          f"budget), concentration {IMPACT_MATERIAL_CONCENTRATION}")
    print("  impact projected, not enforced (R7 owns the decision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
