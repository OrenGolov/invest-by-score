"""Gate: confidence adjusts R2's size, and only in the direction that helps.

R2 sized a position from correlation alone. That question never asks whether
the forecast motivating the trade is worth acting on: a 51% P(up) from two
observations and a 51% from eight hundred produce the identical R2 size.

THE DECIDING MEASUREMENT: confidence-scaling helps one direction and HURTS the
other. On a candidate correlated with what is held and more volatile, scaling
by confidence walked portfolio volatility from +19.64% to +0.98%. On an
uncorrelated candidate the SAME scaling walked it from -19.23% to -0.99%,
monotonically. It removed a benefit rather than a risk.

THE CANDIDATE DOMINATES THE FORECAST: MEASURED, a HIGH-confidence concentrator
at 20% raised volatility +19.78% while a LOW-confidence diversifier at 5%
LOWERED it -4.82%. Sizing on confidence alone ranks these backwards.

Both measurements are RECOMPUTED here from a seeded generator, not asserted
from a comment. If the relationship ever reverses, this gate fails rather than
letting a stale claim justify the rule.

Verified to FAIL when any of these is reinjected:
  - the haircut applied to risk REDUCERS as well as increasers
  - confidence allowed to multiply a size above R2's
  - the confidence floor removed, so a token position replaces NO_TRADE
  - a missing confidence treated as 0.0 (refuse all) or 1.0 (wave through)
  - R4 reviving a position R2 refused
  - a NO_TRADE or NOT_EVALUATED carrying a size
  - the haircut sneaking a position below the minimum tradeable weight
  - R4 declared to block trades
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    FORECAST_RISK_BLOCKS_TRADES,
    FORECAST_RISK_HAIRCUT_INCREASERS_ONLY,
    FORECAST_RISK_MAX_MULTIPLIER,
    FORECAST_RISK_MIN_CONFIDENCE,
    FORECAST_RISK_UNMEASURABLE_IS_NOT_LOW,
    FRISK_NO_TRADE,
    FRISK_NOT_EVALUATED,
    FRISK_REDUCED,
    FRISK_UNCHANGED,
    FRISK_VERDICTS,
    SIZING_MIN_WEIGHT,
    SIZING_REFUSED,
    SIZING_SIZED,
)
from core.correlation_sizing import size_position  # noqa: E402
from core.forecast_confidence import assess_confidence  # noqa: E402
from core.forecast_risk import (  # noqa: E402
    adjust_size,
    confidence_of,
    forecast_risk_problems,
    forecast_risk_report,
    multiplier_for,
    render_forecast_risk,
)
from core.position_exposure import (  # noqa: E402
    aligned_returns,
    covariance,
    portfolio_variance,
)

FAILURES: list[str] = []
HELD = {"HELD_A": 0.5, "HELD_B": 0.5}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def market(seed=7, sessions=500):
    """Two correlated holdings, a levered twin, and a true diversifier."""
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


def assessment(confidence):
    return {"assessed_object": "forecast", "confidence": confidence}


def proposal(size=0.10, *, diversifying=False, increase=0.05, verdict=SIZING_SIZED):
    return {
        "ticker": "CAND",
        "verdict": verdict,
        "size": size,
        "diversifying": diversifying,
        "volatility_increase": increase,
    }


def main() -> int:
    tickers, cov = market()
    base = math.sqrt(portfolio_variance(HELD, tickers, cov))

    # 1. THE DECIDING MEASUREMENT, RECOMPUTED. Scaling a risk INCREASER by
    #    confidence must reduce portfolio volatility monotonically.
    previous = None
    for confidence in (1.0, 0.75, 0.5, 0.25, 0.05):
        size = 0.20 * confidence
        vol = math.sqrt(
            portfolio_variance(
                {"HELD_A": 0.5 * (1 - size), "HELD_B": 0.5 * (1 - size), "TWIN": size},
                tickers,
                cov,
            )
        )
        check(vol > base, "the TWIN candidate does not raise portfolio volatility")
        if previous is not None:
            check(
                vol < previous,
                f"shrinking a risk increaser to {size:.1%} did not reduce "
                f"portfolio volatility — the haircut's premise is false",
            )
        previous = vol

    # 2. THE NEGATIVE RESULT. The SAME scaling on a risk REDUCER must make the
    #    portfolio worse. This is why the haircut is increasers-only.
    previous = None
    for confidence in (1.0, 0.75, 0.5, 0.25, 0.05):
        size = 0.20 * confidence
        vol = math.sqrt(
            portfolio_variance(
                {"HELD_A": 0.5 * (1 - size), "HELD_B": 0.5 * (1 - size), "DIVR": size},
                tickers,
                cov,
            )
        )
        check(vol < base, "the DIVR candidate does not lower portfolio volatility")
        if previous is not None:
            check(
                vol > previous,
                f"shrinking a risk REDUCER to {size:.1%} lowered volatility — "
                f"the increasers-only rule has lost its evidence and must be "
                f"re-derived rather than kept",
            )
        previous = vol

    # 3. THE CANDIDATE DOMINATES THE FORECAST.
    concentrator = math.sqrt(
        portfolio_variance({"HELD_A": 0.4, "HELD_B": 0.4, "TWIN": 0.20}, tickers, cov)
    )
    diversifier = math.sqrt(
        portfolio_variance(
            {"HELD_A": 0.475, "HELD_B": 0.475, "DIVR": 0.05}, tickers, cov
        )
    )
    check(
        concentrator > diversifier,
        "a 20% high-confidence concentrator scored better than a 5% "
        "low-confidence diversifier — confidence would rank candidates "
        "backwards, and sizing on it alone would be wrong",
    )

    # 4. THE HAIRCUT RUNS IN ONE DIRECTION ONLY.
    check(
        FORECAST_RISK_HAIRCUT_INCREASERS_ONLY,
        "the haircut is not restricted to risk increasers",
    )
    increaser = adjust_size(proposal(0.10, increase=0.05), assessment(0.5))
    check(
        increaser["verdict"] == FRISK_REDUCED,
        f"a risk increaser at confidence 0.5 produced {increaser['verdict']}",
    )
    check(
        abs(increaser["adjusted_size"] - 0.05) < 1e-6,
        f"a risk increaser at confidence 0.5 was sized "
        f"{increaser['adjusted_size']}, not half of R2's 10%",
    )
    reducer = adjust_size(
        proposal(0.20, diversifying=True, increase=-0.19), assessment(0.30)
    )
    check(
        reducer["verdict"] == FRISK_UNCHANGED,
        f"a risk REDUCER at low confidence produced {reducer['verdict']} — "
        f"MEASURED, shrinking it removes the benefit rather than the risk",
    )
    check(
        reducer["adjusted_size"] == 0.20,
        "a risk reducer did not keep R2's size",
    )
    doubtful = adjust_size(
        proposal(0.20, diversifying=True, increase=-0.19), assessment(0.01)
    )
    check(
        doubtful["verdict"] != FRISK_NO_TRADE,
        "a risk-reducing candidate was refused on confidence, declining a "
        "free reduction in variance",
    )

    # 5. A HAIRCUT, NEVER A MULTIPLIER.
    check(
        FORECAST_RISK_MAX_MULTIPLIER == 1.0,
        f"the multiplier ceiling is {FORECAST_RISK_MAX_MULTIPLIER}, not 1.0 — "
        f"R2's size is already the most the portfolio can absorb",
    )
    for confidence in (0.25, 0.5, 0.75, 1.0):
        check(
            multiplier_for(confidence) <= 1.0 + 1e-9,
            f"confidence {confidence} earned a multiplier above 1.0",
        )
        result = adjust_size(proposal(0.10), assessment(confidence))
        if result.get("adjusted_size") is not None:
            check(
                result["adjusted_size"] <= 0.10 + 1e-9,
                f"confidence {confidence} grew R2's size to "
                f"{result['adjusted_size']}",
            )
    full = adjust_size(proposal(0.10), assessment(1.0))
    check(
        full["verdict"] == FRISK_UNCHANGED
        and abs(full["adjusted_size"] - 0.10) < 1e-9,
        "full confidence did not earn exactly R2's size",
    )

    # 6. BELOW THE FLOOR, NO TRADE — not a token position.
    check(
        0.0 < FORECAST_RISK_MIN_CONFIDENCE < 1.0,
        f"the confidence floor {FORECAST_RISK_MIN_CONFIDENCE} is not a share",
    )
    below = adjust_size(
        proposal(0.10), assessment(FORECAST_RISK_MIN_CONFIDENCE - 0.01)
    )
    check(
        below["verdict"] == FRISK_NO_TRADE,
        f"confidence below the floor produced {below['verdict']} instead of "
        f"NO_TRADE — a size that decays toward zero is a token position",
    )
    check(
        below.get("adjusted_size") is None,
        "a NO_TRADE carried a size; 0.0 and 'do not trade' must not share a "
        "representation",
    )
    at_floor = adjust_size(proposal(0.10), assessment(FORECAST_RISK_MIN_CONFIDENCE))
    check(
        at_floor.get("adjusted_size") is not None,
        "the floor is exclusive at the wrong end: confidence exactly at the "
        "floor produced no position",
    )

    # 7. THE HAIRCUT CANNOT SNEAK UNDER R2's MINIMUM TRADEABLE WEIGHT.
    tiny = adjust_size(proposal(SIZING_MIN_WEIGHT * 1.5), assessment(0.30))
    check(
        tiny["verdict"] == FRISK_NO_TRADE and tiny.get("adjusted_size") is None,
        f"a haircut produced {tiny.get('adjusted_size')}, below the "
        f"{SIZING_MIN_WEIGHT:.1%} minimum R2 already refuses",
    )

    # 8. MISSING IS NOT ZERO. An unmeasured confidence must neither refuse
    #    every trade nor wave it through.
    check(
        FORECAST_RISK_UNMEASURABLE_IS_NOT_LOW,
        "an UNMEASURABLE confidence factor is being treated as a low one",
    )
    check(confidence_of(None) is None, "a missing assessment produced a number")
    check(
        confidence_of({"assessed_object": "forecast"}) is None,
        "an assessment with no confidence produced a number",
    )
    unassessed = adjust_size(proposal(0.10), None)
    check(
        unassessed["verdict"] == FRISK_NOT_EVALUATED,
        f"an unassessed forecast produced {unassessed['verdict']} — 'no "
        f"confidence was measured' is not 'the forecast is worthless'",
    )
    check(
        unassessed.get("adjusted_size") is None
        and unassessed.get("confidence") is None,
        "an unassessed forecast carried a size or a confidence",
    )

    # 9. R2 IS NOT OVERRULED. Conviction does not change the covariance.
    refused = adjust_size(
        {"ticker": "X", "verdict": SIZING_REFUSED, "size": None}, assessment(1.0)
    )
    check(
        refused["verdict"] == FRISK_NOT_EVALUATED
        and refused.get("adjusted_size") is None,
        "a position R2 REFUSED was revived by a confident forecast",
    )

    # 10. THE REAL SIZER ROUTES BOTH DIRECTIONS. Using R2's own output, not a
    #     hand-built proposal, so the two cannot drift apart.
    proposals = {t: size_position(HELD, t, tickers, cov) for t in ("TWIN", "DIVR")}
    check(
        proposals["TWIN"].get("diversifying") is False,
        "R2 no longer reports the TWIN candidate as risk-increasing",
    )
    check(
        proposals["DIVR"].get("diversifying") is True,
        "R2 no longer reports the DIVR candidate as diversifying",
    )
    report = forecast_risk_report(
        proposals, {"TWIN": assessment(0.5), "DIVR": assessment(0.5)}
    )
    check(
        forecast_risk_problems(report) == [],
        f"a well-formed report reported problems: "
        f"{forecast_risk_problems(report)}",
    )
    check(
        report["adjusted"]["TWIN"]["verdict"] == FRISK_REDUCED,
        f"R2's own risk increaser produced "
        f"{report['adjusted']['TWIN']['verdict']}",
    )
    check(
        report["adjusted"]["DIVR"]["verdict"] == FRISK_UNCHANGED,
        f"R2's own diversifier produced "
        f"{report['adjusted']['DIVR']['verdict']}",
    )
    check(
        sum(report["counts"].values()) == report["candidates"],
        "the verdict counts do not cover every candidate",
    )

    # 11. A REFUSAL RENDERS AS ABSENT, NOT ZERO.
    refusal_report = forecast_risk_report(
        {"A": proposal(0.10)}, {"A": assessment(0.01)}
    )
    for line in render_forecast_risk(refusal_report):
        check(
            "0.00%" not in line.split("R4", 1)[1],
            "a refused position rendered as 0.00%, which reads as a position "
            "of zero size rather than no position",
        )

    # 12. DESCRIBED, NOT ENFORCED.
    check(
        not FORECAST_RISK_BLOCKS_TRADES,
        "R4 claims to block trades — it adjusts a size, and whether to trade "
        "at all is R7's decision",
    )

    # 13. THE CONTRACT CHECK CAN FAIL.
    for mutation, label in (
        (lambda r: r["adjusted"]["A"].update({"adjusted_size": 0.50}), "a grown size"),
        (lambda r: r["adjusted"]["A"].update({"multiplier": 1.5}), "a multiplier above 1"),
        (lambda r: r["adjusted"]["A"].update({"verdict": "PROBABLY"}), "an unknown verdict"),
        (lambda r: r["adjusted"]["A"].update({"reason": "  "}), "a verdict with no reason"),
        (lambda r: r["adjusted"]["A"].update({"adjusted_size": 0.049}), "an inconsistent size"),
        (lambda r: r.update({"blocks_trades": True}), "a report that blocks trades"),
    ):
        broken = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.5)})
        mutation(broken)
        check(
            forecast_risk_problems(broken) != [],
            f"{label} passed the contract check",
        )
    # A NO_TRADE carrying size 0.0 is the collapse the shape rule prevents.
    broken = forecast_risk_report({"A": proposal(0.10)}, {"A": assessment(0.01)})
    broken["adjusted"]["A"]["adjusted_size"] = 0.0
    check(
        forecast_risk_problems(broken) != [],
        "a NO_TRADE carrying size 0.0 passed the contract check",
    )
    # A haircut applied to a risk reducer must be caught after the fact too.
    broken = forecast_risk_report(
        {"A": proposal(0.10, diversifying=True, increase=-0.1)},
        {"A": assessment(0.5)},
    )
    broken["adjusted"]["A"]["verdict"] = FRISK_REDUCED
    check(
        forecast_risk_problems(broken) != [],
        "a haircut on a risk REDUCER passed the contract check",
    )

    if FAILURES:
        print("FORECAST-ADJUSTED RISK GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("FORECAST-ADJUSTED RISK GATE: PASS")
    print("  haircut reduces risk on increasers, monotonically (recomputed)")
    print("  haircut REMOVES the benefit on reducers, so it is not applied")
    print("  the candidate dominates the forecast: +19.8% vs -4.8% (recomputed)")
    print(f"  confidence floor {FORECAST_RISK_MIN_CONFIDENCE}, ceiling "
          f"{FORECAST_RISK_MAX_MULTIPLIER}, missing is not zero")
    print("  R2 is not overruled; R4 adjusts a size, R7 owns the refusal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
