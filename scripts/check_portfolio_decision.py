"""Gate: R7 is the one module that refuses, and it refuses what R2 approves.

R1, R3, R4, R5 and R6 each carry *_BLOCKS_TRADES = False and defer the refusal
here. This gate checks that the deferral actually lands: that R7 blocks, that
nothing else does, and that R7 catches a portfolio-level failure no per-ticker
check can see.

THE DECIDING MEASUREMENT, recomputed here: a set of individually-correct
trades can be collectively impossible. On a diversified 4-name book with 6
candidates in one correlated cluster, R2 sizes each candidate against the
ORIGINAL portfolio and approves all six at 20% - every one diversifying, every
one REDUCING volatility in isolation. The approved set sums to 120% of the
book and raises portfolio volatility +25.52% executed together.

R7 DOES NOT DUPLICATE W2. Every RISK_POLICY_V2 rule reads ONE ticker's inputs;
none can see the portfolio. This gate constructs per-ticker evidence W2
accepts and confirms R7 still refuses, because if it did not, R7 would add
nothing.

Verified to FAIL when any of these is reinjected:
  - R7 declared unable to block trades
  - another Sprint R module declaring that it blocks
  - the total-size veto removed, so an infeasible set passes
  - the set volatility measured on one member instead of the whole set
  - a missing covariance, concentration or stress report passing silently
  - warnings summed into a refusal with no veto
  - a veto that does not produce NO_TRADE
  - a refused candidate counted at size 0.0
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    EXPOSURE_BLOCKS_TRADES,
    FORECAST_RISK_BLOCKS_TRADES,
    IMPACT_BLOCKS_TRADES,
    PORTFOLIO_BLOCKS_TRADES,
    PORTFOLIO_FAIL_CLOSED,
    PORTFOLIO_MAX_TOTAL_SIZE,
    PORTFOLIO_MAX_VOLATILITY_INCREASE,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    PORTFOLIO_REDUCED,
    PORTFOLIO_RULES,
    PORTFOLIO_SEVERITY_VETO,
    SECTOR_BLOCKS_TRADES,
    SIZING_RISK_BUDGET,
    STRESS_BLOCKS_TRADES,
)
from core.correlation_sizing import size_position  # noqa: E402
from core.portfolio_decision import (  # noqa: E402
    accepted_sizes,
    decision_problems,
    evaluate_portfolio,
    render_decision,
    set_volatility_increase,
)
from core.position_exposure import covariance  # noqa: E402
from core.risk_policy import evaluate_risk_policy  # noqa: E402

FAILURES: list[str] = []
BASE = ["D1", "D2", "D3", "D4"]
HELD = {t: 0.25 for t in BASE}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def clustered_market(seed=41, sessions=800, candidates=6):
    """A diversified book plus candidates that all share one factor."""
    rng = random.Random(seed)
    cands = [f"C{i}" for i in range(candidates)]
    names = BASE + cands
    matrix = [[0.0] * sessions for _ in names]
    crash = set(rng.sample(range(sessions), 80))
    for day in range(sessions):
        if day in crash:
            common = rng.gauss(-0.040, 0.005)
            for i in range(len(names)):
                matrix[i][day] = common + rng.gauss(0, 0.025)
        else:
            market = rng.gauss(0.0012, 0.004)
            cluster = rng.gauss(0, 0.020)
            for i in range(len(BASE)):
                matrix[i][day] = market + rng.gauss(0, 0.014)
            for j in range(candidates):
                matrix[len(BASE) + j][day] = market + cluster + rng.gauss(0, 0.005)
    return names, cands, covariance(matrix)


def proposal(ticker, size, **extra):
    payload = {"ticker": ticker, "verdict": "SIZED", "size": size}
    payload.update(extra)
    return payload


def main() -> int:
    names, cands, cov = clustered_market()
    proposals = {t: size_position(HELD, t, names, cov) for t in cands}

    # 1. R2 APPROVES EVERY CANDIDATE IN ISOLATION. Recomputed, because the
    #    whole point of R7 rests on it.
    for ticker, sized in proposals.items():
        check(
            sized.get("size") is not None,
            f"{ticker}: R2 did not size it; this gate needs candidates R2 "
            f"approves individually",
        )
        check(
            bool(sized.get("diversifying")),
            f"{ticker}: R2 no longer reports it as diversifying — the gate's "
            f"scenario no longer demonstrates the finding",
        )
        check(
            sized.get("volatility_increase", 1) < 0,
            f"{ticker}: R2 reports it raising volatility; the finding is "
            f"about candidates that each LOWER it",
        )

    # 2. THE APPROVED SET IS IMPOSSIBLE.
    sizes = accepted_sizes(proposals)
    total = sum(sizes.values())
    check(
        total > 1.0,
        f"the individually-approved set totals {total:.1%}; the gate needs a "
        f"set that cannot be executed",
    )
    joint = set_volatility_increase(HELD, sizes, names, cov)
    check(
        joint is not None and joint > 0,
        f"the set of individually volatility-REDUCING trades did not raise "
        f"volatility together ({joint}). R7's central measurement rests on "
        f"that; if it has gone, the rule must be re-derived rather than kept",
    )

    # 3. R7 REFUSES WHAT R2 APPROVED.
    decision = evaluate_portfolio(HELD, proposals, names, cov)
    check(
        decision["verdict"] == PORTFOLIO_NO_TRADE,
        f"R7 returned {decision['verdict']} on a set totalling {total:.1%} of "
        f"the book",
    )
    check(
        "total_size_exceeds_budget" in decision["veto_rule_ids"],
        "the infeasible total did not trigger its veto",
    )
    check(
        decision_problems(decision) == [],
        f"a well-formed refusal reported problems: {decision_problems(decision)}",
    )

    # 4. R7 DOES NOT DUPLICATE W2. Per-ticker evidence W2 accepts must still
    #    be refused at the portfolio level, or R7 adds nothing.
    context = {
        "market_data_quality": 95.0,
        "market_source_confidence": 0.95,
        "market_timestamp_valid": True,
        "fundamental_point_in_time_valid": True,
        "fundamental_source_confidence": 0.9,
        "fundamental_source_status": "ok",
        "score": 80.0,
        "action": "BUY",
        "confidence": 0.85,
        "confidence_breakdown": {"factors": []},
        "market_regime": "bullish",
    }
    w2 = evaluate_risk_policy(context)
    check(
        not w2["veto"],
        f"the gate's per-ticker context was vetoed by W2 ({w2['veto_rule_ids']}); "
        f"it must be evidence W2 ACCEPTS for the comparison to mean anything",
    )
    check(
        decision["verdict"] == PORTFOLIO_NO_TRADE,
        "W2 accepted the evidence and R7 did not refuse the portfolio-level "
        "problem — R7 would add nothing over W2",
    )

    # 5. R7 IS THE ONLY MODULE THAT BLOCKS. Every other Sprint R module
    #    deferred here precisely so one place owns the refusal.
    check(PORTFOLIO_BLOCKS_TRADES, "R7 does not block trades")
    for name, flag in (
        ("R1 exposure", EXPOSURE_BLOCKS_TRADES),
        ("R3 sector concentration", SECTOR_BLOCKS_TRADES),
        ("R4 forecast risk", FORECAST_RISK_BLOCKS_TRADES),
        ("R5 expected impact", IMPACT_BLOCKS_TRADES),
        ("R6 stress", STRESS_BLOCKS_TRADES),
    ):
        check(
            not flag,
            f"{name} claims to block trades — the refusal is meant to live in "
            f"exactly one place, and two modules refusing means neither owns it",
        )

    # 6. FAIL-CLOSED. A missing input triggers the rule that reads it.
    check(PORTFOLIO_FAIL_CLOSED, "R7 is not configured fail-closed")
    small_names, small_cands, small_cov = clustered_market(candidates=1)
    no_cov = evaluate_portfolio(HELD, {"C0": proposal("C0", 0.05)})
    check(
        "set_volatility_exceeds_budget" in no_cov["veto_rule_ids"],
        "a missing covariance did not trigger the joint-volatility veto — an "
        "unmeasurable effect was treated as a safe one",
    )
    no_evidence = evaluate_portfolio(
        HELD, {"C0": proposal("C0", 0.05)}, small_names, small_cov
    )
    for rule_id in ("sector_concentration_breach", "stress_diversification_collapse"):
        check(
            rule_id in no_evidence["warning_rule_ids"],
            f"{rule_id} did not trigger when its report was absent — a "
            f"portfolio approved without the check is not one found clean",
        )
    unevaluated_stress = evaluate_portfolio(
        HELD,
        {"C0": proposal("C0", 0.05)},
        small_names,
        small_cov,
        concentration={"flags": []},
        stress={"status": "NOT_EVALUATED", "reason": "too few sessions"},
    )
    check(
        "stress_diversification_collapse" in unevaluated_stress["warning_rule_ids"],
        "an unevaluated stress report passed as though stress were absent",
    )

    # 7. THE GATE CAN STAY QUIET. A module that refuses everything decides
    #    nothing.
    clean = evaluate_portfolio(
        HELD,
        {"C0": proposal("C0", 0.05), "C1": proposal("C1", 0.05)},
        names,
        cov,
        concentration={"flags": []},
        stress={"status": "MEASURED", "material": [], "scenarios": {}},
    )
    check(
        clean["verdict"] == PORTFOLIO_PROCEED,
        f"a clean 10% set produced {clean['verdict']} — a decision module "
        f"that cannot approve is not deciding, it is refusing",
    )
    check(
        decision_problems(clean) == [],
        f"a clean approval reported problems: {decision_problems(clean)}",
    )

    # 8. WARNINGS NEVER SUM INTO A REFUSAL.
    warned = evaluate_portfolio(
        HELD, {"C0": proposal("C0", 0.05), "C1": proposal("C1", 0.05)}, names, cov
    )
    check(
        warned["warning_rule_ids"] and not warned["veto_rule_ids"],
        "the gate's warning scenario no longer produces warnings without a veto",
    )
    check(
        warned["verdict"] != PORTFOLIO_NO_TRADE,
        f"{len(warned['warning_rule_ids'])} warning(s) produced a refusal with "
        f"no veto — counting warnings until they become a veto invents a "
        f"threshold nobody measured",
    )

    # 9. AN EMPTY SET IS NOT AN APPROVAL.
    none_survived = evaluate_portfolio(
        HELD,
        {"C0": {"ticker": "C0", "verdict": "NO_TRADE", "adjusted_size": None}},
        names,
        cov,
        concentration={"flags": []},
        stress={"status": "MEASURED", "material": [], "scenarios": {}},
    )
    check(
        none_survived["verdict"] == PORTFOLIO_NO_TRADE,
        f"a set where every candidate was refused upstream produced "
        f"{none_survived['verdict']} — proceeding would execute nothing while "
        f"reporting approval",
    )
    check(
        accepted_sizes({"C0": {"verdict": "NO_TRADE", "adjusted_size": None}}) == {},
        "a refused candidate was counted at size 0.0, which is a position "
        "rather than an absence",
    )

    # 10. THE BUDGETS AGREE WITH R2.
    check(
        PORTFOLIO_MAX_VOLATILITY_INCREASE == SIZING_RISK_BUDGET,
        f"the set-level volatility budget {PORTFOLIO_MAX_VOLATILITY_INCREASE} "
        f"disagrees with R2's per-trade budget {SIZING_RISK_BUDGET}: a set "
        f"must not do collectively what no single trade was allowed to do",
    )
    check(
        0.0 < PORTFOLIO_MAX_TOTAL_SIZE < 1.0,
        f"a total-size budget of {PORTFOLIO_MAX_TOTAL_SIZE} does not leave "
        f"the book intact",
    )

    # 11. EVERY RULE IS EVALUATED AND CARRIES ITS MEASUREMENT.
    evaluated = {r["rule_id"] for r in decision["rules"]}
    check(
        evaluated == set(PORTFOLIO_RULES),
        f"rules evaluated {sorted(evaluated)} do not match the declared "
        f"policy {sorted(PORTFOLIO_RULES)}",
    )
    for rule in decision["rules"]:
        check(
            str(rule.get("measurement") or "").strip() != "",
            f"{rule['rule_id']}: carries no measurement — a rule that cannot "
            f"say what it enforces is an assertion",
        )
        check(
            str(rule.get("source") or "").strip() != "",
            f"{rule['rule_id']}: names no sprint as its source",
        )
    check(
        any(
            spec["severity"] == PORTFOLIO_SEVERITY_VETO
            for spec in PORTFOLIO_RULES.values()
        ),
        "no rule can veto: a decision module that cannot refuse is not one",
    )

    # 12. THE REFUSAL IS VISIBLE IN THE RENDER.
    text = "\n".join(render_decision(decision))
    check(PORTFOLIO_NO_TRADE in text, "the refusal is not rendered")
    check("TRIGGERED" in text, "no triggered rule is marked in the render")

    # 13. THE CONTRACT CHECK CAN FAIL.
    def fresh():
        return evaluate_portfolio(
            HELD,
            {"C0": proposal("C0", 0.05)},
            names,
            cov,
            concentration={"flags": []},
            stress={"status": "MEASURED", "material": [], "scenarios": {}},
        )

    mutations = [
        (
            lambda d: d.update(
                {"veto": True, "veto_rule_ids": ["total_size_exceeds_budget"]}
            ),
            "a fired veto alongside a PROCEED verdict",
        ),
        (lambda d: d.update({"verdict": PORTFOLIO_NO_TRADE}), "a refusal naming no rule"),
        (lambda d: d.update({"blocks_trades": False}), "R7 unable to block"),
        (lambda d: d.update({"fail_closed": False}), "R7 not fail-closed"),
        (lambda d: d.update({"rules": d["rules"][:-1]}), "a missing rule"),
        (lambda d: d.update({"verdict": "PROBABLY"}), "an unknown verdict"),
        (lambda d: d.update({"total_size": 0.99}), "a mismatched total"),
        (lambda d: d.update({"accepted": {}}), "a PROCEED with nothing accepted"),
        (
            lambda d: d["rules"][0].update({"measurement": "  "}),
            "a rule with no measurement",
        ),
    ]
    for mutation, label in mutations:
        broken = fresh()
        mutation(broken)
        check(decision_problems(broken) != [], f"{label} passed the contract check")

    if FAILURES:
        print("PORTFOLIO DECISION GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("PORTFOLIO DECISION GATE: PASS")
    print(f"  R2 approved all {len(cands)} candidates individually, totalling "
          f"{total:.0%} of the book (recomputed)")
    print(f"  executed together they move volatility {joint:+.2%}; R7 refuses")
    print("  W2 accepts the per-ticker evidence; R7 still refuses the book")
    print("  R7 is the only module that blocks; R1/R3/R4/R5/R6 all defer")
    print("  fail-closed: missing covariance, concentration or stress all trigger")
    print("  and it can still approve a clean set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
