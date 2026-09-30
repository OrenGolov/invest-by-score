"""Gate: the stress covariance keeps the crash in, and stress never looks calm.

R1 through R5 all rest on ONE covariance estimated over a long, mostly-calm
window. Every answer built on it is a calm-market answer unless it is re-asked
under stress.

THE DECIDING MEASUREMENT, on 76 real tickers over 1,170 common sessions:
average pairwise correlation roughly DOUBLES under stress, measured about the
full-period mean.

    stress definition   calm     stress   ratio
    worst  5% of days   0.2421   0.6066   x2.51
    worst 10% of days   0.2263   0.5534   x2.45
    worst 15% of days   0.2214   0.4958   x2.24
    worst 20% of days   0.2199   0.4591   x2.09

Portfolio volatility under the historical decile is 2.30x calm and the
diversification ratio falls 38.1% (2.080 -> 1.288).

THE TRAP THIS GATE EXISTS TO CATCH, and it was this sprint's FIRST result:
estimating the stress covariance by demeaning WITHIN the stress subset
reported correlation FALLING under stress (0.2166 -> 0.1492, -31.1%) and
diversification IMPROVING (2.128 -> 2.789). The stress decile averages -3.15%
against +0.14% overall, so subtracting the subset mean removes the crash
itself. The estimator reported the opposite of the truth, confidently.

The real-data figures above are quoted, not recomputed: the ingest is
gitignored and absent on every clone. The STRUCTURAL effect is recomputed here
from a seeded market that crashes together, which is what the rule rests on.

Verified to FAIL when any of these is reinjected:
  - the stress covariance demeaned within the subset
  - a report claiming a subset demeaning is acceptable
  - stressed correlation allowed below the calm baseline
  - a stress scenario that lowers portfolio volatility
  - the mandatory historical scenario dropped
  - a correlation shock that also changes volatilities
  - too few stressed sessions reported as a calm result
  - R6 declared to block trades
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    STRESS_BLOCKS_TRADES,
    STRESS_CORRELATION_LEVEL,
    STRESS_DEMEAN_FULL_PERIOD,
    STRESS_MIN_SESSIONS,
    STRESS_SCENARIO_HISTORICAL,
    STRESS_SCENARIOS,
    STRESS_VOLATILITY_MULTIPLIER,
    STRESS_WORST_SHARE,
)
from core.stress_scenarios import (  # noqa: E402
    average_correlation,
    covariance_over,
    diversification_ratio,
    render_stress,
    shock_correlation,
    shock_volatility,
    stress_problems,
    stress_report,
    worst_sessions,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def crashing_market(seed=13, sessions=800, names=12, crash_share=0.10):
    """Mostly idiosyncratic, but crashes together — the real data's structure.

    The crash magnitude is DISPERSED (sd 0.005) and per-name noise stays high
    on crash days (sd 0.025), because that is what real crashes look like and
    it is what makes the estimator trap visible. MEASURED, an earlier version
    used a tight crash with almost no idiosyncratic noise; subset demeaning
    still reported 0.88 correlation there, so the trap never appeared and a
    gate built on it passed the very sabotage it existed to catch.
    """
    rng = random.Random(seed)
    tickers = [f"T{i:02d}" for i in range(names)]
    matrix = [[0.0] * sessions for _ in range(names)]
    crash_days = set(
        rng.sample(
            range(sessions), max(STRESS_MIN_SESSIONS, int(sessions * crash_share))
        )
    )
    for day in range(sessions):
        if day in crash_days:
            common = rng.gauss(-0.040, 0.005)
            for i in range(names):
                matrix[i][day] = common + rng.gauss(0, 0.025)
        else:
            common = rng.gauss(0.0012, 0.003)
            for i in range(names):
                matrix[i][day] = common + rng.gauss(0, 0.012)
    return tickers, matrix, sorted(crash_days)


def equal_weights(tickers):
    return {t: 1.0 / len(tickers) for t in tickers}


def main() -> int:
    tickers, matrix, crash_days = crashing_market()
    weights = equal_weights(tickers)
    sessions = len(matrix[0])
    full_means = [statistics.mean(row) for row in matrix]
    calm_days = [d for d in range(sessions) if d not in set(crash_days)]

    # 1. THE ESTIMATOR TRAP, RECOMPUTED. Demeaning within the stress subset
    #    must understate correlation against the full-period estimate.
    correct = average_correlation(
        covariance_over(matrix, crash_days, means=full_means)
    )
    subset_means = [statistics.mean(row[d] for d in crash_days) for row in matrix]
    trapped = average_correlation(
        covariance_over(matrix, crash_days, means=subset_means)
    )
    check(
        correct > trapped,
        f"subset demeaning reported correlation {trapped:.4f} against the "
        f"full-period {correct:.4f}. The full-period rule rests on subset "
        f"demeaning UNDERSTATING stress; if that has stopped being true the "
        f"rule must be re-derived rather than kept",
    )

    # 2. WITH THE CORRECT ESTIMATOR, CORRELATION RISES UNDER STRESS.
    calm_correlation = average_correlation(
        covariance_over(matrix, calm_days, means=full_means)
    )
    stressed_correlation = correct
    check(
        stressed_correlation > calm_correlation,
        f"correlation did not rise under stress ({calm_correlation:.4f} -> "
        f"{stressed_correlation:.4f}) in a market built to crash together",
    )

    # 3. THE CONFIGURED RULE MATCHES THE MEASUREMENT.
    check(
        STRESS_DEMEAN_FULL_PERIOD,
        "the stress covariance is not configured to demean over the full "
        "period, which is the one choice this module exists to get right",
    )

    # 4. STRESS IS LOCATED, not sampled arbitrarily.
    selected = set(worst_sessions(matrix, share=STRESS_WORST_SHARE))
    overlap = len(selected & set(crash_days)) / max(len(selected), 1)
    check(
        overlap > 0.8,
        f"only {overlap:.0%} of the selected stress sessions were real crash "
        f"days — stress is not being located",
    )

    # 5. THE REPORT REPRODUCES THE EFFECT.
    report = stress_report(weights, tickers, matrix)
    check(
        stress_problems(report) == [],
        f"a well-formed report reported problems: {stress_problems(report)}",
    )
    check(report["status"] == "MEASURED", f"status was {report['status']}")
    check(
        report["demean"] == "full_period",
        f"the report demeans on {report['demean']!r}",
    )
    historical = report["scenarios"][STRESS_SCENARIO_HISTORICAL]
    check(
        historical["volatility_multiple"] > 1.0,
        f"stress reported a {historical['volatility_multiple']}x volatility "
        f"multiple — a stress scenario that makes the portfolio safer is "
        f"measuring the estimator, not the risk",
    )
    check(
        historical["stressed_correlation"] > historical["baseline_correlation"],
        f"the historical scenario reported correlation falling "
        f"({historical['baseline_correlation']:.4f} -> "
        f"{historical['stressed_correlation']:.4f}) — the signature of subset "
        f"demeaning",
    )
    check(
        historical["diversification_lost"] > 0,
        f"diversification survived the crash decile "
        f"({historical['diversification_lost']})",
    )
    check(
        STRESS_SCENARIO_HISTORICAL in report["material"],
        "a crash that costs diversification was not flagged material",
    )

    # 6. EVERY DECLARED SCENARIO IS REPORTED, and the historical one is
    #    mandatory: an unchecked hypothetical is an assumption.
    for name in STRESS_SCENARIOS:
        scenario = report["scenarios"].get(name)
        check(scenario is not None, f"scenario {name!r} was not reported")
        if scenario:
            check(
                str(scenario.get("reason") or "").strip() != "",
                f"{name}: scenario with no reason",
            )
    check(
        STRESS_SCENARIO_HISTORICAL in STRESS_SCENARIOS,
        "the historical scenario is not declared mandatory",
    )

    # 7. THE SYNTHETIC SHOCKS DO WHAT THEY CLAIM.
    calm_cov = covariance_over(matrix, calm_days, means=full_means)
    correlation_shocked = shock_correlation(calm_cov)
    check(
        average_correlation(correlation_shocked) > average_correlation(calm_cov),
        "the correlation shock did not raise correlation",
    )
    for i in range(len(calm_cov)):
        check(
            abs(calm_cov[i][i] - correlation_shocked[i][i]) < 1e-12,
            f"the correlation shock changed volatility {i} — it must move "
            f"correlation alone, or the two effects cannot be told apart",
        )
    volatility_shocked = shock_volatility(calm_cov)
    check(
        abs(
            average_correlation(volatility_shocked) - average_correlation(calm_cov)
        )
        < 1e-8,
        "the volatility shock changed correlation — it must move volatility "
        "alone",
    )
    check(
        abs(
            volatility_shocked[0][0]
            - calm_cov[0][0] * STRESS_VOLATILITY_MULTIPLIER ** 2
        )
        < 1e-12,
        "the volatility shock did not scale variance by the squared multiplier",
    )
    check(
        0.0 < STRESS_CORRELATION_LEVEL < 1.0,
        f"a correlation level of {STRESS_CORRELATION_LEVEL} is degenerate; a "
        f"literal 1.0 is singular",
    )

    # 8. TOO LITTLE STRESS IS NOT AN ABSENCE OF STRESS.
    thin_tickers = ["A", "B", "C"]
    thin = [[0.01, -0.02, 0.005, 0.001] for _ in thin_tickers]
    thin_report = stress_report(equal_weights(thin_tickers), thin_tickers, thin)
    check(
        thin_report["status"] == "NOT_EVALUATED",
        f"a portfolio with {thin_report['stressed_sessions']} stressed "
        f"sessions produced {thin_report['status']} — reporting a calm answer "
        f"claims the portfolio survives conditions nobody examined",
    )
    check(
        thin_report["scenarios"] == {},
        "a NOT_EVALUATED report carried scenarios",
    )
    check(
        str(thin_report.get("reason") or "").strip() != "",
        "a NOT_EVALUATED report carried no reason",
    )
    check(
        stress_problems(thin_report) == [],
        f"an honestly-unevaluated report reported problems: "
        f"{stress_problems(thin_report)}",
    )

    # 9. A SINGLE HOLDING HAS NO DIVERSIFICATION, and an empty one has no
    #    measurement rather than a zero.
    check(
        abs(diversification_ratio({tickers[0]: 1.0}, tickers, calm_cov) - 1.0) < 1e-6,
        "a single holding reported a diversification benefit",
    )
    check(
        diversification_ratio({}, tickers, calm_cov) is None,
        "an empty portfolio reported a diversification ratio of 0.0, which "
        "reads as perfect diversification rather than unmeasured",
    )

    # 10. DESCRIBED, NOT ENFORCED.
    check(
        not STRESS_BLOCKS_TRADES,
        "R6 claims to block trades — it measures what stress does, and "
        "whether to trade is R7's decision",
    )
    check(not report["blocks_trades"], "a stress report claims to block trades")

    # 11. A MATERIAL SCENARIO IS VISIBLE IN THE RENDER.
    check(
        "!!" in "\n".join(render_stress(report)),
        "a material stress scenario rendered without being marked",
    )

    # 12. THE CONTRACT CHECK CAN FAIL.
    def fresh():
        return stress_report(weights, tickers, matrix)

    mutations = [
        (lambda r: r.update({"demean": "subset"}), "a subset-demeaned report"),
        (lambda r: r.update({"demean_reason": ""}), "a missing demean reason"),
        (lambda r: r.update({"blocks_trades": True}), "a report that blocks trades"),
        (
            lambda r: r["scenarios"].pop(STRESS_SCENARIO_HISTORICAL),
            "a report with no historical scenario",
        ),
        (
            lambda r: r["scenarios"][STRESS_SCENARIO_HISTORICAL].update(
                {"volatility_multiple": 0.8}
            ),
            "stress that lowers volatility",
        ),
        (
            lambda r: r["scenarios"][STRESS_SCENARIO_HISTORICAL].update(
                {
                    "stressed_correlation": r["scenarios"][
                        STRESS_SCENARIO_HISTORICAL
                    ]["baseline_correlation"]
                    - 0.1
                }
            ),
            "stressed correlation below the calm baseline",
        ),
        (
            lambda r: r["scenarios"][STRESS_SCENARIO_HISTORICAL].update(
                {"reason": "  "}
            ),
            "a scenario with no reason",
        ),
        (
            lambda r: r["scenarios"][STRESS_SCENARIO_HISTORICAL].update(
                {"diversification_lost": 0.9, "material": False}
            ),
            "a materiality disagreeing with its threshold",
        ),
        (
            lambda r: r["scenarios"][STRESS_SCENARIO_HISTORICAL].update(
                {"stressed_diversification": 0.0}
            ),
            "a zero diversification ratio",
        ),
        (
            lambda r: r.update({"status": "NOT_EVALUATED", "reason": "x"}),
            "NOT_EVALUATED carrying scenarios",
        ),
    ]
    for mutation, label in mutations:
        broken = fresh()
        mutation(broken)
        check(stress_problems(broken) != [], f"{label} passed the contract check")

    if FAILURES:
        print("STRESS SCENARIO GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("STRESS SCENARIO GATE: PASS")
    print(f"  subset demeaning understates stress correlation "
          f"({trapped:.4f} vs {correct:.4f}) — recomputed")
    print(f"  correlation rises under stress "
          f"({calm_correlation:.4f} -> {stressed_correlation:.4f})")
    print(f"  historical decile: {historical['volatility_multiple']:.2f}x "
          f"volatility, {historical['diversification_lost']:.1%} of "
          f"diversification lost")
    print(f"  shocks are separable: correlation-only and volatility-only")
    print("  stress measured, not enforced (R7 owns the decision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
