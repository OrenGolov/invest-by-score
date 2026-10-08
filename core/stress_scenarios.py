"""R6 stress scenarios — the calm-window covariance is the thing that breaks.

R1 through R5 all rest on ONE covariance estimated over a long, mostly-calm
window. Every answer built on it is a calm-market answer unless it is re-asked
under stress, and stress is precisely when the estimate stops holding.

**THE DECIDING MEASUREMENT, on 76 real tickers over 1,170 common sessions:
correlation roughly DOUBLES under stress.** Average pairwise correlation,
measured about the full-period mean:

    stress definition   calm     stress   ratio
    worst  5% of days   0.2421   0.6066   x2.51
    worst 10% of days   0.2263   0.5534   x2.45
    worst 15% of days   0.2214   0.4958   x2.24
    worst 20% of days   0.2199   0.4591   x2.09

On an equal-weight portfolio of those names, portfolio volatility under stress
is **2.30x** calm and the diversification ratio falls **38.1%** (2.080 ->
1.288). Diversification weakens most in exactly the conditions it is held for,
so a calm-window covariance overstates every hedge in the book.

**THE TRAP, and it is not hypothetical — it was this sprint's first result.**
Estimating the stress covariance by demeaning WITHIN the stress subset
reported average correlation *falling* under stress (0.2166 -> 0.1492, -31.1%)
and the diversification ratio *improving* (2.128 -> 2.789). MEASURED, the
stress decile averages -3.15% against +0.14% over all sessions; subtracting
the subset mean removes the crash itself, so every name looks as though it
barely moved and co-movement collapses. The estimator reported the opposite of
the truth, confidently.

So the stress covariance is measured about the **FULL-PERIOD mean**, which
keeps the crash in the deviations. A module that got this wrong would tell a
reader their portfolio diversifies better in a crash.

**Three scenarios, one of them historical.** A hypothetical shock that is
never checked against the observed record is an assumption wearing a
measurement's clothes, so the worst-decile scenario is mandatory and the two
synthetic shocks are calibrated to it: 0.95 correlation sits beyond the
observed 0.607, and the 2x volatility shock matches the observed 2.30x.

**R6 measures what stress does. It does not refuse trades** — that is R7.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from core.config import (
    STRESS_BLOCKS_TRADES,
    STRESS_CORRELATION_LEVEL,
    STRESS_DEMEAN_FULL_PERIOD,
    STRESS_DEMEAN_REASON,
    STRESS_MATERIAL_DEGRADATION,
    STRESS_MIN_SESSIONS,
    STRESS_SCENARIO_CORRELATION_SHOCK,
    STRESS_SCENARIO_HISTORICAL,
    STRESS_SCENARIO_VERSION,
    STRESS_SCENARIO_VOLATILITY_SHOCK,
    STRESS_SCENARIOS,
    STRESS_VOLATILITY_MULTIPLIER,
    STRESS_WORST_SHARE,
)
from core.position_exposure import portfolio_variance


class StressScenarioError(ValueError):
    """Raised when a stress request is structurally invalid."""


def worst_sessions(
    matrix: Sequence[Sequence[float]],
    *,
    share: float = STRESS_WORST_SHARE,
) -> list[int]:
    """The indices of the worst market-wide sessions.

    Stress is defined by returns falling together, not by quotes being noisy:
    a decision needs to know what happens when everything drops at once.
    """
    if not matrix or not matrix[0]:
        raise StressScenarioError("no return matrix to rank")
    if not 0.0 < share < 1.0:
        raise StressScenarioError(f"share {share!r} must lie in (0, 1)")
    sessions = len(matrix[0])
    market = [
        statistics.mean(row[index] for row in matrix) for index in range(sessions)
    ]
    ranked = sorted(range(sessions), key=lambda index: market[index])
    count = max(1, int(sessions * share))
    return sorted(ranked[:count])


def covariance_over(
    matrix: Sequence[Sequence[float]],
    columns: Sequence[int],
    *,
    means: Sequence[float] | None = None,
) -> list[list[float]]:
    """Covariance over selected sessions, demeaned by `means`.

    `means` defaults to the FULL-PERIOD mean of each row, which is the whole
    point. Demeaning within the selected subset subtracts the crash being
    measured: MEASURED, that reported correlation falling 31.1% under stress
    and diversification improving, both artifacts of the estimator.
    """
    if not matrix:
        raise StressScenarioError("no return matrix")
    columns = list(columns)
    if len(columns) < 2:
        raise StressScenarioError(
            f"a covariance needs at least two observations, got {len(columns)}"
        )
    if means is None:
        means = [statistics.mean(row) for row in matrix]
    if len(means) != len(matrix):
        raise StressScenarioError("one mean is required per series")

    deviations = [
        [float(row[index]) - float(mean) for index in columns]
        for row, mean in zip(matrix, means)
    ]
    size = len(matrix)
    count = len(columns) - 1
    return [
        [
            sum(deviations[i][x] * deviations[j][x] for x in range(len(columns)))
            / count
            for j in range(size)
        ]
        for i in range(size)
    ]


def average_correlation(cov: Sequence[Sequence[float]]) -> float | None:
    """Average pairwise correlation implied by a covariance matrix."""
    size = len(cov)
    if size < 2:
        return None
    values: list[float] = []
    for i in range(size):
        for j in range(i + 1, size):
            spread = cov[i][i] * cov[j][j]
            if spread > 0:
                values.append(cov[i][j] / math.sqrt(spread))
    if not values:
        return None
    return round(statistics.mean(values), 8)


def diversification_ratio(
    weights: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> float | None:
    """Weighted average single-name volatility over portfolio volatility.

    1.0 means no diversification at all. None when it cannot be measured —
    never 0.0, which would read as perfect diversification.
    """
    variance = portfolio_variance(weights, tickers, cov)
    if variance <= 0:
        return None
    index = {t: i for i, t in enumerate(tickers)}
    weighted = 0.0
    for ticker, weight in weights.items():
        position = index.get(str(ticker).upper())
        if position is None:
            continue
        weighted += abs(float(weight)) * math.sqrt(max(cov[position][position], 0.0))
    if weighted <= 0:
        return None
    return round(weighted / math.sqrt(variance), 8)


def shock_correlation(
    cov: Sequence[Sequence[float]], level: float = STRESS_CORRELATION_LEVEL
) -> list[list[float]]:
    """Rebuild a covariance at a uniform correlation, keeping each volatility.

    Not literally 1.0: a perfectly correlated matrix is singular and portfolio
    variance stops being well conditioned. MEASURED, the worst 5% of sessions
    already reach 0.607 average pairwise correlation.
    """
    if not 0.0 < level < 1.0:
        raise StressScenarioError(f"correlation level {level!r} must lie in (0, 1)")
    size = len(cov)
    deviations = [math.sqrt(max(cov[i][i], 0.0)) for i in range(size)]
    return [
        [
            cov[i][i] if i == j else level * deviations[i] * deviations[j]
            for j in range(size)
        ]
        for i in range(size)
    ]


def shock_volatility(
    cov: Sequence[Sequence[float]], multiplier: float = STRESS_VOLATILITY_MULTIPLIER
) -> list[list[float]]:
    """Scale every volatility by `multiplier`, leaving correlations intact."""
    if multiplier <= 0:
        raise StressScenarioError(f"multiplier {multiplier!r} must be positive")
    factor = multiplier * multiplier
    return [[value * factor for value in row] for row in cov]


def _scenario(
    name: str,
    weights: Mapping[str, float],
    tickers: Sequence[str],
    baseline_cov: Sequence[Sequence[float]],
    stressed_cov: Sequence[Sequence[float]],
    reason: str,
    **detail,
) -> dict:
    """One scenario's before/after, on the same weights."""
    base_variance = portfolio_variance(weights, tickers, baseline_cov)
    stressed_variance = portfolio_variance(weights, tickers, stressed_cov)
    base_vol = math.sqrt(max(base_variance, 0.0))
    stressed_vol = math.sqrt(max(stressed_variance, 0.0))

    base_ratio = diversification_ratio(weights, tickers, baseline_cov)
    stressed_ratio = diversification_ratio(weights, tickers, stressed_cov)
    degradation = None
    if base_ratio and stressed_ratio is not None and base_ratio > 0:
        degradation = round(1.0 - (stressed_ratio / base_ratio), 6)

    payload = {
        "scenario": name,
        "baseline_volatility": round(base_vol, 8),
        "stressed_volatility": round(stressed_vol, 8),
        "volatility_multiple": (
            round(stressed_vol / base_vol, 6) if base_vol > 0 else None
        ),
        "baseline_correlation": average_correlation(baseline_cov),
        "stressed_correlation": average_correlation(stressed_cov),
        "baseline_diversification": base_ratio,
        "stressed_diversification": stressed_ratio,
        "diversification_lost": degradation,
        "material": (
            degradation is not None and degradation >= STRESS_MATERIAL_DEGRADATION
        ),
        "threshold": STRESS_MATERIAL_DEGRADATION,
        "reason": reason,
    }
    payload.update(detail)
    return payload


def stress_report(
    weights: Mapping[str, float],
    tickers: Sequence[str],
    matrix: Sequence[Sequence[float]],
    *,
    share: float = STRESS_WORST_SHARE,
    min_sessions: int = STRESS_MIN_SESSIONS,
) -> dict:
    """What every R1-R5 answer looks like when the calm window stops holding.

    `matrix` is the aligned return matrix R1 produces, one row per ticker in
    `tickers` order. The baseline and every scenario are measured on the same
    weights, so a difference is attributable to the stress, not the portfolio.
    """
    if not isinstance(weights, Mapping):
        raise StressScenarioError("a mapping of weights is required")
    if not matrix or not matrix[0]:
        raise StressScenarioError("no return matrix to stress")
    if len(matrix) != len(tickers):
        raise StressScenarioError(
            f"{len(matrix)} return series for {len(tickers)} tickers"
        )

    sessions = len(matrix[0])
    means = [statistics.mean(row) for row in matrix]
    stressed_columns = worst_sessions(matrix, share=share)

    # NOT ENOUGH STRESS TO MEASURE IS NOT AN ABSENCE OF STRESS. Reporting a
    # calm answer here would claim the portfolio survives conditions nobody
    # examined.
    if len(stressed_columns) < min_sessions:
        return {
            "version": STRESS_SCENARIO_VERSION,
            "sessions": sessions,
            "stressed_sessions": len(stressed_columns),
            "status": "NOT_EVALUATED",
            "reason": (
                f"only {len(stressed_columns)} stressed sessions are "
                f"available and {min_sessions} are required; a covariance "
                f"from fewer observations is not an estimate, and reporting "
                f"the calm answer would claim the portfolio survives "
                f"conditions nobody examined"
            ),
            "scenarios": {},
            "demean": "full_period" if STRESS_DEMEAN_FULL_PERIOD else "subset",
            "demean_reason": STRESS_DEMEAN_REASON,
            "blocks_trades": STRESS_BLOCKS_TRADES,
            "note": _NOTE,
        }

    calm_columns = [i for i in range(sessions) if i not in set(stressed_columns)]
    baseline = covariance_over(matrix, calm_columns, means=means)
    # THE WHOLE POINT: demeaned by the FULL-PERIOD mean, so the crash stays in.
    historical = covariance_over(matrix, stressed_columns, means=means)

    scenarios = {
        STRESS_SCENARIO_HISTORICAL: _scenario(
            STRESS_SCENARIO_HISTORICAL,
            weights,
            tickers,
            baseline,
            historical,
            (
                f"the worst {share:.0%} of sessions, measured about the "
                f"full-period mean so the crash stays in the deviations. "
                f"MEASURED on 76 real tickers, correlation roughly doubles "
                f"here (0.2263 -> 0.5534) and diversification falls 38.1%"
            ),
            stressed_sessions=len(stressed_columns),
            share=share,
        ),
        STRESS_SCENARIO_CORRELATION_SHOCK: _scenario(
            STRESS_SCENARIO_CORRELATION_SHOCK,
            weights,
            tickers,
            baseline,
            shock_correlation(baseline),
            (
                f"every pair forced to {STRESS_CORRELATION_LEVEL} correlation "
                f"at unchanged volatilities — beyond the 0.607 observed in "
                f"the worst 5% of sessions, and short of a singular matrix"
            ),
            level=STRESS_CORRELATION_LEVEL,
        ),
        STRESS_SCENARIO_VOLATILITY_SHOCK: _scenario(
            STRESS_SCENARIO_VOLATILITY_SHOCK,
            weights,
            tickers,
            baseline,
            shock_volatility(baseline),
            (
                f"every volatility scaled {STRESS_VOLATILITY_MULTIPLIER}x at "
                f"unchanged correlations, calibrated to the 2.30x portfolio "
                f"volatility observed in the historical decile"
            ),
            multiplier=STRESS_VOLATILITY_MULTIPLIER,
        ),
    }

    material = sorted(n for n, s in scenarios.items() if s.get("material"))
    return {
        "version": STRESS_SCENARIO_VERSION,
        "sessions": sessions,
        "stressed_sessions": len(stressed_columns),
        "status": "MEASURED",
        "reason": "",
        "scenarios": scenarios,
        "scenario_order": list(STRESS_SCENARIOS),
        "material": material,
        "worst_volatility_multiple": max(
            (
                s["volatility_multiple"]
                for s in scenarios.values()
                if s.get("volatility_multiple") is not None
            ),
            default=None,
        ),
        "demean": "full_period" if STRESS_DEMEAN_FULL_PERIOD else "subset",
        "demean_reason": STRESS_DEMEAN_REASON,
        "blocks_trades": STRESS_BLOCKS_TRADES,
        "note": _NOTE,
    }


_NOTE = (
    "R6 measures what stress does to the covariance every other answer rests "
    "on. MEASURED on 76 real tickers, correlation roughly doubles under the "
    "worst decile and diversification falls 38.1% — weakest exactly when it "
    "is most needed. Whether to trade is R7's decision."
)


def stress_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a stress report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("blocks_trades"):
        problems.append(
            "R6 claims to block trades — it measures what stress does, and "
            "whether to trade is R7's decision"
        )
    if report.get("demean") != "full_period":
        problems.append(
            "the stress covariance is not demeaned over the full period. "
            "MEASURED, subset demeaning reported correlation FALLING 31.1% "
            "under stress and diversification IMPROVING, because it subtracts "
            "away the -3.15% crash it is measuring"
        )
    if not str(report.get("demean_reason") or "").strip():
        problems.append("the demeaning choice carries no reason")

    status = report.get("status")
    scenarios = report.get("scenarios") or {}
    if status == "NOT_EVALUATED":
        if scenarios:
            problems.append("stress was NOT_EVALUATED but scenarios were reported")
        if not str(report.get("reason") or "").strip():
            problems.append("stress was NOT_EVALUATED with no reason given")
        return problems

    # THE HISTORICAL SCENARIO IS MANDATORY. A hypothetical shock never checked
    # against the observed record is an assumption, not a measurement.
    if STRESS_SCENARIO_HISTORICAL not in scenarios:
        problems.append(
            "the historical scenario is missing — a synthetic shock that is "
            "never checked against the observed record is an assumption"
        )
    for name in STRESS_SCENARIOS:
        if name not in scenarios:
            problems.append(f"scenario {name!r} was not reported")

    for name, scenario in scenarios.items():
        if not isinstance(scenario, Mapping):
            problems.append(f"{name}: scenario is not a mapping")
            continue
        if not str(scenario.get("reason") or "").strip():
            problems.append(f"{name}: scenario with no reason")

        baseline = scenario.get("baseline_correlation")
        stressed = scenario.get("stressed_correlation")
        # STRESS MUST NOT LOOK CALMER THAN CALM. This is the estimator trap,
        # caught at the contract boundary as well as at the source.
        if baseline is not None and stressed is not None:
            if float(stressed) < float(baseline) - 1e-9:
                problems.append(
                    f"{name}: stressed correlation {stressed:.4f} is BELOW "
                    f"the baseline {baseline:.4f}. MEASURED, that is the "
                    f"signature of demeaning within the stress subset, which "
                    f"subtracts away the crash"
                )

        multiple = scenario.get("volatility_multiple")
        if multiple is not None and float(multiple) < 1.0 - 1e-9:
            problems.append(
                f"{name}: stress LOWERED portfolio volatility "
                f"({multiple:.4f}x) — a stress scenario that makes the "
                f"portfolio safer is measuring the estimator, not the risk"
            )

        lost = scenario.get("diversification_lost")
        if lost is not None and scenario.get("material") is not None:
            expected = float(lost) >= float(
                scenario.get("threshold", STRESS_MATERIAL_DEGRADATION)
            )
            if bool(scenario["material"]) != expected:
                problems.append(
                    f"{name}: materiality {scenario['material']} disagrees "
                    f"with a {lost:.4f} loss against its threshold"
                )
        # THE SHAPE RULE: a figure exists IFF it was measured.
        for field in ("baseline_diversification", "stressed_diversification"):
            value = scenario.get(field)
            if value is not None and float(value) <= 0:
                problems.append(
                    f"{name}: {field} is {value!r} — a ratio of 0 reads as "
                    f"perfect diversification rather than unmeasured"
                )
    return problems


def render_stress(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per scenario."""
    lines: list[str] = []
    if report.get("status") == "NOT_EVALUATED":
        return [f"  stress NOT_EVALUATED: {report.get('reason')}"]
    scenarios = report.get("scenarios") or {}
    for name in report.get("scenario_order") or sorted(scenarios):
        scenario = scenarios.get(name)
        if not isinstance(scenario, Mapping):
            continue
        multiple = scenario.get("volatility_multiple")
        shown = "—" if multiple is None else f"{float(multiple):5.2f}x"
        lost = scenario.get("diversification_lost")
        shown_lost = "—" if lost is None else f"{float(lost):+6.1%}"
        flag = "  !!" if scenario.get("material") else ""
        lines.append(
            f"  {name:26s} vol {shown}   diversification {shown_lost}{flag}"
        )
    return lines
