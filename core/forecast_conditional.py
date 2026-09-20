"""Conditional forecasting (Sprint F4) — P(outcome | context), honestly.

F3 answers "what will this do?". F4 answers "what does this do WHEN the market
looks like THIS?" — `P(+5% in 20D | bullish)` beside `P(+5% in 20D | stress)`.

**The problem conditioning creates.** Slicing history by regime does not give
five datasets; it gives one dataset cut five ways, and the cuts are wildly
uneven. MEASURED over 5 years of SPY (191 PIT-correct sessions):

    bullish 143 | risk_off 32 | range 9 | bearish 5 | stress 2

The roadmap's own example — stress — has TWO observations. Computed naively it
reads `P(up) = 1.00, P(+5%) = 1.00, mean +13.64%`: the most confident-looking
and least trustworthy cell in the whole grid. The point estimate is INVERSELY
informative to its reliability.

**Why there is no single rule here.** Both global policies were measured
against the real slices, and both fail:

- A floor at ``N >= 30`` silences range, bearish AND stress — three of five
  regimes. The grid then cannot answer the question F4 exists to answer. That
  is a loss of ACCURACY, not a safe default.
- No floor publishes the stress cell as fact. That is a loss of RELIABILITY.

So this module does not pick one estimator. It picks, PER CELL, the strongest
claim that cell's own evidence supports — `select_claim`. A cell with 140
observations states a number; a cell with 9 states an interval; a cell with 2
states its sample size and refuses. Every cell says something true, and none
says more than it can.

**The tier ladder** (floors MEASURED, not chosen for roundness — the
derivation is reproduced in `scripts/check_conditional_forecast.py`):

    POINT        N >= 40   |err| > 0.15 in 1.6% of draws (vs 22.7% at N=10)
    INTERVAL     N >= 8    where mean Wilson width first drops below 0.50
    DIRECTIONAL  N >= 20   AND the interval excludes the base rate
    INSUFFICIENT otherwise, carrying N so the reader knows how thin it was

**DIRECTIONAL sits ABOVE INTERVAL, and that inversion is deliberate.** It says
less but demands more evidence. A too-wide interval advertises its own
weakness on its face; a directional claim that simply failed to DETECT an
effect is indistinguishable from one that found none. MEASURED: a large 15pt
effect is detected 24% of the time at N=10 and 40% at N=20. Below 20 a
directional claim is a coin flip wearing the costume of a finding.

**Wilson, not M6.** `core.calibration.prediction_interval` measures spread
ACROSS CV FOLDS. MEASURED: on identical observations it returns width 0.00 at
EVERY N — including N=2. A conditional rate needs uncertainty FROM SAMPLE
SIZE, which is a different quantity that happens to render the same way.
Composing M6 here would be a W5 violation in reverse: reusing the wrong
existing thing instead of building the right one. Wilson's coverage was
verified to hold (0.92–1.00) down to N=2, so the interval is honest at every
tier that uses it.

**Width never stands alone.** MEASURED: a unanimous 5/5 gives Wilson width
0.434 — NARROWER than the well-sampled 9-observation range cell at 0.525.
Width rewards unanimity and unanimity is what tiny samples manufacture, so the
sample floor gates the tier and width only constrains within it.

**The shape rule, inherited from F3.** A cell carries a `value` key IF AND
ONLY IF it is making a POINT claim. `value: None` would coalesce to 0.0 under
the dashboard's `Number(x ?? 0)` idiom and render a withheld probability as
CERTAIN DOWN. An INTERVAL cell carries an interval and NO value; a
DIRECTIONAL cell carries a direction and NO value; an INSUFFICIENT cell
carries neither.

**Governance is untouched.** The W2 veto `market_regime_stress` (STRESS ->
NO_TRADE) is evaluated only in `core/risk_policy.py`. A stress cell reporting
a high P(up) is a REPORT, never permission to trade, and F4 emits no signal
that reaches the veto.

**PIT by delegation.** This module never fetches. The caller supplies the
observation set — each entry carrying the condition that held AT its as_of and
the realized outcome of a window that had ALREADY CLOSED by the time of the
report. Conditioning on a regime computed from future bars would be the exact
leak C3/C5 was fixed for.
"""

from __future__ import annotations

import logging
import math

from core.config import (
    COND_STATUS_EMPTY_SLICE,
    COND_STATUS_INSUFFICIENT,
    COND_STATUS_LABEL_UNBACKED,
    COND_STATUS_NO_CONDITION,
    COND_STATUS_OK,
    COND_STATUS_PENDING,
    COND_STATUS_UNAVAILABLE,
    CONDITIONAL_CLAIM_DIRECTIONAL,
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_INTERVAL,
    CONDITIONAL_CLAIM_POINT,
    CONDITIONAL_CONDITION_REGIME,
    CONDITIONAL_CONTRACT_VERSION,
    CONDITIONAL_EMPTY_SLICE_N,
    CONDITIONAL_FORECAST_VERSION,
    CONDITIONAL_INTERVAL_LEVEL,
    CONDITIONAL_INTERVAL_METHOD,
    CONDITIONAL_INTERVAL_Z,
    CONDITIONAL_MAX_INTERVAL_WIDTH,
    CONDITIONAL_MIN_SAMPLES_DIRECTIONAL,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    FORECAST_HORIZONS,
    FORECAST_TARGETS,
    REGIME_LABELS,
)
from core.forecast_horizons import HORIZON_STATUS_PENDING, HORIZON_STATUS_SCORABLE
from core.forecast_targets import is_probability_target, target_contract

LOGGER = logging.getLogger("core.forecast_conditional")

DIRECTION_HIGHER = "HIGHER"
DIRECTION_LOWER = "LOWER"
DIRECTION_INDISTINGUISHABLE = "INDISTINGUISHABLE"


class ConditionalForecastError(ValueError):
    """Raised when a conditional-forecast request violates the F4 contract."""


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int,
    trials: int,
    z: float = CONDITIONAL_INTERVAL_Z,
) -> dict | None:
    """Wilson score interval for a rate — uncertainty FROM SAMPLE SIZE.

    Returns None at ``trials == 0``: an interval over no observations is
    ``[0, 1]``, which is not an answer but the absence of one, and returning
    it would let an empty slice masquerade as a very uncertain finding.

    Wilson rather than the normal approximation because the normal interval
    collapses to zero width at ``p = 0`` or ``p = 1`` — exactly the unanimous
    tiny samples this module exists to guard against. MEASURED: Wilson keeps
    its 95% promise (coverage 0.92-1.00) down to N=2.
    """
    trials = int(trials)
    successes = int(successes)
    if trials <= 0:
        return None
    if successes < 0 or successes > trials:
        raise ConditionalForecastError(
            f"successes ({successes}) must lie within [0, trials] ({trials})"
        )

    point = successes / trials
    denominator = 1.0 + (z * z) / trials
    centre = (point + (z * z) / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(point * (1.0 - point) / trials + (z * z) / (4 * trials * trials))
        / denominator
    )
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return {
        "method": CONDITIONAL_INTERVAL_METHOD,
        "level": CONDITIONAL_INTERVAL_LEVEL,
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "width": round(upper - lower, 6),
        "samples": trials,
        "successes": successes,
    }


def interval_width(interval: dict | None) -> float | None:
    """The width of an interval, or None when there is no interval."""
    if not interval:
        return None
    lower, upper = interval.get("lower"), interval.get("upper")
    if lower is None or upper is None:
        return None
    return float(upper) - float(lower)


def excludes(interval: dict | None, reference: float | None) -> bool:
    """True when the interval lies strictly to one side of the reference.

    This is the DIRECTIONAL test: it asks whether the conditional rate is
    distinguishable from the unconditional one, not whether it differs
    numerically. Every finite sample differs numerically.
    """
    if not interval or reference is None:
        return False
    return (
        float(interval["lower"]) > float(reference)
        or float(interval["upper"]) < float(reference)
    )


def direction_of(interval: dict | None, reference: float | None) -> str:
    """HIGHER / LOWER / INDISTINGUISHABLE relative to the base rate."""
    if not excludes(interval, reference):
        return DIRECTION_INDISTINGUISHABLE
    return (
        DIRECTION_HIGHER
        if float(interval["lower"]) > float(reference)
        else DIRECTION_LOWER
    )


# ---------------------------------------------------------------------------
# The selection mechanism — the heart of F4
# ---------------------------------------------------------------------------


def select_claim(
    samples: int,
    interval: dict | None = None,
    base_rate: float | None = None,
) -> tuple[str, str]:
    """Pick the STRONGEST claim this cell's own evidence supports.

    Returns ``(claim, reason)``. This is the per-cell method selection the
    sprint turns on: no global estimator, no global refusal threshold. A cell
    with 140 observations states a number, one with 9 states an interval, one
    with 2 states its sample size and stops.

    Tiers are tried STRONGEST-FIRST and the first that qualifies wins, which
    is equivalent to "the last qualifying tier in weakest-first precedence"
    and is the ordering `CONDITIONAL_CLAIM_PRECEDENCE` declares.
    """
    samples = int(samples)
    if samples <= CONDITIONAL_EMPTY_SLICE_N:
        return (
            CONDITIONAL_CLAIM_INSUFFICIENT,
            "the condition never occurred in the observed history",
        )

    width = interval_width(interval)

    # POINT — the strongest claim. Needs the highest floor AND a usable width.
    if samples >= CONDITIONAL_MIN_SAMPLES_POINT:
        if width is not None and width > CONDITIONAL_MAX_INTERVAL_WIDTH:
            # Reachable: a well-sampled but near-50/50 rate can still be wide.
            return (
                CONDITIONAL_CLAIM_INTERVAL,
                f"{samples} observations clear the point-estimate floor, but the "
                f"interval is {width:.3f} wide, so the range is the finding",
            )
        return (
            CONDITIONAL_CLAIM_POINT,
            f"{samples} observations support a point estimate "
            f"(floor {CONDITIONAL_MIN_SAMPLES_POINT})",
        )

    # DIRECTIONAL — says less than INTERVAL but demands MORE evidence. See the
    # module docstring: a directional claim that failed to detect an effect is
    # indistinguishable from one that found none, so it needs real power.
    if samples >= CONDITIONAL_MIN_SAMPLES_DIRECTIONAL and excludes(interval, base_rate):
        return (
            CONDITIONAL_CLAIM_DIRECTIONAL,
            f"{samples} observations cannot pin a number, but the interval "
            f"excludes the unconditional rate of {float(base_rate):.3f}",
        )

    # INTERVAL — honest at low N (Wilson coverage holds), merely wide. A wide
    # interval is self-limiting in a way a wrong point estimate is not.
    if samples >= CONDITIONAL_MIN_SAMPLES_INTERVAL:
        if width is not None and width > CONDITIONAL_MAX_INTERVAL_WIDTH:
            return (
                CONDITIONAL_CLAIM_INSUFFICIENT,
                f"{samples} observations give an interval {width:.3f} wide, which "
                f"spans too much of the range to constrain anything",
            )
        return (
            CONDITIONAL_CLAIM_INTERVAL,
            f"{samples} observations support a range but not a point estimate "
            f"(point floor {CONDITIONAL_MIN_SAMPLES_POINT})",
        )

    # Below every floor. The sample size IS the finding.
    if samples >= CONDITIONAL_MIN_SAMPLES_DIRECTIONAL:
        return (
            CONDITIONAL_CLAIM_INSUFFICIENT,
            f"{samples} observations, and the interval does not separate this "
            f"condition from the unconditional rate",
        )
    return (
        CONDITIONAL_CLAIM_INSUFFICIENT,
        f"{samples} observations is below every claim floor "
        f"(weakest is {CONDITIONAL_MIN_SAMPLES_INTERVAL}); at this sample size a "
        f"rate of 1.00 and a rate of 0.50 are not distinguishable",
    )


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def slice_observations(
    observations,
    condition_value: str,
    condition: str = CONDITIONAL_CONDITION_REGIME,
) -> list[dict]:
    """The observations whose condition held at their own as_of.

    An observation missing the condition key is DROPPED, not treated as
    non-matching: "the regime was not bullish" and "the regime could not be
    established" are different facts, and folding the second into the first
    would inflate every other slice with unresolved sessions.
    """
    kept: list[dict] = []
    for observation in observations or ():
        if not isinstance(observation, dict):
            continue
        value = (observation.get("conditions") or {}).get(condition)
        if value is None:
            continue
        if str(value) == str(condition_value):
            kept.append(observation)
    return kept


def _outcome_hits(observations, target: str) -> tuple[int, int, list[float]]:
    """(successes, trials, raw values) for a target over an observation set.

    For a probability target the outcome is a hit/miss; for a magnitude target
    the raw values are returned and the caller aggregates. An observation with
    no recorded outcome for this target is excluded from the DENOMINATOR as
    well as the numerator — counting it as a miss would bias every rate toward
    zero in exactly the thin slices that can least afford it.
    """
    successes = 0
    trials = 0
    values: list[float] = []
    for observation in observations:
        outcomes = observation.get("outcomes") or {}
        if target not in outcomes:
            continue
        value = outcomes[target]
        if value is None:
            continue
        trials += 1
        values.append(float(value))
        if bool(value) if isinstance(value, bool) else float(value) > 0.0:
            successes += 1
    return successes, trials, values


def condition_counts(
    observations,
    condition: str = CONDITIONAL_CONDITION_REGIME,
) -> dict[str, int]:
    """How many observations each condition value holds — the F4 headline.

    Reported for EVERY known regime including those with zero, because a
    missing row reads as an oversight while a zero reads as a fact.
    """
    counts = {label: 0 for label in REGIME_LABELS}
    for observation in observations or ():
        if not isinstance(observation, dict):
            continue
        value = (observation.get("conditions") or {}).get(condition)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def resolve_cell_status(
    target: str,
    horizon: str,
    *,
    has_observations: bool,
    readiness: str,
    condition_resolved: bool,
    label_backed: bool,
    samples: int,
    claim: str,
) -> tuple[str, str]:
    """The single status for one conditional cell, by DECLARED precedence.

    Ordered in `CONDITIONAL_CELL_PRECEDENCE` so a cell is never ambiguous
    about why it is empty: the most fundamental obstacle wins and the reason
    names it.
    """
    contract = target_contract(target)

    if not has_observations:
        return COND_STATUS_UNAVAILABLE, "no observation set was supplied"

    if readiness == HORIZON_STATUS_PENDING:
        return (
            COND_STATUS_PENDING,
            f"the {horizon} window has not closed; a conditional outcome here "
            f"would be fabricated",
        )
    if readiness != HORIZON_STATUS_SCORABLE:
        return COND_STATUS_UNAVAILABLE, f"no outcome is recorded for {horizon}"

    if not condition_resolved:
        return (
            COND_STATUS_NO_CONDITION,
            "the condition could not be resolved point-in-time for these "
            "observations, so they cannot be attributed to any slice",
        )

    if not label_backed:
        return (
            COND_STATUS_LABEL_UNBACKED,
            f"no realized {contract['label_field']!r} exists at {horizon}, so "
            f"this target could never be scored there — conditioned or not",
        )

    if samples <= CONDITIONAL_EMPTY_SLICE_N:
        return (
            COND_STATUS_EMPTY_SLICE,
            "this condition never occurred in the observed history; that is "
            "different from occurring rarely, and the fix is a longer window",
        )

    if claim == CONDITIONAL_CLAIM_INSUFFICIENT:
        return (
            COND_STATUS_INSUFFICIENT,
            f"{samples} observations cannot support any claim about this cell",
        )

    return COND_STATUS_OK, ""


def build_cell(
    target: str,
    horizon: str,
    condition_value: str,
    *,
    observations=None,
    readiness: str = HORIZON_STATUS_SCORABLE,
    condition: str = CONDITIONAL_CONDITION_REGIME,
    base_rate: float | None = None,
    condition_resolved: bool = True,
    label_backed: bool = True,
    comparisons: int | None = None,
) -> dict:
    """One (target, horizon, condition) cell, at the claim strength it earns.

    THE SHAPE RULE, inherited from F3 and extended: a `value` key appears IF
    AND ONLY IF the claim is POINT. An INTERVAL cell carries an interval and
    no value; a DIRECTIONAL cell carries a direction and no value; a refused
    cell carries neither. `value: None` would coalesce to 0.0 in the dashboard
    and render a withheld probability as certain-down.
    """
    if target not in FORECAST_TARGETS:
        raise ConditionalForecastError(
            f"unknown target {target!r} (known: {sorted(FORECAST_TARGETS)})"
        )
    if horizon not in FORECAST_HORIZONS:
        raise ConditionalForecastError(
            f"unknown horizon {horizon!r} (known: {list(FORECAST_HORIZONS)})"
        )

    contract = target_contract(target)
    has_observations = observations is not None
    matched = slice_observations(observations or (), condition_value, condition)
    successes, trials, values = _outcome_hits(matched, target)

    probability_like = is_probability_target(target)
    interval = wilson_interval(successes, trials) if probability_like else None

    claim, claim_reason = select_claim(trials, interval, base_rate)

    status, status_reason = resolve_cell_status(
        target,
        horizon,
        has_observations=has_observations,
        readiness=readiness,
        condition_resolved=condition_resolved,
        label_backed=label_backed,
        samples=trials,
        claim=claim,
    )

    cell = {
        "target": target,
        "horizon": horizon,
        "condition": condition,
        "condition_value": condition_value,
        "status": status,
        "claim": claim if status == COND_STATUS_OK else CONDITIONAL_CLAIM_INSUFFICIENT,
        "kind": contract["kind"],
        "unit": contract["unit"],
        "question": contract["question"],
        "samples": trials,
        "successes": successes if probability_like else None,
        "base_rate": base_rate,
        # A refused cell supplies NO uncertainty either — an interval beside a
        # withheld estimate is an estimate by another name.
        "interval": None,
        "direction": None,
        "reason": status_reason or claim_reason,
        # Directional claims carry their multiplicity exposure so a reader can
        # weigh them. 180 cells from one history yields ~9 spurious findings
        # at a 5% false-signal rate.
        "comparisons": comparisons,
    }

    if status != COND_STATUS_OK:
        return cell

    if claim == CONDITIONAL_CLAIM_POINT:
        point = (successes / trials) if probability_like else (
            sum(values) / len(values) if values else None
        )
        if point is None:
            # Defensive: POINT was selected but nothing can be computed. Fall
            # back rather than emit a null value through the front door.
            cell["status"] = COND_STATUS_INSUFFICIENT
            cell["claim"] = CONDITIONAL_CLAIM_INSUFFICIENT
            cell["reason"] = "no values were available to average"
            return cell
        cell["value"] = round(float(point), 6)
        cell["interval"] = interval
    elif claim == CONDITIONAL_CLAIM_INTERVAL:
        cell["interval"] = interval
    elif claim == CONDITIONAL_CLAIM_DIRECTIONAL:
        cell["direction"] = direction_of(interval, base_rate)
        cell["interval"] = interval

    return cell


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def build_conditional_forecast(
    ticker: str,
    as_of,
    *,
    observations=None,
    targets: tuple[str, ...] = FORECAST_TARGETS,
    horizons: tuple[str, ...] = FORECAST_HORIZONS,
    conditions: tuple[str, ...] = REGIME_LABELS,
    condition: str = CONDITIONAL_CONDITION_REGIME,
    base_rates: dict | None = None,
    readiness: dict | None = None,
) -> dict:
    """The full conditional grid for one ticker at one as_of.

    `observations` is the PIT-correct evidence set: each entry carries the
    condition that held AT its own as_of and the realized outcomes of windows
    that had ALREADY CLOSED. This module never fetches and never filters —
    conditioning on a regime computed from future bars is the leak C3/C5 was
    fixed for, and the only defence is that the caller owns the timeline.
    """
    if not str(ticker or "").strip():
        raise ConditionalForecastError("a ticker is required")

    base_rates = base_rates or {}
    readiness = readiness or {}
    counts = condition_counts(observations or (), condition)

    # Every DIRECTIONAL claim in this grid was drawn from this many cells.
    comparisons = len(targets) * len(horizons) * len(conditions)

    rows: dict[str, dict] = {}
    emitted_points = 0
    emitted_intervals = 0
    emitted_directions = 0

    for horizon in horizons:
        cells: dict[str, dict] = {}
        horizon_readiness = readiness.get(horizon, HORIZON_STATUS_SCORABLE)
        for target in targets:
            by_condition: dict[str, dict] = {}
            for value in conditions:
                cell = build_cell(
                    target,
                    horizon,
                    value,
                    observations=observations,
                    readiness=horizon_readiness,
                    condition=condition,
                    base_rate=(base_rates.get(target) or {}).get(horizon)
                    if isinstance(base_rates.get(target), dict)
                    else base_rates.get(target),
                    comparisons=comparisons,
                )
                if "value" in cell:
                    emitted_points += 1
                elif cell.get("direction") not in (None, DIRECTION_INDISTINGUISHABLE):
                    emitted_directions += 1
                elif cell.get("interval") is not None:
                    emitted_intervals += 1
                by_condition[value] = cell
            cells[target] = by_condition
        rows[horizon] = {"horizon": horizon, "cells": cells}

    return {
        "ticker": str(ticker).upper(),
        "as_of": str(as_of),
        "condition": condition,
        "condition_order": list(conditions),
        "condition_counts": counts,
        "horizon_order": list(horizons),
        "target_order": list(targets),
        "rows": rows,
        "comparisons": comparisons,
        "emitted_points": emitted_points,
        "emitted_intervals": emitted_intervals,
        "emitted_directions": emitted_directions,
        "observations": len(observations or ()),
        "calculation_version": CONDITIONAL_FORECAST_VERSION,
        "contract_version": CONDITIONAL_CONTRACT_VERSION,
        # The W2 veto is evaluated only in core/risk_policy.py. Stated here so
        # a reader of a favourable stress cell cannot mistake it for a signal.
        "governance_note": (
            "conditional readings are REPORTS; the STRESS -> NO_TRADE veto is "
            "evaluated in core/risk_policy.py and is unaffected by this grid"
        ),
    }


def conditional_problems(forecast: dict) -> list[str]:
    """Validate a built grid against the F4 contract."""
    problems: list[str] = []
    if not isinstance(forecast, dict):
        return ["forecast must be a dict"]

    for field in ("ticker", "as_of", "condition", "calculation_version"):
        if not forecast.get(field):
            problems.append(f"forecast field {field!r} missing/empty")

    for horizon, row in (forecast.get("rows") or {}).items():
        for target, by_condition in (row.get("cells") or {}).items():
            for value, cell in by_condition.items():
                where = f"{target}@{horizon}|{value}"
                claim = cell.get("claim")
                status = cell.get("status")

                # THE SHAPE RULE: a value key exists IFF the claim is POINT.
                if claim == CONDITIONAL_CLAIM_POINT:
                    if "value" not in cell:
                        problems.append(f"{where}: a POINT claim carries no value")
                elif "value" in cell:
                    problems.append(
                        f"{where}: a {claim} claim carries a value key — it would "
                        f"coalesce to 0 in a consumer and render as a real reading"
                    )

                if status != COND_STATUS_OK:
                    if cell.get("interval") is not None:
                        problems.append(
                            f"{where}: a refused cell supplies uncertainty — an "
                            f"interval beside a withheld estimate is an estimate "
                            f"by another name"
                        )
                    if cell.get("direction") is not None:
                        problems.append(f"{where}: a refused cell states a direction")
                    if not cell.get("reason"):
                        problems.append(f"{where}: a refused cell does not explain itself")

                if claim == CONDITIONAL_CLAIM_DIRECTIONAL:
                    if cell.get("direction") in (None, DIRECTION_INDISTINGUISHABLE):
                        problems.append(
                            f"{where}: a DIRECTIONAL claim names no direction"
                        )
                    if not cell.get("comparisons"):
                        problems.append(
                            f"{where}: a DIRECTIONAL claim carries no multiplicity "
                            f"exposure, so a reader cannot weigh it against the "
                            f"other cells it was drawn from"
                        )

                samples = cell.get("samples")
                if claim == CONDITIONAL_CLAIM_POINT and (
                    samples or 0
                ) < CONDITIONAL_MIN_SAMPLES_POINT:
                    problems.append(
                        f"{where}: a POINT claim rests on {samples} observations, "
                        f"below the measured floor of {CONDITIONAL_MIN_SAMPLES_POINT}"
                    )
                if claim == CONDITIONAL_CLAIM_DIRECTIONAL and (
                    samples or 0
                ) < CONDITIONAL_MIN_SAMPLES_DIRECTIONAL:
                    problems.append(
                        f"{where}: a DIRECTIONAL claim rests on {samples} "
                        f"observations, below the floor of "
                        f"{CONDITIONAL_MIN_SAMPLES_DIRECTIONAL}"
                    )

                width = interval_width(cell.get("interval"))
                if width is not None and width > CONDITIONAL_MAX_INTERVAL_WIDTH:
                    problems.append(
                        f"{where}: published an interval {width:.3f} wide, above "
                        f"the cap of {CONDITIONAL_MAX_INTERVAL_WIDTH}"
                    )

    return problems


def render_rows(
    forecast: dict,
    target: str = "probability_up",
) -> list[dict]:
    """One reading row per horizon for one target, in declared order.

    Each row renders every condition side by side — the comparison F4 exists
    to make readable. A cell that earned no claim renders its reason, so the
    grid never has a blank a reader must interpret.
    """
    rows: list[dict] = []
    for horizon in forecast.get("horizon_order") or ():
        row = (forecast.get("rows") or {}).get(horizon) or {}
        by_condition = (row.get("cells") or {}).get(target) or {}
        rendered = {}
        for value in forecast.get("condition_order") or ():
            cell = by_condition.get(value) or {}
            rendered[value] = {
                "claim": cell.get("claim"),
                "samples": cell.get("samples"),
                "value": cell.get("value"),
                "interval": cell.get("interval"),
                "direction": cell.get("direction"),
                "reason": cell.get("reason"),
            }
        rows.append({"horizon": horizon, "target": target, "conditions": rendered})
    return rows
