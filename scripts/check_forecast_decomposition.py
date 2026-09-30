"""CI drift gate for the F6 forecast decomposition.

The load-bearing property is that this decomposition does NOT imply causation
and does NOT pretend to add up. The obvious "improvement" — give each
component a number that sums to the forecast — is arithmetically wrong here,
and it is exactly what a future reader will try to add.

1.  THE DECOMPOSITION IS NOT ADDITIVE, and the reason travels with the code.
    MEASURED over 4,000 observations with a realistic regime/chart
    correlation: regime +0.064, chart +0.067, sum +0.131, ACTUAL joint +0.060.
    The parts overlap and double-count by more than 2x;
2.  no component reports a "contribution" or a "cause" — the vocabulary
    itself is guarded, because the word is what carries the false certainty;
3.  NOT_WIRED stays distinct from a measured zero. Reporting an unwired
    component as 0.0 claims it was measured and found irrelevant, which is a
    different fact;
4.  an unwired component carries NO numbers at all;
5.  every declared component appears, in order. A missing row reads as an
    oversight where an explicit NOT_WIRED reads as a fact;
6.  a non-PRESENT component reports no effect and still explains itself;
7.  the output says it describes the FORECAST, so it cannot be mistaken for
    W1's ensemble breakdown of the SCORE (W5: they are different objects);
8.  claim strength comes from F4's select_claim, not a second sample-size
    policy, so a thin slice cannot publish a rate;
9.  the seven components stay seven: dropping one silently narrows what the
    decomposition claims to cover.

Synthetic and deterministic apart from an optional live read.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CONDITIONAL_MIN_SAMPLES_POINT,
    DECOMP_EFFECT_NARROWED,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_EFFECTS,
    DECOMPOSITION_OVERLAP_EVIDENCE,
    DECOMPOSITION_WIRED_COMPONENTS,
)
from core.forecast_conditional import CONDITIONAL_CLAIM_INSUFFICIENT  # noqa: E402
from core.forecast_decomposition import (  # noqa: E402
    DECOMPOSED_OBJECT,
    DecompositionError,
    decompose_conditional_cell,
    decompose_event_forecast,
    decomposition_problems,
    marginal_effect,
    render_components,
)
from core.forecast_event import build_event_forecast  # noqa: E402
from core.event_memory import EventMemory  # noqa: E402


def _snapshot(**overrides):
    payload = {
        "close": 120.0, "rsi": 55.0, "volatility": 0.22, "volume_ratio_20d": 1.1,
        "atr_14": 2.5, "trend_slope_60d": 0.3, "trend_vs_20d_mean": 0.02,
        "market_regime": "bullish", "change_5d": 0.01, "change_20d": 0.05,
        "change_60d": 0.12, "price_vs_ma_50": 1.03, "price_vs_ma_200": 1.15,
    }
    payload.update(overrides)
    return payload


def _memory(index, ticker="AMD"):
    return EventMemory(
        event_id=f"m{index}", ticker=ticker,
        published_time="2025-06-01 00:00:00", event_type="earnings",
        direction="positive", provenance="observed",
        chart_state=_snapshot(close=300.0, atr_14=7.5, rsi=55.0 + (index % 3) * 0.3),
        response={"20d": {"abnormal_return": 0.03 if index % 3 else -0.02,
                          "stock_return": 0.04}},
        attribution={"20d": "event_associated"},
    )


EVENT = {
    "event_id": "probe", "entity": "NVDA", "event_type": "earnings",
    "direction": "positive", "published_time": "2026-01-05 00:00:00",
    "effective_time": "2026-01-05 00:00:00",
}


def main() -> int:
    failures: list[str] = []

    many = CONDITIONAL_MIN_SAMPLES_POINT + 5
    pool = [_memory(i, ticker=f"T{i}") for i in range(many)]
    forecast = build_event_forecast(
        EVENT, "2026-01-05", chart_state=_snapshot(), regime="bullish",
        memories=pool, base_rate=0.55,
    )
    decomposition = decompose_event_forecast(forecast)

    # ---------------------------------------------------------------- 1
    if DECOMPOSITION_ADDITIVE:
        failures.append(
            "the decomposition is marked ADDITIVE. " + DECOMPOSITION_OVERLAP_EVIDENCE
        )
    if decomposition.get("additive") is not False:
        failures.append("a built decomposition does not declare itself non-additive")
    if "MEASURED" not in (decomposition.get("overlap_evidence") or ""):
        failures.append(
            "the overlap evidence lost its measurement — a future reader "
            "cannot tell a finding from an assertion"
        )
    for phrase in ("not what caused", "do not sum"):
        if phrase not in (decomposition.get("disclaimer") or ""):
            failures.append(
                f"the disclaimer no longer says {phrase!r} — a decomposition "
                f"is where a reader is most likely to read causation into "
                f"association"
            )

    # ---------------------------------------------------------------- 2
    for forbidden in ("CONTRIBUTION", "CONTRIBUTED", "CAUSED"):
        if forbidden in DECOMPOSITION_EFFECTS:
            failures.append(
                f"{forbidden!r} became a reportable effect — the components "
                f"overlap and do not sum to the forecast"
            )
    for name, row in (decomposition.get("components") or {}).items():
        for key in row:
            if "contribution" in str(key).lower():
                failures.append(f"{name}: reports a {key!r} field")
        measured = row.get("measured") or {}
        for key in measured:
            if "contribution" in str(key).lower():
                failures.append(f"{name}: a measured effect reports {key!r}")

    # ---------------------------------------------------------------- 3 + 4
    unwired = set(DECOMPOSITION_COMPONENTS) - set(DECOMPOSITION_WIRED_COMPONENTS)
    if not unwired:
        failures.append(
            "every component is now marked wired — if that is real the gate "
            "must be updated deliberately, because NOT_WIRED existing is what "
            "keeps an unmeasured component from reading as a measured zero"
        )
    for name in unwired:
        row = (decomposition.get("components") or {}).get(name) or {}
        if row.get("status") != DECOMP_STATUS_NOT_WIRED:
            failures.append(
                f"{name}: is not wired but reports {row.get('status')!r}"
            )
        if row.get("effect") is not None:
            failures.append(f"{name}: an unwired component reports an effect")
        for key in ("rate", "rate_difference", "samples", "measured"):
            if key in row:
                failures.append(
                    f"{name}: an unwired component carries {key!r} — it was "
                    f"never measured, and a number here claims otherwise"
                )
        if "not zero" not in (row.get("reason") or ""):
            failures.append(
                f"{name}: does not say it is UNMEASURED rather than zero"
            )

    # The GUARDS themselves, exercised directly. The rows built above never
    # pass detail into a non-PRESENT component, so a gate that only inspects
    # them cannot see whether the guard exists — removing it would pass.
    # These call `_component` with detail and an effect on purpose.
    from core.forecast_decomposition import _component  # noqa: PLC0415

    for blocked_status in (DECOMP_STATUS_NOT_WIRED, DECOMP_STATUS_ABSENT):
        smuggled = _component(
            "macro", blocked_status, reason="probe",
            effect=DECOMP_EFFECT_NARROWED,
            rate=0.0, samples=0, rate_difference=0.0,
        )
        if smuggled.get("effect") is not None:
            failures.append(
                f"a {blocked_status} component kept an effect that was passed "
                f"to it — only a PRESENT component has one to report, and this "
                f"is how an unmeasured component starts looking measured"
            )
        for key in ("rate", "samples", "rate_difference"):
            if key in smuggled:
                failures.append(
                    f"a {blocked_status} component kept {key!r} that was "
                    f"passed to it — a number on an unmeasured row claims it "
                    f"was measured and found to be that value"
                )

    # ---------------------------------------------------------------- 5
    if list((decomposition.get("components") or {})) != list(DECOMPOSITION_COMPONENTS):
        failures.append(
            "not every declared component is reported in order — a missing "
            "row reads as an oversight where NOT_WIRED reads as a fact"
        )

    # ---------------------------------------------------------------- 6
    for name, row in (decomposition.get("components") or {}).items():
        if row.get("status") != DECOMP_STATUS_PRESENT:
            if row.get("effect") is not None:
                failures.append(f"{name}: a non-PRESENT component reports an effect")
            if not row.get("reason"):
                failures.append(f"{name}: a non-PRESENT component does not explain itself")

    # ---------------------------------------------------------------- 7
    if decomposition.get("decomposed_object") != DECOMPOSED_OBJECT:
        failures.append(
            "the decomposition does not name the object it describes — it "
            "could be mistaken for W1's ensemble breakdown of the SCORE"
        )
    if DECOMPOSED_OBJECT != "forecast":
        failures.append("F6 must decompose the FORECAST, not another object")

    # ---------------------------------------------------------------- 8
    # A thin slice must not publish a rate.
    thin = marginal_effect(2, 2, 30, 60)
    if thin.get("claim") != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append(
            f"a 2-observation slice earned {thin.get('claim')!r} — F6 must "
            f"use F4's measured floors, not a second sample-size policy"
        )
    if thin.get("rate") is not None:
        failures.append(
            "a 2-observation slice published a rate — it would render as a "
            "real reading built from two observations"
        )
    if thin.get("interval") is not None:
        failures.append(
            "a refused slice supplies an interval — uncertainty beside a "
            "withheld estimate is an estimate by another name"
        )
    healthy = marginal_effect(26, 45, 30, 60)
    if healthy.get("rate") is None:
        failures.append("a well-sampled slice published no rate")
    if healthy.get("additive") is not False:
        failures.append("a measured effect claims to be additive")

    # The arithmetic guard: a narrowed slice cannot exceed its base.
    try:
        marginal_effect(5, 80, 5, 40)
        failures.append(
            "a narrowed slice larger than its base was accepted — that is not "
            "a subset, and its 'effect' would be meaningless"
        )
    except DecompositionError:
        pass

    # ---------------------------------------------------------------- 9
    if len(DECOMPOSITION_COMPONENTS) != 7:
        failures.append(
            f"the decomposition now covers {len(DECOMPOSITION_COMPONENTS)} "
            f"components, not the seven the sprint names"
        )

    # Contract checks on both surfaces.
    for problem in decomposition_problems(decomposition):
        failures.append(f"event decomposition problem: {problem}")

    cell = {
        "condition_value": "bullish", "samples": 40, "successes": 26,
        "base_rate": 0.5, "claim": "POINT", "horizon": "20d", "value": 0.65,
        "interval": {"lower": 0.5, "upper": 0.78},
    }
    conditional = decompose_conditional_cell(cell, total_observations=100)
    for problem in decomposition_problems(conditional):
        failures.append(f"conditional decomposition problem: {problem}")
    if conditional["components"]["regime"]["status"] != DECOMP_STATUS_PRESENT:
        failures.append("an F4 cell's regime component is not PRESENT")
    if conditional["components"]["technical"]["status"] != DECOMP_STATUS_ABSENT:
        failures.append(
            "an F4 cell reports technical as something other than ABSENT — F4 "
            "slices on the regime alone, and padding the row would be the "
            "false certainty this sprint avoids"
        )

    # A refused forecast decomposes without inventing anything.
    empty = decompose_event_forecast(
        build_event_forecast(EVENT, "2026-01-05", chart_state=_snapshot(),
                             regime="bullish", memories=[])
    )
    for problem in decomposition_problems(empty):
        failures.append(f"empty decomposition problem: {problem}")
    if empty.get("present"):
        failures.append(
            f"a forecast with no analogs reports PRESENT components: "
            f"{empty['present']}"
        )
    if "headline_value" in empty:
        failures.append("a refused forecast's decomposition carries a headline value")

    # Rendering never leaves a blank.
    for row in render_components(empty):
        if not row.get("reason"):
            failures.append(f"{row['component']}: renders blank for a reader")

    if failures:
        print("F6 decomposition gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F6 decomposition gate OK:")
    print(
        f"  {len(DECOMPOSITION_COMPONENTS)} components reported in order; "
        f"{len(DECOMPOSITION_WIRED_COMPONENTS)} wired, "
        f"{len(unwired)} explicitly NOT_WIRED (never a measured zero)."
    )
    print("  the decomposition is NOT additive, and carries the measurement that")
    print("  says why: sum +0.131 vs an ACTUAL joint effect of +0.060.")
    print("  no component reports a 'contribution' or a cause.")
    print("  claim strength comes from F4's floors; a thin slice publishes nothing.")
    print("  the output names the FORECAST as its object, distinct from W1's score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
