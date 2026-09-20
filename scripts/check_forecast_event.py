"""CI drift gate for the F5 event-conditioned forecast.

The load-bearing property of F5 is that it is a PIPELINE that names the stage
which failed, and that it never claims more than its retrieval independently
supports. A future edit that returns a bare number, or that counts one
ticker's history as many observations, would look simpler and would publish
confident nonsense.

1.  the declared stages are complete, ordered, and the forecast is LAST;
2.  the FIRST failure stops the pipeline — later stages read BLOCKED, so a
    reader sees the real obstacle, not a downstream symptom;
3.  an empty memory store REFUSES at `matches` and says the store is empty.
    MEASURED: on a fresh clone data/event_memory.jsonl does not exist, so this
    is the ordinary state of the system, not an edge case;
4.  PSEUDO-REPLICATION IS CAPPED. 45 analogs from ONE ticker are 45 adjacent
    sessions of one situation, not 45 observations. They must NOT reach the
    POINT tier however many rows they contain — this is the F4 stress cell in
    another costume;
5.  ...while genuinely diverse analogs still reach it, so the cap is a floor
    on honesty rather than a blanket refusal;
6.  THE SHAPE RULE — a `value` key exists IFF the claim is POINT, and a
    refused forecast supplies no interval either;
7.  F5 composes E6 retrieval rather than adding a third similarity metric
    beside E6 and C7 (W5: one canonical implementation);
8.  the claim strength comes from F4's select_claim, so an analog set and a
    regime slice of the same size get the same answer — one sample-size
    policy, not two that drift apart;
9.  regime is REPORTED, not filtered on. `market_regime` is already an E6
    similarity field, so filtering again would double-weight it;
10. a confounded analog set is flagged: E5 already separates an
    event-associated response from a confounded one.

Synthetic and deterministic — no network, no wall clock.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CONDITIONAL_MIN_SAMPLES_POINT,
    EVENT_FORECAST_EFFECTIVE_PER_TICKER,
    EVENT_FORECAST_MIN_DISTINCT_FOR_POINT,
    EVENT_FORECAST_OK,
    EVENT_FORECAST_REFUSED,
    EVENT_FORECAST_RETRIEVAL,
    EVENT_FORECAST_STAGE_FORECAST,
    EVENT_FORECAST_STAGE_MATCHES,
    EVENT_FORECAST_STAGES,
    EVENT_MEMORY_SIMILARITY_FIELDS,
    EVENT_STAGE_BLOCKED,
    EVENT_STAGE_DEGRADED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_OK,
)
from core.event_memory import EventMemory  # noqa: E402
from core.forecast_conditional import (  # noqa: E402
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_POINT,
    select_claim,
    wilson_interval,
)
from core.forecast_event import (  # noqa: E402
    build_event_forecast,
    event_forecast_problems,
    render_pipeline,
    represent_event,
)

EVENT = {
    "event_id": "probe", "entity": "NVDA", "event_type": "earnings",
    "direction": "positive", "published_time": "2026-01-05 00:00:00",
    "effective_time": "2026-01-05 00:00:00",
}


def _snapshot(**overrides):
    payload = {
        "close": 120.0, "rsi": 55.0, "volatility": 0.22, "volume_ratio_20d": 1.1,
        "atr_14": 2.5, "trend_slope_60d": 0.3, "trend_vs_20d_mean": 0.02,
        "market_regime": "bullish", "change_5d": 0.01, "change_20d": 0.05,
        "change_60d": 0.12, "price_vs_ma_50": 1.03, "price_vs_ma_200": 1.15,
    }
    payload.update(overrides)
    return payload


def _memory(index, ticker="AMD", associated=True, price=300.0):
    return EventMemory(
        event_id=f"m{index}", ticker=ticker,
        published_time="2025-06-01 00:00:00", event_type="earnings",
        direction="positive",
        chart_state=_snapshot(close=price, atr_14=price / 40,
                              rsi=55.0 + (index % 3) * 0.3),
        response={"20d": {"abnormal_return": 0.03 if index % 3 else -0.02,
                          "stock_return": 0.04}},
        attribution={"20d": "event_associated" if associated else "confounded"},
    )


def _run(memories, regime="bullish", chart=None, event=EVENT):
    return build_event_forecast(
        event, "2026-01-05",
        chart_state=_snapshot() if chart is None else chart,
        regime=regime, memories=memories, base_rate=0.55,
    )


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    if EVENT_FORECAST_STAGES[-1] != EVENT_FORECAST_STAGE_FORECAST:
        failures.append(
            "the forecast is not the LAST stage — anything after it would be "
            "reasoning the published claim did not account for"
        )
    if len(set(EVENT_FORECAST_STAGES)) != len(EVENT_FORECAST_STAGES):
        failures.append("EVENT_FORECAST_STAGES contains a duplicate")
    if EVENT_FORECAST_STAGES.index(EVENT_FORECAST_STAGE_MATCHES) >= len(
        EVENT_FORECAST_STAGES
    ) - 1:
        failures.append("retrieval must precede the forecast")

    # ---------------------------------------------------------------- 3
    empty = _run([])
    if empty["status"] != EVENT_FORECAST_REFUSED:
        failures.append("an empty memory store produced a forecast")
    if empty["failed_stage"] != EVENT_FORECAST_STAGE_MATCHES:
        failures.append(
            f"an empty store failed at {empty['failed_stage']!r} rather than at "
            f"retrieval — the reader would not learn that nothing is stored"
        )
    if "empty" not in (empty["stages"][EVENT_FORECAST_STAGE_MATCHES]["reason"] or ""):
        failures.append(
            "an empty store does not SAY it is empty — 'no analogs' and 'nothing "
            "has ever been recorded' have different fixes"
        )
    if "value" in empty:
        failures.append("a refused forecast carries a value key")

    # ---------------------------------------------------------------- 2
    for name in EVENT_FORECAST_STAGES[
        EVENT_FORECAST_STAGES.index(EVENT_FORECAST_STAGE_MATCHES) + 1:
    ]:
        if empty["stages"][name]["status"] != EVENT_STAGE_BLOCKED:
            failures.append(
                f"stage {name!r} ran after retrieval failed — a reader would see "
                f"a downstream symptom instead of the real obstacle"
            )

    # ---------------------------------------------------------------- 4
    # The defect this gate exists for.
    many = CONDITIONAL_MIN_SAMPLES_POINT + 5
    one_ticker = _run([_memory(i, ticker="NVDA") for i in range(many)])
    if one_ticker.get("claim") == CONDITIONAL_CLAIM_POINT:
        failures.append(
            f"{many} analogs from ONE ticker earned a POINT claim — they are "
            f"adjacent sessions of one situation, not {many} independent "
            f"observations, and this is the F4 stress cell in another costume"
        )
    if "value" in one_ticker:
        failures.append(
            "an all-one-ticker analog set published a value — it would render "
            "as a real reading built from a single stock's own history"
        )
    stage = one_ticker["stages"][EVENT_FORECAST_STAGE_FORECAST]
    if stage.get("effective_samples") != EVENT_FORECAST_EFFECTIVE_PER_TICKER:
        failures.append(
            f"an all-one-ticker set reports effective_samples "
            f"{stage.get('effective_samples')} rather than "
            f"{EVENT_FORECAST_EFFECTIVE_PER_TICKER} — the discount is not applied"
        )
    if one_ticker["stages"][EVENT_FORECAST_STAGE_MATCHES]["status"] != EVENT_STAGE_DEGRADED:
        failures.append("an all-one-ticker analog set is not flagged at retrieval")
    if EVENT_FORECAST_EFFECTIVE_PER_TICKER >= CONDITIONAL_MIN_SAMPLES_POINT:
        failures.append(
            "one ticker alone can reach the POINT floor — a single stock's own "
            "history could publish a point estimate"
        )

    # ---------------------------------------------------------------- 5
    diverse = _run([_memory(i, ticker=f"T{i}") for i in range(many)])
    if diverse.get("claim") != CONDITIONAL_CLAIM_POINT:
        failures.append(
            f"{many} analogs across {many} DISTINCT tickers failed to earn a "
            f"POINT claim (got {diverse.get('claim')}) — the independence cap "
            f"has become a blanket refusal, which destroys accuracy"
        )
    if "value" not in diverse:
        failures.append("a POINT claim carries no value")
    if diverse["status"] != EVENT_FORECAST_OK:
        failures.append("a well-evidenced forecast was refused")

    # ---------------------------------------------------------------- 6
    for label, forecast in (("empty", empty), ("one-ticker", one_ticker),
                            ("diverse", diverse)):
        if forecast.get("claim") == CONDITIONAL_CLAIM_POINT:
            if "value" not in forecast:
                failures.append(f"{label}: a POINT claim carries no value")
        elif "value" in forecast:
            failures.append(
                f"{label}: a {forecast.get('claim')} claim carries a value key — "
                f"it would coalesce to 0 in a consumer and render as a reading"
            )
        if forecast["status"] == EVENT_FORECAST_REFUSED:
            if forecast.get("interval") is not None:
                failures.append(
                    f"{label}: a refused forecast supplies an interval — "
                    f"uncertainty beside a withheld estimate is an estimate"
                )
            if not forecast.get("failed_stage"):
                failures.append(f"{label}: a refusal does not name the stage")
        for problem in event_forecast_problems(forecast):
            failures.append(f"{label}: contract problem: {problem}")

    # ---------------------------------------------------------------- 6b
    # THE REFUSAL PATHS must be exercised, not assumed. A forecast that
    # refuses AFTER retrieval succeeded is the case that can actually carry a
    # stale interval or value: the earlier refusals return before either is
    # computed, so checking only those proves nothing.
    # The analogs must produce a MIXED outcome so a Wilson interval genuinely
    # exists on this path; an all-identical set can hide an interval leak by
    # making the leaked value indistinguishable from the absent one.
    thin = _run([_memory(i, ticker=f"S{i}") for i in range(3)])
    if thin["samples"] != 3:
        failures.append(
            f"the thin fixture yielded {thin['samples']} scorable analogs, not 3 — "
            f"the refusal path is no longer being exercised"
        )
    # Prove an interval WOULD exist here, so `interval is None` is a real
    # withholding rather than an accident of the fixture.
    would_exist = wilson_interval(
        thin["stages"][EVENT_FORECAST_STAGE_FORECAST].get("successes", 0),
        thin["samples"] or 1,
    )
    if would_exist is None:
        failures.append(
            "the thin fixture cannot produce an interval at all, so this check "
            "cannot prove the refusal WITHHELD one"
        )

    # The decisive fixture: retrieval SUCCEEDS and an interval genuinely
    # exists, but the claim is capped to INSUFFICIENT by the independence
    # discount. This is the only path where a leak is both possible and
    # meaningful, so it is the one the shape rule must be tested on.
    capped = _run([_memory(i, ticker="NVDA") for i in range(many)])
    if capped["status"] != EVENT_FORECAST_REFUSED:
        failures.append("the independence-capped fixture is no longer refused")
    if capped["samples"] < CONDITIONAL_MIN_SAMPLES_POINT:
        failures.append(
            "the capped fixture no longer carries enough analogs for an "
            "interval to exist, so it cannot prove the refusal withheld one"
        )
    if wilson_interval(
        capped["stages"][EVENT_FORECAST_STAGE_FORECAST].get("successes", 0),
        capped["samples"] or 1,
    ) is None:
        failures.append("the capped fixture cannot produce an interval")
    if capped.get("interval") is not None:
        failures.append(
            "a forecast refused by the INDEPENDENCE CAP still publishes its "
            "interval — retrieval succeeded and an interval genuinely exists "
            "here, so this is a real leak: uncertainty beside a withheld "
            "estimate is an estimate by another name"
        )
    if "value" in capped:
        failures.append(
            "a forecast refused by the independence cap carries a value — the "
            "one-ticker base rate would render as a real reading"
        )
    if thin["status"] != EVENT_FORECAST_REFUSED:
        failures.append("3 analogs produced a forecast")
    if thin.get("interval") is not None:
        failures.append(
            "a forecast refused at the LAST stage still carries its interval — "
            "uncertainty beside a withheld estimate is an estimate by another "
            "name, and this is the one refusal path where the interval exists"
        )
    if "value" in thin:
        failures.append(
            "a forecast refused at the last stage carries a value — the shape "
            "rule must hold on the path where a value was actually computed"
        )
    if thin.get("claim") != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append("a refused forecast does not report an INSUFFICIENT claim")

    # An INTERVAL-tier claim must also withhold its point estimate: this is
    # the path where `value` is most tempting to emit, because one exists.
    interval_tier = _run([_memory(i, ticker=f"V{i}") for i in range(20)])
    if interval_tier.get("claim") == CONDITIONAL_CLAIM_POINT:
        failures.append(
            "20 analogs earned a POINT claim — below F4's measured floor of "
            f"{CONDITIONAL_MIN_SAMPLES_POINT}"
        )
    if "value" in interval_tier:
        failures.append(
            f"a {interval_tier.get('claim')} claim carries a value key — it "
            f"would coalesce to 0 in a consumer and render as a real reading"
        )

    # ---------------------------------------------------------------- 7
    if EVENT_FORECAST_RETRIEVAL != "event_memory.find_analogs":
        failures.append(
            "F5 no longer composes E6 retrieval — a third analog metric beside "
            "E6 and C7 is the split-brain W5 forbids"
        )
    # ...and the constant must describe what the module ACTUALLY calls. A
    # declaration that drifts from behaviour documents an intention, not a
    # fact, so the retrieval is verified by OBSERVING the call.
    import core.forecast_event as _f5
    import core.event_memory as _e6

    called = {"n": 0}
    original = _f5.find_analogs

    def _observed(*args, **kwargs):
        called["n"] += 1
        return original(*args, **kwargs)

    _f5.find_analogs = _observed
    try:
        _run([_memory(i, ticker=f"W{i}") for i in range(6)])
    finally:
        _f5.find_analogs = original
    if called["n"] != 1:
        failures.append(
            f"E6's find_analogs was called {called['n']} time(s) — F5 must "
            f"compose it exactly once per forecast, not reimplement retrieval"
        )
    if _f5.find_analogs is not _e6.find_analogs:
        failures.append(
            "F5's find_analogs is not E6's — retrieval has been forked"
        )

    if EVENT_FORECAST_EFFECTIVE_PER_TICKER * EVENT_FORECAST_MIN_DISTINCT_FOR_POINT > (
        CONDITIONAL_MIN_SAMPLES_POINT
    ):
        failures.append(
            f"fewer than {EVENT_FORECAST_MIN_DISTINCT_FOR_POINT} tickers can "
            f"reach the POINT floor — a point estimate must rest on several "
            f"distinct names"
        )

    # ---------------------------------------------------------------- 8
    # One sample-size policy, not two. An analog set and a regime slice of the
    # same size must receive the same claim.
    for trials, successes in ((3, 2), (8, 5), (45, 30)):
        pool = [_memory(i, ticker=f"D{i}") for i in range(trials)]
        built = _run(pool)
        direct, _ = select_claim(
            built["samples"], wilson_interval(
                built["stages"][EVENT_FORECAST_STAGE_FORECAST].get("successes", 0),
                built["samples"],
            ) if built["samples"] else None,
            0.55,
        )
        if built["samples"] and built.get("claim") != direct:
            failures.append(
                f"an analog set of {built['samples']} earned {built.get('claim')!r} "
                f"while F4 would give {direct!r} for the same evidence — F5 has "
                f"grown a second sample-size policy that will drift from F4's"
            )

    # ---------------------------------------------------------------- 9
    if "market_regime" not in EVENT_MEMORY_SIMILARITY_FIELDS:
        failures.append(
            "market_regime left the E6 similarity fields — F5 reports regime "
            "agreement INSTEAD of filtering precisely because retrieval already "
            "accounts for it, and that reasoning no longer holds"
        )
    regime_stage = diverse["stages"]["regime"]
    if regime_stage.get("filtered") is not False:
        failures.append(
            "the regime stage claims to filter — filtering on a field that is "
            "already inside the similarity double-weights the same evidence"
        )
    if regime_stage.get("agreement") is None:
        failures.append("a supplied regime produced no agreement measure")

    bogus = _run([_memory(i, ticker=f"T{i}") for i in range(8)], regime="euphoric")
    if bogus["failed_stage"] != "regime":
        failures.append(
            "an ungoverned regime label was accepted — the five-state label is "
            "governed, and a sixth would silently bypass the W2 stress veto"
        )

    # ---------------------------------------------------------------- 10
    confounded = _run(
        [_memory(i, ticker=f"T{i}", associated=(i < 2)) for i in range(10)]
    )
    if confounded["stages"][EVENT_FORECAST_STAGE_MATCHES]["status"] != EVENT_STAGE_DEGRADED:
        failures.append(
            "a mostly-confounded analog set was not flagged — a base rate built "
            "from confounded responses describes the market, not the event"
        )

    # Representation refusals.
    typeless = _run([_memory(i, ticker=f"T{i}") for i in range(8)],
                    event={**EVENT, "event_type": ""})
    if typeless["failed_stage"] != "representation":
        failures.append(
            "an event with no type reached retrieval — matching is BY type, and "
            "an earnings surprise is not a comparable for a regulatory action"
        )

    if not render_pipeline(diverse):
        failures.append("the pipeline renders no rows")
    for row in render_pipeline(empty):
        if row["status"] != EVENT_STAGE_OK and not row["reason"]:
            failures.append(f"stage {row['stage']!r} renders blank for a reader")

    if failures:
        print("F5 event-forecast gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F5 event-forecast gate OK:")
    print(f"  {len(EVENT_FORECAST_STAGES)} stages, forecast last, first failure blocks the rest.")
    print("  an empty store REFUSES at retrieval and says the store is empty.")
    print(
        f"  pseudo-replication capped: {many} analogs from one ticker -> effective "
        f"{EVENT_FORECAST_EFFECTIVE_PER_TICKER}, no POINT claim; {many} across "
        f"{many} tickers -> POINT."
    )
    print("  a value key exists IFF the claim is POINT; refusals carry no interval.")
    print("  claim strength comes from F4 select_claim; retrieval composes E6 (W5).")
    print("  regime is reported, not filtered; ungoverned labels refused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
