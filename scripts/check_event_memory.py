"""CI drift gate for E6 event memory.

This is the store the forecasting sprints retrieve from without re-deriving
anything, so its value depends on not remembering things that were never
true. The gate proves the refusals hold:

1. event, context, chart state and every horizon response are all recorded;
2. an unmeasured event NEVER enters memory — it teaches nothing while
   counting toward every analog total;
3. a memory without chart state is refused (a log is not institutional
   memory);
4. re-recording cannot inflate the count, which would silently make every
   historical base rate wrong;
5. analogs match on chart state AND event type;
6. a "typical response" from too few examples is refused, with no statistics
   at all;
7. the attribution verdict travels with every memory.

Synthetic only; runs in about a second.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.attribution import attribute_study  # noqa: E402
from core.config import (  # noqa: E402
    EVENT_MEMORY_CHART_FIELDS,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_FIELD_SCALE,
    EVENT_MEMORY_MIN_SIMILARITY,
    EVENT_MEMORY_SIMILARITY_FIELDS,
    EVENT_MEMORY_RESPONSE_HORIZONS,
)
from core.event_contract import Event  # noqa: E402
from core.event_memory import (  # noqa: E402
    EventMemory,
    EventMemoryError,
    analog_summary,
    build_memory,
    chart_similarity,
    find_analogs,
    load_memories,
    memory_problems,
    remember,
)
from core.event_study import study_event  # noqa: E402


def _frame(bars: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, bars))
    return pd.DataFrame(
        {
            "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, bars).astype(float),
        },
        index=pd.bdate_range("2024-01-01", periods=bars),
    )


def _snapshot(**overrides) -> dict:
    payload = {
        "close": 120.0, "rsi": 55.0, "volatility": 0.02, "volume_ratio_20d": 1.1,
        "atr_14": 2.5, "trend_slope_60d": 0.3, "trend_vs_20d_mean": 0.02,
        "market_regime": "bullish", "change_5d": 0.01, "change_20d": 0.05,
        "change_60d": 0.12, "price_vs_ma_50": 1.03, "price_vs_ma_200": 1.15,
    }
    payload.update(overrides)
    return payload


def _memory(event_id: str = "e1", **overrides) -> EventMemory:
    payload = dict(
        event_id=event_id, ticker="NVDA", published_time="2026-01-05 00:00:00",
        event_type="earnings", direction="positive", chart_state=_snapshot(),
        response={"20d": {"abnormal_return": 0.03}},
        attribution={"20d": "event_associated"},
    )
    payload.update(overrides)
    return EventMemory(**payload)


def main() -> int:
    failures: list[str] = []
    frame, benchmark = _frame(), _frame(seed=1)

    # 1. Everything E6 requires is recorded.
    event = Event(
        entity="NVDA", published_time=frame.index[200].strftime("%Y-%m-%d %H:%M:%S"),
        event_type="earnings", source="news", actor="Jensen Huang",
        evidence=[{"source_record_id": "r1"}],
    )
    study = study_event(event, frame, benchmark)
    memory = build_memory(event, study, attribute_study(study), _snapshot())

    for field_name in EVENT_MEMORY_CHART_FIELDS:
        if field_name not in memory.chart_state:
            failures.append(f"chart field {field_name!r} was not remembered")
    missing_horizons = set(EVENT_MEMORY_RESPONSE_HORIZONS) - set(memory.response)
    if missing_horizons:
        failures.append(f"response horizons not remembered: {sorted(missing_horizons)}")
    for key in ("benchmark", "sector", "model", "baseline_volatility", "market_regime"):
        if key not in memory.context:
            failures.append(f"context field {key!r} was not remembered")
    if not memory.attribution:
        failures.append(
            "the attribution verdict did not travel with the memory — a "
            "confounded response is not the same thing as an associated one"
        )
    for horizon in memory.attribution:
        if horizon not in memory.response:
            failures.append(f"attribution for {horizon!r} has no matching response")

    # 2. An unmeasured event never enters memory.
    early = Event(
        entity="NVDA", published_time=frame.index[5].strftime("%Y-%m-%d %H:%M:%S"),
        event_type="earnings", source="news", evidence=[{"source_record_id": "r2"}],
    )
    try:
        build_memory(early, study_event(early, frame, benchmark), {}, _snapshot())
        failures.append(
            "an unmeasured event was remembered — it teaches nothing while "
            "counting toward every analog total"
        )
    except EventMemoryError:
        pass

    # 3. Incomplete memories are refused.
    if not memory_problems(_memory(chart_state={})):
        failures.append("a memory without chart state was accepted — that is a log")
    if not memory_problems(_memory(response={})):
        failures.append("a memory with no measured response was accepted")
    if not memory_problems(_memory(attribution={"5d": "event_associated"})):
        failures.append("an attribution without a matching response was accepted")
    for field_name in ("event_id", "ticker", "published_time", "event_type"):
        if not memory_problems(_memory(**{field_name: ""})):
            failures.append(f"a memory missing {field_name} was accepted")

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "event_memory.jsonl"

        try:
            remember(_memory(chart_state={}), store)
            failures.append("an incomplete memory was written to the store")
        except EventMemoryError:
            pass
        if store.exists():
            failures.append("a refused memory still created the store file")

        # 4. Deduplication.
        for _ in range(3):
            remember(_memory("dedupe"), store)
        if len(load_memories(store)) != 1:
            failures.append(
                "re-recording inflated the memory count — every historical base "
                "rate would silently become wrong"
            )

    # 5. Analogs match on chart state AND event type.
    pool = [
        _memory(f"e{index}", chart_state=_snapshot(rsi=55.0 + index * 0.2),
                response={"20d": {"abnormal_return": 0.01 * (index - 3)}})
        for index in range(EVENT_MEMORY_MIN_ANALOGS + 3)
    ]
    analogs = find_analogs(_snapshot(), "earnings", pool)
    if not analogs:
        failures.append("no analogs retrieved for an identical setup")
    scores = [entry["similarity"] for entry in analogs]
    if scores != sorted(scores, reverse=True):
        failures.append("analogs are not sorted by similarity")
    if any(score < EVENT_MEMORY_MIN_SIMILARITY for score in scores):
        failures.append("an analog below the similarity threshold was returned")

    wrong_type = [_memory(f"r{i}", event_type="regulation") for i in range(8)]
    if find_analogs(_snapshot(), "earnings", wrong_type):
        failures.append(
            "a different event type was returned as a comparable — similarity "
            "alone retrieves confident nonsense"
        )

    dissimilar = [_memory("far", chart_state=_snapshot(
        rsi=5.0, change_20d=-0.5, change_60d=-0.6, market_regime="stress",
        price_vs_ma_50=0.6, price_vs_ma_200=0.5, trend_slope_60d=-0.9,
    ))]
    if find_analogs(_snapshot(), "earnings", dissimilar):
        failures.append("a dissimilar chart was returned as an analog")

    # 5b. SHAPE, not price level. MEASURED: with `close` and `atr_14` inside
    # the similarity, two IDENTICAL chart shapes at $180 and $420 scored 0.846,
    # and 84.5% of every pair clearing the 0.70 bar was the SAME TICKER —
    # adjacent sessions of one stock, which is one situation counted many times
    # rather than an analog set. Retrieval had become a price filter.
    for level_field in ("close", "atr_14"):
        if level_field in EVENT_MEMORY_SIMILARITY_FIELDS:
            failures.append(
                f"{level_field!r} is back in the similarity fields — analog "
                f"retrieval becomes a price filter that matches a $180 stock "
                f"only to other $180 stocks"
            )
    cheap = _snapshot(close=180.0, atr_14=3.5)
    pricey = _snapshot(close=420.0, atr_14=8.2)
    if chart_similarity(cheap, pricey) < 0.999:
        failures.append(
            f"two IDENTICAL chart shapes at different price levels scored "
            f"{chart_similarity(cheap, pricey):.3f} — the same setup on a "
            f"pricier stock is the canonical analog, and it must not be "
            f"penalised for the share price"
        )
    # The same guard must not have made everything similar to everything.
    opposed = _snapshot(
        rsi=28.0, volatility=0.55, volume_ratio_20d=3.0, trend_slope_60d=-0.0012,
        trend_vs_20d_mean=-0.09, market_regime="bearish", change_5d=-0.07,
        change_20d=-0.18, change_60d=-0.31, price_vs_ma_50=-0.12,
        price_vs_ma_200=-0.26,
    )
    if chart_similarity(cheap, opposed) >= EVENT_MEMORY_MIN_SIMILARITY:
        failures.append(
            f"an OPPOSITE chart shape scored {chart_similarity(cheap, opposed):.3f}, "
            f"above the retrieval bar — dropping the level fields must sharpen "
            f"discrimination, not loosen it"
        )
    # And it must not have broken what the absolute scales were FOR: a purely
    # relative measure collapses near zero, calling two nearly-flat slopes
    # completely dissimilar.
    # The volatility scale must match the UNIT the chart state carries.
    # MEASURED: the state holds ANNUALIZED vol (0.28 = 28%), but the scale was
    # 0.01 — set for daily vol — so any 1pp gap scored ZERO and the field
    # contributed a mean of 0.029 across 4,000 real pairs. Two ordinary stocks
    # at 20% and 25% vol were called completely dissimilar.

    # ...without going blind to a real volatility regime change. Asserted on
    # the FIELD, not the whole-chart mean: with ten other fields identical, a
    # single differing field cannot move the mean below the retrieval bar, so
    # a whole-chart assertion here would test the scale's neighbours instead
    # of the scale.
    vol_scale = EVENT_MEMORY_FIELD_SCALE["volatility"]
    calm_vs_wild = max(0.0, 1.0 - abs(0.15 - 0.60) / vol_scale)
    if calm_vs_wild > 0.25:
        failures.append(
            f"a 15% and a 60% annualized volatility score {calm_vs_wild:.3f} on "
            f"the volatility field (scale {vol_scale}) — the scale has been "
            f"widened until the field no longer discriminates a genuine "
            f"volatility regime change"
        )
    ordinary_field = max(0.0, 1.0 - abs(0.20 - 0.25) / vol_scale)
    if ordinary_field < 0.5:
        failures.append(
            f"two ordinary volatilities (20% vs 25%) score {ordinary_field:.3f} "
            f"on the volatility field (scale {vol_scale}) — the scale is set "
            f"for DAILY vol while the chart state carries ANNUALIZED"
        )

    flat_a = _snapshot(trend_slope_60d=0.0012)
    flat_b = _snapshot(trend_slope_60d=-0.0009)
    if chart_similarity(flat_a, flat_b) < 0.9:
        failures.append(
            f"two nearly-flat slopes scored {chart_similarity(flat_a, flat_b):.3f} "
            f"— the declared absolute scales exist to stop a relative measure "
            f"collapsing near zero, and that protection is gone"
        )

    if chart_similarity(_snapshot(), _snapshot()) != 1.0:
        failures.append("an identical chart does not score 1.0 similarity")
    if chart_similarity({}, _snapshot()) != 0.0:
        failures.append("an empty chart state scored non-zero similarity")

    # 6. Too few analogs refuses to summarise, with no statistics.
    thin = analog_summary(analogs[: EVENT_MEMORY_MIN_ANALOGS - 1])
    if thin["status"] != "insufficient_analogs":
        failures.append(
            f"{EVENT_MEMORY_MIN_ANALOGS - 1} analogs produced a "
            f"{thin['status']!r} summary — a typical response from this few is "
            f"not typical of anything"
        )
    for forbidden in ("median_response", "mean_response", "positive_share"):
        if forbidden in thin:
            failures.append(f"an insufficient summary reported {forbidden}")

    full = analog_summary(analogs)
    if full["status"] != "measured":
        failures.append(
            f"{len(analogs)} analogs produced {full['status']!r} — the summary "
            f"threshold is unreachable, which is indistinguishable from broken"
        )
    if "not a forecast" not in full.get("detail", ""):
        failures.append("an analog summary does not disclaim being a forecast")
    if "event_associated_share" not in full:
        failures.append(
            "the summary does not report how many analogs were event-associated — "
            "a confounded history is weaker evidence"
        )

    if failures:
        print("E6 event-memory gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E6 event-memory gate OK:")
    print(
        f"  event, context, {len(EVENT_MEMORY_CHART_FIELDS)} chart fields and "
        f"{len(EVENT_MEMORY_RESPONSE_HORIZONS)} horizon responses all recorded."
    )
    print("  unmeasured events and incomplete memories refused; re-recording deduplicates.")
    print("  analogs match on chart state AND event type; dissimilar setups excluded.")
    print(
        f"  a summary needs >={EVENT_MEMORY_MIN_ANALOGS} analogs and disclaims "
        f"being a forecast."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
