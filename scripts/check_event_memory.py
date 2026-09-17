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
    EVENT_MEMORY_MIN_SIMILARITY,
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
