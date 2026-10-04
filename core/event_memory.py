"""Event memory (Sprint E6) — institutional memory of how events landed.

Stores, per event: the event itself, the market context, the chart state at
the moment it happened, the measured 1D/5D/20D/60D response, and the
attribution verdict. This is what the forecasting sprints retrieve from when
asking "what happened last time a setup looked like this?".

    E1 event + E4 study + E5 attribution
        |
        v
    remember()       <- refuses an incomplete memory
        |
        v
    data/event_memory.jsonl  (append-only)
        |
        v
    find_analogs()   <- similar chart state, same event type
        |
        v
    analog_summary() <- REFUSES to summarise too few

Design decisions worth stating:

- **A memory is written only when it is complete.** Event, study and
  attribution must all be present and measured. A half-recorded memory is
  worse than an absent one, because it looks like evidence — and this is the
  store the forecasting layer will retrieve from without re-deriving
  anything.
- **The chart state uses the SHARED snapshot fields.** A remembered chart
  state and a live one are described in identical terms, or the comparison
  is meaningless. Institutional memory that cannot be compared with the
  present is just a log.
- **Analogs match on chart state AND event type.** An earnings surprise into
  an overbought chart is not a comparable for a regulatory action into the
  same chart. Similarity alone would retrieve confident nonsense.
- **A summary needs enough analogs, or it refuses.** A "typical response"
  computed from two examples is not typical of anything, so
  `analog_summary` returns the matches with `status: "insufficient_analogs"`
  rather than a median nobody should act on.
- **The attribution verdict travels with every memory.** A remembered
  response that was `confounded` is a different thing from one that was
  `event_associated`, and forgetting which is how a memory store turns into
  a source of false confidence.
- **Memories are immutable and deduplicated by event id.** Re-recording the
  same event is a no-op, so a re-run cannot inflate the analog count.

Pure and deterministic apart from the append itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.config import (
    EVENT_MEMORY_CHART_FIELDS,
    EVENT_MEMORY_FIELD_SCALE,
    EVENT_MEMORY_INFERENCE_METHODS,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    EVENT_MEMORY_PROVENANCES,
    EVENT_MEMORY_REQUIRE_PROVENANCE,
    EVENT_MEMORY_RESPONSE_HORIZONS,
    EVENT_MEMORY_SIMILARITY_FIELDS,
    EVENT_MEMORY_VERSION,
    MEMORY_PROVENANCE_INFERRED,
    MEMORY_PROVENANCE_OBSERVED,
)

EVENT_MEMORY_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "event_memory.jsonl"
)


class EventMemoryError(ValueError):
    """Raised when a memory is incomplete or a retrieval is invalid."""


@dataclass
class EventMemory:
    """One remembered event and everything needed to learn from it."""

    event_id: str
    ticker: str
    published_time: str
    event_type: str
    direction: str
    actor: str = ""
    actor_type: str = ""
    # Chart state at the moment of the event, in shared snapshot terms.
    chart_state: dict[str, Any] = field(default_factory=dict)
    # Market context: regime, benchmark and sector identities.
    context: dict[str, Any] = field(default_factory=dict)
    # Measured response per horizon: stock, abnormal, volatility, volume.
    response: dict[str, dict[str, Any]] = field(default_factory=dict)
    # E5 verdict per horizon — a confounded response is not the same thing
    # as an event-associated one, and forgetting which breeds false
    # confidence.
    attribution: dict[str, str] = field(default_factory=dict)
    entity_resolution_method: str = ""
    # THE OUTLET that reported the event ("CNBC", "Biztoc.com"), carried
    # here so an outcome can be attributed to it. MEASURED 2026-09-21, 2,084
    # memories carried NO outlet, which is why L4's source-reliability
    # estimator had nothing to learn from: a source cannot acquire a track
    # record if the memory holding the outcome does not say who reported it.
    #
    # Empty for an INFERRED memory and that is correct, not missing — a
    # memory dated from volume cadence has no reporter by construction.
    # `SOURCE_RELIABILITY_ABSENT_IS_ZERO = False` is the matching rule
    # downstream: unmeasured and useless are different facts.
    source: str = ""
    # The pipeline that delivered it ("newsapi_news"). Kept distinct from
    # `source` so the provider is never mistaken for a track record.
    provider: str = ""
    # Was this OBSERVED from a real source, or INFERRED from price behaviour?
    # Defaulted to "" rather than "observed" deliberately: a default of
    # "observed" would silently launder every unlabelled memory into evidence.
    # `memory_problems` refuses a memory that does not state it.
    provenance: str = ""
    # For an inferred memory, the method that dated it — a key of
    # EVENT_MEMORY_INFERENCE_METHODS, so the reasoning is always retrievable.
    inference_method: str = ""
    memory_version: str = EVENT_MEMORY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def response_at(self, horizon: str) -> float | None:
        """The abnormal return at one horizon, or None if unrecorded."""
        return (self.response.get(horizon) or {}).get("abnormal_return")

    def is_event_associated(self, horizon: str = "20d") -> bool:
        return self.attribution.get(horizon) == "event_associated"

    def is_inferred(self) -> bool:
        """True when this memory's event was dated by inference, not observed.

        An inferred memory is real evidence about a real price move; what is
        uncertain is WHICH event produced it, and whether one did at all.
        """
        return self.provenance == MEMORY_PROVENANCE_INFERRED


def memory_problems(memory: EventMemory) -> list[str]:
    """Validate a memory before it is written.

    A memory that looks like evidence but is not is the worst thing this
    store can contain, so the bar for entry is high.
    """
    problems: list[str] = []
    if not str(memory.event_id or "").strip():
        problems.append("event_id is required — memories are deduplicated by it")
    if not str(memory.ticker or "").strip():
        problems.append("ticker is required")
    if not str(memory.published_time or "").strip():
        problems.append("published_time is required")
    if not str(memory.event_type or "").strip():
        problems.append("event_type is required — analogs match on it")
    if not memory.chart_state:
        problems.append(
            "chart_state is required — a memory that cannot be compared with "
            "the present is a log, not institutional memory"
        )
    if not memory.response:
        problems.append(
            "no measured response — an event with no recorded outcome teaches "
            "nothing and should not occupy memory"
        )
    for horizon, verdict in (memory.attribution or {}).items():
        if horizon not in memory.response:
            problems.append(
                f"attribution recorded for {horizon!r} but no response was measured"
            )

    # Provenance is required at the door. Once written, an unlabelled memory
    # is indistinguishable from an observed one, and F5 would count an
    # inferred base rate as though it had been sourced.
    provenance = str(memory.provenance or "")
    if EVENT_MEMORY_REQUIRE_PROVENANCE and not provenance:
        problems.append(
            "provenance is required — an unlabelled memory cannot be told "
            "apart from an observed one once it is in the store"
        )
    elif provenance and provenance not in EVENT_MEMORY_PROVENANCES:
        problems.append(
            f"unknown provenance {provenance!r} "
            f"(known: {list(EVENT_MEMORY_PROVENANCES)})"
        )
    if provenance == MEMORY_PROVENANCE_INFERRED:
        method = str(memory.inference_method or "")
        if not method:
            problems.append(
                "an inferred memory must name the method that dated it, or a "
                "reader cannot tell what produced it"
            )
        elif method not in EVENT_MEMORY_INFERENCE_METHODS:
            problems.append(
                f"unknown inference method {method!r} "
                f"(known: {sorted(EVENT_MEMORY_INFERENCE_METHODS)})"
            )
    elif provenance == MEMORY_PROVENANCE_OBSERVED and memory.inference_method:
        problems.append(
            "an OBSERVED memory names an inference method — it was either "
            "observed or inferred, and claiming both hides which"
        )
    return problems


def build_memory(
    event,
    study,
    attributions: dict | None = None,
    snapshot: dict | None = None,
    provenance: str = MEMORY_PROVENANCE_OBSERVED,
    inference_method: str = "",
) -> EventMemory:
    """Assemble a memory from an E1 event, an E4 study and E5 attributions.

    Refuses an unmeasured study: recording an event whose response was never
    measured would put a row in memory that teaches nothing while counting
    toward every analog total.

    `provenance` defaults to OBSERVED because this function's inputs are a
    real event and a real study — the caller that INFERRED its event must say
    so explicitly, which is the only way round the guard in `memory_problems`.
    """
    if not getattr(study, "is_measured", lambda: False)():
        raise EventMemoryError(
            f"study for {getattr(event, 'entity', '?')} at "
            f"{getattr(event, 'published_time', '?')} is not measured "
            f"(status {getattr(study, 'status', '?')!r}) — an unmeasured event "
            f"teaches nothing and must not occupy memory"
        )

    chart_state = {
        field_name: (snapshot or {}).get(field_name)
        for field_name in EVENT_MEMORY_CHART_FIELDS
        if (snapshot or {}).get(field_name) is not None
    }

    response: dict[str, dict[str, Any]] = {}
    for horizon in EVENT_MEMORY_RESPONSE_HORIZONS:
        reaction = (study.reactions or {}).get(horizon)
        if not reaction:
            continue
        response[horizon] = {
            "stock_return": reaction.get("stock_return"),
            "abnormal_return": reaction.get("abnormal_return"),
            "volatility_ratio": reaction.get("volatility_ratio"),
            "volume_ratio": reaction.get("volume_ratio"),
        }

    verdicts = {
        horizon: attribution.verdict
        for horizon, attribution in (attributions or {}).items()
        if horizon in response
    }

    return EventMemory(
        event_id=str(getattr(event, "event_id", "")),
        ticker=str(getattr(event, "entity", "")).upper(),
        published_time=str(getattr(event, "published_time", "")),
        event_type=str(getattr(event, "event_type", "other")),
        direction=str(getattr(event, "direction", "neutral")),
        actor=str(getattr(event, "actor", "") or ""),
        actor_type=str(getattr(event, "actor_type", "") or ""),
        chart_state=chart_state,
        provenance=provenance,
        inference_method=inference_method,
        source=str(getattr(event, "source", "") or ""),
        provider=str(getattr(event, "provider", "") or ""),
        context={
            "benchmark": getattr(study, "benchmark", None),
            "sector": getattr(study, "sector", None),
            "model": getattr(study, "model", None),
            "baseline_volatility": (getattr(study, "baseline", {}) or {}).get(
                "daily_volatility"
            ),
            "market_regime": (snapshot or {}).get("market_regime"),
        },
        response=response,
        attribution=verdicts,
        entity_resolution_method=str(
            (getattr(event, "entity_resolution", {}) or {}).get("method", "")
        ),
    )


def remember(
    memory: EventMemory,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a memory, deduplicated by event id.

    Re-recording the same event is a no-op rather than an error: a re-run of
    the pipeline must not inflate the analog count, which would quietly make
    every historical base rate wrong.
    """
    problems = memory_problems(memory)
    if problems:
        raise EventMemoryError(
            f"refusing to remember an incomplete memory for "
            f"{memory.ticker or '?'}: " + "; ".join(problems)
        )

    store = Path(path) if path is not None else EVENT_MEMORY_STORE_PATH
    for existing in load_memories(store):
        if existing.get("event_id") == memory.event_id:
            return existing

    record = memory.to_dict()
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def load_memories(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read every memory. Malformed lines raise (integrity is loud)."""
    store = Path(path) if path is not None else EVENT_MEMORY_STORE_PATH
    if not store.exists():
        return []
    records: list[dict[str, Any]] = []
    with store.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise EventMemoryError(
                    f"{store.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return records


def load_memory_objects(path: str | Path | None = None) -> list[EventMemory]:
    """Read memories as typed objects."""
    known = {name for name in EventMemory.__dataclass_fields__}
    return [
        EventMemory(**{key: value for key, value in record.items() if key in known})
        for record in load_memories(path)
    ]


def chart_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: Iterable[str] = EVENT_MEMORY_SIMILARITY_FIELDS,
) -> float:
    """Similarity between two chart SHAPES, in [0, 1].

    Compared field by field over `EVENT_MEMORY_SIMILARITY_FIELDS`, which is
    the recorded chart state MINUS its price-level fields. Fields with a
    declared scale in `EVENT_MEMORY_FIELD_SCALE` use it; the rest fall back to
    a relative comparison.

    **Levels are recorded but never matched on.** `close` and `atr_14` say how
    expensive the stock is, not what the chart is doing, and every other field
    is already scale-free. MEASURED: with the levels included, two IDENTICAL
    chart shapes at $180 and $420 scored 0.846, and 84.5% of all pairs clearing
    the 0.70 retrieval bar were the SAME TICKER — adjacent sessions of one
    stock, which is one situation counted many times rather than an analog set.
    Without them the same pair scores 1.000 while an OPPOSITE-shape pair falls
    from 0.230 to 0.090, so discrimination improved rather than loosened.

    An absolute scale, where one is declared, is what keeps a purely relative
    measure from collapsing near zero: two nearly-flat slopes (+0.0012 vs
    -0.0009) are both "flat", yet relative difference calls them 0.0 similar
    and drags an otherwise-0.97 match below the retrieval bar.

    Fields absent from either side are skipped rather than treated as
    matching, because a missing field is not agreement.
    """
    scores: list[float] = []
    for field_name in fields:
        a, b = left.get(field_name), right.get(field_name)
        if a is None or b is None:
            continue
        if isinstance(a, str) or isinstance(b, str):
            scores.append(1.0 if str(a) == str(b) else 0.0)
            continue
        try:
            a_value, b_value = float(a), float(b)
        except (TypeError, ValueError):
            continue
        # Absolute scale where one exists, relative otherwise. A purely
        # relative measure collapses near zero: two nearly-flat slopes are
        # both "flat", but relative difference calls them 0.0 similar and
        # drags an otherwise-0.97 match below the retrieval bar.
        scale = EVENT_MEMORY_FIELD_SCALE.get(field_name)
        if scale is None:
            scale = max(abs(a_value), abs(b_value), 1e-9)
        scores.append(max(0.0, 1.0 - abs(a_value - b_value) / scale))
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def find_analogs(
    chart_state: dict[str, Any],
    event_type: str,
    memories: list[EventMemory] | None = None,
    min_similarity: float = EVENT_MEMORY_MIN_SIMILARITY,
    exclude_event_id: str | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Historical events whose setup resembled this one.

    Matches on chart state AND event type. An earnings surprise into an
    overbought chart is not a comparable for a regulatory action into the
    same chart, so similarity alone would retrieve confident nonsense.
    """
    pool = memories if memories is not None else load_memory_objects(path)
    analogs: list[dict[str, Any]] = []
    for memory in pool:
        if exclude_event_id and memory.event_id == exclude_event_id:
            continue
        if memory.event_type != event_type:
            continue
        similarity = chart_similarity(chart_state, memory.chart_state)
        if similarity < min_similarity:
            continue
        analogs.append({"similarity": similarity, "memory": memory})
    return sorted(analogs, key=lambda entry: entry["similarity"], reverse=True)


def analog_summary(
    analogs: list[dict[str, Any]],
    horizon: str = "20d",
) -> dict[str, Any]:
    """Summarise what happened to comparable setups, or refuse to.

    A "typical response" from two examples is not typical of anything, so
    below `EVENT_MEMORY_MIN_ANALOGS` the matches are returned with an
    explicit `insufficient_analogs` status and no summary statistics.
    """
    import statistics

    usable = [
        entry for entry in analogs
        if entry["memory"].response_at(horizon) is not None
    ]

    if len(usable) < EVENT_MEMORY_MIN_ANALOGS:
        return {
            "status": "insufficient_analogs",
            "horizon": horizon,
            "analogs": len(usable),
            "required": EVENT_MEMORY_MIN_ANALOGS,
            "detail": (
                f"{len(usable)} comparable setup(s) with a measured {horizon} "
                f"response, below the {EVENT_MEMORY_MIN_ANALOGS} required — a "
                f"typical response from this few examples is not typical of "
                f"anything"
            ),
            "memory_version": EVENT_MEMORY_VERSION,
        }

    returns = [entry["memory"].response_at(horizon) for entry in usable]
    associated = sum(
        1 for entry in usable if entry["memory"].is_event_associated(horizon)
    )
    return {
        "status": "measured",
        "horizon": horizon,
        "analogs": len(usable),
        "median_response": round(statistics.median(returns), 6),
        "mean_response": round(statistics.fmean(returns), 6),
        "positive_share": round(sum(1 for r in returns if r > 0) / len(returns), 4),
        "dispersion": round(statistics.pstdev(returns), 6) if len(returns) > 1 else 0.0,
        "mean_similarity": round(
            statistics.fmean(entry["similarity"] for entry in usable), 4
        ),
        "event_associated_share": round(associated / len(usable), 4),
        "memory_version": EVENT_MEMORY_VERSION,
        "detail": (
            "historical association under comparable conditions — not a forecast, "
            "and not evidence that these events caused these moves"
        ),
    }


def memory_report(path: str | Path | None = None) -> dict[str, Any]:
    """Coverage summary over the whole memory store."""
    memories = load_memory_objects(path)
    by_type: dict[str, int] = {}
    by_ticker: dict[str, int] = {}
    associated = 0
    for memory in memories:
        by_type[memory.event_type] = by_type.get(memory.event_type, 0) + 1
        by_ticker[memory.ticker] = by_ticker.get(memory.ticker, 0) + 1
        if memory.is_event_associated():
            associated += 1
    return {
        "memory_version": EVENT_MEMORY_VERSION,
        "memories": len(memories),
        "distinct_tickers": len(by_ticker),
        "by_event_type": dict(sorted(by_type.items())),
        "event_associated": associated,
        "min_analogs": EVENT_MEMORY_MIN_ANALOGS,
        "min_similarity": EVENT_MEMORY_MIN_SIMILARITY,
    }
