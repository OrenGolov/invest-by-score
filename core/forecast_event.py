"""Event-conditioned forecast (Sprint F5) — what followed setups like this one.

The pipeline the roadmap names, in order:

    new event -> event representation -> historical event matches
              -> current chart -> market regime -> forecast

`build_event_forecast(...)` runs all six stages and returns a PIPELINE, not a
number. Every stage carries its own status and reason, because "no forecast"
tells a reader nothing while "retrieval found 1 analog and needs 5" tells them
exactly what is missing and what would fix it.

**The first failure stops the pipeline.** Later stages report BLOCKED rather
than being attempted, so a reader always sees the most upstream obstacle
instead of a downstream symptom of it. A chart that could not be described
because the event was never valid is not a charting problem.

**Retrieval is the binding constraint, and it is reported, never absorbed.**
MEASURED on a fresh clone: `data/event_memory.jsonl` does not exist and holds
zero memories, so stages 3–6 are unreachable. That is the honest terminal
state of a system that has not yet observed anything, not a defect to paper
over. Even with a populated store of 1,608 real chart states, and after the E6
similarity fix, **39.3% of events retrieve ZERO analogs** and only 24.7% reach
E6's floor of five. Yield travels with every forecast.

**Composition, not a third retrieval (W5).** Two analog systems already exist —
E6 (`find_analogs`: chart numbers **and event type**, floor 5) and C7
(`find_reaction_analogs`: C4 structural phase, floor 3). F5 builds neither. It
calls E6, because E6 is the one that filters by event type, which is what
"event-conditioned" means. A third similarity metric would be the split-brain
the master context forbids.

**The claim strength is F4's, not a second answer.** An analog set is exactly a
conditional slice: N observations of what followed a comparable setup. F4
already decides — measured — what a slice of size N supports, so F5 calls
`core.forecast_conditional.select_claim` rather than inventing a second
sample-size policy that would drift from the first. A set of 3 analogs gets the
same treatment as a 3-observation regime slice, for the same reason.

**Regime is already inside retrieval.** `market_regime` is one of
`EVENT_MEMORY_SIMILARITY_FIELDS`, so the roadmap's "market regime" stage sits
partly upstream of the match. F5 therefore reports the regime AGREEMENT of the
retrieved set rather than filtering on it again — filtering twice would weight
the same evidence twice and shrink an already thin set for no new information.

**Same-ticker concentration is a first-class caveat.** Analogs drawn from one
ticker are not independent observations. MEASURED before the E6 fix, 84.5% of
every pair clearing the retrieval bar was the same ticker — adjacent sessions
of one stock, which is one situation counted many times. Above
`EVENT_FORECAST_MAX_SAME_TICKER_SHARE` the forecast is DEGRADED and says so.

**Attribution is honoured.** E5 already separates an event-ASSOCIATED response
from a confounded one. A base rate built mostly from confounded analogs
describes the market, not the event, so it is reported as DEGRADED.

**PIT by delegation.** This module never fetches. The event, the chart state,
the regime and the memory pool are supplied by the caller, each already
filtered to `as_of`. Retrieving an analog whose response window had not closed
by `as_of` would be the leak C3/C5 was fixed for.
"""

from __future__ import annotations

import logging

from core.config import (
    CONDITIONAL_CLAIM_PRECEDENCE,
    EVENT_FORECAST_CONTRACT_VERSION,
    EVENT_FORECAST_EFFECTIVE_PER_TICKER,
    EVENT_FORECAST_HORIZONS,
    EVENT_FORECAST_MAX_SAME_TICKER_SHARE,
    EVENT_FORECAST_MIN_ASSOCIATED_SHARE,
    EVENT_FORECAST_MIN_OBSERVED_SHARE,
    EVENT_FORECAST_OK,
    EVENT_FORECAST_REFUSED,
    EVENT_FORECAST_RETRIEVAL,
    EVENT_FORECAST_STAGE_CHART,
    EVENT_FORECAST_STAGE_EVENT,
    EVENT_FORECAST_STAGE_FORECAST,
    EVENT_FORECAST_STAGE_MATCHES,
    EVENT_FORECAST_STAGE_REGIME,
    EVENT_FORECAST_STAGE_REPRESENTATION,
    EVENT_FORECAST_STAGES,
    EVENT_FORECAST_VERSION,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    EVENT_STAGE_BLOCKED,
    EVENT_STAGE_DEGRADED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_OK,
    REGIME_LABELS,
)
from core.event_memory import find_analogs
from core.forecast_conditional import (
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_POINT,
    select_claim,
    wilson_interval,
)

LOGGER = logging.getLogger("core.forecast_event")


class EventForecastError(ValueError):
    """Raised when an event-forecast request violates the F5 contract."""


def _stage(name: str, status: str, reason: str = "", **detail) -> dict:
    """One pipeline stage. Every stage carries a reason when it is not OK."""
    return {"stage": name, "status": status, "reason": reason, **detail}


# ---------------------------------------------------------------------------
# Stage 2 — event representation
# ---------------------------------------------------------------------------


def represent_event(event) -> dict:
    """The comparable form of an event: what retrieval will match on.

    Deliberately thin. E1 already defines what an event IS; this extracts only
    the fields that make two events comparable, so the representation cannot
    drift into a second event schema.
    """
    if event is None:
        raise EventForecastError("an event is required to represent")

    def read(name, default=""):
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)

    event_type = str(read("event_type", "") or "")
    return {
        "event_id": str(read("event_id", "") or ""),
        "entity": str(read("entity", "") or "").upper(),
        "event_type": event_type,
        "direction": str(read("direction", "neutral") or "neutral"),
        "actor_type": str(read("actor_type", "") or ""),
        "published_time": str(read("published_time", "") or ""),
        "effective_time": str(read("effective_time", "") or ""),
    }


def representation_problems(representation: dict) -> list[str]:
    """What stops this representation from being matched against history."""
    problems: list[str] = []
    if not representation.get("event_type"):
        problems.append(
            "the event has no type, and retrieval matches on type — an "
            "earnings surprise is not a comparable for a regulatory action"
        )
    if not representation.get("entity"):
        problems.append("the event names no entity")
    if not (representation.get("effective_time") or representation.get("published_time")):
        problems.append(
            "the event has no time, so its analogs could not be restricted to "
            "what was already known"
        )
    return problems


# ---------------------------------------------------------------------------
# Stage 3 — historical matches (composed from E6)
# ---------------------------------------------------------------------------


def same_ticker_share(analogs, entity: str) -> float:
    """Share of the analog set drawn from the SAME ticker as the subject.

    Analogs from one ticker are not independent observations: they are
    adjacent sessions of one situation, counted many times.
    """
    if not analogs:
        return 0.0
    same = sum(
        1 for entry in analogs
        if str(getattr(entry["memory"], "ticker", "")).upper() == str(entity).upper()
    )
    return round(same / len(analogs), 6)


def associated_share(analogs, horizon: str) -> float:
    """Share of analogs whose response E5 judged event-ASSOCIATED.

    A base rate built mostly from confounded analogs describes the market
    rather than the event.
    """
    if not analogs:
        return 0.0
    associated = sum(
        1 for entry in analogs
        if getattr(entry["memory"], "is_event_associated", lambda _h: False)(horizon)
    )
    return round(associated / len(analogs), 6)


def observed_share(analogs) -> float:
    """Share of the analog set whose events were OBSERVED, not inferred.

    An inferred analog carries a real price move; what was never sourced is
    WHICH event produced it, or whether one did at all. A base rate built
    mostly from inferred events is therefore weaker evidence, and MEASURED,
    an inferred date cannot be validated against anything.
    """
    if not analogs:
        return 0.0
    observed = sum(
        1 for entry in analogs
        if not getattr(entry["memory"], "is_inferred", lambda: False)()
    )
    return round(observed / len(analogs), 6)


def regime_agreement(analogs, regime: str | None) -> float | None:
    """Share of analogs that occurred in the SAME regime as now.

    REPORTED, not filtered on. `market_regime` is already a similarity field,
    so filtering again would weight the same evidence twice and shrink an
    already thin set for no new information.
    """
    if not analogs or not regime:
        return None
    matched = sum(
        1 for entry in analogs
        if (getattr(entry["memory"], "chart_state", {}) or {}).get("market_regime")
        == regime
    )
    return round(matched / len(analogs), 6)


def _outcomes(analogs, horizon: str) -> tuple[int, int, list[float]]:
    """(successes, trials, values) of the analog responses at one horizon.

    An analog with no recorded response at this horizon leaves the DENOMINATOR
    as well as the numerator — counting it as a miss would bias every base rate
    toward zero in exactly the thin sets that can least afford it.
    """
    successes = 0
    values: list[float] = []
    for entry in analogs or ():
        memory = entry["memory"]
        value = memory.response_at(horizon)
        if value is None:
            continue
        values.append(float(value))
        if float(value) > 0.0:
            successes += 1
    return successes, len(values), values


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def build_event_forecast(
    event,
    as_of,
    *,
    chart_state: dict | None = None,
    regime: str | None = None,
    memories=None,
    horizon: str = "20d",
    min_similarity: float = EVENT_MEMORY_MIN_SIMILARITY,
    base_rate: float | None = None,
) -> dict:
    """Run the six-stage event-conditioned forecast.

    Returns the whole pipeline. The first failing stage stops it and every
    later stage reads BLOCKED, so the reported obstacle is always the most
    upstream one.
    """
    if horizon not in EVENT_FORECAST_HORIZONS:
        raise EventForecastError(
            f"{horizon!r} is not an event-forecast horizon "
            f"(known: {list(EVENT_FORECAST_HORIZONS)})"
        )

    stages: dict[str, dict] = {}
    failed_at: str | None = None

    def block_remaining(from_stage: str) -> None:
        start = EVENT_FORECAST_STAGES.index(from_stage)
        for name in EVENT_FORECAST_STAGES[start:]:
            if name not in stages:
                stages[name] = _stage(
                    name,
                    EVENT_STAGE_BLOCKED,
                    f"not attempted: the {failed_at!r} stage did not complete",
                )

    # -- 1. the event ------------------------------------------------------
    if event is None:
        stages[EVENT_FORECAST_STAGE_EVENT] = _stage(
            EVENT_FORECAST_STAGE_EVENT, EVENT_STAGE_FAILED, "no event was supplied"
        )
        failed_at = EVENT_FORECAST_STAGE_EVENT
        block_remaining(EVENT_FORECAST_STAGE_REPRESENTATION)
        return _assemble(None, as_of, horizon, stages, failed_at, None, [], base_rate)

    stages[EVENT_FORECAST_STAGE_EVENT] = _stage(
        EVENT_FORECAST_STAGE_EVENT, EVENT_STAGE_OK
    )

    # -- 2. representation -------------------------------------------------
    representation = represent_event(event)
    problems = representation_problems(representation)
    if problems:
        stages[EVENT_FORECAST_STAGE_REPRESENTATION] = _stage(
            EVENT_FORECAST_STAGE_REPRESENTATION,
            EVENT_STAGE_FAILED,
            "; ".join(problems),
            representation=representation,
        )
        failed_at = EVENT_FORECAST_STAGE_REPRESENTATION
        block_remaining(EVENT_FORECAST_STAGE_MATCHES)
        return _assemble(
            representation, as_of, horizon, stages, failed_at, None, [], base_rate
        )

    stages[EVENT_FORECAST_STAGE_REPRESENTATION] = _stage(
        EVENT_FORECAST_STAGE_REPRESENTATION,
        EVENT_STAGE_OK,
        representation=representation,
    )

    # -- 3. historical matches (E6, not a new metric) ----------------------
    if chart_state is None:
        # Retrieval needs a chart state to compare against. This is a
        # RETRIEVAL failure, not a charting one: the chart stage is about
        # describing the CURRENT setup for the reader.
        stages[EVENT_FORECAST_STAGE_MATCHES] = _stage(
            EVENT_FORECAST_STAGE_MATCHES,
            EVENT_STAGE_FAILED,
            "no chart state was supplied, so there is nothing to match against",
            analogs=0,
            retrieval=EVENT_FORECAST_RETRIEVAL,
        )
        failed_at = EVENT_FORECAST_STAGE_MATCHES
        block_remaining(EVENT_FORECAST_STAGE_CHART)
        return _assemble(
            representation, as_of, horizon, stages, failed_at, None, [], base_rate
        )

    analogs = find_analogs(
        chart_state,
        representation["event_type"],
        memories if memories is not None else [],
        min_similarity=min_similarity,
        exclude_event_id=representation["event_id"] or None,
    )
    successes, trials, _values = _outcomes(analogs, horizon)
    same_share = same_ticker_share(analogs, representation["entity"])
    assoc_share = associated_share(analogs, horizon)
    obs_share = observed_share(analogs)

    match_caveats: list[str] = []
    if analogs and same_share > EVENT_FORECAST_MAX_SAME_TICKER_SHARE:
        match_caveats.append(
            f"{same_share:.0%} of the analogs are the same ticker — adjacent "
            f"sessions of one situation are not independent observations"
        )
    if analogs and assoc_share < EVENT_FORECAST_MIN_ASSOCIATED_SHARE:
        match_caveats.append(
            f"only {assoc_share:.0%} of the analog responses were judged "
            f"event-associated; the rest are confounded, and a base rate built "
            f"from them describes the market rather than the event"
        )
    if analogs and obs_share < EVENT_FORECAST_MIN_OBSERVED_SHARE:
        match_caveats.append(
            f"only {obs_share:.0%} of the analogs were OBSERVED events; the "
            f"rest were dated by inference from price behaviour, so which "
            f"event produced the move — or whether one did — was never sourced"
        )

    match_stage = _stage(
        EVENT_FORECAST_STAGE_MATCHES,
        EVENT_STAGE_DEGRADED if match_caveats else EVENT_STAGE_OK,
        "; ".join(match_caveats),
        analogs=len(analogs),
        scorable_analogs=trials,
        retrieval=EVENT_FORECAST_RETRIEVAL,
        min_similarity=min_similarity,
        same_ticker_share=same_share,
        associated_share=assoc_share,
        observed_share=obs_share,
        pool_size=len(memories) if memories is not None else 0,
    )

    if not analogs:
        match_stage["status"] = EVENT_STAGE_FAILED
        pool = len(memories) if memories is not None else 0
        match_stage["reason"] = (
            f"no comparable setup was retrieved from {pool} memories at "
            f"similarity >= {min_similarity} for event type "
            f"{representation['event_type']!r}"
            + (
                " — the memory store is empty, so nothing has been observed yet"
                if pool == 0
                else ""
            )
        )
        stages[EVENT_FORECAST_STAGE_MATCHES] = match_stage
        failed_at = EVENT_FORECAST_STAGE_MATCHES
        block_remaining(EVENT_FORECAST_STAGE_CHART)
        return _assemble(
            representation, as_of, horizon, stages, failed_at, None, analogs, base_rate
        )

    stages[EVENT_FORECAST_STAGE_MATCHES] = match_stage

    # -- 4. the current chart ---------------------------------------------
    stages[EVENT_FORECAST_STAGE_CHART] = _stage(
        EVENT_FORECAST_STAGE_CHART,
        EVENT_STAGE_OK,
        "",
        fields=len(chart_state),
    )

    # -- 5. market regime --------------------------------------------------
    agreement = regime_agreement(analogs, regime)
    if regime is None:
        stages[EVENT_FORECAST_STAGE_REGIME] = _stage(
            EVENT_FORECAST_STAGE_REGIME,
            EVENT_STAGE_DEGRADED,
            "no regime was supplied, so the analogs cannot be checked for "
            "regime agreement; the forecast proceeds without that context",
            regime=None,
            agreement=None,
        )
    elif regime not in REGIME_LABELS:
        stages[EVENT_FORECAST_STAGE_REGIME] = _stage(
            EVENT_FORECAST_STAGE_REGIME,
            EVENT_STAGE_FAILED,
            f"{regime!r} is not a governed regime label "
            f"(known: {list(REGIME_LABELS)})",
            regime=regime,
            agreement=None,
        )
        failed_at = EVENT_FORECAST_STAGE_REGIME
        block_remaining(EVENT_FORECAST_STAGE_FORECAST)
        return _assemble(
            representation, as_of, horizon, stages, failed_at, None, analogs, base_rate
        )
    else:
        stages[EVENT_FORECAST_STAGE_REGIME] = _stage(
            EVENT_FORECAST_STAGE_REGIME,
            EVENT_STAGE_OK,
            "",
            regime=regime,
            agreement=agreement,
            # Reported, never filtered on: market_regime is already one of the
            # similarity fields, so filtering again double-weights it.
            filtered=False,
        )

    # -- 6. the forecast (F4 decides the claim strength) -------------------
    interval = wilson_interval(successes, trials) if trials else None
    claim, claim_reason = select_claim(trials, interval, base_rate)

    # EFFECTIVE sample size, not the raw count. Analogs drawn from one ticker
    # are adjacent sessions of ONE situation, resampled — 45 of them are not
    # 45 independent observations. Without this, a POINT claim gets published
    # from one stock's own history while the caveat sits two levels down in
    # the matches stage, exactly where a dashboard will not show it. This is
    # the F4 stress cell in another costume: the most confident-looking cell
    # resting on the least independent evidence.
    #
    # The discount is deliberately crude — distinct tickers, not a correlation
    # model — because a precise-looking adjustment here would imply a
    # precision the retrieval cannot support. It is a FLOOR on honesty, not an
    # estimate.
    distinct = len({
        str(getattr(entry["memory"], "ticker", "")).upper() for entry in analogs
    })
    effective = min(trials, max(distinct, 1) * EVENT_FORECAST_EFFECTIVE_PER_TICKER)
    if effective < trials:
        capped, capped_reason = select_claim(
            effective, wilson_interval(successes, trials) if trials else None,
            base_rate,
        )
        if CONDITIONAL_CLAIM_PRECEDENCE.index(capped) < CONDITIONAL_CLAIM_PRECEDENCE.index(claim):
            claim = capped
            claim_reason = (
                f"{trials} analogs span only {distinct} distinct ticker(s), so "
                f"the effective sample is {effective}: {capped_reason}"
            )

    forecast_stage = _stage(
        EVENT_FORECAST_STAGE_FORECAST,
        EVENT_STAGE_OK if claim != CONDITIONAL_CLAIM_INSUFFICIENT else EVENT_STAGE_FAILED,
        claim_reason,
        claim=claim,
        samples=trials,
        successes=successes,
        distinct_tickers=distinct,
        effective_samples=effective,
    )
    if claim == CONDITIONAL_CLAIM_INSUFFICIENT:
        failed_at = EVENT_FORECAST_STAGE_FORECAST
    stages[EVENT_FORECAST_STAGE_FORECAST] = forecast_stage

    # The interval is passed through UNCONDITIONALLY. `_assemble` is the single
    # place that decides whether uncertainty is published, so the shape rule
    # has exactly one enforcement point that a gate can actually attack. Two
    # guards in series look safer and make the real one untestable: the
    # upstream one silently masks the downstream one.
    return _assemble(
        representation, as_of, horizon, stages, failed_at, interval,
        analogs, base_rate, claim=claim, successes=successes, trials=trials,
    )


def _assemble(
    representation,
    as_of,
    horizon,
    stages,
    failed_at,
    interval,
    analogs,
    base_rate,
    claim: str = CONDITIONAL_CLAIM_INSUFFICIENT,
    successes: int = 0,
    trials: int = 0,
) -> dict:
    """Build the pipeline object, applying the shape rule.

    THE SHAPE RULE, inherited from F3/F4: a `value` key exists IF AND ONLY IF
    the claim is POINT. `value: None` would coalesce to 0.0 under the
    dashboard's `Number(x ?? 0)` idiom and render a withheld probability as
    certain-down.
    """
    for name in EVENT_FORECAST_STAGES:
        stages.setdefault(
            name, _stage(name, EVENT_STAGE_BLOCKED, "not attempted")
        )

    forecast = {
        "as_of": str(as_of),
        "horizon": horizon,
        "event": representation,
        "stages": {name: stages[name] for name in EVENT_FORECAST_STAGES},
        "stage_order": list(EVENT_FORECAST_STAGES),
        "status": EVENT_FORECAST_REFUSED if failed_at else EVENT_FORECAST_OK,
        "failed_stage": failed_at,
        "claim": claim if not failed_at else CONDITIONAL_CLAIM_INSUFFICIENT,
        "samples": trials,
        "analogs": len(analogs or ()),
        "base_rate": base_rate,
        "interval": None,
        "calculation_version": EVENT_FORECAST_VERSION,
        "contract_version": EVENT_FORECAST_CONTRACT_VERSION,
        "disclaimer": (
            "historical association under comparable conditions — not a "
            "forecast of cause, and not evidence that these events produced "
            "these moves"
        ),
    }

    if failed_at:
        return forecast

    forecast["interval"] = interval
    if claim == CONDITIONAL_CLAIM_POINT and trials:
        forecast["value"] = round(successes / trials, 6)
    return forecast


def event_forecast_problems(forecast: dict) -> list[str]:
    """Validate a built pipeline against the F5 contract."""
    problems: list[str] = []
    if not isinstance(forecast, dict):
        return ["forecast must be a dict"]

    for field in ("as_of", "horizon", "status", "calculation_version"):
        if not forecast.get(field):
            problems.append(f"forecast field {field!r} missing/empty")

    stages = forecast.get("stages") or {}
    if list(stages) != list(EVENT_FORECAST_STAGES):
        problems.append(
            "the pipeline does not report every declared stage in order — a "
            "reader could not tell which stage was skipped"
        )
    for name, stage in stages.items():
        if stage.get("status") != EVENT_STAGE_OK and not stage.get("reason"):
            problems.append(f"stage {name!r} is not OK and does not say why")

    # THE SHAPE RULE
    if forecast.get("claim") == CONDITIONAL_CLAIM_POINT:
        if "value" not in forecast:
            problems.append("a POINT claim carries no value")
    elif "value" in forecast:
        problems.append(
            f"a {forecast.get('claim')} claim carries a value key — it would "
            f"coalesce to 0 in a consumer and render as a real reading"
        )

    if forecast.get("status") == EVENT_FORECAST_REFUSED:
        if forecast.get("interval") is not None:
            problems.append(
                "a refused forecast supplies an interval — uncertainty beside a "
                "withheld estimate is an estimate by another name"
            )
        if not forecast.get("failed_stage"):
            problems.append("a refused forecast does not name the stage that failed")
        if "value" in forecast:
            problems.append("a refused forecast carries a value")

    # A forecast must never claim more than its retrieval supports.
    if forecast.get("status") == EVENT_FORECAST_OK:
        match_stage = stages.get(EVENT_FORECAST_STAGE_MATCHES) or {}
        if not match_stage.get("analogs"):
            problems.append(
                "an OK forecast rests on zero analogs — the claim is not "
                "event-conditioned at all"
            )
        if forecast.get("claim") == CONDITIONAL_CLAIM_POINT:
            if (forecast.get("samples") or 0) < EVENT_MEMORY_MIN_ANALOGS:
                problems.append(
                    f"a POINT claim rests on {forecast.get('samples')} analogs, "
                    f"below E6's floor of {EVENT_MEMORY_MIN_ANALOGS}"
                )

    # The first failure must stop the pipeline.
    failed = forecast.get("failed_stage")
    if failed:
        start = EVENT_FORECAST_STAGES.index(failed)
        for name in EVENT_FORECAST_STAGES[start + 1:]:
            if (stages.get(name) or {}).get("status") != EVENT_STAGE_BLOCKED:
                problems.append(
                    f"stage {name!r} ran after {failed!r} failed — a reader would "
                    f"see a downstream symptom instead of the real obstacle"
                )

    return problems


def render_pipeline(forecast: dict) -> list[dict]:
    """One row per stage, in declared order — the reading view.

    Never leaves a blank: a stage that is not OK renders its reason, so the
    pipeline always explains itself.
    """
    rows: list[dict] = []
    for name in forecast.get("stage_order") or ():
        stage = (forecast.get("stages") or {}).get(name) or {}
        rows.append(
            {
                "stage": name,
                "status": stage.get("status"),
                "reason": stage.get("reason") or "",
                "detail": {
                    key: value
                    for key, value in stage.items()
                    if key not in ("stage", "status", "reason")
                },
            }
        )
    return rows
