"""Chart reaction memory (Sprint C7) — before, event, and what followed.

`build_reaction_memory(...)` retains, for one significant event:

    BEFORE  the C4 structural picture of the chart going in
    EVENT   identity, time, direction
    AFTER   the 1h / intraday / 1d / 5d / 20d / 60d reaction

and `find_reaction_analogs(...)` retrieves past events whose BEFORE-picture
resembles a current setup, so a base rate can be read off history rather than
guessed.

**Composition, not duplication.** E4 (`core.event_study`) already measures the
intraday/1d/5d/20d/60d reaction against a pre-event baseline, and E6
(`core.event_memory`) already stores events and retrieves analogs on a flat
numeric snapshot. C7 consumes E4's measurements rather than recomputing them —
a second implementation of abnormal return would be the split-brain the master
context forbids (W5). What C7 adds is the 1h reaction, and a STRUCTURAL
before-picture from C4 instead of a bag of numbers.

**The 1h honesty rule.** Providers serve roughly one month of hourly bars, while
a 60d reaction needs sixty sessions AFTER the event. So for any event old enough
to have a 60d reaction, hourly data does not exist. `1h` is therefore optional
and reads UNAVAILABLE when absent; it is never interpolated from daily bars,
which would invent a reaction nobody observed. A memory missing only intraday
horizons is still OK — missing a DAILY horizon is INCOMPLETE, because every
historical event can supply those.

**Retrieval matches structure first.** Two charts with the same numbers but
different phases — one breaking out, one failing a breakout — are not analogs,
and averaging their outcomes produces a base rate for a situation that never
existed. Similarity is `PHASE_WEIGHT * phase agreement + NUMERIC_WEIGHT *
numeric closeness`, and the phase term is the larger single lever.

**A thin analog set is reported, not smoothed.** Below `REACTION_MIN_ANALOGS`
the summary carries the matches but refuses a median response, because three
observations are an anecdote rather than a base rate.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from core.chart_structure import describe_structure
from core.config import (
    REACTION_HORIZONS,
    REACTION_HOURLY_MAX_ENTRY_GAP_HOURS,
    REACTION_INTRADAY_HORIZONS,
    REACTION_MEMORY_SCHEMA_VERSION,
    REACTION_MEMORY_VERSION,
    REACTION_MIN_ANALOGS,
    REACTION_MIN_SIMILARITY,
    REACTION_NUMERIC_MATCH_WEIGHT,
    REACTION_PHASE_MATCH_WEIGHT,
    REACTION_PRE_EVENT_SESSIONS,
    REACTION_REQUIRED_HORIZONS,
    REACTION_STATUS_INCOMPLETE,
    REACTION_STATUS_OK,
    REACTION_STATUS_UNAVAILABLE,
)

LOGGER = logging.getLogger("core.reaction_memory")

REACTION_MEMORY_SOURCE_ID = "yahoo_finance_chart"

# Numeric fields compared when matching a before-picture. Deliberately small:
# these are the ones C4 exposes alongside the phase, so a match is explainable.
_NUMERIC_FIELDS = ("consolidation_range", "volatility_regime_ratio", "gap_pct")


class ReactionMemoryError(ValueError):
    """Raised when a reaction-memory request is structurally invalid."""


def hourly_reaction(
    hourly_frame: pd.DataFrame | None,
    event_time,
    hours: int = 1,
) -> dict:
    """The 1h reaction, or an explicit UNAVAILABLE.

    Returns `{status, stock_return, entry_bar, exit_bar, reason}`. Absent hourly
    bars are a status, never a value interpolated from daily data: a 1h reaction
    that nobody observed is worse than no 1h reaction.
    """
    unavailable = {
        "status": REACTION_STATUS_UNAVAILABLE,
        "stock_return": None,
        "entry_bar": None,
        "exit_bar": None,
    }
    if hourly_frame is None or getattr(hourly_frame, "empty", True):
        return {
            **unavailable,
            "reason": (
                "no hourly bars supplied; providers serve about a month of them, "
                "so an event old enough for a 60d reaction has none"
            ),
        }
    if "Close" not in hourly_frame.columns:
        return {**unavailable, "reason": "hourly payload has no Close column"}

    frame = hourly_frame.dropna(subset=["Close"]).sort_index()
    target = pd.Timestamp(event_time)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    index = pd.to_datetime(frame.index)
    at_or_after = index >= target
    if not at_or_after.any():
        return {
            **unavailable,
            "reason": f"no hourly bar at or after {target} — the event predates the window",
        }

    entry = int(at_or_after.argmax())
    # An event predating the hourly window matches EVERY bar, so argmax returns
    # bar 0 and an unrelated hour months later gets reported as the reaction.
    # The entry bar must actually be near the event.
    gap_hours = (pd.Timestamp(frame.index[entry]) - target).total_seconds() / 3600.0
    if gap_hours > REACTION_HOURLY_MAX_ENTRY_GAP_HOURS:
        return {
            **unavailable,
            "reason": (
                f"nearest hourly bar is {gap_hours:.1f}h after the event, beyond the "
                f"{REACTION_HOURLY_MAX_ENTRY_GAP_HOURS}h bound — the event predates "
                f"the hourly window"
            ),
        }

    exit_position = entry + hours
    if exit_position >= len(frame):
        return {
            **unavailable,
            "reason": f"hourly history ends {len(frame) - 1 - entry} bar(s) after the event",
        }

    entry_close = float(frame["Close"].iloc[entry])
    exit_close = float(frame["Close"].iloc[exit_position])
    if entry_close <= 0:
        return {**unavailable, "reason": "entry close is not positive"}

    return {
        "status": REACTION_STATUS_OK,
        "stock_return": round(exit_close / entry_close - 1.0, 6),
        "entry_bar": str(frame.index[entry]),
        "exit_bar": str(frame.index[exit_position]),
        "reason": "",
    }


def before_picture(frame: pd.DataFrame, event_time) -> dict:
    """The C4 structure of the chart going INTO the event.

    Only bars strictly before the event are used. Including the event bar would
    let the reaction describe its own setup, which is how a memory quietly
    learns to predict the past.
    """
    if frame is None or "Close" not in getattr(frame, "columns", []):
        raise ReactionMemoryError("a price frame with Close is required")
    target = pd.Timestamp(event_time)
    if target.tzinfo is not None:
        target = target.tz_localize(None)

    index = pd.to_datetime(frame.index)
    prior = frame.loc[index < target]
    if prior.empty:
        return {
            "status": REACTION_STATUS_UNAVAILABLE,
            "phase": None,
            "swing_structure": None,
            "bars": 0,
            "reason": f"no bars before {target}",
        }

    window = prior.iloc[-REACTION_PRE_EVENT_SESSIONS:]
    structure = describe_structure(window)
    return {
        "status": REACTION_STATUS_OK,
        "phase": structure["phase"],
        "swing_structure": structure["swing_structure"],
        "consolidating": structure["consolidating"],
        "consolidation_range": structure["consolidation_range"],
        "volatility_regime": structure["volatility_regime"],
        "volatility_regime_ratio": structure["volatility_regime_ratio"],
        "gap_pct": structure["gap_pct"],
        "reversal": structure["reversal"],
        "bars": int(len(window)),
        "last_bar": str(window.index[-1]),
        "reason": structure.get("reason", ""),
    }


def _reactions_from_study(study: dict | None) -> dict:
    """E4's measured reactions, consumed as-is."""
    if not study:
        return {}
    reactions = study.get("reactions") if isinstance(study, dict) else None
    return dict(reactions or {})


def memory_hash(payload: dict) -> str:
    """Deterministic identity for one retained memory."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_reaction_memory(
    ticker: str,
    event_time,
    frame: pd.DataFrame,
    event_study: dict | None = None,
    hourly_frame: pd.DataFrame | None = None,
    event_id: str | None = None,
    event_type: str | None = None,
    direction: str | None = None,
) -> dict:
    """Retain one event's before-picture and measured reaction.

    `frame` must already be PIT-filtered by the caller; `event_study` is E4's
    result, consumed rather than recomputed.
    """
    if not str(ticker or "").strip():
        raise ReactionMemoryError("a ticker is required")

    target = pd.Timestamp(event_time)
    before = before_picture(frame, target)
    reactions = _reactions_from_study(event_study)
    hourly = hourly_reaction(hourly_frame, target)

    measured: dict[str, dict] = {}
    for horizon in REACTION_HORIZONS:
        if horizon == "1h":
            measured[horizon] = hourly
            continue
        entry = reactions.get(horizon)
        if entry:
            measured[horizon] = {"status": REACTION_STATUS_OK, **entry}
        else:
            measured[horizon] = {
                "status": REACTION_STATUS_UNAVAILABLE,
                "stock_return": None,
                "reason": f"event study reported no {horizon} reaction",
            }

    missing_required = [
        horizon for horizon in REACTION_REQUIRED_HORIZONS
        if measured[horizon]["status"] != REACTION_STATUS_OK
    ]
    missing_intraday = [
        horizon for horizon in REACTION_INTRADAY_HORIZONS
        if measured[horizon]["status"] != REACTION_STATUS_OK
    ]

    if before["status"] != REACTION_STATUS_OK:
        status = REACTION_STATUS_UNAVAILABLE
        reason = f"no before-picture: {before.get('reason', '')}"
    elif missing_required:
        status = REACTION_STATUS_INCOMPLETE
        reason = f"required horizons unmeasured: {', '.join(missing_required)}"
    else:
        status = REACTION_STATUS_OK
        # An absent intraday horizon is expected on an older event, so it is
        # noted without degrading the memory.
        reason = (
            f"intraday horizons unavailable: {', '.join(missing_intraday)}"
            if missing_intraday else ""
        )

    identity = {
        "ticker": str(ticker).upper(),
        "event_time": target.strftime("%Y-%m-%d %H:%M:%S"),
        "event_id": event_id,
        "before": before,
        "reactions": measured,
    }

    return {
        **identity,
        "event_type": event_type,
        "direction": direction,
        "status": status,
        "missing_required_horizons": missing_required,
        "missing_intraday_horizons": missing_intraday,
        "memory_hash": memory_hash(identity),
        "schema_version": REACTION_MEMORY_SCHEMA_VERSION,
        "calculation_version": REACTION_MEMORY_VERSION,
        "source_id": REACTION_MEMORY_SOURCE_ID,
        "reason": reason,
    }


def before_similarity(left: dict, right: dict) -> float:
    """How alike two before-pictures are, in [0, 1].

    Structure first. Two charts with identical numbers but different phases —
    one breaking out, one failing a breakout — are not analogs, and averaging
    their outcomes yields a base rate for a situation that never occurred.
    """
    if not left or not right:
        return 0.0

    left_phase, right_phase = left.get("phase"), right.get("phase")
    phase_score = 1.0 if (left_phase and left_phase == right_phase) else 0.0

    scores: list[float] = []
    for field in _NUMERIC_FIELDS:
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            continue
        try:
            a_value, b_value = float(a), float(b)
        except (TypeError, ValueError):
            continue
        scale = max(abs(a_value), abs(b_value), 0.01)
        scores.append(max(0.0, 1.0 - abs(a_value - b_value) / scale))
    numeric_score = sum(scores) / len(scores) if scores else 0.0

    total = (
        REACTION_PHASE_MATCH_WEIGHT * phase_score
        + REACTION_NUMERIC_MATCH_WEIGHT * numeric_score
    )
    return round(total, 6)


def find_reaction_analogs(
    setup: dict,
    memories: list[dict],
    minimum_similarity: float = REACTION_MIN_SIMILARITY,
) -> list[dict]:
    """Past events whose before-picture resembles the current setup.

    Only OK and INCOMPLETE memories are eligible: one with no before-picture
    cannot be matched against anything.
    """
    matches: list[dict] = []
    for memory in memories or []:
        before = memory.get("before") or {}
        if before.get("status") != REACTION_STATUS_OK:
            continue
        similarity = before_similarity(setup, before)
        if similarity >= minimum_similarity:
            matches.append({**memory, "similarity": similarity})
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)


def analog_response(analogs: list[dict], horizon: str) -> dict:
    """The historical response at one horizon across matched analogs.

    Below `REACTION_MIN_ANALOGS` this reports the count and refuses a median:
    two observations are an anecdote, and dressing one up as a base rate is how
    a memory starts lying confidently.
    """
    if horizon not in REACTION_HORIZONS:
        raise ReactionMemoryError(f"unknown reaction horizon {horizon!r}")

    returns: list[float] = []
    for analog in analogs or []:
        reaction = (analog.get("reactions") or {}).get(horizon) or {}
        if reaction.get("status") != REACTION_STATUS_OK:
            continue
        value = reaction.get("stock_return")
        if isinstance(value, (int, float)):
            returns.append(float(value))

    if len(returns) < REACTION_MIN_ANALOGS:
        return {
            "horizon": horizon,
            "observations": len(returns),
            "median_response": None,
            "sufficient": False,
            "reason": (
                f"{len(returns)} observation(s); {REACTION_MIN_ANALOGS} required — "
                f"fewer is an anecdote, not a base rate"
            ),
        }

    ordered = sorted(returns)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    positive = sum(1 for value in returns if value > 0)
    return {
        "horizon": horizon,
        "observations": len(returns),
        "median_response": round(median, 6),
        "share_positive": round(positive / len(returns), 6),
        "sufficient": True,
        "reason": "",
    }


def memory_problems(memory: dict) -> list[str]:
    """Validate a retained memory against the C7 contract."""
    problems: list[str] = []
    if not isinstance(memory, dict):
        return ["memory must be a dict"]
    for field in ("ticker", "event_time", "status", "schema_version", "memory_hash"):
        if not memory.get(field):
            problems.append(f"memory field {field!r} missing/empty")

    reactions = memory.get("reactions") or {}
    for horizon in REACTION_HORIZONS:
        if horizon not in reactions:
            problems.append(f"reaction horizon {horizon!r} absent — C7 requires every horizon to answer")
            continue
        entry = reactions[horizon]
        if entry.get("status") == REACTION_STATUS_OK and entry.get("stock_return") is None:
            problems.append(f"horizon {horizon!r} is OK but carries no return")
        if entry.get("status") != REACTION_STATUS_OK and entry.get("stock_return") is not None:
            problems.append(f"horizon {horizon!r} is {entry.get('status')} but still reports a return")

    if memory.get("status") == REACTION_STATUS_OK:
        for horizon in REACTION_REQUIRED_HORIZONS:
            if (reactions.get(horizon) or {}).get("status") != REACTION_STATUS_OK:
                problems.append(f"memory is OK while required horizon {horizon!r} is unmeasured")
        if (memory.get("before") or {}).get("status") != REACTION_STATUS_OK:
            problems.append("memory is OK with no before-picture")
    return problems
