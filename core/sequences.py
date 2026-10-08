"""Temporal sequence dataset (Sprint C5) — T-60 … T0, each step PIT-correct.

`build_sequence(ticker, as_of, ...)` assembles one sequence: 61 steps, each
carrying the state that was knowable AT THAT STEP, plus the market/macro/event
context as of T0, plus the V1 outcome labels.

**C5 is not a second door into a training set.** M2
(`core.training_dataset`) remains the only generator of supervised
`TrainingRow`s, and this module produces none. What it builds is a separate,
versioned, hashed SEQUENCE artifact for C6 sequence-model research. Labels come
from `core.labels.build_outcome_labels` — the same V1 builder M2 uses — so a
sequence and a training row can never disagree about an outcome.

**The step-truncation rule.** Step i is built from `frame.iloc[:i+1]` — the bars
up to and including that step, and nothing after. This is the property that
makes a sequence trainable: if step 5 could see bar 40, a model would learn from
information that did not exist, and the leak would be invisible in the output.
`sequence_problems` re-verifies monotonicity, and the gate injects a violation
to prove the check bites.

**Per-step vs as-of-T0, stated honestly.** Price, volume, the C2 feature surface
and the C4 structure are recomputed at every step, so they genuinely vary across
the window. Market, sector, macro, sentiment, event and fundamental state are
attached ONCE at T0 and labelled `context_scope: "as_of_t0"`. Back-projecting
today's macro reading across sixty past steps would be the revised-data-in-
history failure the master context forbids; carrying it unlabelled would be
worse still. A consumer can see exactly which channels vary and which do not.

**A ragged sequence is not a sequence.** Below `SEQUENCE_MIN_STEP_COVERAGE` the
artifact is INCOMPLETE and carries NO steps, because a model trained on a window
with holes learns the holes.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from core.chart_features import compute_chart_features
from core.chart_structure import describe_structure
from core.config import (
    SEQUENCE_CONTEXT_SCOPE,
    SEQUENCE_CONTRACT_VERSION,
    SEQUENCE_LOOKBACK_STEPS,
    SEQUENCE_MIN_STEP_COVERAGE,
    SEQUENCE_PIPELINE_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    SEQUENCE_STATUS_INCOMPLETE,
    SEQUENCE_STATUS_OK,
    SEQUENCE_STATUS_UNAVAILABLE,
    SEQUENCE_STEP_CHANNELS,
    SEQUENCE_STEP_WARMUP_BARS,
    SEQUENCE_T0_CHANNELS,
)

LOGGER = logging.getLogger("core.sequences")

SEQUENCE_SOURCE_ID = "yahoo_finance_chart"

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class SequenceError(ValueError):
    """Raised when a sequence request is structurally invalid."""


def _usable(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        raise SequenceError("a price frame is required")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SequenceError(f"price frame is missing {', '.join(missing)}")
    usable = frame.dropna(subset=["Close"])
    # Every window here reads backwards from the last row, so a descending or
    # shuffled frame is silently mislabelled rather than rejected: a reversed
    # downtrend reads as a breakout. Refusing is better than sorting behind the
    # caller's back, because an out-of-order frame means the caller has a bug.
    index = pd.to_datetime(usable.index)
    if not index.is_monotonic_increasing:
        raise SequenceError(
            "price frame is not in ascending time order — every window reads "
            "backwards from the last row, so an out-of-order frame produces "
            "confident, wrong labels"
        )
    return usable


def build_step(
    frame: pd.DataFrame,
    position: int,
    benchmark_frame: pd.DataFrame | None = None,
) -> dict:
    """The state knowable at one step.

    `frame.iloc[:position + 1]` is the whole point: a step sees its own bar and
    every bar before it, and nothing after. Slicing here rather than at the
    caller keeps the truncation in one place, where it can be tested.
    """
    if position < 0 or position >= len(frame):
        raise SequenceError(f"step position {position} is outside the frame")

    visible = frame.iloc[: position + 1]
    bar = visible.iloc[-1]
    features = compute_chart_features(visible, benchmark_frame=benchmark_frame)
    structure = describe_structure(visible)

    return {
        "step_time": str(visible.index[-1]),
        "visible_bars": int(len(visible)),
        "price": {
            "open": round(float(bar["Open"]), 6),
            "high": round(float(bar["High"]), 6),
            "low": round(float(bar["Low"]), 6),
            "close": round(float(bar["Close"]), 6),
        },
        "volume": float(bar["Volume"]),
        "technical": features["features"],
        "technical_gaps": features["insufficient_history"],
        "structure": {
            "phase": structure["phase"],
            "swing_structure": structure["swing_structure"],
            "consolidating": structure["consolidating"],
            "reversal": structure["reversal"],
            "volatility_regime": structure["volatility_regime"],
        },
    }


def _t0_context(
    ticker: str,
    as_of: str,
    market_context: dict | None,
    macro_snapshot: dict | None,
    news_snapshot: dict | None,
    sentiment_snapshot: dict | None,
    fundamentals: dict | None,
) -> dict:
    """Context attached once at T0, labelled so it is never mistaken for per-step.

    Each channel carries its own status. A channel that was not supplied reads
    UNAVAILABLE rather than being silently absent — an absent key and a failed
    provider look identical to a consumer otherwise.
    """
    def _channel(payload: dict | None, keys: tuple[str, ...] = ()) -> dict:
        if not payload:
            return {"status": SEQUENCE_STATUS_UNAVAILABLE, "value": None}
        status = str(payload.get("status", SEQUENCE_STATUS_OK))
        value = {key: payload.get(key) for key in keys} if keys else None
        return {"status": status, "value": value}

    return {
        "context_scope": SEQUENCE_CONTEXT_SCOPE,
        "as_of": str(as_of),
        "market": _channel(market_context, ("benchmark", "benchmark_symbol", "returns")),
        "sector": _channel(market_context, ("sector", "industry_benchmark")),
        "macro": _channel(macro_snapshot, ("regime", "regime_score", "series_values")),
        "sentiment": _channel(sentiment_snapshot, ("sentiment_score", "derivation")),
        "events": _channel(news_snapshot, ("event_count", "direction")),
        "fundamentals": _channel(fundamentals, ("valuation_metrics",)),
    }


def sequence_hash(steps: list[dict], ticker: str, as_of: str, labels: dict) -> str:
    """Deterministic identity for a sequence.

    Includes the step states, the ticker/as_of, and the label identity — two
    sequences differing in any of those must never share a hash.
    """
    payload = {
        "ticker": str(ticker).upper(),
        "as_of": str(as_of),
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "contract_version": SEQUENCE_CONTRACT_VERSION,
        "steps": steps,
        "label_version": labels.get("label_version"),
        "label_record_hash": labels.get("record_hash"),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_sequence(
    ticker: str,
    as_of,
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    market_context: dict | None = None,
    macro_snapshot: dict | None = None,
    news_snapshot: dict | None = None,
    sentiment_snapshot: dict | None = None,
    fundamentals: dict | None = None,
    labels: dict | None = None,
    lookback: int = SEQUENCE_LOOKBACK_STEPS,
) -> dict:
    """One T-lookback … T0 sequence for a ticker.

    `frame` must already be PIT-filtered to as_of by the caller (C1's
    `eligible_bars`) — this module does no as_of filtering of its own, so the
    system keeps exactly one notion of "now".

    Labels are passed in rather than fetched, so the caller supplies them from
    `core.labels.build_outcome_labels`; C5 never computes a forward return of
    its own, which is what keeps it from becoming a second label path.
    """
    if not str(ticker or "").strip():
        raise SequenceError("a ticker is required")
    if lookback < 2:
        raise SequenceError("a sequence must span at least two steps")

    usable = _usable(frame)
    labels = labels or {}
    required_bars = lookback + 1 + SEQUENCE_STEP_WARMUP_BARS

    base = {
        "ticker": str(ticker).upper(),
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_steps": lookback,
        "step_channels": list(SEQUENCE_STEP_CHANNELS),
        "t0_channels": list(SEQUENCE_T0_CHANNELS),
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "calculation_version": SEQUENCE_CONTRACT_VERSION,
        "pipeline_version": SEQUENCE_PIPELINE_VERSION,
        "source_id": SEQUENCE_SOURCE_ID,
    }

    if len(usable) < required_bars:
        return {
            **base,
            "status": SEQUENCE_STATUS_INCOMPLETE,
            "steps": [],
            "step_count": 0,
            "context": _t0_context(
                ticker, base["as_of"], market_context, macro_snapshot,
                news_snapshot, sentiment_snapshot, fundamentals,
            ),
            "labels": labels,
            "sequence_hash": None,
            "reason": (
                f"{len(usable)} bars is below the {required_bars} required "
                f"({lookback + 1} steps plus {SEQUENCE_STEP_WARMUP_BARS} warm-up bars, "
                f"so early steps are not thinner than late ones)"
            ),
        }

    first_position = len(usable) - (lookback + 1)
    steps: list[dict] = []
    for position in range(first_position, len(usable)):
        try:
            steps.append(build_step(usable, position, benchmark_frame=benchmark_frame))
        except Exception as exc:  # a step failure degrades, never crashes
            LOGGER.warning("sequence step %s failed for %s: %s", position, ticker, exc)

    expected = lookback + 1
    coverage = len(steps) / expected if expected else 0.0
    if coverage < SEQUENCE_MIN_STEP_COVERAGE:
        return {
            **base,
            "status": SEQUENCE_STATUS_INCOMPLETE,
            # A ragged window is dropped entirely: a model trained on holes
            # learns the holes.
            "steps": [],
            "step_count": 0,
            "context": _t0_context(
                ticker, base["as_of"], market_context, macro_snapshot,
                news_snapshot, sentiment_snapshot, fundamentals,
            ),
            "labels": labels,
            "sequence_hash": None,
            "reason": (
                f"only {len(steps)} of {expected} steps built "
                f"({coverage:.0%} < {SEQUENCE_MIN_STEP_COVERAGE:.0%}); a sequence "
                f"with gaps is not a sequence"
            ),
        }

    context = _t0_context(
        ticker, base["as_of"], market_context, macro_snapshot,
        news_snapshot, sentiment_snapshot, fundamentals,
    )
    degraded = [name for name in SEQUENCE_T0_CHANNELS
                if (context.get(name) or {}).get("status") != SEQUENCE_STATUS_OK]

    return {
        **base,
        "status": SEQUENCE_STATUS_OK if not degraded else SEQUENCE_STATUS_INCOMPLETE,
        "steps": steps,
        "step_count": len(steps),
        "context": context,
        "labels": labels,
        "sequence_hash": sequence_hash(steps, ticker, base["as_of"], labels),
        "reason": "" if not degraded else f"context channels degraded: {', '.join(sorted(degraded))}",
    }


def sequence_problems(sequence: dict) -> list[str]:
    """Validate a built sequence against the C5 contract."""
    problems: list[str] = []
    if not isinstance(sequence, dict):
        return ["sequence must be a dict"]
    for field in ("ticker", "as_of", "status", "schema_version", "calculation_version"):
        if not sequence.get(field):
            problems.append(f"sequence field {field!r} missing/empty")

    steps = sequence.get("steps") or []
    if sequence.get("status") == SEQUENCE_STATUS_OK:
        expected = int(sequence.get("lookback_steps") or 0) + 1
        if len(steps) != expected:
            problems.append(f"an OK sequence must carry {expected} steps, found {len(steps)}")
        if not sequence.get("sequence_hash"):
            problems.append("an OK sequence must carry a hash")

    # The leakage check: visibility must increase by exactly one bar per step,
    # and a step must never see more bars than its position allows.
    previous_visible = None
    previous_time = None
    for index, step in enumerate(steps):
        visible = step.get("visible_bars")
        if previous_visible is not None and visible != previous_visible + 1:
            problems.append(
                f"step {index} sees {visible} bars after {previous_visible} — a step "
                f"must advance by exactly one bar, or it is seeing the future"
            )
        step_time = step.get("step_time")
        if previous_time is not None and step_time is not None and step_time <= previous_time:
            problems.append(f"step {index} is not strictly after the previous step")
        previous_visible = visible
        previous_time = step_time

    context = sequence.get("context") or {}
    if steps and context.get("context_scope") != SEQUENCE_CONTEXT_SCOPE:
        problems.append(
            "context must be labelled as_of_t0 — unlabelled context would be "
            "mistaken for per-step state"
        )
    return problems
