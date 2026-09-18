"""Deterministic chart structure (Sprint C4) — what the chart is DOING, reproducibly.

`describe_structure(frame)` composes a structural description of an OHLCV frame:

    swing pivots -> swing structure (HH/HL, LH/LL, broadening, narrowing)
                 -> consolidation / reversal / trend
                 -> PHASE (one word, by declared precedence)

**Not pattern recognition.** The master context is explicit: avoid screenshot
interpretation and vague pattern labels. So there is no "head and shoulders"
here, no "cup and handle", no visual matching. Every label is derived from
measurable primitives with thresholds declared in `core.config`: the same bars
always produce the same structure, and a reader can check the arithmetic by
hand. A label nobody can recompute is not evidence.

**The PIT hazard: a fractal pivot needs bars AFTER it.** A bar is a swing high
when its High is the maximum of the window CENTRED on it — which cannot be known
until `CHART_SWING_FRACTAL_K` further bars exist. Scanning to the final bar would
let future bars decide a past label, the same class of leak C1's still-forming
bar was. The scan therefore stops k bars short of the end, and those unconfirmed
trailing bars are REPORTED (`unconfirmed_tail_bars`) rather than silently
trimmed, so a consumer can see that the most recent action is not yet structural.

**Composition, not duplication.** Breakout state, gaps and the volatility regime
are already registered C2 features owned by `chart_feature_agent`. C4 consumes
them through `core.chart_features` rather than recomputing — a second
implementation would be the split-brain the master context forbids (W5). What
C4 adds is the swing structure and the composition into a phase.

**Insufficient history yields an undefined phase, never a guess.** Below
`CHART_STRUCTURE_MIN_BARS` the structure is `undefined` with a reason, because a
confident-looking "trending" drawn from thirty bars is worse than an admission.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.chart_features import (
    breakout_state,
    gap_pct,
    volatility_regime,
    volatility_regime_ratio,
)
from core.config import (
    CHART_BREAKOUT_DOWN,
    CHART_BREAKOUT_FAILED_DOWN,
    CHART_BREAKOUT_FAILED_UP,
    CHART_BREAKOUT_UP,
    CHART_CONSOLIDATION_MAX_RANGE,
    CHART_CONSOLIDATION_WINDOW,
    CHART_REVERSAL_MIN_MOVE,
    CHART_REVERSAL_WINDOW,
    CHART_STRUCTURE_MIN_BARS,
    CHART_SWING_FRACTAL_K,
    CHART_SWING_LOOKBACK,
    CHART_SWING_MIN_CHANGE,
    STRUCTURE_CONTRACT_VERSION,
    STRUCTURE_PHASE_BREAKOUT,
    STRUCTURE_PHASE_CONSOLIDATION,
    STRUCTURE_PHASE_FAILED_BREAKOUT,
    STRUCTURE_PHASE_REVERSAL,
    STRUCTURE_PHASE_TRENDING,
    STRUCTURE_PHASE_UNDEFINED,
    STRUCTURE_PIPELINE_VERSION,
    STRUCTURE_SWING_CONTRACTING,
    STRUCTURE_SWING_DOWNTREND,
    STRUCTURE_SWING_EXPANDING,
    STRUCTURE_SWING_MIXED,
    STRUCTURE_SWING_UPTREND,
)

LOGGER = logging.getLogger("core.chart_structure")

STRUCTURE_SOURCE_ID = "yahoo_finance_chart"

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


class ChartStructureError(ValueError):
    """Raised when a structure request is structurally invalid."""


def _usable(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        raise ChartStructureError("a price frame is required")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ChartStructureError(f"price frame is missing {', '.join(missing)}")
    return frame.dropna(subset=["Close"])


def swing_points(
    frame: pd.DataFrame,
    k: int = CHART_SWING_FRACTAL_K,
) -> tuple[list[dict], list[dict]]:
    """Confirmed fractal swing highs and lows.

    A pivot at index i requires bars i-k..i+k, so the scan stops k bars short
    of the end. Those trailing bars cannot yet be classified — claiming a pivot
    there would be future data deciding a past label.
    """
    highs, lows = frame["High"], frame["Low"]
    swing_highs: list[dict] = []
    swing_lows: list[dict] = []
    for position in range(k, len(frame) - k):
        window_high = highs.iloc[position - k: position + k + 1]
        window_low = lows.iloc[position - k: position + k + 1]
        if float(highs.iloc[position]) == float(window_high.max()):
            swing_highs.append(
                {"index": position, "time": str(frame.index[position]), "price": round(float(highs.iloc[position]), 6)}
            )
        if float(lows.iloc[position]) == float(window_low.min()):
            swing_lows.append(
                {"index": position, "time": str(frame.index[position]), "price": round(float(lows.iloc[position]), 6)}
            )
    return swing_highs, swing_lows


def _direction(previous: float, current: float) -> int:
    """+1 higher, -1 lower, 0 inside the noise band."""
    if previous <= 0:
        return 0
    change = (current - previous) / previous
    if change > CHART_SWING_MIN_CHANGE:
        return 1
    if change < -CHART_SWING_MIN_CHANGE:
        return -1
    return 0


def swing_structure(
    swing_highs: list[dict],
    swing_lows: list[dict],
    lookback: int = CHART_SWING_LOOKBACK,
) -> str | None:
    """Label the relationship between the last highs and the last lows.

    The four named outcomes are the ones that carry information: an uptrend
    (higher high AND higher low), a downtrend, a broadening range (higher high
    AND lower low) and a narrowing one. Anything else is honestly `mixed`
    rather than forced into a trend.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    recent_highs = swing_highs[-lookback:]
    recent_lows = swing_lows[-lookback:]
    high_direction = _direction(recent_highs[-2]["price"], recent_highs[-1]["price"])
    low_direction = _direction(recent_lows[-2]["price"], recent_lows[-1]["price"])

    if high_direction > 0 and low_direction > 0:
        return STRUCTURE_SWING_UPTREND
    if high_direction < 0 and low_direction < 0:
        return STRUCTURE_SWING_DOWNTREND
    if high_direction > 0 and low_direction < 0:
        return STRUCTURE_SWING_EXPANDING
    if high_direction < 0 and low_direction > 0:
        return STRUCTURE_SWING_CONTRACTING
    return STRUCTURE_SWING_MIXED


def consolidation_range(
    frame: pd.DataFrame,
    window: int = CHART_CONSOLIDATION_WINDOW,
) -> float | None:
    """The window's high-low range as a fraction of its mean close."""
    if len(frame) < window:
        return None
    recent = frame.iloc[-window:]
    mean_close = float(recent["Close"].mean())
    if mean_close <= 0:
        return None
    span = float(recent["High"].max()) - float(recent["Low"].min())
    return round(span / mean_close, 6)


def is_consolidating(frame: pd.DataFrame) -> bool | None:
    """Whether price is going sideways rather than trending."""
    span = consolidation_range(frame)
    if span is None:
        return None
    return span <= CHART_CONSOLIDATION_MAX_RANGE


def reversal_state(
    frame: pd.DataFrame,
    window: int = CHART_REVERSAL_WINDOW,
) -> str | None:
    """A directional flip where BOTH legs clear the threshold.

    Requiring both legs to be real moves is what separates a reversal from a
    drift that happened to change sign. Returns "reversal_up",
    "reversal_down", or "none".
    """
    if len(frame) < window * 2 + 1:
        return None
    closes = frame["Close"]
    latest = float(closes.iloc[-1])
    mid = float(closes.iloc[-1 - window])
    earliest = float(closes.iloc[-1 - 2 * window])
    if mid <= 0 or earliest <= 0:
        return None
    prior_leg = mid / earliest - 1.0
    recent_leg = latest / mid - 1.0
    if prior_leg <= -CHART_REVERSAL_MIN_MOVE and recent_leg >= CHART_REVERSAL_MIN_MOVE:
        return "reversal_up"
    if prior_leg >= CHART_REVERSAL_MIN_MOVE and recent_leg <= -CHART_REVERSAL_MIN_MOVE:
        return "reversal_down"
    return "none"


def resolve_phase(
    breakout: str | None,
    reversal: str | None,
    consolidating: bool | None,
    swing: str | None,
) -> str:
    """Collapse the structure into one word by DECLARED precedence.

    Precedence, most specific first:

      failed breakout > breakout > reversal > consolidation > trending

    A failed breakout outranks a breakout because it is the more specific
    statement about the same event: price left the range and came back. A
    breakout outranks a reversal because it is a range event, not a swing
    event, and outranks consolidation because a range that broke is no longer
    a range. Trending is last — it is what remains when nothing sharper
    applies.
    """
    if breakout in (CHART_BREAKOUT_FAILED_UP, CHART_BREAKOUT_FAILED_DOWN):
        return STRUCTURE_PHASE_FAILED_BREAKOUT
    if breakout in (CHART_BREAKOUT_UP, CHART_BREAKOUT_DOWN):
        return STRUCTURE_PHASE_BREAKOUT
    if reversal in ("reversal_up", "reversal_down"):
        return STRUCTURE_PHASE_REVERSAL
    if consolidating:
        return STRUCTURE_PHASE_CONSOLIDATION
    if swing in (STRUCTURE_SWING_UPTREND, STRUCTURE_SWING_DOWNTREND):
        return STRUCTURE_PHASE_TRENDING
    return STRUCTURE_PHASE_UNDEFINED


def describe_structure(frame: pd.DataFrame) -> dict:
    """The C4 structure representation for one already-PIT-filtered frame.

    Like C2, this does no fetching and no as_of filtering: the caller passes
    bars already filtered through `core.timeframes.eligible_bars`, so there is
    exactly one notion of "now" in the system.
    """
    usable = _usable(frame)
    bar_count = int(len(usable))

    if bar_count < CHART_STRUCTURE_MIN_BARS:
        return {
            "phase": STRUCTURE_PHASE_UNDEFINED,
            "swing_structure": None,
            "swing_highs": [],
            "swing_lows": [],
            "unconfirmed_tail_bars": min(CHART_SWING_FRACTAL_K, bar_count),
            "consolidation_range": None,
            "consolidating": None,
            "reversal": None,
            "breakout_state": None,
            "gap_pct": None,
            "volatility_regime": None,
            "volatility_regime_ratio": None,
            "bar_count": bar_count,
            "calculation_version": STRUCTURE_CONTRACT_VERSION,
            "pipeline_version": STRUCTURE_PIPELINE_VERSION,
            "source_id": STRUCTURE_SOURCE_ID,
            "reason": (
                f"{bar_count} bars is below the {CHART_STRUCTURE_MIN_BARS} required to "
                f"describe a structure; a confident label drawn from too little "
                f"history is worse than none"
            ),
        }

    swing_highs, swing_lows = swing_points(usable)
    swing = swing_structure(swing_highs, swing_lows)
    # C2 owns these: consume, never recompute (W5).
    breakout = breakout_state(usable)
    ratio = volatility_regime_ratio(usable)
    structure_reversal = reversal_state(usable)
    consolidating = is_consolidating(usable)

    phase = resolve_phase(breakout, structure_reversal, consolidating, swing)
    # `undefined` is a legitimate outcome — a broadening or narrowing range is
    # genuinely neither trending nor consolidating — but it must always say WHY,
    # or it is indistinguishable from a failure to compute.
    undefined_reason = ""
    if phase == STRUCTURE_PHASE_UNDEFINED:
        undefined_reason = (
            f"no sharper phase applies: swing structure is {swing or 'unestablished'}, "
            f"which is neither a directional trend nor a consolidation"
        )

    return {
        "phase": phase,
        "swing_structure": swing,
        "swing_highs": swing_highs[-CHART_SWING_LOOKBACK:],
        "swing_lows": swing_lows[-CHART_SWING_LOOKBACK:],
        # The tail a fractal pivot cannot yet confirm. Reported, not hidden.
        "unconfirmed_tail_bars": CHART_SWING_FRACTAL_K,
        "consolidation_range": consolidation_range(usable),
        "consolidating": consolidating,
        "reversal": structure_reversal,
        "breakout_state": breakout,
        "gap_pct": gap_pct(usable),
        "volatility_regime": volatility_regime(ratio),
        "volatility_regime_ratio": ratio,
        "bar_count": bar_count,
        "calculation_version": STRUCTURE_CONTRACT_VERSION,
        "pipeline_version": STRUCTURE_PIPELINE_VERSION,
        "source_id": STRUCTURE_SOURCE_ID,
        "reason": undefined_reason,
    }


def structure_problems(structure: dict) -> list[str]:
    """Validate a built structure against the C4 contract."""
    problems: list[str] = []
    if not isinstance(structure, dict):
        return ["structure must be a dict"]
    for field in ("phase", "bar_count", "calculation_version"):
        if structure.get(field) is None:
            problems.append(f"structure field {field!r} missing")
    if structure.get("phase") == STRUCTURE_PHASE_UNDEFINED and not structure.get("reason"):
        problems.append("an undefined phase must explain itself")
    # Any structure built from enough bars was genuinely described, even when
    # the phase resolved to `undefined` — so it must still declare its tail.
    # Keying this on the phase alone let an undefined-but-described structure
    # skip the check entirely.
    if int(structure.get("bar_count") or 0) >= CHART_STRUCTURE_MIN_BARS:
        if structure.get("unconfirmed_tail_bars") != CHART_SWING_FRACTAL_K:
            problems.append(
                "a described structure must declare its unconfirmed tail — the "
                "bars a fractal pivot cannot yet classify"
            )
    if structure.get("phase") == STRUCTURE_PHASE_CONSOLIDATION and not structure.get("consolidating"):
        problems.append("phase is consolidation but the range says otherwise")
    return problems
