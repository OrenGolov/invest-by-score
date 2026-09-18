"""Price/volume feature expansion (Sprint C2) — the chart features M1 did not have.

`compute_chart_features(frame, benchmark_frame=None)` reduces an OHLCV frame to
the C2 feature surface. Every value is registered in the M1 feature registry
against the `chart_feature_agent` producer, so a model can consume it; an
unregistered feature cannot enter a production model (M1 acceptance).

**What is NOT here, deliberately.** Returns, momentum, volatility, ATR, volume
surprise and slope are already registered against `market_data_agent`
(`change_*`, `rsi`, `volatility`, `atr_14`, `volume_ratio_20d`,
`trend_slope_60d`). Recomputing them here would create a second implementation
of the same quantity — the split-brain scoring the master context forbids (W5:
one canonical technical truth). C2 adds only what was genuinely absent.

**PIT contract.** This module does no fetching and no as_of filtering. It reads
the frame it is handed, and only ever looks BACKWARDS from the last row. The
caller is responsible for passing bars already filtered to as_of — for a
multi-timeframe caller that is `core.timeframes.eligible_bars`, which excludes
still-forming bars. Keeping the filter in one place means a feature cannot
quietly acquire its own, divergent, notion of "now".

**Insufficient history yields None, never zero.** A drawdown computed over
three bars is not a drawdown, and a neutral-looking 0.0 is indistinguishable
from a real one. Each feature declares its minimum in
`CHART_FEATURE_MIN_HISTORY` and returns None below it, with the shortfall
recorded in the result's `insufficient_history` list.

**Relative strength takes an INJECTED benchmark.** C2 never selects a benchmark
and never fabricates one — benchmark and sector selection is C3's contract. With
no benchmark frame the feature is None and says so, exactly as E4's event study
treats an absent benchmark.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.config import (
    CHART_ACCELERATION_WINDOW,
    CHART_BREAKOUT_CONFIRM_SESSIONS,
    CHART_BREAKOUT_DOWN,
    CHART_BREAKOUT_FAILED_DOWN,
    CHART_BREAKOUT_FAILED_UP,
    CHART_BREAKOUT_MARGIN,
    CHART_BREAKOUT_NONE,
    CHART_BREAKOUT_UP,
    CHART_DRAWDOWN_WINDOW,
    CHART_FEATURE_MIN_HISTORY,
    CHART_FEATURE_VERSION,
    CHART_GAP_MIN_FRACTION,
    CHART_LEVEL_WINDOW,
    CHART_PIPELINE_VERSION,
    CHART_RELATIVE_STRENGTH_WINDOW,
    CHART_VOL_CONTRACTING,
    CHART_VOL_CONTRACTION_RATIO,
    CHART_VOL_EXPANDING,
    CHART_VOL_EXPANSION_RATIO,
    CHART_VOL_LONG_WINDOW,
    CHART_VOL_SHORT_WINDOW,
    CHART_VOL_STABLE,
)

LOGGER = logging.getLogger("core.chart_features")

CHART_FEATURE_SOURCE_ID = "yahoo_finance_chart"

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


class ChartFeatureError(ValueError):
    """Raised when a chart-feature request is structurally invalid."""


def _usable(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        raise ChartFeatureError("a price frame is required")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ChartFeatureError(f"price frame is missing {', '.join(missing)}")
    return frame.dropna(subset=["Close"])


def _has(frame: pd.DataFrame, feature: str) -> bool:
    return len(frame) >= CHART_FEATURE_MIN_HISTORY[feature]


def acceleration(frame: pd.DataFrame, window: int = CHART_ACCELERATION_WINDOW) -> float | None:
    """Change in momentum: this window's return minus the previous window's.

    Momentum says the price rose. Acceleration says whether the rise is
    speeding up or stalling — a decelerating rally and a steady one look
    identical to a single-window return.
    """
    if not _has(frame, "acceleration_10d"):
        return None
    closes = frame["Close"]
    latest = float(closes.iloc[-1])
    mid = float(closes.iloc[-1 - window])
    earliest = float(closes.iloc[-1 - 2 * window])
    if mid <= 0 or earliest <= 0:
        return None
    recent_return = latest / mid - 1.0
    prior_return = mid / earliest - 1.0
    return round(recent_return - prior_return, 6)


def gap_pct(frame: pd.DataFrame) -> float | None:
    """Last session's open against the prior close, as a fraction.

    Below CHART_GAP_MIN_FRACTION this reports 0.0 rather than None: the bars
    exist and the answer is "no gap", which is different from "unknown".
    """
    if not _has(frame, "gap_pct"):
        return None
    previous_close = float(frame["Close"].iloc[-2])
    current_open = float(frame["Open"].iloc[-1])
    if previous_close <= 0:
        return None
    raw = current_open / previous_close - 1.0
    return round(raw, 6) if abs(raw) >= CHART_GAP_MIN_FRACTION else 0.0


def drawdown(frame: pd.DataFrame, window: int = CHART_DRAWDOWN_WINDOW) -> float | None:
    """Current close against the highest close in the window, as a fraction.

    Zero means at the high; negative means below it. Never positive.
    """
    if not _has(frame, "drawdown_60d"):
        return None
    closes = frame["Close"].iloc[-window:]
    peak = float(closes.max())
    if peak <= 0:
        return None
    return round(float(closes.iloc[-1]) / peak - 1.0, 6)


def recovery_speed(frame: pd.DataFrame, window: int = CHART_DRAWDOWN_WINDOW) -> float | None:
    """How much of the window's worst drawdown has been recovered, in [0, 1].

    1.0 means fully recovered to the peak; 0.0 means sitting at the trough.
    Reported only when a real drawdown occurred — recovery from nothing is
    not a measurement.
    """
    if not _has(frame, "recovery_speed_60d"):
        return None
    closes = frame["Close"].iloc[-window:]
    peak = float(closes.max())
    trough = float(closes.min())
    if peak <= 0 or peak <= trough:
        return None
    return round((float(closes.iloc[-1]) - trough) / (peak - trough), 6)


def support_distance(frame: pd.DataFrame, window: int = CHART_LEVEL_WINDOW) -> float | None:
    """Distance from the window's low, as a fraction of the last close."""
    if not _has(frame, "support_distance_60d"):
        return None
    lows = frame["Low"].iloc[-window:] if "Low" in frame.columns else frame["Close"].iloc[-window:]
    support = float(lows.min())
    last = float(frame["Close"].iloc[-1])
    if last <= 0:
        return None
    return round((last - support) / last, 6)


def resistance_distance(frame: pd.DataFrame, window: int = CHART_LEVEL_WINDOW) -> float | None:
    """Distance to the window's high, as a fraction of the last close."""
    if not _has(frame, "resistance_distance_60d"):
        return None
    highs = frame["High"].iloc[-window:] if "High" in frame.columns else frame["Close"].iloc[-window:]
    resistance = float(highs.max())
    last = float(frame["Close"].iloc[-1])
    if last <= 0:
        return None
    return round((resistance - last) / last, 6)


def breakout_state(
    frame: pd.DataFrame,
    window: int = CHART_LEVEL_WINDOW,
    confirm: int = CHART_BREAKOUT_CONFIRM_SESSIONS,
) -> str | None:
    """Breakout, failed breakout, or neither.

    The failed case is the reason this feature earns its place: a break that
    reverses back inside the range is a different event from one that holds,
    and a plain "did it exceed the high" flag cannot tell them apart. The
    range is measured on bars BEFORE the confirmation window, so the breakout
    itself never redefines the level it broke.
    """
    if not _has(frame, "breakout_state_60d"):
        return None
    closes = frame["Close"]
    reference = closes.iloc[-(window + confirm):-confirm]
    if reference.empty:
        return None
    high = float(reference.max())
    low = float(reference.min())
    recent = closes.iloc[-confirm:]
    last = float(closes.iloc[-1])

    broke_up = bool((recent > high * (1.0 + CHART_BREAKOUT_MARGIN)).any())
    broke_down = bool((recent < low * (1.0 - CHART_BREAKOUT_MARGIN)).any())
    holding_up = last > high * (1.0 + CHART_BREAKOUT_MARGIN)
    holding_down = last < low * (1.0 - CHART_BREAKOUT_MARGIN)

    if broke_up and holding_up:
        return CHART_BREAKOUT_UP
    if broke_down and holding_down:
        return CHART_BREAKOUT_DOWN
    if broke_up:
        return CHART_BREAKOUT_FAILED_UP
    if broke_down:
        return CHART_BREAKOUT_FAILED_DOWN
    return CHART_BREAKOUT_NONE


def _realized_volatility(closes: pd.Series, window: int) -> float | None:
    returns = closes.pct_change().dropna().iloc[-window:]
    if len(returns) < 2:
        return None
    value = float(returns.std())
    return value if value == value else None  # NaN guard


def volatility_regime_ratio(frame: pd.DataFrame) -> float | None:
    """Short realized vol over long realized vol.

    Above 1 the market is moving faster than its own recent baseline. The
    ratio is reported rather than only the label, so a caller can see how far
    from stable it is.
    """
    if not _has(frame, "volatility_regime_ratio"):
        return None
    closes = frame["Close"]
    short = _realized_volatility(closes, CHART_VOL_SHORT_WINDOW)
    long = _realized_volatility(closes, CHART_VOL_LONG_WINDOW)
    if short is None or long is None or long <= 0:
        return None
    return round(short / long, 6)


def volatility_regime(ratio: float | None) -> str | None:
    """Label the ratio. None in, None out — never a default 'stable'."""
    if ratio is None:
        return None
    if ratio >= CHART_VOL_EXPANSION_RATIO:
        return CHART_VOL_EXPANDING
    if ratio <= CHART_VOL_CONTRACTION_RATIO:
        return CHART_VOL_CONTRACTING
    return CHART_VOL_STABLE


def relative_strength(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
    window: int = CHART_RELATIVE_STRENGTH_WINDOW,
) -> float | None:
    """Stock return minus benchmark return over the window.

    The benchmark is INJECTED. C2 does not choose one and does not invent one:
    with no benchmark frame this is None, because "+4% while the index did
    +4%" and "+4% while the index did -2%" are different facts and a missing
    benchmark cannot distinguish them.

    Aligned by TIMESTAMP, not by position — a benchmark with a different
    history length would otherwise be measured over a different calendar
    window, and that mismatch would masquerade as relative strength.
    """
    if benchmark_frame is None or not _has(frame, "relative_strength_60d"):
        return None
    if "Close" not in benchmark_frame.columns:
        return None

    stock = frame["Close"]
    benchmark = benchmark_frame.dropna(subset=["Close"])["Close"]
    shared = stock.index.intersection(benchmark.index)
    if len(shared) < window + 1:
        return None
    shared = shared.sort_values()[-(window + 1):]

    # The window must END at the stock's own last bar. Intersecting alone lets a
    # stale benchmark drag the comparison backwards silently: a benchmark ending
    # 20 sessions early moved the reported value 0.4938 -> 0.6164 on the same
    # stock, because the stock's most recent sessions were quietly dropped.
    # Refusing is correct -- a relative strength measured over a window the
    # caller did not ask for is worse than no number.
    if shared[-1] != stock.index[-1]:
        return None

    stock_start, stock_end = float(stock.loc[shared[0]]), float(stock.loc[shared[-1]])
    bench_start, bench_end = float(benchmark.loc[shared[0]]), float(benchmark.loc[shared[-1]])
    if stock_start <= 0 or bench_start <= 0:
        return None
    return round((stock_end / stock_start) - (bench_end / bench_start), 6)


def compute_chart_features(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
) -> dict:
    """The C2 feature surface for one already-PIT-filtered frame.

    Returns `{features, insufficient_history, bar_count, versions}`. A feature
    that could not be computed is present with a None value AND named in
    `insufficient_history`, so a consumer can tell "not enough data" from
    "computed, and the answer is zero".
    """
    usable = _usable(frame)
    ratio = volatility_regime_ratio(usable)
    features: dict[str, object] = {
        "acceleration_10d": acceleration(usable),
        "gap_pct": gap_pct(usable),
        "drawdown_60d": drawdown(usable),
        "recovery_speed_60d": recovery_speed(usable),
        "support_distance_60d": support_distance(usable),
        "resistance_distance_60d": resistance_distance(usable),
        "breakout_state_60d": breakout_state(usable),
        "volatility_regime_ratio": ratio,
        "volatility_regime": volatility_regime(ratio),
        "relative_strength_60d": relative_strength(usable, benchmark_frame),
    }
    insufficient = sorted(
        name for name, value in features.items()
        if value is None and name in CHART_FEATURE_MIN_HISTORY
    )
    return {
        "features": features,
        "insufficient_history": insufficient,
        "bar_count": int(len(usable)),
        "benchmark_supplied": benchmark_frame is not None,
        "calculation_version": CHART_FEATURE_VERSION,
        "pipeline_version": CHART_PIPELINE_VERSION,
        "source_id": CHART_FEATURE_SOURCE_ID,
    }
