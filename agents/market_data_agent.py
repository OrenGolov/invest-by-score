from __future__ import annotations

import pandas as pd

from fetch_data import fetch_price_history


def _coerce_as_of(as_of: str, timestamp: str | None = None) -> pd.Timestamp:
    base = pd.Timestamp(as_of)
    if timestamp is None:
        return base.normalize()
    return pd.Timestamp(f"{base.date()} {timestamp}")


def _pct_change(series: pd.Series, periods: int = 1) -> float:
    if len(series) <= periods:
        return 0.0
    previous = series.iloc[-periods]
    if pd.isna(previous) or previous == 0:
        return 0.0
    return float((series.iloc[-1] / previous) - 1.0)


def fetch_market_snapshot(ticker: str, as_of: str, timestamp: str | None = None) -> dict:
    """Fetch the latest available price bar on or before a requested as-of date."""
    target = _coerce_as_of(as_of, timestamp)
    if target > pd.Timestamp.now():
        raise ValueError(f"Requested as-of date {target} is in the future.")

    days_back = (pd.Timestamp.now().normalize() - target.normalize()).days
    if days_back <= 365:
        period = "1y"
    elif days_back <= 1825:
        period = "5y"
    else:
        period = "10y"

    history = fetch_price_history(ticker, period=period, interval="1d")
    history = history.sort_index()
    history = history[history.index <= target]

    if history.empty:
        raise ValueError(f"No market data available for {ticker} on or before {target}.")

    latest = history.iloc[-1]
    recent_5 = history.tail(5)
    recent_20 = history.tail(20)
    avg_volume_20d = float(recent_20["Volume"].mean()) if not recent_20.empty else float(latest["Volume"])
    close_20d_mean = float(recent_20["Close"].mean()) if not recent_20.empty else float(latest["Close"])

    return {
        "ticker": ticker.upper(),
        "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
        "close": float(latest["Close"]),
        "volume": float(latest["Volume"]),
        "avg_volume_20d": avg_volume_20d,
        "change_1d": _pct_change(history["Close"], 1),
        "change_5d": _pct_change(history["Close"], 5),
        "change_20d": _pct_change(history["Close"], 20),
        "high_20d": float(recent_20["High"].max()),
        "low_20d": float(recent_20["Low"].min()),
        "trend_vs_20d_mean": float(latest["Close"] / close_20d_mean - 1.0) if close_20d_mean else 0.0,
        "recent_5d_min": float(recent_5["Low"].min()),
        "recent_5d_max": float(recent_5["High"].max()),
    }
