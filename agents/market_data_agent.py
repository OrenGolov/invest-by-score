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
    """Fetch and validate the latest market snapshot on or before the requested timestamp."""
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

    try:
        history = fetch_price_history(ticker, period=period, interval="1d")
    except Exception as exc:
        raise ValueError(f"No market data available for {ticker} on or before {target}.") from exc
    history = history.sort_index()
    future_bars_excluded = int((history.index > target).sum())
    history = history[history.index <= target].copy()

    if history.empty:
        raise ValueError(f"No market data available for {ticker} on or before {target}.")

    history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if history.empty:
        raise ValueError(f"No valid market data available for {ticker} on or before {target}.")

    latest = history.iloc[-1]
    recent_5 = history.tail(5)
    recent_20 = history.tail(20)
    avg_volume_20d = float(recent_20["Volume"].mean()) if not recent_20.empty else float(latest["Volume"])
    close_20d_mean = float(recent_20["Close"].mean()) if not recent_20.empty else float(latest["Close"])
    latest_close = float(latest["Close"])
    latest_volume = float(latest["Volume"])

    close = history["Close"].astype(float)
    ma_50 = close.rolling(window=50, min_periods=1).mean().iloc[-1]
    ma_100 = close.rolling(window=100, min_periods=1).mean().iloc[-1]
    ma_150 = close.rolling(window=150, min_periods=1).mean().iloc[-1]
    ma_200 = close.rolling(window=200, min_periods=1).mean().iloc[-1]

    delta = close.diff().dropna()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gain = gains.rolling(window=14, min_periods=1).mean().iloc[-1]
    avg_loss = losses.rolling(window=14, min_periods=1).mean().iloc[-1]
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    returns = close.pct_change().dropna()
    volatility = returns.tail(30).std(ddof=0) if not returns.empty else 0.0
    volume_ratio_20d = latest_volume / avg_volume_20d if avg_volume_20d else 1.0

    price_vs_ma_50 = ((latest_close / ma_50) - 1.0) if ma_50 else 0.0
    price_vs_ma_200 = ((latest_close / ma_200) - 1.0) if ma_200 else 0.0
    price_vs_ma_100 = ((latest_close / ma_100) - 1.0) if ma_100 else 0.0

    bars_available = len(history)
    gaps_detected = int(history.index.to_series().diff().dt.days.gt(1).sum())
    quality_flags: list[str] = []
    quality_score = 100.0

    if future_bars_excluded > 0:
        quality_score -= min(25.0, future_bars_excluded * 0.5)
        quality_flags.append("future_bars_excluded")
    if bars_available < 30:
        quality_score -= 15.0
        quality_flags.append("limited_history")
    if gaps_detected > 0:
        quality_score -= min(25.0, gaps_detected * 10.0)
        quality_flags.append("market_gap_detected")
    if latest_volume <= 0:
        quality_score -= 20.0
        quality_flags.append("zero_volume")
    if volume_ratio_20d < 0.5:
        quality_score -= 15.0
        quality_flags.append("low_volume_relative_to_recent_average")
    if price_vs_ma_200 < -0.15:
        quality_score -= 10.0
        quality_flags.append("below_200d_ma")
    if rsi < 30.0 or rsi > 70.0:
        quality_score -= 5.0
        quality_flags.append("rsi_extreme")
    if quality_score < 50.0:
        quality_flags.append("degraded_quality")

    market_regime = "bullish"
    if latest_close < ma_200 and ma_50 < ma_200:
        market_regime = "bearish"
    elif abs(price_vs_ma_50) < 0.02 and abs(price_vs_ma_200) < 0.02:
        market_regime = "neutral"

    chart_series = []
    close_series = history["Close"].astype(float)
    ma50_series = close_series.rolling(window=50, min_periods=1).mean()
    ma100_series = close_series.rolling(window=100, min_periods=1).mean()
    ma200_series = close_series.rolling(window=200, min_periods=1).mean()
    for index, ts in enumerate(history.index):
        chart_series.append({
            "date": ts.strftime("%Y-%m-%d"),
            "close": round(float(close_series.iloc[index]), 2),
            "ma_50": round(float(ma50_series.iloc[index]), 2),
            "ma_100": round(float(ma100_series.iloc[index]), 2),
            "ma_200": round(float(ma200_series.iloc[index]), 2),
            "volume": int(history["Volume"].iloc[index]),
        })

    timestamp_valid = all(ts <= target for ts in history.index)
    return {
        "ticker": ticker.upper(),
        "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
        "point_in_time_cutoff": target.strftime("%Y-%m-%d %H:%M:%S"),
        "future_bars_excluded": int(future_bars_excluded),
        "point_in_time_policy": "Only bars with timestamp <= as_of are eligible for scoring; all future bars are excluded before indicators are calculated.",
        "source": "Yahoo Finance",
        "source_type": "public_market_feed",
        "source_confidence": 0.8,
        "source_contract": {
            "provider": "Yahoo Finance",
            "source_type": "market_data",
            "source_id": "yahoo_finance_chart",
            "source_confidence": 0.8,
            "source_timestamp": history.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "live_provider",
            "timestamp_valid": bool(timestamp_valid),
            "source_timestamp_must_be_at_or_before_as_of": True,
            "policy": "Future-dated bars are removed before technical indicators are calculated."
        },
        "last_valid_bar": history.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "first_valid_bar": history.index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "bars_available": bars_available,
        "gaps_detected": gaps_detected,
        "missing_data": int(history["Close"].isna().sum()),
        "close": float(latest_close),
        "volume": latest_volume,
        "avg_volume_20d": avg_volume_20d,
        "volume_ratio_20d": round(volume_ratio_20d, 4),
        "price_vs_ma_50": round(price_vs_ma_50, 4),
        "price_vs_ma_100": round(price_vs_ma_100, 4),
        "price_vs_ma_150": round(float((latest_close / ma_150) - 1.0) if ma_150 else 0.0, 4),
        "price_vs_ma_200": round(price_vs_ma_200, 4),
        "change_1d": _pct_change(history["Close"], 1),
        "change_5d": _pct_change(history["Close"], 5),
        "change_20d": _pct_change(history["Close"], 20),
        "high_20d": float(recent_20["High"].max()),
        "low_20d": float(recent_20["Low"].min()),
        "trend_vs_20d_mean": float(latest_close / close_20d_mean - 1.0) if close_20d_mean else 0.0,
        "recent_5d_min": float(recent_5["Low"].min()),
        "recent_5d_max": float(recent_5["High"].max()),
        "moving_averages": {
            "200d": round(float(ma_200), 2),
            "150d": round(float(ma_150), 2),
            "100d": round(float(ma_100), 2),
            "50d": round(float(ma_50), 2),
        },
        "rsi": round(float(rsi), 2),
        "volatility": round(float(volatility), 4),
        "chart_series": chart_series,
        "data_quality_score": round(float(max(0.0, min(100.0, quality_score))), 2),
        "quality_flags": quality_flags,
        "market_regime": market_regime,
        "data_quality": {
            "score": round(float(max(0.0, min(100.0, quality_score))), 2),
            "flags": quality_flags,
            "bars_available": bars_available,
            "gaps_detected": gaps_detected,
            "missing_data": int(history["Close"].isna().sum()),
            "future_bars_excluded": int(future_bars_excluded),
            "point_in_time_cutoff": target.strftime("%Y-%m-%d %H:%M:%S"),
            "point_in_time_policy": "Only bars with timestamp <= as_of are eligible for scoring; all future bars are excluded before indicators are calculated.",
        },
    }
