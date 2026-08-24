from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.schemas import MarketSnapshot


def fetch_market_snapshot(ticker: str, as_of: str) -> MarketSnapshot:
    from fetch_data import fetch_price_history

    data = fetch_price_history(ticker, period="6mo", interval="1d")
    if data.empty:
        raise ValueError(f"No market data available for {ticker}")

    as_of_dt = pd.to_datetime(as_of)
    recent = data.loc[data.index <= as_of_dt].tail(10)
    if recent.empty:
        raise ValueError(f"No market data available up to {as_of} for {ticker}")

    latest_close = float(recent["Close"].iloc[-1])
    returns = ((recent["Close"] / recent["Close"].shift(1)) - 1).dropna().tolist()

    if not returns:
        trend = "neutral"
    else:
        recent_pct = sum(returns[-3:])
        trend = "bullish" if recent_pct > 0 else "bearish" if recent_pct < 0 else "neutral"

    volatility = float(recent["Close"].pct_change().dropna().std()) if len(recent) > 1 else 0.0

    return MarketSnapshot(
        ticker=ticker.upper(),
        as_of=as_of,
        latest_close=latest_close,
        recent_returns=[float(v) for v in returns[-5:]],
        trend=trend,
        volatility=volatility,
    )
