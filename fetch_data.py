"""Fetch OHLCV price history for one or more tickers from Yahoo Finance.

Uses urllib against Yahoo's public chart API directly rather than the
yfinance library. yfinance's curl_cffi HTTP client validates TLS against
its own bundled CA list and fails behind this network's corporate
TLS-inspecting proxy; urllib validates via the OS trust store instead,
so it works without any custom certificate handling.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
DEFAULT_TIMEOUT_SECONDS = 10

# Portfolio tracked by this project.
PORTFOLIO_TICKERS = [
    "VOO",
    "CIBR",
    "V",
    "SOXX",
    "RKLB",
    "GOOGL",
    "NVDA",
    "AVGO",
    "MSFT",
    "VRT",
    "ANET",
    "CAT",
]


class TickerFetchError(Exception):
    """Raised when price history for a ticker cannot be fetched or parsed."""


def fetch_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch OHLCV price history for a single ticker.

    Args:
        ticker: Symbol to fetch, e.g. "AAPL".
        period: Yahoo Finance range string, e.g. "6mo", "1y", "5y".
        interval: Bar size, e.g. "1d", "1wk".
        timeout: Request timeout in seconds.

    Returns:
        DataFrame indexed by timestamp with Open/High/Low/Close/Volume
        columns.

    Raises:
        TickerFetchError: On network failure, an unknown/invalid ticker,
            or a response that doesn't match the expected shape.
    """
    url = YAHOO_CHART_URL.format(ticker=ticker)
    params = f"?range={period}&interval={interval}"
    request = urllib.request.Request(
        url + params, headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise TickerFetchError(f"{ticker}: network request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TickerFetchError(f"{ticker}: response was not valid JSON: {exc}") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise TickerFetchError(f"{ticker}: Yahoo Finance returned an error: {error}")

    results = chart.get("result")
    if not results:
        raise TickerFetchError(f"{ticker}: no data returned (unknown ticker?)")

    result = results[0]
    try:
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        data = pd.DataFrame(
            {
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Volume": quote["volume"],
            },
            index=pd.to_datetime(timestamps, unit="s"),
        )
    except (KeyError, IndexError) as exc:
        raise TickerFetchError(f"{ticker}: response missing expected fields: {exc}") from exc

    if data.empty:
        raise TickerFetchError(f"{ticker}: response contained no price bars")

    data.index.name = "Date"
    return data


def fetch_portfolio(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers.

    A failure on any single ticker (bad symbol, network error, rate limit)
    is logged and skipped rather than raised, so one bad ticker can't crash
    a batch fetch for the whole portfolio.

    Args:
        tickers: Symbols to fetch.
        period: Yahoo Finance range string, e.g. "6mo", "1y", "5y".
        interval: Bar size, e.g. "1d", "1wk".

    Returns:
        Mapping of ticker to its price history DataFrame. Tickers that
        failed to fetch are omitted.
    """
    history: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            history[ticker] = fetch_price_history(ticker, period, interval)
        except TickerFetchError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    portfolio_history = fetch_portfolio(PORTFOLIO_TICKERS)
    for symbol, df in portfolio_history.items():
        print(f"\n{symbol}")
        print(df.tail())
