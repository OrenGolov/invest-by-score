"""Fetch OHLCV price history for one or more tickers from Yahoo Finance.

Uses urllib against Yahoo's public chart API directly rather than the
yfinance library. yfinance's curl_cffi HTTP client validates TLS against
its own bundled CA list and fails behind this network's corporate
TLS-inspecting proxy; urllib validates via the OS trust store instead,
so it works without any custom certificate handling.

Fetched data is cached to disk under CACHE_DIR (Parquet, one file per
ticker/period/interval combination) so repeated runs within the TTL
window don't re-hit Yahoo Finance for every ticker. Parquet was chosen
over CSV because it round-trips dtypes exactly (the datetime index and
Volume's integer type survive without re-parsing) and reads/writes
faster, which matters once fundamentals and indicators are cached
alongside price data too; the tradeoff is that it's a binary format, so
you can't eyeball a cache file in a plain text editor.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
DEFAULT_TIMEOUT_SECONDS = 10

CACHE_DIR = Path(__file__).parent / "data"
DEFAULT_CACHE_TTL = timedelta(hours=24)

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


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    """Return the cache file path for a given ticker/period/interval combo."""
    return CACHE_DIR / f"{ticker}_{period}_{interval}.parquet"


def _read_cache(path: Path, ttl: timedelta) -> pd.DataFrame | None:
    """Return the cached DataFrame at path if it exists and is within ttl."""
    if not path.exists():
        return None
    age = timedelta(seconds=time.time() - path.stat().st_mtime)
    if age > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # corrupt/partial cache file: treat as a miss
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _write_cache(data: pd.DataFrame, path: Path) -> None:
    """Write data to path as Parquet, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)


def fetch_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    use_cache: bool = True,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
) -> pd.DataFrame:
    """Fetch OHLCV price history for a single ticker.

    Args:
        ticker: Symbol to fetch, e.g. "AAPL".
        period: Yahoo Finance range string, e.g. "6mo", "1y", "5y".
        interval: Bar size, e.g. "1d", "1wk".
        timeout: Request timeout in seconds.
        use_cache: If True, serve from and save to the on-disk cache under
            CACHE_DIR instead of hitting Yahoo Finance every call.
        cache_ttl: Maximum age of a cached file before it's considered
            stale and re-fetched.

    Returns:
        DataFrame indexed by timestamp with Open/High/Low/Close/Volume
        columns.

    Raises:
        TickerFetchError: On network failure, an unknown/invalid ticker,
            or a response that doesn't match the expected shape.
    """
    cache_path = _cache_path(ticker, period, interval)
    if use_cache:
        cached = _read_cache(cache_path, cache_ttl)
        if cached is not None:
            logger.info("Using cached data for %s (%s)", ticker, cache_path.name)
            return cached

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

    if use_cache:
        _write_cache(data, cache_path)

    return data


def fetch_portfolio(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
    use_cache: bool = True,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers.

    A failure on any single ticker (bad symbol, network error, rate limit)
    is logged and skipped rather than raised, so one bad ticker can't crash
    a batch fetch for the whole portfolio.

    Args:
        tickers: Symbols to fetch.
        period: Yahoo Finance range string, e.g. "6mo", "1y", "5y".
        interval: Bar size, e.g. "1d", "1wk".
        use_cache: If True, serve from and save to the on-disk cache under
            CACHE_DIR instead of hitting Yahoo Finance every call.
        cache_ttl: Maximum age of a cached file before it's considered
            stale and re-fetched.

    Returns:
        Mapping of ticker to its price history DataFrame. Tickers that
        failed to fetch are omitted.
    """
    history: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            history[ticker] = fetch_price_history(
                ticker, period, interval, use_cache=use_cache, cache_ttl=cache_ttl
            )
        except TickerFetchError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    portfolio_history = fetch_portfolio(PORTFOLIO_TICKERS)
    for symbol, df in portfolio_history.items():
        print(f"\n{symbol}")
        print(df.tail())
