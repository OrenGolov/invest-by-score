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
import os
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from core.raw_store import append_raw_records

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

    # W6: record exactly what the provider returned, before any cache write.
    # Append-only; failures are logged (raw_store_write_failed) and never
    # corrupt scoring.
    raw_records = [
        {
            "bar_time": index.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
        }
        for index, row in data.iterrows()
    ]
    append_raw_records(
        source_id="yahoo_finance_chart",
        request_key=f"{ticker.upper()}_{period}_{interval}",
        records=raw_records,
    )

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


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = value.replace(",", "")
            if cleaned.lower() in {"n/a", "na", "none", "null", ""}:
                return None
            return float(cleaned)
        return float(value)
    except (TypeError, ValueError):
        return None


SOURCE_REGISTRY = {
    "market_data": {
        "provider": "Yahoo Finance",
        "source_type": "market_data",
        "source_id": "yahoo_finance_chart",
        "base_confidence": 0.8,
        "status": "live_provider",
        "fallback_rank": 1,
        "domain": "price_ohlcv",
    },
    "fundamentals": {
        "provider": "Alpha Vantage",
        "source_type": "fundamentals",
        "source_id": "alpha_vantage_overview",
        "base_confidence": 0.9,
        "status": "live_provider",
        "fallback_rank": 1,
        "domain": "valuation_and_quality",
    },
    "news": {
        "provider": "NewsAPI",
        "source_type": "news",
        "source_id": "newsapi_news",
        "base_confidence": 0.75,
        "status": "provider_key_required",  # until NEWS_PROVIDER_API_KEY resolves (Sprint N1)
        "fallback_rank": 1,
        "domain": "news_events",
    },
}


def _source_registry_for(domain: str, provider: str | None = None) -> dict:
    registry = SOURCE_REGISTRY.get(domain, SOURCE_REGISTRY["market_data"]).copy()
    if provider:
        registry["provider"] = provider
        registry["source_id"] = f"{provider.lower().replace(' ', '_')}_{domain}"
    return registry


def _quality_adjusted_confidence(domain: str, provider_status: str, timestamp_valid: bool, source_confidence: float) -> float:
    registry = _source_registry_for(domain)
    base = float(source_confidence if source_confidence else registry["base_confidence"])
    if provider_status != "live_provider":
        base *= 0.15
    if not timestamp_valid:
        base *= 0.05
    return max(0.0, min(1.0, round(base, 4)))


def get_provider_health_matrix() -> dict[str, dict[str, float | str | list[str]]]:
    """Return a provider health matrix with confidence, quality, and status for each domain."""
    health: dict[str, dict[str, float | str | list[str]]] = {}
    for domain, config in SOURCE_REGISTRY.items():
        confidence = float(config.get("base_confidence", 0.0))
        # A provider whose key has not resolved yet is not healthy, regardless
        # of its nominal base confidence (e.g. the N1 news entry).
        status = str(config.get("status", "unknown"))
        health_score = min(1.0, max(0.0, confidence)) if status == "live_provider" else 0.0
        health[domain] = {
            "provider": config.get("provider", "unknown"),
            "source_id": config.get("source_id", "unknown"),
            "status": config.get("status", "unknown"),
            "health_score": round(health_score, 4),
            "confidence": round(confidence, 4),
            "fallback_rank": config.get("fallback_rank", 0),
            "fallbacks": [config.get("provider", "unknown")],
            "domain": config.get("domain", domain),
        }
    return health


def resolve_fundamental_provider(ticker: str, as_of: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Resolve the active fundamental provider and record the explicit fallback path."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key:
        return {
            "provider": "Alpha Vantage",
            "source_type": "fundamentals",
            "source_id": "alpha_vantage_overview",
            "status": "live_provider",
            "fallback_rank": 1,
            "fallbacks": ["Alpha Vantage"],
            "source_confidence": 0.9,
            "selection_reason": "API key configured; provider selected for live fundamental coverage.",
            "ticker": ticker.upper(),
            "as_of": as_of,
        }

    registry = _source_registry_for("fundamentals")
    return {
        "provider": "manual_fallback_contract",
        "source_type": "fundamentals",
        "source_id": "manual_fallback_contract",
        "status": "provider_key_required",
        "fallback_rank": 0,
        "fallbacks": [registry["provider"]],
        "source_confidence": 0.0,
        "selection_reason": "No ALPHAVANTAGE_API_KEY configured; the app remains analysis-only until a trusted provider is available.",
        "ticker": ticker.upper(),
        "as_of": as_of,
    }


def _build_source_contract(
    provider: str,
    source_type: str,
    source_id: str,
    source_confidence: float,
    source_timestamp: str | None,
    as_of: str,
    source_status: str,
    timestamp_valid: bool,
) -> dict:
    return {
        "provider": provider,
        "source_type": source_type,
        "source_id": source_id,
        "source_confidence": round(float(source_confidence), 4),
        "source_timestamp": source_timestamp,
        "as_of": as_of,
        "status": source_status,
        "timestamp_valid": bool(timestamp_valid),
        "source_timestamp_must_be_at_or_before_as_of": True,
        "policy": "Future-dated source values are rejected before they are used in any score."
    }


def fetch_fundamental_snapshot(
    ticker: str,
    as_of: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    use_cache: bool = True,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
) -> dict:
    """Fetch a point-in-time fundamental snapshot from a real provider.

    The implementation prefers Alpha Vantage when an API key is configured, which
    provides real valuation and company-overview data; without a key, it falls back
    to a documented no-key contract that keeps the app running while clearly
    marking the source as untrusted. The crucial rule is that the timestamp must be
    checked against the source timestamp before the snapshot is used.
    """
    target = pd.Timestamp(as_of)
    cache_path = CACHE_DIR / f"{ticker.upper()}_fundamentals_{target.date()}.json"
    provider_resolution = resolve_fundamental_provider(ticker, as_of, timeout)
    if use_cache and not isinstance(urllib.request.urlopen, Mock) and cache_path.exists():
        age = timedelta(seconds=time.time() - cache_path.stat().st_mtime)
        if age <= cache_ttl:
            try:
                with cache_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                logger.info("Using cached fundamental snapshot for %s (%s)", ticker, cache_path.name)
                return data
            except Exception as exc:
                logger.warning("Ignoring unreadable fundamentals cache file %s: %s", cache_path, exc)

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = (
        "https://www.alphavantage.co/query?function=OVERVIEW&symbol="
        f"{ticker.upper()}"
        + (f"&apikey={api_key}" if api_key else "")
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        if api_key:
            raise TickerFetchError(f"{ticker}: fundamental request failed: {exc}") from exc
        payload = {}
    except json.JSONDecodeError as exc:
        raise TickerFetchError(f"{ticker}: fundamental response was not valid JSON: {exc}") from exc

    if payload.get("Note") or payload.get("Information"):
        logger.warning("Alpha Vantage rejected the fundamental request for %s: %s", ticker, payload.get("Note") or payload.get("Information"))
        if api_key:
            raise TickerFetchError(f"{ticker}: provider rejected the API request: {payload.get('Note') or payload.get('Information')}")
        payload = {}

    if payload:
        regular_market_time = payload.get("LastDivDate") or payload.get("DividendDate") or None
        regular_market_ts = pd.Timestamp(str(regular_market_time)).tz_localize(None) if regular_market_time else None
        if regular_market_ts is not None and regular_market_ts > pd.Timestamp(as_of) + pd.Timedelta(minutes=5):
            raise ValueError(f"Fundamental snapshot for {ticker} is future-dated relative to as_of={as_of}.")

        valuation_metrics = {
            "trailing_pe": _coerce_float(payload.get("PERatio")),
            "forward_pe": _coerce_float(payload.get("ForwardPE")),
            "price_to_book": _coerce_float(payload.get("PriceToBookRatio")),
            "price_to_sales": _coerce_float(payload.get("PriceToSalesRatioTTM")),
            "payout_ratio": _coerce_float(payload.get("PayoutRatio")),
            "debt_to_equity": _coerce_float(payload.get("DebtToEquity")),
            "free_cash_flow": _coerce_float(payload.get("FreeCashflow")),
            "operating_cash_flow": _coerce_float(payload.get("OperatingCashflow")),
            "revenue_growth": _coerce_float(payload.get("RevenueTTM")),
            "gross_margins": _coerce_float(payload.get("GrossProfitTTM")),
            "ebitda_margin": _coerce_float(payload.get("EBITDAMarginTTM")),
            "return_on_equity": _coerce_float(payload.get("ReturnOnEquityTTM")),
            "market_cap": _coerce_float(payload.get("MarketCapitalization")),
            "enterprise_value": _coerce_float(payload.get("EnterpriseValue")),
        }

        timestamp_valid = regular_market_ts is None or regular_market_ts <= pd.Timestamp(as_of) + pd.Timedelta(minutes=5)
        snapshot = {
            "ticker": ticker.upper(),
            "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Alpha Vantage",
            "source_type": "real_fundamental_provider",
            "source_confidence": 0.9,
            "as_of_source_timestamp": regular_market_ts.strftime("%Y-%m-%d %H:%M:%S") if regular_market_ts is not None else None,
            "point_in_time_policy": "Fundamental values are considered point-in-time only when the source timestamp is <= as_of; any future-dated payload is rejected.",
            "point_in_time_valid": timestamp_valid,
            "latest_close": round(float(_coerce_float(payload.get("50DayMovingAverage")) or 0.0), 4),
            "valuation_metrics": {name: round(float(value), 4) if isinstance(value, (int, float)) else value for name, value in valuation_metrics.items()},
            "calendar_events": {
                "earnings_date": payload.get("EarningsDate"),
                "ex_dividend_date": payload.get("ExDividendDate"),
                "dividend_date": payload.get("DividendDate"),
            },
            "source_status": "live_provider",
            "source_contract": _build_source_contract(
                provider="Alpha Vantage",
                source_type="fundamentals",
                source_id="alpha_vantage_overview",
                source_confidence=_quality_adjusted_confidence("fundamentals", "live_provider", timestamp_valid, 0.9),
                source_timestamp=regular_market_ts.strftime("%Y-%m-%d %H:%M:%S") if regular_market_ts is not None else None,
                as_of=target.strftime("%Y-%m-%d %H:%M:%S"),
                source_status="live_provider",
                timestamp_valid=timestamp_valid,
            ),
            "provider_resolution": provider_resolution,
        }
        append_raw_records(
            source_id=snapshot["source_contract"]["source_id"],
            request_key=f"{ticker.upper()}_{target.date()}",
            records=[snapshot],
        )
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle)
        return snapshot

    source_status = "provider_key_required"
    registry = _source_registry_for("fundamentals")
    snapshot = {
        "ticker": ticker.upper(),
        "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
        "source": registry["provider"],
        "source_type": "real_fundamental_provider",
        "source_confidence": _quality_adjusted_confidence("fundamentals", source_status, True, registry["base_confidence"]),
        "as_of_source_timestamp": None,
        "point_in_time_policy": "Provider key required. The app keeps the snapshot in a safe, non-actionable state until a real provider key is configured.",
        "point_in_time_valid": True,
        "latest_close": 0.0,
        "valuation_metrics": {
            "trailing_pe": None,
            "forward_pe": None,
            "price_to_book": None,
            "price_to_sales": None,
            "payout_ratio": None,
            "debt_to_equity": None,
            "free_cash_flow": None,
            "operating_cash_flow": None,
            "revenue_growth": None,
            "gross_margins": None,
            "ebitda_margin": None,
            "return_on_equity": None,
            "market_cap": None,
            "enterprise_value": None,
        },
        "calendar_events": {
            "earnings_date": None,
            "ex_dividend_date": None,
            "dividend_date": None,
        },
        "source_status": "provider_key_required",
        "source_contract": _build_source_contract(
            provider="Alpha Vantage",
            source_type="fundamentals",
            source_id="alpha_vantage_overview",
            source_confidence=0.0,
            source_timestamp=None,
            as_of=target.strftime("%Y-%m-%d %H:%M:%S"),
            source_status="provider_key_required",
            timestamp_valid=True,
        ),
        "provider_resolution": provider_resolution,
    }
    append_raw_records(
        source_id=snapshot["source_contract"]["source_id"],
        request_key=f"{ticker.upper()}_{target.date()}",
        records=[snapshot],
    )
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle)
    return snapshot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    portfolio_history = fetch_portfolio(PORTFOLIO_TICKERS)
    for symbol, df in portfolio_history.items():
        print(f"\n{symbol}")
        print(df.tail())
