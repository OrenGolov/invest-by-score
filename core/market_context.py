"""Market context features (Sprint C3) — the benchmark C2 was waiting on.

`build_market_context(ticker, as_of, fetcher=None, macro_snapshot=None)` assembles
the frames and readings that let a stock be interpreted against the market it
trades in, rather than in isolation:

    S&P 500 / Nasdaq 100 / Russell 2000 / USD   -> price bars (ETF proxies)
    VIX / 10Y yield                             -> REFERENCED from the macro snapshot
    sector ETF                                  -> selected from the holding's sector
    industry benchmark                          -> the sector ETF (see below)

**Why this sprint exists.** "+4% while the S&P did +4%" and "+4% while the S&P
did -2%" are the same stock return and completely different evidence. Only the
second is relative strength. C2 registered `relative_strength_60d` with an
INJECTED benchmark and no way to obtain one; C3 is the producer that supplies it.

**VIX and the 10Y are not re-fetched.** Both are already registered macro series
(FRED, vintage-aware, publication-time gated) from N3. Pulling them again as
Yahoo price bars would create a second source of truth for the same quantity —
the split-brain the master context forbids (W5). The context block REFERENCES
the macro snapshot, and says so in `macro_reference_status` when that snapshot
is not OK.

**ETF proxies, not raw index levels.** An index level has no volume and no
tradable history; the ETF shares the same provider contract, split handling and
cache path as every other ticker in the system, so one price truth covers the
whole surface.

**An unmapped ticker gets no sector, not a guessed one.** Funds have no single
sector (and a sector ETF is not a benchmark for itself), and an unknown small-cap
is unknown. Both are reported as `sector: None` with a reason, because a wrong
sector benchmark produces confident, wrong relative strength — worse than none.

**PIT by delegation.** Like C2, this module does no as_of filtering of its own.
It fetches and hands back frames; the caller filters through
`core.timeframes.eligible_bars`, so there is exactly one notion of "now".
"""

from __future__ import annotations

import logging

import pandas as pd

from core.config import (
    CONTEXT_CONTRACT_VERSION,
    CONTEXT_DEFAULT_BENCHMARK,
    CONTEXT_INDEX_PROXIES,
    CONTEXT_MACRO_REFERENCES,
    CONTEXT_PIPELINE_VERSION,
    CONTEXT_RETURN_WINDOW,
    CONTEXT_SECTOR_ETFS,
    CONTEXT_SOURCE_ID,
)
from core.macro_registry import get_symbol_sector

LOGGER = logging.getLogger("core.market_context")

STATUS_OK = "OK"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_UNAVAILABLE = "UNAVAILABLE"

_STATUS_SEVERITY = {STATUS_OK: 0, STATUS_INCOMPLETE: 1, STATUS_UNAVAILABLE: 2}


class MarketContextError(ValueError):
    """Raised when a market-context request is structurally invalid."""


def sector_for(ticker: str) -> str | None:
    """The holding's GICS sector, or None when it cannot be established.

    None is a real answer here. A fund has no single sector, and a sector ETF
    is not a benchmark for itself; an unmapped small-cap is simply unknown. A
    guessed sector produces confident, wrong sector-relative strength.
    """
    return get_symbol_sector(str(ticker or "").upper())


def sector_etf_for(ticker: str) -> str | None:
    """The sector ETF that acts as the holding's industry benchmark."""
    sector = sector_for(ticker)
    return CONTEXT_SECTOR_ETFS.get(sector) if sector else None


def window_return(
    frame: pd.DataFrame | None,
    window: int = CONTEXT_RETURN_WINDOW,
    as_of=None,
) -> float | None:
    """Trailing return over the window, or None when history is short.

    Short history yields None rather than a partial-window return, which would
    silently compare different periods across context series.

    `as_of` truncates the frame before measuring. Without it the return is read
    off the frame's TAIL, which for a historical request is the future: at
    as_of=2025-09-01 a frame running to 2026-02-15 reported +20.1% of market
    return that had not happened yet, and C5 carried that into a sequence
    labelled as_of_t0. The parameter is not optional in practice -- the caller
    always has the timestamp -- but it defaults to None so a caller holding an
    already-filtered frame is not forced to pass it twice.
    """
    if frame is None or "Close" not in getattr(frame, "columns", []):
        return None
    if as_of is not None:
        target = pd.Timestamp(as_of)
        if target.tzinfo is not None:
            target = target.tz_localize(None)
        index = pd.to_datetime(frame.index)
        frame = frame.loc[index <= target]
        if frame.empty:
            return None
    closes = frame.dropna(subset=["Close"])["Close"]
    if len(closes) < window + 1:
        return None
    start = float(closes.iloc[-(window + 1)])
    end = float(closes.iloc[-1])
    if start <= 0:
        return None
    return round(end / start - 1.0, 6)


def _macro_readings(macro_snapshot: dict | None) -> tuple[dict, str, str]:
    """Pull VIX and the 10Y from the macro snapshot, never from price bars.

    Returns `(readings, status, reason)`. A non-OK macro snapshot yields no
    values at all rather than a neutral substitute — the same fail-closed rule
    the macro agent itself applies.
    """
    readings: dict[str, float | None] = {name: None for name in CONTEXT_MACRO_REFERENCES}
    if not macro_snapshot:
        return readings, STATUS_UNAVAILABLE, (
            "no macro snapshot supplied; VIX and the 10Y are macro series (N3) "
            "and are never re-fetched as price bars"
        )
    status = str(macro_snapshot.get("status", STATUS_UNAVAILABLE))
    if status != STATUS_OK:
        return readings, STATUS_UNAVAILABLE, (
            f"macro snapshot is {status}; VIX and the 10Y yield no value rather "
            f"than a neutral substitute"
        )
    values = macro_snapshot.get("series_values") or {}
    for field, macro_key in CONTEXT_MACRO_REFERENCES.items():
        value = values.get(macro_key)
        readings[field] = float(value) if value is not None else None
    missing = [name for name, value in readings.items() if value is None]
    if missing:
        return readings, STATUS_INCOMPLETE, f"macro series absent: {', '.join(sorted(missing))}"
    return readings, STATUS_OK, ""


def build_market_context(
    ticker: str,
    as_of,
    fetcher=None,
    macro_snapshot: dict | None = None,
    benchmark: str = CONTEXT_DEFAULT_BENCHMARK,
) -> dict:
    """Assemble the market context for one holding at one as_of.

    `fetcher(symbol, period, interval) -> DataFrame` is injectable so the
    contract can be tested without a network round trip.

    The returned `frames` are RAW provider bars, deliberately unfiltered: the
    caller applies `core.timeframes.eligible_bars` so one filter governs the
    whole system. `returns` are computed on those same raw frames and are
    therefore only as PIT-correct as the caller's own filtering — they exist
    for reporting, while `frames` is what feeds C2.
    """
    if not str(ticker or "").strip():
        raise MarketContextError("a ticker is required")
    if benchmark not in CONTEXT_INDEX_PROXIES:
        raise MarketContextError(
            f"unknown benchmark {benchmark!r} (known: {sorted(CONTEXT_INDEX_PROXIES)})"
        )

    target = pd.Timestamp(as_of)
    if fetcher is None:
        from fetch_data import fetch_price_history

        def fetcher(symbol, period, interval):  # noqa: ANN001 - thin adapter
            return fetch_price_history(symbol, period=period, interval=interval)

    symbol = str(ticker).upper()
    sector = sector_for(symbol)
    sector_etf = sector_etf_for(symbol)

    wanted: dict[str, str] = dict(CONTEXT_INDEX_PROXIES)
    if sector_etf:
        wanted["sector"] = sector_etf

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for name, proxy in wanted.items():
        try:
            frame = fetcher(proxy, "1y", "1d")
        except Exception as exc:
            # ABSORBS: a provider failure for ONE context proxy. The others are
            # still usable, so the loop records this one as failed and continues
            # — a missing VIX must not cost the S&P context too. The failure is
            # named in `failed`, so a consumer can tell partial context from
            # complete.
            LOGGER.warning("market context fetch failed for %s (%s): %s", name, proxy, exc)
            failed.append(name)
            continue
        if frame is None or getattr(frame, "empty", True):
            failed.append(name)
            continue
        frames[name] = frame

    readings, macro_status, macro_reason = _macro_readings(macro_snapshot)

    reasons: list[str] = []
    if failed:
        reasons.append(f"context series unavailable: {', '.join(sorted(failed))}")
    if sector is None:
        reasons.append(
            f"{symbol} has no mapped sector, so no industry benchmark was selected; "
            f"a guessed sector would produce confident, wrong sector-relative strength"
        )
    if macro_reason:
        reasons.append(macro_reason)

    if not frames:
        status = STATUS_UNAVAILABLE
    elif failed or sector is None or macro_status != STATUS_OK:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_OK

    return {
        "ticker": symbol,
        "as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "source_id": CONTEXT_SOURCE_ID,
        "calculation_version": CONTEXT_CONTRACT_VERSION,
        "pipeline_version": CONTEXT_PIPELINE_VERSION,
        "benchmark": benchmark,
        "benchmark_symbol": CONTEXT_INDEX_PROXIES[benchmark],
        "sector": sector,
        "sector_etf": sector_etf,
        "industry_benchmark": sector_etf,
        "proxies": {name: wanted[name] for name in sorted(wanted)},
        "frames": frames,
        # Measured AT as_of, not off the frame tail. `frames` stay raw so the
        # caller can apply C1's eligible_bars, but a return is a number that
        # leaves this module, so it is filtered here or it is a leak.
        "returns": {
            name: window_return(frames.get(name), as_of=target)
            for name in sorted(wanted)
        },
        "returns_as_of": target.strftime("%Y-%m-%d %H:%M:%S"),
        "macro_readings": readings,
        "macro_reference_status": macro_status,
        "unavailable": sorted(failed),
        "reason": "; ".join(reasons),
    }


def benchmark_frame(context: dict) -> pd.DataFrame | None:
    """The broad-market frame C2's relative strength consumes."""
    return (context.get("frames") or {}).get(context.get("benchmark"))


def sector_frame(context: dict) -> pd.DataFrame | None:
    """The industry-benchmark frame, or None when no sector was established."""
    return (context.get("frames") or {}).get("sector")


def context_problems(context: dict) -> list[str]:
    """Validate a built context against the C3 contract."""
    problems: list[str] = []
    if not isinstance(context, dict):
        return ["context must be a dict"]
    for field in ("ticker", "as_of", "status", "calculation_version", "benchmark"):
        if not context.get(field):
            problems.append(f"context field {field!r} missing/empty")
    if context.get("status") == STATUS_OK:
        if benchmark_frame(context) is None:
            problems.append("context is OK but carries no benchmark frame")
        if context.get("sector") and context.get("industry_benchmark") is None:
            problems.append("a mapped sector must yield an industry benchmark")
    if context.get("sector") is None and context.get("industry_benchmark") is not None:
        problems.append("an industry benchmark was selected without a sector")
    readings = context.get("macro_readings") or {}
    if context.get("macro_reference_status") == STATUS_OK:
        for name, value in readings.items():
            if value is None:
                problems.append(f"macro reference {name!r} is OK but carries no value")
    return problems
