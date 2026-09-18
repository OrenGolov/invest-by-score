"""CI drift gate for the C3 market context.

Proves, on every push, that the context contract C2 depends on holds:

1. every declared index proxy resolves into a frame;
2. VIX and the 10Y are REFERENCED from the macro snapshot and NEVER fetched
   as price bars — a second Yahoo-sourced copy would fork the truth for a
   quantity the macro registry already owns with vintage guarantees (W5);
3. a degraded macro snapshot yields NO reading rather than a neutral
   substitute;
4. an unmapped ticker yields NO sector and NO industry benchmark — a guessed
   sector produces confident, wrong sector-relative strength;
5. every mapped sector has an ETF, and no ETF is claimed by two sectors;
6. the context actually unblocks C2: relative strength is computable with it
   and None without it;
7. a provider failure degrades rather than raising.

Synthetic only, so it runs in well under a second and needs no network.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.chart_features import compute_chart_features  # noqa: E402
from core.config import (  # noqa: E402
    CONTEXT_INDEX_PROXIES,
    CONTEXT_SECTOR_ETFS,
)
from core.macro_registry import SYMBOL_TO_SECTOR  # noqa: E402
from core.market_context import (  # noqa: E402
    STATUS_INCOMPLETE,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    benchmark_frame,
    build_market_context,
    context_problems,
    sector_frame,
)


def _frame(closes, start="2026-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(closes),
        },
        index=index,
    )


def _rising(n=120, base=100.0, step=1.0):
    return _frame([base + step * i for i in range(n)])


def _ok_macro():
    return {"status": "OK", "series_values": {"vix": 17.5, "10y_yield": 4.2}}


def main() -> int:
    failures: list[str] = []
    requested: list[str] = []

    def recording(symbol, period, interval):
        requested.append(symbol)
        return _rising()

    context = build_market_context("NVDA", "2026-09-15", fetcher=recording, macro_snapshot=_ok_macro())

    # 1. every proxy resolves
    for name in CONTEXT_INDEX_PROXIES:
        if name not in context["frames"]:
            failures.append(f"index proxy {name!r} produced no frame")
    if context["status"] != STATUS_OK:
        failures.append(f"a healthy context did not read OK (got {context['status']})")
    for problem in context_problems(context):
        failures.append(f"contract problem: {problem}")

    # 2. VIX/10Y are never price-fetched
    for symbol in requested:
        if "VIX" in symbol.upper() or "TNX" in symbol.upper():
            failures.append(
                f"{symbol!r} was fetched as a price series — VIX and the 10Y are "
                f"macro series (N3) and must be referenced, not re-sourced (W5)"
            )
    if context["macro_readings"].get("vix") != 17.5:
        failures.append("VIX was not read from the macro snapshot")

    # 3. a degraded macro snapshot substitutes nothing
    degraded = build_market_context(
        "NVDA", "2026-09-15", fetcher=recording,
        macro_snapshot={"status": "INCOMPLETE", "series_values": {"vix": 17.5, "10y_yield": 4.2}},
    )
    if degraded["macro_readings"].get("vix") is not None:
        failures.append(
            "a degraded macro snapshot still produced a VIX reading — a non-OK "
            "contract must yield no value, not a plausible-looking one"
        )

    # 4. an unmapped ticker gets nothing guessed
    unmapped = build_market_context("ZZZZ", "2026-09-15", fetcher=recording, macro_snapshot=_ok_macro())
    if unmapped["sector"] is not None or unmapped["industry_benchmark"] is not None:
        failures.append(
            "an unmapped ticker was assigned a sector — a guessed sector yields "
            "confident, wrong sector-relative strength"
        )
    if unmapped["status"] != STATUS_INCOMPLETE:
        failures.append(f"an unmapped ticker must be INCOMPLETE (got {unmapped['status']})")

    # 5. sector table integrity
    for symbol, sector in SYMBOL_TO_SECTOR.items():
        if sector not in CONTEXT_SECTOR_ETFS:
            failures.append(f"{symbol!r} maps to sector {sector!r}, which has no ETF")
    etfs = list(CONTEXT_SECTOR_ETFS.values())
    if len(etfs) != len(set(etfs)):
        failures.append("a sector ETF is claimed by two sectors")

    # 6. C3 actually unblocks C2
    stock = _rising(n=120, step=2.0)
    flat = _frame([100.0] * 120)

    def flat_market(symbol, period, interval):
        return flat if symbol == CONTEXT_INDEX_PROXIES["sp500"] else _rising()

    ctx = build_market_context("NVDA", "2026-09-15", fetcher=flat_market, macro_snapshot=_ok_macro())
    with_context = compute_chart_features(stock, benchmark_frame=benchmark_frame(ctx))
    without_context = compute_chart_features(stock)
    if with_context["features"]["relative_strength_60d"] is None:
        failures.append("relative strength is still None WITH a context — C3 did not unblock C2")
    if without_context["features"]["relative_strength_60d"] is not None:
        failures.append("relative strength resolved WITHOUT a benchmark — C2 must refuse")
    if sector_frame(ctx) is None:
        failures.append("a mapped holding produced no sector frame")

    # 7. provider failure degrades rather than raising
    def failing(symbol, period, interval):
        raise RuntimeError("provider down")

    context_logger = logging.getLogger("core.market_context")
    previous = context_logger.level
    context_logger.setLevel(logging.ERROR)
    try:
        dead = build_market_context("NVDA", "2026-09-15", fetcher=failing, macro_snapshot=_ok_macro())
    finally:
        context_logger.setLevel(previous)
    if dead["status"] != STATUS_UNAVAILABLE:
        failures.append(f"a total provider failure must read UNAVAILABLE (got {dead['status']})")

    if failures:
        print("C3 market-context gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C3 market-context gate OK:")
    print(f"  all {len(CONTEXT_INDEX_PROXIES)} index proxies resolve; "
          f"{len(CONTEXT_SECTOR_ETFS)} sector ETFs mapped, none double-claimed.")
    print("  VIX and the 10Y referenced from the macro snapshot, never price-fetched (W5).")
    print("  a degraded macro contract yields no reading rather than a neutral substitute.")
    print("  an unmapped ticker gets no sector and no industry benchmark.")
    print("  C2 relative strength resolves WITH a context and stays None without one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
