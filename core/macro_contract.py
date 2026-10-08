"""Macroeconomic snapshot contract (Sprint N3) — stable entry point.

Thin, documented wrapper over `core.macro_adapter.build_macro_snapshot`.
All status semantics documented in the MacroSnapshot dataclass and enforced
at the orchestrator level: missing series degrades confidence, no silent
neutrality. FRED API key is required; without it, an explicit UNAVAILABLE
contract is returned (never empty/neutral zero-fill).
"""

from __future__ import annotations

from core.macro_adapter import build_macro_snapshot


def fetch_macro_snapshot(ticker: str, as_of: str) -> dict:
    """Return the point-in-time macroeconomic context snapshot for a ticker.

    Thin, stable entry point over `core.macro_adapter.build_macro_snapshot`
    (Sprint N3). Callers treat `status` as the source of truth:

    - UNAVAILABLE: no provider key configured (explicit placeholder), the provider
      request failed, or no series data was returned. Macro context is never
      derived from price, RSI, volume, or any other technical indicator — that
      would present fabricated data as if it were real economic coverage.
    - OK: all series were ingested, PIT-filtered, and aggregated inside the
      point-in-time window. Risk regime, sector sensitivities, and per-series
      contributions are present.
    - INCOMPLETE: some series are missing or unavailable; confidence is degraded
      by the missing-series penalty per series (MACRO_MISSING_SERIES_PENALTY).
    - INVALID: the provider payload violated the timestamp policy (future-dated
      or unparseable publication times).
    """
    return build_macro_snapshot(ticker, as_of)
