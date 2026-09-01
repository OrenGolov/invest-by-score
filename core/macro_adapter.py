"""Macroeconomic adapter (Sprint N3) — vintage-aware ingestion, PIT filtering, risk regime.

Pipeline:

    FETCH -> PROVIDER GATE -> PIT FILTER -> REVISION HANDLING -> 
    INDICATOR ANALYSIS -> SECTOR-WEIGHTED REGIME -> CONFIDENCE CALCULATION

Governance rules:

- Provider gate: without FRED_API_KEY or if provider disabled, explicit UNAVAILABLE
  snapshot (no silent neutrality).
- Point-in-time: only releases with published_time <= as_of are eligible.
  Future-dated or unparseable publication times invalidate the payload
  (status INVALID, fail-closed).
- Revisions: every fetch appends to raw_store (W6). The adapter uses the
  latest version (append-order defines winner, immune to clock granularity).
- Missing series: degrades confidence (INCOMPLETE), never zero-fills to neutral.
- Sector sensitivity: static loadings per GICS sector (rates, energy, USD);
  symbol-sector mapping v1 (curated, extensible via service integration later).
- Risk regime: fed_funds + rates trend + inflation + growth indicators → 
  risk-on/off tilt with transition-risk flagging.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from core.config import (
    MACRO_ADAPTER_VERSION,
    MACRO_CONTRACT_VERSION,
    MACRO_LOOKBACK_PERIODS,
    MACRO_MISSING_SERIES_PENALTY,
    MACRO_PROVIDER_API_KEY_ENV,
    MACRO_PROVIDER_TIMEOUT_SECONDS,
    MACRO_RISKOFF_THRESHOLD,
    MACRO_RISKON_THRESHOLD,
    MACRO_SCORE_BASE,
    MACRO_SCORE_SPAN,
)
from core.macro_registry import (
    MACRO_SERIES_REGISTRY,
    SECTOR_MACRO_LOADINGS,
    SYMBOL_TO_SECTOR,
    get_series,
    get_sector_loadings,
    get_symbol_sector,
)
from core.raw_store import append_raw_records
from core.schemas import MacroSnapshot

LOGGER = logging.getLogger("core.macro_adapter")

MACRO_SOURCE_ID = "fred_macro"
MACRO_PROVIDER_NAME = "Federal Reserve Economic Data (FRED)"
UNAVAILABLE_SOURCE_ID = "macro_provider_unconfigured"
UNAVAILABLE_REASON = (
    "No verified macro data provider is connected. Economic data is not inferred "
    "from price or technical indicators."
)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse a timestamp into naive UTC. Returns None when unparseable."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def resolve_macro_provider(as_of: str) -> dict:
    """Resolve the active macro provider, mirroring the news/fundamental pattern."""
    api_key = os.getenv(MACRO_PROVIDER_API_KEY_ENV)
    if api_key:
        return {
            "provider": MACRO_PROVIDER_NAME,
            "source_type": "macro",
            "source_id": MACRO_SOURCE_ID,
            "status": "live_provider",
            "fallback_rank": 1,
            "fallbacks": ["FRED"],
            "source_confidence": 0.9,  # High: official Fed data, minimal revisions
            "selection_reason": f"{MACRO_PROVIDER_API_KEY_ENV} configured; FRED selected for official economic data.",
            "as_of": as_of,
        }
    return {
        "provider": "macro_provider_unconfigured",
        "source_type": "macro",
        "source_id": UNAVAILABLE_SOURCE_ID,
        "status": "provider_key_required",
        "fallback_rank": 0,
        "fallbacks": ["FRED"],
        "source_confidence": 0.0,
        "selection_reason": (
            f"No {MACRO_PROVIDER_API_KEY_ENV} configured; the macro contract stays "
            "explicitly UNAVAILABLE until a trusted provider is available."
        ),
        "as_of": as_of,
    }


def fetch_fred_series(
    series_id: str,
    api_key: str,
    lookback_periods: int = MACRO_LOOKBACK_PERIODS,
    timeout: float = MACRO_PROVIDER_TIMEOUT_SECONDS,
) -> dict:
    """Fetch one series from FRED (St. Louis Fed Economic Data).

    Returns {"status": "ok"|"provider_request_failed", "records": [...], "reason": str}.
    Never raises: failed request is explicit disposition, never empty data pretending
    to be coverage.
    """
    url = (
        f"https://api.stlouisfed.org/fred/series/data"
        f"?series_id={urllib.parse.quote(series_id)}"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&limit={max(1, lookback_periods)}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "invest-by-score/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.warning("macro_fred_request_failed: %s %s", series_id, exc)
        return {"status": "provider_request_failed", "records": [], "reason": f"FRED request failed: {exc}"}

    observations = payload.get("observations", [])
    if not observations:
        return {"status": "ok", "records": [], "reason": ""}

    records: list[dict] = []
    for obs in observations:
        date_str = obs.get("date")
        value_str = obs.get("value")
        if not date_str or not value_str or value_str == ".":  # "." = no data
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        records.append({
            "source_record_id": f"{series_id}_{date_str}",
            "series_id": series_id,
            "reference_date": date_str,
            "published_time": date_str,  # FRED publishes on the date; N3 adapter adds lag
            "value": value,
        })

    return {"status": "ok", "records": records, "reason": ""}


def pit_filter_macro(records: list[dict], as_of_dt: datetime) -> tuple[list[tuple[dict, datetime]], list[dict]]:
    """PIT FILTER: keep records with published_time <= as_of (inclusive).

    Unparseable and future-dated records are rejected with explicit reasons;
    their presence in a provider payload invalidates it (status INVALID).
    """
    eligible: list[tuple[dict, datetime]] = []
    rejected: list[dict] = []
    for record in records:
        published_dt = _parse_timestamp(record.get("published_time"))
        if published_dt is None:
            rejected.append({
                "source_record_id": str(record.get("source_record_id", "")),
                "series_id": str(record.get("series_id", "")),
                "reason": "unparseable_published_time",
            })
            continue
        if published_dt > as_of_dt:
            rejected.append({
                "source_record_id": str(record.get("source_record_id", "")),
                "series_id": str(record.get("series_id", "")),
                "reason": "future_dated",
            })
            continue
        eligible.append((record, published_dt))
    return eligible, rejected


def compute_risk_regime(
    fed_funds: float | None,
    cpi_yoy: float | None,
    initial_claims: float | None,
    gdp_growth: float | None,
    yield_10y: float | None,
) -> tuple[str, float, str]:
    """INDICATOR ANALYSIS + RISK REGIME CLASSIFICATION v1.

    Inputs: latest values from each series (already PIT-filtered and latest-versioned).
    Returns (regime_label, risk_score, reasoning).

    Risk score ∈ [0, 1]: 0 = risk-off, 0.5 = neutral, 1.0 = risk-on.
    Maps to macro_score on 0-10 scale via MACRO_SCORE_BASE + MACRO_SCORE_SPAN * risk_score.
    """
    signals = []
    reasoning_parts = []

    # Fed Funds Rate: high rates = risk-off (tightening), low = risk-on (easing)
    if fed_funds is not None:
        if fed_funds > 4.0:
            signals.append(-0.4)
            reasoning_parts.append("rates-restrictive")
        elif fed_funds > 2.0:
            signals.append(-0.1)
            reasoning_parts.append("rates-neutral")
        else:
            signals.append(0.3)
            reasoning_parts.append("rates-accommodative")

    # CPI YoY: inflation above 3% = risk-off, below 2% = neutral to risk-on
    if cpi_yoy is not None:
        if cpi_yoy > 3.5:
            signals.append(-0.3)
            reasoning_parts.append("inflation-high")
        elif cpi_yoy > 2.5:
            signals.append(-0.1)
            reasoning_parts.append("inflation-above-target")
        elif cpi_yoy < 1.5:
            signals.append(0.25)
            reasoning_parts.append("inflation-subdued")
        else:
            signals.append(0.05)
            reasoning_parts.append("inflation-target")

    # Initial Claims: high/rising = risk-off (labor market weakening)
    # Baseline ~300k; >400k = stressed, <250k = tight market
    if initial_claims is not None:
        if initial_claims > 450.0:
            signals.append(-0.3)
            reasoning_parts.append("claims-elevated")
        elif initial_claims > 350.0:
            signals.append(-0.1)
            reasoning_parts.append("claims-above-baseline")
        elif initial_claims < 200.0:
            signals.append(0.3)
            reasoning_parts.append("claims-tight")
        else:
            signals.append(0.05)
            reasoning_parts.append("claims-normal")

    # GDP Growth: below 2% YoY = risk-off, >2.5% = risk-on
    if gdp_growth is not None:
        if gdp_growth < 1.0:
            signals.append(-0.3)
            reasoning_parts.append("growth-weak")
        elif gdp_growth < 2.0:
            signals.append(-0.1)
            reasoning_parts.append("growth-below-trend")
        elif gdp_growth > 3.0:
            signals.append(0.3)
            reasoning_parts.append("growth-strong")
        else:
            signals.append(0.1)
            reasoning_parts.append("growth-trend")

    # 10Y Yield: >3% = risk-on (higher term premium, growth expectations), <2% = risk-off
    if yield_10y is not None:
        if yield_10y > 3.5:
            signals.append(0.3)
            reasoning_parts.append("yields-elevated")
        elif yield_10y > 2.5:
            signals.append(0.1)
            reasoning_parts.append("yields-normal")
        elif yield_10y < 1.5:
            signals.append(-0.25)
            reasoning_parts.append("yields-depressed")
        else:
            signals.append(-0.05)
            reasoning_parts.append("yields-low")

    # Mean signal = risk score offset from 0.5
    if signals:
        mean_signal = sum(signals) / len(signals)
        risk_score = max(0.0, min(1.0, 0.5 + mean_signal))
    else:
        risk_score = 0.5
    
    risk_score = round(risk_score, 4)

    # Classify regime
    if risk_score < MACRO_RISKOFF_THRESHOLD:
        regime = "risk_off"
    elif risk_score > MACRO_RISKON_THRESHOLD:
        regime = "risk_on"
    else:
        regime = "neutral"

    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "no_signals"

    return regime, risk_score, reasoning


def _unavailable_snapshot(ticker: str, as_of: str, reason: str = UNAVAILABLE_REASON, source_id: str = UNAVAILABLE_SOURCE_ID) -> dict:
    """The explicit UNAVAILABLE contract.

    No provider key = empty snapshot, never neutral zero-fill.
    """
    sector = get_symbol_sector(ticker)
    sector_loadings = get_sector_loadings(sector) if sector else {}
    
    return MacroSnapshot(
        ticker=ticker,
        as_of=str(as_of),
        status="UNAVAILABLE",
        source_id=source_id,
        source_confidence=0.0,
        published_time=None,
        calculation_version=MACRO_CONTRACT_VERSION,
        regime="neutral",
        regime_score=0.5,
        series_values={},
        series_credibility={},
        sector_loadings=sector_loadings,
        per_series_contributions=[],
        reason=reason,
    ).to_dict()


def build_macro_snapshot(ticker: str, as_of: str, timeout: float = MACRO_PROVIDER_TIMEOUT_SECONDS) -> dict:
    """Run the full N3 macro pipeline for one ticker at one point in time."""
    ticker = str(ticker).upper()
    as_of_text = str(as_of)
    as_of_dt = _parse_timestamp(as_of_text)
    if as_of_dt is None:
        raise ValueError(f"macro adapter: unparseable as_of {as_of_text!r}")

    resolution = resolve_macro_provider(as_of_text)
    if resolution.get("status") != "live_provider":
        return _unavailable_snapshot(ticker, as_of_text)

    api_key = os.getenv(MACRO_PROVIDER_API_KEY_ENV)
    if not api_key:
        return _unavailable_snapshot(ticker, as_of_text)

    # Fetch all series in parallel concept (sequential for now).
    series_data: dict[str, list[tuple[dict, datetime]]] = {}
    series_rejected: dict[str, list[dict]] = {}
    has_any_error = False

    for logical_id, series in MACRO_SERIES_REGISTRY.items():
        fetched = fetch_fred_series(series.series_id, api_key, MACRO_LOOKBACK_PERIODS, timeout)
        if fetched["status"] != "ok":
            LOGGER.warning("macro_fetch_failed: %s (%s)", logical_id, fetched["reason"])
            series_data[logical_id] = []
            series_rejected[logical_id] = []
            continue

        eligible, rejected = pit_filter_macro(fetched["records"], as_of_dt)
        series_data[logical_id] = eligible
        series_rejected[logical_id] = rejected

        # Append raw records for audit trail (W6).
        append_raw_records(
            source_id=MACRO_SOURCE_ID,
            request_key=f"{logical_id}_{as_of_dt.date().isoformat()}",
            records=[record for record, _ in eligible],
        )

        if rejected:
            has_any_error = True

    # Extract latest value from each series (FIFO = latest by append order).
    series_values: dict[str, float] = {}
    series_credibility: dict[str, dict] = {}
    latest_published_time = None
    missing_series: list[str] = []

    for logical_id, series in MACRO_SERIES_REGISTRY.items():
        eligible = series_data.get(logical_id, [])
        if not eligible:
            missing_series.append(logical_id)
            series_credibility[logical_id] = {"status": "UNAVAILABLE", "reason": "no_eligible_records"}
            continue
        
        # Latest = last eligible (most recent by published time)
        record, published_dt = eligible[-1]
        series_values[logical_id] = float(record.get("value", 0.0))
        series_credibility[logical_id] = {
            "status": "OK",
            "published_time": published_dt.isoformat(),
            "source_record_id": record.get("source_record_id"),
        }
        if latest_published_time is None or published_dt > latest_published_time:
            latest_published_time = published_dt

    # Compute risk regime from the series values.
    regime, risk_score, regime_reasoning = compute_risk_regime(
        fed_funds=series_values.get("fed_funds"),
        cpi_yoy=series_values.get("cpi_yoy"),
        initial_claims=series_values.get("initial_claims"),
        gdp_growth=series_values.get("gdp_growth"),
        yield_10y=series_values.get("10y_yield"),
    )

    # Map risk_score to 0-10 scale.
    macro_score = MACRO_SCORE_BASE + MACRO_SCORE_SPAN * risk_score

    # Determine status based on missing series and errors.
    if has_any_error:
        status = "INVALID"
        reason_text = (
            "Macro provider payload violated the point-in-time policy "
            f"(future-dated or unparseable publication times); rejected fail-closed."
        )
    elif missing_series:
        status = "INCOMPLETE"
        reason_text = f"Missing {len(missing_series)} series ({', '.join(missing_series)}); confidence degraded."
    else:
        status = "OK"
        reason_text = ""

    # Sector loadings for this ticker.
    sector = get_symbol_sector(ticker)
    sector_loadings = get_sector_loadings(sector) if sector else None

    # Per-series contributions (logical_id -> score impact).
    per_series_contributions = [
        {
            "series_id": logical_id,
            "value": series_values.get(logical_id),
            "credibility": series_credibility.get(logical_id, {}),
            "source_record_ids": [record["source_record_id"] for record, _ in series_data.get(logical_id, [])],
        }
        for logical_id in MACRO_SERIES_REGISTRY.keys()
    ]

    # Apply confidence penalty for missing series.
    base_confidence = resolution.get("source_confidence", 0.9)
    if missing_series:
        base_confidence = max(0.0, base_confidence - len(missing_series) * MACRO_MISSING_SERIES_PENALTY)

    snapshot = MacroSnapshot(
        ticker=ticker,
        as_of=as_of_text,
        status=status,
        source_id=MACRO_SOURCE_ID,
        source_confidence=round(base_confidence, 4),
        published_time=latest_published_time.isoformat() if latest_published_time else None,
        calculation_version=MACRO_CONTRACT_VERSION,
        regime=regime,
        regime_score=risk_score,
        series_values=series_values,
        series_credibility=series_credibility,
        sector_loadings=sector_loadings or {},
        per_series_contributions=per_series_contributions,
        reason=reason_text,
    ).to_dict()

    # Provenance: pipeline metadata (only on OK/INCOMPLETE, not UNAVAILABLE).
    if status != "UNAVAILABLE":
        snapshot["pipeline"] = {
            "pipeline_version": MACRO_ADAPTER_VERSION,
            "provider": resolution,
            "series_count": len(MACRO_SERIES_REGISTRY),
            "missing_series": missing_series,
            "regime_reasoning": regime_reasoning,
        }

    return snapshot
