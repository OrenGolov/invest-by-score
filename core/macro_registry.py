"""Macroeconomic series registry (Sprint N3) — vintage-aware metadata for FRED-like data.

Each series carries complete provenance:
- source provider (FRED, BLS, etc.)
- publication-time semantics (first-release vs revision)
- reference period and reporting lag
- unit, frequency, and transformations
- PIT policy enforcement (eligibility by published_time, never reference-period end)
- feature version for reproducibility

Revisions are appended to raw_store (W6) with new version lines; the adapter
uses published_time to gate eligibility, so a release published after as_of
is provably excluded. A missing series degrades confidence (INCOMPLETE), never
silently zero-fills.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Federal Reserve Economic Data (FRED) series registry ----------------------

@dataclass(frozen=True)
class MacroSeries:
    """Complete metadata for one macroeconomic series."""
    series_id: str                # e.g. "FEDFUNDS", "CPIAUCSL", "ICSA"
    provider: str                 # "FRED", "BLS", "CENSUS", etc.
    provider_key_env: str         # Environment variable for API key
    base_url: str                 # Provider API endpoint base
    name: str                     # Human-readable name
    unit: str                     # "percent", "index_1982_84=100", "thousands", etc.
    frequency: str                # "daily", "weekly", "monthly", "quarterly", "annual"
    transformation: str           # "none", "log_diff", "percent_change", "yoy_change", etc.
    lag_days: float               # Publication lag (e.g., CPI released ~month-end+12d)
    feature_version: str          # Versioned constant for reproducibility
    lookback_periods: int         # How many periods to fetch (typically 10+ years)
    reference_period_field: str   # JSON field for date/period (e.g. "date", "period")
    published_time_field: str     # JSON field for publication timestamp (e.g. "publication_date")
    first_release_marker: str     # Marker in payload for first release (e.g. "notes")
    description: str              # Role in regime / economic context

    def to_dict(self) -> dict:
        """Serialize to dict for caching / audit."""
        return {
            "series_id": self.series_id,
            "provider": self.provider,
            "name": self.name,
            "unit": self.unit,
            "frequency": self.frequency,
            "transformation": self.transformation,
            "lag_days": self.lag_days,
            "feature_version": self.feature_version,
            "lookback_periods": self.lookback_periods,
            "description": self.description,
        }


# --- Curated series (complete, non-negotiable for v1 macro regime classification) --

MACRO_SERIES_REGISTRY: dict[str, MacroSeries] = {
    "fed_funds": MacroSeries(
        series_id="FEDFUNDS",
        provider="FRED",
        provider_key_env="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred/series/data",
        name="Federal Funds Effective Rate",
        unit="percent",
        frequency="daily",
        transformation="none",
        lag_days=1.0,
        feature_version="macro-feature-fed_funds-v1",
        lookback_periods=500,  # ~2 years of daily data
        reference_period_field="date",
        published_time_field="date",  # FRED daily data published same day or next morning
        first_release_marker="",
        description="Central bank target rate; inversely linked to risk-on appetite and tech multiples.",
    ),
    "cpi_yoy": MacroSeries(
        series_id="CPIAUCSL",
        provider="FRED",
        provider_key_env="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred/series/data",
        name="Consumer Price Index (CPI) YoY % Change",
        unit="percent",
        frequency="monthly",
        transformation="percent_change_yoy",
        lag_days=12.0,  # Typically released ~month 13 at 8:30 ET
        feature_version="macro-feature-cpi_yoy-v1",
        lookback_periods=120,  # 10 years of monthly
        reference_period_field="date",
        published_time_field="publication_date",
        first_release_marker="real_time_start",
        description="Inflation momentum; above-target readings compress growth multiples and increase volatility regime risk.",
    ),
    "initial_claims": MacroSeries(
        series_id="ICSA",
        provider="FRED",
        provider_key_env="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred/series/data",
        name="Initial Jobless Claims (Seasonally Adjusted)",
        unit="thousands",
        frequency="weekly",
        transformation="none",
        lag_days=3.0,  # Released Thursday morning for prior week
        feature_version="macro-feature-initial_claims-v1",
        lookback_periods=260,  # 5 years of weekly
        reference_period_field="date",
        published_time_field="publication_date",
        first_release_marker="notes",
        description="Labor market health; spikes correlate with regime downturns and elevated drawdown risk.",
    ),
    "gdp_growth": MacroSeries(
        series_id="A191RL1Q225SBEA",
        provider="FRED",
        provider_key_env="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred/series/data",
        name="Real GDP Growth (YoY %)",
        unit="percent",
        frequency="quarterly",
        transformation="percent_change_yoy",
        lag_days=30.0,  # Preliminary release ~month after quarter end
        feature_version="macro-feature-gdp_growth-v1",
        lookback_periods=40,  # 10 years of quarterly
        reference_period_field="date",
        published_time_field="publication_date",
        first_release_marker="real_time_start",
        description="Economic growth rate; below-trend readings increase bear case and regime-downshift risk.",
    ),
    "10y_yield": MacroSeries(
        series_id="DGS10",
        provider="FRED",
        provider_key_env="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred/series/data",
        name="10-Year Treasury Yield (%)",
        unit="percent",
        frequency="daily",
        transformation="none",
        lag_days=0.5,  # Published intraday by FRED
        feature_version="macro-feature-10y_yield-v1",
        lookback_periods=500,  # ~2 years of daily
        reference_period_field="date",
        published_time_field="date",
        first_release_marker="",
        description="Long-term risk-free rate; drives equity risk premium and discount-rate sensitivity (esp. growth stocks).",
    ),
}


# --- Sector sensitivity mapping (static, v1) -----------------------------------
# Maps GICS sectors to factor-loading vectors: {rates_sensitivity, energy_beta, usd_beta}.
# Positive values = headwind when factors move up; rates sensitivity is negative for
# duration-driven multiples compression.

MACRO_SENSITIVITY_VERSION = "macro-sensitivity-v1"

SECTOR_MACRO_LOADINGS: dict[str, dict[str, float]] = {
    "Energy": {
        "rates": 0.3,      # Moderate capex sensitivity to rates
        "energy": 0.8,     # High direct commodity linkage
        "usd": -0.5,       # Exporters benefit from weak dollar
    },
    "Financials": {
        "rates": -0.6,     # Banks benefit from higher rates (NIM expansion)
        "energy": 0.0,     # No direct exposure
        "usd": 0.2,        # Strong USD helps net interest margins
    },
    "Information Technology": {
        "rates": -0.8,     # High duration (multiples compression in rising-rate regimes)
        "energy": 0.0,     # Energy-agnostic
        "usd": 0.4,        # US-domiciled, earnings headwind if dollar weakens
    },
    "Consumer Discretionary": {
        "rates": 0.2,      # Moderate sensitivity (consumer credit + discount rates)
        "energy": 0.1,     # Fuel costs matter
        "usd": 0.1,        # Mixed (domestic focused but export exposure)
    },
    "Consumer Staples": {
        "rates": 0.1,      # Lower sensitivity (defensive, less leveraged)
        "energy": 0.3,     # Supply-chain and shipping cost exposure
        "usd": 0.0,        # Domestic-focused
    },
    "Health Care": {
        "rates": 0.0,      # Stable cash flows, moderate capex
        "energy": 0.0,     # Energy-agnostic
        "usd": 0.3,        # Pharma exporters, R&D in foreign currencies
    },
    "Industrials": {
        "rates": 0.5,      # Capex-sensitive, leveraged balance sheets
        "energy": 0.2,     # Supply-chain exposure
        "usd": 0.3,        # Exporters sensitive to USD strength
    },
    "Materials": {
        "rates": 0.4,      # Capital-intensive, long-term project financing
        "energy": 0.6,     # Commodity-linked production costs
        "usd": -0.4,       # Commodity exporters benefit from weak USD
    },
    "Real Estate": {
        "rates": -0.7,     # Duration asset (mortgage rates, cap rates)
        "energy": 0.1,     # Operational (heating/cooling costs)
        "usd": 0.0,        # Domestic-focused
    },
    "Utilities": {
        "rates": -0.5,     # Duration asset (regulated ROE on long-lived assets)
        "energy": 0.4,     # Fuel cost pass-through
        "usd": 0.0,        # Domestic-focused
    },
    "Communication Services": {
        "rates": -0.5,     # Duration asset (long-life licenses, subscriber value)
        "energy": 0.0,     # Energy-agnostic
        "usd": 0.2,        # Global revenue streams, hedging variability
    },
}

# Symbol → GICS Sector mapping (v1, curated set; extensible but pinned for reproducibility).
# Used by N3 to apply sector-specific macro sensitivities.
# TODO: expand to full Russell 3000 + sector classification service integration (deferred).

SYMBOL_TO_SECTOR: dict[str, str] = {
    # Information Technology
    "MSFT": "Information Technology",
    "AAPL": "Information Technology",
    "NVDA": "Information Technology",
    "CRWD": "Information Technology",
    "PANW": "Information Technology",
    "ANET": "Information Technology",
    # Energy
    # (placeholder; no energy holdings in current portfolio)
    # Materials
    # (placeholder; no materials holdings in current portfolio)
    # Industrials
    "RKLB": "Industrials",
    # Consumer Discretionary
    "TSLA": "Consumer Discretionary",
    "VRT": "Consumer Discretionary",
    # Other sectors
    "GOOG": "Information Technology",
    "GOOGL": "Information Technology",
    "META": "Information Technology",
    "IREN": "Utilities",  # (or Energy if renewable; treating as Utilities here)
    "AVGO": "Information Technology",
    "NBIS": "Information Technology",
}


def get_series(logical_id: str) -> MacroSeries | None:
    """Look up a series by its logical identifier."""
    return MACRO_SERIES_REGISTRY.get(logical_id)


def get_sector_loadings(sector: str) -> dict[str, float] | None:
    """Look up macro sensitivities for a GICS sector."""
    return SECTOR_MACRO_LOADINGS.get(sector)


def get_symbol_sector(symbol: str) -> str | None:
    """Map a ticker symbol to its GICS sector."""
    return SYMBOL_TO_SECTOR.get(symbol.upper())
