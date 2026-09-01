"""Temporary N3 smoke check — deleted after the sprint lands."""
from unittest.mock import patch

from core.macro_adapter import build_macro_snapshot
from core.macro_contract import fetch_macro_snapshot

AS_OF = "2024-01-05 12:00:00"
ENV = {"FRED_API_KEY": "smoke-key"}


def _fred_record(series_id, date, value):
    return {
        "source_record_id": f"{series_id}_{date}",
        "series_id": series_id,
        "reference_date": date,
        "published_time": date,
        "value": value,
    }


def _snapshot(records_by_provider_id):
    """Build mock FRED data: {provider_id (FEDFUNDS, etc): [records]}."""
    def mock_fetch(provider_id, *args, **kwargs):
        records = records_by_provider_id.get(provider_id, [])
        return {"status": "ok", "records": records, "reason": ""}

    with patch.dict("os.environ", ENV), \
            patch("core.macro_adapter.append_raw_records", return_value=True), \
            patch("core.macro_adapter.fetch_fred_series", side_effect=mock_fetch):
        return build_macro_snapshot("TEST", AS_OF)


legacy = {
    "ticker": "TEST", "as_of": "2024-01-05 12:00:00", "status": "UNAVAILABLE",
    "source_id": "macro_provider_unconfigured", "source_confidence": 0.0,
    "published_time": None, "calculation_version": "macro-contract-v1",
    "regime": "neutral", "regime_score": 0.5, "series_values": {}, "series_credibility": {},
    "sector_loadings": {}, "per_series_contributions": [], "reason": (
        "No verified macro data provider is connected. Economic data is not inferred "
        "from price or technical indicators."
    ),
}
assert fetch_macro_snapshot("TEST", "2024-01-05 12:00:00") == legacy, "no-key contract drifted"
print("1. no-key UNAVAILABLE byte-for-byte: OK")

# OK path: all series with reasonable values (keyed by provider series_id)
snap = _snapshot({
    "FEDFUNDS": [_fred_record("FEDFUNDS", "2024-01-05", 5.5)],
    "CPIAUCSL": [_fred_record("CPIAUCSL", "2024-01-05", 2.5)],
    "ICSA": [_fred_record("ICSA", "2024-01-05", 210.0)],
    "A191RL1Q225SBEA": [_fred_record("A191RL1Q225SBEA", "2024-01-05", 2.5)],
    "DGS10": [_fred_record("DGS10", "2024-01-05", 3.5)],
})
print("2. OK path:", snap["status"], round(snap["source_confidence"], 2), snap["regime"])
assert snap["status"] == "OK"
assert snap["regime"] in ["risk_on", "risk_off", "neutral"]

# INCOMPLETE path: missing series
snap = _snapshot({
    "FEDFUNDS": [_fred_record("FEDFUNDS", "2024-01-05", 5.5)],
    # CPIAUCSL: Missing
    "ICSA": [_fred_record("ICSA", "2024-01-05", 210.0)],
    # A191RL1Q225SBEA: Missing
    "DGS10": [_fred_record("DGS10", "2024-01-05", 3.5)],
})
print("3. INCOMPLETE path:", snap["status"], "missing series count =", len(snap["pipeline"]["missing_series"]) if "pipeline" in snap else 0)
assert snap["status"] == "INCOMPLETE"
assert len(snap["pipeline"]["missing_series"]) == 2

# INVALID path: future-dated records
snap = _snapshot({
    "FEDFUNDS": [_fred_record("FEDFUNDS", "2024-01-10", 5.5)],  # After as_of
    "CPIAUCSL": [_fred_record("CPIAUCSL", "2024-01-05", 2.5)],
    "ICSA": [_fred_record("ICSA", "2024-01-05", 210.0)],
    "A191RL1Q225SBEA": [_fred_record("A191RL1Q225SBEA", "2024-01-05", 2.5)],
    "DGS10": [_fred_record("DGS10", "2024-01-05", 3.5)],
})
print("4. INVALID path:", snap["status"])
assert snap["status"] == "INVALID"

# Risk regime classification: risk-on scenario
snap = _snapshot({
    "FEDFUNDS": [_fred_record("FEDFUNDS", "2024-01-05", 1.5)],  # Accommodative
    "CPIAUCSL": [_fred_record("CPIAUCSL", "2024-01-05", 1.5)],  # Subdued
    "ICSA": [_fred_record("ICSA", "2024-01-05", 180.0)],  # Tight
    "A191RL1Q225SBEA": [_fred_record("A191RL1Q225SBEA", "2024-01-05", 3.5)],  # Strong
    "DGS10": [_fred_record("DGS10", "2024-01-05", 4.0)],  # Elevated
})
print("5. Risk-on scenario:", snap["regime"], snap["regime_score"])
assert snap["regime"] == "risk_on" and snap["regime_score"] > 0.7

# Risk regime classification: risk-off scenario
snap = _snapshot({
    "FEDFUNDS": [_fred_record("FEDFUNDS", "2024-01-05", 5.0)],  # Restrictive
    "CPIAUCSL": [_fred_record("CPIAUCSL", "2024-01-05", 4.0)],  # High
    "ICSA": [_fred_record("ICSA", "2024-01-05", 500.0)],  # Elevated
    "A191RL1Q225SBEA": [_fred_record("A191RL1Q225SBEA", "2024-01-05", 0.5)],  # Weak
    "DGS10": [_fred_record("DGS10", "2024-01-05", 1.5)],  # Depressed
})
print("6. Risk-off scenario:", snap["regime"], snap["regime_score"])
assert snap["regime"] == "risk_off" and snap["regime_score"] < 0.3

print("ALL SMOKE CHECKS PASSED")
