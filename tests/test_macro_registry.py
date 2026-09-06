"""Sprint N3: macro registry — vintage awareness, PIT gating, born-wired wiring.

Hermetic: the FRED fetch is patched with canned fixtures (the real registry
lag conventions apply through the same code path the adapter uses), raw-store
writes are mocked, no wall-clock reads. Gates covered: eligibility by release
published_time (never reference-period end); pending releases excluded but
never invalidating; revisions after as_of can never leak; missing series
degrade confidence (INCOMPLETE), never zero-fill.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from core import config as core_config
from core.macro_adapter import (
    MACRO_SOURCE_ID,
    UNAVAILABLE_SOURCE_ID,
    build_macro_snapshot,
    compute_risk_regime,
    fetch_fred_series,
    pit_filter_macro,
    resolve_macro_provider,
)
from core.macro_registry import (
    MACRO_REGISTRY_VERSION,
    MACRO_SERIES_REGISTRY,
    MACRO_SENSITIVITY_VERSION,
    get_sector_loadings,
    get_symbol_sector,
)
from core.orchestrator import _derive_agent_statuses, orchestrate_score
from core.score_engine import _ensemble_blend, build_score

AS_OF = "2024-03-15 12:00:00"
KEY_ENV = {"FRED_API_KEY": "test-key"}

# Canned fixtures: reference dates chosen so the REAL registry lag conventions
# (applied through fetch_fred_series) land before as_of.
HEALTHY = {
    "FEDFUNDS": [{"reference_date": "2024-03-01", "value": 5.33}],
    "CPIAUCSL": [{"reference_date": "2024-01-01", "value": 3.1}],
    "ICSA": [{"reference_date": "2024-03-09", "value": 190.0}],
    "A191RL1Q225SBEA": [{"reference_date": "2023-10-01", "value": 3.1}],
    "DGS10": [{"reference_date": "2024-03-13", "value": 4.2}],
}
RISK_OFF = {
    "FEDFUNDS": [{"reference_date": "2024-03-01", "value": 5.5}],
    "CPIAUCSL": [{"reference_date": "2024-01-01", "value": 4.0}],
    "ICSA": [{"reference_date": "2024-03-09", "value": 500.0}],
    "A191RL1Q225SBEA": [{"reference_date": "2023-10-01", "value": 0.5}],
    "DGS10": [{"reference_date": "2024-03-13", "value": 1.0}],
}
RISK_ON = {
    "FEDFUNDS": [{"reference_date": "2024-03-01", "value": 1.5}],
    "CPIAUCSL": [{"reference_date": "2024-01-01", "value": 1.5}],
    "ICSA": [{"reference_date": "2024-03-09", "value": 180.0}],
    "A191RL1Q225SBEA": [{"reference_date": "2023-10-01", "value": 3.5}],
    "DGS10": [{"reference_date": "2024-03-13", "value": 4.0}],
}


def _fake_fetch(values_by_series):
    """Stand-in for fetch_fred_series; the caller passes the registry lag_days."""

    def _fetch(series_id, api_key, lag_days=0.0, lookback_periods=120, timeout=10.0):
        records = []
        for entry in values_by_series.get(series_id, []):
            reference_dt = datetime.fromisoformat(entry["reference_date"])
            published_dt = reference_dt + timedelta(days=lag_days)
            records.append({
                "source_record_id": f"{series_id}_{entry['reference_date']}",
                "series_id": series_id,
                "reference_date": entry["reference_date"],
                "published_time": published_dt.isoformat(),
                "published_time_source": "lag_convention",
                "value": entry["value"],
            })
        return {"status": "ok", "records": records, "reason": ""}

    return _fetch


def _snapshot(values_by_series, ticker="MSFT", as_of=AS_OF):
    with patch.dict(os.environ, KEY_ENV), \
            patch("core.macro_adapter.append_raw_records", return_value=True), \
            patch("core.macro_adapter.fetch_fred_series", side_effect=_fake_fetch(values_by_series)):
        return build_macro_snapshot(ticker, as_of)


def _rec(rid="r1", published="2024-03-10T00:00:00", value=5.33, series_id="DFF"):
    return {
        "source_record_id": rid,
        "series_id": series_id,
        "reference_date": published[:10],
        "published_time": published,
        "published_time_source": "lag_convention",
        "value": value,
    }


class RegistryTests(unittest.TestCase):
    """Every series carries the full N3 field contract from the master spec."""

    REQUIRED_FIELDS = (
        "series_id", "provider", "provider_key_env", "base_url", "name",
        "unit", "frequency", "transformation", "lag_days", "feature_version",
        "lookback_periods", "description",
    )

    def test_five_logical_series_exist(self):
        self.assertEqual(
            set(MACRO_SERIES_REGISTRY),
            {"fed_funds", "cpi_yoy", "initial_claims", "gdp_growth", "10y_yield"},
        )

    def test_every_series_carries_the_full_field_contract(self):
        for logical_id, series in MACRO_SERIES_REGISTRY.items():
            for field in self.REQUIRED_FIELDS:
                self.assertTrue(hasattr(series, field), f"{logical_id} missing {field}")
                if field in ("series_id", "provider", "name", "unit", "frequency", "transformation"):
                    self.assertTrue(getattr(series, field), f"{logical_id}.{field} empty")
            self.assertTrue(series.feature_version.startswith("macro-feature-"))

    def test_registry_and_sensitivity_versions_exist(self):
        self.assertEqual(MACRO_REGISTRY_VERSION, "macro-registry-v1")
        self.assertEqual(MACRO_SENSITIVITY_VERSION, "macro-sensitivity-v1")

    def test_lag_conventions_are_publication_time_safe(self):
        # Lookahead-bias guard: a monthly series dated the 1st must not be
        # treated as published mid-same-month; quarterly must wait ~a quarter.
        self.assertGreaterEqual(MACRO_SERIES_REGISTRY["cpi_yoy"].lag_days, 40.0)
        self.assertGreaterEqual(MACRO_SERIES_REGISTRY["gdp_growth"].lag_days, 100.0)
        self.assertGreaterEqual(MACRO_SERIES_REGISTRY["initial_claims"].lag_days, 4.0)
        self.assertGreaterEqual(MACRO_SERIES_REGISTRY["fed_funds"].lag_days, 0.0)

    def test_sector_sensitivity_mapping(self):
        self.assertEqual(get_symbol_sector("MSFT"), "Information Technology")
        loadings = get_sector_loadings(get_symbol_sector("MSFT"))
        self.assertIn("rates", loadings)
        self.assertIn("energy", loadings)
        self.assertIn("usd", loadings)
        self.assertIsNone(get_symbol_sector("VOO"))  # unknown -> never guessed
        self.assertIsNone(get_sector_loadings("Nonexistent Sector"))


class ProviderResolutionTests(unittest.TestCase):
    def test_no_key_is_provider_key_required(self):
        with patch.dict(os.environ, {}, clear=True):
            resolution = resolve_macro_provider(AS_OF)
        self.assertEqual(resolution["status"], "provider_key_required")
        self.assertEqual(resolution["source_confidence"], 0.0)

    def test_with_key_is_live(self):
        with patch.dict(os.environ, KEY_ENV):
            resolution = resolve_macro_provider(AS_OF)
        self.assertEqual(resolution["status"], "live_provider")
        self.assertEqual(resolution["source_id"], MACRO_SOURCE_ID)


class FetchTests(unittest.TestCase):
    FRED_PAYLOAD = {
        "observations": [
            {"date": "2024-03-01", "value": "5.33"},
            {"date": "2024-03-02", "value": "."},  # FRED missing-data marker
            {"date": "2024-03-04", "value": "5.38"},
        ],
    }

    def test_lag_convention_is_applied_to_published_time(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return buffer.read()

        buffer = io.BytesIO(json.dumps(self.FRED_PAYLOAD).encode("utf-8"))
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.urllib.request.urlopen", return_value=FakeResponse()):
            disposition = fetch_fred_series("FEDFUNDS", "k", lag_days=1.0)
        self.assertEqual(disposition["status"], "ok")
        first = disposition["records"][0]
        self.assertEqual(first["published_time"], "2024-03-02T00:00:00")  # reference + 1d lag
        self.assertEqual(first["published_time_source"], "lag_convention")
        # Missing-data markers and malformed values are skipped, never zero-filled.
        self.assertEqual(len(disposition["records"]), 2)

    def test_request_failure_is_an_explicit_disposition(self):
        import urllib.error
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
            disposition = fetch_fred_series("FEDFUNDS", "k", lag_days=1.0)
        self.assertEqual(disposition["status"], "provider_request_failed")
        self.assertEqual(disposition["records"], [])


class PitFilterTests(unittest.TestCase):
    def test_eligible_at_the_boundary_and_pending_after(self):
        as_of = datetime(2024, 3, 15, 12, 0, 0)
        records = [
            _rec(published="2024-03-15 12:00:00"),
            _rec(rid="f", published="2024-03-16 09:00:00"),
        ]
        eligible, rejected = pit_filter_macro(records, as_of)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(rejected[0]["reason"], "future_dated")

    def test_unparseable_publication_time_is_rejected(self):
        as_of = datetime(2024, 3, 15, 12, 0, 0)
        eligible, rejected = pit_filter_macro([_rec(rid="x", published="not-a-date")], as_of)
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]["reason"], "unparseable_published_time")


class RegimeTests(unittest.TestCase):
    def test_risk_on_classification_and_score_mapping(self):
        regime, score, reasoning = compute_risk_regime(
            fed_funds=1.5, cpi_yoy=1.2, initial_claims=180.0, gdp_growth=3.5, yield_10y=4.0,
        )
        self.assertEqual(regime, "risk_on")
        self.assertGreater(score, core_config.MACRO_RISKON_THRESHOLD)
        self.assertAlmostEqual(core_config.MACRO_SCORE_BASE + core_config.MACRO_SCORE_SPAN * score, 8.95, places=2)
        self.assertIn("growth-strong", reasoning)

    def test_risk_off_classification(self):
        regime, score, reasoning = compute_risk_regime(
            fed_funds=5.5, cpi_yoy=4.0, initial_claims=500.0, gdp_growth=0.5, yield_10y=1.0,
        )
        self.assertEqual(regime, "risk_off")
        self.assertLess(score, core_config.MACRO_RISKOFF_THRESHOLD)

    def test_no_signals_is_neutral(self):
        regime, score, reasoning = compute_risk_regime(
            fed_funds=None, cpi_yoy=None, initial_claims=None, gdp_growth=None, yield_10y=None,
        )
        self.assertEqual(regime, "neutral")
        self.assertEqual(score, 0.5)
        self.assertEqual(reasoning, "no_signals")


class SnapshotTests(unittest.TestCase):
    def test_no_key_is_unavailable_never_neutral(self):
        with patch("core.macro_adapter.append_raw_records") as append_mock:
            snapshot = build_macro_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "UNAVAILABLE")
        self.assertEqual(snapshot["source_id"], UNAVAILABLE_SOURCE_ID)
        self.assertIsNone(snapshot["published_time"])
        self.assertEqual(snapshot["series_values"], {})
        self.assertNotIn("pipeline", snapshot)
        append_mock.assert_not_called()

    def test_failed_fetch_degrades_to_incomplete_never_fabricates(self):
        # A configured provider whose per-series requests fail is INCOMPLETE
        # (missing series, degraded confidence), not UNAVAILABLE: the provider
        # gate (no key / not live) owns the UNAVAILABLE contract.
        def _failing_fetch(*args, **kwargs):
            return {"status": "provider_request_failed", "records": [], "reason": "FRED request failed: boom"}

        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.append_raw_records", return_value=True), \
                patch("core.macro_adapter.fetch_fred_series", side_effect=_failing_fetch):
            snapshot = build_macro_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertEqual(snapshot["source_id"], MACRO_SOURCE_ID)
        self.assertIsNone(snapshot["published_time"])
        self.assertEqual(snapshot["series_values"], {})

    def test_all_empty_payloads_degrade_to_incomplete(self):
        # Missing series degrade confidence (INCOMPLETE) and are never
        # zero-filled into neutral evidence (N3 contract).
        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.append_raw_records", return_value=True), \
                patch("core.macro_adapter.fetch_fred_series", side_effect=_fake_fetch({})):
            snapshot = build_macro_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertEqual(len(snapshot["pipeline"]["missing_series"]), 5)

    def test_healthy_run_is_ok_with_per_series_evidence(self):
        snapshot = _snapshot(HEALTHY)
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(len(snapshot["series_values"]), 5)
        self.assertEqual(snapshot["series_values"]["fed_funds"], 5.33)
        self.assertEqual(snapshot["regime"], "neutral")  # mixed signals net to 0.58
        self.assertAlmostEqual(snapshot["regime_score"], 0.58, places=4)
        self.assertEqual(len(snapshot["per_series_contributions"]), 5)
        for entry in snapshot["per_series_contributions"]:
            self.assertTrue(entry["source_record_ids"], f"{entry['series_id']} has no evidence id")
        self.assertEqual(snapshot["sector_loadings"]["rates"], -0.8)  # MSFT -> Information Technology
        pipeline = snapshot["pipeline"]
        self.assertEqual(pipeline["pending_releases"], 0)
        self.assertEqual(pipeline["series_count"], 5)
        self.assertIn("regime_reasoning", pipeline)

    def test_risk_on_and_risk_off_tilts(self):
        self.assertEqual(_snapshot(RISK_ON)["regime"], "risk_on")
        self.assertEqual(_snapshot(RISK_OFF)["regime"], "risk_off")

    def test_missing_series_degrades_to_incomplete_without_zero_fill(self):
        partial = {key: value for key, value in HEALTHY.items() if key != "ICSA"}
        snapshot = _snapshot(partial)
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertEqual(snapshot["pipeline"]["missing_series"], ["initial_claims"])
        self.assertNotIn("initial_claims", snapshot["series_values"])  # absent, never zero-filled
        self.assertAlmostEqual(snapshot["source_confidence"], 0.9 - core_config.MACRO_MISSING_SERIES_PENALTY, places=4)
        credibility = {entry["series_id"]: entry["credibility"] for entry in snapshot["per_series_contributions"]}
        self.assertEqual(credibility["initial_claims"]["status"], "UNAVAILABLE")
        self.assertEqual(credibility["fed_funds"]["status"], "OK")

    def test_pending_releases_are_excluded_but_never_invalidate(self):
        future_heavy = {key: list(value) for key, value in HEALTHY.items()}
        future_heavy["FEDFUNDS"].append({"reference_date": "2024-03-20", "value": 5.4})
        snapshot = _snapshot(future_heavy)
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(snapshot["series_values"]["fed_funds"], 5.33)  # only the eligible vintage
        self.assertEqual(snapshot["pipeline"]["pending_releases"], 1)

    def test_unparseable_publication_time_invalidates_the_payload(self):
        def _malformed_fetch(*args, **kwargs):
            return {"status": "ok", "records": [_rec(rid="bad", published="not-a-date")], "reason": ""}

        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.append_raw_records", return_value=True), \
                patch("core.macro_adapter.fetch_fred_series", side_effect=_malformed_fetch):
            snapshot = build_macro_snapshot("MSFT", AS_OF)
        self.assertEqual(snapshot["status"], "INVALID")

    def test_macro_snapshot_is_deterministic(self):
        self.assertEqual(_snapshot(HEALTHY), _snapshot(HEALTHY))


class EnsembleWiringTests(unittest.TestCase):
    """Born-wired rule: the macro agent enters the ensemble with real weight."""

    @staticmethod
    def _contributions(macro):
        return {
            "market_data": {"score_current": 10.0, "score_long": 10.0, "status": "OK", "note": ""},
            "technical_analysis": {"score_current": 8.0, "score_long": 6.0, "status": "OK", "note": ""},
            "fundamental_analysis": {"score_current": 5.0, "score_long": 5.0, "status": "OK", "note": ""},
            "news_intelligence": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "sentiment": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            "macroeconomic": macro,
            "market_regime": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        }

    def test_macro_ok_gets_its_dedicated_weight_in_both_horizons(self):
        current, long_term, breakdown = _ensemble_blend(
            self._contributions({"score_current": 7.9, "score_long": 7.9, "status": "OK", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        self.assertAlmostEqual(current, (0.70 * 8.0 + 0.10 * 5.0 + 0.10 * 7.9) / 0.90, places=2)
        self.assertAlmostEqual(long_term, (0.70 * 6.0 + 0.20 * 5.0 + 0.10 * 7.9) / 1.0, places=2)
        macro = breakdown["agents"]["macroeconomic"]
        self.assertAlmostEqual(macro["effective_weight_current"], 0.10 / 0.90, places=6)
        self.assertTrue(macro["eligible_current"])
        self.assertTrue(macro["eligible_long"])

    def test_macro_unavailable_renormalizes_instead_of_counting_neutral(self):
        current, _, breakdown = _ensemble_blend(
            self._contributions({"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        self.assertAlmostEqual(current, (0.70 * 8.0 + 0.10 * 5.0) / 0.80, places=2)
        macro = breakdown["agents"]["macroeconomic"]
        self.assertEqual(macro["effective_weight_current"], 0.0)
        self.assertFalse(macro["eligible_current"])

    def test_macro_incomplete_is_ineligible(self):
        _, _, breakdown = _ensemble_blend(
            self._contributions({"score_current": None, "score_long": None, "status": "INCOMPLETE", "note": ""}),
            core_config.ENSEMBLE_WEIGHTS_CURRENT,
            core_config.ENSEMBLE_WEIGHTS_LONG,
        )
        self.assertFalse(breakdown["agents"]["macroeconomic"]["eligible_current"])


class ScoreEngineIntegrationTests(unittest.TestCase):
    """The macro contract flows into build_score through its own ensemble line."""

    @staticmethod
    def _ok_macro_snapshot():
        return _snapshot(HEALTHY)

    @staticmethod
    def _incomplete_macro_snapshot():
        partial = {key: value for key, value in HEALTHY.items() if key != "ICSA"}
        return _snapshot(partial)

    @staticmethod
    def _ok_news():
        return {
            "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": "OK",
            "source_id": "newsapi_news", "source_confidence": 0.7,
            "published_time": "2024-01-02 09:00:00",
            "calculation_version": "news-contract-v1", "lookback_period": "7d",
            "sentiment_score": 0.5, "articles": [], "reason": "",
            "pipeline": {"pipeline_version": core_config.NEWS_PIPELINE_VERSION},
        }

    def test_no_key_keeps_macro_unavailable_and_ineligible(self):
        result = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(result.macro_snapshot["status"], "UNAVAILABLE")
        entry = result.ensemble_breakdown["agents"]["macroeconomic"]
        self.assertFalse(entry["eligible_current"])
        self.assertIsNone(entry["score_current"])
        self.assertIn("vintage-aware", entry["note"])

    def test_macro_line_moves_the_ensemble_not_the_technical_view(self):
        ok_macro = self._ok_macro_snapshot()
        with patch("core.score_engine.fetch_macro_snapshot", return_value=ok_macro):
            with_macro = build_score("MSFT", "2024-01-02", persist_audit=False)
        baseline = build_score("MSFT", "2024-01-02", persist_audit=False)
        macro_line = with_macro.ensemble_breakdown["agents"]["macroeconomic"]
        self.assertEqual(macro_line["score_current"], 7.9)  # 5 + 5 * 0.58
        self.assertAlmostEqual(macro_line["effective_weight_current"], 0.10 / 0.90, places=6)
        self.assertEqual(
            with_macro.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
            baseline.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
        )
        self.assertIn("macro_snapshot_hash", with_macro.replay_metadata)

    def test_macro_incomplete_floors_a_healthy_news_run_to_analysis_only(self):
        # Born-wired floor: with news OK but macro INCOMPLETE, the data-agent
        # posture is INCOMPLETE -> ANALYSIS_ONLY, and the mode can never be
        # PAPER while any live data agent is degraded.
        with patch("core.score_engine.fetch_news_snapshot", return_value=self._ok_news()), \
                patch("core.score_engine.fetch_macro_snapshot", return_value=self._incomplete_macro_snapshot()):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(decision.mode, "ANALYSIS_ONLY")
        self.assertNotIn("agent_status_no_trade", decision.veto_reasons)
        macro_agent = next(a for a in decision.agent_outputs if a.agent == "macroeconomic")
        self.assertEqual(macro_agent.status, "INCOMPLETE")
        self.assertIn("macro_not_fully_usable", macro_agent.warnings)

    def test_macro_unparseable_release_forces_no_trade(self):
        def _malformed_fetch(*args, **kwargs):
            return {"status": "ok", "records": [_rec(rid="bad", published="not-a-date")], "reason": ""}

        with patch.dict(os.environ, KEY_ENV), \
                patch("core.macro_adapter.append_raw_records", return_value=True), \
                patch("core.macro_adapter.fetch_fred_series", side_effect=_malformed_fetch), \
                patch("core.score_engine.fetch_news_snapshot", return_value=self._ok_news()):
            decision = orchestrate_score("MSFT", "2024-01-02")
        self.assertEqual(decision.mode, "NO_TRADE")
        self.assertIn("agent_status_no_trade", decision.veto_reasons)
        macro_agent = next(a for a in decision.agent_outputs if a.agent == "macroeconomic")
        self.assertEqual(macro_agent.status, "INVALID")

    def test_macro_determinism(self):
        ok_macro = self._ok_macro_snapshot()
        with patch("core.score_engine.fetch_macro_snapshot", return_value=ok_macro):
            first = build_score("MSFT", "2024-01-02", persist_audit=False)
            second = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(first.replay_metadata["replay_hash"], second.replay_metadata["replay_hash"])


if __name__ == "__main__":
    unittest.main()