"""Canonical feature registry tests (Sprint M1).

Every feature that enters a production model must be registered with complete
metadata. These tests pin the registry contract.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.feature_registry import (
    FEATURE_REGISTRY_STORE_PATH,
    FEATURE_REGISTRY_VERSION,
    FeatureRegistry,
    FeatureSpec,
    build_default_registry,
    feature_contract_problems,
    load_feature_registry,
    persist_feature_registry,
    registry_hash,
    spec_problems,
)
from core.contract_verification import feature_registry_problems


def _valid_spec(name: str = "test_feature", **overrides) -> FeatureSpec:
    base = dict(
        name=name, owner="market_data_agent", domain="market",
        formula="close / sma(close, 20) - 1", version="market-feature-v1",
        unit="ratio", frequency="daily", lookback="20d", minimum_history=20,
        null_policy="exclude", pit_rule="Only bars <= as_of are used.",
        source_dependencies=["yahoo_finance_chart"], feature_family="trend",
        model_compatibility=["technical_analysis"],
    )
    base.update(overrides)
    return FeatureSpec(**base)


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"Open": list(closes), "High": list(closes), "Low": list(closes),
         "Close": list(closes), "Volume": [1_000_000.0] * len(closes)},
        index=index)


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * i for i in range(sessions)]

class TestFeatureSpec(unittest.TestCase):
    def test_to_dict_round_trips(self):
        spec = _valid_spec()
        d = spec.to_dict()
        self.assertEqual(d["name"], "test_feature")
        rebuilt = FeatureSpec(**d)
        self.assertEqual(rebuilt.name, spec.name)

    def test_canonical_hash_is_deterministic(self):
        spec = _valid_spec()
        self.assertEqual(spec.canonical_hash(), spec.canonical_hash())

    def test_same_spec_same_hash(self):
        self.assertEqual(_valid_spec().canonical_hash(), _valid_spec().canonical_hash())

    def test_different_field_different_hash(self):
        self.assertNotEqual(
            _valid_spec().canonical_hash(),
            _valid_spec(version="market-feature-v2").canonical_hash())

    def test_hash_covers_all_fields(self):
        base = _valid_spec()
        for fn in ("name", "owner", "domain", "formula", "version", "unit",
                   "frequency", "lookback", "minimum_history", "null_policy",
                   "pit_rule", "source_dependencies", "feature_family",
                   "model_compatibility"):
            kw = {fn: "DIFFERENT" if isinstance(base.to_dict()[fn], str) else 999}
            if fn in ("source_dependencies", "model_compatibility"):
                kw[fn] = ["different"]
            self.assertNotEqual(base.canonical_hash(), _valid_spec(**kw).canonical_hash(),
                                f"hash did not change when {fn} changed")


class TestSpecProblems(unittest.TestCase):
    def test_valid_spec_has_no_problems(self):
        self.assertEqual(spec_problems(_valid_spec()), [])

    def test_empty_name(self):
        self.assertTrue(any("name is empty" in p for p in spec_problems(_valid_spec(name=""))))

    def test_unknown_owner(self):
        self.assertTrue(any("not a known producer" in p for p in spec_problems(_valid_spec(owner="ghost"))))

    def test_invalid_domain(self):
        self.assertTrue(any("not a valid domain" in p for p in spec_problems(_valid_spec(domain="magic"))))

    def test_empty_formula(self):
        self.assertTrue(any("formula is empty" in p for p in spec_problems(_valid_spec(formula=""))))

    def test_empty_version(self):
        self.assertTrue(any("version is empty" in p for p in spec_problems(_valid_spec(version=""))))

    def test_invalid_null_policy(self):
        self.assertTrue(any("null_policy" in p for p in spec_problems(_valid_spec(null_policy="guess"))))

    def test_invalid_frequency(self):
        self.assertTrue(any("frequency" in p for p in spec_problems(_valid_spec(frequency="whenever"))))

    def test_minimum_history_too_low(self):
        self.assertTrue(any("minimum_history" in p for p in spec_problems(_valid_spec(minimum_history=0))))

    def test_empty_pit_rule(self):
        self.assertTrue(any("pit_rule is empty" in p for p in spec_problems(_valid_spec(pit_rule=""))))


class TestFeatureRegistry(unittest.TestCase):
    def test_register_valid(self):
        reg = FeatureRegistry(); reg.register(_valid_spec())
        self.assertTrue(reg.is_registered("test_feature"))

    def test_register_invalid_raises(self):
        reg = FeatureRegistry()
        with self.assertRaises(ValueError):
            reg.register(_valid_spec(name=""))

    def test_get_and_hash(self):
        reg = FeatureRegistry(); spec = _valid_spec(); reg.register(spec)
        self.assertEqual(reg.get("test_feature"), spec)
        self.assertIsNone(reg.get("nope"))
        self.assertEqual(reg.feature_hash("test_feature"), spec.canonical_hash())
        self.assertIsNone(reg.feature_hash("nope"))

    def test_problems_empty_vs_populated(self):
        reg = FeatureRegistry()
        self.assertTrue(len(reg.problems()) > 0)
        reg.register(_valid_spec())
        self.assertEqual(reg.problems(), [])


class TestFeatureContractProblems(unittest.TestCase):
    def _snap(self, extras=None):
        snap = {"rsi": {"name": "rsi", "value": 50.0, "as_of": "2024-01-15 00:00:00",
                        "source_id": "yahoo_finance_chart", "published_time": "2024-01-15 00:00:00",
                        "calculation_version": "market-feature-v1", "lookback_period": "14d"}}
        if extras:
            snap.update(extras)
        return {"features": snap}

    def test_unregistered_rejected(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems(self._snap({"ghost": {"name": "ghost", "value": 1.0,
            "as_of": "2024-01-15 00:00:00", "source_id": "x", "published_time": "2024-01-15 00:00:00",
            "calculation_version": "market-feature-v1", "lookback_period": "1d"}}), reg)
        self.assertTrue(any("ghost" in p and "not registered" in p for p in problems))

    def test_pit_violation_rejected(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems(self._snap({"rsi": {"name": "rsi", "value": 50.0,
            "as_of": "2024-01-15 00:00:00", "source_id": "yahoo_finance_chart",
            "published_time": "2024-01-16 00:00:00", "calculation_version": "market-feature-v1",
            "lookback_period": "14d"}}), reg)
        self.assertTrue(any("published_time" in p and "after" in p for p in problems))

    def test_empty_surface(self):
        reg = FeatureRegistry()
        problems = feature_contract_problems({"features": {}}, reg)
        self.assertTrue(any("no feature contracts" in p for p in problems))

    def test_clean_registered(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        self.assertEqual(feature_contract_problems(self._snap(), reg), [])


class TestRegistryHash(unittest.TestCase):
    def test_deterministic(self):
        reg = build_default_registry()
        self.assertEqual(registry_hash(reg), registry_hash(reg))

    def test_changes_on_spec_change(self):
        r1 = FeatureRegistry(); r1.register(_valid_spec("a", version="v1"))
        r2 = FeatureRegistry(); r2.register(_valid_spec("a", version="v2"))
        self.assertNotEqual(registry_hash(r1), registry_hash(r2))

    def test_changes_on_add(self):
        r = FeatureRegistry(); r.register(_valid_spec("a")); h1 = registry_hash(r)
        r.register(_valid_spec("b")); self.assertNotEqual(h1, registry_hash(r))


class TestBuildDefaultRegistry(unittest.TestCase):
    def test_builds_22_features(self):
        self.assertEqual(len(build_default_registry().all_features()), 22)

    def test_expected_names_present(self):
        expected = {
            "change_1d", "change_5d", "change_20d", "change_60d",
            "change_50d", "change_100d", "change_150d", "change_200d",
            "atr_14", "trend_slope_60d", "trend_vs_20d_mean", "rsi",
            "volatility", "volume_ratio_20d",
            "price_vs_ma_50", "price_vs_ma_100", "price_vs_ma_150", "price_vs_ma_200",
            "ma_50", "ma_100", "ma_150", "ma_200"}
        self.assertEqual(set(build_default_registry().all_features()), expected)

    def test_all_specs_valid(self):
        reg = build_default_registry()
        for name, spec in reg.all_features().items():
            self.assertEqual(spec_problems(spec), [], f"{name}: {spec_problems(spec)}")

    def test_version_matches_live_constant(self):
        from core.config import MARKET_FEATURE_VERSION
        for spec in build_default_registry().all_features().values():
            self.assertEqual(spec.version, MARKET_FEATURE_VERSION)

    def test_all_owned_by_market_data_agent(self):
        for spec in build_default_registry().all_features().values():
            self.assertEqual(spec.owner, "market_data_agent")

    def test_all_compatible_with_technical_analysis(self):
        for spec in build_default_registry().all_features().values():
            self.assertIn("technical_analysis", spec.model_compatibility)


class TestPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.jsonl"
            reg = build_default_registry()
            persist_feature_registry(reg, path)
            loaded = load_feature_registry(path)
            self.assertEqual(set(loaded.all_features()), set(reg.all_features()))

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.jsonl"
            reg = build_default_registry()
            self.assertTrue(persist_feature_registry(reg, path))
            self.assertFalse(persist_feature_registry(reg, path))

    def test_integrity_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.jsonl"
            reg = FeatureRegistry(); reg.register(_valid_spec("a", version="v1"))
            persist_feature_registry(reg, path)
            record = {"registry_version": FEATURE_REGISTRY_VERSION,
                      "registry_hash": registry_hash(reg),
                      "features": {"a": _valid_spec("a", version="v2").to_dict()}}
            path.write_text(json.dumps(record, sort_keys=True, default=str) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                persist_feature_registry(reg, path)
            self.assertIn("integrity violation", str(ctx.exception))

    def test_malformed_line_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.jsonl"
            path.write_text("this is not json\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_feature_registry(path)
            self.assertIn("not valid JSON", str(ctx.exception))

    def test_load_empty_store(self):
        reg = load_feature_registry(pathlib.Path("/nonexistent/path/registry.jsonl"))
        self.assertEqual(reg.all_features(), {})


class TestContractVerifierIntegration(unittest.TestCase):
    def test_clean_snapshot(self):
        from agents.market_data_agent import fetch_market_snapshot
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with patch("agents.market_data_agent.fetch_price_history", lambda *a, **k: frame.copy()), \
             patch("core.regime_agent.fetch_price_history", lambda *a, **k: frame.copy()), \
             patch.dict(os.environ, {"NEWS_PROVIDER_API_KEY": "", "FRED_API_KEY": ""}):
            snapshot = fetch_market_snapshot("TEST", as_of)
        self.assertEqual(feature_registry_problems(snapshot), [])

    def test_unregistered_injected(self):
        from agents.market_data_agent import fetch_market_snapshot
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with patch("agents.market_data_agent.fetch_price_history", lambda *a, **k: frame.copy()), \
             patch("core.regime_agent.fetch_price_history", lambda *a, **k: frame.copy()), \
             patch.dict(os.environ, {"NEWS_PROVIDER_API_KEY": "", "FRED_API_KEY": ""}):
            snapshot = fetch_market_snapshot("TEST", as_of)
        snapshot["features"]["ghost"] = {"name": "ghost", "value": 1.0, "as_of": as_of,
            "source_id": "x", "published_time": as_of,
            "calculation_version": "market-feature-v1", "lookback_period": "1d"}
        problems = feature_registry_problems(snapshot)
        self.assertTrue(any("ghost" in p and "not registered" in p for p in problems))


class TestEngineIntegration(unittest.TestCase):
    def test_run_declares_registry_version(self):
        from core.backtest import run_walk_forward_backtest
        frame = _frame(_ramp(260, step=1.0))
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "backtest_runs.jsonl"
            with patch("core.backtest.engine.build_score") as mock_score:
                mock_score.return_value = SimpleNamespace(
                    score=5.0, action="ANALYSIS_ONLY", current_time_score=5.0,
                    long_term_score=5.0, confidence=0.5, explanation="",
                    risk_flags=[], moving_averages={}, rsi=None, volatility=None,
                    market_context={}, data_quality={}, source_metadata={},
                    recommended_actions={}, latest_financial_report={},
                    next_expected_report={}, insights={}, scoring_breakdown={},
                    source_reliability={}, technical_features={}, feature_metadata={},
                    governance={}, evidence_ledger={}, fundamental_score=0.0,
                    fundamental_features={}, source_quality={}, replay_metadata={},
                    news_snapshot={}, sentiment_snapshot={}, macro_snapshot={},
                    market_regime_snapshot={}, confidence_breakdown={},
                    ensemble_breakdown={})
                result = run_walk_forward_backtest(
                    "TEST", frame, fold_sessions=50, embargo_sessions=60,
                    holdout_sessions=40, manifest_store_path=store)
        self.assertEqual(
            result["manifest"]["versions"].get("feature_registry"),
            FEATURE_REGISTRY_VERSION)


if __name__ == "__main__":
    unittest.main()

