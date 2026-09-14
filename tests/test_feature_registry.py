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
    FEATURE_FAMILIES,
    FEATURE_REGISTRY_STORE_PATH,
    FEATURE_REGISTRY_VERSION,
    KNOWN_UNITS,
    MODEL_FAMILIES,
    PRODUCERS,
    PRODUCER_BY_NAME,
    FeatureContractError,
    FeatureRegistry,
    FeatureSpec,
    build_default_registry,
    feature_contract_problems,
    feature_set_hash,
    feature_surface_digest,
    load_feature_registry,
    model_feature_problems,
    persist_feature_registry,
    producer_problems,
    registry_hash,
    require_model_features,
    spec_problems,
    unwired_producers,
)
from core.contract_verification import feature_registry_problems
from core.backtest import BacktestFeatureContractError, COST_TABLE_V2, run_walk_forward_backtest
from core.backtest.engine import (
    _exposed_feature_surface,
    _replay_window,
    offline_replay_seam,
)
from core import score_engine


def _valid_spec(name: str = "test_feature", **overrides) -> FeatureSpec:
    base = dict(
        name=name, owner="market_data_agent", domain="market",
        formula="close / sma(close, 20) - 1", version="market-feature-v1",
        unit="ratio", frequency="daily", lookback="14d", minimum_history=15,
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


class TestSpecMetadataCompleteness(unittest.TestCase):
    """Every required metadata field is enforced, not merely stored."""

    def test_unit_enforced(self):
        self.assertTrue(any("unit is empty" in p for p in spec_problems(_valid_spec(unit=""))))
        problems = spec_problems(_valid_spec(unit="bananas"))
        self.assertTrue(any("not a known unit" in p for p in problems))

    def test_lookback_enforced(self):
        self.assertTrue(any("lookback is empty" in p for p in spec_problems(_valid_spec(lookback=""))))
        problems = spec_problems(_valid_spec(lookback="two weeks"))
        self.assertTrue(any("lookback" in p for p in problems))

    def test_minimum_history_must_cover_the_lookback(self):
        problems = spec_problems(_valid_spec(lookback="60d", minimum_history=20))
        self.assertTrue(any("below the 60d lookback" in p for p in problems))
        self.assertEqual(
            spec_problems(_valid_spec(lookback="60d", minimum_history=60)), [])

    def test_source_dependencies_enforced(self):
        problems = spec_problems(_valid_spec(source_dependencies=[]))
        self.assertTrue(any("source_dependencies is empty" in p for p in problems))
        problems = spec_problems(_valid_spec(source_dependencies=["mystery_feed"]))
        self.assertTrue(any("not a known source" in p for p in problems))

    def test_feature_family_enforced(self):
        problems = spec_problems(_valid_spec(feature_family=""))
        self.assertTrue(any("feature_family is empty" in p for p in problems))
        problems = spec_problems(_valid_spec(feature_family="vibes"))
        self.assertTrue(any("not a known family" in p for p in problems))

    def test_model_compatibility_enforced(self):
        problems = spec_problems(_valid_spec(model_compatibility=[]))
        self.assertTrue(any("model_compatibility is empty" in p for p in problems))
        problems = spec_problems(_valid_spec(model_compatibility=["oracle"]))
        self.assertTrue(any("not a known model family" in p for p in problems))

    def test_name_must_be_lower_snake_case(self):
        problems = spec_problems(_valid_spec(name="Not A Feature"))
        self.assertTrue(any("lower_snake_case" in p for p in problems))

    def test_metadata_is_complete_on_the_default_registry(self):
        for name, spec in build_default_registry().all_features().items():
            self.assertEqual(spec_problems(spec), [], name)
            self.assertIn(spec.unit, KNOWN_UNITS, name)
            self.assertIn(spec.feature_family, FEATURE_FAMILIES, name)
            self.assertTrue(set(spec.model_compatibility) <= set(MODEL_FAMILIES), name)


class TestProducerRegistry(unittest.TestCase):
    """A producer must exist as a module + callable, not just as a name."""

    def test_every_producer_resolves_to_a_real_module_and_callable(self):
        for producer in PRODUCERS:
            self.assertEqual(producer_problems(producer.name), [], producer.name)

    def test_unknown_producer_reported(self):
        problems = producer_problems("ghost_agent")
        self.assertTrue(any("not in the producer registry" in p for p in problems))

    def test_missing_module_reported(self):
        broken = PRODUCER_BY_NAME["market_data_agent"]
        replacement = type(broken)("market_data_agent", "agents.no_such_module", "x")
        with patch.dict("core.feature_registry.PRODUCER_BY_NAME",
                        {"market_data_agent": replacement}):
            problems = producer_problems("market_data_agent")
        self.assertTrue(any("does not exist" in p for p in problems))

    def test_missing_callable_reported(self):
        broken = PRODUCER_BY_NAME["market_data_agent"]
        replacement = type(broken)(
            "market_data_agent", "agents.market_data_agent", "ghost_callable")
        with patch.dict("core.feature_registry.PRODUCER_BY_NAME",
                        {"market_data_agent": replacement}):
            problems = producer_problems("market_data_agent")
        self.assertTrue(any("not defined in agents.market_data_agent" in p for p in problems))

    def test_registry_problems_include_a_broken_producer(self):
        registry = build_default_registry()
        broken = PRODUCER_BY_NAME["market_data_agent"]
        replacement = type(broken)("market_data_agent", "agents.no_such_module", "x")
        with patch.dict("core.feature_registry.PRODUCER_BY_NAME",
                        {"market_data_agent": replacement}):
            problems = registry.problems()
        self.assertTrue(any("does not exist" in p for p in problems))

    def test_registering_a_spec_with_a_broken_producer_is_refused(self):
        broken = PRODUCER_BY_NAME["market_data_agent"]
        replacement = type(broken)("market_data_agent", "agents.no_such_module", "x")
        with patch.dict("core.feature_registry.PRODUCER_BY_NAME",
                        {"market_data_agent": replacement}):
            with self.assertRaises(ValueError) as ctx:
                FeatureRegistry().register(_valid_spec())
        self.assertIn("does not exist", str(ctx.exception))

    def test_unwired_producers_are_reported_not_failed(self):
        registry = build_default_registry()
        unwired = unwired_producers(registry)
        self.assertNotIn("market_data_agent", unwired)
        self.assertEqual(sorted(unwired), sorted(
            name for name in PRODUCER_BY_NAME if name != "market_data_agent"))
        self.assertEqual(registry.problems(), [])


class TestRevisedInputRejected(unittest.TestCase):
    """A definition change without a version bump is revised input."""

    def test_identical_reregistration_is_a_no_op(self):
        registry = FeatureRegistry()
        self.assertTrue(registry.register(_valid_spec()))
        self.assertFalse(registry.register(_valid_spec()))
        self.assertEqual(registry.revisions(), [])

    def test_same_version_different_definition_is_refused(self):
        registry = FeatureRegistry()
        registry.register(_valid_spec())
        with self.assertRaises(ValueError) as ctx:
            registry.register(_valid_spec(formula="totally different formula"))
        self.assertIn("revised input is rejected", str(ctx.exception))

    def test_version_bump_supersedes_and_keeps_history(self):
        registry = FeatureRegistry()
        registry.register(_valid_spec(version="market-feature-v1"))
        self.assertTrue(registry.register(_valid_spec(version="market-feature-v2")))
        revisions = registry.revisions()
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]["superseded_version"], "market-feature-v1")
        self.assertEqual(revisions[0]["version"], "market-feature-v2")
        self.assertNotEqual(revisions[0]["hash"], revisions[0]["superseded_hash"])
        self.assertEqual(registry.get("test_feature").version, "market-feature-v2")

    def test_persistence_refuses_an_invalid_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "registry.jsonl"
            with self.assertRaises(ValueError) as ctx:
                persist_feature_registry(FeatureRegistry(), path)
            self.assertIn("refusing to persist an invalid feature registry", str(ctx.exception))
            self.assertFalse(path.exists())


class TestModelFeatureGate(unittest.TestCase):
    """An unregistered feature cannot enter a production model."""

    def setUp(self):
        self.registry = build_default_registry()

    def test_declared_features_pass(self):
        self.assertEqual(
            model_feature_problems(["rsi", "ma_50"], self.registry, "technical_analysis"),
            [])

    def test_unregistered_feature_refused(self):
        problems = model_feature_problems(["rsi", "ghost"], self.registry)
        self.assertTrue(any("ghost" in p and "not registered" in p for p in problems))

    def test_incompatible_model_family_refused(self):
        problems = model_feature_problems(["rsi"], self.registry, "boosting")
        self.assertTrue(any("does not declare compatibility" in p for p in problems))

    def test_unknown_model_family_refused(self):
        problems = model_feature_problems(["rsi"], self.registry, "oracle")
        self.assertTrue(any("not a known model family" in p for p in problems))

    def test_empty_feature_set_refused(self):
        problems = model_feature_problems([], self.registry)
        self.assertTrue(any("at least one feature" in p for p in problems))

    def test_require_model_features_raises_or_returns_hash(self):
        digest = require_model_features(["rsi", "ma_50"], self.registry, "technical_analysis")
        self.assertEqual(digest, feature_set_hash(["rsi", "ma_50"], self.registry))
        with self.assertRaises(FeatureContractError):
            require_model_features(["ghost"], self.registry)

    def test_feature_set_hash_is_deterministic_and_order_insensitive(self):
        first = feature_set_hash(("rsi", "ma_50"), self.registry)
        second = feature_set_hash(("ma_50", "rsi", "rsi"), self.registry)
        self.assertEqual(first, second)
        self.assertNotEqual(first, feature_set_hash(("rsi",), self.registry))

    def test_feature_set_hash_changes_when_a_definition_changes(self):
        bumped = FeatureRegistry()
        bumped.register(_valid_spec("rsi"))
        bumped.register(_valid_spec("rsi", version="market-feature-v2"))
        self.assertNotEqual(
            feature_set_hash(["rsi"], bumped), feature_set_hash(["rsi"], self.registry))

    def test_feature_set_hash_marks_unregistered_names(self):
        with_ghost = feature_set_hash(["rsi", "ghost"], self.registry)
        self.assertNotEqual(with_ghost, feature_set_hash(["rsi"], self.registry))
        self.assertEqual(with_ghost, feature_set_hash(["ghost", "rsi"], self.registry))


def _feature_contract(
    name: str,
    value: float = 1.0,
    as_of: str = "2024-01-15 00:00:00",
    lookback: str = "14d",
    source_id: str = "yahoo_finance_chart",
    calculation_version: str = "market-feature-v1",
) -> dict:
    return {
        "name": name, "value": value, "as_of": as_of, "source_id": source_id,
        "published_time": as_of, "calculation_version": calculation_version,
        "lookback_period": lookback,
    }


def _score_result(contracts) -> SimpleNamespace:
    return SimpleNamespace(
        score=5.0, action="ANALYSIS_ONLY",
        feature_metadata={
            "current_score_features": list(contracts),
            "long_term_score_features": [],
        },
    )


class TestSnapshotConformanceExtras(unittest.TestCase):
    """Version, lookback, source and shape drift are all fatal."""

    def _snap(self, **overrides):
        contract = _feature_contract("rsi")
        contract.update(overrides)
        return {"features": {"rsi": contract}}

    def test_registered_lookback_mismatch_reported(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems(self._snap(lookback_period="20d"), reg)
        self.assertTrue(any("registered lookback" in p for p in problems))

    def test_contract_version_mismatch_reported(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems(
            self._snap(calculation_version="market-feature-v0"), reg)
        self.assertTrue(any("revised definition must be registered" in p for p in problems))

    def test_undeclared_source_reported(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems(self._snap(source_id="fred_macro"), reg)
        self.assertTrue(any("not a declared source dependency" in p for p in problems))

    def test_non_dict_contract_reported(self):
        reg = FeatureRegistry(); reg.register(_valid_spec("rsi"))
        problems = feature_contract_problems({"features": {"rsi": 5.0}}, reg)
        self.assertTrue(any("contract is not a dict" in p for p in problems))

    def test_real_snapshot_is_fully_conformant(self):
        from agents.market_data_agent import fetch_market_snapshot
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with patch("agents.market_data_agent.fetch_price_history", lambda *a, **k: frame.copy()):
            snapshot = fetch_market_snapshot("TEST", as_of)
        self.assertEqual(feature_contract_problems(snapshot, build_default_registry()), [])

    def test_feature_surface_digest_is_deterministic(self):
        reg = build_default_registry()
        surface = {"rsi": _feature_contract("rsi")}
        self.assertEqual(
            feature_surface_digest(surface, reg),
            feature_surface_digest(dict(surface), reg))
        changed = {"rsi": _feature_contract("rsi", value=2.0)}
        self.assertNotEqual(
            feature_surface_digest(surface, reg),
            feature_surface_digest(changed, reg))


class TestEngineFeatureGate(unittest.TestCase):
    """The walk-forward engine is a production-model consumer of the registry."""

    @staticmethod
    def _gate_frame():
        # Smallest geometry that still produces folds + a tail holdout:
        # 2 validation folds of 30 sessions and a 20-session holdout.
        return _frame(_ramp(180))

    def _run(self, store, contracts):
        with patch("core.backtest.engine.build_score") as mock_score:
            mock_score.return_value = _score_result(contracts)
            return run_walk_forward_backtest(
                "TEST", self._gate_frame(), fold_sessions=30, embargo_sessions=60,
                holdout_sessions=20, manifest_store_path=store)

    def test_declared_unregistered_feature_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "runs.jsonl"
            with patch.object(score_engine, "CURRENT_SCORE_FEATURES",
                              ("rsi", "ghost_feature")):
                with self.assertRaises(BacktestFeatureContractError) as ctx:
                    self._run(store, [])
        self.assertIn("ghost_feature", str(ctx.exception))
        self.assertIn("not registered", str(ctx.exception))

    def test_unregistered_feature_in_consumed_surface_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "runs.jsonl"
            with self.assertRaises(BacktestFeatureContractError) as ctx:
                self._run(store, [_feature_contract("ghost_feature")])
        self.assertIn("ghost_feature", str(ctx.exception))
        self.assertIn("non-conformant feature surface", str(ctx.exception))

    def test_version_drift_in_a_consumed_surface_refuses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "runs.jsonl"
            drifted = _feature_contract("rsi", calculation_version="market-feature-v0")
            with self.assertRaises(BacktestFeatureContractError) as ctx:
                self._run(store, [drifted])
        self.assertIn("revised definition", str(ctx.exception))

    def test_conformant_surface_is_verified_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "runs.jsonl"
            result = self._run(store, [_feature_contract("rsi")])
        self.assertEqual(result["folds"][0]["feature_surface"], "verified")
        self.assertGreater(result["folds"][0]["feature_surface_bars"], 0)
        self.assertEqual(result["holdout"]["feature_surface"], "verified")

    def test_real_scoring_path_verifies_the_production_surface(self):
        frame = _frame(_ramp(320))
        with offline_replay_seam({"TEST": frame}):
            replay = _replay_window(
                "TEST", frame, 280, 299, initial_capital=100_000.0,
                trade_notional=50_000.0, injected_labels=None,
                cost_table=COST_TABLE_V2,
            )
        self.assertEqual(replay["feature_surface"], "verified")
        self.assertEqual(replay["feature_surface_bars"], 20)

    def test_real_score_result_consumes_exactly_its_declared_features(self):
        """The live model consumes its declared set — all of it registered."""
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with offline_replay_seam({"TEST": frame}):
            result = score_engine.build_score("TEST", as_of, persist_audit=False)
            surface = _exposed_feature_surface(result)
        declared = set(score_engine.CURRENT_SCORE_FEATURES) | set(score_engine.LONG_TERM_SCORE_FEATURES)
        self.assertEqual(set(surface), declared)
        self.assertEqual(
            model_feature_problems(surface, build_default_registry(), "technical_analysis"),
            [])

    def test_score_result_without_a_surface_is_reported_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "runs.jsonl"
            result = self._run(store, [])
        self.assertEqual(
            result["folds"][0]["feature_surface"], "unexposed_by_score_result")
        self.assertEqual(result["folds"][0]["feature_surface_bars"], 0)


if __name__ == "__main__":
    unittest.main()

