"""Framing contract tests (Sprint V8) — a backtest is evidence about historical
behavior under explicit assumptions, not proof the future behaves the same way.

The framing block is part of the engine contract now: every completed run carries
a versioned framing block, and the canonical interpretation never changes with
performance. The tests below pin that behavior.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.backtest import (
    FRAMING_VERSION,
    BacktestManifestError,
    build_framing_block,
    framing_problems,
    load_framing_snapshots,
    persist_framing_snapshot,
    run_walk_forward_backtest,
)
from core.framing import (
    FRAMING_LIMITATIONS,
    FRAMING_NON_CLAIMS,
    FRAMING_STATEMENT,
    FRAMING_STORE_PATH,
    REQUIRED_ASSUMPTION_KEYS,
    assumption_entries,
)
from core.backtest.costs import COST_TABLE_V2
# shared manifest input value; kept in sync with core via the cost_table we
# pass into the manifest under versions.cost_table
_COST_TABLE_MANIFEST_VALUE: str = "backtest-cost-table-v2"


def _manifest_base() -> dict:
    return {
        "ticker": "AAPL",
        "versions": {
            "cost_table": _COST_TABLE_MANIFEST_VALUE,
            "strategy": "score-engine-v1",
            "universe": "universe-ledger-v1",
            "framing": FRAMING_VERSION,
        },
        "config": {
            "embargo_sessions": 5,
            "fold_sessions": 252,
            "holdout_sessions": 63,
            "enter_score": 0.65,
            "exit_score": 0.30,
            "universe": "universe-ledger-v1",
        },
        "provider_overrides": {"price_basis": "price_return_split_adjusted"},
        "data_digest": "alpha_vantage_aapl_2020_2024_v1",
    }


def _aggregate(decision_count: int = 3) -> dict:
    return {
        "decision_count": decision_count,
        "total_trades": decision_count,
        "deltas_usd": [100.0, -50.0, 25.0][:decision_count],
        "win_rate": 0.6,
        "avg_win": 100.0,
        "avg_loss": -50.0,
        "profit_factor": 1.5,
        "max_drawdown_pct": 0.10,
        "volatility_annualized_pct": 0.18,
        "annualized_return_pct": 0.12,
        "sharpe_ratio": 1.2,
        "calmar_ratio": 1.2,
    }


def _frame(closes):
    index = pd.date_range("2022-01-03", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": list(closes),
            "Low": list(closes),
            "Close": list(closes),
            "Volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * i for i in range(sessions)]


# --- unit: assumption_entries -------------------------------------------------


class TestAssumptionEntries(unittest.TestCase):
    def test_sources_match_manifest_surface(self):
        m = _manifest_base()
        entries = assumption_entries(m)
        keys = {e["key"] for e in entries}
        self.assertEqual(keys, set(REQUIRED_ASSUMPTION_KEYS))
        by_key = {e["key"]: e for e in entries}
        self.assertEqual(by_key["cost_table"]["value"], _COST_TABLE_MANIFEST_VALUE)
        self.assertEqual(by_key["cost_table"]["sourced_from"], "manifest.versions.cost_table")
        self.assertIsNotNone(by_key["strategy"]["value"])
        self.assertIn(by_key["strategy"]["value"], ("score-engine-v1", "score-engine-v2"))
        self.assertEqual(by_key["strategy"]["sourced_from"], "manifest.versions.strategy")
        self.assertEqual(by_key["price_basis"]["sourced_from"], "manifest.provider_overrides.price_basis")
        self.assertEqual(by_key["universe"]["sourced_from"], "manifest.config.universe")
        self.assertEqual(by_key["data_digest"]["sourced_from"], "manifest.data_digest")
        self.assertEqual(by_key["data_digest"]["value"], "alpha_vantage_aapl_2020_2024_v1")

    def test_unset_values_understandable_not_clean(self):
        m = {"ticker": "XYZ"}
        entries = assumption_entries(m)
        by_key = {e["key"]: e for e in entries}
        self.assertIsNone(by_key["cost_table"]["value"])
        self.assertIsNone(by_key["price_basis"]["value"])


# --- unit: framing_problems ---------------------------------------------------


class TestFramingProblems(unittest.TestCase):
    def test_full_manifest_is_framable(self):
        m = _manifest_base()
        self.assertEqual(framing_problems(m), [])

    def test_missing_cost_table_is_flagged(self):
        m = _manifest_base()
        del m["versions"]["cost_table"]
        problems = framing_problems(m)
        self.assertTrue(any("cost_table" in p for p in problems))
        self.assertFalse(any("strategy" in p for p in problems))

    def test_missing_framing_version_is_flagged(self):
        m = _manifest_base()
        del m["versions"]["framing"]
        problems = framing_problems(m)
        self.assertTrue(any("framing version is missing" in p for p in problems))

    def test_missing_price_basis_is_flagged(self):
        m = _manifest_base()
        del m["provider_overrides"]["price_basis"]
        problems = framing_problems(m)
        self.assertTrue(any("price_basis" in p for p in problems))

    def test_unframable_drives_manifest_error(self):
        m = _manifest_base()
        del m["versions"]["cost_table"]
        problems = framing_problems(m)
        self.assertTrue(problems)
        with self.assertRaises(BacktestManifestError):
            raise BacktestManifestError("; ".join(problems))


# --- unit: build_framing_block ------------------------------------------------


class TestBuildFramingBlock(unittest.TestCase):
    def test_versioned_and_complete(self):
        m = _manifest_base()
        block = build_framing_block(m)
        self.assertEqual(block["framing_version"], FRAMING_VERSION)
        self.assertEqual(block["statement"], FRAMING_STATEMENT)
        self.assertIn(block["evidence_status"], {"historical_evidence", "historical_evidence_no_decisions"})
        self.assertIsInstance(block["assumptions"], list)
        self.assertIsInstance(block["limitations"], list)
        self.assertIsInstance(block["does_not_claim"], list)
        expected_limitations = [
            {"key": k, "detail": v} for k, v in FRAMING_LIMITATIONS.items()
        ]
        expected_non_claims = [
            {"key": k, "detail": v} for k, v in FRAMING_NON_CLAIMS.items()
        ]
        self.assertEqual(block["limitations"], expected_limitations)
        self.assertEqual(block["does_not_claim"], expected_non_claims)

    def test_source_is_manifest_never_hardcoded(self):
        m = _manifest_base()
        block = build_framing_block(m)
        by_key = {a["key"]: a for a in block["assumptions"]}
        self.assertEqual(by_key["cost_table"]["value"], _COST_TABLE_MANIFEST_VALUE)
        self.assertEqual(by_key["data_digest"]["value"], "alpha_vantage_aapl_2020_2024_v1")
        self.assertEqual(by_key["price_basis"]["value"], "price_return_split_adjusted")


# --- unit: framing does not upgrade with performance --------------------------


class TestFramingDoesNotUpgradeWithPerformance(unittest.TestCase):
    def test_strong_results_stay_historical_evidence(self):
        m = _manifest_base()
        strong = {
            "decision_count": 120,
            "total_trades": 120,
            "deltas_usd": [100.0] * 120,
            "win_rate": 0.85,
            "avg_win": 200.0,
            "avg_loss": -50.0,
            "profit_factor": 6.0,
            "max_drawdown_pct": 0.02,
            "volatility_annualized_pct": 0.12,
            "annualized_return_pct": 0.40,
            "sharpe_ratio": 4.0,
            "calmar_ratio": 20.0,
        }
        weak = {
            "decision_count": 120,
            "total_trades": 120,
            "deltas_usd": [-50.0] * 120,
            "win_rate": 0.25,
            "avg_win": 100.0,
            "avg_loss": -200.0,
            "profit_factor": 0.2,
            "max_drawdown_pct": 0.55,
            "volatility_annualized_pct": 0.40,
            "annualized_return_pct": -0.25,
            "sharpe_ratio": -0.5,
            "calmar_ratio": -0.45,
        }
        strong_block = build_framing_block(m, strong)
        weak_block = build_framing_block(m, weak)
        self.assertEqual(strong_block["framing_version"], weak_block["framing_version"])
        self.assertEqual(strong_block["statement"], weak_block["statement"])
        self.assertEqual(strong_block["evidence_status"], weak_block["evidence_status"])
        self.assertEqual(strong_block["evidence_status"], "historical_evidence")
        self.assertEqual(strong_block["assumptions"], weak_block["assumptions"])
        self.assertEqual(strong_block["limitations"], weak_block["limitations"])
        self.assertEqual(strong_block["does_not_claim"], weak_block["does_not_claim"])
        claim_keys = {d["key"] for d in strong_block["does_not_claim"]}
        self.assertIn("not_proof_of_future", claim_keys)

    def test_no_decisions_stays_framed_evidence(self):
        m = _manifest_base()
        block = build_framing_block(m, {"decision_count": 0})
        self.assertEqual(block["evidence_status"], "historical_evidence_no_decisions")
        self.assertIn("no trade evidence", block["evidence_note"])


# --- unit: append-only framing store ------------------------------------------

class TestFramingStore(unittest.TestCase):
    def test_persist_then_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "framing.jsonl"
            snapshot = build_framing_block(_manifest_base(), _aggregate())
            appended = persist_framing_snapshot(
                run_type="backtest",
                run_hash="abc123",
                snapshot=snapshot,
                path=path,
            )
            self.assertTrue(appended)
            records = load_framing_snapshots(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_type"], "backtest")
            self.assertEqual(records[0]["run_hash"], "abc123")
            self.assertEqual(records[0]["snapshot"]["framing_version"], FRAMING_VERSION)

    def test_idempotent_recompute_appends_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "framing.jsonl"
            snapshot = build_framing_block(_manifest_base(), _aggregate(seed := 2))
            self.assertTrue(persist_framing_snapshot("backtest", "hash-1", snapshot, path))
            self.assertFalse(persist_framing_snapshot("backtest", "hash-1", snapshot, path))
            self.assertEqual(len(load_framing_snapshots(path)), 1)

    def test_same_key_different_content_raises_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "framing.jsonl"
            a = build_framing_block(_manifest_base(), _aggregate(seed := 1))
            b = build_framing_block(_manifest_base(), _aggregate(seed := 3))
            persist_framing_snapshot("backtest", "hash-1", a, path)
            with self.assertRaises(ValueError) as ctx:
                persist_framing_snapshot("backtest", "hash-1", b, path)
            self.assertIn("framing integrity violation", str(ctx.exception))

    def test_malformed_json_raises_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "framing.jsonl"
            path.write_text("not-json-at-all\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_framing_snapshots(path)
            self.assertRegex(str(ctx.exception), r"line 1 is not valid JSON")

    def test_store_path_default_is_data_framing_jsonl(self):
        expected = pathlib.Path(__file__).resolve().parent.parent / "data" / "framing.jsonl"
        self.assertEqual(FRAMING_STORE_PATH, expected)


# --- engine integration -------------------------------------------------------

class _FramingEngineTestsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        closes = _ramp(200, step=2.0) + _ramp(200, step=-2.0, base=100.0 + 2.0 * 199)
        cls.frame = _frame(closes)
        cls.geometry_kwargs = dict(fold_sessions=50, embargo_sessions=60, holdout_sessions=40)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.manifest_store = pathlib.Path(cls._tmp.name) / "backtest_runs.jsonl"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _scripted_run(self, **kwargs):
        def fake_build_score(ticker, as_of, persist_audit=False, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

        with patch("core.backtest.engine.build_score", fake_build_score):
            return run_walk_forward_backtest(
                "TEST",
                self.frame,
                manifest_store_path=self.manifest_store,
                **kwargs,
            )


class FramingBlockCarriedByCompletedRun(_FramingEngineTestsBase):
    def test_run_contains_framing_field(self):
        run = self._scripted_run()
        self.assertIn("framing", run)
        block = run["framing"]
        self.assertEqual(block["framing_version"], FRAMING_VERSION)
        self.assertEqual(block["statement"], FRAMING_STATEMENT)
        self.assertIn("assumptions", block)
        self.assertIn("limitations", block)
        self.assertIn("does_not_claim", block)
        claim_keys = {d["key"] for d in block["does_not_claim"]}
        self.assertIn("not_proof_of_future", claim_keys)

    def test_framing_block_is_sourced_from_manifest(self):
        run = self._scripted_run()
        block = run["framing"]
        by_key = {a["key"]: a for a in block["assumptions"]}
        self.assertEqual(by_key["cost_table"]["value"], _COST_TABLE_MANIFEST_VALUE)
        u = by_key["universe"]["value"]
        self.assertIsInstance(u, dict)
        self.assertEqual(u["type"], "single_ticker_unverified")
        self.assertIn(u["survivorship_status"], ("unverifiable_no_ledger",))
        pb = by_key["price_basis"]["value"]
        self.assertIsInstance(pb, str)
        self.assertIn("split-adjusted", pb)
        self.assertIn("dividends NOT reinvested", pb)

    def test_framing_does_not_change_between_identical_reruns(self):
        first = self._scripted_run()
        second = self._scripted_run()
        self.assertEqual(first["framing"], second["framing"])


class FramingGateRefusesUnframableRun(_FramingEngineTestsBase):
    def test_missing_framing_version_in_manifest_refuses_run(self):
        def broken_build_manifest(*args, **kwargs):
            m = {
                "manifest_version": "backtest-manifest-v1",
                "ticker": "TEST",
                "versions": {"cost_table": _COST_TABLE_MANIFEST_VALUE, "universe": "universe-ledger-v1"},
                "config": {
                    "embargo_sessions": 60,
                    "fold_sessions": 50,
                    "holdout_sessions": 40,
                    "enter_score": 0.65,
                    "exit_score": 0.30,
                    "universe": "universe-ledger-v1",
                },
                "provider_overrides": {"price_basis": "price_return_split_adjusted"},
                "data_digest": "test_digest",
                "run_hash": "fake",
            }
            return m

        with patch("core.backtest.engine.build_manifest", broken_build_manifest):
            with self.assertRaises(BacktestManifestError) as ctx:
                self._scripted_run()
        self.assertIn("framing version is missing", str(ctx.exception))


class FramingPersistedBesideManifest(_FramingEngineTestsBase):
    def test_framing_record_persisted(self):
        run = self._scripted_run()
        path = self.manifest_store.parent / "framing.jsonl"
        records = load_framing_snapshots(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["run_type"], "backtest")
        self.assertEqual(records[0]["run_hash"], run["manifest"]["run_hash"])
        self.assertEqual(records[0]["snapshot"]["framing_version"], FRAMING_VERSION)

    def test_framing_persistence_is_idempotent(self):
        first = self._scripted_run()
        second = self._scripted_run()
        path = self.manifest_store.parent / "framing.jsonl"
        records = load_framing_snapshots(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["run_hash"], first["manifest"]["run_hash"])


if __name__ == "__main__":
    unittest.main()

