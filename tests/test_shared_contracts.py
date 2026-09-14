"""Sprint V5: shared feature/scoring contracts — research == production.

The binding rule (docs/validation.md, Shared Research Contract): the
backtester consumes the same feature contracts and scorers as live — no
side research dataset, no parallel ad hoc path. These tests pin that rule
through `core.contract_verification`:

- feature contracts on the live snapshot are complete and version-current;
- the offline research seam changes only the DATA SOURCE: a direct call
  and the same call under the seam produce byte-identical snapshots, score
  layers, and V1 labels (diff-precise verifiers must find zero problems);
- the engine's manifest versions must equal the live config constants —
  version drift is a failure, not a warning;
- structural check: research modules call the canonical producers and
  never define parallel implementations (the detector itself is tested
  against a seeded violation).

Hermetic: every fetch is served from the same synthetic frame on both
paths, so the comparison is purely seam-vs-direct computation.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core import config as core_config
from core.contract_verification import (
    feature_contract_problems,
    engine_version_problems,
    parallel_path_problems,
    score_identity_problems,
    snapshot_field_problems,
)
from agents.market_data_agent import fetch_market_snapshot
from core.labels import build_outcome_labels
from core.score_engine import build_score
from core.backtest.engine import offline_replay_seam, run_walk_forward_backtest


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
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
    return [base + step * index for index in range(sessions)]


@contextmanager
def _direct_patches(frame):
    """The production-reference configuration: canonical providers served
    the same frame, keyless provider env — NO research seam involved."""

    def provider(ticker, *args, **kwargs):
        return frame.copy()

    with patch("agents.market_data_agent.fetch_price_history", provider), \
            patch("core.regime_agent.fetch_price_history", provider), \
            patch("core.labels.fetch_price_history", provider), \
            patch.dict(os.environ, {"NEWS_PROVIDER_API_KEY": "", "FRED_API_KEY": ""}):
        yield


class FeatureContractConformanceTests(unittest.TestCase):
    def test_feature_contracts_conform_on_a_snapshot(self):
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with _direct_patches(frame):
            snapshot = fetch_market_snapshot("TEST", as_of)
        self.assertEqual(feature_contract_problems(snapshot), [])

    def test_feature_contract_problems_are_diff_precise(self):
        frame = _frame(_ramp(320))
        as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        with _direct_patches(frame):
            snapshot = fetch_market_snapshot("TEST", as_of)
        snapshot["features"]["rsi"]["lookback_period"] = ""
        snapshot["features"]["atr_14"]["calculation_version"] = "market-feature-v0"
        problems = feature_contract_problems(snapshot)
        self.assertIn("rsi: provenance field lookback_period missing/empty", problems)
        self.assertIn(
            "atr_14: calculation_version 'market-feature-v0' != 'market-feature-v1'", problems
        )
        self.assertNotIn("rsi: contract carries no value key", problems)

    def test_snapshot_without_features_is_a_problem(self):
        problems = feature_contract_problems({"features": {}})
        self.assertEqual(problems, ["snapshot carries no feature contracts"])


class ResearchProductionIdentityTests(unittest.TestCase):
    """The seam changes only the data source — never the computation."""

    @classmethod
    def setUpClass(cls):
        cls.frame = _frame(_ramp(320))
        cls.as_of = cls.frame.index[250].strftime("%Y-%m-%d %H:%M:%S")
        with _direct_patches(cls.frame):
            cls.live_snapshot = fetch_market_snapshot("TEST", cls.as_of)
            cls.live_score = build_score("TEST", cls.as_of, persist_audit=False)
            cls.live_labels = build_outcome_labels("TEST", cls.as_of)
        with offline_replay_seam({"TEST": cls.frame}):
            cls.replay_snapshot = fetch_market_snapshot("TEST", cls.as_of)
            cls.replay_score = build_score("TEST", cls.as_of, persist_audit=False)
            cls.replay_labels = build_outcome_labels("TEST", cls.as_of)

    def test_snapshots_are_identical_across_paths(self):
        self.assertEqual(
            snapshot_field_problems(self.live_snapshot, self.replay_snapshot), []
        )

    def test_score_layers_are_identical_across_paths(self):
        self.assertEqual(
            score_identity_problems(self.live_score, self.replay_score), []
        )

    def test_v1_labels_are_identical_across_paths(self):
        self.assertEqual(self.live_labels, self.replay_labels)
        self.assertEqual(self.live_labels["labels_hash"], self.replay_labels["labels_hash"])
        for horizon in ("1d", "5d", "20d", "60d"):
            self.assertEqual(
                self.live_labels["horizons"][horizon],
                self.replay_labels["horizons"][horizon],
                horizon,
            )


class EngineVersionConsistencyTests(unittest.TestCase):
    """Manifest versions must equal the live config constants — no copies."""

    def test_engine_manifest_versions_match_live_constants(self):
        frame = _frame(_ramp(260, step=1.0))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "backtest_runs.jsonl"

            def fake_build_score(ticker, as_of, persist_audit=False, **kwargs):
                return SimpleNamespace(score=5.0, action="ANALYSIS_ONLY")

            with patch("core.backtest.engine.build_score", fake_build_score):
                result = run_walk_forward_backtest(
                    "TEST", frame,
                    fold_sessions=50, embargo_sessions=60, holdout_sessions=40,
                    manifest_store_path=store,
                )
        self.assertEqual(engine_version_problems(result["manifest"]["versions"]), [])

    def test_version_drift_is_reported_precisely(self):
        problems = engine_version_problems({
            **core_config.__dict__.copy(),
            "ensemble": "ensemble-v1",
            "outcome_label": "outcome-label-v0",
        })
        self.assertIn(
            f"versions[ensemble]: 'ensemble-v1' != '{core_config.ENSEMBLE_VERSION}'", problems
        )
        self.assertIn(
            f"versions[outcome_label]: 'outcome-label-v0' != '{core_config.OUTCOME_LABEL_VERSION}'",
            problems,
        )
        self.assertTrue(
            any("versions[cost_table] missing" in problem for problem in problems),
            problems,
        )

    def test_feature_registry_version_is_required_and_drift_detected(self):
        """M1: the registry version participates in the manifest contract."""
        problems = engine_version_problems({
            **core_config.__dict__.copy(),
            "feature_registry": "feature-registry-v0",
        })
        self.assertIn(
            f"versions[feature_registry]: 'feature-registry-v0' "
            f"!= '{core_config.FEATURE_REGISTRY_VERSION}'",
            problems,
        )
        missing = engine_version_problems({
            **core_config.__dict__.copy(),
            "feature_registry": None,
        })
        self.assertTrue(
            any("versions[feature_registry]" in problem for problem in missing), missing)

    def test_expected_versions_come_from_the_live_constants(self):
        from core.contract_verification import expected_engine_versions

        expected = expected_engine_versions()
        self.assertEqual(expected["ensemble"], core_config.ENSEMBLE_VERSION)
        self.assertEqual(expected["market_feature"], core_config.MARKET_FEATURE_VERSION)
        self.assertEqual(expected["outcome_label"], core_config.OUTCOME_LABEL_VERSION)


class NoParallelPathTests(unittest.TestCase):
    """Structural pin: research calls production, never re-implements it."""

    def test_research_modules_are_clean(self):
        self.assertEqual(parallel_path_problems(), [])

    def test_the_detector_catches_a_seeded_violation(self):
        probe = Path(__file__).resolve().parent.parent / "core" / "backtest" / "_probe_parallel.py"
        try:
            probe.write_text(
                "def _score_current_time(snapshot, momentum_damping=1.0):\n"
                "    return 0.0  # a parallel scorer sneaking into research\n",
                encoding="utf-8",
            )
            problems = parallel_path_problems()
            self.assertTrue(
                any("_probe_parallel.py" in problem and "def _score_current_time" in problem
                    for problem in problems),
                problems,
            )
        finally:
            probe.unlink(missing_ok=True)
        # The violation is gone once the probe module is removed.
        self.assertEqual(parallel_path_problems(), [])


if __name__ == "__main__":
    unittest.main()
