"""Model registry and artifact tracking tests (Sprint M2, board numbering).

The M2 acceptance criterion, pinned:

    "Referencing an unapproved/retired model in the score path is impossible
    without an explicit override that itself is audited."

Plus the rules the board states in code terms: a model referenced by a live
decision must be `approved`; promotion candidate->approved requires an
out-of-sample comparison against the incumbent plus a human `approved_by`;
retirement never deletes artifacts.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
import unittest.mock

from core.config import (
    MODEL_REGISTRY_VERSION,
    MODEL_STATUS_APPROVED,
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_RETIRED,
)
from core.model_registry import (
    ModelEntry,
    ModelNotApprovedError,
    ModelRegistry,
    ModelRegistryError,
    approver_problems,
    build_default_model_registry,
    entry_problems,
    load_model_manifest,
    load_model_registry,
    oos_comparison_problems,
    persist_model_registry,
    require_live_model,
)

_APPROVED_AT = "2026-09-16T00:00:00+00:00"


def _candidate(version: str = "technical-v2", family: str = "technical_analysis") -> ModelEntry:
    return ModelEntry(
        model_version=version,
        family=family,
        feature_set_version="feature-set-1",
        training_data_cutoff="2026-06-30",
        artifact_uri=f"models/{version}.joblib",
        dataset_hash="d" * 64,
    )


def _winning_comparison(incumbent: str | None = "technical-v1") -> dict:
    return {
        "primary_metric": "oos_sharpe",
        "candidate_value": 1.4,
        "incumbent_value": 1.1,
        "sample": 500,
        "candidate_version": "technical-v2",
        "incumbent_version": incumbent,
        "higher_is_better": True,
    }


class TestEntryValidation(unittest.TestCase):
    def test_valid_candidate_has_no_problems(self) -> None:
        self.assertEqual(entry_problems(_candidate()), [])

    def test_unknown_family_is_refused(self) -> None:
        entry = _candidate()
        entry.family = "telepathy"
        self.assertTrue(any("not a known model family" in p for p in entry_problems(entry)))

    def test_unknown_status_is_refused(self) -> None:
        entry = _candidate()
        entry.status = "probably_fine"
        self.assertTrue(any("not a known status" in p for p in entry_problems(entry)))

    def test_feature_set_version_is_required(self) -> None:
        entry = _candidate()
        entry.feature_set_version = ""
        self.assertTrue(any("feature_set_version" in p for p in entry_problems(entry)))

    def test_approved_entry_must_record_its_approver(self) -> None:
        entry = _candidate()
        entry.status = MODEL_STATUS_APPROVED
        problems = entry_problems(entry)
        self.assertTrue(any("approved_by" in p for p in problems))
        self.assertTrue(any("approved_at" in p for p in problems))

    def test_model_cannot_be_its_own_parent(self) -> None:
        entry = _candidate()
        entry.parent_version = entry.model_version
        self.assertTrue(any("parent_version" in p for p in entry_problems(entry)))

    def test_duplicate_registration_is_refused(self) -> None:
        registry = ModelRegistry()
        registry.register(_candidate())
        with self.assertRaises(ModelRegistryError) as ctx:
            registry.register(_candidate())
        self.assertIn("already registered", str(ctx.exception))

    def test_unknown_parent_is_refused(self) -> None:
        entry = _candidate()
        entry.parent_version = "does-not-exist-v1"
        with self.assertRaises(ModelRegistryError):
            ModelRegistry().register(entry)

    def test_entry_hash_is_deterministic(self) -> None:
        self.assertEqual(_candidate().canonical_hash(), _candidate().canonical_hash())

    def test_status_change_changes_the_hash(self) -> None:
        entry = _candidate()
        before = entry.canonical_hash()
        entry.status = MODEL_STATUS_APPROVED
        self.assertNotEqual(before, entry.canonical_hash())


class TestPromotionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_model_registry()
        self.registry.register(_candidate())

    def test_valid_promotion_succeeds(self) -> None:
        entry = self.registry.promote(
            "technical-v2", "oren", _APPROVED_AT, _winning_comparison()
        )
        self.assertEqual(entry.status, MODEL_STATUS_APPROVED)
        self.assertEqual(entry.approved_by, "oren")
        self.assertEqual(entry.oos_comparison["primary_metric"], "oos_sharpe")

    def test_promotion_without_oos_comparison_is_refused(self) -> None:
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.promote("technical-v2", "oren", _APPROVED_AT, {})
        self.assertIn("out-of-sample comparison", str(ctx.exception))

    def test_candidate_that_loses_cannot_be_promoted(self) -> None:
        """'It did not win' is a valid, successful outcome."""
        losing = {**_winning_comparison(), "candidate_value": 0.9}
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.promote("technical-v2", "oren", _APPROVED_AT, losing)
        self.assertIn("does not beat incumbent", str(ctx.exception))

    def test_loss_metric_direction_is_honoured(self) -> None:
        """For a loss metric, lower must win."""
        comparison = {
            **_winning_comparison(),
            "primary_metric": "brier_score",
            "candidate_value": 0.18,
            "incumbent_value": 0.22,
            "higher_is_better": False,
        }
        entry = self.registry.promote("technical-v2", "oren", _APPROVED_AT, comparison)
        self.assertEqual(entry.status, MODEL_STATUS_APPROVED)

    def test_automated_approver_is_refused(self) -> None:
        """Governance sign-off cannot be automated."""
        for actor in ("ci", "bot", "system", "automatic", ""):
            with self.subTest(actor=actor):
                with self.assertRaises(ModelRegistryError):
                    self.registry.promote(
                        "technical-v2", actor, _APPROVED_AT, _winning_comparison()
                    )

    def test_comparison_must_name_the_real_incumbent(self) -> None:
        wrong = {**_winning_comparison(), "incumbent_version": "ghost-v1"}
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.promote("technical-v2", "oren", _APPROVED_AT, wrong)
        self.assertIn("incumbent", str(ctx.exception))

    def test_non_positive_sample_is_refused(self) -> None:
        bad = {**_winning_comparison(), "sample": 0}
        with self.assertRaises(ModelRegistryError):
            self.registry.promote("technical-v2", "oren", _APPROVED_AT, bad)

    def test_promoting_an_unregistered_model_is_refused(self) -> None:
        with self.assertRaises(ModelRegistryError):
            self.registry.promote("ghost-v9", "oren", _APPROVED_AT, _winning_comparison())

    def test_already_approved_cannot_be_repromoted(self) -> None:
        with self.assertRaises(ModelRegistryError):
            self.registry.promote("technical-v1", "oren", _APPROVED_AT, _winning_comparison())

    def test_retired_model_cannot_be_promoted(self) -> None:
        self.registry.retire("technical-v2", _APPROVED_AT, "abandoned")
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.promote("technical-v2", "oren", _APPROVED_AT, _winning_comparison())
        self.assertIn("retired", str(ctx.exception))

    def test_approver_problems_flags_non_humans(self) -> None:
        self.assertTrue(approver_problems("pipeline"))
        self.assertEqual(approver_problems("oren"), [])

    def test_oos_problems_requires_every_field(self) -> None:
        for missing in ("primary_metric", "candidate_value", "incumbent_value", "sample"):
            with self.subTest(missing=missing):
                comparison = {k: v for k, v in _winning_comparison().items() if k != missing}
                self.assertTrue(oos_comparison_problems(comparison, "technical-v2", "technical-v1"))


class TestRetirementKeepsArtifacts(unittest.TestCase):
    def test_retirement_never_deletes(self) -> None:
        registry = build_default_model_registry()
        before = registry.get("technical-v1").artifact_uri
        registry.retire("technical-v1", _APPROVED_AT, "superseded")
        entry = registry.get("technical-v1")
        self.assertIsNotNone(entry, "the entry was deleted")
        self.assertEqual(entry.status, MODEL_STATUS_RETIRED)
        self.assertEqual(entry.artifact_uri, before)
        self.assertEqual(entry.retired_reason, "superseded")

    def test_retired_model_leaves_the_approved_set(self) -> None:
        registry = build_default_model_registry()
        registry.retire("technical-v1", _APPROVED_AT, "superseded")
        self.assertNotIn("technical-v1", registry.approved())

    def test_retirement_requires_a_timestamp(self) -> None:
        registry = build_default_model_registry()
        with self.assertRaises(ModelRegistryError):
            registry.retire("technical-v1", "")


class TestLivePathGate(unittest.TestCase):
    """The M2 acceptance criterion."""

    def setUp(self) -> None:
        self.registry = build_default_model_registry()

    def test_approved_model_resolves(self) -> None:
        record = require_live_model("technical-v1", self.registry)
        self.assertEqual(record["status"], MODEL_STATUS_APPROVED)
        self.assertFalse(record["override"])

    def test_unregistered_model_is_refused(self) -> None:
        with self.assertRaises(ModelNotApprovedError):
            require_live_model("ghost-v9", self.registry)

    def test_candidate_model_is_refused(self) -> None:
        self.registry.register(_candidate())
        with self.assertRaises(ModelNotApprovedError) as ctx:
            require_live_model("technical-v2", self.registry)
        self.assertIn("candidate", str(ctx.exception))

    def test_retired_model_is_refused(self) -> None:
        self.registry.retire("technical-v1", _APPROVED_AT, "superseded")
        with self.assertRaises(ModelNotApprovedError) as ctx:
            require_live_model("technical-v1", self.registry)
        self.assertIn("retired", str(ctx.exception))

    def test_override_is_possible_but_audited(self) -> None:
        """Acceptance: impossible WITHOUT an explicit override that is audited."""
        record = require_live_model(
            "ghost-v9", self.registry, override_reason="incident-42 manual review"
        )
        self.assertTrue(record["override"])
        self.assertEqual(record["override_reason"], "incident-42 manual review")
        self.assertEqual(record["status"], "unregistered")

    def test_blank_override_reason_does_not_unlock(self) -> None:
        for reason in ("", "   ", None):
            with self.subTest(reason=reason):
                with self.assertRaises(ModelNotApprovedError):
                    require_live_model("ghost-v9", self.registry, override_reason=reason)


class TestOrchestratorUsesTheRegistry(unittest.TestCase):
    def test_no_hardcoded_model_version_literals_remain(self) -> None:
        """The board: those strings become registry lookups."""
        source = (
            pathlib.Path(__file__).resolve().parent.parent / "core" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        for literal in ('model_version="market-data-v1"',
                        'model_version="technical-v1"',
                        'model_version="fundamental-v1"'):
            with self.subTest(literal=literal):
                self.assertNotIn(literal, source)

    def test_seeded_models_are_all_approved(self) -> None:
        registry = build_default_model_registry()
        self.assertEqual(len(registry.all_models()), 3)
        self.assertEqual(len(registry.approved()), 3)

    def test_incumbent_lookup_finds_an_approved_model(self) -> None:
        registry = build_default_model_registry()
        self.assertIsNotNone(registry.incumbent("technical_analysis"))

    def test_incumbent_is_none_when_all_retired(self) -> None:
        registry = build_default_model_registry()
        for version in list(registry.all_models()):
            registry.retire(version, _APPROVED_AT, "cleared")
        self.assertIsNone(registry.incumbent("technical_analysis"))


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "manifest.json"
        self.addCleanup(self._tmp.cleanup)

    def test_round_trip_preserves_the_registry(self) -> None:
        registry = build_default_model_registry()
        persist_model_registry(registry, self.path)
        loaded = load_model_registry(self.path)
        self.assertEqual(loaded.registry_hash(), registry.registry_hash())
        self.assertEqual(set(loaded.all_models()), set(registry.all_models()))

    def test_persist_is_idempotent(self) -> None:
        registry = build_default_model_registry()
        first = persist_model_registry(registry, self.path)
        second = persist_model_registry(registry, self.path)
        self.assertEqual(first["registry_hash"], second["registry_hash"])

    def test_manifest_declares_its_version(self) -> None:
        persist_model_registry(build_default_model_registry(), self.path)
        self.assertEqual(
            load_model_manifest(self.path)["registry_version"], MODEL_REGISTRY_VERSION
        )

    def test_promotion_changes_the_persisted_hash(self) -> None:
        registry = build_default_model_registry()
        first = persist_model_registry(registry, self.path)
        registry.register(_candidate())
        registry.promote("technical-v2", "oren", _APPROVED_AT, _winning_comparison())
        second = persist_model_registry(registry, self.path)
        self.assertNotEqual(first["registry_hash"], second["registry_hash"])

    def test_missing_manifest_reads_empty(self) -> None:
        self.assertEqual(load_model_manifest(self.path), {})

    def test_malformed_manifest_raises_loudly(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelRegistryError):
            load_model_manifest(self.path)

    def test_committed_manifest_matches_the_default_registry(self) -> None:
        """The tracked manifest must not drift from the code."""
        manifest = load_model_manifest()
        if not manifest:
            self.skipTest("models/manifest.json not present")
        self.assertEqual(
            manifest["registry_hash"], build_default_model_registry().registry_hash()
        )


if __name__ == "__main__":
    unittest.main()


class TestAuditTrailCarriesResolutions(unittest.TestCase):
    """The override is only 'audited' if it reaches the permanent record."""

    def test_audit_event_carries_model_resolutions(self) -> None:
        from core.audit_store import AUDIT_SCHEMA_VERSION, persist_decision_audit

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "audit.jsonl"
            with unittest.mock.patch("core.audit_store.AUDIT_LOG_PATH", log):
                event = persist_decision_audit({
                    "ticker": "NVDA",
                    "as_of": "2026-01-05 00:00:00",
                    "model_resolutions": {
                        "technical_analysis": {
                            "model_version": "ghost-v9",
                            "status": "unregistered",
                            "override": True,
                            "override_reason": "incident-42",
                        }
                    },
                })
        self.assertEqual(AUDIT_SCHEMA_VERSION, "audit-event-v3")
        resolution = event["model_resolutions"]["technical_analysis"]
        self.assertTrue(resolution["override"])
        self.assertEqual(resolution["override_reason"], "incident-42")

    def test_orchestrator_resolutions_reach_the_event(self) -> None:
        """A real run stamps approved statuses, with no override."""
        from core.orchestrator import orchestrate_score

        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "audit.jsonl"
            with unittest.mock.patch("core.audit_store.AUDIT_LOG_PATH", log):
                orchestrate_score("MSFT", "2024-01-02")
                events = [
                    json.loads(line)
                    for line in log.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        resolutions = events[-1]["model_resolutions"]
        self.assertEqual(set(resolutions), {
            "market_data", "technical_analysis", "fundamental_analysis"
        })
        for agent, record in resolutions.items():
            with self.subTest(agent=agent):
                self.assertEqual(record["status"], MODEL_STATUS_APPROVED)
                self.assertFalse(record["override"])
