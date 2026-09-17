"""Champion / challenger tests (Sprint M7).

The M7 rule, pinned: "One Champion, multiple Challengers (A/B/C). New models
always begin in `SHADOW`."

Before M7 the registry had no role at all — `incumbent()` returned whichever
approved model sorted last by version string, so the most important question
in the registry ("which model is actually serving?") was an accident of
naming. These tests pin the role model that replaced it:

- every model is born shadow, and no path registers one into a serving role;
- shadow -> challenger requires accumulated out-of-sample evidence;
- challenger -> champion requires governance approval AND demotes the
  incumbent in the same act, so "exactly one champion" cannot be violated
  halfway through a swap;
- a champion serves, a challenger is measured, a shadow model is not even
  consulted.
"""

from __future__ import annotations

import unittest

from core.config import (
    MODEL_ROLE_CHALLENGER,
    MODEL_ROLE_CHAMPION,
    MODEL_ROLE_SHADOW,
    MODEL_STATUS_APPROVED,
    SHADOW_MIN_OBSERVATIONS,
)
from core.model_registry import (
    ModelEntry,
    ModelNotApprovedError,
    ModelRegistryError,
    build_default_model_registry,
    entry_problems,
    require_live_model,
)

_AT = "2026-09-17T00:00:00+00:00"
_CONTRACT = ("expected_return", "20d", "u1")


class ChampionChallengerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_model_registry()

    def _model(self, version: str, universe: str = "u1", horizon: str = "20d") -> ModelEntry:
        return ModelEntry(
            model_version=version,
            family="boosting",
            feature_set_version="fs-1",
            dataset_hash="d" * 64,
            code_commit="abc1234",
            hyperparameters={"n_estimators": 100},
            seed=42,
            artifact_hash="a" * 64,
            horizon=horizon,
            universe=universe,
        )

    def _comparison(self, candidate: str, incumbent: str | None = None) -> dict:
        return {
            "primary_metric": "oos_sharpe",
            "candidate_value": 1.4,
            "incumbent_value": 1.1,
            "sample": 500,
            "candidate_version": candidate,
            "incumbent_version": incumbent,
        }

    def _challenger(self, version: str, **kwargs) -> ModelEntry:
        entry = self.registry.register(self._model(version, **kwargs))
        self.registry.record_shadow_observations(version, SHADOW_MIN_OBSERVATIONS)
        self.registry.promote_to_challenger(version, _AT)
        return entry

    def _approved(self, version: str, incumbent: str | None = None, **kwargs) -> ModelEntry:
        entry = self._challenger(version, **kwargs)
        self.registry.promote(version, "oren", _AT, self._comparison(version, incumbent))
        return entry

    def _crowned(self, version: str, incumbent: str | None = None, **kwargs) -> ModelEntry:
        entry = self._approved(version, incumbent, **kwargs)
        self.registry.crown_champion(version, _AT)
        return entry


class TestEveryModelBeginsInShadow(ChampionChallengerTestCase):
    def test_new_model_is_born_shadow(self) -> None:
        self.assertEqual(self.registry.register(self._model("gb-v1")).role, MODEL_ROLE_SHADOW)

    def test_a_shadow_model_does_not_serve(self) -> None:
        self.assertFalse(self.registry.register(self._model("gb-v1")).is_live_eligible())

    def test_the_live_gate_refuses_a_shadow_model(self) -> None:
        self.registry.register(self._model("gb-v1"))
        with self.assertRaises(ModelNotApprovedError):
            require_live_model("gb-v1", self.registry)

    def test_governance_approval_alone_does_not_make_a_model_serve(self) -> None:
        """Approved is a lifecycle status; champion is a serving role."""
        entry = self._approved("gb-v1")
        self.assertEqual(entry.status, MODEL_STATUS_APPROVED)
        self.assertEqual(entry.role, MODEL_ROLE_CHALLENGER)
        self.assertFalse(entry.is_live_eligible())


class TestPromotionToChallenger(ChampionChallengerTestCase):
    def test_unmeasured_shadow_cannot_become_challenger(self) -> None:
        self.registry.register(self._model("gb-v1"))
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.promote_to_challenger("gb-v1", _AT)
        self.assertIn("earned nothing", str(ctx.exception))

    def test_shadow_observations_accrue(self) -> None:
        self.registry.register(self._model("gb-v1"))
        self.registry.record_shadow_observations("gb-v1", 60)
        self.registry.record_shadow_observations("gb-v1", 40)
        self.assertEqual(self.registry.get("gb-v1").shadow_observations, 100)

    def test_observations_only_accrue_while_in_shadow(self) -> None:
        self._challenger("gb-v1")
        with self.assertRaises(ModelRegistryError):
            self.registry.record_shadow_observations("gb-v1", 10)

    def test_negative_observations_are_refused(self) -> None:
        self.registry.register(self._model("gb-v1"))
        with self.assertRaises(ModelRegistryError):
            self.registry.record_shadow_observations("gb-v1", -5)

    def test_measured_shadow_becomes_a_challenger(self) -> None:
        entry = self._challenger("gb-v1")
        self.assertEqual(entry.role, MODEL_ROLE_CHALLENGER)
        self.assertEqual(entry.role_changed_at, _AT)

    def test_a_challenger_is_measured_not_trusted(self) -> None:
        self.assertFalse(self._approved("gb-v1").is_live_eligible())

    def test_a_challenger_cannot_be_re_promoted(self) -> None:
        self._challenger("gb-v1")
        with self.assertRaises(ModelRegistryError):
            self.registry.promote_to_challenger("gb-v1", _AT)

    def test_transition_must_be_timestamped(self) -> None:
        self.registry.register(self._model("gb-v1"))
        self.registry.record_shadow_observations("gb-v1", SHADOW_MIN_OBSERVATIONS)
        with self.assertRaises(ModelRegistryError):
            self.registry.promote_to_challenger("gb-v1", "")


class TestCrowning(ChampionChallengerTestCase):
    def test_shadow_cannot_be_crowned_directly(self) -> None:
        self.registry.register(self._model("gb-v1"))
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.crown_champion("gb-v1", _AT)
        self.assertIn("only a challenger is crowned", str(ctx.exception))

    def test_unapproved_challenger_cannot_be_crowned(self) -> None:
        """Crowning is a serving decision; it cannot replace the M2 gate."""
        self._challenger("gb-v1")
        with self.assertRaises(ModelRegistryError) as ctx:
            self.registry.crown_champion("gb-v1", _AT)
        self.assertIn("promotion gate", str(ctx.exception))

    def test_approved_challenger_is_crowned_and_serves(self) -> None:
        entry = self._crowned("gb-v1")
        self.assertEqual(entry.role, MODEL_ROLE_CHAMPION)
        self.assertTrue(entry.is_live_eligible())

    def test_crowning_requires_a_timestamp(self) -> None:
        self._approved("gb-v1")
        with self.assertRaises(ModelRegistryError):
            self.registry.crown_champion("gb-v1", "")

    def test_unregistered_model_cannot_be_crowned(self) -> None:
        with self.assertRaises(ModelRegistryError):
            self.registry.crown_champion("ghost-v9", _AT)


class TestExactlyOneChampion(ChampionChallengerTestCase):
    def test_crowning_demotes_the_incumbent_in_one_act(self) -> None:
        """The swap is atomic, so the invariant cannot break halfway."""
        self._crowned("gb-v1")
        self._approved("gb-v2", incumbent="gb-v1")

        result = self.registry.crown_champion("gb-v2", _AT, "better OOS")
        self.assertEqual(result["champion"], "gb-v2")
        self.assertEqual(result["demoted"], "gb-v1")
        self.assertEqual(self.registry.get("gb-v1").role, MODEL_ROLE_CHALLENGER)
        self.assertEqual(self.registry.role_problems(), [])

    def test_a_demoted_champion_keeps_its_entry_and_reason(self) -> None:
        self._crowned("gb-v1")
        self._approved("gb-v2", incumbent="gb-v1")
        self.registry.crown_champion("gb-v2", _AT)
        demoted = self.registry.get("gb-v1")
        self.assertIsNotNone(demoted, "the demoted champion was deleted")
        self.assertIn("demoted", demoted.role_reason)
        self.assertEqual(demoted.artifact_hash, "a" * 64)

    def test_champion_lookup_returns_the_single_server(self) -> None:
        self._crowned("gb-v1")
        self._approved("gb-v2", incumbent="gb-v1")
        self.registry.crown_champion("gb-v2", _AT)
        self.assertEqual(self.registry.champion(_CONTRACT).model_version, "gb-v2")

    def test_one_champion_many_challengers(self) -> None:
        """The M7 shape: one Champion, Challengers A/B/C."""
        self._crowned("gb-a")
        for version in ("gb-b", "gb-c", "gb-d"):
            self._approved(version, incumbent="gb-a")
        self.assertEqual(self.registry.champion(_CONTRACT).model_version, "gb-a")
        self.assertEqual(
            [e.model_version for e in self.registry.challengers(_CONTRACT)],
            ["gb-b", "gb-c", "gb-d"],
        )
        self.assertEqual(self.registry.role_problems(), [])

    def test_different_contracts_have_independent_champions(self) -> None:
        """A 20d champion and a 5d champion are not rivals."""
        self._crowned("gb-20d", horizon="20d")
        self._crowned("gb-5d", horizon="5d")
        self.assertEqual(self.registry.role_problems(), [])
        self.assertEqual(
            self.registry.champion(("expected_return", "20d", "u1")).model_version, "gb-20d"
        )
        self.assertEqual(
            self.registry.champion(("expected_return", "5d", "u1")).model_version, "gb-5d"
        )

    def test_role_problems_detects_a_double_champion(self) -> None:
        """Hand-forced corruption must be reported, not tolerated."""
        self._crowned("gb-v1")
        second = self._approved("gb-v2", incumbent="gb-v1")
        second.role = MODEL_ROLE_CHAMPION  # bypasses crown_champion deliberately
        problems = self.registry.role_problems()
        self.assertTrue(any("2 champions" in p for p in problems), problems)

    def test_champion_lookup_raises_on_a_corrupted_registry(self) -> None:
        self._crowned("gb-v1")
        second = self._approved("gb-v2", incumbent="gb-v1")
        second.role = MODEL_ROLE_CHAMPION
        with self.assertRaises(ModelRegistryError):
            self.registry.champion(_CONTRACT)

    def test_no_champion_for_an_unserved_contract(self) -> None:
        self.assertIsNone(self.registry.champion(("expected_return", "60d", "nobody")))


class TestShadowRoster(ChampionChallengerTestCase):
    def test_shadows_are_listed(self) -> None:
        self.registry.register(self._model("gb-v1"))
        self.registry.register(self._model("gb-v2"))
        self.assertEqual(
            [e.model_version for e in self.registry.shadows(_CONTRACT)], ["gb-v1", "gb-v2"]
        )

    def test_promotion_removes_a_model_from_the_shadow_roster(self) -> None:
        self._challenger("gb-v1")
        self.assertEqual(self.registry.shadows(_CONTRACT), [])


class TestRoleValidation(ChampionChallengerTestCase):
    def test_champion_role_cannot_outrank_governance(self) -> None:
        entry = self._model("gb-v1")
        entry.role = MODEL_ROLE_CHAMPION
        entry.role_changed_at = _AT
        self.assertTrue(any("cannot outrank governance" in p for p in entry_problems(entry)))

    def test_leaving_shadow_must_be_timestamped(self) -> None:
        entry = self._model("gb-v1")
        entry.role = MODEL_ROLE_CHALLENGER
        self.assertTrue(any("role_changed_at" in p for p in entry_problems(entry)))

    def test_unknown_role_is_refused(self) -> None:
        entry = self._model("gb-v1")
        entry.role = "emperor"
        self.assertTrue(any("not a known role" in p for p in entry_problems(entry)))

    def test_role_is_part_of_artifact_identity(self) -> None:
        base = self._model("gb-v1")
        other = self._model("gb-v1")
        other.role = MODEL_ROLE_CHALLENGER
        self.assertNotEqual(base.canonical_hash(), other.canonical_hash())


class TestSeededScorers(ChampionChallengerTestCase):
    def test_seeded_scorers_serve_their_own_contracts(self) -> None:
        """They are complementary ensemble components, not rivals."""
        for entry in self.registry.all_models().values():
            with self.subTest(model=entry.model_version):
                self.assertEqual(entry.role, MODEL_ROLE_CHAMPION)
                self.assertTrue(entry.is_live_eligible())
        self.assertEqual(self.registry.role_problems(), [])

    def test_each_seeded_scorer_has_a_distinct_contract(self) -> None:
        contracts = [e.forecast_contract() for e in self.registry.all_models().values()]
        self.assertEqual(len(contracts), len(set(contracts)))

    def test_the_orchestrator_can_still_resolve_them(self) -> None:
        for version in ("market-data-v1", "technical-v1", "fundamental-v1"):
            with self.subTest(model=version):
                self.assertEqual(
                    require_live_model(version, self.registry)["status"],
                    MODEL_STATUS_APPROVED,
                )


if __name__ == "__main__":
    unittest.main()
