"""X9 release-snapshot tests.

The behaviour under test is that a freeze must NOTICE A THAW. MEASURED, the
commit hash reports the same value before and after an edit, so a snapshot
carrying only a commit cannot tell a frozen tree from a modified one.

These tests run against a TEMPORARY GIT REPOSITORY, not the working directory:
the repo this suite runs in is dirty whenever someone is working in it, so
asserting anything about its cleanliness would test the checkout rather than the
code — the mistake recorded as open item 6.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.config import (
    RELEASE_BLOCKS_TRADES,
    RELEASE_COMPONENTS,
    RELEASE_DIRTY,
    RELEASE_DIRTY_IS_NOT_RELEASABLE,
    RELEASE_FROZEN,
    RELEASE_INCOMPLETE,
    RELEASE_NOT_EVALUATED,
    RELEASE_RECORDS_COMMIT,
    RELEASE_REQUIRES_EXPLICIT_ABSENCE,
    RELEASE_REQUIRES_WORKTREE_DIGEST,
    RELEASE_SNAPSHOT_VERDICTS,
)
from core.release_snapshot import (
    COMPONENT_ABSENT,
    build_release_snapshot,
    COMPONENT_PRESENT,
    RELEASE_REASON_DIRTY,
    RELEASE_REASON_MISSING,
    RELEASE_REASON_NO_GIT,
    ReleaseSnapshotError,
    _component,
    collect_components,
    commit_id,
    config_digest,
    is_dirty,
    render_release_snapshot,
    snapshot_digest,
    release_snapshot_problems,
    tracked_files,
    verify_snapshot,
    worktree_digest,
)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


class TemporaryRepository:
    """A real git repository with one tracked file, created per test.

    A real repo rather than a mock, because what is under test is precisely how
    git answers — and a mock would let me assert my own assumptions about
    `ls-files` and `hash-object` instead of their behaviour.
    """

    def __enter__(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name)
        git("init", "--quiet", cwd=self.path)
        git("config", "user.email", "test@example.com", cwd=self.path)
        git("config", "user.name", "Test", cwd=self.path)
        self.tracked = self.path / "tracked.txt"
        self.tracked.write_text("original\n", encoding="utf-8")
        git("add", "tracked.txt", cwd=self.path)
        git("commit", "--quiet", "-m", "initial", cwd=self.path)
        return self

    def __exit__(self, *exc):
        self._directory.cleanup()
        return False


class TheCommitCannotDetectAThawTests(unittest.TestCase):
    """The deciding measurement, reproduced on a repository of our own."""

    def test_the_commit_is_unchanged_by_an_edit(self):
        with TemporaryRepository() as repo:
            before = commit_id(repo.path)
            repo.tracked.write_text("modified\n", encoding="utf-8")
            after = commit_id(repo.path)
            self.assertEqual(before, after)
            # ...while the tree is demonstrably no longer that commit.
            self.assertTrue(is_dirty(repo.path))

    def test_the_worktree_digest_is_changed_by_an_edit(self):
        with TemporaryRepository() as repo:
            before = worktree_digest(repo.path)
            repo.tracked.write_text("modified\n", encoding="utf-8")
            after = worktree_digest(repo.path)
            self.assertNotEqual(before, after)

    def test_the_digest_returns_exactly_on_revert(self):
        with TemporaryRepository() as repo:
            before = worktree_digest(repo.path)
            repo.tracked.write_text("modified\n", encoding="utf-8")
            repo.tracked.write_text("original\n", encoding="utf-8")
            self.assertEqual(worktree_digest(repo.path), before)

    def test_the_digest_is_stable_across_calls(self):
        with TemporaryRepository() as repo:
            self.assertEqual(worktree_digest(repo.path), worktree_digest(repo.path))

    def test_a_rename_changes_the_digest(self):
        # The digest binds paths to contents, so moving a file without editing
        # it is still a change to what the release ships.
        with TemporaryRepository() as repo:
            before = worktree_digest(repo.path)
            git("mv", "tracked.txt", "renamed.txt", cwd=repo.path)
            self.assertNotEqual(worktree_digest(repo.path), before)

    def test_the_contract_requires_the_digest(self):
        self.assertTrue(RELEASE_REQUIRES_WORKTREE_DIGEST)
        self.assertTrue(RELEASE_RECORDS_COMMIT)


class UntrackedFilesAreNotDirtTests(unittest.TestCase):
    """A release freezes what it SHIPS."""

    def test_an_untracked_file_does_not_make_the_tree_dirty(self):
        # Otherwise every working directory is dirty forever, and the gitignored
        # data stores would make a release impossible on the machine that has
        # them.
        with TemporaryRepository() as repo:
            (repo.path / "scratch.tmp").write_text("junk\n", encoding="utf-8")
            self.assertFalse(is_dirty(repo.path))

    def test_an_untracked_file_does_not_change_the_digest(self):
        with TemporaryRepository() as repo:
            before = worktree_digest(repo.path)
            (repo.path / "scratch.tmp").write_text("junk\n", encoding="utf-8")
            self.assertEqual(worktree_digest(repo.path), before)

    def test_a_modified_tracked_file_does_make_it_dirty(self):
        with TemporaryRepository() as repo:
            repo.tracked.write_text("modified\n", encoding="utf-8")
            self.assertTrue(is_dirty(repo.path))


class GitUnavailableTests(unittest.TestCase):
    """None, never a sentinel string."""

    def test_a_non_repository_yields_none(self):
        with tempfile.TemporaryDirectory() as directory:
            # `_code_commit()` returns the literal "unknown" here, which is
            # indistinguishable from a commit named "unknown" once recorded.
            self.assertIsNone(commit_id(Path(directory)))
            self.assertIsNone(worktree_digest(Path(directory)))
            self.assertIsNone(is_dirty(Path(directory)))

    def test_a_non_repository_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            components = collect_components(
                runs=[{"artifact_hash": "a"}],
                validation={"X1": {"verdict": "OK"}},
                repo_root=Path(directory),
            )
            self.assertEqual(components["code"]["state"], COMPONENT_ABSENT)


class EveryComponentStatesItselfTests(unittest.TestCase):
    """An omitted component is not the same as an absent one."""

    def test_every_declared_component_gets_an_entry(self):
        components = collect_components()
        self.assertEqual(sorted(components), sorted(RELEASE_COMPONENTS))

    def test_every_entry_names_a_state(self):
        for name, record in collect_components().items():
            with self.subTest(component=name):
                self.assertIn(record["state"], (COMPONENT_PRESENT, COMPONENT_ABSENT))
                self.assertTrue(record["detail"].strip())

    def test_absent_validation_is_explicit(self):
        components = collect_components(validation=None)
        self.assertEqual(components["validation"]["state"], COMPONENT_ABSENT)

    def test_supplied_validation_is_present(self):
        components = collect_components(validation={"X8": {"verdict": "NOT_EVALUATED"}})
        self.assertEqual(components["validation"]["state"], COMPONENT_PRESENT)

    def test_a_not_evaluated_verdict_still_freezes(self):
        # "Gate not met" is a legitimate result and must be freezable; what
        # cannot freeze is the absence of any verdict.
        components = collect_components(validation={"X1": {"verdict": "NOT_EVALUATED"}})
        self.assertEqual(components["validation"]["state"], COMPONENT_PRESENT)

    def test_a_verdictless_validation_result_is_absent(self):
        components = collect_components(validation={"X1": {}})
        self.assertEqual(components["validation"]["state"], COMPONENT_ABSENT)

    def test_no_runs_makes_models_absent(self):
        components = collect_components(runs=[])
        self.assertEqual(components["models"]["state"], COMPONENT_ABSENT)

    def test_a_run_without_an_artifact_hash_makes_models_absent(self):
        components = collect_components(runs=[{"estimator": "ridge"}])
        self.assertEqual(components["models"]["state"], COMPONENT_ABSENT)

    def test_the_contract_requires_explicit_absence(self):
        self.assertTrue(RELEASE_REQUIRES_EXPLICIT_ABSENCE)

    def test_a_raising_probe_records_absent_rather_than_crashing(self):
        # The point of X9 is to say which components are missing; a traceback
        # says nothing.
        with tempfile.TemporaryDirectory() as directory:
            components = collect_components(repo_root=Path(directory))
            self.assertEqual(components["code"]["state"], COMPONENT_ABSENT)

    def test_an_invalid_component_state_is_refused(self):
        with self.assertRaises(ReleaseSnapshotError):
            _component("MAYBE", "no")


class VerdictTests(unittest.TestCase):
    """All four verdicts must be reachable."""

    def frozen_components(self, dirty=False):
        return {
            name: _component(COMPONENT_PRESENT, f"{name} ok")
            for name in RELEASE_COMPONENTS
        } | {
            "code": _component(
                COMPONENT_PRESENT,
                "commit abc1234",
                commit="abc1234",
                worktree_digest="d" * 64,
                dirty=dirty,
            )
        }

    def build(self, components):
        # Exercise the verdict logic directly, so the test does not depend on
        # the state of the repository it runs in.
        from core.release_snapshot import _result

        absent = sorted(
            name
            for name, record in components.items()
            if record.get("state") != COMPONENT_PRESENT
        )
        code = components["code"]
        if code.get("state") != COMPONENT_PRESENT:
            return _result(
                RELEASE_NOT_EVALUATED,
                "no git",
                reason_code=RELEASE_REASON_NO_GIT,
                components=components,
                absent=absent,
            )
        if absent:
            return _result(
                RELEASE_INCOMPLETE,
                f"absent: {absent}",
                reason_code=RELEASE_REASON_MISSING,
                components=components,
                absent=absent,
            )
        if code.get("dirty"):
            return _result(
                RELEASE_DIRTY,
                "tracked files differ from HEAD",
                reason_code=RELEASE_REASON_DIRTY,
                components=components,
                absent=absent,
                commit=code["commit"],
                worktree_digest=code["worktree_digest"],
            )
        return _result(
            RELEASE_FROZEN,
            "frozen",
            reason_code="",
            components=components,
            absent=absent,
            commit=code["commit"],
            worktree_digest=code["worktree_digest"],
        )

    def test_a_clean_complete_snapshot_is_frozen(self):
        report = self.build(self.frozen_components())
        self.assertEqual(report["verdict"], RELEASE_FROZEN)
        self.assertEqual(release_snapshot_problems(report), [])

    def test_a_dirty_tree_is_not_frozen(self):
        self.assertTrue(RELEASE_DIRTY_IS_NOT_RELEASABLE)
        report = self.build(self.frozen_components(dirty=True))
        self.assertEqual(report["verdict"], RELEASE_DIRTY)
        self.assertEqual(release_snapshot_problems(report), [])

    def test_an_absent_component_is_incomplete(self):
        components = self.frozen_components()
        components["validation"] = _component(COMPONENT_ABSENT, "no results")
        report = self.build(components)
        self.assertEqual(report["verdict"], RELEASE_INCOMPLETE)
        self.assertEqual(report["absent"], ["validation"])
        self.assertEqual(release_snapshot_problems(report), [])

    def test_absent_code_is_not_evaluated(self):
        components = self.frozen_components()
        components["code"] = _component(COMPONENT_ABSENT, "no git")
        report = self.build(components)
        self.assertEqual(report["verdict"], RELEASE_NOT_EVALUATED)
        self.assertEqual(release_snapshot_problems(report), [])


class TheShippedBuilderIsExercisedTests(unittest.TestCase):
    """`build_release_snapshot` itself, not a reimplementation of its logic.

    A SABOTAGE TEST CAUGHT THIS GAP: disabling the `if absent:` branch in
    `build_release_snapshot` left every test and the gate green, because both
    drove the verdict through a local helper that duplicated the rules. A copy
    of the logic proves nothing about the shipped path.
    """

    def test_the_builder_reports_incomplete_when_validation_is_absent(self):
        report = build_release_snapshot(validation=None)
        self.assertEqual(report["verdict"], RELEASE_INCOMPLETE)
        self.assertIn("validation", report["absent"])
        self.assertEqual(release_snapshot_problems(report), [])

    def test_the_builder_reports_incomplete_when_models_are_absent(self):
        report = build_release_snapshot(
            runs=[], validation={"X1": {"verdict": "OK"}}
        )
        self.assertEqual(report["verdict"], RELEASE_INCOMPLETE)
        self.assertIn("models", report["absent"])

    def test_the_builder_reports_not_evaluated_outside_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_release_snapshot(
                runs=[{"artifact_hash": "a"}],
                validation={"X1": {"verdict": "OK"}},
                repo_root=Path(directory),
            )
            self.assertEqual(report["verdict"], RELEASE_NOT_EVALUATED)
            self.assertEqual(report["reason_code"], RELEASE_REASON_NO_GIT)
            self.assertEqual(release_snapshot_problems(report), [])

    def test_the_builder_always_records_every_component(self):
        report = build_release_snapshot(validation=None)
        self.assertEqual(sorted(report["components"]), sorted(RELEASE_COMPONENTS))

    def test_the_builder_produces_a_snapshot_digest(self):
        report = build_release_snapshot(validation={"X1": {"verdict": "OK"}})
        self.assertEqual(len(report["snapshot_digest"]), 64)

    def test_a_snapshot_the_builder_took_verifies_against_its_own_tree(self):
        report = build_release_snapshot(validation={"X1": {"verdict": "OK"}})
        self.assertIs(verify_snapshot(report)["matches"], True)

    def test_the_builder_refuses_to_freeze_a_dirty_tree(self):
        """The dirty branch of the SHIPPED builder, on a repo of our own.

        ALSO FOUND BY SABOTAGE: disabling the dirty check was caught by the gate
        but not here, because the other tests deliberately avoid asserting
        anything about the cleanliness of the tree the suite runs in. A temporary
        repository lets the shipped branch be driven both ways.
        """
        with TemporaryRepository() as repo:
            # Everything but `code` is probed from this process, so a complete
            # snapshot of the temporary repo is not achievable; drive the
            # builder's dirty branch through the code component directly.
            components = collect_components(
                runs=[{"artifact_hash": "a"}],
                validation={"X1": {"verdict": "OK"}},
                repo_root=repo.path,
            )
            self.assertFalse(components["code"]["dirty"])

            repo.tracked.write_text("modified\n", encoding="utf-8")
            dirtied = collect_components(
                runs=[{"artifact_hash": "a"}],
                validation={"X1": {"verdict": "OK"}},
                repo_root=repo.path,
            )
            self.assertTrue(dirtied["code"]["dirty"])

    def test_a_complete_but_dirty_snapshot_is_dirty_not_frozen(self):
        # The shipped `build_release_snapshot`, driven through its dirty branch
        # by making every component present and the code component dirty.
        import core.release_snapshot as module

        real = module.collect_components

        def dirty_components(**kwargs):
            built = {
                name: _component(COMPONENT_PRESENT, f"{name} ok")
                for name in RELEASE_COMPONENTS
            }
            built["code"] = _component(
                COMPONENT_PRESENT,
                "commit abc1234",
                commit="abc1234",
                worktree_digest="d" * 64,
                dirty=True,
            )
            return built

        try:
            module.collect_components = dirty_components
            report = module.build_release_snapshot(
                validation={"X1": {"verdict": "OK"}}
            )
            self.assertEqual(report["verdict"], RELEASE_DIRTY)
            self.assertEqual(report["reason_code"], RELEASE_REASON_DIRTY)
            self.assertEqual(release_snapshot_problems(report), [])
        finally:
            module.collect_components = real

    def test_a_complete_clean_snapshot_is_frozen(self):
        # The same path with `dirty=False`, so the branch is proved both ways
        # and FROZEN is reachable from the shipped builder.
        import core.release_snapshot as module

        real = module.collect_components

        def clean_components(**kwargs):
            built = {
                name: _component(COMPONENT_PRESENT, f"{name} ok")
                for name in RELEASE_COMPONENTS
            }
            built["code"] = _component(
                COMPONENT_PRESENT,
                "commit abc1234",
                commit="abc1234",
                worktree_digest="d" * 64,
                dirty=False,
            )
            return built

        try:
            module.collect_components = clean_components
            report = module.build_release_snapshot(
                validation={"X1": {"verdict": "OK"}}
            )
            self.assertEqual(report["verdict"], RELEASE_FROZEN)
            self.assertEqual(release_snapshot_problems(report), [])
        finally:
            module.collect_components = real


class AComponentWithoutAProbeIsAbsentTests(unittest.TestCase):
    """The fallback for a component declared but never probed.

    ALSO FOUND BY SABOTAGE: disabling this loop was invisible, because every
    name in RELEASE_COMPONENTS currently has a probe. It is the guarantee that
    the NEXT component added cannot be silently omitted, so it needs a test that
    actually reaches it.
    """

    def test_a_declared_component_with_no_probe_records_absent(self):
        import core.config as config_module
        import core.release_snapshot as module

        original = config_module.RELEASE_COMPONENTS
        try:
            extended = original + ("unprobed_component",)
            config_module.RELEASE_COMPONENTS = extended
            module.RELEASE_COMPONENTS = extended
            components = collect_components()
            self.assertIn("unprobed_component", components)
            self.assertEqual(
                components["unprobed_component"]["state"], COMPONENT_ABSENT
            )
            self.assertIn("no probe", components["unprobed_component"]["detail"])
        finally:
            config_module.RELEASE_COMPONENTS = original
            module.RELEASE_COMPONENTS = original

    def test_an_unprobed_component_makes_the_release_incomplete(self):
        import core.config as config_module
        import core.release_snapshot as module

        original = config_module.RELEASE_COMPONENTS
        try:
            extended = original + ("unprobed_component",)
            config_module.RELEASE_COMPONENTS = extended
            module.RELEASE_COMPONENTS = extended
            report = build_release_snapshot(validation={"X1": {"verdict": "OK"}})
            self.assertEqual(report["verdict"], RELEASE_INCOMPLETE)
            self.assertIn("unprobed_component", report["absent"])
        finally:
            config_module.RELEASE_COMPONENTS = original
            module.RELEASE_COMPONENTS = original


class VerifyTests(unittest.TestCase):
    """The question a freeze exists to answer."""

    def test_an_unchanged_tree_matches(self):
        with TemporaryRepository() as repo:
            snapshot = {"worktree_digest": worktree_digest(repo.path)}
            self.assertTrue(verify_snapshot(snapshot, repo_root=repo.path)["matches"])

    def test_a_changed_tree_does_not_match(self):
        with TemporaryRepository() as repo:
            snapshot = {"worktree_digest": worktree_digest(repo.path)}
            repo.tracked.write_text("modified\n", encoding="utf-8")
            result = verify_snapshot(snapshot, repo_root=repo.path)
            self.assertFalse(result["matches"])
            self.assertIn("changed", result["reason"])

    def test_a_reverted_tree_matches_again(self):
        with TemporaryRepository() as repo:
            snapshot = {"worktree_digest": worktree_digest(repo.path)}
            repo.tracked.write_text("modified\n", encoding="utf-8")
            repo.tracked.write_text("original\n", encoding="utf-8")
            self.assertTrue(verify_snapshot(snapshot, repo_root=repo.path)["matches"])

    def test_a_snapshot_without_a_digest_cannot_be_verified(self):
        # None, not False: "cannot tell" is not "does not match", and this is
        # exactly the state a commit-only snapshot leaves you in.
        result = verify_snapshot({"commit": "abc1234"})
        self.assertIsNone(result["matches"])

    def test_a_non_mapping_snapshot_is_refused(self):
        with self.assertRaises(ReleaseSnapshotError):
            verify_snapshot("not a mapping")


class DigestTests(unittest.TestCase):
    def test_the_config_digest_is_deterministic(self):
        self.assertEqual(config_digest(), config_digest())

    def test_the_config_digest_is_a_sha256(self):
        self.assertEqual(len(config_digest()), 64)

    def test_the_snapshot_digest_ignores_prose_and_dirt(self):
        # `detail` is prose and `dirty` is a property of the moment; including
        # either would make the digest differ across reruns that froze identical
        # material.
        first = {"code": _component(COMPONENT_PRESENT, "one", commit="a", dirty=False)}
        second = {"code": _component(COMPONENT_PRESENT, "two", commit="a", dirty=True)}
        self.assertEqual(snapshot_digest(first), snapshot_digest(second))

    def test_the_snapshot_digest_notices_a_changed_component(self):
        first = {"code": _component(COMPONENT_PRESENT, "d", commit="a")}
        second = {"code": _component(COMPONENT_PRESENT, "d", commit="b")}
        self.assertNotEqual(snapshot_digest(first), snapshot_digest(second))

    def test_the_snapshot_digest_is_order_independent(self):
        one = {
            "code": _component(COMPONENT_PRESENT, "d", commit="a"),
            "data": _component(COMPONENT_PRESENT, "d", digest="x"),
        }
        two = dict(reversed(list(one.items())))
        self.assertEqual(snapshot_digest(one), snapshot_digest(two))

    def test_tracked_files_are_sorted(self):
        with TemporaryRepository() as repo:
            files = tracked_files(repo.path)
            self.assertEqual(files, sorted(files))


class ContractTests(unittest.TestCase):
    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(RELEASE_SNAPSHOT_VERDICTS[0], RELEASE_NOT_EVALUATED)
        self.assertEqual(RELEASE_SNAPSHOT_VERDICTS[-1], RELEASE_FROZEN)

    def test_dirty_is_not_frozen(self):
        self.assertNotEqual(RELEASE_DIRTY, RELEASE_FROZEN)

    def test_incomplete_is_not_not_evaluated(self):
        self.assertNotEqual(RELEASE_INCOMPLETE, RELEASE_NOT_EVALUATED)

    def test_x9_reports_and_does_not_block(self):
        self.assertFalse(RELEASE_BLOCKS_TRADES)

    def test_every_roadmap_component_is_declared(self):
        for required in (
            "code",
            "data",
            "features",
            "models",
            "weights",
            "calibration",
            "configuration",
            "validation",
        ):
            self.assertIn(required, RELEASE_COMPONENTS)

    def test_a_forged_frozen_report_without_a_digest_is_caught(self):
        forged = {
            "verdict": RELEASE_FROZEN,
            "reason": "trust me",
            "commit": "abc1234",
            "components": {
                name: _component(COMPONENT_PRESENT, "d") for name in RELEASE_COMPONENTS
            },
        }
        self.assertTrue(release_snapshot_problems(forged))

    def test_a_forged_frozen_report_with_absences_is_caught(self):
        forged = {
            "verdict": RELEASE_FROZEN,
            "reason": "r",
            "commit": "abc1234",
            "worktree_digest": "d" * 64,
            "absent": ["validation"],
            "components": {
                name: _component(COMPONENT_PRESENT, "d") for name in RELEASE_COMPONENTS
            },
        }
        self.assertTrue(release_snapshot_problems(forged))

    def test_a_frozen_report_from_a_dirty_tree_is_caught(self):
        components = {
            name: _component(COMPONENT_PRESENT, "d") for name in RELEASE_COMPONENTS
        }
        components["code"] = _component(
            COMPONENT_PRESENT, "d", commit="a", worktree_digest="d" * 64, dirty=True
        )
        forged = {
            "verdict": RELEASE_FROZEN,
            "reason": "r",
            "commit": "a",
            "worktree_digest": "d" * 64,
            "components": components,
        }
        self.assertTrue(release_snapshot_problems(forged))

    def test_a_report_missing_a_component_state_is_caught(self):
        forged = {
            "verdict": RELEASE_FROZEN,
            "reason": "r",
            "commit": "a",
            "worktree_digest": "d" * 64,
            "components": {"code": _component(COMPONENT_PRESENT, "d")},
        }
        self.assertTrue(release_snapshot_problems(forged))

    def test_an_incomplete_report_with_no_absences_is_caught(self):
        forged = {
            "verdict": RELEASE_INCOMPLETE,
            "reason": "r",
            "reason_code": RELEASE_REASON_MISSING,
            "absent": [],
            "components": {
                name: _component(COMPONENT_PRESENT, "d") for name in RELEASE_COMPONENTS
            },
        }
        self.assertTrue(release_snapshot_problems(forged))

    def test_a_blocking_report_is_caught(self):
        self.assertTrue(
            release_snapshot_problems(
                {"verdict": RELEASE_FROZEN, "reason": "r", "blocks_trades": True}
            )
        )

    def test_a_non_mapping_report_is_caught(self):
        self.assertTrue(release_snapshot_problems("not a mapping"))

    def test_an_unknown_verdict_is_refused(self):
        from core.release_snapshot import _result

        with self.assertRaises(ReleaseSnapshotError):
            _result("MAYBE", "no")


class RenderTests(unittest.TestCase):
    def test_absent_facts_render_as_absent(self):
        text = "\n".join(
            render_release_snapshot(
                {"verdict": RELEASE_NOT_EVALUATED, "reason": "r", "components": {}}
            )
        )
        self.assertIn("ABSENT", text)

    def test_every_component_gets_a_line(self):
        components = {
            name: _component(COMPONENT_PRESENT, "d") for name in RELEASE_COMPONENTS
        }
        lines = render_release_snapshot(
            {"verdict": RELEASE_FROZEN, "reason": "r", "components": components}
        )
        for name in RELEASE_COMPONENTS:
            self.assertTrue(any(name in line for line in lines))

    def test_render_returns_lines_not_a_blob(self):
        lines = render_release_snapshot({"verdict": RELEASE_FROZEN, "reason": "r"})
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


if __name__ == "__main__":
    unittest.main()
