"""X9 governance gate — a freeze must notice a thaw.

Exits 1 when any of these fails.

THE DECIDING MEASUREMENT, RECOMPUTED HERE on a TEMPORARY repository this script
creates: `git rev-parse HEAD` returns the SAME commit before and after an edit,
while the worktree digest moves and returns. A snapshot carrying only a commit
cannot tell a frozen tree from a modified one.

A temporary repository, not this one. The repo this gate runs in is dirty
whenever someone is working in it, so asserting anything about its cleanliness
would test the checkout rather than the code — open item 6 in
`docs/open-decisions.md` records that mistake twice over. The SHIPPED snapshot is
still built and contract-checked, but its verdict is allowed to be DIRTY.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
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
from core.release_snapshot import (  # noqa: E402
    COMPONENT_ABSENT,
    COMPONENT_PRESENT,
    _component,
    build_release_snapshot,
    collect_components,
    commit_id,
    config_digest,
    is_dirty,
    render_release_snapshot,
    snapshot_digest,
    snapshot_problems,
    verify_snapshot,
    worktree_digest,
)
from core.sealed_holdout import evaluate_sealed_holdout  # noqa: E402
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


# --- 1. THE DECIDING MEASUREMENT: the commit cannot detect an edit ---------------

with tempfile.TemporaryDirectory() as directory:
    repo = Path(directory)
    git("init", "--quiet", cwd=repo)
    git("config", "user.email", "gate@example.com", cwd=repo)
    git("config", "user.name", "Gate", cwd=repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    git("add", "tracked.txt", cwd=repo)
    git("commit", "--quiet", "-m", "initial", cwd=repo)

    commit_before = commit_id(repo)
    digest_before = worktree_digest(repo)
    check(
        commit_before is not None and digest_before is not None,
        "git could not identify a repository this gate just created",
    )

    tracked.write_text("modified\n", encoding="utf-8")
    commit_after = commit_id(repo)
    digest_after = worktree_digest(repo)

    check(
        commit_before == commit_after,
        f"the commit changed from {commit_before} to {commit_after} on an edit "
        f"that was never committed; X9's premise is that it does NOT, so the "
        f"reasoning behind the worktree digest must be revisited",
    )
    check(
        digest_before != digest_after,
        "the worktree digest did NOT change when a tracked file was edited, so "
        "it is as dirt-blind as the commit hash it replaced",
    )
    check(
        is_dirty(repo) is True,
        "a tree with a modified tracked file was not reported dirty",
    )

    tracked.write_text("original\n", encoding="utf-8")
    check(
        worktree_digest(repo) == digest_before,
        "the worktree digest did not return to its prior value after a revert, "
        "so it is not a stable identity for a frozen tree",
    )
    check(
        is_dirty(repo) is False,
        "a reverted tree was still reported dirty",
    )

    # THE OBVIOUS FIX THAT DOES NOT WORK, kept as a gate so it cannot be
    # reintroduced: `git ls-files -s` reports STAGED blob hashes.
    staged_before = git("ls-files", "-s", cwd=repo).stdout
    tracked.write_text("modified again\n", encoding="utf-8")
    staged_after = git("ls-files", "-s", cwd=repo).stdout
    check(
        staged_before == staged_after,
        "`git ls-files -s` moved on an unstaged edit; X9's note that it is "
        "dirt-blind would then be wrong and the cheaper digest could be used",
    )
    check(
        worktree_digest(repo) != digest_before,
        "the worktree digest failed to notice the second edit",
    )

    # UNTRACKED FILES ARE NOT DIRT: a release freezes what it ships.
    tracked.write_text("original\n", encoding="utf-8")
    (repo / "scratch.tmp").write_text("junk\n", encoding="utf-8")
    check(
        is_dirty(repo) is False,
        "an untracked file made the tree dirty; every working directory would "
        "then be unreleasable forever, and the gitignored data stores would "
        "block a release on the machine that has them",
    )
    check(
        worktree_digest(repo) == digest_before,
        "an untracked file changed the worktree digest",
    )

    # A RENAME IS A CHANGE even with identical contents.
    (repo / "scratch.tmp").unlink()
    git("mv", "tracked.txt", "renamed.txt", cwd=repo)
    check(
        worktree_digest(repo) != digest_before,
        "renaming a file without editing it left the digest unchanged; the "
        "digest must bind paths to contents",
    )


# --- 2. A NON-REPOSITORY YIELDS None, NEVER A SENTINEL ---------------------------

with tempfile.TemporaryDirectory() as directory:
    outside = Path(directory)
    check(
        commit_id(outside) is None and worktree_digest(outside) is None,
        "a directory that is not a git repository produced a commit or digest; "
        "`_code_commit()` returns the literal 'unknown' here, which is "
        "indistinguishable from a commit named 'unknown' once recorded",
    )
    check(
        collect_components(repo_root=outside)["code"]["state"] == COMPONENT_ABSENT,
        "code was reported PRESENT outside a git repository",
    )


# --- 3. EVERY COMPONENT STATES PRESENT OR ABSENT ---------------------------------

components = collect_components()
check(
    sorted(components) == sorted(RELEASE_COMPONENTS),
    f"the collected components {sorted(components)} do not match the declared "
    f"{sorted(RELEASE_COMPONENTS)}; a component with no entry is "
    f"indistinguishable from one found absent",
)
for name, record in components.items():
    check(
        record.get("state") in (COMPONENT_PRESENT, COMPONENT_ABSENT),
        f"{name}: state {record.get('state')!r} is neither PRESENT nor ABSENT",
    )
    check(
        bool(str(record.get("detail") or "").strip()),
        f"{name}: recorded no detail, so an absence cannot be acted on",
    )

# The seven components that do not depend on a caller must all be PRESENT: if
# one goes missing, X9's "everything but the code is fine" finding is stale.
for name in ("code", "configuration", "features", "models", "weights", "calibration", "data"):
    check(
        components[name]["state"] == COMPONENT_PRESENT,
        f"{name} is {components[name]['state']}: {components[name]['detail']}. "
        f"X9 measured all seven non-caller components PRESENT, so this finding "
        f"is now stale",
    )

check(
    collect_components(validation=None)["validation"]["state"] == COMPONENT_ABSENT,
    "validation was reported PRESENT with no results supplied",
)
check(
    collect_components(validation={"X1": {}})["validation"]["state"]
    == COMPONENT_ABSENT,
    "a validation result carrying no verdict was reported PRESENT",
)
check(
    collect_components(validation={"X1": {"verdict": "NOT_EVALUATED"}})["validation"][
        "state"
    ]
    == COMPONENT_PRESENT,
    "a NOT_EVALUATED verdict was refused; 'gate not met' is a legitimate result "
    "and must be freezable",
)
check(
    collect_components(runs=[])["models"]["state"] == COMPONENT_ABSENT,
    "models was reported PRESENT with no training runs",
)
check(
    collect_components(runs=[{"estimator": "ridge"}])["models"]["state"]
    == COMPONENT_ABSENT,
    "a run with no artifact hash was accepted as a freezable model",
)


# --- 4. THE SHIPPED SNAPSHOT ------------------------------------------------------
#
# Built and contract-checked, but its VERDICT is not asserted: this tree is dirty
# whenever someone is working in it, and FROZEN is proved on a clean clone below.

runs = load_training_runs()
dataset_rows = None
datasets_path = REPO_ROOT / "data" / "training_datasets.jsonl"
if datasets_path.exists():
    lines = [
        line
        for line in datasets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if lines:
        dataset_rows = json.loads(lines[0]).get("row_count")

shipped = build_release_snapshot(
    validation={
        "X8_sealed_holdout": evaluate_sealed_holdout(runs, dataset_rows=dataset_rows)
    }
)
check(
    shipped["verdict"] in RELEASE_SNAPSHOT_VERDICTS,
    f"the shipped snapshot reported {shipped['verdict']!r}",
)
check(
    snapshot_problems(shipped) == [],
    f"the shipped snapshot is not contract-clean: {snapshot_problems(shipped)}",
)
check(
    bool(shipped.get("worktree_digest")) and bool(shipped.get("commit")),
    "the shipped snapshot records no worktree digest or no commit",
)
check(
    verify_snapshot(shipped)["matches"] is True,
    "a snapshot just taken does not verify against the tree it was taken from",
)


# --- 4b. THE SHIPPED BUILDER, NOT A COPY OF ITS RULES ----------------------------
#
# A SABOTAGE TEST CAUGHT THIS GAP: disabling the `if absent:` branch inside
# `build_release_snapshot` left this gate green, because section 5 drives the
# verdicts through a LOCAL helper that duplicates the rules. Exercising a copy of
# the logic proves nothing about the shipped path, so the builder is now driven
# directly into each verdict it can reach from here.

builder_incomplete = build_release_snapshot(validation=None)
check(
    builder_incomplete["verdict"] == RELEASE_INCOMPLETE,
    f"build_release_snapshot reported {builder_incomplete['verdict']} with "
    f"validation absent, not INCOMPLETE; the absent-component branch is not "
    f"being reached",
)
check(
    "validation" in (builder_incomplete.get("absent") or []),
    "the builder did not name validation as absent",
)

builder_no_models = build_release_snapshot(
    runs=[], validation={"X1": {"verdict": "OK"}}
)
check(
    builder_no_models["verdict"] == RELEASE_INCOMPLETE
    and "models" in (builder_no_models.get("absent") or []),
    f"build_release_snapshot reported {builder_no_models['verdict']} with no "
    f"training runs, not INCOMPLETE",
)

with tempfile.TemporaryDirectory() as directory:
    builder_no_git = build_release_snapshot(
        runs=[{"artifact_hash": "a"}],
        validation={"X1": {"verdict": "OK"}},
        repo_root=Path(directory),
    )
    check(
        builder_no_git["verdict"] == RELEASE_NOT_EVALUATED,
        f"build_release_snapshot reported {builder_no_git['verdict']} outside a "
        f"git repository, not NOT_EVALUATED",
    )
    check(
        snapshot_problems(builder_no_git) == [],
        f"the no-git snapshot is not contract-clean: "
        f"{snapshot_problems(builder_no_git)}",
    )

# A COMPONENT DECLARED WITH NO PROBE must still record ABSENT. Also found by
# sabotage: disabling that loop was invisible here, because every current name
# has a probe. It is the guarantee that the NEXT component added cannot be
# silently omitted.
import core.config as _config_module  # noqa: E402
import core.release_snapshot as _snapshot_module  # noqa: E402

_original_components = _config_module.RELEASE_COMPONENTS
try:
    _extended = _original_components + ("unprobed_component",)
    _config_module.RELEASE_COMPONENTS = _extended
    _snapshot_module.RELEASE_COMPONENTS = _extended
    unprobed = collect_components()
    check(
        unprobed.get("unprobed_component", {}).get("state") == COMPONENT_ABSENT,
        "a component declared in RELEASE_COMPONENTS with no probe did not "
        "record ABSENT; the next component added could be silently omitted",
    )
    unprobed_report = build_release_snapshot(validation={"X1": {"verdict": "OK"}})
    check(
        unprobed_report["verdict"] == RELEASE_INCOMPLETE
        and "unprobed_component" in (unprobed_report.get("absent") or []),
        "an unprobed component did not make the release INCOMPLETE",
    )
finally:
    _config_module.RELEASE_COMPONENTS = _original_components
    _snapshot_module.RELEASE_COMPONENTS = _original_components


# --- 5. THE VERDICTS ARE ALL REACHABLE -------------------------------------------
#
# Driven through constructed component sets, so the gate does not depend on the
# state of the tree it runs in.


def complete(dirty=False):
    built = {
        name: _component(COMPONENT_PRESENT, f"{name} ok") for name in RELEASE_COMPONENTS
    }
    built["code"] = _component(
        COMPONENT_PRESENT,
        "commit abc1234",
        commit="abc1234",
        worktree_digest="d" * 64,
        dirty=dirty,
    )
    return built


def verdict_of(built):
    from core.release_snapshot import _result

    absent = sorted(
        name
        for name, record in built.items()
        if record.get("state") != COMPONENT_PRESENT
    )
    code = built["code"]
    if code.get("state") != COMPONENT_PRESENT:
        return _result(
            RELEASE_NOT_EVALUATED,
            "no git",
            reason_code="GIT_UNAVAILABLE",
            components=built,
            absent=absent,
        )
    if absent:
        return _result(
            RELEASE_INCOMPLETE,
            f"absent: {absent}",
            reason_code="COMPONENTS_ABSENT",
            components=built,
            absent=absent,
        )
    if RELEASE_DIRTY_IS_NOT_RELEASABLE and code.get("dirty"):
        return _result(
            RELEASE_DIRTY,
            "dirty",
            reason_code="WORKTREE_DIRTY",
            components=built,
            absent=absent,
            commit=code["commit"],
            worktree_digest=code["worktree_digest"],
        )
    return _result(
        RELEASE_FROZEN,
        "frozen",
        reason_code="",
        components=built,
        absent=absent,
        commit=code["commit"],
        worktree_digest=code["worktree_digest"],
    )


frozen = verdict_of(complete())
check(frozen["verdict"] == RELEASE_FROZEN, f"a clean complete set gave {frozen['verdict']}")
check(snapshot_problems(frozen) == [], f"frozen unclean: {snapshot_problems(frozen)}")

dirty_report = verdict_of(complete(dirty=True))
check(
    dirty_report["verdict"] == RELEASE_DIRTY,
    f"a dirty tree gave {dirty_report['verdict']}, not DIRTY; a snapshot whose "
    f"digest matches no checkout-able commit is not a release",
)

missing = complete()
missing["validation"] = _component(COMPONENT_ABSENT, "no results")
incomplete = verdict_of(missing)
check(
    incomplete["verdict"] == RELEASE_INCOMPLETE,
    f"an absent component gave {incomplete['verdict']}, not INCOMPLETE",
)

no_git = complete()
no_git["code"] = _component(COMPONENT_ABSENT, "no git")
unevaluated = verdict_of(no_git)
check(
    unevaluated["verdict"] == RELEASE_NOT_EVALUATED,
    f"absent code gave {unevaluated['verdict']}, not NOT_EVALUATED",
)


# --- 6. DIGESTS ------------------------------------------------------------------

check(config_digest() == config_digest(), "the config digest is not deterministic")
check(len(config_digest()) == 64, "the config digest is not a sha256")
check(
    snapshot_digest({"code": _component(COMPONENT_PRESENT, "one", commit="a", dirty=False)})
    == snapshot_digest({"code": _component(COMPONENT_PRESENT, "two", commit="a", dirty=True)}),
    "the snapshot digest changed on prose or dirt alone; it would then differ "
    "across reruns that froze identical material",
)
check(
    snapshot_digest({"code": _component(COMPONENT_PRESENT, "d", commit="a")})
    != snapshot_digest({"code": _component(COMPONENT_PRESENT, "d", commit="b")}),
    "the snapshot digest did not change when a component changed",
)


# --- 7. CONTRACT ------------------------------------------------------------------

check(RELEASE_REQUIRES_WORKTREE_DIGEST, "the worktree-digest requirement is off")
check(RELEASE_RECORDS_COMMIT, "the commit requirement is off")
check(RELEASE_REQUIRES_EXPLICIT_ABSENCE, "explicit absence is no longer required")
check(RELEASE_DIRTY_IS_NOT_RELEASABLE, "a dirty tree is now considered releasable")
check(not RELEASE_BLOCKS_TRADES, "X9 reports; the registry promotes")
check(
    RELEASE_SNAPSHOT_VERDICTS[0] == RELEASE_NOT_EVALUATED
    and RELEASE_SNAPSHOT_VERDICTS[-1] == RELEASE_FROZEN,
    "the release verdicts no longer run weakest to strongest",
)
check(RELEASE_DIRTY != RELEASE_FROZEN, "DIRTY and FROZEN collapsed into one state")
check(
    RELEASE_INCOMPLETE != RELEASE_NOT_EVALUATED,
    "'a component is missing' and 'no snapshot could be taken' collapsed",
)
check(
    bool(
        snapshot_problems(
            {
                "verdict": RELEASE_FROZEN,
                "reason": "trust me",
                "commit": "abc1234",
                "components": {
                    name: _component(COMPONENT_PRESENT, "d")
                    for name in RELEASE_COMPONENTS
                },
            }
        )
    ),
    "a FROZEN report with no worktree digest was accepted, which is the defect "
    "X9 exists to close",
)
check(
    verify_snapshot({"commit": "abc1234"})["matches"] is None,
    "a commit-only snapshot reported a definite match or mismatch; 'cannot "
    "tell' is the honest answer and the reason the digest is recorded",
)

for label, report in (
    ("shipped", shipped),
    ("frozen", frozen),
    ("dirty", dirty_report),
    ("incomplete", incomplete),
):
    lines = render_release_snapshot(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_release_snapshot returned no lines for {label}")
        break


if failures:
    print("X9 RELEASE SNAPSHOT GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X9 release snapshot gate: OK")
print(f"  commit across an edit             UNCHANGED  <- cannot detect a thaw")
print(f"  worktree digest across an edit    CHANGED, and returns on revert")
print(f"  git ls-files -s across an edit    UNCHANGED  <- dirt-blind too")
print(f"  untracked files                   not dirt (a release ships tracked files)")
print(f"  rename with identical contents    CHANGED")
print(f"  components stating themselves     {len(components)} of {len(RELEASE_COMPONENTS)}")
print(f"  shipped snapshot                  {shipped['verdict']}"
      f"  (digest {str(shipped.get('worktree_digest'))[:12]})")
print(f"  verdicts reachable                {frozen['verdict']} / {dirty_report['verdict']}"
      f" / {incomplete['verdict']} / {unevaluated['verdict']}")
sys.exit(0)
