"""X9 release snapshot — freeze the release, and notice when it thaws.

"Freeze code, data, features, models, weights, calibration, configuration,
validation results."

**Every component is present and deterministically hashable except the one the
whole snapshot rests on.** MEASURED, `_code_commit()` returns
`git rev-parse --short HEAD`, and appending a line to `core/config.py` does not
change it:

    commit before the edit    ea29e33
    commit after the edit     ea29e33      <- the code changed, the identity did not

A snapshot recording that commit claims a reproducibility it cannot deliver.

**An obvious fix that does not work, checked before using it.** `git ls-files -s`
looks like a cheap tracked-content digest, but it reports STAGED blob hashes, so
an unstaged edit leaves it unchanged — dirt-blind in exactly the same way.

**What does work** is hashing the worktree through `git hash-object`: stable when
clean, moves on any edit, and returns to the prior value on revert. 0.4 s over
339 files, affordable because a release snapshot is taken once per release.

So a snapshot records BOTH: the commit for provenance (where this came from) and
the worktree digest for integrity (whether it still is that). Neither answers
the other's question.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    RELEASE_SNAPSHOT_VERSION,
)


class ReleaseSnapshotError(ValueError):
    """Raised when a snapshot request is structurally invalid."""


# Why a snapshot could not be frozen. Distinct codes, because the fixes differ.
RELEASE_REASON_NO_GIT = "GIT_UNAVAILABLE"
RELEASE_REASON_DIRTY = "WORKTREE_DIRTY"
RELEASE_REASON_MISSING = "COMPONENTS_ABSENT"

# Component states. ABSENT is explicit and load-bearing: a component omitted from
# the manifest is a builder bug, one recorded ABSENT is a release blocker.
COMPONENT_PRESENT = "PRESENT"
COMPONENT_ABSENT = "ABSENT"
COMPONENT_STATES: tuple[str, ...] = (COMPONENT_PRESENT, COMPONENT_ABSENT)


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command, or None when git cannot answer.

    None rather than a sentinel string: `_code_commit()` returns the literal
    "unknown" on failure, which is indistinguishable from a commit named
    "unknown" once it is written into a record.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
            cwd=str(cwd) if cwd else None,
        )
    except Exception:
        return None
    return result.stdout


def commit_id(cwd: Path | None = None) -> str | None:
    """The current commit, or None when git cannot say."""
    out = _git("rev-parse", "--short", "HEAD", cwd=cwd)
    return out.strip() if out else None


def tracked_files(cwd: Path | None = None) -> list[str] | None:
    out = _git("ls-files", cwd=cwd)
    if out is None:
        return None
    return sorted(line for line in out.splitlines() if line.strip())


def worktree_digest(cwd: Path | None = None) -> str | None:
    """A digest of the tracked files AS THEY ARE ON DISK.

    Hashes the worktree, not the index. MEASURED, `git ls-files -s` reports
    STAGED blob hashes and does not move on an unstaged edit, so it would carry
    the same defect as the bare commit hash while looking like a fix.
    """
    files = tracked_files(cwd)
    if files is None:
        return None
    try:
        result = subprocess.run(
            ["git", "hash-object", "--stdin-paths"],
            input="\n".join(files),
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            cwd=str(cwd) if cwd else None,
        )
    except Exception:
        return None
    # Hash the per-file hashes with their paths, so a rename changes the digest
    # even when the contents do not.
    blobs = result.stdout.split()
    if len(blobs) != len(files):
        return None
    payload = json.dumps(dict(zip(files, blobs)), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_dirty(cwd: Path | None = None) -> bool | None:
    """Whether tracked files differ from HEAD. None when git cannot say.

    UNTRACKED FILES DO NOT COUNT. A release freezes what it ships, and an
    untracked scratch file or a gitignored data store is not part of that —
    `--porcelain` alone would call every working directory dirty forever.
    """
    out = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    if out is None:
        return None
    return bool(out.strip())


def config_digest() -> str:
    """A deterministic digest over every public config constant."""
    from core import config as config_module

    constants = {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if name.isupper()
    }

    def canonical(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    payload = {name: canonical(value) for name, value in sorted(constants.items())}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _component(state: str, detail: str, **extra) -> dict:
    if state not in COMPONENT_STATES:
        raise ReleaseSnapshotError(f"{state!r} is not a component state")
    record = {"state": state, "detail": detail}
    record.update(extra)
    return record


def _probe(
    name: str, probe: Callable[[], dict]
) -> dict:
    """Run one component probe, turning a failure into an explicit ABSENT.

    A probe that raises must not abort the snapshot: the point of X9 is to say
    which components are missing, and a traceback says nothing.
    """
    try:
        return probe()
    except Exception as error:  # noqa: BLE001 - the reason is recorded, not raised
        return _component(
            COMPONENT_ABSENT,
            f"probing {name} raised {type(error).__name__}: {error}",
        )


def collect_components(
    *,
    runs: Sequence[Mapping[str, Any]] | None = None,
    validation: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, dict]:
    """Probe every component X9 freezes, each stating PRESENT or ABSENT.

    Every name in RELEASE_COMPONENTS gets an entry, always. A component that is
    simply missing from the mapping would be indistinguishable from one checked
    and found absent, and only the second is a release blocker.
    """
    root = repo_root or Path(__file__).resolve().parent.parent

    def code() -> dict:
        commit = commit_id(root)
        digest = worktree_digest(root)
        dirty = is_dirty(root)
        if commit is None or digest is None:
            return _component(
                COMPONENT_ABSENT,
                "git could not identify the working tree, so the code cannot "
                "be frozen or verified",
            )
        return _component(
            COMPONENT_PRESENT,
            f"commit {commit}, worktree digest {digest[:12]}",
            commit=commit,
            worktree_digest=digest,
            dirty=dirty,
        )

    def configuration() -> dict:
        digest = config_digest()
        return _component(
            COMPONENT_PRESENT, f"config digest {digest[:12]}", digest=digest
        )

    def features() -> dict:
        from core.feature_registry import build_default_registry

        registry = build_default_registry()
        names = sorted(registry.feature_names())
        if not names:
            return _component(COMPONENT_ABSENT, "the feature registry is empty")
        return _component(
            COMPONENT_PRESENT,
            f"{len(names)} registered features",
            count=len(names),
            feature_set_hash=registry.feature_set_hash(names),
        )

    def models() -> dict:
        records = list(runs) if runs is not None else _load_runs()
        if not records:
            return _component(
                COMPONENT_ABSENT,
                "no training runs are recorded, so there is no model to freeze",
            )
        hashes = sorted(
            str(record.get("artifact_hash") or "") for record in records
        )
        if not all(hashes):
            return _component(
                COMPONENT_ABSENT,
                "a training run carries no artifact hash, so its model cannot "
                "be identified",
            )
        return _component(
            COMPONENT_PRESENT,
            f"{len(records)} runs with artifact hashes",
            count=len(records),
            digest=hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )

    def weights() -> dict:
        # The two versioned sets W1 established. Named explicitly rather than
        # discovered by prefix, so adding a third set is a deliberate change to
        # what a release freezes and not a silent one.
        from core.config import (
            ENSEMBLE_VERSION,
            ENSEMBLE_WEIGHTS_CURRENT,
            ENSEMBLE_WEIGHTS_LONG,
        )

        sets = {
            "CURRENT": ENSEMBLE_WEIGHTS_CURRENT,
            "LONG": ENSEMBLE_WEIGHTS_LONG,
        }
        empty = sorted(name for name, mapping in sets.items() if not mapping)
        if empty:
            return _component(
                COMPONENT_ABSENT, f"empty ensemble weight set(s): {', '.join(empty)}"
            )
        return _component(
            COMPONENT_PRESENT,
            f"{len(sets)} weight sets, {ENSEMBLE_VERSION}",
            sets=sorted(sets),
            ensemble_version=ENSEMBLE_VERSION,
            digest=hashlib.sha256(
                json.dumps(
                    {
                        name: dict(sorted(dict(mapping).items()))
                        for name, mapping in sorted(sets.items())
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        )

    def calibration() -> dict:
        from core.config import CALIBRATION_VERSION

        return _component(
            COMPONENT_PRESENT,
            f"calibration {CALIBRATION_VERSION}",
            version=CALIBRATION_VERSION,
        )

    def data() -> dict:
        # Only TRACKED ledgers count. A gitignored store is absent on a fresh
        # clone, so freezing a digest of one would record a number nobody else
        # can reproduce — the defect open item 6 records twice over.
        ledgers = {
            "training_datasets.jsonl": root / "data" / "training_datasets.jsonl",
            "training_runs.jsonl": root / "data" / "training_runs.jsonl",
            "research_trials.jsonl": root / "data" / "research_trials.jsonl",
        }
        missing = sorted(name for name, path in ledgers.items() if not path.exists())
        if missing:
            return _component(
                COMPONENT_ABSENT,
                f"tracked ledgers absent: {', '.join(missing)}",
            )
        digests = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(ledgers.items())
        }
        return _component(
            COMPONENT_PRESENT,
            f"{len(digests)} tracked ledgers",
            ledgers=digests,
        )

    def validation_component() -> dict:
        # X1-X8 produce the validation results X9 freezes. NOT_EVALUATED
        # verdicts are a legitimate result and still freeze; what cannot freeze
        # is the absence of any verdict at all.
        if not validation:
            return _component(
                COMPONENT_ABSENT,
                "no validation results were supplied, so the release carries no "
                "evidence about its own performance",
            )
        verdicts = {
            str(gate): str((result or {}).get("verdict") or "")
            for gate, result in sorted(validation.items())
        }
        unnamed = sorted(gate for gate, verdict in verdicts.items() if not verdict)
        if unnamed:
            return _component(
                COMPONENT_ABSENT,
                f"validation results without a verdict: {', '.join(unnamed)}",
            )
        return _component(
            COMPONENT_PRESENT,
            f"{len(verdicts)} gate verdicts",
            verdicts=verdicts,
            digest=hashlib.sha256(
                json.dumps(verdicts, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )

    probes: dict[str, Callable[[], dict]] = {
        "code": code,
        "configuration": configuration,
        "features": features,
        "models": models,
        "weights": weights,
        "calibration": calibration,
        "data": data,
        "validation": validation_component,
    }
    collected = {
        name: _probe(name, probes[name])
        for name in RELEASE_COMPONENTS
        if name in probes
    }
    # EVERY declared component gets an entry, even one with no probe: silence
    # would read as "fine".
    for name in RELEASE_COMPONENTS:
        collected.setdefault(
            name,
            _component(
                COMPONENT_ABSENT, f"no probe exists for the {name!r} component"
            ),
        )
    return collected


def _load_runs() -> list[dict[str, Any]]:
    try:
        from core.training import load_training_runs

        return list(load_training_runs())
    except Exception:
        return []


def _result(verdict: str, reason: str, **detail) -> dict:
    if verdict not in RELEASE_SNAPSHOT_VERDICTS:
        raise ReleaseSnapshotError(f"{verdict!r} is not a release verdict")
    record = {
        "version": RELEASE_SNAPSHOT_VERSION,
        "verdict": verdict,
        "reason": reason,
        "blocks_trades": RELEASE_BLOCKS_TRADES,
    }
    record.update(detail)
    return record


def snapshot_digest(components: Mapping[str, Mapping[str, Any]]) -> str:
    """One digest over every component's own digest. Order-independent."""
    payload = {
        name: {
            key: value
            for key, value in sorted((record or {}).items())
            # `detail` is prose and `dirty` is a property of the moment, not of
            # the content: including either would make the digest unstable
            # across reruns that froze identical material.
            if key not in ("detail", "dirty")
        }
        for name, record in sorted(components.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()


def build_release_snapshot(
    *,
    runs: Sequence[Mapping[str, Any]] | None = None,
    validation: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Freeze the release, or say precisely why it cannot be frozen."""
    components = collect_components(
        runs=runs, validation=validation, repo_root=repo_root
    )
    absent = sorted(
        name
        for name, record in components.items()
        if record.get("state") != COMPONENT_PRESENT
    )
    code = components.get("code") or {}
    digest = snapshot_digest(components)

    if code.get("state") != COMPONENT_PRESENT:
        return _result(
            RELEASE_NOT_EVALUATED,
            (
                f"the code could not be identified ({code.get('detail')}), so "
                f"nothing else in the snapshot can be tied to a version"
            ),
            reason_code=RELEASE_REASON_NO_GIT,
            components=components,
            absent=absent,
            snapshot_digest=digest,
        )

    if absent:
        return _result(
            RELEASE_INCOMPLETE,
            (
                f"{len(absent)} component(s) absent: {', '.join(absent)}. A "
                f"release cannot be frozen around a component that does not "
                f"exist"
            ),
            reason_code=RELEASE_REASON_MISSING,
            components=components,
            absent=absent,
            snapshot_digest=digest,
            commit=code.get("commit"),
            worktree_digest=code.get("worktree_digest"),
        )

    if RELEASE_DIRTY_IS_NOT_RELEASABLE and code.get("dirty"):
        return _result(
            RELEASE_DIRTY,
            (
                f"every component is present, but tracked files differ from "
                f"commit {code.get('commit')}: this digest matches no commit "
                f"anyone else can check out"
            ),
            reason_code=RELEASE_REASON_DIRTY,
            components=components,
            absent=absent,
            snapshot_digest=digest,
            commit=code.get("commit"),
            worktree_digest=code.get("worktree_digest"),
        )

    return _result(
        RELEASE_FROZEN,
        (
            f"all {len(components)} components frozen at commit "
            f"{code.get('commit')}, worktree digest "
            f"{str(code.get('worktree_digest'))[:12]}"
        ),
        reason_code="",
        components=components,
        absent=absent,
        snapshot_digest=digest,
        commit=code.get("commit"),
        worktree_digest=code.get("worktree_digest"),
    )


def verify_snapshot(
    snapshot: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict:
    """Does the tree still match a snapshot taken earlier?

    This is the question a freeze exists to answer, and the reason the worktree
    digest is recorded: MEASURED, the commit alone reports a match across an
    edit that changed the code.
    """
    if not isinstance(snapshot, Mapping):
        raise ReleaseSnapshotError("a snapshot must be a mapping")
    recorded = snapshot.get("worktree_digest")
    if not recorded:
        return {
            "matches": None,
            "reason": (
                "the snapshot records no worktree digest, so whether the tree "
                "changed cannot be determined — which is the defect X9 measured"
            ),
        }
    current = worktree_digest(repo_root or Path(__file__).resolve().parent.parent)
    if current is None:
        return {
            "matches": None,
            "reason": "git could not digest the working tree",
        }
    matches = current == recorded
    return {
        "matches": matches,
        "recorded": recorded,
        "current": current,
        "reason": (
            "the working tree is identical to the snapshot"
            if matches
            else "the working tree has changed since the snapshot was taken"
        ),
    }


def release_snapshot_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a release snapshot. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    verdict = report.get("verdict")
    if verdict not in RELEASE_SNAPSHOT_VERDICTS:
        problems.append(f"unknown release verdict {verdict!r}")
    if not str(report.get("reason") or "").strip():
        problems.append("no reason given")
    if report.get("blocks_trades"):
        problems.append("X9 reports; the registry promotes")

    components = report.get("components")
    if RELEASE_REQUIRES_EXPLICIT_ABSENCE:
        if not isinstance(components, Mapping):
            problems.append("the snapshot records no component states")
        else:
            for name in RELEASE_COMPONENTS:
                record = components.get(name)
                if not isinstance(record, Mapping):
                    problems.append(
                        f"{name}: no state recorded — an omitted component is "
                        f"indistinguishable from one found absent"
                    )
                elif record.get("state") not in COMPONENT_STATES:
                    problems.append(
                        f"{name}: state {record.get('state')!r} is not "
                        f"PRESENT or ABSENT"
                    )
                elif not str(record.get("detail") or "").strip():
                    problems.append(f"{name}: no detail given")

    if verdict == RELEASE_FROZEN:
        if RELEASE_REQUIRES_WORKTREE_DIGEST and not report.get("worktree_digest"):
            problems.append(
                "FROZEN without a worktree digest: MEASURED, the commit hash "
                "alone cannot tell a frozen tree from an edited one"
            )
        if RELEASE_RECORDS_COMMIT and not report.get("commit"):
            problems.append("FROZEN without a commit")
        if report.get("absent"):
            problems.append(
                f"FROZEN while {report['absent']} are absent; a release cannot "
                f"freeze around a component that does not exist"
            )
        code = (components or {}).get("code") or {}
        if RELEASE_DIRTY_IS_NOT_RELEASABLE and code.get("dirty"):
            problems.append("FROZEN from a dirty worktree")
    if verdict == RELEASE_INCOMPLETE and not report.get("absent"):
        problems.append("INCOMPLETE but nothing is recorded as absent")
    if verdict in (RELEASE_NOT_EVALUATED, RELEASE_INCOMPLETE, RELEASE_DIRTY):
        if not str(report.get("reason_code") or "").strip():
            problems.append(f"{verdict} without a reason code")
    return problems


def render_release_snapshot(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per component plus the verdict."""
    lines = [f"  verdict            {report.get('verdict')}"]
    commit = report.get("commit")
    lines.append(f"  commit             {commit or 'ABSENT'}")
    digest = report.get("worktree_digest")
    lines.append(
        f"  worktree digest    {str(digest)[:12] if digest else 'ABSENT'}"
    )
    snapshot = report.get("snapshot_digest")
    lines.append(
        f"  snapshot digest    {str(snapshot)[:12] if snapshot else 'ABSENT'}"
    )
    for name, record in sorted((report.get("components") or {}).items()):
        state = (record or {}).get("state", "ABSENT")
        lines.append(f"    {name:14s} {state:8s} {(record or {}).get('detail', '')}")
    if str(report.get("reason_code") or "").strip():
        lines.append(f"  reason code        {report['reason_code']}")
    return lines


# C4: `snapshot_problems` was defined under that name in THREE modules — this
# one for X9's release snapshot, plus the other two snapshots. They are genuinely different
# objects with different contracts, so this is not a W5 duplicate; the shared
# NAME was the problem. A reader greping it found three functions with no way to
# tell which a call site meant, and importing two into one file would have
# silently shadowed one.
#
# The alias keeps any external caller working. In-repo callers all use the
# explicit name.
snapshot_problems = release_snapshot_problems
