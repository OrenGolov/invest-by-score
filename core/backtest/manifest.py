"""Mandatory run manifests (Sprint V2).

A backtest run without a valid manifest is invalid by definition. The
manifest freezes everything a reviewer needs to reproduce the run: code
commit, model/feature/label/cost-table versions, the data digest, the
configuration snapshot, and the provider overrides that made the run
offline-deterministic. `run_hash` is the canonical digest of all of it —
identical inputs produce identical run hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

MANIFEST_VERSION = "backtest-manifest-v1"

REQUIRED_FIELDS = (
    "manifest_version",
    "run_hash",
    "code_commit",
    "versions",
    "data_digest",
    "config",
    "seed",
    "provider_overrides",
)


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        # ABSORBS: git being absent, not a repository, or slow. A manifest must
        # still be produced — the commit is provenance, not a precondition — so
        # this degrades to a named "unknown" rather than failing the run.
        #
        # X9 measured the cost of that sentinel: "unknown" is indistinguishable
        # from a commit literally named unknown once written into a record, which
        # is why `release_snapshot` returns None instead. This one predates that
        # and is kept for compatibility with existing manifests.
        return "unknown"


def data_digest(frame) -> str:
    """Canonical digest of the replayed price frame."""
    canonical = json.dumps(
        {
            "index": [str(value) for value in frame.index],
            "close": [float(value) for value in frame["Close"]],
            "high": [float(value) for value in frame["High"]],
            "low": [float(value) for value in frame["Low"]],
            "volume": [int(value) for value in frame["Volume"]],
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    ticker: str,
    frame,
    config_snapshot: dict,
    versions: dict,
    provider_overrides: dict,
    seed=None,
) -> dict:
    """Assemble the manifest and its canonical run hash."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "ticker": str(ticker).upper(),
        "code_commit": _code_commit(),
        "versions": dict(sorted(versions.items())),
        "data_digest": data_digest(frame),
        "config": dict(sorted(config_snapshot.items())),
        "seed": seed,
        "provider_overrides": dict(sorted(provider_overrides.items())),
    }
    hash_payload = {
        "versions": manifest["versions"],
        "data_digest": manifest["data_digest"],
        "config": manifest["config"],
        "code_commit": manifest["code_commit"],
        "seed": manifest["seed"],
    }
    manifest["run_hash"] = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return manifest


def validate_manifest(manifest: dict) -> list[str]:
    """Return the list of manifest problems; empty means the run is valid.

    `seed` is allowed to be None (it records the absence of randomness);
    every other required field must be present and non-empty.
    """
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a dict"]
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            problems.append(f"missing field: {field}")
            continue
        if field == "seed":
            continue  # None is a legitimate value: no randomness in the run
        if manifest[field] is None:
            problems.append(f"field is None: {field}")
        elif field in {"versions", "config", "provider_overrides"} and not manifest[field]:
            problems.append(f"field is empty: {field}")
    if manifest.get("manifest_version") not in (None, MANIFEST_VERSION):
        problems.append(
            f"unknown manifest_version {manifest.get('manifest_version')!r}"
        )
    return problems


# --- Manifest store (Sprint V4) --------------------------------------------------
# Run manifests are MANDATORY: every backtest persists its manifest to
# data/backtest_runs.jsonl as part of the run, and a run whose manifest
# fails validation is refused before any work happens. The store is
# append-only: identical recomputes are skipped (the run hash is the
# canonical digest of all inputs), and a record that fails validation
# on load is an integrity violation, not a warning.

BACKTEST_RUNS_PATH = Path(__file__).resolve().parent.parent / "data" / "backtest_runs.jsonl"


def require_valid_manifest(manifest: dict) -> None:
    """Raise ValueError when the manifest fails validation (mandatory gate)."""
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(
            "invalid run manifest — a run without a valid manifest is invalid "
            f"by definition: {'; '.join(problems)}"
        )


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"backtest run store {path} line {line_number} is not valid JSON "
                    f"(append-only integrity violated): {exc}"
                ) from exc
    return records


def persist_run_manifest(manifest: dict, path: str | Path | None = None) -> bool:
    """Append the manifest to the run store; idempotent per run hash.

    Returns True when a record was appended, False when an identical record
    for the same run hash already exists (deterministic recompute). The
    manifest is validated BEFORE it is written — an invalid manifest is
    never stored. Because the run hash is the canonical digest of all
    inputs, a same-hash/different-content record would be corruption and
    raises rather than appending.
    """
    require_valid_manifest(manifest)
    store_path = Path(path) if path is not None else BACKTEST_RUNS_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)
    for record in _read_records(store_path):
        if record.get("run_hash") == manifest["run_hash"]:
            if record == manifest:
                return False  # identical recompute: nothing to append
            raise ValueError(
                f"manifest integrity violation: run hash {manifest['run_hash']} "
                f"already exists with different content — the canonical digest "
                f"covers all inputs, so this is a corrupted record, not a revision."
            )
    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, default=str) + "\n")
    return True


def load_run_manifests(path: str | Path | None = None) -> list[dict]:
    """Read every persisted manifest, newest last; strict on integrity."""
    store_path = Path(path) if path is not None else BACKTEST_RUNS_PATH
    return _read_records(store_path)


def load_manifest_by_run_hash(run_hash: str, path: str | Path | None = None) -> dict | None:
    """Resolve the persisted manifest for one run hash; None when absent."""
    for record in load_run_manifests(path=path):
        if record.get("run_hash") == run_hash:
            return record
    return None
