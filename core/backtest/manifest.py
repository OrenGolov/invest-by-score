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
