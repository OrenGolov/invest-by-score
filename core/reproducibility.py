"""ML reproducibility gate (Sprint M8).

The rule:

    **Same data, features, seed, code, configuration must produce identical
    model artifacts/predictions — within explicitly defined reproducibility
    guarantees.**

The last clause carries the weight. A blanket "bit-identical forever" claim
would be false the moment numpy changes a summation order or sklearn retunes
an estimator's internals, and a guarantee that is quietly false is worse than
none. So this module states exactly what is promised, verifies it, and
records the environment needed to explain a divergence that falls outside
the promise.

GUARANTEED, same environment and same inputs:
  - identical fitted parameters, hence an identical artifact hash
  - identical predictions, metrics and run hash
  - identical results across separate processes on the same machine

NOT GUARANTEED, and deliberately so:
  - bit-identical artifacts across library versions or platforms. BLAS
    summation order and estimator internals change between releases; the
    environment block records versions so a divergence is attributable
    rather than mysterious.
  - stability across a code change that alters a formula. That is what the
    `*_VERSION` constants exist to make visible.

Two identities, deliberately separate:

  - `artifact_hash` identifies the FITTED MODEL: estimator, feature surface
    and fitted parameters. It does NOT include the seed, because a seed that
    changed nothing about the parameters did not produce a different
    artifact — for `historical_mean` and `ridge` the fit is deterministic and
    the seed is inert.
  - `run_hash` identifies the CONFIGURATION that produced it, seed included.

Conflating them would report a false difference for every deterministic
estimator, which is exactly the kind of noise that trains people to ignore a
gate.

Comparison tolerance is zero. Within one environment the same inputs must
produce the same floats, not merely close ones — a tolerance would hide the
nondeterminism this gate exists to catch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from core.config import (
    REPRODUCIBILITY_ENVIRONMENT_KEYS,
    REPRODUCIBILITY_GUARANTEES,
    REPRODUCIBILITY_NON_GUARANTEES,
    REPRODUCIBILITY_TOLERANCE,
    REPRODUCIBILITY_VERSION,
)


class ReproducibilityError(ValueError):
    """Raised when a reproducibility check cannot be performed."""


@dataclass
class ReproducibilityReport:
    """The outcome of comparing two runs that should be identical."""

    reproducible: bool
    estimator: str
    divergences: list[str] = field(default_factory=list)
    environment_matches: bool = True
    environment_differences: list[str] = field(default_factory=list)
    guarantees: list[str] = field(default_factory=lambda: list(REPRODUCIBILITY_GUARANTEES))
    non_guarantees: list[str] = field(
        default_factory=lambda: list(REPRODUCIBILITY_NON_GUARANTEES)
    )
    tolerance: float = REPRODUCIBILITY_TOLERANCE
    version: str = REPRODUCIBILITY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def explanation(self) -> str:
        """Why the runs diverged, in the terms the guarantees are stated in."""
        if self.reproducible:
            return "runs are identical under the stated guarantees"
        if not self.environment_matches:
            return (
                "runs diverged AND the environments differ ("
                + "; ".join(self.environment_differences)
                + ") — cross-environment bitwise identity is NOT guaranteed, so "
                "this is an expected divergence, not a defect"
            )
        return (
            "runs diverged within the SAME environment — this violates a stated "
            "guarantee and is a defect: " + "; ".join(self.divergences[:3])
        )


def environment_differences(left: dict, right: dict) -> list[str]:
    """Which recorded environment fields differ between two runs."""
    differences: list[str] = []
    for key in REPRODUCIBILITY_ENVIRONMENT_KEYS:
        left_value = (left or {}).get(key)
        right_value = (right or {}).get(key)
        if left_value != right_value:
            differences.append(f"{key}: {left_value!r} != {right_value!r}")
    return differences


def predictions_of(run) -> list[float]:
    """Every fold's out-of-sample predictions, in fold order."""
    values: list[float] = []
    for fold in sorted(getattr(run, "folds", []) or [], key=lambda f: f.fold_id):
        values.extend(fold.predictions or [])
    return values


def compare_predictions(left: Sequence[float], right: Sequence[float]) -> list[str]:
    """Exact comparison. Tolerance is zero by design."""
    if len(left) != len(right):
        return [f"prediction count differs: {len(left)} != {len(right)}"]
    divergences: list[str] = []
    for index, (a, b) in enumerate(zip(left, right)):
        if abs(float(a) - float(b)) > REPRODUCIBILITY_TOLERANCE:
            divergences.append(f"prediction[{index}]: {a!r} != {b!r}")
            if len(divergences) >= 5:
                divergences.append("... further prediction differences truncated")
                break
    return divergences


def compare_runs(first, second) -> ReproducibilityReport:
    """Verify two runs that SHOULD be identical actually are.

    Checks the whole chain — artifact, predictions, metrics, per-fold
    results, run identity — rather than trusting any single hash. A hash
    match with differing predictions would mean the hash covers the wrong
    thing, so the parts are compared independently.
    """
    for run, label in ((first, "first"), (second, "second")):
        if not getattr(run, "folds", None):
            raise ReproducibilityError(f"{label} run has no folds — nothing to compare")

    estimator = getattr(first, "estimator", "?")
    if estimator != getattr(second, "estimator", "?"):
        raise ReproducibilityError(
            f"cannot compare different estimators: {estimator!r} vs "
            f"{getattr(second, 'estimator', '?')!r}"
        )

    divergences: list[str] = []

    if first.artifact_hash != second.artifact_hash:
        divergences.append(
            f"artifact_hash: {first.artifact_hash[:16]}... != {second.artifact_hash[:16]}..."
        )
    if first.run_hash() != second.run_hash():
        divergences.append(
            f"run_hash: {first.run_hash()[:16]}... != {second.run_hash()[:16]}..."
        )
    if first.metrics != second.metrics:
        differing = sorted(
            key for key in set(first.metrics) | set(second.metrics)
            if first.metrics.get(key) != second.metrics.get(key)
        )
        divergences.append(f"metrics differ on {differing}")
    if first.dataset_hash != second.dataset_hash:
        divergences.append("dataset_hash differs — the runs used different data")
    if first.feature_set_hash != second.feature_set_hash:
        divergences.append("feature_set_hash differs — the runs used different features")
    if first.seed != second.seed:
        divergences.append(f"seed: {first.seed} != {second.seed}")

    if len(first.folds) != len(second.folds):
        divergences.append(f"fold count: {len(first.folds)} != {len(second.folds)}")
    else:
        for left_fold, right_fold in zip(
            sorted(first.folds, key=lambda f: f.fold_id),
            sorted(second.folds, key=lambda f: f.fold_id),
        ):
            if left_fold.metrics != right_fold.metrics:
                divergences.append(f"fold {left_fold.fold_id} metrics differ")

    divergences.extend(compare_predictions(predictions_of(first), predictions_of(second)))

    differences = environment_differences(
        getattr(first, "environment", {}), getattr(second, "environment", {})
    )
    return ReproducibilityReport(
        reproducible=not divergences,
        estimator=estimator,
        divergences=divergences,
        environment_matches=not differences,
        environment_differences=differences,
    )


def verify_reproducible(dataset, estimator: str, seed: int, **fold_kwargs) -> ReproducibilityReport:
    """Train the same configuration twice and prove the results match.

    This is the gate's core assertion, and it deliberately RE-TRAINS rather
    than re-reading a stored hash: comparing a run against itself would
    prove nothing.
    """
    from core.training import train_baseline

    first = train_baseline(dataset, estimator, seed=seed, **fold_kwargs)
    second = train_baseline(dataset, estimator, seed=seed, **fold_kwargs)
    return compare_runs(first, second)


def reproducibility_manifest(run) -> dict[str, Any]:
    """Everything needed to reproduce a run, plus the terms of the promise.

    Shipped with an artifact so a future reader knows what was guaranteed at
    the time, not what the current code happens to guarantee.
    """
    missing = [
        key for key in REPRODUCIBILITY_ENVIRONMENT_KEYS
        if not (getattr(run, "environment", {}) or {}).get(key)
    ]
    if missing:
        raise ReproducibilityError(
            f"run does not record {missing} — a divergence could not be "
            f"attributed to an environment change"
        )
    return {
        "version": REPRODUCIBILITY_VERSION,
        "estimator": run.estimator,
        "artifact_hash": run.artifact_hash,
        "artifact_hash_basis": run.artifact_hash_basis,
        "run_hash": run.run_hash(),
        "dataset_hash": run.dataset_hash,
        "feature_set_hash": run.feature_set_hash,
        "seed": run.seed,
        "hyperparameters": dict(run.hyperparameters),
        "pipeline_version": run.pipeline_version,
        "environment": dict(run.environment),
        "guarantees": list(REPRODUCIBILITY_GUARANTEES),
        "non_guarantees": list(REPRODUCIBILITY_NON_GUARANTEES),
        "tolerance": REPRODUCIBILITY_TOLERANCE,
    }
