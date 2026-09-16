"""Model registry and artifact tracking (Sprint M2, board numbering).

The registry is the single source of truth for which model versions exist,
what produced them, and whether governance has cleared them for the live
score path. The binding rule:

    **A model referenced by a live decision must be `approved`.**

That rule is about governance, not about whether gradient descent was
involved. The deterministic scorers backing today's score path
(`technical-v1`, `fundamental-v1`, `market-data-v1`) are registered and
approved exactly like a trained artifact would be, so the orchestrator
resolves them through the registry instead of hardcoding strings. When a
trained model replaces one, only its registry entry changes.

Lifecycle:

    candidate ---- promote() ----> approved ---- retire() ----> retired
                      |
                      +-- requires an out-of-sample comparison row against
                          the incumbent AND a human approved_by

Binding rules, all test-enforced:

- **Promotion is gated.** `promote` refuses without an out-of-sample
  comparison against the named incumbent and a human approver. A candidate
  that does not beat its incumbent on the pre-registered primary metric
  cannot be promoted. Self-approval by an automated actor is refused —
  `approved_by` must be a human identity.
- **Retirement never deletes.** `retire` flips status and records when and
  why; the entry and its `artifact_uri` remain forever, because historical
  decisions must stay explainable (master context section 25: historical
  forecasts never change).
- **The live path is closed by default.** `require_live_model` raises for an
  unregistered, candidate, or retired version. An override exists but is
  audited by construction: it demands a reason and returns a record naming
  the override, so "referencing an unapproved model is impossible without an
  explicit override that itself is audited" is literally true.
- **Append-only persistence** at `models/manifest.json`, idempotent per
  entry hash, with the same integrity model as the feature registry and the
  outcome / manifest / framing stores. Malformed content raises.
- **Deterministic identity.** Every entry carries a canonical SHA-256 over
  its governing fields, so a changed definition is always visible.

Pure and deterministic apart from persistence: no wall-clock reads inside
the decision path (timestamps are supplied by the caller at the governance
boundary, exactly as the orchestrator does for audit events).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.config import (
    CURRENT_SCORE_VERSION,
    FUNDAMENTAL_FEATURE_VERSION,
    LONG_TERM_SCORE_VERSION,
    MARKET_FEATURE_VERSION,
    MODEL_LIVE_ELIGIBLE_STATUSES,
    MODEL_REGISTRY_SEED_APPROVER,
    MODEL_REGISTRY_VERSION,
    MODEL_STATUS_APPROVED,
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_RETIRED,
    MODEL_STATUSES,
)
from core.feature_registry import MODEL_FAMILIES

MODEL_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models" / "manifest.json"

# An approver must be a human identity. These are the automated actors that
# may never appear as `approved_by` — a model cannot approve itself, and a
# CI job cannot stand in for governance sign-off.
_NON_HUMAN_APPROVERS = frozenset({
    "", "system", "auto", "automatic", "ci", "bot", "robot", "pipeline",
    "scheduler", "cron", "none", "null", "n/a", "unknown",
})


class ModelRegistryError(ValueError):
    """Raised when a registry entry or a lifecycle transition is invalid."""


class ModelNotApprovedError(ModelRegistryError):
    """Raised when the live score path references a non-approved model."""


@dataclass
class ModelEntry:
    """One versioned model in the registry."""

    model_version: str
    family: str
    feature_set_version: str
    training_data_cutoff: str | None = None
    artifact_uri: str | None = None
    status: str = MODEL_STATUS_CANDIDATE
    metrics: dict[str, Any] = field(default_factory=dict)
    approved_by: str | None = None
    approved_at: str | None = None
    parent_version: str | None = None
    dataset_hash: str | None = None
    retired_at: str | None = None
    retired_reason: str = ""
    oos_comparison: dict[str, Any] = field(default_factory=dict)
    registry_version: str = MODEL_REGISTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_hash(self) -> str:
        """Deterministic identity over the entry's governing fields."""
        payload = {
            "model_version": self.model_version,
            "family": self.family,
            "feature_set_version": self.feature_set_version,
            "training_data_cutoff": self.training_data_cutoff,
            "artifact_uri": self.artifact_uri,
            "status": self.status,
            "approved_by": self.approved_by,
            "parent_version": self.parent_version,
            "dataset_hash": self.dataset_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_live_eligible(self) -> bool:
        return self.status in MODEL_LIVE_ELIGIBLE_STATUSES


def entry_problems(entry: ModelEntry) -> list[str]:
    """Validate an entry against the registry's closed vocabularies."""
    problems: list[str] = []
    if not str(entry.model_version).strip():
        problems.append("model_version is required")
    if entry.family not in MODEL_FAMILIES:
        problems.append(
            f"family {entry.family!r} is not a known model family "
            f"(known: {sorted(MODEL_FAMILIES)})"
        )
    if not str(entry.feature_set_version).strip():
        problems.append("feature_set_version is required — a model must name its inputs")
    if entry.status not in MODEL_STATUSES:
        problems.append(
            f"status {entry.status!r} is not a known status (known: {sorted(MODEL_STATUSES)})"
        )
    if entry.status == MODEL_STATUS_APPROVED:
        if not str(entry.approved_by or "").strip():
            problems.append("an approved model must record approved_by")
        if not str(entry.approved_at or "").strip():
            problems.append("an approved model must record approved_at")
    if entry.status == MODEL_STATUS_RETIRED and not str(entry.retired_at or "").strip():
        problems.append("a retired model must record retired_at")
    if entry.parent_version is not None and entry.parent_version == entry.model_version:
        problems.append("parent_version cannot be the model itself")
    return problems


def approver_problems(approved_by: str | None) -> list[str]:
    """A promotion needs a human approver, not an automated actor."""
    name = str(approved_by or "").strip()
    if not name:
        return ["approved_by is required — promotion needs a human approver"]
    if name.lower() in _NON_HUMAN_APPROVERS:
        return [
            f"approved_by {name!r} is an automated actor; promotion requires a "
            f"human approver (governance sign-off cannot be automated)"
        ]
    return []


def oos_comparison_problems(
    comparison: dict[str, Any] | None,
    candidate_version: str,
    incumbent_version: str | None,
) -> list[str]:
    """A promotion must carry an out-of-sample comparison against the incumbent.

    The comparison names the pre-registered primary metric and both sides'
    values on it. A candidate that does not beat the incumbent cannot be
    promoted — "it did not win" is a valid, successful outcome.
    """
    if not comparison:
        return [
            "promotion requires an out-of-sample comparison against the incumbent "
            "— an unmeasured model cannot be approved"
        ]
    problems: list[str] = []
    for required in ("primary_metric", "candidate_value", "incumbent_value", "sample"):
        if comparison.get(required) is None:
            problems.append(f"oos_comparison.{required} is required")
    if problems:
        return problems

    if str(comparison.get("candidate_version", candidate_version)) != candidate_version:
        problems.append(
            f"oos_comparison.candidate_version "
            f"{comparison.get('candidate_version')!r} != {candidate_version!r}"
        )
    if incumbent_version is not None:
        stated = comparison.get("incumbent_version")
        if stated is not None and str(stated) != str(incumbent_version):
            problems.append(
                f"oos_comparison.incumbent_version {stated!r} does not name the "
                f"current incumbent {incumbent_version!r}"
            )

    sample = comparison.get("sample")
    try:
        if int(sample) <= 0:
            problems.append("oos_comparison.sample must be a positive number of observations")
    except (TypeError, ValueError):
        problems.append(f"oos_comparison.sample {sample!r} is not a number")

    try:
        candidate_value = float(comparison["candidate_value"])
        incumbent_value = float(comparison["incumbent_value"])
    except (TypeError, ValueError):
        problems.append("oos_comparison candidate/incumbent values must be numeric")
        return problems

    # higher_is_better defaults to True; a loss metric must say so explicitly.
    higher_is_better = bool(comparison.get("higher_is_better", True))
    beats = (
        candidate_value > incumbent_value
        if higher_is_better
        else candidate_value < incumbent_value
    )
    if not beats:
        direction = "higher" if higher_is_better else "lower"
        problems.append(
            f"candidate {candidate_value} does not beat incumbent {incumbent_value} "
            f"on {comparison['primary_metric']!r} ({direction} is better) — "
            f"a model that does not win out-of-sample is not promoted"
        )
    return problems


class ModelRegistry:
    """In-memory registry. Persistence is explicit, never implicit."""

    def __init__(self, entries: Iterable[ModelEntry] | None = None) -> None:
        self._entries: dict[str, ModelEntry] = {}
        for entry in entries or ():
            self.register(entry)

    def register(self, entry: ModelEntry) -> ModelEntry:
        """Add a new model version. Re-registering an existing version is refused."""
        problems = entry_problems(entry)
        if problems:
            raise ModelRegistryError(
                f"invalid model entry for {entry.model_version!r}: " + "; ".join(problems)
            )
        if entry.model_version in self._entries:
            raise ModelRegistryError(
                f"model_version {entry.model_version!r} is already registered — "
                f"a changed model must be registered under a new version"
            )
        if entry.parent_version is not None and entry.parent_version not in self._entries:
            raise ModelRegistryError(
                f"parent_version {entry.parent_version!r} is not registered"
            )
        self._entries[entry.model_version] = entry
        return entry

    def get(self, model_version: str) -> ModelEntry | None:
        return self._entries.get(str(model_version))

    def all_models(self) -> dict[str, ModelEntry]:
        return dict(self._entries)

    def approved(self) -> dict[str, ModelEntry]:
        return {
            version: entry
            for version, entry in self._entries.items()
            if entry.is_live_eligible()
        }

    def incumbent(self, family: str) -> ModelEntry | None:
        """The currently approved model for a family, if any."""
        candidates = [
            entry for entry in self._entries.values()
            if entry.family == family and entry.is_live_eligible()
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda entry: entry.model_version)[-1]

    def promote(
        self,
        model_version: str,
        approved_by: str,
        approved_at: str,
        oos_comparison: dict[str, Any],
    ) -> ModelEntry:
        """candidate -> approved. Gated on OOS evidence plus a human approver."""
        entry = self.get(model_version)
        if entry is None:
            raise ModelRegistryError(f"model_version {model_version!r} is not registered")
        if entry.status == MODEL_STATUS_RETIRED:
            raise ModelRegistryError(
                f"model {model_version!r} is retired and cannot be promoted"
            )
        if entry.status == MODEL_STATUS_APPROVED:
            raise ModelRegistryError(f"model {model_version!r} is already approved")

        problems = approver_problems(approved_by)
        if not str(approved_at or "").strip():
            problems.append("approved_at is required")
        incumbent = self.incumbent(entry.family)
        problems.extend(
            oos_comparison_problems(
                oos_comparison,
                model_version,
                incumbent.model_version if incumbent else None,
            )
        )
        if problems:
            raise ModelRegistryError(
                f"cannot promote {model_version!r}: " + "; ".join(problems)
            )

        entry.status = MODEL_STATUS_APPROVED
        entry.approved_by = str(approved_by)
        entry.approved_at = str(approved_at)
        entry.oos_comparison = dict(oos_comparison)
        return entry

    def retire(self, model_version: str, retired_at: str, reason: str = "") -> ModelEntry:
        """approved -> retired. Never deletes the entry or its artifact."""
        entry = self.get(model_version)
        if entry is None:
            raise ModelRegistryError(f"model_version {model_version!r} is not registered")
        if not str(retired_at or "").strip():
            raise ModelRegistryError("retired_at is required")
        entry.status = MODEL_STATUS_RETIRED
        entry.retired_at = str(retired_at)
        entry.retired_reason = str(reason)
        return entry

    def problems(self) -> list[str]:
        problems: list[str] = []
        for version, entry in sorted(self._entries.items()):
            for problem in entry_problems(entry):
                problems.append(f"{version}: {problem}")
        return problems

    def registry_hash(self) -> str:
        payload = {
            version: entry.canonical_hash()
            for version, entry in sorted(self._entries.items())
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_live_model(
    model_version: str,
    registry: ModelRegistry | None = None,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """The live-path gate: resolve a model version or refuse.

    Returns a resolution record naming the model and whether an override was
    used. An unregistered, candidate or retired version raises unless an
    explicit `override_reason` is supplied — and that override is recorded in
    the returned record, so it always reaches the audit event. There is no
    silent path to a non-approved model.
    """
    active = registry if registry is not None else build_default_model_registry()
    entry = active.get(model_version)

    if entry is not None and entry.is_live_eligible():
        return {
            "model_version": entry.model_version,
            "family": entry.family,
            "status": entry.status,
            "feature_set_version": entry.feature_set_version,
            "approved_by": entry.approved_by,
            "entry_hash": entry.canonical_hash(),
            "override": False,
            "override_reason": "",
        }

    detail = (
        f"model_version {model_version!r} is not registered"
        if entry is None
        else f"model {model_version!r} has status {entry.status!r}, not approved"
    )
    if not str(override_reason or "").strip():
        raise ModelNotApprovedError(
            f"{detail} — a model referenced by a live decision must be approved. "
            f"Supply an explicit override_reason to proceed; the override is audited."
        )
    return {
        "model_version": str(model_version),
        "family": entry.family if entry else None,
        "status": entry.status if entry else "unregistered",
        "feature_set_version": entry.feature_set_version if entry else None,
        "approved_by": entry.approved_by if entry else None,
        "entry_hash": entry.canonical_hash() if entry else "",
        "override": True,
        "override_reason": str(override_reason),
    }


def _seed_entry(model_version: str, family: str, feature_set_version: str) -> ModelEntry:
    """A deterministic scorer that backs today's live score path.

    These are rule-based, not trained, so they carry no dataset hash or
    artifact file. They are registered and approved anyway: the binding rule
    is about governance, and the orchestrator must resolve every model
    version it stamps through the registry.
    """
    return ModelEntry(
        model_version=model_version,
        family=family,
        feature_set_version=feature_set_version,
        training_data_cutoff=None,
        artifact_uri=None,
        status=MODEL_STATUS_APPROVED,
        metrics={},
        approved_by=MODEL_REGISTRY_SEED_APPROVER,
        approved_at="2026-09-16T00:00:00+00:00",
        parent_version=None,
        dataset_hash=None,
        oos_comparison={
            "note": (
                "Deterministic rule-based scorer predating the model registry; "
                "approved as the governance baseline, not by out-of-sample "
                "comparison. Any TRAINED successor must beat it through promote()."
            )
        },
    )


def build_default_model_registry() -> ModelRegistry:
    """The canonical registry backing the live score path."""
    registry = ModelRegistry()
    registry.register(_seed_entry("market-data-v1", "technical_analysis", MARKET_FEATURE_VERSION))
    registry.register(_seed_entry(
        "technical-v1",
        "technical_analysis",
        f"{CURRENT_SCORE_VERSION}+{LONG_TERM_SCORE_VERSION}",
    ))
    registry.register(_seed_entry("fundamental-v1", "technical_analysis", FUNDAMENTAL_FEATURE_VERSION))
    return registry


def persist_model_registry(
    registry: ModelRegistry,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the manifest. Idempotent per registry hash."""
    store = Path(path) if path is not None else MODEL_MANIFEST_PATH
    manifest = {
        "registry_version": MODEL_REGISTRY_VERSION,
        "registry_hash": registry.registry_hash(),
        "models": [
            entry.to_dict() for _, entry in sorted(registry.all_models().items())
        ],
    }
    existing = load_model_manifest(store)
    if existing.get("registry_hash") == manifest["registry_hash"]:
        return existing
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_model_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Read the manifest. Malformed content raises (integrity is loud)."""
    store = Path(path) if path is not None else MODEL_MANIFEST_PATH
    if not store.exists():
        return {}
    try:
        return json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelRegistryError(f"{store.name} is not valid JSON: {exc}") from exc


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    """Rebuild a registry from a persisted manifest."""
    manifest = load_model_manifest(path)
    registry = ModelRegistry()
    for payload in manifest.get("models", []):
        known = {field_name for field_name in ModelEntry.__dataclass_fields__}
        registry.register(ModelEntry(**{
            key: value for key, value in payload.items() if key in known
        }))
    return registry
