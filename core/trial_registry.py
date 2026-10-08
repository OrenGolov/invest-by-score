"""Research trial registry (Sprint M3) — protection against cherry-picking.

Every research experiment is recorded BEFORE its metrics are known. The
registry exists for one reason: to make uncontrolled experimentation and
post-hoc story-telling structurally difficult rather than merely discouraged.

    register_trial(hypothesis, config, primary_metric)   <- no metrics yet
              |
              +-- complete_trial(metrics)     -> completed, immutable
              +-- abandon_trial(reason)       -> abandoned, recorded

Two rules carry most of the weight:

- **Pre-registration.** A trial records its hypothesis, its full
  configuration and its *pre-registered primary metric* at registration
  time, before any result exists. `complete_trial` refuses metrics that do
  not contain the declared primary metric, so a trial cannot quietly be
  judged on whichever number happened to look best.
- **Immutability.** A completed trial cannot be re-completed, re-registered
  or edited. A changed configuration is a NEW trial with a new id. There is
  no code path that rewrites a recorded result.

What a trial records (the M3 field list, all validated):

    trial_id, hypothesis, feature_set_version, model_family,
    hyperparameters, label_version, horizons, training_window,
    validation_scheme, costs, seed, dataset_hash, metrics

`trial_id` is a deterministic SHA-256 over the trial's *configuration* —
hypothesis included — so the same experiment always has the same identity,
and re-running an experiment cannot masquerade as a new independent result.
That last property is what makes the multiple-testing count (X7) honest: the
registry knows how many DISTINCT experiments were actually run.

Abandoned trials are recorded, never deleted. A drawer full of quietly
discarded failures is exactly the bias this registry exists to prevent.

Pure and deterministic apart from persistence; timestamps are supplied by
the caller at the governance boundary, never read from the wall clock here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.config import (
    LABEL_HORIZON_SESSIONS,
    OUTCOME_LABEL_VERSION,
    TRIAL_REGISTRY_VERSION,
    TRIAL_STATUS_ABANDONED,
    TRIAL_STATUS_COMPLETED,
    TRIAL_STATUS_REGISTERED,
    TRIAL_STATUSES,
    TRIAL_VALIDATION_SCHEMES,
)
from core.feature_registry import MODEL_FAMILIES

TRIAL_REGISTRY_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "research_trials.jsonl"
)

_MIN_HYPOTHESIS_CHARS = 20


class TrialRegistryError(ValueError):
    """Raised when a trial record or a lifecycle transition is invalid."""


@dataclass
class Trial:
    """One recorded research experiment."""

    hypothesis: str
    feature_set_version: str
    model_family: str
    label_version: str
    horizons: list[str]
    training_window: dict[str, Any]
    validation_scheme: str
    costs: dict[str, Any]
    seed: int
    dataset_hash: str
    primary_metric: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    trial_id: str = ""
    status: str = TRIAL_STATUS_REGISTERED
    metrics: dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""
    completed_at: str = ""
    abandoned_at: str = ""
    abandoned_reason: str = ""
    parent_trial_id: str | None = None
    registry_version: str = TRIAL_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if not self.trial_id:
            self.trial_id = self.config_hash()

    def config_hash(self) -> str:
        """Deterministic identity over the trial's CONFIGURATION only.

        Metrics and timestamps are excluded on purpose: the same experiment
        must hash the same before and after it runs, so re-running it cannot
        be passed off as a new independent result.
        """
        payload = {
            "hypothesis": self.hypothesis.strip(),
            "feature_set_version": self.feature_set_version,
            "model_family": self.model_family,
            "hyperparameters": self.hyperparameters,
            "label_version": self.label_version,
            "horizons": sorted(str(h) for h in self.horizons),
            "training_window": self.training_window,
            "validation_scheme": self.validation_scheme,
            "costs": self.costs,
            "seed": self.seed,
            "dataset_hash": self.dataset_hash,
            "primary_metric": self.primary_metric,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_final(self) -> bool:
        return self.status in (TRIAL_STATUS_COMPLETED, TRIAL_STATUS_ABANDONED)


def trial_problems(trial: Trial) -> list[str]:
    """Validate a trial against the M3 field requirements."""
    problems: list[str] = []

    hypothesis = str(trial.hypothesis or "").strip()
    if not hypothesis:
        problems.append("hypothesis is required — an unstated hypothesis cannot be tested")
    elif len(hypothesis) < _MIN_HYPOTHESIS_CHARS:
        problems.append(
            f"hypothesis is {len(hypothesis)} chars; at least {_MIN_HYPOTHESIS_CHARS} "
            f"are required — it must state what is predicted and why"
        )

    if not str(trial.feature_set_version or "").strip():
        problems.append("feature_set_version is required — a trial must name its inputs")
    if trial.model_family not in MODEL_FAMILIES:
        problems.append(
            f"model_family {trial.model_family!r} is not a known model family "
            f"(known: {sorted(MODEL_FAMILIES)})"
        )
    if trial.label_version != OUTCOME_LABEL_VERSION:
        problems.append(
            f"label_version {trial.label_version!r} != {OUTCOME_LABEL_VERSION!r} — "
            f"a trial must be judged against the current label contract"
        )

    if not trial.horizons:
        problems.append("at least one horizon is required")
    for horizon in trial.horizons:
        if str(horizon) not in LABEL_HORIZON_SESSIONS:
            problems.append(
                f"horizon {horizon!r} is not a declared label horizon "
                f"(known: {sorted(LABEL_HORIZON_SESSIONS)})"
            )

    if not trial.training_window:
        problems.append("training_window is required")
    else:
        for key in ("start", "end"):
            if not str(trial.training_window.get(key) or "").strip():
                problems.append(f"training_window.{key} is required")

    if trial.validation_scheme not in TRIAL_VALIDATION_SCHEMES:
        problems.append(
            f"validation_scheme {trial.validation_scheme!r} is not recognised "
            f"(known: {sorted(TRIAL_VALIDATION_SCHEMES)}) — a trial without a "
            f"validation scheme is not a trial"
        )

    if not trial.costs:
        problems.append(
            "costs are required — a backtest without explicit cost assumptions "
            "is not evidence (Sprint V3)"
        )
    elif not str(trial.costs.get("cost_table_version") or "").strip():
        problems.append("costs.cost_table_version is required")

    if not isinstance(trial.seed, int) or isinstance(trial.seed, bool):
        problems.append(f"seed must be an int, got {type(trial.seed).__name__}")

    if not str(trial.dataset_hash or "").strip():
        problems.append(
            "dataset_hash is required — a trial must name the exact dataset it used"
        )

    if not str(trial.primary_metric or "").strip():
        problems.append(
            "primary_metric must be pre-registered — choosing the metric after "
            "seeing results is cherry-picking"
        )

    if trial.status not in TRIAL_STATUSES:
        problems.append(
            f"status {trial.status!r} is not a known status (known: {sorted(TRIAL_STATUSES)})"
        )
    if trial.status == TRIAL_STATUS_REGISTERED and trial.metrics:
        problems.append(
            "a registered trial must not carry metrics — pre-registration means "
            "the hypothesis is recorded before the result exists"
        )
    if trial.status == TRIAL_STATUS_COMPLETED:
        if not trial.metrics:
            problems.append("a completed trial must carry metrics")
        elif trial.primary_metric not in trial.metrics:
            problems.append(
                f"completed trial does not report its pre-registered primary "
                f"metric {trial.primary_metric!r} (reported: {sorted(trial.metrics)})"
            )
        if not str(trial.completed_at or "").strip():
            problems.append("a completed trial must record completed_at")
    if trial.status == TRIAL_STATUS_ABANDONED:
        if not str(trial.abandoned_at or "").strip():
            problems.append("an abandoned trial must record abandoned_at")
        if not str(trial.abandoned_reason or "").strip():
            problems.append(
                "an abandoned trial must record why — a silently dropped trial "
                "is the bias this registry exists to prevent"
            )

    if trial.parent_trial_id is not None and trial.parent_trial_id == trial.trial_id:
        problems.append("parent_trial_id cannot be the trial itself")

    if trial.trial_id and trial.trial_id != trial.config_hash():
        problems.append(
            "trial_id does not match the configuration hash — the configuration "
            "changed after registration, which is a NEW trial, not an edit"
        )
    return problems


class TrialRegistry:
    """In-memory registry of research trials. Persistence is explicit."""

    def __init__(self, trials: Iterable[Trial] | None = None) -> None:
        self._trials: dict[str, Trial] = {}
        for trial in trials or ():
            self._trials[trial.trial_id] = trial

    def register(self, trial: Trial) -> Trial:
        """Record a trial before it runs. Duplicate configurations are refused."""
        problems = trial_problems(trial)
        if problems:
            raise TrialRegistryError(
                f"invalid trial {trial.trial_id!r}: " + "; ".join(problems)
            )
        if trial.status != TRIAL_STATUS_REGISTERED:
            raise TrialRegistryError(
                f"a trial must enter the registry as {TRIAL_STATUS_REGISTERED!r}, "
                f"got {trial.status!r} — metrics are attached by complete_trial"
            )
        existing = self._trials.get(trial.trial_id)
        if existing is not None:
            raise TrialRegistryError(
                f"trial {trial.trial_id!r} is already registered (status "
                f"{existing.status!r}) — an identical configuration is the SAME "
                f"experiment, not a new one. Change the configuration or cite the "
                f"existing trial."
            )
        if trial.parent_trial_id is not None and trial.parent_trial_id not in self._trials:
            raise TrialRegistryError(
                f"parent_trial_id {trial.parent_trial_id!r} is not registered"
            )
        self._trials[trial.trial_id] = trial
        return trial

    def get(self, trial_id: str) -> Trial | None:
        return self._trials.get(str(trial_id))

    def all_trials(self) -> dict[str, Trial]:
        return dict(self._trials)

    def by_status(self, status: str) -> dict[str, Trial]:
        return {
            trial_id: trial
            for trial_id, trial in self._trials.items()
            if trial.status == status
        }

    def complete(self, trial_id: str, metrics: dict[str, Any], completed_at: str) -> Trial:
        """Attach results. Exactly once, and the primary metric must be reported."""
        trial = self._require(trial_id)
        if trial.is_final():
            raise TrialRegistryError(
                f"trial {trial_id!r} is already {trial.status!r} — a recorded result "
                f"is immutable. Re-running the experiment does not overwrite it."
            )
        if not metrics:
            raise TrialRegistryError("metrics are required to complete a trial")
        if trial.primary_metric not in metrics:
            raise TrialRegistryError(
                f"metrics do not report the pre-registered primary metric "
                f"{trial.primary_metric!r} (reported: {sorted(metrics)}) — a trial "
                f"is judged on the metric it declared, not the one that looks best"
            )
        if not str(completed_at or "").strip():
            raise TrialRegistryError("completed_at is required")

        trial.metrics = dict(metrics)
        trial.status = TRIAL_STATUS_COMPLETED
        trial.completed_at = str(completed_at)
        return trial

    def abandon(self, trial_id: str, reason: str, abandoned_at: str) -> Trial:
        """Record an abandoned trial. Never delete it."""
        trial = self._require(trial_id)
        if trial.is_final():
            raise TrialRegistryError(
                f"trial {trial_id!r} is already {trial.status!r} and is immutable"
            )
        if not str(reason or "").strip():
            raise TrialRegistryError(
                "an abandoned trial must record why — a silently dropped trial is "
                "the file-drawer bias this registry exists to prevent"
            )
        if not str(abandoned_at or "").strip():
            raise TrialRegistryError("abandoned_at is required")
        trial.status = TRIAL_STATUS_ABANDONED
        trial.abandoned_reason = str(reason)
        trial.abandoned_at = str(abandoned_at)
        return trial

    def _require(self, trial_id: str) -> Trial:
        trial = self.get(trial_id)
        if trial is None:
            raise TrialRegistryError(
                f"trial {trial_id!r} is not registered — results from an "
                f"unregistered experiment are not evidence"
            )
        return trial

    def problems(self) -> list[str]:
        problems: list[str] = []
        for trial_id, trial in sorted(self._trials.items()):
            for problem in trial_problems(trial):
                problems.append(f"{trial_id}: {problem}")
        return problems

    def multiple_testing_report(self) -> dict[str, Any]:
        """How many DISTINCT experiments were run, for X7 honesty.

        A result's significance depends on how many things were tried. This
        is the count that makes that correction possible — including the
        abandoned trials, which is exactly the number people forget.
        """
        completed = self.by_status(TRIAL_STATUS_COMPLETED)
        abandoned = self.by_status(TRIAL_STATUS_ABANDONED)
        registered = self.by_status(TRIAL_STATUS_REGISTERED)
        by_metric: dict[str, int] = {}
        for trial in completed.values():
            by_metric[trial.primary_metric] = by_metric.get(trial.primary_metric, 0) + 1
        return {
            "registry_version": TRIAL_REGISTRY_VERSION,
            "distinct_trials": len(self._trials),
            "completed": len(completed),
            "abandoned": len(abandoned),
            "still_registered": len(registered),
            "completed_by_primary_metric": dict(sorted(by_metric.items())),
        }


def persist_trial(trial: Trial, path: str | Path | None = None) -> dict[str, Any]:
    """Append a trial's current state to the append-only ledger.

    Append-only: registration and completion are separate lines, so the
    ledger shows that the hypothesis was recorded BEFORE the result. That
    ordering is the evidence of pre-registration.
    """
    store = Path(path) if path is not None else TRIAL_REGISTRY_STORE_PATH
    problems = trial_problems(trial)
    if problems:
        raise TrialRegistryError(
            f"refusing to persist invalid trial {trial.trial_id!r}: " + "; ".join(problems)
        )
    record = trial.to_dict()
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def load_trial_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read every persisted line, oldest first. Malformed lines raise."""
    store = Path(path) if path is not None else TRIAL_REGISTRY_STORE_PATH
    if not store.exists():
        return []
    records: list[dict[str, Any]] = []
    with store.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise TrialRegistryError(
                    f"{store.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return records


def load_trial_registry(path: str | Path | None = None) -> TrialRegistry:
    """Rebuild a registry from the ledger, newest state per trial winning."""
    known = {name for name in Trial.__dataclass_fields__}
    latest: dict[str, Trial] = {}
    for payload in load_trial_records(path):
        trial = Trial(**{key: value for key, value in payload.items() if key in known})
        latest[trial.trial_id] = trial
    return TrialRegistry(latest.values())
