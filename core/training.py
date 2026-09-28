"""Offline training pipeline (Sprint M3, board numbering) — the only trainer.

Consumes an M2-DS dataset, splits it with the V2 walk-forward harness, trains
the baseline families, and records everything needed to reproduce the result.
Sequence and temporal models are explicitly out of scope until these
baselines survive validation.

    TrainingDataset (M2-DS)
        |
        v
    build_walk_forward_folds()   <- V2 harness: anchored folds + embargo
        |
        v
    fit baselines per fold       <- ridge / elastic_net / random_forest /
        |                           gradient_boosting, plus the pure
        v                           historical-mean and momentum baselines
    TrainingRun (metrics + artifact hash + manifest)

Binding rules, all test-enforced:

- **Splits come from the V2 harness, never sklearn.** `train_test_split`,
  `KFold` and `cross_val_score` shuffle by default and are not time-safe;
  using them on financial panel data leaks the future into training. The
  folds here are the same anchored windows with the same embargo the
  backtest engine uses, and an import-hygiene test proves no sklearn
  splitter is imported anywhere in this module.
- **The registry gates training.** Every feature named by the dataset must be
  registered and declare compatibility with the model family being trained;
  an unregistered feature aborts the run rather than training on it.
- **Rows are used in time order.** A fold's train indices are strictly
  earlier than its validation indices, separated by the embargo. The
  dataset is sorted by prediction_time before any split.
- **Determinism.** Same dataset hash + same seed + same configuration
  produce identical metrics and an identical artifact hash. The hash is
  taken over FITTED PARAMETERS rather than pickle bytes, because joblib
  output embeds library versions and is not byte-stable across environments
  — hashing it would make the reproducibility claim untestable.
- **A run without a dataset hash is refused.** A model that cannot name its
  training data is not reproducible and therefore not evidence.

Pure and deterministic apart from the fit itself, which is seeded.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.config import (
    BACKTEST_EMBARGO_SESSIONS,
    BACKTEST_FOLD_SESSIONS,
    BACKTEST_HOLDOUT_SESSIONS,
    TRAINING_ARTIFACT_HASH_BASIS,
    TRAINING_DEFAULT_SEED,
    TRAINING_PIPELINE_VERSION,
)
from core.feature_registry import (
    FeatureRegistry,
    build_default_registry,
    feature_set_hash,
    model_feature_problems,
)

TRAINING_RUN_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "training_runs.jsonl"
)

# Estimator name -> (model family for the registry gate, constructor).
# Constructors are resolved lazily so importing this module does not require
# scikit-learn until a trained baseline is actually requested.
BASELINE_ESTIMATORS: tuple[str, ...] = (
    "historical_mean",
    "momentum",
    "mean_reversion",
    "ridge",
    "elastic_net",
    "logistic",
    "random_forest",
    "gradient_boosting",
)

# The pure baselines need no third-party library and must be beaten before
# any trained model earns its place (master context M4).
_PURE_BASELINES = ("historical_mean", "momentum", "mean_reversion")

_FAMILY_BY_ESTIMATOR = {
    "historical_mean": "baseline_mean",
    "momentum": "momentum",
    "mean_reversion": "mean_reversion",
    "logistic": "logistic",
    "ridge": "linear",
    "elastic_net": "linear",
    "random_forest": "tree",
    "gradient_boosting": "boosting",
}


class TrainingError(ValueError):
    """Raised when a training request or a fitted run violates the contract."""


@dataclass
class FoldResult:
    """One fold's out-of-sample result. Validation metrics only."""

    fold_id: int
    train_rows: int
    validation_rows: int
    train_end_time: str
    validation_start_time: str
    metrics: dict[str, float] = field(default_factory=dict)
    # M6: calibration must be fitted on VALIDATION folds only, so the raw
    # out-of-sample predictions and their actuals are retained per fold.
    # Without them a calibration map could only be fitted in-sample, which
    # is exactly the mistake calibration exists to avoid.
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    # A1: WHERE the fold sat, and under WHAT CONDITIONS its observations fell.
    #
    # `train_rows`/`validation_rows` say HOW MANY, never WHERE. X8 needs
    # position to prove a fold stopped short of the sealed holdout, and a count
    # cannot express that — a fold of 120 rows could sit anywhere.
    #
    # `regimes`/`events`/`sources` are one entry PER VALIDATION OBSERVATION,
    # aligned index-for-index with `predictions` and `actuals`, so X3 can slice
    # per regime and X4 can leave one event or source out at a time.
    #
    # An empty list means NO CONTEXT WAS RECORDED, which is why X3 and X4 report
    # NOT_EVALUATED rather than inventing a single bucket. A list of the right
    # length with `None` entries means the context was looked for and absent for
    # those observations — a different and more useful fact.
    train: list[int] = field(default_factory=list)
    validation: list[int] = field(default_factory=list)
    regimes: list[str | None] = field(default_factory=list)
    events: list[str | None] = field(default_factory=list)
    sources: list[str | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def context_problems(self) -> list[str]:
        """Why this fold's recorded context cannot be trusted. Empty means clean.

        THE ALIGNMENT IS THE WHOLE CONTRACT. A context list shorter than
        `predictions` would silently misalign every observation after the gap,
        and a per-regime comparison built on a misaligned slice is worse than no
        comparison at all — it looks like evidence.
        """
        problems: list[str] = []
        width = len(self.predictions)
        for name in ("regimes", "events", "sources"):
            values = getattr(self, name)
            if values and len(values) != width:
                problems.append(
                    f"{name} has {len(values)} entries against {width} "
                    f"predictions; a misaligned slice reads as evidence while "
                    f"describing the wrong observations"
                )
        for name in ("train", "validation"):
            window = getattr(self, name)
            if window and len(window) != 2:
                problems.append(
                    f"{name} must be [start, end] absolute row indices, got "
                    f"{len(window)} values"
                )
            if len(window) == 2 and window[0] > window[1]:
                problems.append(f"{name} window {window} runs backwards")
        return problems


@dataclass
class TrainingRun:
    """A completed training run and everything needed to reproduce it."""

    estimator: str
    model_family: str
    dataset_hash: str
    feature_set_hash: str
    feature_names: list[str]
    target_horizon: str
    seed: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    folds: list[FoldResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_hash: str = ""
    artifact_hash_basis: str = TRAINING_ARTIFACT_HASH_BASIS
    pipeline_version: str = TRAINING_PIPELINE_VERSION
    environment: dict[str, str] = field(default_factory=dict)
    # A1/X8: THE SEAL, RECORDED RATHER THAN INFERRED.
    #
    # `build_walk_forward_folds` computed `holdout: [start, end]` and this class
    # discarded it, so recovering where the sealed tail fell required a second
    # file plus a guessed geometry. X8 measured that none of the four facts a
    # seal needs were on disk; these are those four.
    #
    # NOT part of `run_hash()` — the hash identifies the MODEL (data, features,
    # seed, hyperparameters, metrics), and recording where the holdout sat does
    # not change which model was fitted. Adding it would invalidate every
    # existing artifact hash for no gain in identity.
    dataset_rows: int | None = None
    geometry: dict[str, int] = field(default_factory=dict)
    holdout: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["folds"] = [fold.to_dict() for fold in self.folds]
        return payload

    def run_hash(self) -> str:
        """Deterministic identity over the run's inputs and its results."""
        payload = {
            "estimator": self.estimator,
            "dataset_hash": self.dataset_hash,
            "feature_set_hash": self.feature_set_hash,
            "target_horizon": self.target_horizon,
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifact_hash": self.artifact_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _environment() -> dict[str, str]:
    """The environment facts that can change a fitted artifact.

    Recorded so an M8 reproducibility comparison can say WHY two runs
    diverged instead of just that they did.
    """
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "absent"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn_version,
        "platform": platform.system(),
    }


def _build_estimator(name: str, seed: int):
    """Construct a seeded estimator. Imports sklearn only when needed."""
    if name in _PURE_BASELINES:
        return None
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Ridge  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise TrainingError(
            f"estimator {name!r} requires scikit-learn, which is not installed"
        ) from exc

    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, max_iter=2000)
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "elastic_net":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=5000)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=seed, n_jobs=1
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=seed
        )
    raise TrainingError(f"unknown estimator {name!r} (known: {sorted(BASELINE_ESTIMATORS)})")


def _fit_predict(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one estimator and predict the validation fold.

    Returns predictions plus the fitted parameters used for artifact hashing.
    """
    if name == "historical_mean":
        # The simplest honest baseline: predict the training mean.
        mean = float(np.mean(train_y)) if train_y.size else 0.0
        return np.full(len(validation_x), mean), {"mean": round(mean, 10)}

    if name == "momentum":
        # Sign-following on the first feature column, scaled by training
        # magnitude. Deliberately naive — it exists to be beaten.
        magnitude = float(np.mean(np.abs(train_y))) if train_y.size else 0.0
        direction = np.sign(validation_x[:, 0]) if validation_x.size else np.zeros(0)
        return direction * magnitude, {"magnitude": round(magnitude, 10)}

    if name == "mean_reversion":
        # The opposing hypothesis to momentum: fade the recent move. Mirror
        # of the momentum baseline so the two are directly comparable — if
        # momentum has an edge, this must lose by the same margin.
        magnitude = float(np.mean(np.abs(train_y))) if train_y.size else 0.0
        direction = -np.sign(validation_x[:, 0]) if validation_x.size else np.zeros(0)
        return direction * magnitude, {"magnitude": round(magnitude, 10)}

    estimator = _build_estimator(name, seed)
    if name == "logistic":
        # Direction is the natural target for a classifier. A single-class
        # training window has no decision to make, so it falls back to that
        # class rather than raising — recorded in the parameters.
        labels = np.sign(train_y)
        magnitude = float(np.mean(np.abs(train_y))) if train_y.size else 0.0
        if np.unique(labels).size < 2:
            only = float(labels[0]) if labels.size else 0.0
            return (
                np.full(len(validation_x), only * magnitude),
                {"degenerate_single_class": only, "magnitude": round(magnitude, 10)},
            )
        estimator.fit(train_x, labels)
        predicted_direction = np.asarray(estimator.predict(validation_x), dtype=float)
        return (
            predicted_direction * magnitude,
            {
                "coef": [round(float(v), 10) for v in np.ravel(estimator.coef_)],
                "intercept": round(float(np.ravel(estimator.intercept_)[0]), 10),
                "magnitude": round(magnitude, 10),
            },
        )


    estimator.fit(train_x, train_y)
    predictions = np.asarray(estimator.predict(validation_x), dtype=float)

    parameters: dict[str, Any] = {}
    if hasattr(estimator, "coef_"):
        parameters["coef"] = [round(float(value), 10) for value in np.ravel(estimator.coef_)]
    if hasattr(estimator, "intercept_"):
        parameters["intercept"] = round(float(np.ravel(estimator.intercept_)[0]), 10)
    if hasattr(estimator, "feature_importances_"):
        parameters["feature_importances"] = [
            round(float(value), 10) for value in estimator.feature_importances_
        ]
    return predictions, parameters


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Regression + directional metrics, computed directly.

    No sklearn.metrics import: these are three lines of numpy each, and
    keeping them local means the module's only sklearn surface is the
    estimators themselves.
    """
    if actual.size == 0:
        return {}
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    directional = float(np.mean(np.sign(predicted) == np.sign(actual)))
    return {
        "mae": round(mae, 8),
        "rmse": round(rmse, 8),
        "directional_accuracy": round(directional, 8),
        "observations": int(actual.size),
    }


def training_request_problems(
    dataset,
    estimator: str,
    registry: FeatureRegistry | None = None,
) -> list[str]:
    """Validate a training request before any fitting happens."""
    active = registry if registry is not None else build_default_registry()
    problems: list[str] = []

    if estimator not in BASELINE_ESTIMATORS:
        problems.append(
            f"estimator {estimator!r} is not a baseline "
            f"(known: {sorted(BASELINE_ESTIMATORS)})"
        )
        return problems

    if not getattr(dataset, "rows", None):
        problems.append("dataset carries no rows — there is nothing to train on")
    if not str(getattr(dataset, "dataset_hash", "") or "").strip():
        problems.append(
            "dataset carries no dataset_hash — a model that cannot name its "
            "training data is not reproducible"
        )

    feature_names = list(getattr(dataset, "feature_names", []) or [])
    if not feature_names:
        problems.append("dataset declares no features")
    else:
        # M1 gate: unregistered features cannot enter a production model, and
        # training is where a model's inputs are decided.
        problems.extend(
            model_feature_problems(feature_names, active, _FAMILY_BY_ESTIMATOR[estimator])
        )
    return problems


def train_baseline(
    dataset,
    estimator: str = "ridge",
    seed: int = TRAINING_DEFAULT_SEED,
    fold_sessions: int | None = None,
    embargo_sessions: int | None = None,
    holdout_sessions: int | None = None,
    registry: FeatureRegistry | None = None,
) -> TrainingRun:
    """Train one baseline on an M2-DS dataset using V2 walk-forward folds.

    Every fold trains strictly before its validation window, separated by the
    embargo. Metrics are out-of-sample by construction: a model never scores
    itself on rows it trained on.
    """
    # Imported here so this module does not pull the backtest package (and
    # its pandas/provider surface) at import time.
    from core.backtest.engine import build_walk_forward_folds

    active = registry if registry is not None else build_default_registry()
    problems = training_request_problems(dataset, estimator, active)
    if problems:
        raise TrainingError(
            f"invalid training request for {estimator!r}: " + "; ".join(problems)
        )

    feature_names = sorted(dataset.feature_names)
    rows = sorted(dataset.rows, key=lambda row: (row.prediction_time, row.ticker))
    matrix = np.array(
        [[float(row.features[name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    targets = np.array([float(row.forward_return) for row in rows], dtype=float)

    try:
        geometry = build_walk_forward_folds(
            len(rows), fold_sessions, embargo_sessions, holdout_sessions
        )
    except ValueError as exc:
        # Thin history is an ordinary, expected condition — recent listings
        # and short cached frames hit it constantly. Raise it as a
        # TrainingError so a caller's `except TrainingError` sees it, and
        # name the shortfall rather than leaving a bare geometry message.
        raise TrainingError(
            f"{estimator!r} cannot be trained on {len(rows)} rows: {exc}. "
            f"A walk-forward fit needs at least "
            f"{2 * (fold_sessions or BACKTEST_FOLD_SESSIONS) + (embargo_sessions or BACKTEST_EMBARGO_SESSIONS) + (holdout_sessions or BACKTEST_HOLDOUT_SESSIONS)} "
            f"rows at this geometry — a recent listing or a short cached frame "
            f"will not have them."
        ) from exc

    fold_results: list[FoldResult] = []
    parameters_by_fold: list[dict[str, Any]] = []
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for fold in geometry["folds"]:
        train_start, train_end = fold["train"]
        validation_start, validation_end = fold["validation"]
        train_x = matrix[train_start:train_end + 1]
        train_y = targets[train_start:train_end + 1]
        validation_x = matrix[validation_start:validation_end + 1]
        validation_y = targets[validation_start:validation_end + 1]
        if train_x.size == 0 or validation_x.size == 0:
            continue

        predicted, parameters = _fit_predict(estimator, train_x, train_y, validation_x, seed)
        parameters_by_fold.append(parameters)
        all_actual.append(validation_y)
        all_predicted.append(predicted)
        # A1: the validation rows THEMSELVES, so the recorded context is aligned
        # with the predictions by construction rather than by a later join.
        validation_rows_slice = rows[validation_start:validation_end + 1]
        fold_results.append(FoldResult(
            fold_id=int(fold["fold_id"]),
            train_rows=int(train_x.shape[0]),
            validation_rows=int(validation_x.shape[0]),
            train_end_time=rows[train_end].prediction_time,
            validation_start_time=rows[validation_start].prediction_time,
            metrics=_metrics(validation_y, predicted),
            predictions=[round(float(value), 10) for value in predicted],
            actuals=[round(float(value), 10) for value in validation_y],
            train=[int(train_start), int(train_end)],
            validation=[int(validation_start), int(validation_end)],
            regimes=[getattr(row, "regime", None) for row in validation_rows_slice],
            events=[getattr(row, "event_id", None) for row in validation_rows_slice],
            sources=[getattr(row, "source_id", None) for row in validation_rows_slice],
        ))

    if not fold_results:
        raise TrainingError("no fold produced a usable train/validation split")

    pooled_actual = np.concatenate(all_actual)
    pooled_predicted = np.concatenate(all_predicted)
    pooled = _metrics(pooled_actual, pooled_predicted)
    pooled["folds"] = len(fold_results)

    # Hash the FITTED PARAMETERS, not pickle bytes: joblib output embeds
    # library versions and is not byte-stable across environments, which
    # would make the M8 reproducibility guarantee untestable in CI.
    # M8: the artifact hash identifies the FITTED MODEL, so it covers the
    # parameters, the estimator and the feature surface — but NOT the seed.
    # A seed that changed nothing about the fitted parameters did not produce
    # a different artifact, and hashing it in would report a false difference
    # for the deterministic estimators (historical_mean, ridge). The seed is
    # part of the RUN identity instead, via run_hash().
    artifact_payload = {
        "estimator": estimator,
        "feature_names": feature_names,
        "folds": parameters_by_fold,
    }
    artifact_hash = hashlib.sha256(
        json.dumps(artifact_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    return TrainingRun(
        estimator=estimator,
        model_family=_FAMILY_BY_ESTIMATOR[estimator],
        dataset_hash=dataset.dataset_hash,
        feature_set_hash=feature_set_hash(feature_names, active),
        feature_names=feature_names,
        target_horizon=dataset.target_horizon,
        seed=seed,
        hyperparameters=_hyperparameters(estimator, seed),
        folds=fold_results,
        metrics=pooled,
        artifact_hash=artifact_hash,
        environment=_environment(),
        # A1/X8: the seal, recorded so it need never be guessed again.
        dataset_rows=len(rows),
        geometry={
            "fold_sessions": int(geometry["fold_sessions"]),
            "embargo_sessions": int(geometry["embargo_sessions"]),
            "holdout_sessions": int(geometry["holdout_sessions"]),
        },
        holdout=[int(geometry["holdout"][0]), int(geometry["holdout"][1])],
    )


def _hyperparameters(estimator: str, seed: int) -> dict[str, Any]:
    """The hyperparameters a run used, recorded for the trial registry."""
    if estimator in _PURE_BASELINES:
        return {"seed": seed}
    built = _build_estimator(estimator, seed)
    return {
        key: value
        for key, value in built.get_params().items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def train_baseline_suite(
    dataset,
    estimators: Iterable[str] | None = None,
    seed: int = TRAINING_DEFAULT_SEED,
    **fold_kwargs,
) -> dict[str, TrainingRun]:
    """Train several baselines on one dataset, for like-for-like comparison."""
    selected = list(estimators) if estimators is not None else list(BASELINE_ESTIMATORS)
    return {
        name: train_baseline(dataset, name, seed, **fold_kwargs)
        for name in selected
    }


def persist_training_run(run: TrainingRun, path: str | Path | None = None) -> dict[str, Any]:
    """Append a run to the ledger, idempotent per run hash."""
    store = Path(path) if path is not None else TRAINING_RUN_STORE_PATH
    if not run.artifact_hash:
        raise TrainingError("run carries no artifact_hash — refusing to persist")
    run_hash = run.run_hash()
    for existing in load_training_runs(store):
        if existing.get("run_hash") == run_hash:
            return existing
    record = {**run.to_dict(), "run_hash": run_hash}
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def load_training_runs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read persisted runs. Malformed lines raise (integrity is loud)."""
    store = Path(path) if path is not None else TRAINING_RUN_STORE_PATH
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
                raise TrainingError(
                    f"{store.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return records
