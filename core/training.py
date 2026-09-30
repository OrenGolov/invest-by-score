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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from core.config import (
    LABEL_HORIZON_SESSIONS,
    OUTCOME_LABEL_VERSION,
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


def rows_per_date(rows: Sequence[Any]) -> float:
    """Average rows per distinct prediction time. 1.0 for a single-ticker series.

    THE CONVERSION FACTOR between the geometry (rows) and the horizon (sessions).
    A panel with 14 tickers packs 14 rows into each date, so a row count means
    one fourteenth of the calendar distance it appears to.
    """
    if not rows:
        return 1.0
    dates = {getattr(row, "prediction_time", None) for row in rows}
    dates.discard(None)
    if not dates:
        return 1.0
    return len(rows) / len(dates)


def required_embargo_rows(rows: Sequence[Any], horizon: str) -> int:
    """The embargo, IN ROWS, that covers `horizon` for this dataset's density.

    MEASURED, the scaling is stark and is the reason the default cannot simply be
    raised once:

        tickers   20d horizon   60d        252d
              1        20 rows    60       252
             14       280 rows   840     3,528
             50     1,000 rows 3,000    12,600

    A panel therefore needs an embargo proportional to BOTH the horizon and the
    universe size, which is why this is computed rather than configured.
    """
    sessions = LABEL_HORIZON_SESSIONS.get(horizon)
    if sessions is None:
        raise TrainingError(
            f"unknown label horizon {horizon!r}; its embargo cannot be computed"
        )
    density = rows_per_date(rows)
    return int(-(-sessions * density // 1))  # ceil


def minimum_rows_for(rows: Sequence[Any], horizon: str, fold_rows: int, holdout_rows: int) -> int:
    """Rows needed for one fold plus a sealed tail at this density and horizon."""
    return 2 * int(fold_rows) + required_embargo_rows(rows, horizon) + int(holdout_rows)


def embargo_problems(
    rows: Sequence[Any],
    folds: Sequence[Mapping[str, Any]],
    horizon: str,
) -> list[str]:
    """Whether each fold's train/validation separation covers the label horizon.

    **THE GEOMETRY COUNTS ROWS; THE HORIZON IS IN SESSIONS.** A panel dataset has
    one row per (ticker, prediction_time), so an embargo of N rows spans only
    N/tickers distinct dates. MEASURED, with 14 tickers a 252-ROW embargo is 18
    DATES — short of even the 20d horizon.

    This checks the separation that actually matters: the CALENDAR distance
    between the last training observation and the first validation observation,
    against the horizon's session count. A label computed from prices inside the
    training window is leakage however the geometry was specified.
    """
    # pandas imported lazily, as elsewhere in this module, so importing
    # core.training does not pull the pandas/provider surface.
    import pandas as pd

    problems: list[str] = []
    needed = LABEL_HORIZON_SESSIONS.get(horizon)
    if needed is None:
        return [f"unknown label horizon {horizon!r}; its separation cannot be checked"]
    for fold in folds or []:
        train_end = fold.get("train", [None, None])[1]
        validation_start = fold.get("validation", [None, None])[0]
        if train_end is None or validation_start is None:
            continue
        if not (0 <= train_end < len(rows)) or not (0 <= validation_start < len(rows)):
            continue
        try:
            end = pd.Timestamp(rows[train_end].prediction_time)
            start = pd.Timestamp(rows[validation_start].prediction_time)
        except Exception:
            # ABSORBS: a prediction_time that will not parse as a timestamp.
            # Reported as a PROBLEM rather than skipped, because an unverifiable
            # embargo is exactly the condition A2 added this check to catch — a
            # silent skip would restore the label leak it closed.
            problems.append(
                f"fold {fold.get('fold_id')}: prediction times are not timestamps, "
                f"so the embargo cannot be verified"
            )
            continue
        # Sessions are ~252/year, so a session is ~1.45 calendar days. Using
        # CALENDAR days with that conversion is conservative in the right
        # direction: it demands at least as much separation as the horizon.
        separation_days = (start - end).days
        needed_days = int(needed * 365.0 / 252.0)
        if separation_days < needed_days:
            problems.append(
                f"fold {fold.get('fold_id')}: {separation_days} calendar days "
                f"separate the last training row from the first validation row, "
                f"against the {needed} sessions (~{needed_days} days) the "
                f"{horizon} label needs. The geometry's embargo is counted in "
                f"ROWS, and a panel dataset packs many tickers into one date"
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

    # A2: THE EMBARGO IS VERIFIED IN DATES BEFORE ANY FOLD IS FITTED.
    #
    # The geometry's embargo is counted in ROWS while the horizon is in SESSIONS,
    # and a panel dataset packs many tickers into one date. Fitting first and
    # checking later would produce metrics from leaked labels and then discard
    # them, which is slower and invites someone to read the numbers anyway.
    leakage = embargo_problems(rows, geometry["folds"], dataset.target_horizon)
    if leakage:
        raise TrainingError(
            "the walk-forward embargo does not cover the label horizon in "
            "calendar terms: " + "; ".join(leakage)
        )

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


def trial_for_run(
    run: Any,
    *,
    hypothesis: str = "",
    primary_metric: str = "directional_accuracy",
    validation_scheme: str = "walk_forward_embargo",
) -> Any:
    """The trial that describes ONE run, for the M3 registry.

    A5. **ONE TRIAL PER ESTIMATOR.** MEASURED, the registry held 1 distinct trial
    against 8 trained estimators because the registering code passed every
    estimator as a single `hyperparameters` entry — and `trial_id` is a hash over
    the configuration, so they collapsed. M3 exists to count the number of things
    tried; a registry that merges eight into one cannot do that, and X7's
    correction then understates the search by 8x.

    The trial is derived FROM THE RUN, so its dataset hash, feature set, seed and
    horizons cannot disagree with what was actually fitted.
    """
    from core.backtest.costs import COST_TABLE_VERSION
    from core.trial_registry import Trial

    folds = run.folds if hasattr(run, "folds") else (run or {}).get("folds", [])
    first = folds[0] if folds else None
    last = folds[-1] if folds else None

    def stamp(fold, key):
        if fold is None:
            return ""
        return getattr(fold, key, None) or (fold.get(key) if isinstance(fold, dict) else "") or ""

    # M3 requires a hypothesis of at least 20 characters stating what is
    # predicted and why. Defaulting HERE rather than at the call site means an
    # empty string cannot reach the registry from any caller.
    stated = str(hypothesis or "").strip() or (
        f"Baseline {run.estimator} predicts {run.target_horizon} forward returns "
        f"from the registered feature set; registered automatically when the run "
        f"was persisted so the family count matches the search"
    )
    return Trial(
        hypothesis=stated,
        feature_set_version=run.feature_set_hash,
        model_family=run.model_family,
        label_version=OUTCOME_LABEL_VERSION,
        horizons=[run.target_horizon],
        training_window={
            "start": stamp(first, "train_end_time"),
            "end": stamp(last, "validation_start_time"),
        },
        validation_scheme=validation_scheme,
        costs={"cost_table_version": COST_TABLE_VERSION},
        seed=run.seed,
        dataset_hash=run.dataset_hash,
        primary_metric=primary_metric,
        # THE ESTIMATOR IS PART OF THE CONFIGURATION, so two estimators on the
        # same data are two trials rather than one.
        hyperparameters={
            "estimator": run.estimator,
            **(run.hyperparameters or {}),
        },
        # NO METRICS. A REGISTERED trial must not carry them: registration
        # precedes the result, and the append-only ledger showing registration
        # before completion IS the evidence of pre-registration. M3 refuses a
        # registered trial with metrics, and that refusal is the mechanism.
        # Metrics go on the completion line, via `complete_trial`.
    )


def horizon_metrics(runs_by_horizon: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Pooled out-of-sample metrics per horizon, in the shape X6 consumes.

    A3. X6 asks whether an edge holds ACROSS horizons, and reported
    HORIZONS_MISSING because every run in the ledger trained the single `20d`
    target. It needs `{horizon: {"directional_accuracy": x, "observations": n}}`.

    Directional accuracy is horizon-agnostic — computed from the SIGN of the
    forward return, not from `label_up`, which the label builder only emits for
    `20d`. So a horizon needs no special label support to be compared.

    A horizon whose run FAILED is omitted rather than recorded as zero: X6
    distinguishes MISSING from "showed no edge", and a 0.0 accuracy would claim
    the second where the truth is the first.
    """
    metrics: dict[str, dict[str, Any]] = {}
    for horizon, run in (runs_by_horizon or {}).items():
        if run is None:
            continue
        pooled = run.metrics if hasattr(run, "metrics") else (run or {}).get("metrics")
        if not pooled:
            continue
        accuracy = pooled.get("directional_accuracy")
        observations = pooled.get("observations")
        if accuracy is None or observations is None:
            continue
        metrics[str(horizon)] = {
            "directional_accuracy": float(accuracy),
            "observations": int(observations),
        }
    return metrics


def train_across_horizons(
    datasets_by_horizon: Mapping[str, Any],
    estimator: str = "ridge",
    seed: int = TRAINING_DEFAULT_SEED,
    *,
    fold_sessions: int | None = None,
    embargo_sessions: int | None = None,
    holdout_sessions: int | None = None,
    registry: FeatureRegistry | None = None,
) -> dict[str, Any]:
    """Train one run per horizon on that horizon's own dataset.

    ONE DATASET PER HORIZON, not one dataset scored several ways: the label — and
    therefore which rows have a MATURED outcome — differs per horizon. A 252d
    label needs 252 more sessions of future than a 1d label, so sharing rows
    across horizons would either leak (reusing an unmatured label) or silently
    drop the longest horizon's newest rows from the others.

    Returns `{horizon: TrainingRun}` for the horizons that trained, and records
    the failures separately under `"failed"`. A horizon that could not train is
    NOT the same as one that trained and showed nothing.
    """
    trained: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for horizon in sorted(datasets_by_horizon or {}):
        dataset = datasets_by_horizon[horizon]
        if dataset is None:
            failed[horizon] = "no dataset was built for this horizon"
            continue
        try:
            trained[horizon] = train_baseline(
                dataset,
                estimator=estimator,
                seed=seed,
                fold_sessions=fold_sessions,
                embargo_sessions=embargo_sessions,
                holdout_sessions=holdout_sessions,
                registry=registry,
            )
        except TrainingError as error:
            # An expected, ordinary outcome: a long horizon may not fit the
            # available history at this geometry. Recording WHY keeps that
            # distinguishable from "trained and found no edge".
            failed[horizon] = str(error)
    return {"runs": trained, "failed": failed}


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


def persist_training_run(
    run: TrainingRun,
    path: str | Path | None = None,
    *,
    hypothesis: str = "",
    register_trial: bool = True,
) -> dict[str, Any]:
    """Append a run to the ledger, idempotent per run hash.

    A5: REGISTRATION IS NOT OPTIONAL. MEASURED, the registry held 1 distinct trial
    against 8 trained estimators because registering was opt-in behind a flag —
    and the runs that skipped it are exactly the ones M3 was built to see. A
    persisted run now registers its trial by default and records the `trial_id`,
    so the link is auditable from either side.

    `register_trial=False` exists for the tests that persist a run in isolation;
    it is not a production path.
    """
    store = Path(path) if path is not None else TRAINING_RUN_STORE_PATH
    if not run.artifact_hash:
        raise TrainingError("run carries no artifact_hash — refusing to persist")
    run_hash = run.run_hash()
    for existing in load_training_runs(store):
        if existing.get("run_hash") == run_hash:
            return existing

    trial_id = ""
    if register_trial:
        # THE TRIAL STORE FOLLOWS THE RUN STORE. A caller persisting to a
        # temporary run ledger must not have its trials land in the tracked one:
        # the test suite does exactly that, and a first version of this appended
        # synthetic trials to `data/research_trials.jsonl` every time the suite
        # ran.
        trial_path = None if path is None else store.with_name("research_trials.jsonl")
        trial_id = _register_run_trial(run, hypothesis, trial_path)

    record = {**run.to_dict(), "run_hash": run_hash, "trial_id": trial_id}
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def _register_run_trial(
    run: TrainingRun, hypothesis: str, path: Path | None = None
) -> str:
    """Register this run's trial and return its id. Idempotent.

    An IDENTICAL configuration is the SAME experiment — `trial_id` is a hash over
    the configuration — so re-persisting a run must not register a second trial.
    M3 already refuses to re-register an existing id; that refusal is the
    idempotence, and it is caught rather than raised so persisting stays safe to
    repeat.
    """
    from core.trial_registry import TrialRegistryError, persist_trial

    trial = trial_for_run(
        run,
        hypothesis=hypothesis,
    )
    try:
        persist_trial(trial) if path is None else persist_trial(trial, path)
    except TrialRegistryError as error:
        # ALREADY REGISTERED is the idempotent case and is fine: an identical
        # configuration is the same experiment. Anything else is a genuine
        # contract breach and must be loud — a first version caught everything
        # here and silently wrote no trial at all, which is precisely the
        # unregistered-experiment problem A5 exists to close.
        if "already" not in str(error).lower():
            raise
    return trial.trial_id


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
