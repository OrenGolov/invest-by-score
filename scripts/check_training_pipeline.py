"""CI drift gate for the M3 offline training pipeline.

Proves, on every push, that the board's M3 acceptance criteria hold:

1. two runs with identical inputs produce identical metrics AND artifact
   hashes (determinism);
2. a feature without a registry entry aborts training;
3. splits are time-safe — every fold trains strictly before it validates,
   separated by at least the embargo, and no sklearn splitter is imported;
4. seeds and the environment are recorded;
5. the guards are non-vacuous — each refusal is exercised and must fire.

Kept deliberately small (one cheap estimator, few rows) so CI stays fast;
the full suite covers every baseline.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import BACKTEST_EMBARGO_SESSIONS  # noqa: E402

from core.config import TRAINING_DEFAULT_SEED  # noqa: E402
from core.training import (  # noqa: E402
    TrainingError,
    train_baseline,
    training_request_problems,
)
from core.training_dataset import build_training_dataset  # noqa: E402

FIXTURE = REPO_ROOT / "data" / "NVDA_5y_1d.parquet"
# Derived: the embargo must cover the longest label horizon. A fixed 60
# stopped being legal the moment F2 declared a 252d horizon, and the row
# slice widened with it (a fit needs 2*fold + embargo + holdout rows).
FOLDS = {
    "fold_sessions": 60,
    "embargo_sessions": BACKTEST_EMBARGO_SESSIONS,
    "holdout_sessions": 60,
}


def main() -> int:
    if not FIXTURE.exists():
        print(f"M3 training-pipeline gate SKIPPED: fixture {FIXTURE.name} not present")
        return 0

    failures: list[str] = []

    # 3a. Import hygiene — checked before anything is trained, since a
    # shuffling splitter would invalidate every result below.
    source = (REPO_ROOT / "core" / "training.py").read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
    for banned in (
        "train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit",
        "cross_val_score", "cross_validate", "TimeSeriesSplit",
    ):
        if any(banned in name for name in imported):
            failures.append(
                f"core/training.py imports {banned} — sklearn splitters shuffle "
                f"by default and leak the future into training"
            )

    frame = pd.read_parquet(FIXTURE)
    frame.index = pd.DatetimeIndex(frame.index)
    times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[-900:-200]]
    dataset = build_training_dataset({"NVDA": times}, {"NVDA": frame})
    if not dataset.rows:
        failures.append("dataset builder produced no rows for the gate")
        _report(failures)
        return 1

    # 1. Determinism.
    first = train_baseline(dataset, "ridge", **FOLDS)
    second = train_baseline(dataset, "ridge", **FOLDS)
    if first.artifact_hash != second.artifact_hash:
        failures.append(
            f"artifact hash is not deterministic:\n"
            f"    first  {first.artifact_hash}\n"
            f"    second {second.artifact_hash}"
        )
    if first.metrics != second.metrics:
        failures.append(f"metrics differ between identical runs: {first.metrics} vs {second.metrics}")
    if first.run_hash() != second.run_hash():
        failures.append("run hash is not deterministic")

    # 2. The registry gate must abort training.
    original = list(dataset.feature_names)
    dataset.feature_names = original + ["definitely_not_a_registered_feature"]
    try:
        train_baseline(dataset, "ridge", **FOLDS)
        failures.append("an unregistered feature did NOT abort training")
    except TrainingError:
        pass
    except Exception as exc:
        # The registry gate must reject BEFORE the matrix is built. A
        # KeyError here means training started on an unregistered feature and
        # only failed by accident, which is not the guarantee.
        failures.append(
            f"an unregistered feature was not rejected by the registry gate; "
            f"training failed incidentally with {type(exc).__name__}: {exc}"
        )
    finally:
        dataset.feature_names = original

    if training_request_problems(dataset, "ridge"):
        failures.append("a valid training request was wrongly rejected")
    if not training_request_problems(dataset, "not_an_estimator"):
        failures.append("an unknown estimator was NOT rejected")

    # 3b. Time-safe folds.
    rows = sorted(dataset.rows, key=lambda row: (row.prediction_time, row.ticker))
    stamps = [row.prediction_time for row in rows]
    for fold in first.folds:
        if fold.train_end_time >= fold.validation_start_time:
            failures.append(
                f"fold {fold.fold_id} validates at {fold.validation_start_time} "
                f"but trains through {fold.train_end_time} — not time-safe"
            )
            continue
        gap = stamps.index(fold.validation_start_time) - stamps.index(fold.train_end_time)
        if gap < FOLDS["embargo_sessions"]:
            failures.append(
                f"fold {fold.fold_id} has a {gap}-session gap, below the "
                f"{FOLDS['embargo_sessions']}-session embargo"
            )
    if len(first.folds) < 2:
        failures.append(f"expected multiple folds, got {len(first.folds)}")

    expected_observations = sum(fold.validation_rows for fold in first.folds)
    if first.metrics.get("observations") != expected_observations:
        failures.append(
            "pooled metrics do not match the validation rows — metrics may "
            "include in-sample observations"
        )

    # 4. Seed and environment recorded.
    if first.seed != TRAINING_DEFAULT_SEED:
        failures.append(f"run recorded seed {first.seed}, expected {TRAINING_DEFAULT_SEED}")
    for key in ("python", "numpy", "sklearn", "platform"):
        if not first.environment.get(key):
            failures.append(f"environment.{key} was not recorded")
    if first.dataset_hash != dataset.dataset_hash:
        failures.append("run does not carry the dataset hash it trained on")

    if failures:
        _report(failures)
        return 1

    print("M3 training-pipeline gate OK:")
    print(
        f"  {len(dataset.rows)} rows, {len(first.folds)} walk-forward folds, "
        f"embargo {FOLDS['embargo_sessions']} sessions"
    )
    print(f"  artifact hash deterministic ({first.artifact_hash[:16]}...)")
    print("  unregistered feature aborts training; no sklearn splitter imported.")
    print("  seed and environment recorded for M8 reproducibility.")
    return 0


def _report(failures: list[str]) -> None:
    print("M3 training-pipeline gate FAILED:")
    for failure in failures:
        print(f"  - {failure}")


if __name__ == "__main__":
    sys.exit(main())
