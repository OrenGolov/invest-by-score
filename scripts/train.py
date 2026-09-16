"""Offline training entrypoint (Sprint M3, board numbering).

The board names this file specifically. It wires the existing pieces
together and adds nothing of its own:

    M2-DS dataset  ->  V2 walk-forward folds  ->  baselines  ->  run ledger

Every run is optionally pre-registered in the M3 trial registry, so an
experiment run from the command line is recorded before its metrics exist
exactly like one run from code.

Usage:
    python scripts/train.py --ticker NVDA --estimator ridge
    python scripts/train.py --ticker NVDA --all --register-trial \
        --hypothesis "Ridge on 20d returns beats the historical-mean baseline"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    OUTCOME_LABEL_VERSION,
    TRAINING_DEFAULT_SEED,
)
from core.training import (  # noqa: E402
    BASELINE_ESTIMATORS,
    persist_training_run,
    train_baseline,
)
from core.training_dataset import build_training_dataset  # noqa: E402


def _load_frame(ticker: str) -> pd.DataFrame:
    """Load a cached price frame. Training is offline by construction."""
    candidates = sorted((REPO_ROOT / "data").glob(f"{ticker.upper()}_*_1d.parquet"))
    if not candidates:
        raise SystemExit(
            f"no cached frame for {ticker!r} under data/ — training reads the "
            f"cache, never a live provider"
        )
    longest = max(candidates, key=lambda path: path.stat().st_size)
    frame = pd.read_parquet(longest)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--estimator", default="ridge", choices=BASELINE_ESTIMATORS)
    parser.add_argument("--all", action="store_true", help="train every baseline")
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=600, help="prediction times to use")
    parser.add_argument("--fold-sessions", type=int, default=80)
    parser.add_argument("--embargo-sessions", type=int, default=60)
    parser.add_argument("--holdout-sessions", type=int, default=60)
    parser.add_argument("--persist", action="store_true", help="append to the run ledger")
    parser.add_argument("--register-trial", action="store_true")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--primary-metric", default="directional_accuracy")
    args = parser.parse_args(argv)

    frame = _load_frame(args.ticker)
    # Leave the tail unused: the newest bars cannot have a matured 20d label.
    times = [
        ts.strftime("%Y-%m-%d %H:%M:%S")
        for ts in frame.index[-(args.rows + 100):-100]
    ]
    dataset = build_training_dataset({args.ticker.upper(): times}, {args.ticker.upper(): frame})
    print(f"dataset: {len(dataset)} rows, {len(dataset.feature_names)} features")
    print(f"  hash {dataset.dataset_hash}")
    if dataset.excluded:
        print(f"  excluded: {dataset.report()['exclusion_reasons']}")
    if not dataset.rows:
        raise SystemExit("dataset is empty — nothing to train on")

    estimators = list(BASELINE_ESTIMATORS) if args.all else [args.estimator]
    folds = {
        "fold_sessions": args.fold_sessions,
        "embargo_sessions": args.embargo_sessions,
        "holdout_sessions": args.holdout_sessions,
    }

    trial_id = ""
    if args.register_trial:
        if not args.hypothesis.strip():
            raise SystemExit("--register-trial requires --hypothesis")
        trial_id = _register_trial(args, dataset, estimators)
        print(f"trial registered: {trial_id}")

    print()
    print(f"{'estimator':20} {'folds':>5} {'rmse':>11} {'dir_acc':>8}  artifact")
    results = {}
    for estimator in estimators:
        run = train_baseline(dataset, estimator, seed=args.seed, **folds)
        results[estimator] = run
        metrics = run.metrics
        print(
            f"{estimator:20} {metrics['folds']:>5} {metrics['rmse']:>11.6f} "
            f"{metrics['directional_accuracy']:>8.3f}  {run.artifact_hash[:12]}"
        )
        if args.persist:
            persist_training_run(run)

    best = max(results.values(), key=lambda run: run.metrics["directional_accuracy"])
    print()
    print(
        f"best directional accuracy: {best.estimator} "
        f"({best.metrics['directional_accuracy']:.3f})"
    )
    if best.estimator in ("historical_mean", "momentum"):
        print(
            "  NOTE: a pure baseline won. No trained model has earned promotion "
            "— that is a valid result, not a failure."
        )
    return 0


def _register_trial(args, dataset, estimators: list[str]) -> str:
    """Pre-register the experiment before its metrics exist (M3-TR)."""
    from core.backtest.costs import COST_TABLE_VERSION
    from core.trial_registry import Trial, TrialRegistry, persist_trial

    trial = Trial(
        hypothesis=args.hypothesis.strip(),
        feature_set_version=dataset.feature_set_hash,
        model_family="linear" if args.estimator in ("ridge", "elastic_net") else "tree",
        label_version=OUTCOME_LABEL_VERSION,
        horizons=[dataset.target_horizon],
        training_window={
            "start": dataset.rows[0].prediction_time,
            "end": dataset.rows[-1].prediction_time,
        },
        validation_scheme="walk_forward_embargo",
        costs={"cost_table_version": COST_TABLE_VERSION},
        seed=args.seed,
        dataset_hash=dataset.dataset_hash,
        primary_metric=args.primary_metric,
        hyperparameters={"estimators": sorted(estimators)},
    )
    TrialRegistry().register(trial)
    persist_trial(trial)
    return trial.trial_id


if __name__ == "__main__":
    sys.exit(main())
