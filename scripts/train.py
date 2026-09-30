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
    LABEL_HORIZON_SESSIONS,
    TEMPORAL_HORIZONS,
    TRAINING_DEFAULT_SEED,
)
from core.temporal_robustness import evaluate_temporal_robustness  # noqa: E402
from core.training import (  # noqa: E402
    BASELINE_ESTIMATORS,
    horizon_metrics,
    train_across_horizons,
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


# A2: derived from the label horizons, so a horizon change cannot strand the
# defaults again. One fold plus a sealed tail needs 2*fold + embargo + holdout
# rows; `--rows` is set above that with room for a second fold.
_MAX_HORIZON = max(LABEL_HORIZON_SESSIONS.values())
# 1,464 rows is the SMALLEST count that yields TWO walk-forward folds AND a
# holdout embargoed against the 252-session horizon at this geometry. Below it
# the gap to the tail is a remainder short of the horizon and X8 reports
# NOT_EVALUATED; at 1,164 one fold seals, and 1,464 buys the second fold.
# MEASURED by sweeping the fold loop, not guessed.
_DEFAULT_ROWS = 1464


def _train_horizons(args, frame, folds) -> int:
    """A3: one run per X6 horizon, then the temporal-robustness verdict.

    ONE DATASET PER HORIZON. The label decides which rows have a MATURED
    outcome, so a 120d horizon must drop 120 more sessions of tail than a 1d
    one. Sharing rows would either reuse an unmatured label or silently trim the
    short horizons to the longest one's usable window.
    """
    ticker = args.ticker.upper()
    datasets = {}
    print(f"building {len(TEMPORAL_HORIZONS)} datasets, one per horizon")
    for horizon in TEMPORAL_HORIZONS:
        sessions = LABEL_HORIZON_SESSIONS[horizon]
        usable = frame.iloc[: -(sessions + 5)]
        times = [
            stamp.strftime("%Y-%m-%d %H:%M:%S")
            for stamp in usable.index[-args.rows:]
        ]
        dataset = build_training_dataset({ticker: times}, {ticker: frame}, target_horizon=horizon)
        datasets[horizon] = dataset
        print(f"  {horizon:5s} {len(dataset):5d} rows  excluded {len(dataset.excluded):3d}")

    outcome = train_across_horizons(
        datasets, estimator=args.estimator, seed=args.seed, **folds
    )
    for horizon, reason in sorted(outcome["failed"].items()):
        print(f"  FAILED {horizon}: {reason[:120]}")

    metrics = horizon_metrics(outcome["runs"])
    print()
    print(f"{'horizon':10} {'dir_acc':>8} {'observations':>13}")
    for horizon in TEMPORAL_HORIZONS:
        entry = metrics.get(horizon)
        if entry is None:
            print(f"{horizon:10} {'MISSING':>8} {'—':>13}")
        else:
            print(
                f"{horizon:10} {entry['directional_accuracy']:>8.3f} "
                f"{entry['observations']:>13d}"
            )

    if args.persist:
        for run in outcome["runs"].values():
            persist_training_run(run, hypothesis=args.hypothesis.strip())

    report = evaluate_temporal_robustness(metrics)
    print()
    print(f"X6 temporal robustness: {report['verdict']}")
    if report.get("reason"):
        print(f"  {report['reason']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--estimator", default="ridge", choices=BASELINE_ESTIMATORS)
    parser.add_argument("--all", action="store_true", help="train every baseline")
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULT_SEED)
    # A2: THE DEFAULTS ARE DERIVED FROM THE HORIZON, NOT HARDCODED.
    #
    # These were fold=80 / embargo=60 / holdout=60, set when the longest label
    # horizon was 60 sessions. F2 raised it to 252 and
    # `build_walk_forward_folds` requires `embargo >= max horizon`, so the
    # script HAS BEEN UNABLE TO RUN WITH ITS OWN DEFAULTS since 2026-09-19.
    #
    # Deriving them means the next horizon change cannot strand them again.
    # `fold > horizon` is required for a sealed holdout, not merely preferred:
    # A1 measured that the gap the walk-forward loop leaves before the tail is a
    # remainder bounded by `fold_sessions - 4`, so a fold at or below the horizon
    # can never embargo the holdout.
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS, help="prediction times to use")
    parser.add_argument("--fold-sessions", type=int, default=_MAX_HORIZON + 48)
    parser.add_argument("--embargo-sessions", type=int, default=_MAX_HORIZON)
    parser.add_argument("--holdout-sessions", type=int, default=60)
    parser.add_argument(
        "--horizons",
        action="store_true",
        help=(
            "train one run per X6 horizon (1d/5d/20d/60d/120d) and report "
            "temporal robustness; each horizon gets its OWN dataset, because a "
            "horizon's label decides which rows have a matured outcome"
        ),
    )
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

    if args.horizons:
        return _train_horizons(args, frame, folds)

    # A5: registration is no longer opt-in, and no longer one trial for the whole
    # batch. `persist_training_run` registers ONE TRIAL PER ESTIMATOR, because
    # `trial_id` hashes the configuration and passing every estimator as one
    # hyperparameter collapsed eight experiments into one id. --hypothesis is
    # still honoured; without it a generated one naming the estimator and target
    # is recorded.
    if args.register_trial and not args.hypothesis.strip():
        raise SystemExit("--register-trial requires --hypothesis")

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
            persist_training_run(run, hypothesis=args.hypothesis.strip())

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


# `_register_trial` REMOVED in A5. It registered ONE trial for the whole batch by
# passing every estimator as a single `hyperparameters` entry — and `trial_id` is
# a hash over the configuration, so eight experiments collapsed into one id. That
# is why the registry held 1 distinct trial against 8 trained estimators.
# `persist_training_run` now registers one trial per estimator instead.


if __name__ == "__main__":
    sys.exit(main())
