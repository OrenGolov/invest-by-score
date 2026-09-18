"""Establish the measured baseline C6 requires (M2 -> M4 -> M3, on real data).

C6 says "evaluate only after baseline validation", and the master context is
blunt about it: do not build a complex neural network before proving simple
baselines, and promote nothing without beating the incumbent out-of-sample.

Before this script existed the repository had the MACHINERY for all of that —
an M2 dataset builder, an M4 baseline suite, an M3 trial registry — and zero
recorded trials and zero persisted datasets. The three registered "champions"
are rule-based scorers carrying `metrics: {}`, documented as approved for
governance rather than by comparison. So there was no number for a sequence
model to beat, and any C6 result would have been unfalsifiable.

This script produces that number:

    build_training_dataset   (M2, the only sanctioned generator)
        -> train_baseline_suite  (M4, like-for-like across families)
        -> persist_trial         (M3, so the experiment is on the record)
        -> persist_training_run  (the run ledger)

It trains nothing fancy and adds no dependency. Run it, and C6 finally has an
incumbent to argue with.

Usage:
    python scripts/establish_baseline.py [--tickers N] [--times N] [--horizon 20d]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.baseline_suite import best_simple_baseline  # noqa: E402
from core.backtest.costs import COST_TABLE_V2  # noqa: E402
from core.config import TRAINING_DEFAULT_SEED  # noqa: E402
from core.trial_registry import Trial, load_trial_registry, persist_trial  # noqa: E402
from core.training import persist_training_run, train_baseline_suite  # noqa: E402
from core.training_dataset import (  # noqa: E402
    build_training_dataset,
    persist_training_dataset,
)
from fetch_data import PORTFOLIO_TICKERS, fetch_price_history  # noqa: E402

# Funds and unclassified names: a broad fund is not a company, and the
# fundamental path has nothing to say about it.
_EXCLUDED = {"VOO", "CIBR", "SOXX", "NASA", "SPCX", "CBRS", "KEEL"}


def _prediction_times(frame: pd.DataFrame, count: int, horizon_sessions: int) -> list[str]:
    """Evenly spaced prediction times whose horizon has fully matured.

    The last `horizon_sessions` bars are excluded: a row whose outcome has not
    realized is an EXCLUDED row, not a zero, and there is no point building
    one only for the label builder to reject it.
    """
    usable = frame.iloc[:-horizon_sessions] if horizon_sessions else frame
    if len(usable) < count:
        return [str(stamp) for stamp in usable.index]
    step = max(1, len(usable) // count)
    return [str(stamp) for stamp in usable.index[::step][:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=int, default=12, help="how many holdings to include")
    parser.add_argument("--times", type=int, default=25, help="prediction times per ticker")
    parser.add_argument("--horizon", default="20d", help="target label horizon")
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULT_SEED)
    args = parser.parse_args()

    from core.config import LABEL_HORIZON_SESSIONS

    if args.horizon not in LABEL_HORIZON_SESSIONS:
        print(f"unknown horizon {args.horizon!r}; known: {sorted(LABEL_HORIZON_SESSIONS)}")
        return 1
    horizon_sessions = LABEL_HORIZON_SESSIONS[args.horizon]

    tickers = [t for t in PORTFOLIO_TICKERS if t not in _EXCLUDED][: args.tickers]
    print(f"Loading history for {len(tickers)} holdings...")

    history: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            history[ticker] = fetch_price_history(ticker, period="2y", interval="1d")
        except Exception as exc:
            print(f"  {ticker}: skipped ({type(exc).__name__})")
    if not history:
        print("no history could be loaded; aborting rather than building an empty dataset")
        return 1

    times = {
        ticker: _prediction_times(frame, args.times, horizon_sessions)
        for ticker, frame in history.items()
    }
    total = sum(len(v) for v in times.values())
    print(f"  {len(history)} tickers, {total} candidate prediction times")

    print(f"\nBuilding the M2 dataset (horizon {args.horizon})...")
    dataset = build_training_dataset(
        prediction_times_by_ticker=times,
        history_by_ticker=history,
        target_horizon=args.horizon,
    )
    rows = len(dataset.rows)
    print(f"  rows admitted: {rows}")
    print(f"  dataset hash : {dataset.dataset_hash[:20]}")
    excluded = getattr(dataset, "excluded", None) or []
    print(f"  excluded     : {len(excluded)} (never imputed)")
    if rows < 40:
        print(
            f"\nOnly {rows} rows admitted — too thin to call a baseline measured.\n"
            f"Increase --tickers/--times, or inspect the exclusion reasons above.\n"
            f"Reporting this rather than persisting a baseline nobody should trust."
        )
        return 1

    persisted = persist_training_dataset(dataset)
    print(f"  persisted    : {str(persisted.get('dataset_hash', ''))[:20]}")

    # M3 PRE-REGISTRATION. The registry refuses a trial that arrives carrying
    # metrics: the hypothesis must be on the record BEFORE the result exists,
    # or a disappointing run could quietly be rewritten as a different
    # experiment. This is the anti-cherry-picking discipline M3 was built for.
    print("\nPre-registering the M3 trial (before any result exists)...")
    first_time = min(min(stamps) for stamps in times.values() if stamps)
    last_time = max(max(stamps) for stamps in times.values() if stamps)
    trial = Trial(
        hypothesis=(
            "Simple baselines establish a measured out-of-sample incumbent on the "
            "official M2 dataset, so any later sequence model (C6) has a number to "
            "beat rather than an unfalsifiable claim."
        ),
        feature_set_version=dataset.feature_set_hash,
        # The suite spans several families; the trial is registered against the
        # family whose baseline the incumbent is expected to come from. M3's
        # vocabulary already includes 'sequence', which is what C6 will use.
        model_family="momentum",
        label_version=getattr(dataset, "label_version", "outcome-label-v1"),
        horizons=[args.horizon],
        training_window={
            "start": str(first_time),
            "end": str(last_time),
            "tickers": sorted(history),
            "prediction_times": total,
            "rows_admitted": rows,
        },
        validation_scheme="walk_forward",
        costs={
            "cost_table_version": COST_TABLE_V2["cost_table_version"],
            "note": "the dataset's declared cost assumptions apply unchanged",
        },
        seed=args.seed,
        dataset_hash=dataset.dataset_hash,
        primary_metric="directional_accuracy",
    )
    registered = persist_trial(trial)
    trial_id = str(registered.get("trial_id", ""))
    print(f"  registered: {trial_id[:20]} status={registered.get('status')}")

    print("\nTraining the M4 baseline suite (like-for-like, same folds/seed)...")
    runs = train_baseline_suite(dataset, seed=args.seed)
    for name, run in sorted(runs.items()):
        metrics = run.metrics or {}
        accuracy = metrics.get("directional_accuracy")
        shown = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "n/a"
        print(f"  {name:22} directional_accuracy={shown}")

    # best_simple_baseline returns the RUN, not a (name, run) pair.
    incumbent_run = best_simple_baseline(runs)
    incumbent_name = incumbent_run.estimator
    incumbent_metrics = incumbent_run.metrics or {}
    print(f"\nIncumbent: {incumbent_name}")
    print(f"  metrics: {incumbent_metrics}")

    print("\nRecording the run ledger and completing the trial...")
    for name, run in sorted(runs.items()):
        try:
            persist_training_run(run)
        except Exception as exc:
            print(f"  run {name}: not persisted ({type(exc).__name__}: {exc})")

    registry = load_trial_registry()
    completed = registry.complete(
        trial_id,
        metrics=incumbent_metrics,
        completed_at=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    )
    record = persist_trial(completed)
    print(f"  completed: {str(record.get('trial_id', ''))[:20]} status={record.get('status')}")

    print("\nBaseline established.")
    print(f"  dataset  {dataset.dataset_hash[:20]}  rows {rows}")
    print(f"  incumbent {incumbent_name}  {incumbent_metrics.get('directional_accuracy')}")
    print("  C6 may now be evaluated against this, and only against this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
