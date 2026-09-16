"""CI drift gate for the M2 official training dataset builder.

Proves, on every push, that the dataset contract still holds:

1. the builder produces rows from the live scoring path against real cached
   history;
2. every emitted row is registry-conformant and leakage-free;
3. the dataset hash is deterministic — a rebuild reproduces it exactly;
4. a planted future-dated feature IS caught (the guard is not vacuous);
5. an unregistered feature cannot enter a dataset;
6. persistence is idempotent per dataset hash.

Exit 1 on any drift, with a diff-precise message.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.feature_registry import build_default_registry  # noqa: E402
from core.training_dataset import (  # noqa: E402
    TrainingDatasetError,
    build_training_dataset,
    load_training_datasets,
    persist_training_dataset,
    row_problems,
)

FIXTURE = REPO_ROOT / "data" / "NVDA_5y_1d.parquet"


def fail(message: str) -> None:
    print(f"M2 training-dataset gate FAILED:\n  {message}")
    raise SystemExit(1)


def main() -> None:
    if not FIXTURE.exists():
        print(f"M2 training-dataset gate SKIPPED: fixture {FIXTURE.name} not present")
        return

    frame = pd.read_parquet(FIXTURE)
    frame.index = pd.DatetimeIndex(frame.index)
    times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[-400:-385]]

    dataset = build_training_dataset({"NVDA": times}, {"NVDA": frame})
    if not dataset.rows:
        fail(f"builder produced no rows from {len(times)} prediction times")
    if dataset.excluded:
        fail(f"unexpected exclusions: {dataset.report()['exclusion_reasons']}")

    registry = build_default_registry()
    for row in dataset.rows:
        problems = row_problems(row, registry)
        if problems:
            fail(f"row {row.ticker}@{row.prediction_time} not conformant: {problems[:3]}")

    # Leakage: no feature may be published after the prediction instant.
    for row in dataset.rows:
        prediction = pd.Timestamp(row.prediction_time)
        for name, contract in row.feature_contracts.items():
            if pd.Timestamp(contract["published_time"]) > prediction:
                fail(f"{name} at {row.prediction_time} is published in the future")

    # Determinism.
    rebuilt = build_training_dataset({"NVDA": times}, {"NVDA": frame})
    if rebuilt.dataset_hash != dataset.dataset_hash:
        fail(
            f"dataset hash is not deterministic:\n"
            f"    first  {dataset.dataset_hash}\n"
            f"    second {rebuilt.dataset_hash}"
        )

    # The leakage guard must actually fire — a gate that cannot fail is theatre.
    planted = build_training_dataset({"NVDA": times[:2]}, {"NVDA": frame})
    victim = planted.rows[0]
    feature_name = sorted(victim.feature_contracts)[0]
    victim.feature_contracts[feature_name]["published_time"] = "2099-01-01 00:00:00"
    if not row_problems(victim, registry):
        fail("a future-dated feature contract was NOT rejected — the leakage guard is vacuous")

    # The M1 gate must reach datasets too.
    try:
        build_training_dataset({}, {}, feature_names=["definitely_not_a_registered_feature"])
    except TrainingDatasetError:
        pass
    else:
        fail("an unregistered feature was admitted into a dataset")

    # Persistence idempotency.
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "training_datasets.jsonl"
        persist_training_dataset(dataset, store)
        persist_training_dataset(dataset, store)
        if len(load_training_datasets(store)) != 1:
            fail("persistence is not idempotent per dataset hash")

    print("M2 training-dataset gate OK:")
    print(
        f"  {len(dataset.rows)} rows, {len(dataset.feature_names)} features, "
        f"target {dataset.target_horizon}"
    )
    print(f"  dataset hash deterministic ({dataset.dataset_hash[:16]}...)")
    print("  rows registry-conformant and leakage-free; guards proven non-vacuous.")


if __name__ == "__main__":
    main()
