"""Sprint V2 — walk-forward validation harness.

Modules:
- costs.py    versioned execution cost table (spread/impact/commission)
- metrics.py  pure, deterministic performance metrics
- manifest.py mandatory run manifests (a run without one is invalid)
- engine.py   walk-forward folding + offline replay of the live scoring path

Simulation only: nothing in this package is a production trading path.
"""

from core.backtest.costs import COST_TABLE_V1, COST_TABLE_VERSION, execution_price, liquidity_bucket, total_side_cost_bps
from core.backtest.engine import (
    BacktestLeakageError,
    build_walk_forward_folds,
    offline_replay_seam,
    run_walk_forward_backtest,
)
from core.backtest.manifest import MANIFEST_VERSION, build_manifest, validate_manifest
from core.backtest.metrics import compute_metrics

__all__ = [
    "COST_TABLE_V1",
    "COST_TABLE_VERSION",
    "BacktestLeakageError",
    "MANIFEST_VERSION",
    "build_manifest",
    "build_walk_forward_folds",
    "compute_metrics",
    "execution_price",
    "liquidity_bucket",
    "offline_replay_seam",
    "run_walk_forward_backtest",
    "total_side_cost_bps",
    "validate_manifest",
]
