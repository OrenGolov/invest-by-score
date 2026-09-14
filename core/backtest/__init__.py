"""Sprint V2 — walk-forward validation harness.

Modules:
- costs.py    versioned execution cost table (spread/impact/commission)
- metrics.py  pure, deterministic performance metrics
- manifest.py mandatory run manifests (a run without one is invalid)
- engine.py   walk-forward folding + offline replay of the live scoring path

V7: every completed run also stamps and persists a monitoring snapshot
(`core.monitoring`, monitoring-v1): score-distribution drift (PSI),
confidence drift, stale-data rate, veto rate by rule, and label coverage —
computed by pure functions, versioned, append-only stored.

Simulation only: nothing in this package is a production trading path.
"""

from core.backtest.costs import (
    COST_TABLE_V1,
    COST_TABLE_V2,
    COST_TABLE_VERSION,
    execution_cost_record,
    execution_price,
    liquidity_bucket,
    realized_vol_daily,
    total_side_cost_bps,
)
from core.backtest.engine import (
    BacktestFeatureContractError,
    BacktestLeakageError,
    BacktestManifestError,
    BacktestUniverseError,
    build_walk_forward_folds,
    offline_replay_seam,
    run_walk_forward_backtest,
)
from core.backtest.manifest import (
    BACKTEST_RUNS_PATH,
    build_manifest,
    load_manifest_by_run_hash,
    load_run_manifests,
    persist_run_manifest,
    require_valid_manifest,
    validate_manifest,
)
from core.backtest.metrics import compute_metrics
from core.framing import (
    FRAMING_STORE_PATH,
    FRAMING_VERSION,
    build_framing_block,
    framing_problems,
    load_framing_snapshots,
    persist_framing_snapshot,
)
from core.feature_registry import (
    FEATURE_REGISTRY_STORE_PATH,
    FEATURE_REGISTRY_VERSION,
    FeatureContractError,
    FeatureRegistry,
    FeatureSpec,
    build_default_registry,
    feature_contract_problems,
    feature_set_hash,
    feature_surface_digest,
    load_feature_registry,
    model_feature_problems,
    persist_feature_registry,
    producer_problems,
    registry_hash,
    require_model_features,
    spec_problems,
    unwired_producers,
)
from core.monitoring import (
    MONITORING_VERSION,
    MONITORING_STORE_PATH,
    confidence_drift,
    label_coverage,
    load_monitoring_snapshots,
    monitoring_snapshot,
    persist_monitoring_snapshot,
    score_drift_psi,
    stale_data_rate,
    veto_rate_by_rule,
)

__all__ = [
    "BACKTEST_RUNS_PATH",
    "COST_TABLE_V1",
    "COST_TABLE_V2",
    "COST_TABLE_VERSION",
    "BacktestFeatureContractError",
    "BacktestLeakageError",
    "BacktestManifestError",
    "BacktestUniverseError",
    "FEATURE_REGISTRY_STORE_PATH",
    "FEATURE_REGISTRY_VERSION",
    "FRAMING_STORE_PATH",
    "FRAMING_VERSION",
    "FeatureContractError",
    "FeatureRegistry",
    "FeatureSpec",
    "MANIFEST_VERSION",
    "MONITORING_STORE_PATH",
    "MONITORING_VERSION",
    "build_default_registry",
    "build_framing_block",
    "build_manifest",
    "build_walk_forward_folds",
    "compute_metrics",
    "confidence_drift",
    "execution_cost_record",
    "execution_price",
    "feature_contract_problems",
    "feature_set_hash",
    "feature_surface_digest",
    "framing_problems",
    "load_feature_registry",
    "load_framing_snapshots",
    "load_manifest_by_run_hash",
    "load_monitoring_snapshots",
    "load_run_manifests",
    "liquidity_bucket",
    "model_feature_problems",
    "monitoring_snapshot",
    "offline_replay_seam",
    "persist_feature_registry",
    "persist_framing_snapshot",
    "persist_monitoring_snapshot",
    "persist_run_manifest",
    "producer_problems",
    "realized_vol_daily",
    "registry_hash",
    "require_model_features",
    "require_valid_manifest",
    "run_walk_forward_backtest",
    "score_drift_psi",
    "spec_problems",
    "stale_data_rate",
    "total_side_cost_bps",
    "unwired_producers",
    "validate_manifest",
    "veto_rate_by_rule",
]
