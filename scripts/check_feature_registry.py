"""CI drift gate for the M1 canonical feature registry.

Verifies, with no network and no wall-clock dependence in the checked logic:
  1. the default registry builds with zero problems (every spec complete);
  2. every declared producer resolves to a real module + callable;
  3. the live market snapshot's emitted feature-contract surface is exactly
     the registered market-data surface — drift is a build failure with a
     diff-precise message (added/removed/renamed features cannot slide in);
  4. every emitted contract satisfies the registry (PIT, version, lookback,
     declared source);
  5. the registry hash is deterministic across builds;
  6. persistence round-trips and is idempotent per registry hash.

Exit code 0 = conforming; 1 = drift or violation (diff-precise messages).
The M1 acceptance rule this script pins: an unregistered feature cannot
enter a production model, and registry/snapshot drift is a build failure,
never a runtime surprise.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402  (repo-root import bootstrap above)

from agents.market_data_agent import fetch_market_snapshot  # noqa: E402
from core.contract_verification import feature_registry_problems  # noqa: E402
from core.feature_registry import (  # noqa: E402
    PRODUCERS,
    build_default_registry,
    load_feature_registry,
    persist_feature_registry,
    producer_problems,
    registry_hash,
    unwired_producers,
)


def _ramp(sessions: int, step: float = 1.0, base: float = 100.0) -> list[float]:
    return [base + step * i for i in range(sessions)]


def _offline_snapshot():
    """A deterministic offline market snapshot (patched price fetch, no network)."""
    frame = pd.DataFrame(
        {
            "Open": _ramp(320), "High": _ramp(320), "Low": _ramp(320),
            "Close": _ramp(320), "Volume": [1_000_000.0] * 320,
        },
        index=pd.date_range("2022-01-03", periods=320, freq="B"),
    )
    as_of = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    with patch("agents.market_data_agent.fetch_price_history", lambda *a, **k: frame.copy()):
        return fetch_market_snapshot("TEST", as_of), as_of


def main() -> int:
    failures: list[str] = []

    # 1. Registry validity — every spec complete, every producer resolvable.
    registry = build_default_registry()
    problems = registry.problems()
    if problems:
        failures.append(f"registry invalid: {'; '.join(problems)}")

    # 2. Producers resolve to real modules + callables.
    for producer in PRODUCERS:
        producer_failures = producer_problems(producer.name)
        if producer_failures:
            failures.append(f"producer {producer.name}: {'; '.join(producer_failures)}")

    unwired = unwired_producers(registry)
    print(
        f"registry: {len(registry.all_features())} features, "
        f"hash {registry_hash(registry)[:16]}..."
    )
    print(
        f"producers: {len(PRODUCERS)} declared, "
        f"unwired (forward wiring, informational): {sorted(unwired)}"
    )

    # 3+4. Live snapshot conformance + drift vs the registered surface.
    snapshot, as_of = _offline_snapshot()
    emitted = set(snapshot.get("features") or {})
    registered_market = {
        name for name, spec in registry.all_features().items()
        if spec.owner == "market_data_agent"
    }
    conformance_problems = feature_registry_problems(snapshot)
    if conformance_problems:
        failures.append(
            f"live snapshot violates the registry: {'; '.join(conformance_problems[:5])}"
        )
    if emitted != registered_market:
        added = sorted(emitted - registered_market)
        removed = sorted(registered_market - emitted)
        failures.append(
            "feature-surface drift: emitted != registered — "
            f"unregistered/new: {added}; registered-but-missing: {removed}"
        )

    # 5. Deterministic hash across builds.
    if registry_hash(build_default_registry()) != registry_hash(registry):
        failures.append("registry hash is not deterministic across builds")

    # 6. Persistence round-trip + idempotency.
    with tempfile.TemporaryDirectory() as tmp:
        store = pathlib.Path(tmp) / "feature_registry.jsonl"
        if not persist_feature_registry(registry, store):
            failures.append("first persist did not append the record")
        if persist_feature_registry(registry, store):
            failures.append("second persist was not idempotent")
        if registry_hash(load_feature_registry(store)) != registry_hash(registry):
            failures.append("persistence round-trip changed the registry hash")

    if failures:
        print("M1 feature-registry gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M1 feature-registry gate OK:")
    print(
        f"  emitted surface == registered market surface "
        f"({len(emitted)} contracts, registry-conformant at {as_of})"
    )
    print("  registry valid, producers resolve, hash deterministic, persistence idempotent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())