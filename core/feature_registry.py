"""Canonical feature registry (Sprint M1).

Every feature that enters a production model must be registered here with
complete metadata. The registry is the single source of truth for what
features exist, how they are computed, who owns them, and which models may
consume them.

Binding rules (enforced, not aspirational):

- An unregistered feature CANNOT enter a production model. The contract
  verifier and the backtest engine both refuse a snapshot that carries an
  unregistered feature — it is a hard failure, not a warning.
- A feature's producer (owner) must exist. `spec_problems` rejects any
  feature whose owner is not in KNOWN_PRODUCERS.
- Future/revised input is rejected. A feature contract whose published_time
  is after as_of violates the PIT rule and is rejected by
  `feature_contract_problems`.
- Each feature carries a deterministic hash over its canonical spec. Any
  change to the feature's definition produces a different hash.

Design notes:

- Pure data + pure functions only. No wall-clock, no randomness.
- Append-only persistence at `data/feature_registry.jsonl`, idempotent per
  registry hash, with the same integrity model as outcomes / manifests /
  framing / monitoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import MARKET_FEATURE_VERSION

FEATURE_REGISTRY_VERSION = "feature-registry-v1"
FEATURE_REGISTRY_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "feature_registry.jsonl"

FEATURE_DOMAINS: tuple[str, ...] = (
    "market", "fundamental", "news", "sentiment", "macro", "regime", "technical",
)

NULL_POLICIES: tuple[str, ...] = ("exclude", "fail", "default", "flag")

FREQUENCIES: tuple[str, ...] = ("daily", "per_bar", "per_session")

KNOWN_PRODUCERS: tuple[str, ...] = (
    "market_data_agent", "technical_agent", "fundamental_agent",
    "news_agent", "sentiment_agent", "macro_agent", "regime_agent",
)


@dataclass
class FeatureSpec:
    """Canonical specification for a single feature."""

    name: str
    owner: str
    domain: str
    formula: str
    version: str
    unit: str
    frequency: str
    lookback: str
    minimum_history: int
    null_policy: str
    pit_rule: str
    source_dependencies: list[str] = field(default_factory=list)
    feature_family: str = ""
    model_compatibility: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_hash(self) -> str:
        """Deterministic SHA-256 over the canonical JSON of this spec."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FeatureRegistry:
    """Canonical feature registry — the single source of truth."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        problems = spec_problems(spec)
        if problems:
            raise ValueError(
                f"invalid feature spec for {spec.name!r}: {'; '.join(problems)}"
            )
        self._features[spec.name] = spec

    def get(self, name: str) -> FeatureSpec | None:
        return self._features.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._features

    def feature_hash(self, name: str) -> str | None:
        spec = self._features.get(name)
        return spec.canonical_hash() if spec is not None else None

    def all_features(self) -> dict[str, FeatureSpec]:
        return dict(self._features)

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not self._features:
            problems.append("feature registry is empty — no features registered")
        return problems


def spec_problems(spec: FeatureSpec) -> list[str]:
    """Return the list of problems with a feature spec; empty means valid."""
    problems: list[str] = []
    if not spec.name:
        problems.append("feature name is empty")
    if spec.owner not in KNOWN_PRODUCERS:
        problems.append(
            f"owner {spec.owner!r} is not a known producer "
            f"(known: {sorted(KNOWN_PRODUCERS)})"
        )
    if spec.domain not in FEATURE_DOMAINS:
        problems.append(
            f"domain {spec.domain!r} is not a valid domain "
            f"(valid: {sorted(FEATURE_DOMAINS)})"
        )
    if not spec.formula:
        problems.append("formula is empty")
    if not spec.version:
        problems.append("version is empty")
    if spec.null_policy not in NULL_POLICIES:
        problems.append(
            f"null_policy {spec.null_policy!r} is not valid "
            f"(valid: {sorted(NULL_POLICIES)})"
        )
    if spec.frequency not in FREQUENCIES:
        problems.append(
            f"frequency {spec.frequency!r} is not valid "
            f"(valid: {sorted(FREQUENCIES)})"
        )
    if spec.minimum_history < 1:
        problems.append(
            f"minimum_history must be >= 1, got {spec.minimum_history!r}"
        )
    if not spec.pit_rule:
        problems.append("pit_rule is empty")
    return problems


def feature_contract_problems(
    snapshot: dict,
    registry: FeatureRegistry,
) -> list[str]:
    """Return problems with the feature contracts on a snapshot.

    Binding rules:
    1. Every feature must be registered — unregistered features cannot enter
       a production model.
    2. published_time must be <= as_of — future/revised input is rejected.
    """
    problems: list[str] = []
    features = snapshot.get("features") or {}
    if not features:
        problems.append(
            "snapshot carries no feature contracts — a producer returned an empty feature surface"
        )
        return problems

    for name in sorted(features):
        contract = features[name]
        if not registry.is_registered(name):
            problems.append(
                f"feature {name!r} is not registered — "
                f"unregistered features cannot enter a production model"
            )
            continue

        published = contract.get("published_time")
        as_of = contract.get("as_of")
        if published is not None and as_of is not None:
            if str(published) > str(as_of):
                problems.append(
                    f"feature {name!r} published_time {published!r} is after "
                    f"as_of {as_of!r} — future/revised input is rejected by the PIT rule"
                )
    return problems


def registry_hash(registry: FeatureRegistry) -> str:
    """Deterministic hash over the entire registry."""
    payload = {
        name: spec.to_dict()
        for name, spec in sorted(registry.all_features().items())
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

_MARKET_PIT_RULE = (
    "Feature is computed using only bars with timestamp <= as_of; "
    "future bars are removed before indicators are calculated. "
    "Source data must be published on or before as_of."
)


def _market_feature(
    name: str,
    formula: str,
    unit: str,
    lookback: str,
    minimum_history: int,
    null_policy: str,
    feature_family: str,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        owner="market_data_agent",
        domain="market",
        formula=formula,
        version=MARKET_FEATURE_VERSION,
        unit=unit,
        frequency="daily",
        lookback=lookback,
        minimum_history=minimum_history,
        null_policy=null_policy,
        pit_rule=_MARKET_PIT_RULE,
        source_dependencies=["yahoo_finance_chart"],
        feature_family=feature_family,
        model_compatibility=["technical_analysis"],
    )


def build_default_registry() -> FeatureRegistry:
    """Build the canonical default registry (22 market-data features)."""
    registry = FeatureRegistry()

    _RETURN = (
        "Fractional price change over {window} sessions: "
        "(close / close_{window}_sessions_ago) - 1. "
        "Returns 0.0 when fewer than {need} bars are available or the anchor is zero."
    )
    for window, need, lookback in [
        (1, 2, "1d"), (5, 6, "5d"), (20, 21, "20d"), (60, 61, "60d"),
        (50, 51, "50d"), (100, 101, "100d"), (150, 151, "150d"), (200, 201, "200d"),
    ]:
        registry.register(_market_feature(
            name=f"change_{window}d",
            formula=_RETURN.format(window=window, need=need),
            unit="ratio", lookback=lookback, minimum_history=need,
            null_policy="default", feature_family="momentum",
        ))

    registry.register(_market_feature(
        name="atr_14",
        formula=(
            "14-period Average True Range (simple mean): "
            "true_range = max(high-low, |high-prev_close|, |low-prev_close|); "
            "atr_14 = mean(true_range, 14). None until 15 bars exist."
        ),
        unit="price", lookback="14d", minimum_history=15,
        null_policy="exclude", feature_family="volatility",
    ))
    registry.register(_market_feature(
        name="volatility",
        formula=(
            "30-period close-to-close annualized volatility: "
            "std(ln(close/prev_close), 30) * sqrt(252). None until 30 bars."
        ),
        unit="ratio", lookback="30d", minimum_history=30,
        null_policy="exclude", feature_family="volatility",
    ))
    registry.register(_market_feature(
        name="trend_slope_60d",
        formula=(
            "Least-squares slope of last 60 closes vs session position "
            "(price/session, signed). None until 60 bars."
        ),
        unit="price_per_session", lookback="60d", minimum_history=60,
        null_policy="exclude", feature_family="trend",
    ))
    registry.register(_market_feature(
        name="trend_vs_20d_mean",
        formula=(
            "Ratio of latest close to 20-session SMA: "
            "close/sma(close,20) - 1. None until 20 bars."
        ),
        unit="ratio", lookback="20d", minimum_history=20,
        null_policy="exclude", feature_family="trend",
    ))


    for window in (50, 100, 150, 200):
        registry.register(_market_feature(
            name=f"price_vs_ma_{window}",
            formula=(
                f"Ratio of latest close to {window}-session SMA: "
                f"close/sma(close,{window}) - 1. "
                f"Returns 0.0 when the moving average is zero."
            ),
            unit="ratio", lookback=f"{window}d", minimum_history=window,
            null_policy="default", feature_family="trend",
        ))
    for window in (50, 100, 150, 200):
        registry.register(_market_feature(
            name=f"ma_{window}",
            formula=(
                f"{window}-session simple moving average of close: "
                f"sum(close,{window})/{window}. None until {window} bars."
            ),
            unit="price", lookback=f"{window}d", minimum_history=window,
            null_policy="exclude", feature_family="trend",
        ))

    registry.register(_market_feature(
        name="rsi",
        formula=(
            "14-period Relative Strength Index (Wilder-smoothed): "
            "avg_gain/avg_loss over 14 sessions; "
            "rsi = 100 - (100/(1 + avg_gain/avg_loss)). "
            "Normalized 0-100. None until 15 bars."
        ),
        unit="index", lookback="14d", minimum_history=15,
        null_policy="exclude", feature_family="momentum",
    ))
    registry.register(_market_feature(
        name="volume_ratio_20d",
        formula=(
            "Ratio of latest volume to 20-session average volume: "
            "volume/mean(volume,20). Returns 0.0 when average volume is zero."
        ),
        unit="ratio", lookback="20d", minimum_history=20,
        null_policy="default", feature_family="volume",
    ))

    return registry


def persist_feature_registry(
    registry: FeatureRegistry,
    path: str | Path | None = None,
) -> bool:
    """Persist the registry to append-only JSONL. Idempotent per registry hash."""
    store_path = Path(path) if path is not None else FEATURE_REGISTRY_STORE_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "registry_version": FEATURE_REGISTRY_VERSION,
        "registry_hash": registry_hash(registry),
        "features": {
            name: spec.to_dict()
            for name, spec in sorted(registry.all_features().items())
        },
    }

    if store_path.exists():
        for raw in store_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"feature registry store {store_path} has a malformed line: {exc}"
                ) from exc
            if existing.get("registry_hash") == record["registry_hash"]:
                if existing == record:
                    return False
                raise ValueError(
                    f"feature registry integrity violation: hash "
                    f"{record['registry_hash']} already exists with different content."
                )

    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return True


def load_feature_registry(
    path: str | Path | None = None,
) -> FeatureRegistry:
    """Load the latest registry from the append-only JSONL store."""
    store_path = Path(path) if path is not None else FEATURE_REGISTRY_STORE_PATH
    if not store_path.exists():
        return FeatureRegistry()

    records: list[dict] = []
    with store_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"feature registry store {store_path} line {line_number} "
                    f"is not valid JSON: {exc}"
                ) from exc

    if not records:
        return FeatureRegistry()

    latest = records[-1]
    registry = FeatureRegistry()
    for name, spec_dict in (latest.get("features") or {}).items():
        try:
            spec = FeatureSpec(**spec_dict)
        except TypeError as exc:
            raise ValueError(
                f"feature {name!r} in registry store is malformed: {exc}"
            ) from exc
        registry._features[name] = spec
    return registry

