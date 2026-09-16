"""Canonical feature registry (Sprint M1) — the only door into a model.

Every feature that enters a production model must be registered here with
complete metadata. The registry is the single source of truth for what
features exist, how they are computed, who owns them, which source data they
depend on, and which model families may consume them.

Required metadata — all of it validated by `spec_problems`, not aspirational:
name, owner, domain, formula, version, unit, frequency, lookback, minimum
history, null policy, PIT rule, source dependencies, feature family, model
compatibility. The vocabularies are closed: an unknown unit, family, domain,
frequency, source, null policy, or model family is a rejection, never a
silent default.

Binding rules (enforced, not aspirational):

- **An unregistered feature cannot enter a production model.** Three gates
  apply the rule: `model_feature_problems` (a model's declared feature set),
  `feature_contract_problems` (the feature contracts a producer emitted), and
  the walk-forward engine, which refuses to start when the scored model's
  declared inputs are not registry-conformant and refuses mid-run when a
  score result exposes a non-conformant feature surface.
- **A producer must exist.** `PRODUCERS` maps every owner to the module and
  callable that actually produces its features, and `producer_problems`
  verifies that both exist — a feature cannot name an owner that is only a
  string. A producer with no registered feature is reported by
  `unwired_producers` (informational: forward wiring, sprint by sprint).
- **Future/revised input is rejected.** A contract whose `published_time` is
  after `as_of` violates the PIT rule; a contract whose `calculation_version`
  or `lookback_period` disagrees with the registered spec, or that arrives
  from a source the spec does not declare, is a revised/undeclared input; and
  re-registering a feature definition without bumping its version is refused
  by `FeatureRegistry.register`.
- **Deterministic feature hash.** A spec, the whole registry, a named feature
  set, and a consumed feature surface each carry a canonical SHA-256, so a
  definition change is always visible and no two different feature sets share
  an identity.

Design notes:

- Pure data + pure functions only. No wall-clock, no randomness.
- Append-only persistence at `data/feature_registry.jsonl`, idempotent per
  registry hash, with the same integrity model as outcomes / manifests /
  framing / monitoring.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.config import (
    FEATURE_REGISTRY_VERSION,
    FUNDAMENTAL_FEATURE_VERSION,
    MACRO_CONTRACT_VERSION,
    MARKET_FEATURE_VERSION,
    NEWS_CONTRACT_VERSION,
    REGIME_CONTRACT_VERSION,
    REGIME_REQUIRED_SESSIONS,
    SENTIMENT_CONTRACT_VERSION,
)

FEATURE_REGISTRY_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "feature_registry.jsonl"
_REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DOMAINS: tuple[str, ...] = (
    "market", "fundamental", "news", "sentiment", "macro", "regime", "technical", "event",
)

NULL_POLICIES: tuple[str, ...] = ("exclude", "fail", "default", "flag")

FREQUENCIES: tuple[str, ...] = (
    "per_bar", "per_session", "daily", "weekly", "monthly", "quarterly", "event",
)

KNOWN_UNITS: tuple[str, ...] = (
    "ratio", "price", "price_per_session", "index", "percent", "count",
    "sessions", "shares", "currency", "score", "zscore", "bool", "text",
)

FEATURE_FAMILIES: tuple[str, ...] = (
    "momentum", "volatility", "trend", "volume", "mean_reversion", "structure",
    "relative_strength", "liquidity", "breadth", "fundamental", "quality",
    "valuation", "growth", "news_event", "sentiment", "macro", "regime",
)

MODEL_FAMILIES: tuple[str, ...] = (
    "technical_analysis", "baseline_mean", "momentum", "mean_reversion",
    "linear", "logistic", "tree", "boosting", "sequence",
)

KNOWN_SOURCES: tuple[str, ...] = (
    "yahoo_finance_chart", "alpha_vantage_overview", "fred_macro",
    "newsapi_news", "portfolio_list_snapshot",
    # The sentiment agent's two declared source identities (N2). Neither is a
    # live provider today: the contract is permanently UNAVAILABLE unless the
    # sanctioned news-derived path is used, which labels itself explicitly.
    # They are declared here so the feature can be registered and gated like
    # any other rather than living outside the registry.
    "sentiment_provider_unconfigured", "sentiment_derived_from_news",
)

FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
LOOKBACK_PATTERN = re.compile(r"^[1-9][0-9]*[dwqmy]$")


@dataclass(frozen=True)
class Producer:
    """The module + callable that actually produces a feature's raw inputs."""

    name: str
    module: str
    callable_name: str


PRODUCERS: tuple[Producer, ...] = (
    Producer("market_data_agent", "agents.market_data_agent", "fetch_market_snapshot"),
    Producer("technical_agent", "agents.technical_agent", "score_technical"),
    Producer("fundamental_agent", "core.score_engine", "_build_fundamental_score"),
    Producer("news_agent", "core.news_adapter", "build_news_snapshot"),
    Producer("sentiment_agent", "core.sentiment_contract", "fetch_sentiment_snapshot"),
    Producer("macro_agent", "core.macro_adapter", "build_macro_snapshot"),
    Producer("regime_agent", "core.regime_agent", "build_regime_snapshot"),
)

KNOWN_PRODUCERS: tuple[str, ...] = tuple(producer.name for producer in PRODUCERS)
PRODUCER_BY_NAME: dict[str, Producer] = {producer.name: producer for producer in PRODUCERS}


_MODULE_SOURCE_CACHE: dict[str, str] = {}


class FeatureContractError(ValueError):
    """Raised when a feature spec, model feature set, or snapshot violates the registry."""


def _module_source(module_path: Path) -> str | None:
    """Read a module source file once per process (deterministic, cached)."""
    key = str(module_path)
    if key not in _MODULE_SOURCE_CACHE:
        try:
            _MODULE_SOURCE_CACHE[key] = module_path.read_text(encoding="utf-8")
        except OSError:
            return None
    return _MODULE_SOURCE_CACHE[key]


def producer_problems(owner: str) -> list[str]:
    """Verify a declared producer exists: module file + producer callable on disk.

    The producer must be a real, importable producer — a name in a tuple is not
    evidence that anything can actually produce the feature.
    """
    producer = PRODUCER_BY_NAME.get(owner)
    if producer is None:
        return [
            f"producer {owner!r} is not in the producer registry "
            f"(known: {sorted(KNOWN_PRODUCERS)})"
        ]
    module_path = _REPO_ROOT / Path(*producer.module.split(".")).with_suffix(".py")
    if not module_path.is_file():
        return [
            f"producer {owner!r} declares module {producer.module!r}, which does not "
            f"exist at {module_path}"
        ]
    source = _module_source(module_path)
    if source is None:
        return [
            f"producer {owner!r} declares module {producer.module!r}, which could not "
            f"be read at {module_path}"
        ]
    if f"def {producer.callable_name}(" not in source:
        return [
            f"producer {owner!r} declares callable {producer.callable_name!r}, which is "
            f"not defined in {producer.module}"
        ]
    return []


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
        self._revisions: list[dict[str, Any]] = []

    def register(self, spec: FeatureSpec) -> bool:
        """Register a spec. Returns True when the registry changed.

        Fail-closed rules:
        - an invalid spec (any missing/invalid metadata field) is refused;
        - a declared producer that does not exist is refused;
        - re-registering an existing feature with the SAME version but a
          different definition is refused as revised input — changing a
          feature's meaning requires a version bump (the history is kept).
        """
        problems = spec_problems(spec)
        if problems:
            raise ValueError(
                f"invalid feature spec for {spec.name!r}: {'; '.join(problems)}"
            )
        producer_issues = producer_problems(spec.owner)
        if producer_issues:
            raise ValueError(
                f"invalid feature spec for {spec.name!r}: {'; '.join(producer_issues)}"
            )

        existing = self._features.get(spec.name)
        if existing is not None:
            if existing.canonical_hash() == spec.canonical_hash():
                return False
            if existing.version == spec.version:
                raise ValueError(
                    f"feature {spec.name!r} is already registered at version "
                    f"{spec.version!r} with a different definition — revised input "
                    f"is rejected: bump the version to change a feature"
                )
            self._revisions.append({
                "name": spec.name,
                "superseded_version": existing.version,
                "superseded_hash": existing.canonical_hash(),
                "version": spec.version,
                "hash": spec.canonical_hash(),
            })
        self._features[spec.name] = spec
        return True

    def get(self, name: str) -> FeatureSpec | None:
        return self._features.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._features

    def feature_hash(self, name: str) -> str | None:
        spec = self._features.get(name)
        return spec.canonical_hash() if spec is not None else None

    def feature_names(self) -> list[str]:
        return sorted(self._features)

    def feature_set_hash(self, names: Iterable[str]) -> str:
        """Deterministic hash of a named feature set against this registry."""
        return feature_set_hash(names, self)

    def all_features(self) -> dict[str, FeatureSpec]:
        return dict(self._features)

    def revisions(self) -> list[dict]:
        """Version-bump history (superseded definitions are never deleted)."""
        return [dict(entry) for entry in self._revisions]

    def problems(self) -> list[str]:
        """Structural registry health: non-empty, every spec valid, producers real."""
        problems: list[str] = []
        if not self._features:
            problems.append("feature registry is empty — no features registered")
        for name in sorted(self._features):
            spec = self._features[name]
            if spec.name != name:
                problems.append(
                    f"registry key {name!r} disagrees with spec name {spec.name!r}"
                )
            problems.extend(f"feature {name!r}: {p}" for p in spec_problems(spec))
        problems.extend(registry_producer_problems(self))
        return problems


def registry_producer_problems(registry: FeatureRegistry) -> list[str]:
    """Every producer referenced by the registry must actually exist."""
    problems: list[str] = []
    for owner in sorted({spec.owner for spec in registry.all_features().values()}):
        problems.extend(producer_problems(owner))
    return problems


def unwired_producers(registry: FeatureRegistry) -> list[str]:
    """Producers that exist but own no registered feature yet (informational).

    Forward wiring is expected to be incremental (news/sentiment/macro/regime
    features arrive in their own sprints), so this is reported rather than
    treated as a registry failure.
    """
    owned = {spec.owner for spec in registry.all_features().values()}
    return [producer.name for producer in PRODUCERS if producer.name not in owned]


def _lookback_problems(lookback: str) -> list[str]:
    if not lookback:
        return ["lookback is empty"]
    if not LOOKBACK_PATTERN.match(str(lookback)):
        return [
            f"lookback {lookback!r} must be a positive integer followed by "
            f"d|w|m|q|y (e.g. '20d', '3m')"
        ]
    return []


def _lookback_sessions(lookback: str) -> int | None:
    """Sessions implied by a day-unit lookback (None for non-day units)."""
    if not isinstance(lookback, str) or not LOOKBACK_PATTERN.match(lookback):
        return None
    if not lookback.endswith("d"):
        return None
    return int(lookback[:-1])


def spec_problems(spec: FeatureSpec) -> list[str]:
    """Return the list of problems with a feature spec; empty means valid.

    Every field the M1 contract requires is checked against a closed
    vocabulary, so a feature cannot enter a registry with an unspecified unit,
    an undeclared source, an unknown family, or a compatibility list that no
    model recognizes.
    """
    problems: list[str] = []
    if not spec.name:
        problems.append("feature name is empty")
    elif not FEATURE_NAME_PATTERN.match(str(spec.name)):
        problems.append(f"feature name {spec.name!r} must be lower_snake_case")
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
    if not spec.unit:
        problems.append("unit is empty")
    elif spec.unit not in KNOWN_UNITS:
        problems.append(
            f"unit {spec.unit!r} is not a known unit (known: {sorted(KNOWN_UNITS)})"
        )
    if spec.frequency not in FREQUENCIES:
        problems.append(
            f"frequency {spec.frequency!r} is not valid "
            f"(valid: {sorted(FREQUENCIES)})"
        )
    problems.extend(_lookback_problems(spec.lookback))
    if not isinstance(spec.minimum_history, int) or isinstance(spec.minimum_history, bool):
        problems.append(
            f"minimum_history must be an int, got {spec.minimum_history!r}"
        )
    elif spec.minimum_history < 1:
        problems.append(
            f"minimum_history must be >= 1, got {spec.minimum_history!r}"
        )
    else:
        sessions = _lookback_sessions(spec.lookback)
        if sessions is not None and spec.minimum_history < sessions:
            problems.append(
                f"minimum_history {spec.minimum_history} is below the {spec.lookback} "
                f"lookback — a feature cannot be computed before its own window exists"
            )
    if spec.null_policy not in NULL_POLICIES:
        problems.append(
            f"null_policy {spec.null_policy!r} is not valid "
            f"(valid: {sorted(NULL_POLICIES)})"
        )
    if not spec.pit_rule:
        problems.append("pit_rule is empty")
    if not spec.source_dependencies:
        problems.append(
            "source_dependencies is empty — a feature must declare the source "
            "data it depends on"
        )
    else:
        for source in spec.source_dependencies:
            if source not in KNOWN_SOURCES:
                problems.append(
                    f"source_dependency {source!r} is not a known source "
                    f"(known: {sorted(KNOWN_SOURCES)})"
                )
    if not spec.feature_family:
        problems.append("feature_family is empty")
    elif spec.feature_family not in FEATURE_FAMILIES:
        problems.append(
            f"feature_family {spec.feature_family!r} is not a known family "
            f"(known: {sorted(FEATURE_FAMILIES)})"
        )
    if not spec.model_compatibility:
        problems.append(
            "model_compatibility is empty — a feature must declare which model "
            "families may consume it"
        )
    else:
        for family in spec.model_compatibility:
            if family not in MODEL_FAMILIES:
                problems.append(
                    f"model_compatibility {family!r} is not a known model family "
                    f"(known: {sorted(MODEL_FAMILIES)})"
                )
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
    3. calculation_version must equal the registered version — a changed
       definition must be registered as a new version.
    4. lookback_period must equal the registered lookback.
    5. source_id must be one of the feature's declared source dependencies.
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
        if not isinstance(contract, dict):
            problems.append(
                f"feature {name!r} contract is not a dict, got {type(contract).__name__}"
            )
            continue
        spec = registry.get(name)
        if spec is None:
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
        version = contract.get("calculation_version")
        if version is not None and version != spec.version:
            problems.append(
                f"feature {name!r} calculation_version {version!r} != registered "
                f"version {spec.version!r} — a revised definition must be "
                f"registered as a new version"
            )
        lookback = contract.get("lookback_period")
        if lookback is not None and str(lookback) != spec.lookback:
            problems.append(
                f"feature {name!r} lookback_period {lookback!r} != registered "
                f"lookback {spec.lookback!r}"
            )
        source_id = contract.get("source_id")
        if (
            source_id is not None
            and spec.source_dependencies
            and str(source_id) not in spec.source_dependencies
        ):
            problems.append(
                f"feature {name!r} arrived from source {source_id!r}, which is not a "
                f"declared source dependency of the feature "
                f"(declared: {sorted(spec.source_dependencies)})"
            )
    return problems


def feature_set_hash(names: Iterable[str], registry: FeatureRegistry) -> str:
    """Deterministic hash identifying a named feature set (names + definitions).

    This is the identity a model, dataset, or trial records: two feature sets
    with different definitions can never share a hash, and an unregistered
    name hashes as an explicit null so an invalid set is still identifiable.
    """
    payload = []
    for name in sorted({str(entry) for entry in names}):
        spec = registry.get(name)
        payload.append([name, spec.canonical_hash() if spec is not None else None])
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_surface_digest(surface: dict, registry: FeatureRegistry) -> str:
    """Deterministic hash over the feature contracts a model actually consumed."""
    payload = []
    for name in sorted(surface or {}):
        contract = surface[name]
        contract_dict = contract if isinstance(contract, dict) else {}
        spec = registry.get(name)
        payload.append({
            "name": name,
            "spec_hash": spec.canonical_hash() if spec is not None else None,
            "value": contract_dict.get("value"),
            "as_of": contract_dict.get("as_of"),
            "published_time": contract_dict.get("published_time"),
            "calculation_version": contract_dict.get("calculation_version"),
            "lookback_period": contract_dict.get("lookback_period"),
            "source_id": contract_dict.get("source_id"),
        })
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_feature_problems(
    feature_names: Iterable[str],
    registry: FeatureRegistry,
    model_family: str | None = None,
) -> list[str]:
    """The production-model gate: what a model declares, it must be allowed.

    An unregistered feature cannot enter a production model; a feature that
    does not declare compatibility with the consuming model family cannot
    enter that family either.
    """
    problems: list[str] = []
    names = [str(name) for name in feature_names]
    if not names:
        problems.append(
            "a model must declare at least one feature — an empty feature set "
            "is not a model contract"
        )
    if model_family is not None and model_family not in MODEL_FAMILIES:
        problems.append(
            f"model_family {model_family!r} is not a known model family "
            f"(known: {sorted(MODEL_FAMILIES)})"
        )
        return problems
    for name in sorted(set(names)):
        spec = registry.get(name)
        if spec is None:
            problems.append(
                f"feature {name!r} is not registered — an unregistered feature "
                f"cannot enter a production model"
            )
            continue
        if model_family is not None and model_family not in spec.model_compatibility:
            problems.append(
                f"feature {name!r} does not declare compatibility with model "
                f"family {model_family!r} (declares {sorted(spec.model_compatibility)})"
            )
    return problems


def require_model_features(
    feature_names: Iterable[str],
    registry: FeatureRegistry,
    model_family: str | None = None,
) -> str:
    """Fail-closed helper: return the feature-set hash or raise."""
    problems = model_feature_problems(feature_names, registry, model_family)
    if problems:
        raise FeatureContractError(
            "model feature set violates the canonical feature registry: "
            + "; ".join(problems)
        )
    return feature_set_hash(feature_names, registry)


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
        # Market features back the technical scorers AND are legitimate ML
        # inputs: same PIT rule, same numeric contract the fundamental
        # factors already declare for these families. Declaring only
        # technical_analysis would have barred the M3 trainer from the very
        # features the dataset builder emits.
        model_compatibility=[
            "technical_analysis", "baseline_mean", "momentum", "mean_reversion",
            "linear", "logistic", "tree", "boosting",
        ],
    )


# The PIT rule every fundamental factor inherits: metrics may only enter
# through a fundamentals snapshot whose source contract is timestamped at or
# before as_of. Later-published metrics are rejected upstream and can never
# reach the factor (M1 future/revised-input rule).
_FUNDAMENTAL_PIT_RULE = (
    "Fundamental metrics enter only through the fundamentals snapshot whose "
    "source contract is timestamped at or before as_of; a metric published "
    "after as_of is rejected upstream and never reaches the factor."
)


def _fundamental_feature(name: str, formula: str, feature_family: str) -> FeatureSpec:
    """A fundamental factor produced by core.score_engine._build_fundamental_score."""
    return FeatureSpec(
        name=name,
        owner="fundamental_agent",
        domain="fundamental",
        formula=formula,
        version=FUNDAMENTAL_FEATURE_VERSION,
        unit="score",
        frequency="per_session",
        lookback="1d",
        minimum_history=1,
        null_policy="default",
        pit_rule=_FUNDAMENTAL_PIT_RULE,
        source_dependencies=["alpha_vantage_overview"],
        feature_family=feature_family,
        model_compatibility=["technical_analysis", "linear", "logistic", "tree", "boosting"],
    )


# The PIT rule the contextual agents inherit. Each contract carries its own
# published_time and is filtered to as_of by its adapter before the value is
# produced; a contract published after as_of is rejected upstream.
_CONTEXTUAL_PIT_RULE = (
    "The agent's snapshot is filtered to evidence published at or before "
    "as_of before the signal is computed; a contract published after as_of "
    "is rejected upstream and never reaches the feature. A non-OK agent "
    "status yields no value at all rather than a neutral substitute."
)


def _contextual_feature(
    name: str,
    owner: str,
    domain: str,
    formula: str,
    version: str,
    unit: str,
    source_dependencies: list[str],
    feature_family: str,
    lookback: str = "1d",
    frequency: str = "per_session",
    minimum_history: int = 1,
) -> FeatureSpec:
    """A signal produced by one of the Sprint N contextual agents.

    null_policy is "exclude", never "default": these agents are governed by
    the fail-closed rule that a missing or non-OK contract must not become a
    neutral value (master context sections 9 and 16). A model that cannot
    obtain the feature loses the row, it does not receive a fabricated one.
    """
    return FeatureSpec(
        name=name,
        owner=owner,
        domain=domain,
        formula=formula,
        version=version,
        unit=unit,
        frequency=frequency,
        lookback=lookback,
        minimum_history=minimum_history,
        null_policy="exclude",
        pit_rule=_CONTEXTUAL_PIT_RULE,
        source_dependencies=source_dependencies,
        feature_family=feature_family,
        model_compatibility=["linear", "logistic", "tree", "boosting"],
    )


def build_default_registry() -> FeatureRegistry:
    """Build the canonical default registry (22 market + 5 fundamental + 4 contextual)."""
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

    # --- Fundamental factors (M1) ------------------------------------------------
    # The five 0-10 factors produced by core.score_engine._build_fundamental_features
    # and consumed by _build_fundamental_score (the ensemble's fundamental leg).
    # Exactly the features that enter the production model are registered — no
    # more, no less. Missing metrics degrade to the factor's explicit neutral
    # default (null_policy "default"); the fundamentals layer's own
    # source-status gating keeps degraded evidence out of actionable decisions.
    registry.register(_fundamental_feature(
        name="revenue_growth",
        formula=(
            "0-10 growth factor from the fundamentals snapshot: "
            "clamp_0_10(revenue_growth * 10.0); neutral 5.0 when "
            "revenue_growth is 0 or missing."
        ),
        feature_family="growth",
    ))
    registry.register(_fundamental_feature(
        name="margin_quality",
        formula=(
            "0-10 margin factor: clamp_0_10(gross_margins * 10.0); "
            "neutral 5.0 when gross_margins is 0 or missing."
        ),
        feature_family="quality",
    ))
    registry.register(_fundamental_feature(
        name="free_cash_flow_quality",
        formula=(
            "0-10 cash-flow factor: 8.0 when free_cash_flow > 0, else 3.0 "
            "(positive free cash flow supports balance-sheet flexibility)."
        ),
        feature_family="quality",
    ))
    registry.register(_fundamental_feature(
        name="balance_sheet_quality",
        formula=(
            "0-10 leverage factor: clamp_0_10(10.0 - debt_to_equity * 5.0); "
            "leverage above 2.0 scores 0."
        ),
        feature_family="quality",
    ))
    registry.register(_fundamental_feature(
        name="valuation_quality",
        formula=(
            "0-10 valuation factor: clamp_0_10(10.0 - (price_to_book - 2.0) * 1.5); "
            "price_to_book defaults to 4.0 when missing."
        ),
        feature_family="valuation",
    ))

    # --- Sprint N contextual agents (M1b) ------------------------------------
    # The signals that actually reach the published score from the news,
    # macro, regime and sentiment agents. Before these were registered, the
    # news and macro lines contributed 0.10 each to the ensemble while being
    # invisible to the M1 drift gate — the registry rule was enforced only on
    # the surface that happened to be registered. Exactly the values that
    # enter the production path are registered here; no aspirational entries.
    registry.register(_contextual_feature(
        name="news_sentiment_score",
        owner="news_agent",
        domain="news",
        formula=(
            "Aggregate tone of the credible, PIT-eligible article set in "
            "[-1, 1], from core.news_adapter.build_news_snapshot. Enters the "
            "ensemble as NEWS_SCORE_BASE + NEWS_SCORE_SPAN * sentiment_score. "
            "Produced only when the news contract status is OK; CONTRADICTORY "
            "never averages to neutral."
        ),
        version=NEWS_CONTRACT_VERSION,
        unit="ratio",
        source_dependencies=["newsapi_news"],
        feature_family="news_event",
    ))
    registry.register(_contextual_feature(
        name="macro_regime_score",
        owner="macro_agent",
        domain="macro",
        formula=(
            "Vintage-aware macro regime tilt in [0, 1] (0 risk-off, 0.5 "
            "neutral, 1 risk-on) from core.macro_adapter.build_macro_snapshot. "
            "Enters the ensemble as MACRO_SCORE_BASE + MACRO_SCORE_SPAN * "
            "regime_score. Series are resolved at their publication-time "
            "vintage (N3), never a later revision."
        ),
        version=MACRO_CONTRACT_VERSION,
        unit="ratio",
        source_dependencies=["fred_macro"],
        feature_family="macro",
    ))
    registry.register(_contextual_feature(
        name="regime_probability_proxy",
        owner="regime_agent",
        domain="regime",
        formula=(
            "Continuous proxy in [0, 1] for the five-state regime "
            "classification from core.regime_agent.build_regime_snapshot. "
            "Carries zero ensemble weight by design — the regime agent gates "
            "(STRESS forces NO_TRADE, RISK_OFF damps momentum) rather than "
            "votes — but is registered so a model may condition on it."
        ),
        version=REGIME_CONTRACT_VERSION,
        unit="ratio",
        lookback="283d",
        minimum_history=REGIME_REQUIRED_SESSIONS,
        source_dependencies=["yahoo_finance_chart"],
        feature_family="regime",
    ))
    registry.register(_contextual_feature(
        name="sentiment_score",
        owner="sentiment_agent",
        domain="sentiment",
        formula=(
            "Social/positioning sentiment in [-1, 1] from "
            "core.sentiment_contract.fetch_sentiment_snapshot. No legitimate "
            "provider is connected, so the contract is UNAVAILABLE and the "
            "feature yields no value; the only sanctioned derived path is "
            "derive_sentiment_from_news, which labels itself derived_from_news "
            "and scales confidence. Anti-proxying rule (N2): never inferred "
            "from RSI, price direction, or technical indicators."
        ),
        version=SENTIMENT_CONTRACT_VERSION,
        unit="ratio",
        source_dependencies=["sentiment_provider_unconfigured", "sentiment_derived_from_news"],
        feature_family="sentiment",
    ))

    return registry


def persist_feature_registry(
    registry: FeatureRegistry,
    path: str | Path | None = None,
) -> bool:
    """Persist the registry to append-only JSONL. Idempotent per registry hash.

    Fail-closed: an invalid registry (empty, invalid spec, or a producer that
    does not exist) is never committed to the store.
    """
    problems = registry.problems()
    if problems:
        raise ValueError(
            f"refusing to persist an invalid feature registry: {'; '.join(problems)}"
        )
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

