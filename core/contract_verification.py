"""Shared-contract verification (Sprint V5).

Binding rule (docs/validation.md, Shared Research Contract): research and
production consume the same feature contracts and the same scorers — no
parallel ad hoc path, no side research dataset, ever. This module is the
enforcement toolkit that keeps that rule true:

- `feature_contract_problems(snapshot)`: every feature contract in a market
  snapshot must carry complete provenance and the current
  MARKET_FEATURE_VERSION.
- `snapshot_field_problems(left, right)`: diff-precise comparison of the
  shared snapshot fields — used to prove the offline research seam changes
  only the DATA SOURCE, never the computation.
- `score_identity_problems(first, second)`: diff-precise comparison of the
  published score layer (headline scores, confidence, ensemble breakdown,
  governance) across the two paths.
- `expected_engine_versions()` / `engine_version_problems(...)`: the version
  set every backtest manifest must declare, read from the same config
  constants the live path uses — version drift between research and
  production is a failure, not a warning.
- `parallel_path_problems()`: structural source checks — the research
  package must CALL the canonical producers (`build_score`, the label
  builder) and must never define parallel implementations of them.

Pure and deterministic; every function returns a list of diff-precise
problem strings (empty = conforming) so callers can assert or gate. Tests
in `tests/test_shared_contracts.py` pin the invariants;
`.github/workflows/ci.yml` re-runs them on every push and pull request.
"""

from __future__ import annotations

from pathlib import Path

from core.config import (
    BACKTEST_STRATEGY_VERSION,
    CURRENT_SCORE_VERSION,
    ENSEMBLE_VERSION,
    FEATURE_REGISTRY_VERSION,
    FRAMING_VERSION,
    LONG_TERM_SCORE_VERSION,
    MARKET_FEATURE_VERSION,
    OUTCOME_LABEL_VERSION,
    UNIVERSE_VERSION,
)

SHARED_CONTRACT_VERSION = "shared-contract-verification-v1"

REQUIRED_PROVENANCE_FIELDS = (
    "name", "as_of", "source_id", "published_time", "calculation_version", "lookback_period",
)

# The snapshot surface research must not diverge from production on. The N6
# slotted features are included on purpose: registered-and-deferred still
# means the same value on both paths.
SHARED_SNAPSHOT_FIELDS = (
    "ticker", "as_of", "close", "volume", "avg_volume_20d", "volume_ratio_20d",
    "rsi", "volatility",
    "change_1d", "change_5d", "change_20d", "change_60d",
    "change_50d", "change_100d", "change_150d", "change_200d",
    "atr_14", "trend_slope_60d",
    "price_vs_ma_50", "price_vs_ma_100", "price_vs_ma_150", "price_vs_ma_200",
    "trend_vs_20d_mean", "market_regime",
)

# Research modules must never define these — they belong to the canonical
# producers (score engine, label builder, regime classifier, fundamentals).
_FORBIDDEN_DEFINITIONS = (
    "def build_score",
    "def _score_current_time",
    "def _score_long_term",
    "def _build_fundamental_score",
    "def build_outcome_labels",
    "def _horizon_record",
    "def classify_regime",
    "def evaluate_rules",
    "def _compute_confidence",
)

# The canonical producers the research engine must call by name.
_REQUIRED_ENGINE_CALLS = ("build_score(", "build_outcome_labels(")


def feature_contract_problems(snapshot: dict) -> list[str]:
    """Every feature contract must be complete and version-current.

    Sprint M1: also enforces the feature-registry contract — every feature
    must be registered and satisfy the PIT rule.
    """
    from core.feature_registry import build_default_registry, feature_contract_problems as _fr_problems
    features = snapshot.get("features") or {}
    if not features:
        return ["snapshot carries no feature contracts"]
    problems: list[str] = []
    for name in sorted(features):
        contract = features[name]
        if not isinstance(contract, dict):
            problems.append(f"{name}: contract is not a dict")
            continue
        if contract.get("name") != name:
            problems.append(f"{name}: contract name mismatch ({contract.get('name')!r})")
        if "value" not in contract:
            problems.append(f"{name}: contract carries no value key")
        for field in REQUIRED_PROVENANCE_FIELDS:
            value = contract.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                problems.append(f"{name}: provenance field {field} missing/empty")
        version = contract.get("calculation_version")
        if version != MARKET_FEATURE_VERSION:
            problems.append(
                f"{name}: calculation_version {version!r} != {MARKET_FEATURE_VERSION!r}"
            )
    # Sprint M1: enforce the feature-registry contract.
    registry = build_default_registry()
    problems.extend(_fr_problems(snapshot, registry))
    return problems


def feature_registry_problems(snapshot: dict) -> list[str]:
    """Enforce the feature-registry contract on a snapshot.

    Every feature must be registered, and every feature contract must satisfy
    the PIT rule (published_time <= as_of). This is the M1 binding rule: an
    unregistered feature cannot enter a production model.
    """
    from core.feature_registry import build_default_registry, feature_contract_problems as _fr_problems
    registry = build_default_registry()
    return _fr_problems(snapshot, registry)


# The value each Sprint N contextual agent contributes to the published
# score, and the registered feature it must conform to (M1b). Registering
# the features was only half the gap: nothing built a surface for the
# auditor to check, so news and macro reached the score unexamined.
_CONTEXTUAL_FEATURE_SOURCES = (
    ("news_sentiment_score", "news_snapshot", "sentiment_score"),
    ("macro_regime_score", "macro_snapshot", "regime_score"),
    ("regime_probability_proxy", "market_regime_snapshot", "probability_proxy"),
    ("sentiment_score", "sentiment_snapshot", "sentiment_score"),
)


def contextual_feature_surface(score_result) -> dict:
    """Build the feature surface for the Sprint N contextual agents.

    Only OK contracts carrying a usable value produce a feature. A non-OK
    agent yields no feature at all rather than a neutral one — the
    fail-closed rule these agents are governed by (null_policy "exclude").
    Returns `{}` when no contextual agent is live, which is the normal
    offline posture and is not itself a violation.
    """
    from core.feature_registry import build_default_registry

    registry = build_default_registry()
    surface: dict = {}
    for feature_name, snapshot_attr, value_key in _CONTEXTUAL_FEATURE_SOURCES:
        snapshot = getattr(score_result, snapshot_attr, None)
        if not isinstance(snapshot, dict):
            continue
        if str(snapshot.get("status", "UNAVAILABLE")) != "OK":
            continue
        value = snapshot.get(value_key)
        if value is None:
            continue
        spec = registry.get(feature_name)
        if spec is None:
            # Unregistered: emit the contract as-is so the gate rejects it
            # rather than silently dropping a live contribution.
            surface[feature_name] = {
                "name": feature_name,
                "value": value,
                "as_of": snapshot.get("as_of"),
                "source_id": snapshot.get("source_id"),
                "published_time": snapshot.get("published_time") or snapshot.get("as_of"),
                "calculation_version": snapshot.get("calculation_version"),
                "lookback_period": snapshot.get("lookback_period"),
            }
            continue
        surface[feature_name] = {
            "name": feature_name,
            "value": value,
            "as_of": snapshot.get("as_of"),
            "source_id": snapshot.get("source_id"),
            "published_time": snapshot.get("published_time") or snapshot.get("as_of"),
            "calculation_version": snapshot.get("calculation_version"),
            # The registered lookback is the DEFINITION's window. Some agents
            # (regime) report the sessions actually consumed, which varies per
            # run and could never equal a fixed registered value; the observed
            # count stays visible in that agent's own payload.
            "lookback_period": spec.lookback,
        }
    return surface


# The five registered FUNDAMENTAL features and the key each is published under
# in `fundamental_features`. Registered since M1 with `fundamental_agent` as
# producer; MEASURED, none had ever entered a training row.
# The only status meaning the fundamental numbers were OBSERVED rather than
# imputed from defaults.
_FUNDAMENTAL_LIVE_STATUSES = frozenset({"live_provider", "ok"})

# Each registered fundamental feature and the RAW metric it derives from.
#
# THE RAW INPUT IS THE GATE, NOT THE STATUS. MEASURED on VOO, `source_status` is
# "live_provider" while every entry in `valuation_metrics` is None, and the
# derived scores are still 5.0 / 5.0 / 7.0 / 3.0 / 10.0 — the default ladder in
# `_build_fundamental_features`. Those values are CONSTANT for every ticker
# without fundamentals, so admitting them would let a model learn "this row is an
# ETF" and let X5 call the family incremental on the strength of a survivorship
# marker.
_FUNDAMENTAL_FEATURE_INPUTS = (
    ("revenue_growth", "revenue_growth"),
    ("margin_quality", "gross_margin"),
    ("free_cash_flow_quality", "free_cash_flow"),
    ("balance_sheet_quality", "debt_to_equity"),
    ("valuation_quality", "price_to_book"),
)

# Raw inputs whose "value" is a substituted default that
# `_build_fundamental_features` then republishes as though observed. A feature
# derived from one of these is excluded until the provider returns a real number.
#
# MEASURED: price_to_book is 4.0 for VOO, NVDA and MSFT alike — three companies
# whose actual price-to-book differs by multiples. score_engine.py:502 reads it
# with `metrics.get("price_to_book", 4.0) or 4.0` and line 541 writes the result
# back into `valuation_metrics`, so provenance is lost in transit.
_FUNDAMENTAL_LAUNDERED_DEFAULTS = {"price_to_book": 4.0}
_FUNDAMENTAL_FEATURE_KEYS = tuple(name for name, _ in _FUNDAMENTAL_FEATURE_INPUTS)


def fundamental_feature_surface(score_result) -> dict:
    """Build the feature surface for the fundamental agent.

    A4. The fundamental scores were computed and published but never exposed as
    registry-conformant contracts, so they could not enter a dataset.

    **THE FAIL-CLOSED RULE IS LOAD-BEARING HERE, MORE THAN ELSEWHERE.**
    `_build_fundamental_features` coalesces missing inputs to defaults — a
    missing `revenue_growth` becomes a growth score of 5.0, a missing
    `price_to_book` becomes 4.0. Those are *imputed* numbers wearing a
    measurement's clothes, and admitting them would train a model on the
    imputation rather than on the business.

    So a feature is emitted ONLY when the fundamental source itself reports OK.
    When it does not, no feature is emitted at all and the row is excluded by the
    dataset builder's `TRAINING_REQUIRE_COMPLETE_FEATURES` rule — the same
    posture the contextual agents take.
    """
    from core.feature_registry import build_default_registry

    features = getattr(score_result, "fundamental_features", None)
    if not isinstance(features, dict):
        return {}
    # The fundamental layer is only actionable when its source passes. MEASURED,
    # the statuses this field takes are "live_provider" (the provider answered),
    # "provider_key_required" (no key configured) and "unknown". Only the first
    # means the numbers below are OBSERVATIONS; the others mean
    # `_build_fundamental_features` coalesced its defaults, and admitting those
    # would train a model on the imputation.
    if str(features.get("source_status", "unknown")).strip().lower() not in _FUNDAMENTAL_LIVE_STATUSES:
        return {}

    contract = features.get("source_contract")
    contract = contract if isinstance(contract, dict) else {}
    as_of = contract.get("as_of") or getattr(score_result, "as_of", None)
    published = contract.get("published_time") or as_of

    # THE PROVIDER'S OWN PAYLOAD, not the republished view.
    #
    # `_build_fundamental_features` applies DEFAULTS while reading
    # (`price_to_book` -> 4.0) and then writes those same defaults back into
    # `valuation_metrics`, so the republished view cannot distinguish an observed
    # 4.0 from an absent one. MEASURED, it reports price_to_book 4.0 and
    # valuation_quality 7.0 identically for VOO, NVDA and MSFT.
    #
    # A constant dressed as a measurement is worse than a missing value: it
    # survives every downstream null check, and a model trained on it learns the
    # default rather than the business. So the gate reads the raw snapshot the
    # provider returned.
    raw = getattr(score_result, "fundamental_snapshot", None)
    if isinstance(raw, dict):
        metrics = raw.get("valuation_metrics")
    else:
        metrics = features.get("valuation_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    registry = build_default_registry()
    surface: dict = {}
    for name, raw_key in _FUNDAMENTAL_FEATURE_INPUTS:
        value = features.get(name)
        if value is None:
            continue
        # THE IMPUTATION GUARD. A derived score whose raw input is absent is the
        # default ladder, not a measurement.
        raw_value = metrics.get(raw_key)
        if raw_value is None:
            continue
        # ...and a raw input equal to its own substituted default is
        # indistinguishable from an absent one, because the default is written
        # back into the published view.
        laundered = _FUNDAMENTAL_LAUNDERED_DEFAULTS.get(raw_key)
        if laundered is not None and float(raw_value) == float(laundered):
            continue
        spec = registry.get(name)
        surface[name] = {
            "name": name,
            "value": value,
            "as_of": as_of,
            "source_id": contract.get("source_id"),
            "published_time": published,
            "calculation_version": contract.get("calculation_version"),
            "lookback_period": spec.lookback if spec is not None else None,
        }
    return surface


def fundamental_feature_problems(score_result) -> list[str]:
    """Enforce the registry contract on the fundamental surface.

    An empty surface conforms — it means the fundamental source is not OK, which
    is the ordinary offline posture. What this refuses is a LIVE fundamental
    contribution that is unregistered, PIT-violating or version-drifted.
    """
    from core.feature_registry import (
        build_default_registry,
        feature_contract_problems as _fr_problems,
    )

    surface = fundamental_feature_surface(score_result)
    if not surface:
        return []
    return _fr_problems({"features": surface}, build_default_registry())


def contextual_feature_problems(score_result) -> list[str]:
    """Enforce the registry contract on the contextual agents' surface.

    An empty surface is conforming: it means no contextual agent is OK, which
    the agent-status taxonomy already governs. What this refuses is a LIVE
    contextual agent whose contribution is unregistered, PIT-violating,
    version-drifted, or from an undeclared source.
    """
    from core.feature_registry import build_default_registry, feature_contract_problems as _fr_problems

    surface = contextual_feature_surface(score_result)
    if not surface:
        return []
    return _fr_problems({"features": surface}, build_default_registry())


def snapshot_field_problems(left: dict, right: dict) -> list[str]:
    """Diff-precise comparison of the shared snapshot surface."""
    problems: list[str] = []
    for field in SHARED_SNAPSHOT_FIELDS:
        left_value = left.get(field)
        right_value = right.get(field)
        if left_value != right_value:
            problems.append(f"{field}: {left_value!r} != {right_value!r}")
    if left.get("moving_averages") != right.get("moving_averages"):
        problems.append("moving_averages differ")
    left_features = left.get("features") or {}
    right_features = right.get("features") or {}
    if set(left_features) != set(right_features):
        difference = sorted(set(left_features) ^ set(right_features))
        problems.append(f"feature contract keys differ: {difference}")
    else:
        for name in sorted(left_features):
            if left_features[name] != right_features[name]:
                problems.append(f"feature contract {name} differs")
    return problems


def score_identity_problems(first, second) -> list[str]:
    """Diff-precise comparison of the published score layer."""
    problems: list[str] = []
    for field in ("score", "current_time_score", "long_term_score", "confidence"):
        first_value = getattr(first, field, None)
        second_value = getattr(second, field, None)
        if first_value != second_value:
            problems.append(f"{field}: {first_value!r} != {second_value!r}")
    if first.ensemble_breakdown != second.ensemble_breakdown:
        problems.append("ensemble_breakdown differs")
    if first.governance != second.governance:
        problems.append("governance differs")
    return problems


def expected_engine_versions() -> dict:
    """The version set research manifests must declare, from one source.

    Read from the same config constants the live path stamps — the engine
    must never hardcode a copy.
    """
    return {
        "market_feature": MARKET_FEATURE_VERSION,
        "ensemble": ENSEMBLE_VERSION,
        "current_score": CURRENT_SCORE_VERSION,
        "long_term_score": LONG_TERM_SCORE_VERSION,
        "outcome_label": OUTCOME_LABEL_VERSION,
        "strategy": BACKTEST_STRATEGY_VERSION,
        "metrics": "backtest-metrics-v1",
        "universe": UNIVERSE_VERSION,
        "framing": FRAMING_VERSION,
        "feature_registry": FEATURE_REGISTRY_VERSION,
    }


def engine_version_problems(manifest_versions: dict) -> list[str]:
    """Version drift between a research run and the live constants."""
    expected = expected_engine_versions()
    problems: list[str] = []
    for key, value in expected.items():
        actual = (manifest_versions or {}).get(key)
        if actual != value:
            problems.append(f"versions[{key}]: {actual!r} != {value!r}")
    if "cost_table" not in (manifest_versions or {}):
        problems.append("versions[cost_table] missing — the cost table version must be declared")
    return problems


def parallel_path_problems() -> list[str]:
    """Structural check: research must call, never re-implement, production.

    Walks `core/backtest/*.py` for parallel definitions of canonical
    producers, and requires the engine to call `build_score` and the label
    builder by name.
    """
    problems: list[str] = []
    backtest_dir = Path(__file__).resolve().parent / "backtest"
    for module_path in sorted(backtest_dir.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for forbidden in _FORBIDDEN_DEFINITIONS:
            if forbidden in source:
                problems.append(
                    f"{module_path.name}: defines {forbidden!r} — a parallel "
                    f"implementation of a production contract is forbidden"
                )
    engine_source = (backtest_dir / "engine.py").read_text(encoding="utf-8")
    for required_call in _REQUIRED_ENGINE_CALLS:
        if required_call not in engine_source:
            problems.append(
                f"engine.py no longer calls the canonical producer {required_call}"
            )
    return problems