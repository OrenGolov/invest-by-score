"""Official training dataset builder (Sprint M2) — the only dataset generator.

Every training row is the same shape, and nothing else may build one:

    prediction_time
        -> information available at prediction_time
        -> features
        -> future outcome

There is exactly one door into a training set, and it is this module. A
"quick" side extractor that pulls features some other way is the classic
route to leakage, so the parallel-path rule that governs research
(`core.contract_verification.parallel_path_problems`) applies here too: this
builder CALLS the canonical producers and never reimplements them.

- Features come from `core.score_engine.build_score` — the live scoring path,
  replayed through `core.backtest.engine.offline_replay_seam`. The dataset
  therefore sees exactly what production saw.
- Outcomes come from `core.labels.build_outcome_labels` — the V1
  leakage-safe label builder, not a locally computed forward return.
- Admission is gated by the M1 canonical feature registry: an unregistered,
  PIT-violating, version-drifted or undeclared-source feature cannot enter a
  dataset any more than it can enter a production model.

Binding rules, all test-enforced:

- **Leakage is structurally impossible, not merely avoided.** A row's
  features are built at `prediction_time` from bars at or before it; its
  outcome is read from the label builder, which computes strictly forward
  from the same instant. The two never share a code path, and
  `row_problems` re-verifies every feature contract's `published_time <=
  prediction_time` before the row is admitted.
- **An unmatured horizon is an excluded row, never a zero.** Rows whose
  target horizon has not fully realized are dropped and counted in the build
  report under `excluded`. Coverage is never silently thinned.
- **Missing features exclude the row.** No imputation, no neutral fill (see
  master context section 32). The exclusion reason is recorded.
- **Deterministic dataset hash.** `dataset_hash` is a canonical SHA-256 over
  every row's identity plus the feature-set definition and label version, so
  two datasets with any difference in data, features, or label semantics can
  never share a hash. Rebuilding the same request reproduces it exactly.
- **Append-only persistence**, idempotent per dataset hash, matching the
  integrity model used by outcomes / manifests / framing / the registry.

Pure and deterministic apart from the provider reads, which the replay seam
serves from preloaded frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.config import (
    ENSEMBLE_VERSION,
    FEATURE_REGISTRY_VERSION,
    LABEL_HORIZON_SESSIONS,
    MARKET_FEATURE_VERSION,
    OUTCOME_LABEL_VERSION,
    TRAINING_DATASET_SCHEMA_VERSION,
    TRAINING_DATASET_VERSION,
    TRAINING_DEFAULT_TARGET_HORIZON,
    TRAINING_REQUIRE_COMPLETE_FEATURES,
)
from core.feature_registry import (
    FeatureRegistry,
    build_default_registry,
    feature_contract_problems,
    feature_set_hash,
    model_feature_problems,
)

TRAINING_DATASET_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "training_datasets.jsonl"
)


class TrainingDatasetError(ValueError):
    """Raised when a dataset request or a built row violates the contract."""


def context_snapshot_of(score_result: Any) -> dict[str, Any]:
    """The per-observation context available in a point-in-time score result.

    A1. Pulled from the SCORE RESULT rather than recomputed, so the context a
    training row records is the same context the live path saw at that
    timestamp. Recomputing it later would risk using information published
    after `prediction_time`, which is the one thing a PIT system may not do.
    """
    payload = score_result if isinstance(score_result, dict) else getattr(
        score_result, "__dict__", {}
    )
    return {
        "regime": payload.get("market_regime_snapshot") or {},
        "news": payload.get("news_snapshot") or {},
        "sentiment": payload.get("sentiment_snapshot") or {},
    }


def observed_regime(context_snapshot: dict | None) -> str | None:
    """The regime label at prediction time, or None when none was classified.

    None, NEVER a default label. A regime of "unknown" would become its own
    bucket in X3's per-regime comparison, and a bucket of unclassified
    observations tells you nothing about how the model behaves in a regime.
    """
    regime = ((context_snapshot or {}).get("regime") or {})
    if not isinstance(regime, dict):
        return None
    label = str(regime.get("regime") or "").strip()
    if not label:
        return None
    # A regime snapshot that failed still carries a status; only a classified
    # one counts.
    status = str(regime.get("status") or "").strip().upper()
    if status and status not in ("OK", "PRESENT"):
        return None
    return label.lower()


def observed_event(context_snapshot: dict | None) -> str | None:
    """The dominant news event id at prediction time, or None.

    X4 leaves one event out at a time, so it needs an id that identifies WHICH
    event — not a count and not a polarity. When the provider was unavailable
    there is no event, and that must read as absent rather than as "no news",
    which are different facts: the first is ignorance, the second is evidence.
    """
    news = ((context_snapshot or {}).get("news") or {})
    if not isinstance(news, dict):
        return None
    if str(news.get("status") or "").strip().upper() != "OK":
        return None
    articles = news.get("articles")
    if not isinstance(articles, (list, tuple)) or not articles:
        return None
    first = articles[0] if isinstance(articles[0], dict) else {}
    for key in ("event_id", "id", "url", "title"):
        value = str(first.get(key) or "").strip()
        if value:
            return value
    return None


def observed_source(context_snapshot: dict | None) -> str | None:
    """The news source id at prediction time, or None.

    The SOURCE axis of X4, and the join L4 needs. Recorded only when the
    provider actually answered: `source_id` is present on an UNAVAILABLE
    snapshot too (it names who failed), and recording that would make every
    failed fetch look like an observation from that source.
    """
    news = ((context_snapshot or {}).get("news") or {})
    if not isinstance(news, dict):
        return None
    if str(news.get("status") or "").strip().upper() != "OK":
        return None
    articles = news.get("articles")
    if isinstance(articles, (list, tuple)) and articles:
        first = articles[0] if isinstance(articles[0], dict) else {}
        for key in ("source_id", "source", "source_name"):
            value = first.get(key)
            if isinstance(value, dict):
                value = value.get("id") or value.get("name")
            value = str(value or "").strip()
            if value:
                return value
    value = str(news.get("source_id") or "").strip()
    return value or None


@dataclass
class TrainingRow:
    """One supervised example. Immutable once built."""

    ticker: str
    prediction_time: str
    features: dict[str, float]
    feature_contracts: dict[str, dict]
    target_horizon: str
    forward_return: float
    label_up: bool | None
    realized_vol: float | None
    adverse_excursion: float | None
    label_version: str
    label_record_hash: str
    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION
    # A1: PER-OBSERVATION CONTEXT, for the robustness gates.
    #
    # X3 could not compare per-regime performance, X4 had nothing to leave out
    # in a leave-one-out test, and neither could be fixed by a better estimator
    # — the information was never recorded. These fields carry it from the
    # point-in-time score result that produced the row.
    #
    # DELIBERATELY OUTSIDE `identity()`. The dataset hash is M8's
    # reproducibility anchor; adding fields to it would invalidate every
    # existing `dataset_hash` and every model artifact keyed to one. Context
    # describes a row's provenance, it does not define which example the row
    # IS, so two rows differing only in recorded context are the same training
    # example and must hash alike.
    #
    # None means NOT OBSERVED, never a default. A regime of "unknown" or a
    # source of "" would be a synthetic bucket, and X4's module refuses derived
    # attribution for exactly that reason: bucketing every observation under
    # one label leaves leave-one-out nothing to leave, and the gate would
    # return ROBUST having tested nothing.
    regime: str | None = None
    event_id: str | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def context(self) -> dict[str, Any]:
        """The per-observation context, with absences explicit.

        Separate from `to_dict` so a consumer asking "what do we know about the
        conditions of this observation" gets only that, and can tell an absence
        from a value.
        """
        return {
            "regime": self.regime,
            "event_id": self.event_id,
            "source_id": self.source_id,
        }

    def identity(self) -> dict[str, Any]:
        """The fields that define this row for hashing purposes.

        Feature CONTRACTS are included, not just values: two rows with the
        same number produced under different calculation versions or from
        different sources are different training data.

        CONTEXT IS EXCLUDED — see the note on the context fields above. It is
        provenance, not identity, and including it would break every existing
        dataset hash.
        """
        return {
            "ticker": self.ticker,
            "prediction_time": self.prediction_time,
            "target_horizon": self.target_horizon,
            "features": {name: self.features[name] for name in sorted(self.features)},
            "feature_contracts": {
                name: {
                    "calculation_version": contract.get("calculation_version"),
                    "lookback_period": contract.get("lookback_period"),
                    "published_time": contract.get("published_time"),
                    "source_id": contract.get("source_id"),
                }
                for name, contract in sorted((self.feature_contracts or {}).items())
            },
            "forward_return": self.forward_return,
            "label_up": self.label_up,
            "label_version": self.label_version,
            "label_record_hash": self.label_record_hash,
        }


@dataclass
class TrainingDataset:
    """A built dataset plus the provenance needed to reproduce it."""

    rows: list[TrainingRow] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    target_horizon: str = TRAINING_DEFAULT_TARGET_HORIZON
    excluded: list[dict[str, str]] = field(default_factory=list)
    dataset_version: str = TRAINING_DATASET_VERSION
    dataset_hash: str = ""
    feature_set_hash: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    # V6 survivorship verdict for the tickers this dataset was built from.
    # A single-ticker dataset cannot be survivorship-biased in the usual
    # sense, but a MULTI-ticker one silently can: every name someone types
    # today is a name that survived to be typed.
    survivorship: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "feature_set_hash": self.feature_set_hash,
            "target_horizon": self.target_horizon,
            "feature_names": list(self.feature_names),
            "row_count": len(self.rows),
            "excluded_count": len(self.excluded),
            "excluded": list(self.excluded),
            "versions": dict(self.versions),
            "rows": [row.to_dict() for row in self.rows],
            "survivorship": dict(self.survivorship),
        }

    def report(self) -> dict[str, Any]:
        """The build report — coverage without the row payload."""
        reasons: dict[str, int] = {}
        for entry in self.excluded:
            reason = str(entry.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "feature_set_hash": self.feature_set_hash,
            "target_horizon": self.target_horizon,
            "feature_names": list(self.feature_names),
            "row_count": len(self.rows),
            "excluded_count": len(self.excluded),
            "exclusion_reasons": dict(sorted(reasons.items())),
            "versions": dict(self.versions),
            "survivorship": dict(self.survivorship),
        }


def dataset_hash(rows: Iterable[TrainingRow], feature_set_digest: str, target_horizon: str) -> str:
    """Deterministic identity for a dataset.

    Rows are sorted by (prediction_time, ticker) so build order cannot change
    the hash, while any difference in data, feature definitions, label
    semantics or target horizon must.
    """
    payload = {
        "dataset_version": TRAINING_DATASET_VERSION,
        "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "feature_set_hash": feature_set_digest,
        "target_horizon": target_horizon,
        "label_version": OUTCOME_LABEL_VERSION,
        "rows": sorted(
            (row.identity() for row in rows),
            key=lambda entry: (entry["prediction_time"], entry["ticker"]),
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def row_problems(
    row: TrainingRow,
    registry: FeatureRegistry | None = None,
) -> list[str]:
    """Verify one row against the registry and the leakage rule.

    Re-checks what the builder already enforced. That redundancy is
    deliberate: a row loaded back from disk, or built by a future caller,
    gets the same gate as a freshly built one.
    """
    active = registry if registry is not None else build_default_registry()
    problems: list[str] = []

    if not row.ticker:
        problems.append("row carries no ticker")
    if not row.prediction_time:
        problems.append("row carries no prediction_time")
    if row.target_horizon not in LABEL_HORIZON_SESSIONS:
        problems.append(
            f"target_horizon {row.target_horizon!r} is not a declared label horizon "
            f"(known: {sorted(LABEL_HORIZON_SESSIONS)})"
        )
    if not row.features:
        problems.append("row carries no features — an empty row is not training data")
    if row.label_version != OUTCOME_LABEL_VERSION:
        problems.append(
            f"label_version {row.label_version!r} != {OUTCOME_LABEL_VERSION!r}"
        )
    if not row.label_record_hash:
        problems.append("row carries no label_record_hash — the outcome is untraceable")

    # Registry conformance: unregistered / PIT-violating / drifted contracts.
    problems.extend(feature_contract_problems({"features": row.feature_contracts}, active))

    # The leakage rule, stated independently of the registry check: no feature
    # may be published after the instant the prediction was made.
    if row.prediction_time:
        try:
            prediction_ts = pd.Timestamp(row.prediction_time)
        except (ValueError, TypeError):
            problems.append(f"prediction_time {row.prediction_time!r} is not a timestamp")
        else:
            for name in sorted(row.feature_contracts or {}):
                published = (row.feature_contracts[name] or {}).get("published_time")
                if published is None:
                    continue
                try:
                    published_ts = pd.Timestamp(published)
                except (ValueError, TypeError):
                    problems.append(f"{name}: published_time {published!r} is not a timestamp")
                    continue
                if published_ts > prediction_ts:
                    problems.append(
                        f"{name}: published_time {published} is after prediction_time "
                        f"{row.prediction_time} — future information in a training row"
                    )

    # Every declared feature must have a contract behind it.
    for name in sorted(row.features):
        if name not in (row.feature_contracts or {}):
            problems.append(f"{name}: feature value has no contract behind it")
    return problems


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN check


def build_training_row(
    ticker: str,
    prediction_time: str,
    feature_surface: dict,
    label_set: dict,
    feature_names: Iterable[str],
    target_horizon: str = TRAINING_DEFAULT_TARGET_HORIZON,
    context_snapshot: dict | None = None,
) -> tuple[TrainingRow | None, str]:
    """Assemble one row, or explain why it cannot exist.

    Returns `(row, "")` on success and `(None, reason)` on exclusion. The
    reason is recorded in the build report — a dropped row always says why.
    """
    horizons = (label_set or {}).get("horizons") or {}
    outcome = horizons.get(target_horizon)
    if not isinstance(outcome, dict):
        return None, "label_horizon_missing"
    if outcome.get("status") != "OK":
        return None, f"label_{str(outcome.get('status', 'unavailable')).lower()}"

    forward_return = _numeric(outcome.get("forward_return"))
    if forward_return is None:
        return None, "label_forward_return_unusable"

    features: dict[str, float] = {}
    contracts: dict[str, dict] = {}
    for name in sorted({str(entry) for entry in feature_names}):
        contract = (feature_surface or {}).get(name)
        if not isinstance(contract, dict):
            if TRAINING_REQUIRE_COMPLETE_FEATURES:
                return None, "feature_contract_missing"
            continue
        value = _numeric(contract.get("value"))
        if value is None:
            if TRAINING_REQUIRE_COMPLETE_FEATURES:
                return None, "feature_value_missing"
            continue
        features[name] = value
        contracts[name] = contract

    if not features:
        return None, "no_usable_features"

    row = TrainingRow(
        ticker=str(ticker).upper(),
        prediction_time=str(prediction_time),
        features=features,
        feature_contracts=contracts,
        target_horizon=target_horizon,
        forward_return=forward_return,
        label_up=outcome.get("label_up"),
        realized_vol=_numeric(outcome.get("realized_vol")),
        adverse_excursion=_numeric(outcome.get("adverse_excursion")),
        label_version=str(label_set.get("label_version", OUTCOME_LABEL_VERSION)),
        label_record_hash=str(outcome.get("record_hash", "")),
        regime=observed_regime(context_snapshot),
        event_id=observed_event(context_snapshot),
        source_id=observed_source(context_snapshot),
    )
    return row, ""


def build_training_dataset(
    prediction_times_by_ticker: dict[str, list[str]],
    history_by_ticker: dict,
    feature_names: Iterable[str] | None = None,
    target_horizon: str = TRAINING_DEFAULT_TARGET_HORIZON,
    registry: FeatureRegistry | None = None,
    model_family: str | None = None,
    require_survivorship_safe: bool = False,
) -> TrainingDataset:
    """Build the official training dataset. The only sanctioned generator.

    For every (ticker, prediction_time) the live scoring path is replayed
    offline to produce the feature surface, and the V1 label builder produces
    the forward outcome. Rows that cannot be built are excluded with a
    recorded reason, never imputed.
    """
    # Imported here: core.backtest imports core.labels and core.score_engine,
    # and importing it at module scope makes the dependency cycle explicit
    # rather than load-order-dependent.
    from core.backtest.engine import _exposed_feature_surface, offline_replay_seam
    from core.contract_verification import (
        contextual_feature_surface,
        fundamental_feature_surface,
    )
    from core.labels import build_outcome_labels
    from core.score_engine import CURRENT_SCORE_FEATURES, LONG_TERM_SCORE_FEATURES, build_score

    active = registry if registry is not None else build_default_registry()

    if target_horizon not in LABEL_HORIZON_SESSIONS:
        raise TrainingDatasetError(
            f"target_horizon {target_horizon!r} is not a declared label horizon "
            f"(known: {sorted(LABEL_HORIZON_SESSIONS)})"
        )

    declared = (
        list(feature_names)
        if feature_names is not None
        else sorted({*CURRENT_SCORE_FEATURES, *LONG_TERM_SCORE_FEATURES})
    )
    # M1 gate: an unregistered feature cannot enter a dataset, exactly as it
    # cannot enter a production model.
    gate_problems = model_feature_problems(declared, active, model_family)
    if gate_problems:
        raise TrainingDatasetError(
            "requested feature set violates the canonical feature registry: "
            + "; ".join(gate_problems)
        )

    rows: list[TrainingRow] = []
    excluded: list[dict[str, str]] = []

    with offline_replay_seam(history_by_ticker):
        for ticker in sorted(prediction_times_by_ticker):
            for prediction_time in sorted(prediction_times_by_ticker[ticker]):
                try:
                    score_result = build_score(ticker, prediction_time, persist_audit=False)
                except Exception as exc:
                    # ABSORBS: any failure replaying the live scoring path at
                    # this timestamp — a thin history, a provider gap, a feature
                    # that cannot be computed. The row is EXCLUDED with a named
                    # reason rather than built from partial features: M2 requires
                    # every row to be complete, and the exclusion report is how a
                    # reader learns how many rows were lost and to what.
                    excluded.append({
                        "ticker": str(ticker).upper(),
                        "prediction_time": str(prediction_time),
                        "reason": "score_build_failed",
                        "detail": str(exc),
                    })
                    continue

                # A4: THE TECHNICAL SURFACE IS NOT THE WHOLE SURFACE.
                #
                # `_exposed_feature_surface` returns only what the current-time
                # and long-term SCORE views consumed — 16 price/volume
                # derivatives. MEASURED, 24 of the 40 registered features never
                # reached a model, and among them was every fundamental, news,
                # sentiment and macro feature. The project's central claim, that
                # this evidence predicts returns, had never been tested because
                # the features were never in a training row.
                #
                # `contextual_feature_surface` already built exactly these
                # contracts, fail-closed, for the AUDITOR to check (M1b). It was
                # simply never merged in here. A non-OK agent yields no feature
                # rather than a neutral one, so an unavailable provider still
                # cannot fabricate evidence.
                surface = _exposed_feature_surface(score_result)
                surface.update(contextual_feature_surface(score_result))
                surface.update(fundamental_feature_surface(score_result))
                try:
                    label_set = build_outcome_labels(ticker, prediction_time)
                except Exception as exc:
                    # ABSORBS: any failure building the outcome labels — usually
                    # a horizon that has not matured yet. Excluded with its
                    # reason, never labelled zero: a forward return of 0.0 is a
                    # fabricated outcome, and V1 refuses to emit one.
                    excluded.append({
                        "ticker": str(ticker).upper(),
                        "prediction_time": str(prediction_time),
                        "reason": "label_build_failed",
                        "detail": str(exc),
                    })
                    continue

                row, reason = build_training_row(
                    ticker,
                    prediction_time,
                    surface,
                    label_set,
                    declared,
                    target_horizon,
                    # A1: the context the LIVE path saw at this timestamp, so
                    # the recorded regime/event/source is PIT-correct by
                    # construction rather than by a later lookup.
                    context_snapshot=context_snapshot_of(score_result),
                )
                if row is None:
                    excluded.append({
                        "ticker": str(ticker).upper(),
                        "prediction_time": str(prediction_time),
                        "reason": reason,
                    })
                    continue

                problems = row_problems(row, active)
                if problems:
                    excluded.append({
                        "ticker": str(ticker).upper(),
                        "prediction_time": str(prediction_time),
                        "reason": "row_not_conformant",
                        "detail": "; ".join(problems[:3]),
                    })
                    continue
                rows.append(row)

    rows.sort(key=lambda entry: (entry.prediction_time, entry.ticker))
    feature_digest = feature_set_hash(declared, active)
    survivorship = _survivorship_verdict(
        sorted(prediction_times_by_ticker), rows, require_survivorship_safe
    )
    return TrainingDataset(
        rows=rows,
        feature_names=sorted({str(name) for name in declared}),
        target_horizon=target_horizon,
        excluded=excluded,
        dataset_hash=dataset_hash(rows, feature_digest, target_horizon),
        feature_set_hash=feature_digest,
        survivorship=survivorship,
        versions={
            "dataset": TRAINING_DATASET_VERSION,
            "schema": TRAINING_DATASET_SCHEMA_VERSION,
            "market_feature": MARKET_FEATURE_VERSION,
            "outcome_label": OUTCOME_LABEL_VERSION,
            "feature_registry": FEATURE_REGISTRY_VERSION,
            "ensemble": ENSEMBLE_VERSION,
        },
    )


def _survivorship_verdict(
    tickers: list[str],
    rows: list[TrainingRow],
    require_safe: bool,
) -> dict[str, Any]:
    """Judge a dataset's ticker set against the V6 point-in-time ledger.

    The backtest engine already enforces this; the dataset builder did not,
    so a MULTI-ticker training set could silently inherit survivorship bias.
    Every ticker someone types today is one that survived to be typed, and a
    model trained only on survivors learns that companies do not fail.

    Single-ticker datasets are reported `single_ticker`: the bias is a
    property of universe CONSTRUCTION, and a one-name set makes no claim
    about a universe. Fail-closed when `require_safe` is set.
    """
    from core.universe import (
        STATUS_POINT_IN_TIME_COMPLETE,
        latest_entries_by_ticker,
        universe_survivorship_status,
    )

    if len(tickers) <= 1:
        return {
            "status": "single_ticker",
            "tickers": len(tickers),
            "detail": (
                "a one-ticker dataset makes no universe claim, so survivorship "
                "bias does not apply; it does apply as soon as a second ticker "
                "is added"
            ),
        }

    earliest = rows[0].prediction_time if rows else None
    entries = latest_entries_by_ticker()
    if not entries:
        verdict = {
            "status": "unverifiable",
            "tickers": len(tickers),
            "detail": (
                "the point-in-time universe ledger (data/universe.jsonl) is "
                "empty, so this ticker set cannot be checked for survivorship "
                "bias — seed it with seed_portfolio_universe()"
            ),
        }
    else:
        status = universe_survivorship_status(list(tickers), entries, earliest)
        verdict = {
            "status": status,
            "tickers": len(tickers),
            "as_of": earliest,
            "detail": (
                "ticker set is point-in-time complete at the earliest "
                "prediction time"
                if status == STATUS_POINT_IN_TIME_COMPLETE
                else f"ticker set is {status} against the universe ledger"
            ),
        }

    if require_safe and verdict["status"] != STATUS_POINT_IN_TIME_COMPLETE:
        raise TrainingDatasetError(
            f"refusing to build a multi-ticker dataset that is not "
            f"survivorship-safe: {verdict['detail']}"
        )
    return verdict


def persist_training_dataset(
    dataset: TrainingDataset,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a dataset's manifest, idempotent per dataset hash.

    The row payload is not written here — this is the provenance record a
    trial cites. Re-persisting the same dataset is a no-op, so a rebuild can
    never fork the ledger.
    """
    store = Path(path) if path is not None else TRAINING_DATASET_STORE_PATH
    if not dataset.dataset_hash:
        raise TrainingDatasetError("dataset carries no dataset_hash — refusing to persist")

    for existing in load_training_datasets(store):
        if existing.get("dataset_hash") == dataset.dataset_hash:
            return existing

    record = dataset.report()
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def load_training_datasets(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read persisted dataset manifests. Malformed lines raise (integrity is loud)."""
    store = Path(path) if path is not None else TRAINING_DATASET_STORE_PATH
    if not store.exists():
        return []
    records: list[dict[str, Any]] = []
    with store.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise TrainingDatasetError(
                    f"{store.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return records
