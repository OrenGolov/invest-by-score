"""Outcome label builder (Sprint V1) — leakage-safe, versioned, append-only.

Pipeline (every stage a pure, deterministic function; `build_outcome_labels`
wires them in order):

    FETCH -> ENTRY BAR (last bar <= as_of) -> WINDOW (as_of, as_of + h]
         -> LABELS -> STATUS -> (optional) APPEND-ONLY PERSISTENCE

Governance rules enforced here:

- Leakage safety: labels are computed strictly from bars in
  (as_of, as_of + h] — h trading sessions strictly after the as_of moment.
  The entry bar (last bar <= as_of) supplies only the base close. Bars
  beyond the horizon can never influence a horizon's label (test-pinned),
  and no feature/decision input is recomputed here.
- Boundary rule: a horizon's label is null until the horizon has fully
  elapsed RELATIVE TO THE DATA'S LATEST BAR — eligibility is derived from
  the fetched frame's newest bar, never from wall-clock, so the same
  dataset state always produces the same labels. Partially elapsed
  horizons are never emitted (no partial-window leakage) and never
  persisted.
- Single price truth: the same Yahoo-adjusted close series the scoring
  path uses; no separate adjustment, so research and production share one
  contract.
- Versioning: every record carries OUTCOME_LABEL_VERSION; changing the
  up-threshold or window definitions creates a new version and never
  rewrites history.
- Storage: append-only `data/outcomes.jsonl` keyed
  {ticker, as_of, horizon, label_version}. Re-appending a byte-identical
  latest record is a no-op (idempotent recompute); a genuinely different
  value for the same key is appended with `supersedes` pointing at the
  previous record hash — history is preserved, never mutated.
- Determinism: no wall-clock reads in the label math; `record_hash` and
  `labels_hash` are canonical SHA-256 digests (the latter is the future
  seed for dataset hashing in Sprint M2).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from core.config import (
    LABEL_CALENDAR_COVERAGE_DAYS,
    LABEL_HORIZON_SESSIONS,
    OUTCOME_LABEL_UP_THRESHOLD,
    OUTCOME_LABEL_VERSION,
)
from core.schemas import OutcomeLabelSet
from fetch_data import fetch_price_history

LOGGER = logging.getLogger("core.labels")

LABEL_SOURCE_ID = "yahoo_finance_chart"
OUTCOMES_PATH = Path(__file__).resolve().parent.parent / "data" / "outcomes.jsonl"


def _record_hash(record: dict) -> str:
    """Canonical SHA-256 over the record payload (excluding the hash itself)."""
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _labels_hash(records: dict[str, dict]) -> str:
    """Deterministic digest of the matured horizon payloads for one decision."""
    payload = {name: {k: v for k, v in record.items() if k != "record_hash"}
               for name, record in sorted(records.items())}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _label_fetch_period(days_back: int) -> str:
    """Deterministic provider period covering as_of plus the future windows.

    Wall-clock only selects how much history the provider returns; the
    eligibility math itself never reads the clock.
    """
    needed = int(days_back) + LABEL_CALENDAR_COVERAGE_DAYS
    if needed <= 365:
        return "1y"
    if needed <= 1825:
        return "5y"
    return "10y"


def _unavailable_set(ticker: str, as_of: str, reason: str) -> dict:
    return OutcomeLabelSet(
        ticker=ticker.upper(),
        as_of=as_of,
        status="UNAVAILABLE",
        label_version=OUTCOME_LABEL_VERSION,
        entry_bar=None,
        entry_close=None,
        latest_bar=None,
        horizons={},
        matured_horizons=[],
        pending_horizons=[],
        labels_hash="",
        reason=reason,
    ).to_dict()


def _horizon_record(
    ticker: str,
    as_of: str,
    horizon: str,
    sessions: int,
    entry_bar: str,
    entry_close: float,
    window: pd.DataFrame,
    latest_bar: str,
) -> dict:
    """Compute one matured horizon's labels strictly from the window bars."""
    window_closes = [float(value) for value in window["Close"]]
    exit_close = window_closes[-1]
    forward_return = exit_close / entry_close - 1.0

    # Realized volatility over the window: the h close-to-close returns that
    # realize inside (as_of, as_of + h], the first anchored at the entry
    # close. A single observation (1d) has no defined dispersion.
    returns = []
    previous = entry_close
    for close in window_closes:
        returns.append(close / previous - 1.0)
        previous = close
    realized_vol = None
    if len(returns) >= 2:
        series = pd.Series(returns)
        realized_vol = round(float(series.std(ddof=0)), 6)

    risk_adjusted = None

    record = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "horizon": horizon,
        "label_version": OUTCOME_LABEL_VERSION,
        "source_id": LABEL_SOURCE_ID,
        "status": "OK",
        "entry_bar": entry_bar,
        "entry_close": round(entry_close, 4),
        "exit_bar": window.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "window_sessions": sessions,
        "forward_return": round(forward_return, 6),
        "realized_vol": realized_vol,
        "risk_adjusted": risk_adjusted,
        "latest_bar_at_build": latest_bar,
    }
    if realized_vol not in (None, 0.0) and sessions >= 2:
        # Self-consistent: recomputable from the published numerator and
        # denominator alone (no hidden-precision ratio in the record).
        record["risk_adjusted"] = round(record["forward_return"] / record["realized_vol"], 4)
    if horizon == "20d":
        lows = [float(value) for value in window["Low"]]
        record["adverse_excursion"] = round(min(low / entry_close - 1.0 for low in lows), 6)
        record["label_up"] = bool(forward_return > OUTCOME_LABEL_UP_THRESHOLD)
    record["record_hash"] = _record_hash(record)
    return record


def build_outcome_labels(ticker: str, as_of: str, timestamp: str | None = None) -> dict:
    """Build the versioned, leakage-safe outcome label set for one decision.

    Pure with respect to the dataset state: eligibility derives from the
    fetched frame's latest bar, never wall-clock. A future as_of raises
    (timestamp violation, mirroring the market data agent); provider
    failures and missing entry bars are UNAVAILABLE statuses, never
    fabricated labels.
    """
    ticker = str(ticker).upper()
    as_of_text = pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")
    as_of_ts = pd.Timestamp(as_of_text)
    if as_of_ts > pd.Timestamp.now():
        raise ValueError(f"Requested as-of date {as_of_ts} is in the future.")

    days_back = (pd.Timestamp.now().normalize() - as_of_ts.normalize()).days
    period = _label_fetch_period(days_back)
    try:
        frame = fetch_price_history(ticker, period=period, interval="1d")
    except Exception as exc:
        LOGGER.warning("Outcome label fetch failed for %s: %s", ticker, exc)
        return _unavailable_set(ticker, as_of_text, f"Price history fetch failed; no labels can be computed. ({exc})")

    if frame is None or frame.empty or "Close" not in frame.columns:
        return _unavailable_set(ticker, as_of_text, "Provider returned no price history; no labels can be computed.")

    frame = frame.sort_index()
    eligible = frame[frame.index <= as_of_ts]
    if eligible.empty:
        return _unavailable_set(
            ticker, as_of_text,
            f"No bars at or before {as_of_text}; no entry close exists, so no labels can be computed.",
        )

    entry_position = len(eligible) - 1
    entry_bar = frame.index[entry_position].strftime("%Y-%m-%d %H:%M:%S")
    entry_close = float(frame["Close"].iloc[entry_position])
    bars_after = frame[frame.index > as_of_ts]
    latest_bar = frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")

    horizons: dict[str, dict | None] = {}
    matured: list[str] = []
    pending: list[str] = []
    for horizon, sessions in LABEL_HORIZON_SESSIONS.items():
        window = bars_after.iloc[:sessions]
        if len(window) < sessions:
            horizons[horizon] = None
            pending.append(horizon)
            continue
        horizons[horizon] = _horizon_record(
            ticker=ticker,
            as_of=as_of_text,
            horizon=horizon,
            sessions=sessions,
            entry_bar=entry_bar,
            entry_close=entry_close,
            window=window,
            latest_bar=latest_bar,
        )
        matured.append(horizon)

    if len(matured) == len(LABEL_HORIZON_SESSIONS):
        status = "OK"
        reason = (
            f"All {len(matured)} horizons matured from bars strictly after {as_of_text}; "
            f"entry close {entry_close:.4f} at {entry_bar}."
        )
    elif matured:
        status = "PARTIAL"
        reason = (
            f"{len(matured)} of {len(LABEL_HORIZON_SESSIONS)} horizons matured relative to the data's "
            f"latest bar ({latest_bar}); pending horizons are null — no partial-window leakage."
        )
    else:
        status = "PENDING"
        reason = (
            f"No horizon has fully elapsed relative to the data's latest bar ({latest_bar}); "
            f"all labels are null — no partial-window leakage."
        )

    matured_records = {name: horizons[name] for name in matured}
    return OutcomeLabelSet(
        ticker=ticker,
        as_of=as_of_text,
        status=status,
        label_version=OUTCOME_LABEL_VERSION,
        entry_bar=entry_bar,
        entry_close=round(entry_close, 4),
        latest_bar=latest_bar,
        horizons=horizons,
        matured_horizons=matured,
        pending_horizons=pending,
        labels_hash=_labels_hash(matured_records) if matured_records else "",
        reason=reason,
    ).to_dict()


def build_and_persist_outcome_labels(ticker: str, as_of: str, timestamp: str | None = None, path: str | Path | None = None) -> dict:
    """Build the label set and append every matured horizon to the store.

    Returns the snapshot with a `persisted` count (0 when every record was
    already present byte-identically — idempotent recompute).
    """
    snapshot = build_outcome_labels(ticker, as_of, timestamp)
    matured_records = [
        snapshot["horizons"][name] for name in snapshot.get("matured_horizons", [])
    ]
    snapshot["persisted"] = append_outcome_records(matured_records, path=path)
    return snapshot


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Outcomes store {path} line {line_number} is not valid JSON "
                    f"(append-only integrity violated): {exc}"
                ) from exc
    return records


def append_outcome_records(records: list[dict], path: str | Path | None = None) -> int:
    """Append matured label records; append-only with idempotent recompute.

    A record whose key {ticker, as_of, horizon, label_version} already ends
    with a byte-identical (same record_hash) latest entry is skipped. A
    genuinely different value for the same key is appended with `supersedes`
    pointing at the previous record hash — history is preserved, never
    mutated. Returns the number of records actually appended.
    """
    store_path = Path(path) if path is not None else OUTCOMES_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_records(store_path)
    latest_by_key: dict[tuple, dict] = {}
    for record in existing:
        key = (record.get("ticker"), record.get("as_of"), record.get("horizon"), record.get("label_version"))
        latest_by_key[key] = record

    appended = 0
    with store_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record.get("status") != "OK":
                # Partially elapsed horizons are never emitted (V1 boundary rule).
                continue
            key = (record.get("ticker"), record.get("as_of"), record.get("horizon"), record.get("label_version"))
            prior = latest_by_key.get(key)
            if prior is not None and prior.get("record_hash") == record.get("record_hash"):
                continue  # idempotent recompute: identical label, nothing to append
            enriched = dict(record)
            if prior is not None:
                enriched["supersedes"] = prior.get("record_hash")
            handle.write(json.dumps(enriched, sort_keys=True, default=str) + "\n")
            existing.append(enriched)
            latest_by_key[key] = enriched
            appended += 1
    return appended


def load_outcome_records(
    ticker: str | None = None,
    as_of: str | None = None,
    horizon: str | None = None,
    label_version: str | None = None,
    path: str | Path | None = None,
) -> list[dict]:
    """Read the append-only outcomes store in file order, optionally filtered."""
    store_path = Path(path) if path is not None else OUTCOMES_PATH
    records = _read_records(store_path)
    if ticker is not None:
        records = [r for r in records if r.get("ticker") == str(ticker).upper()]
    if as_of is not None:
        wanted = pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")
        records = [r for r in records if r.get("as_of") == wanted]
    if horizon is not None:
        records = [r for r in records if r.get("horizon") == horizon]
    if label_version is not None:
        records = [r for r in records if r.get("label_version") == label_version]
    return records


def latest_outcome_labels(ticker: str, as_of: str, path: str | Path | None = None) -> dict:
    """Resolve the latest persisted record per horizon for one decision.

    Returns {ticker, as_of, label_version, horizons, matured_horizons,
    pending_horizons, status, count}. Only matured horizons exist in the
    store; pending horizons resolve to None.
    """
    records = load_outcome_records(ticker=ticker, as_of=as_of, path=path)
    by_horizon: dict[str, dict] = {}
    for record in records:
        if record.get("label_version") != OUTCOME_LABEL_VERSION:
            continue
        by_horizon[record["horizon"]] = record  # file order -> last wins
    horizons = {
        horizon: by_horizon.get(horizon)
        for horizon in LABEL_HORIZON_SESSIONS
    }
    matured = [name for name, record in horizons.items() if record is not None]
    pending = [name for name, record in horizons.items() if record is None]
    if len(matured) == len(LABEL_HORIZON_SESSIONS):
        status = "OK"
    elif matured:
        status = "PARTIAL"
    else:
        status = "PENDING"
    return {
        "ticker": str(ticker).upper(),
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S"),
        "label_version": OUTCOME_LABEL_VERSION,
        "horizons": horizons,
        "matured_horizons": matured,
        "pending_horizons": pending,
        "status": status,
        "count": len(matured),
    }

