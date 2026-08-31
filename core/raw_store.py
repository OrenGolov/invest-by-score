"""Append-only raw-record store (W6) — the system's provenance ledger.

Every provider fetch records exactly what it saw, before any cache write:
one JSON line per fetch under `data/raw/{source_id}/{YYYY-MM-DD}.jsonl`:

    {"ingested_time": ..., "source_id": ..., "request_key": ...,
     "payload_sha256": ..., "schema_version": ..., "records": [...]}

Rules:
- Append-only: a re-fetch of the same bar/field appends a new version line;
  nothing is ever mutated or deleted.
- Supersede-on-read: `load_raw_records(record_key_field=...)` keeps every
  version and marks older ones with `superseded_by` (the winning payload
  hash) — mirrors the data-model doc's versioning rule.
- Fail-soft: a store failure logs `raw_store_write_failed` and returns
  False; it must never corrupt scoring (degraded-but-flagged is
  acceptable). `rebuild_price_frame` reconstructs the market history from
  raw records alone, which is the acceptance proof for raw immutability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, date, timezone
from pathlib import Path

import pandas as pd

RAW_STORE_SCHEMA_VERSION = "raw-store-v1"
RAW_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LOGGER = logging.getLogger("core.raw_store")


def _records_digest(records: list[dict]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _store_dir(source_id: str) -> Path:
    return RAW_STORE_DIR / source_id


def append_raw_records(
    source_id: str,
    request_key: str,
    records: list[dict],
    schema_version: str = RAW_STORE_SCHEMA_VERSION,
) -> bool:
    """Append one fetch payload to the raw ledger. Never raises.

    Returns True on success; on any failure logs `raw_store_write_failed`
    and returns False so adapters can keep scoring (degraded-but-flagged).
    """
    line = {
        "ingested_time": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "request_key": request_key,
        "payload_sha256": _records_digest(records),
        "schema_version": schema_version,
        "records": records,
    }
    try:
        directory = _store_dir(source_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{date.today().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True, default=str) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("raw_store_write_failed: %s", exc)
        return False


def load_raw_records(
    source_id: str,
    request_key: str,
    record_key_field: str | None = None,
) -> list[dict]:
    """Read every version of every record for a request key.

    Returns entries shaped `{"ingested_time", "payload_sha256", "record"}`.
    When `record_key_field` is given (e.g. "bar_time"), older versions are
    marked `superseded_by: <winning payload hash>` — both versions remain
    in the output, per the append-only versioning rule. Malformed lines are
    skipped; a missing store yields an empty list.
    """
    entries: list[dict] = []
    directory = _store_dir(source_id)
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            LOGGER.warning("raw_store_read_failed for %s: %s", path, exc)
            continue
        for raw_line in lines:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if line.get("request_key") != request_key:
                continue
            for record in line.get("records", []):
                entries.append({
                    "ingested_time": line.get("ingested_time", ""),
                    "payload_sha256": line.get("payload_sha256", ""),
                    "record": record,
                })

    if record_key_field:
        # Append order == read order (sorted files, in-order lines), so the
        # LAST occurrence of a record key is the newest version regardless
        # of clock granularity; earlier ones get superseded_by its hash.
        last_index_by_key: dict[str, int] = {}
        for index, entry in enumerate(entries):
            key = str(entry["record"].get(record_key_field))
            last_index_by_key[key] = index
        for index, entry in enumerate(entries):
            key = str(entry["record"].get(record_key_field))
            if index != last_index_by_key[key]:
                entry["superseded_by"] = entries[last_index_by_key[key]]["payload_sha256"]
    return entries


def rebuild_price_frame(source_id: str, request_key: str) -> pd.DataFrame | None:
    """Reconstruct the market OHLCV frame purely from raw records.

    Uses the newest non-superseded version of every bar. Returns None when
    the raw store holds no usable records for the request key.
    """
    entries = load_raw_records(source_id, request_key, record_key_field="bar_time")
    rows = [entry["record"] for entry in entries if not entry.get("superseded_by")]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    # Match the market-data index resolution (Yahoo timestamps land at ms
    # precision in the cache; pandas defaults string parsing to us).
    frame["bar_time"] = pd.to_datetime(frame["bar_time"]).astype("datetime64[ms]")
    frame = frame.sort_values("bar_time")
    frame.index = pd.DatetimeIndex(frame["bar_time"], name="Date")
    frame = frame.drop(columns=["bar_time"])
    # Canonical market column order (JSONL storage alphabetizes keys).
    frame = frame[["Open", "High", "Low", "Close", "Volume"]]
    frame = frame.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
    return frame
