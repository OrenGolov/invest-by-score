from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.audit_policy import stable_hash

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_audit.jsonl"
AUDIT_SCHEMA_VERSION = "audit-event-v2"


def _read_audit_events() -> list[dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                try:
                    events.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    return events


def persist_decision_audit(decision: dict[str, Any]) -> dict[str, Any]:
    """Persist a score decision to a lightweight append-only audit log."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "event_id": uuid.uuid4().hex,
        "event_type": "score_decision",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": str(decision.get("ticker", "UNKNOWN")).upper(),
        "as_of": str(decision.get("as_of", "")),
        "mode": str(decision.get("mode", "ANALYSIS_ONLY")),
        "action": str(decision.get("action", "ANALYSIS_ONLY")),
        "score": float(decision.get("score", 0.0) or 0.0),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "replay_hash": str(decision.get("replay_hash") or decision.get("replay_metadata", {}).get("replay_hash", "")),
        "source_quality": decision.get("source_quality") or {},
        "ensemble_version": str(decision.get("ensemble_version", "") or ""),
        "model_versions": dict(decision.get("model_versions") or {}),
        "agent_statuses": dict(decision.get("agent_statuses") or {}),
        "veto": dict(decision.get("veto") or {}),
        "confidence_breakdown_digest": (
            stable_hash(decision["confidence_breakdown"])
            if decision.get("confidence_breakdown") is not None
            else ""
        ),
        "event_payload": {
            "ticker": str(decision.get("ticker", "UNKNOWN")).upper(),
            "as_of": str(decision.get("as_of", "")),
            "mode": str(decision.get("mode", "ANALYSIS_ONLY")),
            "action": str(decision.get("action", "ANALYSIS_ONLY")),
        },
    }

    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    return event


def get_audit_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read the persisted decision events with optional limiting."""
    events = _read_audit_events()
    if limit is not None:
        return events[-limit:]
    return events


def get_events_since(cursor: str | None = None) -> list[dict[str, Any]]:
    """Return events strictly after `cursor` (an event_id) in file order.

    A None or unknown cursor returns the full history — the log is
    append-only, so this is a cheap suffix scan and feeds the timeline API.
    """
    events = _read_audit_events()
    if not cursor:
        return events
    for index, event in enumerate(events):
        if event.get("event_id") == cursor:
            return events[index + 1:]
    return events


def get_decision_by_replay_hash(replay_hash: str) -> list[dict[str, Any]]:
    """Look up a persisted score decision by deterministic replay hash."""
    return [event for event in _read_audit_events() if event.get("replay_hash") == replay_hash]


def get_decision_by_ticker_and_as_of(ticker: str, as_of: str) -> list[dict[str, Any]]:
    """Look up a persisted score decision by exact ticker and timestamp key."""
    target_ticker = str(ticker).upper()
    target_as_of = str(as_of)
    return [
        event for event in _read_audit_events()
        if str(event.get("ticker", "")).upper() == target_ticker and str(event.get("as_of", "")) == target_as_of
    ]
