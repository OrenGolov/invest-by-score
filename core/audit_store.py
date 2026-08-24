from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_audit.jsonl"


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": str(decision.get("ticker", "UNKNOWN")).upper(),
        "as_of": str(decision.get("as_of", "")),
        "mode": str(decision.get("mode", "ANALYSIS_ONLY")),
        "action": str(decision.get("action", "ANALYSIS_ONLY")),
        "score": float(decision.get("score", 0.0) or 0.0),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "replay_hash": str(decision.get("replay_hash") or decision.get("replay_metadata", {}).get("replay_hash", "")),
        "source_quality": decision.get("source_quality") or {},
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


def get_decision_by_replay_hash(replay_hash: str) -> list[dict[str, Any]]:
    """Look up a persisted score decision by deterministic replay hash."""
    return [event for event in _read_audit_events() if event.get("replay_hash") == replay_hash]
