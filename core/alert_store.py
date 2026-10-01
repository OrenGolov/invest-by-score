"""A8 alert store — append-only, nothing ever deleted, queryable by the dashboard.

The operator was explicit: every alert appears in the Monitoring tab and none is
ever removed. That matches W6's append-only ledger, so this module stores rather
than caches, and the dashboard reads history rather than a snapshot.

**THE ALERT ID IS THE DEDUPLICATION KEY, AND IT MUST NOT CONTAIN THE TIMESTAMP.**
This is the trap A1 and A7 both measured, arriving a third time. A1 MEASURED four
consecutive days of an identical, entirely-unavailable forecast producing *four
distinct digests*, because the digest covers ``as_of``. A7 MEASURED the same thing
for suppression keys. So an id computed over the whole record is unique every day
by construction, and "prevent duplicate emails for the same event" would prevent
nothing at all.

The id therefore covers the IDENTITY of a finding — which detector, which ticker,
which horizon, which state — and deliberately excludes ``as_of``, ``detected_at``
and the free-text reason. Two runs on the same unchanged state produce the SAME
id, which is what makes "has this already been emailed?" answerable.

**AN ID IS NOT A SUPPRESSION DECISION.** The id says "this is the same finding";
A7's ``disposition`` says whether the reader should see it again. Keeping them
separate matters because the cooldown is measured in sessions (15, from A7's
measured mean episode length) while the id is timeless.

**WHY FINDINGS AND DELIVERIES ARE TWO FILES.** An alert is something observed; a
delivery is something done. Storing them together would make "we found this" and
"we emailed this" indistinguishable, and only the second is retryable — if the
SMTP host is down, the finding is still true and must not be re-detected to be
re-sent. ``data/alerts.jsonl`` holds findings; ``data/alert_deliveries.jsonl``
holds acts.

**TIMESTAMPS ARE STORED IN UTC.** The dashboard renders them in the viewer's zone.
Storing local time would make the ledger unsortable across a daylight-saving
boundary and wrong when read from another machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    ALERT_DELIVERY_PATH,
    ALERT_DELIVERY_VERSION,
    ALERT_PRIORITIES,
    ALERT_STORE_APPEND_ONLY,
    ALERT_STORE_PATH,
)

LOGGER = logging.getLogger("alert_store")

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = REPO_ROOT / ALERT_STORE_PATH
DELIVERY_PATH = REPO_ROOT / ALERT_DELIVERY_PATH

# Fields that establish WHICH FINDING this is. Declared as data so the identity
# rule is readable without tracing code, and so widening it is a deliberate edit.
ALERT_ID_FIELDS: tuple[str, ...] = ("alert", "ticker", "horizon", "state")

# Fields that must NEVER enter the id. Each advances on its own schedule, so an
# id containing one is unique every run and deduplicates nothing. MEASURED three
# times in this codebase: A1 on digests, A7 on suppression keys, and here.
ALERT_ID_FORBIDDEN: tuple[str, ...] = (
    "as_of",
    "detected_at",
    "timestamp",
    "digest",
    "reason",
    "note",
)

# Delivery channels, named so a delivery row says what was actually done.
CHANNEL_DASHBOARD = "dashboard"
CHANNEL_EMAIL = "email"
CHANNEL_DIGEST = "email_digest"
DELIVERY_CHANNELS: tuple[str, ...] = (
    CHANNEL_DASHBOARD,
    CHANNEL_EMAIL,
    CHANNEL_DIGEST,
)

DELIVERY_SENT = "SENT"
DELIVERY_FAILED = "FAILED"
DELIVERY_SKIPPED = "SKIPPED"      # deliberately not sent (no credential, dry run)
DELIVERY_DUPLICATE = "DUPLICATE"  # already delivered under this id


class AlertStoreError(ValueError):
    """Raised when the store is asked to do something that would lose a finding."""


def alert_id(record: Mapping[str, Any]) -> str:
    """The stable identity of a finding, independent of when it was observed.

    Two observations of the same unchanged state share an id; a state change
    produces a new one. This is what makes duplicate-email prevention possible.
    """
    if not isinstance(record, Mapping):
        raise AlertStoreError("an alert record must be a mapping")
    for forbidden in ALERT_ID_FORBIDDEN:
        if forbidden in ALERT_ID_FIELDS:
            raise AlertStoreError(
                f"{forbidden!r} must never enter the alert id: it advances on "
                f"its own schedule, so the id would be unique every run and "
                f"deduplicate nothing -- MEASURED, four consecutive days of an "
                f"identical unavailable forecast produced four distinct digests"
            )
    parts = []
    for field in ALERT_ID_FIELDS:
        value = record.get(field)
        parts.append("" if value is None else str(value))
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_record(
    alert: Mapping[str, Any],
    grading: Mapping[str, Any],
    *,
    title: str,
    summary: Mapping[str, Any] | None = None,
    disposition: Mapping[str, Any] | None = None,
    detected_at: str | None = None,
) -> dict:
    """One storable finding, assembled from a detector alert and its grading.

    ``alert`` is the raw A1-A6 (or news) output; ``grading`` is
    ``alert_priority.grade``. Both are kept whole rather than flattened, so the
    stored row carries the evidence for its own priority and nobody has to re-run
    the grader to learn why a band was assigned.
    """
    if not isinstance(alert, Mapping):
        raise AlertStoreError("an alert must be a mapping")
    if not isinstance(grading, Mapping):
        raise AlertStoreError("a grading must be a mapping")

    priority = grading.get("priority")
    if priority is not None and priority not in ALERT_PRIORITIES:
        raise AlertStoreError(
            f"unknown priority {priority!r}; storing it would place the alert "
            f"in a band the dashboard cannot filter or colour"
        )

    # THE TIMESTAMP IS VALIDATED, BECAUSE THE FEED IS ORDERED BY IT.
    #
    # CAUGHT BY PROBE: a generated stamp of "2026-10-01T24:00:00+00:00" was
    # stored without complaint. Hour 24 does not exist, and because the feed
    # sorts on this string, that row would sit at the top of the operator's
    # Monitoring tab permanently -- above every genuinely newer alert. An
    # unparseable stamp is a corrupt row, not a very recent one.
    stamp = detected_at or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError) as exc:
        raise AlertStoreError(
            f"detected_at {stamp!r} is not an ISO-8601 timestamp ({exc}); the "
            f"feed is ordered on this field as a string, so an unparseable "
            f"value would outrank every real alert forever"
        ) from None
    if parsed.tzinfo is None:
        raise AlertStoreError(
            f"detected_at {stamp!r} carries no timezone; alerts are stored in "
            f"UTC so that they sort correctly across a daylight-saving "
            f"boundary and read correctly from another machine"
        )

    state = grading.get("state")
    record = {
        "version": ALERT_DELIVERY_VERSION,
        "alert": alert.get("alert"),
        "ticker": alert.get("ticker"),
        "horizon": alert.get("horizon"),
        "state": state,
        "as_of": alert.get("as_of"),
        "detected_at": stamp,
        "priority": priority,
        "priority_rank": grading.get("priority_rank"),
        "priority_reason": grading.get("priority_reason"),
        "severity": alert.get("severity"),
        "action": grading.get("action"),
        "action_reason": grading.get("action_reason"),
        "title": title,
        "reason": alert.get("reason"),
        # The whole detector payload, so the dashboard's expanded view and any
        # later audit read the same evidence the grader saw.
        "detail": dict(alert),
    }
    if summary is not None:
        record["summary"] = dict(summary)
    if disposition is not None:
        record["disposition"] = dict(disposition)
    record["alert_id"] = alert_id(record)
    return record


def _append(path: Path, row: Mapping[str, Any]) -> bool:
    """Append one JSON line. Never raises; a write failure is logged and flagged.

    Follows ``raw_store.append_raw_records``: the caller keeps working in a
    degraded-but-flagged state rather than losing the run to a disk error.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("alert_store_write_failed: %s", exc)
        return False


def append_alert(record: Mapping[str, Any], path: Path | None = None) -> bool:
    """Store one finding. Append-only: an existing row is never rewritten."""
    if not ALERT_STORE_APPEND_ONLY:
        raise AlertStoreError(
            "the store is append-only by contract; the operator requires that "
            "no alert is ever deleted"
        )
    return _append(Path(path) if path is not None else STORE_PATH, record)


def load_alerts(path: Path | None = None) -> list[dict]:
    """Every stored finding, oldest first. A missing store is empty, not an error.

    Returns rows in FILE order. Ordering for display is the reader's choice and
    is applied in ``query``; returning them pre-sorted here would hide the fact
    that the file itself is append-ordered.
    """
    store = Path(path) if path is not None else STORE_PATH
    if not store.exists():
        return []
    rows: list[dict] = []
    try:
        text = store.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("alert_store_read_failed: %s", exc)
        return []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # ONE BAD LINE MUST NOT HIDE THE REST. A truncated final write is
            # the realistic failure (power loss mid-append), and discarding the
            # whole history for it would lose every earlier finding.
            LOGGER.warning("alert_store_bad_line %d: %s", number, exc)
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def record_delivery(
    alert_record: Mapping[str, Any],
    *,
    channel: str,
    status: str,
    detail: str = "",
    path: Path | None = None,
) -> dict:
    """Log that a finding was (or was not) delivered on one channel.

    Written even when the status is SKIPPED or FAILED: "we chose not to email
    this" and "the host refused it" are different facts from "it was sent", and
    only a record of the attempt makes the difference auditable.
    """
    if channel not in DELIVERY_CHANNELS:
        raise AlertStoreError(
            f"unknown delivery channel {channel!r}; the known channels are "
            f"{list(DELIVERY_CHANNELS)}"
        )
    if status not in (
        DELIVERY_SENT,
        DELIVERY_FAILED,
        DELIVERY_SKIPPED,
        DELIVERY_DUPLICATE,
    ):
        raise AlertStoreError(f"unknown delivery status {status!r}")

    row = {
        "version": ALERT_DELIVERY_VERSION,
        "alert_id": alert_record.get("alert_id"),
        "ticker": alert_record.get("ticker"),
        "priority": alert_record.get("priority"),
        "channel": channel,
        "status": status,
        "detail": detail,
        "delivered_at": datetime.now(timezone.utc).isoformat(),
    }
    _append(Path(path) if path is not None else DELIVERY_PATH, row)
    return row


def load_deliveries(path: Path | None = None) -> list[dict]:
    """Every delivery attempt, oldest first."""
    return load_alerts(Path(path) if path is not None else DELIVERY_PATH)


def delivered_ids(
    channel: str | None = None,
    path: Path | None = None,
) -> set[str]:
    """Alert ids already SENT, optionally on one channel.

    Only SENT counts. A FAILED or SKIPPED attempt must NOT mark a finding as
    delivered, or an SMTP outage would permanently silence every alert it
    touched -- the failure mode where a transient becomes a blackout.
    """
    out: set[str] = set()
    for row in load_deliveries(path):
        if row.get("status") != DELIVERY_SENT:
            continue
        if channel is not None and row.get("channel") != channel:
            continue
        identifier = row.get("alert_id")
        if identifier:
            out.add(str(identifier))
    return out


def query(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    priority: str | Iterable[str] | None = None,
    ticker: str | None = None,
    event_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    search: str | None = None,
    newest_first: bool = True,
    limit: int | None = None,
    path: Path | None = None,
) -> list[dict]:
    """The dashboard's read path: filter, search, order, page.

    Every filter is INDEPENDENT and absent means "do not filter", never "match
    nothing". ``start``/``end`` compare ISO date prefixes, which sorts correctly
    because ``detected_at`` is stored in UTC.
    """
    items = list(rows) if rows is not None else load_alerts(path)

    if priority is not None:
        wanted = {priority} if isinstance(priority, str) else set(priority)
        unknown = wanted - set(ALERT_PRIORITIES)
        if unknown:
            raise AlertStoreError(
                f"unknown priority filter {sorted(unknown)}; the known bands "
                f"are {list(ALERT_PRIORITIES)}"
            )
        items = [r for r in items if r.get("priority") in wanted]

    if ticker:
        # Case-insensitive, because the operator types `nvda` and the store
        # holds `NVDA`. An exact match would silently return nothing.
        needle = str(ticker).strip().upper()
        items = [r for r in items if str(r.get("ticker") or "").upper() == needle]

    if event_type:
        needle = str(event_type).strip().lower()
        items = [r for r in items if str(r.get("alert") or "").lower() == needle]

    if start:
        items = [r for r in items if str(r.get("detected_at") or "") >= str(start)]
    if end:
        # Inclusive of the whole end DAY. A bare date compares as midnight, so
        # `end="2026-10-01"` would otherwise exclude everything that day.
        boundary = str(end)
        if len(boundary) == 10:
            boundary += "T99"
        items = [r for r in items if str(r.get("detected_at") or "") <= boundary]

    if search:
        needle = str(search).strip().lower()
        if needle:
            def matches(row: Mapping[str, Any]) -> bool:
                for field in (
                    "ticker",
                    "title",
                    "alert",
                    "state",
                    "reason",
                    "priority",
                    "action",
                    "priority_reason",
                ):
                    if needle in str(row.get(field) or "").lower():
                        return True
                return False

            items = [r for r in items if matches(r)]

    items.sort(key=lambda r: str(r.get("detected_at") or ""), reverse=newest_first)

    if limit is not None:
        if limit < 1:
            raise AlertStoreError("a limit must be at least 1")
        items = items[:limit]
    return items


def tickers_seen(path: Path | None = None) -> list[str]:
    """Every ticker that has ever produced an alert, for the dashboard filter."""
    return sorted(
        {
            str(row["ticker"]).upper()
            for row in load_alerts(path)
            if row.get("ticker")
        }
    )


def event_types_seen(path: Path | None = None) -> list[str]:
    """Every detector that has ever produced an alert."""
    return sorted(
        {str(row["alert"]) for row in load_alerts(path) if row.get("alert")}
    )


def counts_by_priority(path: Path | None = None) -> dict[str, int]:
    """How many findings sit in each band, including the ungraded ones.

    The ungraded count is reported under ``"(ungraded)"`` rather than omitted: a
    detector that cannot reach a verdict is the one thing a summary must not hide.
    """
    out: dict[str, int] = {band: 0 for band in ALERT_PRIORITIES}
    out["(ungraded)"] = 0
    for row in load_alerts(path):
        band = row.get("priority")
        key = band if band in out else "(ungraded)"
        out[key] += 1
    return out
