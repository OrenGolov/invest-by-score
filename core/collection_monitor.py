"""Collection monitoring — notice when the collector stops, not months later.

**The failure this exists to catch.** MEASURED 2026-10-04, before the
collector was scheduled: 10 of 25 business days had no record at all, and
nobody knew. `check_data_coverage.py` detects it, but a CI gate only runs
when somebody pushes — exactly the wrong trigger for a job whose failure
mode is silence. The 2026-09-27 -> 2026-10-04 gap is 7 calendar days of
news gone permanently, and it passed unnoticed.

So monitoring here answers one question: **did today's run capture what
could never be captured again?**

**What is worth waking somebody for, and what is not.** Not every failure
is an alert, and sending one for each is how a channel trains its reader to
ignore it (A7's measured finding: 93.3% of alert-days are repeats, and an
unsuppressed channel emits 15 alerts per episode).

    PERISHABLE LOSS      alert   news gone forever; the day cannot return
    CONSECUTIVE SILENCE  alert   the collector itself has stopped
    quota exhausted      no      self-clearing, nothing to fix, expected
    macro unavailable    no      FRED serves vintages; rebuildable later
    price/fundamentals   no      re-fetchable in full

**Quota exhaustion is deliberately NOT an alert**, which is the judgement
most likely to be questioned. MEASURED: the free tier is 100 requests/24h
on a rolling window and one full sweep costs 75, so a quota day is the
EXPECTED steady state, not an incident. Alerting on it would produce a
daily notification the operator cannot act on — and would bury the
perishable-loss alert that matters.

**A gap is measured in BUSINESS days.** A Saturday with no collection is
not a failure, and a monitor that cannot tell a weekend from an outage
reports an outage every Monday.

**Nothing here sends anything.** This module decides; the caller delivers.
That keeps the decision testable without a network and lets the same
verdict reach either channel (email or Telegram), which is why both share
one status vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from core.config import (
    COLLECTION_ALERT_ON_PERISHABLE_LOSS,
    COLLECTION_ALERT_ON_QUOTA,
    COLLECTION_MAX_SILENT_BUSINESS_DAYS,
    COLLECTION_MONITOR_VERSION,
)

SEVERITY_NONE = "none"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"


@dataclass(frozen=True)
class CollectionVerdict:
    """Whether the operator needs to know, and what to tell them."""

    should_alert: bool
    severity: str
    subject: str = ""
    body: str = ""
    reasons: tuple[str, ...] = field(default_factory=tuple)
    version: str = COLLECTION_MONITOR_VERSION

    def __post_init__(self) -> None:
        if self.should_alert and not self.reasons:
            raise ValueError(
                "an alert must name WHY it fired; a notification the reader "
                "cannot act on is how a channel gets muted"
            )
        if self.should_alert and self.severity == SEVERITY_NONE:
            raise ValueError(
                f"severity {SEVERITY_NONE!r} contradicts should_alert=True"
            )


def business_days_between(start: date, end: date) -> int:
    """Business days strictly after `start`, up to and including `end`.

    Weekends only — no holiday calendar. A holiday therefore counts as a
    silent day, which errs toward alerting; the alternative (a stale
    hardcoded holiday list) errs toward silence, and silence is the failure
    mode this module exists to catch.
    """
    if end <= start:
        return 0
    days = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor += timedelta(days=1)
    return days


def load_reports(path: Path) -> list[dict[str, Any]]:
    """Every collection report on disk, oldest first. Never raises."""
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue  # a truncated line is not a reason to lose the rest
        if isinstance(parsed, dict):
            reports.append(parsed)
    return reports


def last_successful_capture(reports: list[dict[str, Any]], source: str) -> date | None:
    """The most recent day `source` was actually captured.

    Keys on the report's own `as_of`, not on file mtime: a re-run of an old
    day must not look like today's success.
    """
    best: date | None = None
    for report in reports:
        if source in (report.get("perishable_lost") or []):
            continue
        sources = report.get("sources") or {}
        status = str((sources.get(source) or {}).get("status", "")).upper()
        if status not in ("OK", "PARTIAL"):
            continue
        try:
            when = date.fromisoformat(str(report.get("as_of", ""))[:10])
        except ValueError:
            continue
        if best is None or when > best:
            best = when
    return best


def evaluate_collection(
    report: dict[str, Any] | None,
    history: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> CollectionVerdict:
    """Does this collection run warrant waking the operator?

    `report` is the run that just finished (None when the collector did not
    run at all). `history` is every prior report, used to detect silence.
    """
    history = history or []
    today = today or date.today()
    reasons: list[str] = []
    severity = SEVERITY_NONE
    lines: list[str] = []

    quota = False
    if report:
        news = (report.get("sources") or {}).get("news") or {}
        quota = bool(news.get("quota_exhausted"))

    # --- 1. perishable loss: the day cannot be recovered --------------
    lost = list((report or {}).get("perishable_lost") or [])
    if lost and COLLECTION_ALERT_ON_PERISHABLE_LOSS:
        # A quota day loses coverage for the UNVISITED tickers, which the
        # rotation retries tomorrow, and the cursor is deliberately not
        # advanced. That is the expected steady state on a 100/day tier,
        # so it is reported in the body but does not raise severity.
        if quota and not COLLECTION_ALERT_ON_QUOTA:
            lines.append(
                "news partially lost to the provider quota (expected on the "
                "free tier; the rotation cursor was not advanced, so these "
                "tickers are retried next run)"
            )
        else:
            severity = SEVERITY_CRITICAL
            reasons.append(f"perishable data lost: {', '.join(sorted(lost))}")
            lines.append(
                f"PERISHABLE LOSS: {', '.join(sorted(lost))} — this day "
                f"cannot be recovered later."
            )

    # --- 2. silence: the collector itself has stopped -----------------
    if report is None:
        reasons.append("the collector did not run")
        severity = SEVERITY_CRITICAL
        lines.append("The collector did not run at all today.")
    else:
        last = last_successful_capture(history + [report], "news")
        if last is not None:
            silent = business_days_between(last, today)
            if silent > COLLECTION_MAX_SILENT_BUSINESS_DAYS:
                reasons.append(
                    f"no news captured for {silent} business days "
                    f"(threshold {COLLECTION_MAX_SILENT_BUSINESS_DAYS})"
                )
                severity = SEVERITY_CRITICAL
                lines.append(
                    f"SILENCE: the last successful news capture was {last}, "
                    f"{silent} business days ago. A collector that stopped "
                    f"running leaves exactly this trace."
                )

    # --- 3. the run's own summary, always included --------------------
    if report:
        lines.append("")
        lines.append(f"run {report.get('as_of', '?')} -> {report.get('status', '?')}")
        for name, detail in sorted((report.get("sources") or {}).items()):
            status = str((detail or {}).get("status", "?"))
            lines.append(f"  {name:14} {status}")

    if not reasons:
        return CollectionVerdict(
            should_alert=False,
            severity=SEVERITY_NONE,
            body="\n".join(lines),
        )

    subject = f"invest-by-score: collection {severity.upper()} — {reasons[0]}"
    return CollectionVerdict(
        should_alert=True,
        severity=severity,
        subject=subject[:200],
        body="\n".join(lines),
        reasons=tuple(reasons),
    )


def render_verdict(verdict: CollectionVerdict) -> str:
    """One operator-readable line for the console or the log."""
    if not verdict.should_alert:
        return "[OK] collection monitor: nothing worth alerting on"
    return f"[{verdict.severity.upper()}] {'; '.join(verdict.reasons)}"
