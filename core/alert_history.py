"""Alert history for the Alerts Monitoring view, filterable by date and ticker.

**What this module can and cannot show.** Nothing in this repo persists an
alert. MEASURED 2026-10-07: no module appends to an alert ledger, and the seven
evaluators in `core/*_alert.py` return verdicts that `scripts/monitor_collection.py`
renders and discards. The only durable record of alerting is
`data/collection_report.jsonl`, which records what each COLLECTION RUN found.

So this view reconstructs alert history by re-evaluating the stored reports
through `evaluate_collection` -- the same function the 22:00 job calls. That has
one honest consequence the UI must not hide:

  * A row here means "this run WOULD alert, judged now". It is not proof that a
    message was ever delivered. On the Aman network SMTP is intercepted, so a
    historical CRITICAL may have reached nobody.

Re-evaluation is deterministic given the report, so the verdict is faithful to
what the job saw -- but `evaluate_collection` takes `today`, and silence-based
reasons are relative to it. Each row is therefore judged with `today` pinned to
that report's own `as_of`, not to the clock, or every old row would read as
days-stale and the view would be a wall of false CRITICALs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core import config
from core.collection_monitor import (
    SEVERITY_CRITICAL,
    SEVERITY_NONE,
    SEVERITY_WARN,
    evaluate_collection,
    explain_ticker_news_gap,
    load_reports,
)

ALERT_HISTORY_VERSION = "1.0.0"

_SEVERITY_RANK = {SEVERITY_NONE: 0, SEVERITY_WARN: 1, SEVERITY_CRITICAL: 2}


@dataclass(frozen=True)
class AlertRow:
    """One collection run, and whether it warranted waking the operator."""

    as_of: str
    collected_at: str
    severity: str
    should_alert: bool
    subject: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    tickers_failed: tuple[str, ...] = field(default_factory=tuple)
    news_written: int = 0
    news_attempted: int = 0
    delivered: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "collected_at": self.collected_at,
            "severity": self.severity,
            "should_alert": self.should_alert,
            "subject": self.subject,
            "reasons": list(self.reasons),
            "tickers_failed": list(self.tickers_failed),
            "news_written": self.news_written,
            "news_attempted": self.news_attempted,
            # None, never False: we do not know, and claiming "not delivered"
            # would be as wrong as claiming it was.
            "delivered": self.delivered,
        }


def _report_date(report: dict[str, Any]) -> str:
    return str(report.get("as_of") or "")


def _failed_tickers(report: dict[str, Any]) -> tuple[str, ...]:
    """Tickers whose fetch FAILED in this run, across sources.

    A ticker merely not scheduled is not included -- that distinction is the
    whole point of the 10-06 fix, and collapsing it here would reintroduce the
    bug at the presentation layer.
    """
    out: list[str] = []
    for source in (report.get("sources") or {}).values():
        if not isinstance(source, dict):
            continue
        for ticker in source.get("failed") or ():
            name = str(ticker).strip().upper()
            if name and name not in out:
                out.append(name)
    return tuple(sorted(out))


def _news_counts(report: dict[str, Any]) -> tuple[int, int]:
    sources = report.get("sources") or {}
    for key in ("news", "events"):
        block = sources.get(key)
        if isinstance(block, dict):
            written = block.get("written")
            attempted = block.get("attempted")
            if written is not None or attempted is not None:
                return int(written or 0), int(attempted or 0)
    return 0, 0


def build_alert_rows(
    reports: list[dict[str, Any]] | None,
) -> list[AlertRow]:
    """Re-evaluate every stored report, newest first.

    Each run is judged with `today` pinned to its own `as_of` (see module
    docstring) and with only the reports that preceded it as history, so a row
    reflects what was knowable at the time rather than what is known now.
    """
    reports = [r for r in (reports or []) if isinstance(r, dict)]
    rows: list[AlertRow] = []
    for index, report in enumerate(reports):
        as_of_raw = _report_date(report)
        try:
            pinned = date.fromisoformat(as_of_raw)
        except ValueError:
            pinned = None
        verdict = evaluate_collection(
            report,
            history=reports[:index],
            today=pinned,
        )
        written, attempted = _news_counts(report)
        rows.append(
            AlertRow(
                as_of=as_of_raw,
                collected_at=str(report.get("collected_at") or ""),
                severity=verdict.severity,
                should_alert=verdict.should_alert,
                subject=verdict.subject,
                reasons=tuple(verdict.reasons),
                tickers_failed=_failed_tickers(report),
                news_written=written,
                news_attempted=attempted,
                delivered=None,
            )
        )
    rows.sort(key=lambda row: (row.collected_at, row.as_of), reverse=True)
    return rows


def filter_alert_rows(
    rows: list[AlertRow],
    start: str = "",
    end: str = "",
    ticker: str = "",
    severity: str = "",
    only_alerting: bool = False,
) -> list[AlertRow]:
    """Narrow the history. An unparseable bound is ignored, not fatal.

    `start`/`end` are inclusive ISO dates against `as_of`. `ticker` matches a
    run whose failure list contains it. `severity` is a floor, not an equality
    test, so asking for `warn` includes `critical` -- the opposite would hide
    the worst rows behind the milder filter.
    """
    out = list(rows)

    def _parse(value: str) -> date | None:
        try:
            return date.fromisoformat(value.strip())
        except (ValueError, AttributeError):
            return None

    lower, upper = _parse(start), _parse(end)
    if lower or upper:
        kept: list[AlertRow] = []
        for row in out:
            try:
                stamp = date.fromisoformat(row.as_of)
            except ValueError:
                continue  # a row with no usable date cannot satisfy a range
            if lower and stamp < lower:
                continue
            if upper and stamp > upper:
                continue
            kept.append(row)
        out = kept

    name = (ticker or "").strip().upper()
    if name:
        out = [row for row in out if name in row.tickers_failed]

    floor = _SEVERITY_RANK.get((severity or "").strip().lower())
    if floor:
        out = [
            row for row in out if _SEVERITY_RANK.get(row.severity, 0) >= floor
        ]

    if only_alerting:
        out = [row for row in out if row.should_alert]

    return out


def summarise(rows: list[AlertRow]) -> dict[str, Any]:
    """Counts for the header strip."""
    return {
        "runs": len(rows),
        "alerting": sum(1 for row in rows if row.should_alert),
        "critical": sum(1 for row in rows if row.severity == SEVERITY_CRITICAL),
        "warn": sum(1 for row in rows if row.severity == SEVERITY_WARN),
        "clean": sum(1 for row in rows if not row.should_alert),
        "newest": rows[0].as_of if rows else "",
        "oldest": rows[-1].as_of if rows else "",
    }


def ticker_explanations(
    rows: list[AlertRow],
    reports: list[dict[str, Any]],
    ticker: str,
) -> list[dict[str, str]]:
    """Per-run reason THIS ticker has no news, for the ticker filter.

    Delegates to `explain_ticker_news_gap` so the view cannot drift from the
    digest's own wording.
    """
    name = (ticker or "").strip().upper()
    if not name:
        return []
    by_date: dict[str, dict[str, Any]] = {}
    for report in reports:
        if isinstance(report, dict):
            by_date.setdefault(_report_date(report), report)
    out: list[dict[str, str]] = []
    for row in rows:
        report = by_date.get(row.as_of)
        if report is None:
            continue
        explained = explain_ticker_news_gap(name, row.as_of, report)
        out.append(
            {
                "as_of": row.as_of,
                "kind": str(explained.get("kind") or ""),
                "reason": str(explained.get("reason") or ""),
                "action": str(explained.get("action") or ""),
            }
        )
    return out


def alert_history(
    start: str = "",
    end: str = "",
    ticker: str = "",
    severity: str = "",
    only_alerting: bool = False,
    limit: int = 200,
    path: Path | None = None,
) -> dict[str, Any]:
    """The Alerts Monitoring payload: filtered rows, counts, and provenance."""
    source = path or Path(config.COLLECT_REPORT_PATH)
    reports = load_reports(source)
    rows = build_alert_rows(reports)
    filtered = filter_alert_rows(
        rows,
        start=start,
        end=end,
        ticker=ticker,
        severity=severity,
        only_alerting=only_alerting,
    )
    capped = filtered[: max(1, int(limit or 200))]
    return {
        "version": ALERT_HISTORY_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "source_exists": source.exists(),
        "filters": {
            "start": start,
            "end": end,
            "ticker": (ticker or "").strip().upper(),
            "severity": severity,
            "only_alerting": only_alerting,
        },
        "summary": summarise(filtered),
        "total_runs_on_disk": len(rows),
        "returned": len(capped),
        "rows": [row.to_dict() for row in capped],
        "ticker_detail": ticker_explanations(capped, reports, ticker),
        # Stated in the payload, not just the UI, so any consumer inherits it.
        "delivery_caveat": (
            "Rows are re-evaluated from stored collection reports. A row shows "
            "what the monitor WOULD report for that run; it is not proof a "
            "message was delivered. No alert ledger exists yet."
        ),
    }
