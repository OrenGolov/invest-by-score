"""News coverage metrics — whether the ML is still being fed, in numbers.

**The gap this closes.** The collector reported each run's outcome and the
monitor decided whether to alert, but nothing measured the TREND. MEASURED
2026-10-06, news capture had been dead since 2026-10-04 and the only visible
symptom was 77 identically-worded alerts; the facts that would have named the
problem in one line — 34.7% of tickers had any event memory, the newest event
was 4 days old, the rotation cursor had not moved in 2 days — were computable
from data on disk and computed by nobody.

So this module answers five questions a reader can act on:

    news coverage %        how much of the eligible universe has fresh news
    quota consumption      requests spent against the window's allowance
    failed requests        how many, and of which KIND
    oldest missing date    the front edge of the permanent loss
    event memory freshness how stale the newest OBSERVED memory is

**Freshness is measured against the provider window, not against zero.** News
older than NEWS_LOOKBACK_DAYS cannot be refetched, so a memory that old is not
merely stale — the evidence behind it is gone. That is the threshold at which
staleness becomes irreversible, and it is the one worth reporting.

**Nothing here fetches anything.** Every number comes from the collection
report ledger and the event memory store, so the metrics cost no quota and can
run when the provider is down — which is exactly when they are needed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import (
    COLLECT_NEWS_BATCH_SIZE,
    COLLECT_NEWS_REQUEST_BUDGET,
    NEWS_LOOKBACK_DAYS,
)

NEWS_COVERAGE_VERSION = "news-coverage-v1"


def _as_date(value: Any) -> date | None:
    """Parse a date from a report field or an ISO timestamp. Never raises."""
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def event_memory_freshness(
    store_path: Path,
    today: date | None = None,
) -> dict[str, Any]:
    """How current the OBSERVED event memory is, and how wide its coverage.

    `expired` is the judgement that matters: once the newest memory is older
    than the provider's lookback window, the news that would extend it can no
    longer be fetched, so the gap is permanent rather than pending.
    """
    today = today or date.today()
    newest: date | None = None
    tickers: set[str] = set()
    provenance: dict[str, int] = {}
    total = 0

    if store_path.exists():
        try:
            lines = store_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a truncated line must not lose the rest
            if not isinstance(record, dict):
                continue
            total += 1
            kind = str(record.get("provenance") or "unknown")
            provenance[kind] = provenance.get(kind, 0) + 1
            if record.get("ticker"):
                tickers.add(str(record["ticker"]).upper())
            when = _as_date(record.get("published_time"))
            if when and (newest is None or when > newest):
                newest = when

    stale_days = (today - newest).days if newest else None
    return {
        "records": total,
        "distinct_tickers": len(tickers),
        "provenance": provenance,
        "newest_event_date": newest.isoformat() if newest else None,
        "stale_days": stale_days,
        # Beyond the lookback window the underlying news is unfetchable, so
        # the staleness can never be repaired by running the collector again.
        "expired": bool(stale_days is not None and stale_days > NEWS_LOOKBACK_DAYS),
        "lookback_days": NEWS_LOOKBACK_DAYS,
    }


def quota_consumption(reports: list[dict[str, Any]], window_hours: int = 24) -> dict[str, Any]:
    """Requests spent inside the trailing window, against the per-run budget.

    Older reports predate the `requests_spent` field. They are counted as
    `unknown_runs` rather than as zero, because reporting "0 requests spent"
    for a run that in fact exhausted the quota is worse than reporting that
    the number is not known.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    spent = 0
    runs = 0
    unknown = 0
    for report in reports:
        try:
            when = datetime.fromisoformat(str(report.get("collected_at", "")))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when < cutoff:
            continue
        runs += 1
        news = (report.get("sources") or {}).get("news") or {}
        if "requests_spent" in news:
            spent += int(news.get("requests_spent") or 0)
        else:
            unknown += 1
    return {
        "window_hours": window_hours,
        "runs_in_window": runs,
        "requests_spent": spent,
        "unknown_runs": unknown,
        "per_run_budget": COLLECT_NEWS_REQUEST_BUDGET,
        "batch_size": COLLECT_NEWS_BATCH_SIZE,
    }


def news_coverage(
    reports: list[dict[str, Any]],
    eligible_tickers: set[str] | None = None,
    today: date | None = None,
    window_days: int | None = None,
) -> dict[str, Any]:
    """Share of the eligible universe with a SUCCESSFUL news fetch recently.

    Counts only `ok_tickers` — tickers the provider actually answered for.
    A ticker that was attempted and failed is not covered, however many times
    it was attempted, because an attempt buys no data.

    The window defaults to the provider lookback: coverage outside it is not
    current coverage, it is history that can no longer be extended.
    """
    today = today or date.today()
    window = window_days or NEWS_LOOKBACK_DAYS
    earliest = today - timedelta(days=window)

    covered: set[str] = set()
    attempted: set[str] = set()
    failure_kinds: dict[str, int] = {}
    for report in reports:
        when = _as_date(report.get("as_of"))
        if when is None or when < earliest:
            continue
        news = (report.get("sources") or {}).get("news") or {}
        covered |= {str(t).upper() for t in (news.get("ok_tickers") or [])}
        for key in ("unavailable", "failed"):
            attempted |= {str(t).upper() for t in (news.get(key) or [])}
        for kind, count in (news.get("failure_kinds") or {}).items():
            failure_kinds[kind] = failure_kinds.get(kind, 0) + int(count or 0)

    eligible = {str(t).upper() for t in (eligible_tickers or set())}
    # Without an explicit universe, infer it from what the runs themselves
    # touched rather than reporting a percentage of an unknown denominator.
    universe = eligible or (covered | attempted)
    missing = sorted(universe - covered)
    pct = round(100.0 * len(covered & universe) / len(universe), 1) if universe else 0.0

    return {
        "window_days": window,
        "eligible": len(universe),
        "covered": len(covered & universe),
        "coverage_pct": pct,
        "missing": missing,
        "failure_kinds": failure_kinds,
        "failed_requests": sum(failure_kinds.values()),
    }


def oldest_missing_news_date(
    reports: list[dict[str, Any]],
    today: date | None = None,
    window_days: int = 30,
) -> dict[str, Any]:
    """The front edge of the loss: the oldest recent day that captured no news.

    A day with no report at all counts as missing — the collector not running
    is indistinguishable, in data terms, from it running and capturing nothing,
    and both lose the day.
    """
    today = today or date.today()
    captured: set[date] = set()
    for report in reports:
        when = _as_date(report.get("as_of"))
        if when is None:
            continue
        news = (report.get("sources") or {}).get("news") or {}
        if int(news.get("ok", 0) or 0) > 0:
            captured.add(when)

    missing: list[date] = []
    cursor = today - timedelta(days=window_days)
    while cursor <= today:
        if cursor.weekday() < 5 and cursor not in captured:
            missing.append(cursor)
        cursor += timedelta(days=1)

    unrecoverable = [d for d in missing if (today - d).days > NEWS_LOOKBACK_DAYS]
    return {
        "window_days": window_days,
        "missing_business_days": len(missing),
        "oldest_missing": missing[0].isoformat() if missing else None,
        "newest_missing": missing[-1].isoformat() if missing else None,
        # Past the lookback window the day cannot be refetched at any price.
        "unrecoverable_days": len(unrecoverable),
    }


def coverage_report(
    reports: list[dict[str, Any]],
    memory_path: Path,
    eligible_tickers: set[str] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Every metric in one dict, for a log line, an alert body or a dashboard."""
    return {
        "version": NEWS_COVERAGE_VERSION,
        "as_of": (today or date.today()).isoformat(),
        "coverage": news_coverage(reports, eligible_tickers, today),
        "quota": quota_consumption(reports),
        "gaps": oldest_missing_news_date(reports, today),
        "event_memory": event_memory_freshness(memory_path, today),
    }


def render_coverage(report: dict[str, Any]) -> str:
    """The metrics as operator-readable lines.

    Written for a 22:00 log nobody is watching live: each line states a number
    and what it means, so the reader does not need this module's source to
    interpret it.
    """
    coverage = report.get("coverage") or {}
    quota = report.get("quota") or {}
    gaps = report.get("gaps") or {}
    memory = report.get("event_memory") or {}

    lines = [f"news coverage [{report.get('as_of','?')}]"]
    lines.append(
        f"  coverage        {coverage.get('coverage_pct', 0)}% "
        f"({coverage.get('covered', 0)}/{coverage.get('eligible', 0)} eligible "
        f"tickers fetched in the last {coverage.get('window_days','?')}d)"
    )
    lines.append(
        f"  quota           {quota.get('requests_spent', 0)} requests in the last "
        f"{quota.get('window_hours','?')}h over {quota.get('runs_in_window',0)} run(s); "
        f"budget {quota.get('per_run_budget','?')}/run"
    )
    if quota.get("unknown_runs"):
        lines.append(
            f"                  ({quota['unknown_runs']} older run(s) predate "
            f"request accounting and are not counted)"
        )
    failures = coverage.get("failure_kinds") or {}
    if failures:
        detail = ", ".join(f"{kind}={count}" for kind, count in sorted(failures.items()))
        lines.append(f"  failures        {coverage.get('failed_requests',0)} ({detail})")
    else:
        lines.append("  failures        none recorded")
    lines.append(
        f"  gaps            {gaps.get('missing_business_days',0)} business day(s) "
        f"with no news in the last {gaps.get('window_days','?')}d; oldest "
        f"{gaps.get('oldest_missing') or 'none'}"
    )
    if gaps.get("unrecoverable_days"):
        lines.append(
            f"                  {gaps['unrecoverable_days']} of those are past the "
            f"{memory.get('lookback_days','?')}-day provider window and can never "
            f"be recovered"
        )
    newest = memory.get("newest_event_date") or "never"
    stale = memory.get("stale_days")
    lines.append(
        f"  event memory    {memory.get('records',0)} record(s) across "
        f"{memory.get('distinct_tickers',0)} ticker(s); newest {newest}"
        + (f", {stale}d old" if stale is not None else "")
    )
    if memory.get("expired"):
        lines.append(
            "                  EXPIRED: the newest OBSERVED memory is older than "
            "the provider window, so no run can extend it from live news"
        )
    return "\n".join(lines)
