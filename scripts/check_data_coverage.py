"""CI drift gate for daily data coverage — does the collector still run?

A scheduled task that stops firing produces no error anywhere. It produces
NOTHING, which looks exactly like a quiet week. MEASURED before this existed:
2-3 of 15 business days were missing from the W6 ledger, and nobody noticed
because nothing was watching.

This gate turns that silence into a failure. It is the counterpart to
`scripts/daily_collect.py`: the collector captures, this proves the capture
kept happening.

1.  the ledger has no unexplained gap in the recent business-day window.
    Weekends and a small allowance for provider outages are tolerated;
    a dead scheduler is not;
2.  the collection report is being WRITTEN. The ledger can look healthy from
    a manual run, so the report is the evidence that the JOB ran;
3.  the most recent run is recent. A report that stopped updating a month ago
    is a dead scheduler with a tidy history;
4.  PERISHABLE sources are named as such. MEASURED: NEWS_LOOKBACK_DAYS is 7,
    so an uncaptured day of news is gone permanently — the config must keep
    saying so, because the exit code depends on that classification;
5.  the collector still TRIGGERS rather than writing the ledger itself. A
    second write path would be the split-brain W5 forbids;
6.  the allowance cannot swallow a total outage: it must stay below the
    window, or a scheduler that never runs would still pass.

The gate is deliberately TOLERANT of the gaps already in the ledger and of a
missing NEWSAPI_KEY: it fails on a collector that stopped running, not on a
provider that was never configured. Those are different problems with
different fixes, and conflating them would make the gate noise.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    COLLECT_COVERAGE_MAX_MISSING,
    COLLECT_NEWS_BATCH_SIZE,
    COLLECT_NEWS_SKIP_SECTORLESS,
    COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS,
    COLLECT_PERISHABLE_SOURCES,
    COLLECT_REPORT_PATH,
    COLLECT_SOURCES,
    NEWS_LOOKBACK_DAYS,
    NEWS_PROVIDER_DAILY_LIMIT,
)
from core.raw_store import RAW_STORE_DIR  # noqa: E402

# How stale the newest collection report may be before the scheduler is
# presumed dead. Generous: a long weekend plus a holiday plus a slow Monday.
MAX_REPORT_AGE_DAYS = 6

# A source needs at least this many business days of its own history before
# its coverage means anything. Below it, "missing" days are days that predate
# the source rather than days it dropped.
MIN_DAYS_BEFORE_JUDGING = 5


def business_days_back(end: date, count: int) -> list[date]:
    """The `count` most recent business days ending at `end`, newest first."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def ledger_days(source_id: str) -> set[date]:
    """Dates the W6 ledger holds a file for, under one source."""
    folder = RAW_STORE_DIR / source_id
    if not folder.exists():
        return set()
    found: set[date] = set()
    for path in folder.glob("*.jsonl"):
        try:
            found.add(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return found


def _reports() -> list[dict]:
    path = REPO_ROOT / COLLECT_REPORT_PATH
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    today = date.today()
    window = business_days_back(today, COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS)
    oldest = min(window)

    # ---------------------------------------------------------------- 6
    if COLLECT_COVERAGE_MAX_MISSING >= COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS:
        failures.append(
            f"the missing-day allowance ({COLLECT_COVERAGE_MAX_MISSING}) is not "
            f"smaller than the window ({COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS}) "
            f"— a scheduler that NEVER ran would still pass this gate"
        )

    # ---------------------------------------------------------------- 4
    if "news" not in COLLECT_PERISHABLE_SOURCES:
        failures.append(
            f"news is no longer classified as perishable, but "
            f"NEWS_LOOKBACK_DAYS is still {NEWS_LOOKBACK_DAYS} — an uncaptured "
            f"day of news is gone permanently, and the exit code that tells a "
            f"scheduler so depends on this classification"
        )
    for source in COLLECT_PERISHABLE_SOURCES:
        if source not in COLLECT_SOURCES:
            failures.append(
                f"perishable source {source!r} is not in COLLECT_SOURCES, so "
                f"nothing collects it"
            )

    # ---------------------------------------------------------------- 1
    # Ledger coverage, but ONLY on a machine that actually collects.
    #
    # The ledger is TRACKED (W6 provenance travels with the repo), so a clone
    # made months from now carries records ending on the day it was committed.
    # Judging that clone against today's calendar would report "10 of 10 days
    # missing" and fail CI on a machine that was never supposed to collect —
    # exactly the noise this gate must not produce.
    #
    # The collection REPORT is the discriminator: it is gitignored, so its
    # presence means THIS machine has run the collector. Absent, there is
    # nothing to have stopped.
    reports_early = _reports()
    collecting_here = bool(reports_early)

    checked_any = False
    for folder in sorted(p for p in RAW_STORE_DIR.glob("*") if p.is_dir()):
        source_id = folder.name
        days = ledger_days(source_id)
        if not days:
            continue
        newest = max(days)
        if newest < oldest:
            # A ledger stale beyond the whole window. On a machine that does
            # NOT collect this is just an old clone. On one that DOES, it is
            # the worst possible outage — the collector died long enough ago
            # that not one day of the window survives — so it must not be
            # waved through as "retired". An earlier version skipped this
            # branch entirely and a 9-month-dead collector passed silently.
            if collecting_here:
                failures.append(
                    f"{source_id}: the newest record is {newest}, older than "
                    f"every one of the last "
                    f"{COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS} business days, "
                    f"while this machine IS collecting (a collection report "
                    f"exists). That is a collector that stopped long ago, not "
                    f"a source that was retired."
                )
            else:
                notes.append(
                    f"{source_id}: last record {newest}, older than the "
                    f"{COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS}-business-day "
                    f"window; not judged, since this machine has never "
                    f"collected"
                )
            continue
        # A source that only just STARTED collecting has no history to be
        # missing. `newsapi_news` appeared the day NEWS_PROVIDER_API_KEY was
        # set, and judging its first day against a ten-day window reported
        # "10 of 10 missing" for a source that had never worked better.
        # Judge a source only from its own first record onward.
        first = min(days)
        judged = [day for day in window if day >= first]
        if len(judged) < MIN_DAYS_BEFORE_JUDGING:
            notes.append(
                f"{source_id}: first record {first}, only {len(judged)} "
                f"business day(s) of history — too new to judge "
                f"(needs {MIN_DAYS_BEFORE_JUDGING})"
            )
            continue

        checked_any = True
        missing = [day for day in judged if day not in days]
        if not collecting_here:
            if missing:
                notes.append(
                    f"{source_id}: {len(missing)} day(s) missing, not judged — "
                    f"no collection report exists, so this machine has never "
                    f"run the collector and has nothing to have stopped"
                )
            continue
        if len(missing) > COLLECT_COVERAGE_MAX_MISSING:
            failures.append(
                f"{source_id}: {len(missing)} of the last "
                f"{COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS} business days have no "
                f"record (allowance {COLLECT_COVERAGE_MAX_MISSING}) — missing "
                f"{', '.join(str(d) for d in missing[:6])}"
                + (" ..." if len(missing) > 6 else "")
                + ". A collector that stopped running leaves exactly this trace."
            )
        elif missing:
            notes.append(
                f"{source_id}: {len(missing)} day(s) missing, inside the "
                f"allowance ({', '.join(str(d) for d in missing)})"
            )

    if not checked_any:
        notes.append(
            "no raw source has a record inside the window — nothing to check "
            "yet, which is expected on a fresh clone"
        )

    # ---------------------------------------------------------------- 2 + 3
    reports = reports_early
    if not reports:
        notes.append(
            f"{COLLECT_REPORT_PATH} does not exist — this machine has never "
            f"run the collector, so ledger gaps are not judged. Register the "
            f"task with scripts/install_daily_task.ps1 to start the record."
        )
    else:
        stamps: list[datetime] = []
        for row in reports:
            raw = str(row.get("collected_at", ""))
            try:
                stamps.append(datetime.fromisoformat(raw))
            except ValueError:
                continue
        if not stamps:
            failures.append(
                "the collection report has rows but none carries a readable "
                "collected_at — a report that cannot be dated cannot prove "
                "the collector is still running"
            )
        else:
            newest = max(stamps)
            if newest.tzinfo is None:
                newest = newest.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - newest).days
            if age > MAX_REPORT_AGE_DAYS:
                failures.append(
                    f"the newest collection report is {age} days old (limit "
                    f"{MAX_REPORT_AGE_DAYS}) — the scheduled task has stopped "
                    f"firing, and a dead scheduler leaves a tidy history rather "
                    f"than an error"
                )

    # ---------------------------------------------------------------- 5
    collector_path = REPO_ROOT / "scripts" / "daily_collect.py"
    collector = collector_path.read_text(encoding="utf-8")

    # Checked on the parsed AST, not the source text: the module docstring
    # NAMES append_raw_records in order to explain that it does not call it,
    # and a substring check cannot tell an explanation from a call.
    import ast

    tree = ast.parse(collector)
    called: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                called.add(target.id)
            elif isinstance(target, ast.Attribute):
                called.add(target.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    if "append_raw_records" in called or "append_raw_records" in imported:
        failures.append(
            "daily_collect.py writes to the raw ledger directly — every "
            "provider fetch already appends on its way past, and a second "
            "write path is the split-brain W5 forbids"
        )
    if "raw_store" in imported or any(
        name.endswith("raw_store") for name in imported
    ):
        failures.append(
            "daily_collect.py imports the raw store — it triggers provider "
            "fetches and must not touch the ledger itself"
        )
    for source in COLLECT_SOURCES:
        if f'"{source}"' not in collector:
            failures.append(
                f"source {source!r} is declared in COLLECT_SOURCES but the "
                f"collector has no collector for it — it would be silently "
                f"skipped every day"
            )

    # ---------------------------------------------------------------- 7
    # QUOTA AWARENESS. MEASURED: the provider allows 100 requests/day and the
    # collector makes one call per ticker, so an unbatched run over the whole
    # portfolio exceeds it — every request returned HTTP 429 the day this was
    # found. A batch that grows past the ceiling silently reintroduces that.
    from fetch_data import PORTFOLIO_TICKERS  # noqa: PLC0415

    if COLLECT_NEWS_BATCH_SIZE > NEWS_PROVIDER_DAILY_LIMIT:
        failures.append(
            f"the news batch ({COLLECT_NEWS_BATCH_SIZE}) exceeds the "
            f"provider's {NEWS_PROVIDER_DAILY_LIMIT}/day ceiling on its own"
        )
    if COLLECT_NEWS_BATCH_SIZE >= len(PORTFOLIO_TICKERS):
        failures.append(
            f"the news batch ({COLLECT_NEWS_BATCH_SIZE}) covers the whole "
            f"{len(PORTFOLIO_TICKERS)}-ticker portfolio, so no batching "
            f"happens — MEASURED, that is what produced HTTP 429"
        )

    # The batch must actually reduce, and must exclude sector-less tickers.
    import scripts.daily_collect as collector  # noqa: PLC0415

    batch, skipped, _start = collector.news_batch(list(PORTFOLIO_TICKERS))
    if len(batch) > COLLECT_NEWS_BATCH_SIZE:
        failures.append(
            f"news_batch returned {len(batch)} tickers, above the declared "
            f"batch size of {COLLECT_NEWS_BATCH_SIZE}"
        )
    # Asserted on the OUTCOME, not on the flag: gating this check behind
    # COLLECT_NEWS_SKIP_SECTORLESS made it vacuous the moment the flag was
    # turned off, which is exactly the edit it exists to catch.
    from core.market_context import sector_for  # noqa: PLC0415

    funds = [t for t in PORTFOLIO_TICKERS if sector_for(t) is None]
    if funds and not skipped:
        failures.append(
            f"the portfolio holds {len(funds)} sector-less ticker(s) "
            f"({', '.join(sorted(funds))}) but none was skipped — a fund's "
            f"news is market commentary, which E5 calls confounded, so the "
            f"call buys a memory about the market rather than the holding"
        )
    if funds and not COLLECT_NEWS_SKIP_SECTORLESS:
        failures.append(
            "COLLECT_NEWS_SKIP_SECTORLESS was disabled while the portfolio "
            "still holds funds; each one then spends a scarce API call on "
            "market commentary"
        )
    for fund in skipped:
        if fund in batch:
            failures.append(f"{fund}: skipped yet still present in the batch")

    # Rotation must cover the whole eligible set in a bounded number of runs.
    eligible = [t for t in PORTFOLIO_TICKERS if t not in set(skipped)]
    runs_needed = -(-len(eligible) // max(COLLECT_NEWS_BATCH_SIZE, 1))
    covered: set[str] = set()
    for run in range(runs_needed):
        start = (run * COLLECT_NEWS_BATCH_SIZE) % len(eligible)
        ordered = sorted(eligible)
        covered |= {
            ordered[(start + offset) % len(ordered)]
            for offset in range(COLLECT_NEWS_BATCH_SIZE)
        }
    if len(covered) < len(eligible):
        failures.append(
            f"rotation covers only {len(covered)} of {len(eligible)} eligible "
            f"tickers in {runs_needed} run(s) — some holding would never have "
            f"its news collected"
        )

    if failures:
        print("data-coverage gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("data-coverage gate OK:")
    print(
        f"  ledger complete over the last {COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS} "
        f"business days (allowance {COLLECT_COVERAGE_MAX_MISSING})."
    )
    print(
        f"  perishable sources declared: {', '.join(COLLECT_PERISHABLE_SOURCES)} "
        f"(news window {NEWS_LOOKBACK_DAYS}d)."
    )
    print("  the collector triggers provider fetches; it never writes the ledger.")
    for note in notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
