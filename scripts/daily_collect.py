"""Daily data collection — the capture step continuous learning depends on.

The system learns from what it recorded. Nothing recorded it on a schedule:
the W6 ledger accrued only when somebody happened to run a command, and
MEASURED over three weeks, **2-3 of 15 business days are missing**. Those gaps
are permanent for anything perishable.

**What is actually at risk.** Most inputs are rebuildable and lose nothing by
waiting:

    price bars        REBUILDABLE   15y of daily history, re-fetchable
    hourly bars       REBUILDABLE   MEASURED: 2y available, not ~1 month
    macro series      REBUILDABLE   FRED/ALFRED serve vintages
    regime labels     REBUILDABLE   computed from bars
    inferred memories REBUILDABLE   the backfill regenerates them

    NEWS ARTICLES     PERISHABLE    NEWS_LOOKBACK_DAYS = 7, then GONE
    sentiment         PERISHABLE    derived from news; dies with it
    observed memories PERISHABLE    need the news that produced them
    fundamentals      PARTLY        values are re-fetchable; the VINTAGE is not

So the argument for scheduling is irreversibility, not convenience. A day of
news missed is a day that can never be learned from, and it is exactly the
data F5's `observed` memories require.

**This script triggers, it does not duplicate.** Every provider fetch already
appends to the W6 ledger on its way past (`core.raw_store.append_raw_records`
is called inside `fetch_price_history`, `fetch_fundamental_snapshot`,
`build_news_snapshot` and `build_macro_snapshot`). The collector's job is to
make those calls happen every business day — writing records itself would be
a second ledger path, which W5 forbids.

**Idempotent by construction.** Re-running on the same day re-fetches and the
ledger appends a new version line; `load_raw_records` supersedes older
versions by payload hash, so a double run costs bandwidth and changes no
answer. Running it twice is safe; the danger is running it zero times.

**Fail-soft per source, fail-loud in aggregate.** One provider being down must
not stop the others — a missing macro series is not a reason to lose the day's
news. Each source reports its own status, and the exit code reflects whether
anything PERISHABLE was lost, not whether everything succeeded.

Usage:
    python scripts/daily_collect.py                 # everything, today
    python scripts/daily_collect.py --sources news  # just the perishable one
    python scripts/daily_collect.py --dry-run
    python scripts/daily_collect.py --tickers AAPL,MSFT

Scheduling (Windows Task Scheduler, daily on business days):
    schtasks /create /tn "invest-by-score daily" /sc weekly ^
      /d MON,TUE,WED,THU,FRI /st 22:00 ^
      /tr "cmd /c cd /d C:\\Users\\oreng\\invest-by-score && .venv\\Scripts\\python.exe scripts\\daily_collect.py >> data\\collect.log 2>&1"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    COLLECT_MAX_TICKERS_PER_RUN,
    COLLECT_NEWS_BATCH_SIZE,
    COLLECT_NEWS_CURSOR_PATH,
    COLLECT_NEWS_SKIP_SECTORLESS,
    COLLECT_NEWS_TRACK_ANYWAY,
    COLLECT_PERISHABLE_SOURCES,
    COLLECT_REPORT_PATH,
    COLLECT_SOURCES,
    COLLECT_THROTTLE_SECONDS,
    FINNHUB_API_KEY_ENV,
    NEWS_PROVIDER_API_KEY_ENV,
)

LOGGER = logging.getLogger("daily_collect")

STATUS_OK = "OK"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"


def _tickers(explicit: str) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    from fetch_data import PORTFOLIO_TICKERS

    return sorted({str(t).upper() for t in PORTFOLIO_TICKERS if str(t).strip()})


def collect_prices(tickers, as_of: str, dry_run: bool) -> dict:
    """Daily bars for every holding. Rebuildable, but cheap to keep current."""
    from fetch_data import fetch_price_history

    ok, failed = 0, []
    for ticker in tickers:
        if dry_run:
            ok += 1
            continue
        try:
            frame = fetch_price_history(ticker, period="1y", interval="1d")
            if frame is None or getattr(frame, "empty", True):
                failed.append(ticker)
            else:
                ok += 1
        except Exception as exc:
            LOGGER.warning("price fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
        time.sleep(COLLECT_THROTTLE_SECONDS)
    return {"attempted": len(tickers), "ok": ok, "failed": failed}


def news_batch(tickers) -> tuple[list[str], list[str], int]:
    """The subset of tickers this run should fetch, and what it skipped.

    Two reductions, for two different reasons:

    1. SECTOR-LESS tickers are dropped entirely. MEASURED, C3 gives VOO, SOXX,
       CIBR and NASA no sector, and E6 matches analogs on event_type against a
       company's chart state. A fund-level headline is market commentary,
       which E5 attribution calls CONFOUNDED — so the call buys a memory about
       the market, not the holding. They keep their inferred memories from
       price history, which cost no quota.
    2. The remainder is ROTATED in batches. MEASURED: the provider allows 100
       requests/day and each ticker is one call, so 77 plus any ad-hoc work
       exceeds it — today every request returned HTTP 429. A batch of 40
       covers all 73 in two runs and leaves 60 calls spare.

    The cursor advances sequentially rather than by a date-derived stride:
    MEASURED, sequential gives a visit spread of 1 over 20 runs where a stride
    left 2.
    """
    from core.market_context import sector_for

    skipped: list[str] = []
    eligible = list(tickers)
    if COLLECT_NEWS_SKIP_SECTORLESS:
        # ...except the narrow thematic funds named in the allow-list. SOXX
        # and CIBR track one industry each, so their news is closer to a
        # sector event than to market commentary, and the portfolio holds
        # many of their constituents.
        tracked = {t.upper() for t in COLLECT_NEWS_TRACK_ANYWAY}
        skipped = [
            t for t in eligible
            if sector_for(t) is None and t.upper() not in tracked
        ]
        eligible = [t for t in eligible if t not in set(skipped)]

    eligible.sort()
    if not eligible or len(eligible) <= COLLECT_NEWS_BATCH_SIZE:
        return eligible, skipped, 0

    cursor_file = REPO_ROOT / COLLECT_NEWS_CURSOR_PATH
    cursor = 0
    try:
        cursor = int(json.loads(cursor_file.read_text(encoding="utf-8"))["cursor"])
    except Exception:
        cursor = 0  # a missing or unreadable cursor starts at the head

    start = cursor % len(eligible)
    batch = [
        eligible[(start + offset) % len(eligible)]
        for offset in range(COLLECT_NEWS_BATCH_SIZE)
    ]
    return batch, skipped, start


def _advance_cursor(eligible_count: int, start: int) -> None:
    """Move the cursor on, so the next run fetches the next batch."""
    if eligible_count <= 0:
        return
    cursor_file = REPO_ROOT / COLLECT_NEWS_CURSOR_PATH
    try:
        cursor_file.parent.mkdir(parents=True, exist_ok=True)
        cursor_file.write_text(
            json.dumps({
                "cursor": (start + COLLECT_NEWS_BATCH_SIZE) % eligible_count,
                "updated": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )
    except Exception as exc:  # a cursor failure must not lose the day's news
        LOGGER.warning("could not advance the news cursor: %s", exc)


def collect_news(tickers, as_of: str, dry_run: bool) -> dict:
    """THE PERISHABLE ONE. A day missed here is a day lost permanently."""
    from core.news_adapter import build_news_snapshot

    batch, skipped, start = news_batch(tickers)
    eligible_count = len(tickers) - len(skipped)

    ok, unavailable, failed = 0, [], []
    quota_exhausted = False
    for ticker in batch:
        if dry_run:
            ok += 1
            continue
        try:
            snapshot = build_news_snapshot(ticker, as_of)
            status = str(snapshot.get("status", "")).upper()
            if status == "UNAVAILABLE":
                unavailable.append(ticker)
                # A 429 means the quota is gone; every further call this run
                # is wasted and would only deepen the overage.
                if "429" in str(snapshot.get("reason", "")):
                    quota_exhausted = True
                    break
            else:
                ok += 1
        except Exception as exc:
            LOGGER.warning("news fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
        time.sleep(COLLECT_THROTTLE_SECONDS)

    if not dry_run and not quota_exhausted:
        _advance_cursor(eligible_count, start)

    return {
        "attempted": len(batch), "ok": ok,
        "unavailable": unavailable, "failed": failed,
        "batch": len(batch), "eligible": eligible_count,
        "skipped_sectorless": sorted(skipped),
        "cursor_start": start,
        "quota_exhausted": quota_exhausted,
    }


def collect_fundamentals(tickers, as_of: str, dry_run: bool) -> dict:
    """Vintage capture: the values are re-fetchable, today's reading is not."""
    from fetch_data import fetch_fundamental_snapshot

    ok, failed = 0, []
    for ticker in tickers:
        if dry_run:
            ok += 1
            continue
        try:
            fetch_fundamental_snapshot(ticker, as_of)
            ok += 1
        except Exception as exc:
            LOGGER.warning("fundamentals fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
        time.sleep(COLLECT_THROTTLE_SECONDS)
    return {"attempted": len(tickers), "ok": ok, "failed": failed}


def collect_macro(tickers, as_of: str, dry_run: bool) -> dict:
    """One macro snapshot per run — the series are shared, not per-ticker."""
    from core.macro_adapter import build_macro_snapshot

    if dry_run:
        return {"attempted": 1, "ok": 1, "failed": []}
    probe = tickers[0] if tickers else "SPY"
    try:
        snapshot = build_macro_snapshot(probe, as_of)
        status = str(snapshot.get("status", "")).upper()
        return {
            "attempted": 1,
            "ok": 1 if status != "UNAVAILABLE" else 0,
            "failed": [] if status != "UNAVAILABLE" else [probe],
            "status": status,
        }
    except Exception as exc:
        LOGGER.warning("macro snapshot failed: %s", exc)
        return {"attempted": 1, "ok": 0, "failed": [probe], "status": "ERROR"}


def collect_events(tickers, as_of: str, dry_run: bool) -> dict:
    """OBSERVED event memories, from the day's news.

    This is the only path that produces `observed` memories. It depends
    entirely on the news provider: with no key, nothing here can run, and that
    is reported rather than silently producing zero.
    """
    if dry_run:
        # A dry run writes nothing by definition, so it must not be reported
        # as a LOST day. `attempted: 0` makes the status SKIPPED rather than
        # FAILED — the distinction matters because the exit code drives a
        # scheduler, and a dry run must never look like a real outage.
        return {"attempted": 0, "written": 0, "note": "dry run"}

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_event_memory import forward

    # The SAME batch the news stage fetched. `forward` fetches news itself, so
    # passing the full list here would spend the quota a second time on
    # tickers this run already covered — and on the sector-less ones the news
    # stage deliberately skipped. The cursor is NOT advanced by this call;
    # collect_news owns it.
    batch, _skipped, _start = news_batch(tickers)
    report = forward(batch, as_of, False, None)
    unavailable = len(report.get("news_unavailable") or [])
    return {
        # `attempted` counts what this source COULD act on. When news is
        # unavailable there are no events to study, so nothing was attempted —
        # but the day is still lost, which `news_blocked` records. Reporting
        # it as a failure of THIS source would blame the wrong stage; the
        # aggregate already fails on news, and one lost day should not be
        # counted twice.
        "attempted": report.get("considered", 0),
        "written": report.get("written", 0),
        "news_unavailable": unavailable,
        "news_blocked": unavailable > 0,
    }


COLLECTORS = {
    "prices": collect_prices,
    "news": collect_news,
    "fundamentals": collect_fundamentals,
    "macro": collect_macro,
    "events": collect_events,
}


def _source_status(source: str, result: dict) -> str:
    """One source's verdict. PERISHABLE sources are judged more strictly."""
    attempted = int(result.get("attempted", 0) or 0)
    if attempted == 0:
        return STATUS_SKIPPED

    if source == "news":
        if result.get("ok"):
            return STATUS_OK if not result.get("unavailable") else STATUS_PARTIAL
        return STATUS_FAILED
    if source == "events":
        if result.get("written"):
            return STATUS_OK
        # Blocked upstream is not a failure HERE. News already reports the
        # lost day; double-counting it would make the aggregate look worse
        # than the evidence supports and hide which stage actually broke.
        return STATUS_SKIPPED if result.get("news_blocked") else STATUS_FAILED
    if source == "macro":
        return STATUS_OK if result.get("ok") else STATUS_FAILED

    ok = int(result.get("ok", 0) or 0)
    if ok == attempted:
        return STATUS_OK
    return STATUS_PARTIAL if ok else STATUS_FAILED


def run(sources, tickers, as_of: str, dry_run: bool) -> dict:
    results: dict[str, dict] = {}
    statuses: dict[str, str] = {}

    for source in sources:
        collector = COLLECTORS.get(source)
        if collector is None:
            continue
        started = time.time()
        try:
            result = collector(tickers, as_of, dry_run)
        except Exception as exc:
            # Fail-SOFT per source: a dead provider must not cost the others
            # their day. The aggregate verdict below decides what it means.
            LOGGER.error("collector %s raised: %s", source, exc)
            result = {"attempted": 0, "ok": 0, "failed": [], "error": str(exc)}
        result["seconds"] = round(time.time() - started, 1)
        results[source] = result
        statuses[source] = _source_status(source, result)

    # Only a REAL loss counts. A source that was never attempted (a dry run,
    # or a source the caller did not request) is not a lost day, and saying
    # so would train a reader to ignore the one message that matters.
    lost = (
        []
        if dry_run
        else [
            source for source in COLLECT_PERISHABLE_SOURCES
            if statuses.get(source) == STATUS_FAILED
        ]
    )

    return {
        "as_of": as_of,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "tickers": len(tickers),
        "dry_run": dry_run,
        "sources": results,
        "statuses": statuses,
        "perishable_lost": lost,
        "status": STATUS_FAILED if lost else (
            STATUS_PARTIAL if STATUS_PARTIAL in statuses.values() else STATUS_OK
        ),
    }


def _write_report(report: dict) -> None:
    """Append one line per run, so a silent outage is visible in the record."""
    path = REPO_ROOT / COLLECT_REPORT_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, sort_keys=True, default=str) + "\n")
    except Exception as exc:  # a report failure must not fail the collection
        LOGGER.warning("could not write the collection report: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(COLLECT_SOURCES),
                        help=f"comma-separated; known: {', '.join(COLLECT_SOURCES)}")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=COLLECT_MAX_TICKERS_PER_RUN)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in COLLECT_SOURCES]
    if unknown:
        print(f"unknown source(s): {', '.join(unknown)}")
        print(f"known: {', '.join(COLLECT_SOURCES)}")
        return 2

    tickers = _tickers(args.tickers)[: max(1, args.limit)]
    report = run(requested, tickers, args.as_of, args.dry_run)
    if not args.dry_run:
        _write_report(report)

    print(f"daily collection [{report['as_of']}] -> {report['status']}")
    print(f"  tickers: {report['tickers']}" + ("  (dry run)" if args.dry_run else ""))
    for source in requested:
        result = report["sources"].get(source) or {}
        status = report["statuses"].get(source, STATUS_SKIPPED)
        perishable = " PERISHABLE" if source in COLLECT_PERISHABLE_SOURCES else ""
        detail = {
            key: value for key, value in result.items()
            if key not in ("seconds",) and value not in ([], 0, None, "")
        }
        print(f"  {source:<13} {status:<8}{perishable:<12} {detail}")

    news = report["sources"].get("news") or {}
    if news.get("skipped_sectorless"):
        print()
        print(
            f"  news covered {news.get('batch')} of {news.get('eligible')} "
            f"eligible tickers (cursor {news.get('cursor_start')}); "
            f"{len(news['skipped_sectorless'])} fund(s) skipped: "
            f"{', '.join(news['skipped_sectorless'])}"
        )
        print(
            "      A fund has no sector, so its news is market commentary "
            "rather than a company event — E5 would call it confounded. They "
            "keep their inferred memories from price history."
        )
    if news.get("quota_exhausted"):
        print()
        print("  PROVIDER QUOTA EXHAUSTED (HTTP 429): the run stopped early.")
        print("      The rotation cursor was NOT advanced, so the next run")
        print("      retries these same tickers rather than skipping them.")

    if report["perishable_lost"]:
        print()
        print("  PERISHABLE DATA WAS NOT CAPTURED:")
        for source in report["perishable_lost"]:
            print(f"      {source} — this day cannot be recovered later")
        if "news" in report["perishable_lost"]:
            # WHY THE CAUSE IS NAMED RATHER THAN ASSUMED. PORTED from the
            # `alert-delivery` branch, which measured it on its first scheduled
            # run; I then read the same misleading line out of data/collect.log
            # while diagnosing the quota problem from the other direction.
            #
            # This printed "News needs NEWS_PROVIDER_API_KEY" unconditionally,
            # so the one message an operator reads at 22:00 sent them to fix a
            # credential that was already correct. A no-key day and a
            # quota-exhausted day need OPPOSITE actions -- set a variable, or
            # wait for the rolling window -- and naming the wrong one is worse
            # than naming none.
            if news.get("quota_exhausted"):
                print("      The key is set and working; the provider quota is spent.")
                print("      NewsAPI's free tier is 100 requests / 24h on a ROLLING")
                print("      window, so this clears on its own. Nothing to fix --")
                print("      but the uncovered tickers lose today.")
                print("      A Finnhub key (60 calls/MINUTE, free) removes the limit:")
                print("      set FINNHUB_API_KEY and it leads the provider order.")
            elif not os.getenv(NEWS_PROVIDER_API_KEY_ENV) and not os.getenv(
                FINNHUB_API_KEY_ENV
            ):
                print(f"      News needs {NEWS_PROVIDER_API_KEY_ENV} or")
                print(f"      {FINNHUB_API_KEY_ENV}. Until one is set, no OBSERVED")
                print("      event memory can ever exist, and F5 has only inferred")
                print("      analogs to work from.")
            else:
                print("      A key is set, so this is a provider or network failure")
                print("      rather than a setup one. Check the lines above for the")
                print("      reason the provider gave.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
