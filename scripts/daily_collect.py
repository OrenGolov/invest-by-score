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

Scheduling (Windows Task Scheduler, business days):
    powershell -ExecutionPolicy Bypass -File scripts\\install_daily_task.ps1

    Use the installer rather than a hand-written schtasks line. It resolves
    the virtualenv (both `venv` and `.venv` spellings) and REFUSES to
    register when that interpreter cannot import what the collector needs.
    MEASURED 2026-10-04: the bare interpreter on this machine has pandas but
    NOT sklearn, so a task pointed at it collects happily while every model
    path fails at 22:00 into a log nobody reads.
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
    COLLECT_FUNDAMENTALS_BATCH_SIZE,
    COLLECT_FUNDAMENTALS_CURSOR_PATH,
    COLLECT_FUNDAMENTALS_THROTTLE_SECONDS,
    COLLECT_MAX_TICKERS_PER_RUN,
    COLLECT_NEWS_ADVANCE_CURSOR_ON_QUOTA,
    COLLECT_NEWS_BATCH_SIZE,
    COLLECT_NEWS_CURSOR_PATH,
    COLLECT_NEWS_PRIORITIZE_STALE,
    COLLECT_NEWS_REQUEST_BUDGET,
    COLLECT_NEWS_REUSE_SNAPSHOTS,
    COLLECT_NEWS_SKIP_SECTORLESS,
    COLLECT_NEWS_TRACK_ANYWAY,
    COLLECT_PERISHABLE_SOURCES,
    COLLECT_REPORT_PATH,
    COLLECT_SOURCES,
    COLLECT_THROTTLE_SECONDS,
    NEWS_PROVIDER_API_KEY_ENV,
)
from core.news_adapter import (  # noqa: E402
    FAILURE_AUTH,
    FAILURE_NO_ARTICLES,
    FAILURE_GUIDANCE,
    FAILURE_NO_KEY,
    FAILURE_QUOTA,
    FAILURE_UNKNOWN,
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


def _rotate(tickers, batch_size: int, cursor_path: str) -> tuple[list[str], int]:
    """The slice of `tickers` this run should cover, and where it started.

    The same rotation the news stage uses, factored out because fundamentals hit
    the identical wall: a provider allowance smaller than the universe. Returns
    the whole list unchanged when it already fits.
    """
    ordered = sorted(tickers)
    if not ordered or len(ordered) <= batch_size:
        return ordered, 0
    cursor = 0
    try:
        raw = (REPO_ROOT / cursor_path).read_text(encoding="utf-8")
        cursor = int(json.loads(raw)["cursor"])
    except Exception:
        cursor = 0  # a missing or unreadable cursor starts at the head
    start = cursor % len(ordered)
    batch = [ordered[(start + offset) % len(ordered)] for offset in range(batch_size)]
    return batch, start


def _advance_simple_cursor(cursor_path: str, start: int, served: int, total: int) -> None:
    """Move a rotation cursor on by the work actually served.

    Same rule as the news cursor, and for the same measured reason: advancing by
    the intended batch would skip tickers a short run never reached, while never
    advancing pins the rotation on whatever failed first.
    """
    if total <= 0:
        return
    step = max(1, int(served))
    try:
        path = REPO_ROOT / cursor_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "cursor": (start + step) % total,
                "updated": datetime.now(timezone.utc).isoformat(),
                "advanced_by": step,
            }),
            encoding="utf-8",
        )
    except Exception as exc:  # a cursor failure must not fail the collection
        LOGGER.warning("could not advance the cursor %s: %s", cursor_path, exc)


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

    # STALENESS FIRST, within the batch the cursor already chose.
    #
    # The cursor decides WHICH tickers this run covers (that is what keeps the
    # rotation fair); this only decides the ORDER they are fetched in. The
    # distinction matters when the quota dies mid-run: whatever the run managed
    # to fetch should be the tickers closest to losing their news permanently,
    # not whichever happened to sort first alphabetically.
    #
    # Deliberately does NOT change the batch membership, because reordering the
    # selection would let a permanently-stale ticker (one the provider has no
    # articles for) monopolise every run and starve the rotation.
    if COLLECT_NEWS_PRIORITIZE_STALE:
        try:
            batch = _order_by_staleness(batch)
        except Exception as exc:  # ordering is an optimisation, never a gate
            LOGGER.warning("could not order the batch by staleness: %s", exc)

    return batch, skipped, start


def _last_news_capture() -> dict[str, str]:
    """The most recent date each ticker was SUCCESSFULLY fetched.

    Read from the collection report ledger, which costs no quota. Only
    `ok_tickers` counts: a ticker that was attempted and failed has no news,
    so treating the attempt as coverage would push it to the back of the queue
    precisely when it most needs fetching.
    """
    path = REPO_ROOT / COLLECT_REPORT_PATH
    seen: dict[str, str] = {}
    if not path.exists():
        return seen
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return seen
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            report = json.loads(line)
        except ValueError:
            continue
        as_of = str(report.get("as_of") or "")[:10]
        if not as_of:
            continue
        news = (report.get("sources") or {}).get("news") or {}
        for ticker in news.get("ok_tickers") or []:
            key = str(ticker).upper()
            if as_of > seen.get(key, ""):
                seen[key] = as_of
    return seen


def _order_by_staleness(batch: list[str]) -> list[str]:
    """Stalest first. Never-fetched tickers lead, then oldest capture date.

    Ties break alphabetically so the order is deterministic — a reproducible
    run order is worth more than an arbitrary tiebreak, and the W-series
    reproducibility gates depend on it.
    """
    last = _last_news_capture()
    # "" sorts before any ISO date, so a never-covered ticker is stalest.
    return sorted(batch, key=lambda t: (last.get(str(t).upper(), ""), str(t).upper()))


def _advance_cursor(eligible_count: int, start: int, served: int | None = None,
                    reason: str = "batch_complete") -> None:
    """Move the cursor on, so the next run fetches tickers it has not seen.

    `served` is how many tickers this run actually REACHED (fetched, whether
    the fetch succeeded or returned no articles). It defaults to the full
    batch.

    **Why progress is measured in work completed.** The old version always
    advanced by the full batch, and its caller skipped the call entirely on a
    429 to avoid "skipping" tickers. Both halves were wrong:

      * Never advancing means the next run restarts at the same head and fails
        on the same ticker. MEASURED: the cursor sat at 5 for two days while
        news capture was dead, and only 26 of 75 tickers had any memory.
      * Advancing by the full batch after a partial run WOULD skip the
        tickers the run never reached.

    Advancing by `served` does neither. A run stopped after 3 of 25 tickers
    moves the cursor 3, so the next run begins exactly where this one stopped.
    Forward progress is guaranteed and nothing is skipped.
    """
    if eligible_count <= 0:
        return
    step = COLLECT_NEWS_BATCH_SIZE if served is None else max(0, int(served))
    if step <= 0:
        # Nothing was served -- not even one ticker. Advancing by zero would
        # retry the same head forever, which is the deadlock this function
        # exists to prevent, so move on by ONE to guarantee the rotation
        # cannot be pinned by a single permanently-failing ticker.
        step = 1
        reason = f"{reason}_no_progress_nudge"
    cursor_file = REPO_ROOT / COLLECT_NEWS_CURSOR_PATH
    try:
        cursor_file.parent.mkdir(parents=True, exist_ok=True)
        cursor_file.write_text(
            json.dumps({
                "cursor": (start + step) % eligible_count,
                "updated": datetime.now(timezone.utc).isoformat(),
                "advanced_by": step,
                "reason": reason,
            }),
            encoding="utf-8",
        )
    except Exception as exc:  # a cursor failure must not lose the day's news
        LOGGER.warning("could not advance the news cursor: %s", exc)


# The news snapshots this run fetched, keyed by (ticker, as_of).
#
# WHY A MODULE-LEVEL CACHE. `collect_news` and `collect_events` both need the
# same ticker's news on the same date, and `core.news_adapter` caches nothing,
# so the events stage used to spend the quota a SECOND time on tickers the news
# stage had already fetched -- doubling a 25-ticker run to 50 requests against
# a 50/12h rolling allowance. MEASURED 2026-10-06: the news stage stopped at 1
# ticker on a 429 while the events stage went on to call all 40 again.
#
# Scoped to one process, cleared per run, and keyed by as_of so a backfill
# across dates cannot serve one date's articles for another.
_NEWS_CACHE: dict[tuple[str, str], dict] = {}


def _reset_news_cache() -> None:
    _NEWS_CACHE.clear()


def news_snapshot_cached(ticker: str, as_of: str) -> dict:
    """One news snapshot per (ticker, as_of) per run, however many stages ask."""
    from core.news_adapter import build_news_snapshot

    key = (str(ticker).upper(), str(as_of))
    if COLLECT_NEWS_REUSE_SNAPSHOTS and key in _NEWS_CACHE:
        return _NEWS_CACHE[key]
    snapshot = build_news_snapshot(ticker, as_of)
    if COLLECT_NEWS_REUSE_SNAPSHOTS:
        _NEWS_CACHE[key] = snapshot
    return snapshot


def _failure_kind(snapshot: dict) -> str:
    """The typed reason a snapshot is UNAVAILABLE.

    Reads the `failure_kind` the adapter now attaches. The substring fallback
    exists only for a snapshot produced by an older adapter; it is NOT the
    primary path, because searching prose for "429" is exactly the bug that
    let a quota day be reported as an outage.
    """
    kind = str(snapshot.get("failure_kind") or "").strip()
    if kind:
        return kind
    reason = str(snapshot.get("reason", ""))
    if "429" in reason or "too many requests" in reason.lower():
        return FAILURE_QUOTA
    return FAILURE_UNKNOWN


def collect_news(tickers, as_of: str, dry_run: bool) -> dict:
    """THE PERISHABLE ONE. A day missed here is a day lost permanently.

    Quota-aware in three ways the previous version was not:

      1. A hard REQUEST BUDGET is counted down, so the run cannot overrun the
         rolling window even if the batch arithmetic is wrong.
      2. Failures are classified by KIND, so a quota stop (self-clearing, keep
         the rotation moving) is handled differently from an auth failure
         (nothing will work until a human acts) and from an outage.
      3. The cursor advances by WORK SERVED, so a run that stops early resumes
         where it stopped instead of restarting at the same failing head.
    """
    batch, skipped, start = news_batch(tickers)
    eligible_count = len(tickers) - len(skipped)

    ok, unavailable, failed = 0, [], []
    quiet: list[str] = []       # fetched fine, provider simply had no articles
    ok_tickers: list[str] = []  # so an alert can say "fetched, no articles"
                                # rather than lumping it in with "not tried"
    served = 0            # tickers this run actually reached
    requests_spent = 0    # provider calls made, for the budget and the report
    kinds: dict[str, int] = {}
    stop_kind = ""        # the kind that ended the run early, if any

    for ticker in batch:
        if dry_run:
            ok += 1
            ok_tickers.append(ticker)
            served += 1
            continue
        if requests_spent >= COLLECT_NEWS_REQUEST_BUDGET:
            # The budget, not the provider, stopped us. Everything already
            # fetched is kept and the cursor still advances by what was served.
            stop_kind = "budget_exhausted"
            break
        try:
            snapshot = news_snapshot_cached(ticker, as_of)
            requests_spent += 1
            served += 1
            status = str(snapshot.get("status", "")).upper()
            if status == "UNAVAILABLE":
                kind = _failure_kind(snapshot)
                if kind == FAILURE_NO_ARTICLES:
                    # NOT A FAILURE AND NOT A GAP. The provider answered and had
                    # nothing for this ticker in the window. MEASURED 2026-10-07,
                    # five tickers took this path and were reported as
                    # `unknown_error`, which reads as five broken requests when
                    # nothing broke. Absence of news is an observation, so it is
                    # counted as served-and-quiet rather than banked as an error.
                    quiet.append(ticker)
                    served += 1
                    time.sleep(COLLECT_THROTTLE_SECONDS)
                    continue
                unavailable.append(ticker)
                kinds[kind] = kinds.get(kind, 0) + 1
                # THREE kinds end the run; the rest are per-ticker noise.
                # Quota and auth mean every further call is certain to fail,
                # so continuing would waste the window and deepen an overage.
                if kind in (FAILURE_QUOTA, FAILURE_AUTH, FAILURE_NO_KEY):
                    stop_kind = kind
                    break
            else:
                ok += 1
                ok_tickers.append(ticker)
        except Exception as exc:
            LOGGER.warning("news fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
            served += 1
            kinds[FAILURE_UNKNOWN] = kinds.get(FAILURE_UNKNOWN, 0) + 1
        time.sleep(COLLECT_THROTTLE_SECONDS)

    quota_exhausted = stop_kind == FAILURE_QUOTA

    if not dry_run:
        # ALWAYS advance, by what was served. An auth failure or a missing key
        # serves nothing, so the nudge in `_advance_cursor` moves on by one
        # rather than pinning the rotation on a ticker that cannot be fetched.
        if stop_kind and not COLLECT_NEWS_ADVANCE_CURSOR_ON_QUOTA:
            pass  # legacy behaviour, retained only behind the flag
        else:
            _advance_cursor(
                eligible_count, start, served=served,
                reason=stop_kind or "batch_complete",
            )

    return {
        "attempted": len(batch), "ok": ok,
        "unavailable": unavailable, "failed": failed,
        "ok_tickers": ok_tickers,
        "no_articles": quiet,
        "batch": len(batch), "eligible": eligible_count,
        "skipped_sectorless": sorted(skipped),
        "cursor_start": start,
        "served": served,
        "requests_spent": requests_spent,
        "request_budget": COLLECT_NEWS_REQUEST_BUDGET,
        "failure_kinds": kinds,
        "stopped_early_because": stop_kind,
        "quota_exhausted": quota_exhausted,
    }


def collect_fundamentals(tickers, as_of: str, dry_run: bool) -> dict:
    """Vintage capture: the values are re-fetchable, today's reading is not.

    Paced by COLLECT_FUNDAMENTALS_THROTTLE_SECONDS rather than the shared
    throttle. MEASURED 2026-10-06, Alpha Vantage's free tier wants ~1 request
    per second and answers a faster one with HTTP 200 plus an advisory body —
    so a throttled call does NOT raise. It returns a snapshot with no metrics,
    the score substitutes neutral defaults, and every ticker ends up sharing
    one identical vector. Being rate-limited is indistinguishable from having
    no key at all unless it is counted, which is what `throttled` is for.
    """
    from fetch_data import fetch_fundamental_snapshot

    # ROTATE, like news. MEASURED 2026-10-07: attempting all 77 against a
    # 25/day allowance spent the day on the first ~25 and failed the rest, and a
    # second run that day found nothing left. A batch plus a cursor sweeps the
    # universe in 4 runs and keeps 5 calls spare.
    batch, start = _rotate(
        tickers, COLLECT_FUNDAMENTALS_BATCH_SIZE, COLLECT_FUNDAMENTALS_CURSOR_PATH
    )

    ok, failed, throttled = 0, [], []
    for ticker in batch:
        if dry_run:
            ok += 1
            continue
        try:
            snapshot = fetch_fundamental_snapshot(ticker, as_of) or {}
            # A snapshot whose valuation metrics are entirely null came back
            # empty-handed, whatever its status field says.
            metrics = snapshot.get("valuation_metrics") or {}
            if metrics and all(value is None for value in metrics.values()):
                throttled.append(ticker)
            else:
                ok += 1
        except Exception as exc:
            LOGGER.warning("fundamentals fetch failed for %s: %s", ticker, exc)
            failed.append(ticker)
        time.sleep(COLLECT_FUNDAMENTALS_THROTTLE_SECONDS)

    if not dry_run:
        _advance_simple_cursor(
            COLLECT_FUNDAMENTALS_CURSOR_PATH, start, len(batch), len(tickers)
        )

    result = {
        "attempted": len(batch), "ok": ok, "failed": failed,
        "batch": len(batch), "eligible": len(tickers), "cursor_start": start,
    }
    if throttled:
        # Named separately from `failed`: nothing broke, the allowance ran out.
        # The values are re-fetchable tomorrow, so this is not a lost day.
        result["no_metrics_returned"] = throttled
    return result


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

    # The SAME batch the news stage fetched, and now the same SNAPSHOTS too.
    # Passing `news_fetcher` lets `forward` read this run's cache instead of
    # calling the provider again: MEASURED, the refetch doubled a run's cost
    # and was the difference between a batch that fits the rolling window and
    # one that cannot complete. The cursor is NOT advanced by this call;
    # collect_news owns it.
    batch, _skipped, _start = news_batch(tickers)
    report = forward(batch, as_of, False, None, news_fetcher=news_snapshot_cached)
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

    # One run, one set of news snapshots. Cleared at the start rather than the
    # end so a caller that invokes run() twice in a process (the tests do)
    # cannot be served the previous run's articles.
    _reset_news_cache()

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

    # The TREND, not just this run. A single run's numbers cannot show that
    # capture has been dead for two days, which is the fact that mattered most
    # on 2026-10-06 and the one nothing printed.
    if not args.dry_run:
        try:
            from core.collection_monitor import load_reports
            from core.news_coverage import coverage_report, render_coverage

            history = load_reports(REPO_ROOT / COLLECT_REPORT_PATH)
            metrics = coverage_report(
                history,
                REPO_ROOT / "data" / "event_memory.jsonl",
                set(tickers),
            )
            print()
            print(render_coverage(metrics))
        except Exception as exc:  # metrics must never fail a collection run
            LOGGER.warning("could not render coverage metrics: %s", exc)

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
    stopped = str(news.get("stopped_early_because") or "")
    if stopped:
        print()
        guidance = FAILURE_GUIDANCE.get(stopped)
        if stopped == "budget_exhausted":
            print(f"  PER-RUN BUDGET REACHED ({news.get('requests_spent')} requests):")
            print("      the run stopped itself before the provider did. This is the")
            print("      guard that keeps one run from eating the next run's window.")
        elif stopped == FAILURE_QUOTA:
            print("  PROVIDER QUOTA EXHAUSTED: the run stopped early.")
            print(f"      {guidance}")
        elif stopped in (FAILURE_AUTH, FAILURE_NO_KEY):
            print("  AUTHENTICATION PROBLEM — THIS DOES NOT CLEAR ON ITS OWN:")
            print(f"      {guidance}")
        else:
            print(f"  RUN STOPPED EARLY ({stopped}):")
            print(f"      {guidance or 'see the provider message above'}")
        print(f"      Served {news.get('served')} ticker(s); the cursor advanced by")
        print("      that many, so the next run RESUMES where this one stopped")
        print("      instead of retrying the same head of the list.")

    if report["perishable_lost"]:
        print()
        print("  PERISHABLE DATA WAS NOT CAPTURED:")
        for source in report["perishable_lost"]:
            print(f"      {source} — this day cannot be recovered later")
        if "news" in report["perishable_lost"]:
            # WHY THE CAUSE IS NAMED RATHER THAN ASSUMED. This used to print
            # "News needs NEWS_PROVIDER_API_KEY" unconditionally. MEASURED
            # 2026-10-04 on the first scheduled run, the key WAS set and the
            # real cause was the 100/day quota (HTTP 429) — so the one
            # message the operator reads at 22:00 sent them to fix a
            # credential that was already correct. A no-key day and a
            # quota-exhausted day need different actions: set a variable, or
            # wait for the rolling window.
            if news.get("quota_exhausted"):
                print("      The key is set and working; the provider's quota is spent.")
                print("      NewsAPI's free tier is 100 requests / 24h, enforced as")
                print("      50 / 12h on a ROLLING window, so this clears on its own.")
                print("      Nothing to fix — but the uncovered tickers lose today.")
            elif not os.getenv(NEWS_PROVIDER_API_KEY_ENV):
                print("      News needs NEWS_PROVIDER_API_KEY. Until it is set, no OBSERVED")
                print("      event memory can ever exist, and F5 has only inferred")
                print("      analogs to work from.")
            else:
                print("      The key is set, so this is a provider or network")
                print("      failure rather than a setup one. Check the lines above")
                print("      for the reason the provider gave.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
