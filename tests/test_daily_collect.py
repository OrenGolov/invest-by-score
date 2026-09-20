"""Tests for daily collection and the coverage gate.

The behaviour under test is that a day which cannot be recovered is treated
differently from one that can, and that a collector which stops running
becomes visible rather than silent.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from unittest.mock import patch

import scripts.daily_collect as daily_collect
from core.config import (
    COLLECT_COVERAGE_MAX_MISSING,
    COLLECT_NEWS_TRACK_ANYWAY,
    COLLECT_NEWS_BATCH_SIZE,
    NEWS_PROVIDER_DAILY_LIMIT,
    COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS,
    COLLECT_MAX_TICKERS_PER_RUN,
    COLLECT_PERISHABLE_SOURCES,
    COLLECT_REPORT_PATH,
    COLLECT_SOURCES,
    COLLECT_THROTTLE_SECONDS,
    NEWS_LOOKBACK_DAYS,
)
from scripts.check_data_coverage import business_days_back
from scripts.daily_collect import (
    COLLECTORS,
    news_batch,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    _source_status,
    run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def PORTFOLIO_TICKERS_FIXTURE():
    from fetch_data import PORTFOLIO_TICKERS

    return PORTFOLIO_TICKERS


class PerishabilityTests(unittest.TestCase):
    """The classification the exit code depends on."""

    def test_news_is_perishable(self):
        # NEWS_LOOKBACK_DAYS is 7: a day not captured within a week is gone.
        self.assertIn("news", COLLECT_PERISHABLE_SOURCES)
        self.assertLessEqual(NEWS_LOOKBACK_DAYS, 30)

    def test_every_perishable_source_is_actually_collected(self):
        for source in COLLECT_PERISHABLE_SOURCES:
            self.assertIn(source, COLLECT_SOURCES)

    def test_rebuildable_sources_are_not_marked_perishable(self):
        # Prices and macro are re-fetchable; calling them perishable would
        # make the exit code fire on days nothing was actually lost.
        for source in ("prices", "macro"):
            self.assertNotIn(source, COLLECT_PERISHABLE_SOURCES)

    def test_every_declared_source_has_a_collector(self):
        for source in COLLECT_SOURCES:
            self.assertIn(source, COLLECTORS)


class SourceStatusTests(unittest.TestCase):
    def test_a_fully_successful_source_is_ok(self):
        self.assertEqual(
            _source_status("prices", {"attempted": 5, "ok": 5}), STATUS_OK
        )

    def test_a_partially_successful_source_is_partial(self):
        self.assertEqual(
            _source_status("prices", {"attempted": 5, "ok": 3}), STATUS_PARTIAL
        )

    def test_a_totally_failed_source_is_failed(self):
        self.assertEqual(
            _source_status("prices", {"attempted": 5, "ok": 0}), STATUS_FAILED
        )

    def test_an_unattempted_source_is_skipped_not_failed(self):
        # A dry run, or a source the caller did not request, is not an outage.
        self.assertEqual(_source_status("news", {"attempted": 0}), STATUS_SKIPPED)

    def test_news_with_some_coverage_is_partial(self):
        self.assertEqual(
            _source_status(
                "news", {"attempted": 3, "ok": 2, "unavailable": ["X"]}
            ),
            STATUS_PARTIAL,
        )

    def test_news_with_no_coverage_is_failed(self):
        self.assertEqual(
            _source_status(
                "news", {"attempted": 3, "ok": 0, "unavailable": ["A", "B", "C"]}
            ),
            STATUS_FAILED,
        )

    def test_events_blocked_by_news_is_skipped_not_failed(self):
        # News already reports the lost day; blaming events too would
        # double-count one loss and hide which stage actually broke.
        self.assertEqual(
            _source_status(
                "events", {"attempted": 0, "written": 0, "news_blocked": True}
            ),
            STATUS_SKIPPED,
        )

    def test_events_failing_on_its_own_is_failed(self):
        self.assertEqual(
            _source_status(
                "events", {"attempted": 4, "written": 0, "news_blocked": False}
            ),
            STATUS_FAILED,
        )


class RunVerdictTests(unittest.TestCase):
    """The aggregate verdict is what a scheduler acts on."""

    def test_a_dry_run_never_reports_a_lost_day(self):
        report = run(list(COLLECT_SOURCES), ["AAPL"], "2026-09-20", True)
        self.assertEqual(report["perishable_lost"], [])

    def test_losing_a_rebuildable_source_is_not_a_lost_day(self):
        report = run(["prices"], [], "2026-09-20", True)
        self.assertEqual(report["perishable_lost"], [])

    def test_the_report_names_every_requested_source(self):
        report = run(["prices", "macro"], ["AAPL"], "2026-09-20", True)
        self.assertEqual(set(report["statuses"]), {"prices", "macro"})

    def test_an_unknown_source_is_ignored_rather_than_crashing(self):
        report = run(["prices", "not_a_source"], ["AAPL"], "2026-09-20", True)
        self.assertIn("prices", report["statuses"])
        self.assertNotIn("not_a_source", report["statuses"])

    def test_a_collector_that_raises_does_not_stop_the_others(self):
        # Fail-SOFT per source: a dead provider must not cost the day's news.
        original = COLLECTORS["macro"]

        def exploding(*_args, **_kwargs):
            raise RuntimeError("provider down")

        COLLECTORS["macro"] = exploding
        try:
            report = run(["macro", "prices"], ["AAPL"], "2026-09-20", True)
        finally:
            COLLECTORS["macro"] = original
        self.assertIn("error", report["sources"]["macro"])
        self.assertEqual(report["statuses"]["prices"], STATUS_OK)

    def test_the_report_carries_a_dateable_timestamp(self):
        # The coverage gate proves the collector still runs by reading this.
        report = run(["prices"], ["AAPL"], "2026-09-20", True)
        parsed = datetime.fromisoformat(report["collected_at"])
        self.assertIsNotNone(parsed)


class CollectorIsATriggerTests(unittest.TestCase):
    """It must not become a second write path into the W6 ledger (W5)."""

    def setUp(self):
        self.tree = ast.parse(
            (REPO_ROOT / "scripts" / "daily_collect.py").read_text(encoding="utf-8")
        )

    def _names(self):
        called, imported = set(), set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    called.add(target.id)
                elif isinstance(target, ast.Attribute):
                    called.add(target.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported.add(alias.name)
        return called, imported

    def test_the_collector_never_appends_raw_records_itself(self):
        called, imported = self._names()
        self.assertNotIn("append_raw_records", called)
        self.assertNotIn("append_raw_records", imported)

    def test_the_collector_does_not_import_the_raw_store(self):
        _called, imported = self._names()
        self.assertFalse(any("raw_store" in name for name in imported))


class CoverageWindowTests(unittest.TestCase):
    def test_business_days_skip_weekends(self):
        # 2026-09-21 is a Monday; walking back must skip Sat/Sun.
        days = business_days_back(date(2026, 9, 21), 3)
        self.assertEqual(days, [date(2026, 9, 21), date(2026, 9, 18), date(2026, 9, 17)])
        for day in days:
            self.assertLess(day.weekday(), 5)

    def test_the_window_returns_exactly_what_was_asked_for(self):
        for count in (1, 5, 10, 22):
            self.assertEqual(len(business_days_back(date(2026, 9, 21), count)), count)

    def test_the_allowance_cannot_swallow_the_whole_window(self):
        # Otherwise a scheduler that NEVER ran would still pass the gate.
        self.assertLess(
            COLLECT_COVERAGE_MAX_MISSING, COLLECT_COVERAGE_WINDOW_BUSINESS_DAYS
        )

    def test_the_allowance_is_not_zero(self):
        # A provider outage or an unlisted market holiday must not break CI.
        self.assertGreater(COLLECT_COVERAGE_MAX_MISSING, 0)


class CollectConfigTests(unittest.TestCase):
    def test_sources_are_unique(self):
        self.assertEqual(len(set(COLLECT_SOURCES)), len(COLLECT_SOURCES))

    def test_the_throttle_is_not_negative(self):
        self.assertGreaterEqual(COLLECT_THROTTLE_SECONDS, 0)

    def test_a_run_covers_the_whole_portfolio(self):
        from fetch_data import PORTFOLIO_TICKERS

        self.assertGreaterEqual(COLLECT_MAX_TICKERS_PER_RUN, len(PORTFOLIO_TICKERS))

    def test_the_report_path_is_inside_data(self):
        self.assertTrue(str(COLLECT_REPORT_PATH).startswith("data/"))


class CloneSafetyTests(unittest.TestCase):
    """The gate must judge machines that COLLECT, not every checkout.

    The W6 ledger is tracked, so a clone made months from now carries records
    ending on the day they were committed. Judging that against today's
    calendar would fail CI on a machine that was never meant to collect —
    exactly the noise this gate must not produce. The collection report is
    gitignored, so its presence is what proves this machine collects.
    """

    def _repo(self, folder: Path, ledger_days, report: bool):
        (folder / "data" / "raw" / "yahoo_finance_chart").mkdir(parents=True)
        for day in ledger_days:
            (folder / "data" / "raw" / "yahoo_finance_chart" / f"{day}.jsonl").write_text(
                "{}" + chr(10), encoding="utf-8"
            )
        if report:
            (folder / "data" / "collection_report.jsonl").write_text(
                json.dumps({
                    "as_of": date.today().isoformat(),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "status": "OK", "statuses": {},
                }) + chr(10),
                encoding="utf-8",
            )
        for item in ("core", "scripts", "fetch_data.py"):
            source = REPO_ROOT / item
            if source.is_dir():
                shutil.copytree(source, folder / item)
            else:
                shutil.copy2(source, folder / item)

    def _run(self, ledger_days, report: bool) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "repo"
            folder.mkdir()
            self._repo(folder, ledger_days, report)
            result = subprocess.run(
                [sys.executable, str(folder / "scripts" / "check_data_coverage.py")],
                capture_output=True, text=True, cwd=str(folder),
            )
            return result.returncode

    def test_a_fresh_clone_with_a_stale_ledger_passes(self):
        # Nothing has stopped, because nothing ever started here.
        self.assertEqual(
            self._run(["2026-01-05", "2026-01-06", "2026-01-07"], report=False), 0
        )

    def test_a_collecting_machine_with_a_long_dead_ledger_fails(self):
        # The worst outage: not one day of the window survives. An earlier
        # version classified this as "source retired" and passed it silently.
        self.assertEqual(
            self._run(["2026-01-05", "2026-01-06", "2026-01-07"], report=True), 1
        )

    def test_a_collecting_machine_with_a_current_ledger_passes(self):
        current = [d.isoformat() for d in business_days_back(date.today(), 10)]
        self.assertEqual(self._run(current, report=True), 0)


class QuotaAwareNewsTests(unittest.TestCase):
    """The provider allows 100 calls/day and each ticker is one call.

    MEASURED: 77 tickers plus ad-hoc testing exhausted the quota, and every
    request returned HTTP 429. The cost is PER CALL, not per byte, so an ETF
    costs exactly what a stock does — funds are excluded for a different
    reason, and the batch is what protects the quota.
    """

    def test_the_batch_is_bounded(self):
        from fetch_data import PORTFOLIO_TICKERS

        batch, _skipped, _start = news_batch(list(PORTFOLIO_TICKERS))
        self.assertLessEqual(len(batch), COLLECT_NEWS_BATCH_SIZE)

    def test_the_batch_does_not_cover_the_whole_portfolio(self):
        from fetch_data import PORTFOLIO_TICKERS

        # If it did, no batching would happen — which is what produced the 429.
        self.assertLess(COLLECT_NEWS_BATCH_SIZE, len(PORTFOLIO_TICKERS))

    def test_the_batch_stays_inside_the_daily_ceiling(self):
        self.assertLessEqual(COLLECT_NEWS_BATCH_SIZE, NEWS_PROVIDER_DAILY_LIMIT)

    def test_sector_less_tickers_are_skipped_unless_allow_listed(self):
        from core.market_context import sector_for
        from fetch_data import PORTFOLIO_TICKERS

        batch, skipped, _start = news_batch(list(PORTFOLIO_TICKERS))
        tracked = {t.upper() for t in COLLECT_NEWS_TRACK_ANYWAY}
        funds = [t for t in PORTFOLIO_TICKERS if sector_for(t) is None]
        self.assertTrue(funds, "fixture expects the portfolio to hold funds")
        for fund in funds:
            if fund.upper() in tracked:
                self.assertNotIn(fund, skipped, fund)
            else:
                self.assertIn(fund, skipped, fund)
                self.assertNotIn(fund, batch, fund)

    def test_the_broad_index_fund_stays_excluded(self):
        # VOO tracks the whole S&P 500: its "news" IS the market, which E5
        # attribution already calls confounded.
        _batch, skipped, _start = news_batch(list(PORTFOLIO_TICKERS_FIXTURE()))
        self.assertIn("VOO", skipped)

    def test_the_allow_listed_thematic_funds_are_tracked(self):
        # SOXX and CIBR track ONE industry each, so a headline is closer to a
        # sector event than to broad commentary, and the portfolio holds many
        # of their constituents.
        _batch, skipped, _start = news_batch(list(PORTFOLIO_TICKERS_FIXTURE()))
        for etf in COLLECT_NEWS_TRACK_ANYWAY:
            self.assertNotIn(etf, skipped, etf)

    def test_the_allow_list_only_names_sector_less_tickers(self):
        # An allow-list entry that already HAS a sector would be a no-op
        # pretending to be a decision.
        from core.market_context import sector_for

        for ticker in COLLECT_NEWS_TRACK_ANYWAY:
            self.assertIsNone(sector_for(ticker), ticker)

    def test_a_short_list_is_not_batched(self):
        batch, _skipped, start = news_batch(["AAPL", "MSFT"])
        self.assertEqual(sorted(batch), ["AAPL", "MSFT"])
        self.assertEqual(start, 0)

    def test_rotation_covers_every_eligible_ticker(self):
        from core.market_context import sector_for
        from fetch_data import PORTFOLIO_TICKERS

        eligible = sorted(t for t in PORTFOLIO_TICKERS if sector_for(t) is not None)
        runs = -(-len(eligible) // COLLECT_NEWS_BATCH_SIZE)
        covered = set()
        for run in range(runs):
            start = (run * COLLECT_NEWS_BATCH_SIZE) % len(eligible)
            covered |= {
                eligible[(start + offset) % len(eligible)]
                for offset in range(COLLECT_NEWS_BATCH_SIZE)
            }
        self.assertEqual(covered, set(eligible))

    def test_the_cursor_advances_between_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            cursor = Path(folder) / "cursor.json"
            with patch.object(daily_collect, "COLLECT_NEWS_CURSOR_PATH", str(cursor)),                     patch.object(daily_collect, "REPO_ROOT", Path(folder)):
                from fetch_data import PORTFOLIO_TICKERS

                first, skipped, start = news_batch(list(PORTFOLIO_TICKERS))
                daily_collect._advance_cursor(
                    len(PORTFOLIO_TICKERS) - len(skipped), start
                )
                second, _s, second_start = news_batch(list(PORTFOLIO_TICKERS))
            self.assertNotEqual(start, second_start)
            self.assertNotEqual(first, second)

    def test_an_unreadable_cursor_starts_at_the_head(self):
        with tempfile.TemporaryDirectory() as folder:
            cursor = Path(folder) / "cursor.json"
            cursor.write_text("not json", encoding="utf-8")
            with patch.object(daily_collect, "COLLECT_NEWS_CURSOR_PATH", str(cursor)),                     patch.object(daily_collect, "REPO_ROOT", Path(folder)):
                from fetch_data import PORTFOLIO_TICKERS

                _batch, _skipped, start = news_batch(list(PORTFOLIO_TICKERS))
            self.assertEqual(start, 0)

    def test_a_quota_error_stops_the_run_early(self):
        # Burning the rest of the batch on doomed calls only deepens the
        # overage, and the cursor must NOT advance past untried tickers.
        calls = {"n": 0}

        def exhausted(ticker, as_of):
            calls["n"] += 1
            return {
                "status": "UNAVAILABLE",
                "reason": "news provider request failed: HTTP Error 429: Too Many Requests",
            }

        with tempfile.TemporaryDirectory() as folder:
            cursor = Path(folder) / "cursor.json"
            with patch("core.news_adapter.build_news_snapshot", exhausted),                     patch.object(daily_collect, "COLLECT_NEWS_CURSOR_PATH", str(cursor)),                     patch.object(daily_collect, "REPO_ROOT", Path(folder)):
                result = daily_collect.collect_news(
                    ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"] * 20, "2026-09-20", False
                )
                self.assertFalse(cursor.exists())
        self.assertTrue(result["quota_exhausted"])
        self.assertEqual(calls["n"], 1)

    def test_the_events_stage_reuses_the_news_batch(self):
        # forward() fetches news itself, so passing the full list would spend
        # the quota a second time on tickers this run already covered.
        source = (REPO_ROOT / "scripts" / "daily_collect.py").read_text(encoding="utf-8")
        events = source[source.index("def collect_events("):source.index("COLLECTORS =")]
        self.assertIn("news_batch(tickers)", events)
        self.assertNotIn("forward(tickers", events)


if __name__ == "__main__":
    unittest.main()
