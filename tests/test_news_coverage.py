"""News coverage metrics — the numbers that would have named the outage.

MEASURED 2026-10-06: news capture had been dead for three days and the only
visible symptom was 77 identically-worded alerts. Every fact that would have
identified the problem in one line — 0% coverage, the newest event 4 days old,
22 business days missing, 16 of them permanently unrecoverable — was computable
from data already on disk and computed by nobody.

These tests pin the arithmetic, and in particular the distinctions that make the
metrics honest rather than merely present: coverage counts only SUCCESSFUL
fetches, a run that predates request accounting is `unknown` rather than zero,
and staleness past the provider window is reported as irreversible instead of
just large.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from core.config import NEWS_LOOKBACK_DAYS
from core.news_coverage import (
    coverage_report,
    event_memory_freshness,
    news_coverage,
    oldest_missing_news_date,
    quota_consumption,
    render_coverage,
)

TODAY = date(2026, 10, 6)


def _report(as_of: str, news: dict, collected_at: str = "") -> dict:
    return {
        "as_of": as_of,
        "collected_at": collected_at or f"{as_of}T06:20:00+00:00",
        "sources": {"news": news},
    }


class NewsCoverageTests(unittest.TestCase):

    def test_an_attempt_that_failed_is_not_coverage(self):
        # THE DISTINCTION THAT MATTERS. On 2026-10-06 AMZN was attempted and
        # 429'd. Counting attempts would have reported coverage for a day that
        # captured nothing -- the precise reassurance that let this run three
        # days unnoticed.
        reports = [_report("2026-10-06", {
            "ok_tickers": [], "unavailable": ["AMZN"],
            "failure_kinds": {"quota_exceeded": 1},
        })]
        result = news_coverage(reports, {"AMZN", "AAPL"}, TODAY)
        self.assertEqual(result["coverage_pct"], 0.0)
        self.assertEqual(result["covered"], 0)
        self.assertEqual(result["failed_requests"], 1)
        self.assertIn("AMZN", result["missing"])

    def test_coverage_accumulates_across_runs_inside_the_window(self):
        # The rotation covers the universe over several days, so coverage must
        # be measured over the window rather than per run -- otherwise a
        # healthy 3-day sweep reads as permanent 33% coverage.
        reports = [
            _report("2026-10-05", {"ok_tickers": ["AAPL"]}),
            _report("2026-10-06", {"ok_tickers": ["MSFT"]}),
        ]
        result = news_coverage(reports, {"AAPL", "MSFT"}, TODAY)
        self.assertEqual(result["coverage_pct"], 100.0)

    def test_coverage_outside_the_provider_window_does_not_count(self):
        # News older than the lookback cannot be extended, so counting it as
        # current coverage would hide an active outage behind old successes.
        reports = [_report("2026-09-01", {"ok_tickers": ["AAPL"]})]
        result = news_coverage(reports, {"AAPL"}, TODAY)
        self.assertEqual(result["covered"], 0)

    def test_an_unknown_universe_is_inferred_rather_than_assumed(self):
        reports = [_report("2026-10-06", {
            "ok_tickers": ["AAPL"], "unavailable": ["MSFT"],
        })]
        result = news_coverage(reports, None, TODAY)
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["coverage_pct"], 50.0)


class QuotaConsumptionTests(unittest.TestCase):

    def test_a_run_predating_request_accounting_is_unknown_not_zero(self):
        # Reporting "0 requests spent" for a run that in fact exhausted the
        # quota is worse than reporting that the number is not known: it reads
        # as spare capacity that does not exist.
        reports = [_report("2026-10-06", {"ok": 0})]  # no requests_spent key
        result = quota_consumption(reports)
        self.assertEqual(result["unknown_runs"], 1)
        self.assertEqual(result["requests_spent"], 0)

    def test_spend_is_summed_across_runs_in_the_window(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        reports = [
            _report("2026-10-06", {"requests_spent": 12},
                    collected_at=(now - timedelta(hours=1)).isoformat()),
            _report("2026-10-06", {"requests_spent": 13},
                    collected_at=(now - timedelta(hours=2)).isoformat()),
            # Outside the 24h window: must not be counted.
            _report("2026-10-01", {"requests_spent": 99},
                    collected_at=(now - timedelta(hours=72)).isoformat()),
        ]
        result = quota_consumption(reports)
        self.assertEqual(result["requests_spent"], 25)
        self.assertEqual(result["runs_in_window"], 2)


class EventMemoryFreshnessTests(unittest.TestCase):

    def _store(self, records) -> Path:
        folder = tempfile.mkdtemp()
        path = Path(folder) / "event_memory.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        return path

    def test_staleness_past_the_provider_window_is_reported_as_expired(self):
        # Beyond NEWS_LOOKBACK_DAYS the underlying articles are unfetchable, so
        # the gap is permanent rather than pending. A reader who cannot tell
        # those apart will wait for a backfill that can never happen.
        stale = date(2026, 10, 6).toordinal() - (NEWS_LOOKBACK_DAYS + 3)
        path = self._store([{
            "ticker": "AAPL", "provenance": "observed",
            "published_time": date.fromordinal(stale).isoformat() + "T00:00:00Z",
        }])
        result = event_memory_freshness(path, TODAY)
        self.assertTrue(result["expired"])
        self.assertEqual(result["stale_days"], NEWS_LOOKBACK_DAYS + 3)

    def test_recent_memory_is_not_expired(self):
        path = self._store([{
            "ticker": "AAPL", "provenance": "observed",
            "published_time": "2026-10-05T00:00:00Z",
        }])
        result = event_memory_freshness(path, TODAY)
        self.assertFalse(result["expired"])
        self.assertEqual(result["stale_days"], 1)

    def test_a_truncated_line_does_not_lose_the_rest_of_the_store(self):
        folder = tempfile.mkdtemp()
        path = Path(folder) / "event_memory.jsonl"
        path.write_text(
            '{"ticker": "AAPL", "provenance": "observed", '
            '"published_time": "2026-10-05T00:00:00Z"}\n'
            '{"ticker": "MSFT", truncated...\n',
            encoding="utf-8",
        )
        result = event_memory_freshness(path, TODAY)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["distinct_tickers"], 1)

    def test_a_missing_store_reports_empty_rather_than_raising(self):
        result = event_memory_freshness(Path("does/not/exist.jsonl"), TODAY)
        self.assertEqual(result["records"], 0)
        self.assertIsNone(result["newest_event_date"])


class MissingDayTests(unittest.TestCase):

    def test_days_past_the_window_are_counted_as_unrecoverable(self):
        result = oldest_missing_news_date([], TODAY, window_days=30)
        # Every business day in the window is missing, and those older than the
        # lookback can never be refetched at any price.
        self.assertGreater(result["missing_business_days"], 0)
        self.assertGreater(result["unrecoverable_days"], 0)
        self.assertLess(result["unrecoverable_days"], result["missing_business_days"])

    def test_a_day_with_a_successful_capture_is_not_missing(self):
        reports = [_report("2026-10-06", {"ok": 5})]
        result = oldest_missing_news_date(reports, TODAY, window_days=1)
        self.assertNotEqual(result["oldest_missing"], TODAY.isoformat())

    def test_weekends_are_not_counted_as_outages(self):
        # A monitor that cannot tell a Saturday from a failure reports an
        # outage every Monday.
        saturday = date(2026, 10, 3)
        self.assertEqual(saturday.weekday(), 5)
        result = oldest_missing_news_date([], date(2026, 10, 4), window_days=1)
        self.assertEqual(result["missing_business_days"], 0)


class RenderingTests(unittest.TestCase):

    def test_the_rendered_report_states_each_number_and_its_meaning(self):
        folder = tempfile.mkdtemp()
        memory = Path(folder) / "event_memory.jsonl"
        memory.write_text(json.dumps({
            "ticker": "AAPL", "provenance": "observed",
            "published_time": "2026-10-05T00:00:00Z",
        }), encoding="utf-8")
        reports = [_report("2026-10-06", {
            "ok_tickers": ["AAPL"], "requests_spent": 25, "ok": 1,
        })]
        text = render_coverage(coverage_report(reports, memory, {"AAPL"}, TODAY))
        for expected in ("coverage", "quota", "failures", "gaps", "event memory"):
            self.assertIn(expected, text)

    def test_the_rendered_report_is_ascii(self):
        # It lands in a 22:00 log, an email and a Telegram message; a cp1252
        # console turns a non-ASCII dash into a replacement character.
        reports = [_report("2026-10-06", {
            "ok_tickers": [], "unavailable": ["AMZN"],
            "failure_kinds": {"quota_exceeded": 1}, "requests_spent": 1,
        })]
        text = render_coverage(
            coverage_report(reports, Path("none.jsonl"), {"AMZN"}, TODAY)
        )
        text.encode("ascii")  # raises if anything non-ASCII crept in


if __name__ == "__main__":
    unittest.main()
