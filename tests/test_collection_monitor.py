"""Tests for collection monitoring.

Pure and deterministic: `today` is always injected, so no test depends on
the wall clock, and nothing touches a network or a delivery channel.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from core.collection_monitor import (
    explain_ticker_news_gap,
    SEVERITY_CRITICAL,
    SEVERITY_NONE,
    CollectionVerdict,
    business_days_between,
    evaluate_collection,
    last_successful_capture,
    load_reports,
    render_verdict,
)
from core.config import COLLECTION_MAX_SILENT_BUSINESS_DAYS


def _report(as_of="2026-10-04", status="OK", lost=None, news="OK", quota=False):
    return {
        "as_of": as_of,
        "status": status,
        "perishable_lost": list(lost or []),
        "sources": {
            "news": {"status": news, "quota_exhausted": quota},
            "prices": {"status": "OK"},
        },
    }


class BusinessDayTests(unittest.TestCase):
    def test_friday_to_monday_is_one_business_day(self):
        self.assertEqual(
            business_days_between(date(2026, 10, 2), date(2026, 10, 5)), 1
        )

    def test_a_weekend_alone_is_zero(self):
        """A Saturday with no collection is not a failure."""
        self.assertEqual(
            business_days_between(date(2026, 10, 2), date(2026, 10, 3)), 0
        )

    def test_seven_calendar_days_is_five_business_days(self):
        self.assertEqual(
            business_days_between(date(2026, 9, 27), date(2026, 10, 4)), 5
        )

    def test_the_same_day_is_zero(self):
        self.assertEqual(
            business_days_between(date(2026, 10, 4), date(2026, 10, 4)), 0
        )

    def test_an_end_before_the_start_is_zero_not_negative(self):
        self.assertEqual(
            business_days_between(date(2026, 10, 4), date(2026, 10, 1)), 0
        )


class QuotaIsNotAnAlertTests(unittest.TestCase):
    """The judgement most likely to be questioned, so tested explicitly."""

    def test_a_single_quota_day_does_not_alert(self):
        verdict = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED", quota=True),
            [_report(as_of="2026-10-03")],
            today=date(2026, 10, 4),
        )
        self.assertFalse(verdict.should_alert)

    def test_a_quota_day_still_explains_itself_in_the_body(self):
        """Suppressed is not hidden: the reader can still see what happened."""
        verdict = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED", quota=True),
            [_report(as_of="2026-10-03")],
            today=date(2026, 10, 4),
        )
        self.assertIn("quota", verdict.body.lower())

    def test_persistent_quota_starvation_DOES_alert(self):
        """The case that makes the suppression safe.

        A quota day alone is expected. Quota every day, with no successful
        news capture for longer than the threshold, is the collector
        effectively stopped — and must fire despite the quota suppression.
        """
        history = [
            _report(as_of="2026-09-25"),
            _report(as_of="2026-09-28", status="FAILED", lost=["news"], news="FAILED"),
            _report(as_of="2026-10-01", status="FAILED", lost=["news"], news="FAILED"),
        ]
        verdict = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED", quota=True),
            history,
            today=date(2026, 10, 4),
        )
        self.assertTrue(verdict.should_alert)
        self.assertEqual(verdict.severity, SEVERITY_CRITICAL)
        self.assertTrue(any("business days" in r for r in verdict.reasons))


class PerishableLossTests(unittest.TestCase):
    def test_news_loss_without_quota_is_critical(self):
        verdict = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED"),
            [_report(as_of="2026-10-03")],
            today=date(2026, 10, 4),
        )
        self.assertTrue(verdict.should_alert)
        self.assertEqual(verdict.severity, SEVERITY_CRITICAL)

    def test_the_alert_says_the_day_cannot_be_recovered(self):
        verdict = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED"),
            [_report(as_of="2026-10-03")],
            today=date(2026, 10, 4),
        )
        self.assertIn("cannot be recovered", verdict.body)

    def test_a_clean_run_does_not_alert(self):
        verdict = evaluate_collection(
            _report(), [_report(as_of="2026-10-03")], today=date(2026, 10, 4)
        )
        self.assertFalse(verdict.should_alert)
        self.assertEqual(verdict.severity, SEVERITY_NONE)

    def test_a_clean_run_still_reports_the_source_summary(self):
        verdict = evaluate_collection(_report(), [], today=date(2026, 10, 4))
        self.assertIn("news", verdict.body)
        self.assertIn("prices", verdict.body)


class SilenceTests(unittest.TestCase):
    def test_no_run_at_all_is_critical(self):
        verdict = evaluate_collection(None, [], today=date(2026, 10, 4))
        self.assertTrue(verdict.should_alert)
        self.assertTrue(any("did not run" in r for r in verdict.reasons))

    def test_a_successful_run_today_clears_a_historical_gap(self):
        """Today's success means the collector is working NOW.

        A historical gap is real but is `check_data_coverage`'s job — a
        monitor that alerts on it would fire forever, because the gap
        never heals.
        """
        verdict = evaluate_collection(
            _report(),
            [_report(as_of="2026-09-25")],
            today=date(2026, 10, 4),
        )
        self.assertFalse(verdict.should_alert)

    def test_the_threshold_is_respected_exactly(self):
        """At the threshold it stays quiet; one business day past, it fires."""
        at_threshold = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED", quota=True),
            [_report(as_of="2026-10-02")],
            today=date(2026, 10, 6),
        )
        self.assertEqual(
            business_days_between(date(2026, 10, 2), date(2026, 10, 6)),
            COLLECTION_MAX_SILENT_BUSINESS_DAYS,
        )
        self.assertFalse(at_threshold.should_alert)

        past_threshold = evaluate_collection(
            _report(status="FAILED", lost=["news"], news="FAILED", quota=True),
            [_report(as_of="2026-10-02")],
            today=date(2026, 10, 7),
        )
        self.assertTrue(past_threshold.should_alert)


class LastSuccessTests(unittest.TestCase):
    def test_the_newest_success_wins(self):
        reports = [_report(as_of="2026-09-30"), _report(as_of="2026-10-02")]
        self.assertEqual(
            last_successful_capture(reports, "news"), date(2026, 10, 2)
        )

    def test_a_failed_day_is_not_a_success(self):
        reports = [
            _report(as_of="2026-09-30"),
            _report(as_of="2026-10-02", status="FAILED", lost=["news"], news="FAILED"),
        ]
        self.assertEqual(
            last_successful_capture(reports, "news"), date(2026, 9, 30)
        )

    def test_partial_counts_as_captured(self):
        """PARTIAL means some tickers were fetched, so news did arrive."""
        reports = [_report(as_of="2026-10-02", news="PARTIAL")]
        self.assertEqual(
            last_successful_capture(reports, "news"), date(2026, 10, 2)
        )

    def test_no_history_is_none_not_an_error(self):
        self.assertIsNone(last_successful_capture([], "news"))

    def test_an_unparseable_date_is_skipped(self):
        reports = [_report(as_of="not-a-date"), _report(as_of="2026-10-02")]
        self.assertEqual(
            last_successful_capture(reports, "news"), date(2026, 10, 2)
        )


class LoadReportTests(unittest.TestCase):
    def test_a_missing_file_is_empty_not_an_error(self):
        self.assertEqual(load_reports(Path("does/not/exist.jsonl")), [])

    def test_a_truncated_line_does_not_lose_the_rest(self):
        """A half-written final line is normal in an append-only log."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.jsonl"
            path.write_text(
                json.dumps(_report(as_of="2026-10-01")) + "\n"
                + '{"as_of": "2026-10-02", "trunc',
                encoding="utf-8",
            )
            reports = load_reports(path)
        self.assertEqual(len(reports), 1)


class VerdictContractTests(unittest.TestCase):
    def test_an_alert_must_name_a_reason(self):
        with self.assertRaises(ValueError):
            CollectionVerdict(
                should_alert=True, severity=SEVERITY_CRITICAL, reasons=()
            )

    def test_an_alert_cannot_carry_severity_none(self):
        with self.assertRaises(ValueError):
            CollectionVerdict(
                should_alert=True, severity=SEVERITY_NONE, reasons=("x",)
            )

    def test_a_quiet_verdict_needs_no_reason(self):
        verdict = CollectionVerdict(should_alert=False, severity=SEVERITY_NONE)
        self.assertFalse(verdict.should_alert)

    def test_render_names_the_reason_when_alerting(self):
        verdict = evaluate_collection(None, [], today=date(2026, 10, 4))
        self.assertIn("did not run", render_verdict(verdict))

    def test_render_says_so_when_quiet(self):
        verdict = evaluate_collection(_report(), [], today=date(2026, 10, 4))
        self.assertIn("nothing worth alerting", render_verdict(verdict))


class TickerExplanationTests(unittest.TestCase):
    """The 77-identical-alerts bug, pinned.

    MEASURED 2026-10-06, the digest told the operator that every one of 77
    tickers "was not looked at today" because the collector "rotates 75 of the
    eligible tickers per run". The batch was 40, 75 was the ELIGIBLE count, and
    AMZN -- named among the un-looked-at -- was the single ticker that had been
    fetched. These tests exist so one explanation can never again be reused for
    causes that need different responses.
    """

    REPORT = {
        "as_of": "2026-10-06",
        "sources": {
            "news": {
                "batch": 25,
                "eligible": 75,
                "unavailable": ["AMZN"],
                "ok_tickers": ["MSFT"],
                "skipped_sectorless": ["VOO", "NASA"],
                "failure_kinds": {"quota_exceeded": 1},
                "stopped_early_because": "quota_exceeded",
                "served": 1,
            }
        },
    }

    def test_a_fetched_and_failed_ticker_is_not_called_unvisited(self):
        # THE EXACT INVERSION THAT SHIPPED. AMZN was fetched and 429'd.
        detail = explain_ticker_news_gap("AMZN", "2026-10-06", self.REPORT)
        self.assertEqual(detail["kind"], "quota_exceeded")
        self.assertIn("WAS fetched", detail["reason"])
        self.assertNotIn("not looked at", detail["reason"])
        self.assertNotIn("not scheduled", detail["reason"])

    def test_an_unscheduled_ticker_reports_the_real_batch_size(self):
        # The old text invented "75 per run" by conflating the batch with the
        # eligible count. Both numbers must appear, correctly.
        detail = explain_ticker_news_gap("AAPL", "2026-10-06", self.REPORT)
        self.assertEqual(detail["kind"], "not_scheduled")
        self.assertIn("25 of 75", detail["reason"])

    def test_a_deliberately_untracked_fund_is_not_reported_as_a_gap(self):
        detail = explain_ticker_news_gap("VOO", "2026-10-06", self.REPORT)
        self.assertEqual(detail["kind"], "not_tracked")
        self.assertIn("by design", detail["reason"])

    def test_a_successful_fetch_with_no_articles_is_its_own_answer(self):
        # Absence of news is an observation, not a failure, and must not be
        # grouped with the tickers that could not be reached.
        detail = explain_ticker_news_gap("MSFT", "2026-10-06", self.REPORT)
        self.assertEqual(detail["kind"], "no_articles")

    def test_every_kind_carries_a_distinct_actionable_instruction(self):
        # A notification the reader cannot act on is how a channel gets muted,
        # and four identical actions would be the original bug wearing new text.
        kinds = {
            detail["kind"]: detail["action"]
            for detail in (
                explain_ticker_news_gap(t, "2026-10-06", self.REPORT)
                for t in ("AMZN", "AAPL", "VOO", "MSFT")
            )
        }
        self.assertEqual(len(kinds), 4)
        self.assertEqual(len(set(kinds.values())), 4)

    def test_an_auth_failure_is_never_described_as_self_clearing(self):
        # Quota and auth present the same "no news" surface. Telling the
        # operator to wait out a window that will never reopen is the one
        # mistake this taxonomy exists to prevent.
        report = {
            "as_of": "2026-10-06",
            "sources": {"news": {
                "batch": 25, "eligible": 75, "unavailable": ["AMZN"],
                "failure_kinds": {"authentication_failed": 1},
                "stopped_early_because": "authentication_failed",
            }},
        }
        detail = explain_ticker_news_gap("AMZN", "2026-10-06", report)
        self.assertEqual(detail["kind"], "authentication_failed")
        self.assertIn("NEWS_PROVIDER_API_KEY", detail["action"])
        self.assertNotIn("clears on its own", detail["reason"])

    def test_a_legacy_report_without_typed_kinds_still_reads_as_quota(self):
        # Reports written before the taxonomy carry only `quota_exhausted`.
        # Calling those days "unknown_error" would misdescribe history.
        report = {
            "as_of": "2026-10-06",
            "sources": {"news": {
                "batch": 40, "eligible": 75,
                "unavailable": ["AMZN"], "quota_exhausted": True,
            }},
        }
        detail = explain_ticker_news_gap("AMZN", "2026-10-06", report)
        self.assertEqual(detail["kind"], "quota_exceeded")

    def test_no_run_at_all_concludes_nothing(self):
        detail = explain_ticker_news_gap("AAPL", "2026-10-06", None)
        self.assertEqual(detail["kind"], "no_run_record")
        self.assertIn("nothing can be concluded", detail["action"].lower())


if __name__ == "__main__":
    unittest.main()
