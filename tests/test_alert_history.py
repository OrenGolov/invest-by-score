"""Tests for the Alerts Monitoring history view."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.alert_history import (
    alert_history,
    build_alert_rows,
    filter_alert_rows,
    summarise,
    ticker_explanations,
)


def _report(as_of: str, **kwargs) -> dict:
    """A minimal collection report shaped like the real ledger's lines."""
    news = {
        "attempted": kwargs.pop("attempted", 25),
        "written": kwargs.pop("written", 25),
        "eligible": kwargs.pop("eligible", 75),
    }
    news.update(kwargs.pop("news", {}))
    report = {
        "as_of": as_of,
        "collected_at": f"{as_of}T22:00:00+00:00",
        "sources": {"news": news},
    }
    report.update(kwargs)
    return report


class BuildRowsTests(unittest.TestCase):
    def test_rows_are_newest_first(self) -> None:
        rows = build_alert_rows(
            [_report("2026-10-01"), _report("2026-10-03"), _report("2026-10-02")]
        )
        self.assertEqual(
            [row.as_of for row in rows],
            ["2026-10-03", "2026-10-02", "2026-10-01"],
        )

    def test_non_dict_lines_are_skipped(self) -> None:
        rows = build_alert_rows([_report("2026-10-01"), "garbage", None])  # type: ignore[list-item]
        self.assertEqual(len(rows), 1)

    def test_delivery_is_unknown_not_false(self) -> None:
        # Claiming "not delivered" would be as wrong as claiming delivered:
        # nothing records it either way.
        rows = build_alert_rows([_report("2026-10-01")])
        self.assertIsNone(rows[0].delivered)

    def test_failed_tickers_are_collected_and_deduped(self) -> None:
        report = _report("2026-10-01")
        report["sources"]["fundamentals"] = {"attempted": 2, "failed": ["msft", "AAPL"]}
        report["sources"]["news"]["failed"] = ["AAPL"]
        rows = build_alert_rows([report])
        self.assertEqual(rows[0].tickers_failed, ("AAPL", "MSFT"))

    def test_each_row_is_judged_against_its_own_date(self) -> None:
        # The whole history judged with today=now would read as days-stale and
        # every old row would be a false CRITICAL.
        rows = build_alert_rows([_report(f"2026-10-0{n}") for n in range(1, 6)])
        self.assertTrue(all(not row.should_alert for row in rows), [r.reasons for r in rows])

    def test_unparseable_as_of_does_not_raise(self) -> None:
        rows = build_alert_rows([_report("not-a-date")])
        self.assertEqual(len(rows), 1)


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        reports = [_report(f"2026-10-0{n}") for n in range(1, 6)]
        reports[2]["sources"]["fundamentals"] = {"attempted": 1, "failed": ["NVDA"]}
        self.reports = reports
        self.rows = build_alert_rows(reports)

    def test_date_range_is_inclusive(self) -> None:
        kept = filter_alert_rows(self.rows, start="2026-10-02", end="2026-10-04")
        self.assertEqual(
            [row.as_of for row in kept],
            ["2026-10-04", "2026-10-03", "2026-10-02"],
        )

    def test_open_ended_range(self) -> None:
        self.assertEqual(len(filter_alert_rows(self.rows, start="2026-10-04")), 2)
        self.assertEqual(len(filter_alert_rows(self.rows, end="2026-10-02")), 2)

    def test_unparseable_bound_is_ignored_not_fatal(self) -> None:
        self.assertEqual(len(filter_alert_rows(self.rows, start="garbage")), len(self.rows))

    def test_ticker_filter_matches_failures(self) -> None:
        kept = filter_alert_rows(self.rows, ticker="nvda")
        self.assertEqual([row.as_of for row in kept], ["2026-10-03"])

    def test_ticker_filter_excludes_runs_without_that_failure(self) -> None:
        self.assertEqual(filter_alert_rows(self.rows, ticker="TSLA"), [])

    def test_severity_is_a_floor_so_warn_includes_critical(self) -> None:
        # The opposite would hide the worst rows behind the milder filter.
        reports = [_report("2026-10-01"), _report("2026-10-02", perishable_lost=["news"])]
        rows = build_alert_rows(reports)
        critical = [row for row in rows if row.severity == "critical"]
        self.assertTrue(critical, "fixture did not produce a critical row")
        kept = filter_alert_rows(rows, severity="warn")
        self.assertIn("critical", {row.severity for row in kept})

    def test_only_alerting_drops_clean_runs(self) -> None:
        reports = [_report("2026-10-01"), _report("2026-10-02", perishable_lost=["news"])]
        rows = build_alert_rows(reports)
        kept = filter_alert_rows(rows, only_alerting=True)
        self.assertTrue(kept)
        self.assertTrue(all(row.should_alert for row in kept))

    def test_filters_compose(self) -> None:
        kept = filter_alert_rows(
            self.rows, start="2026-10-01", end="2026-10-05", ticker="NVDA"
        )
        self.assertEqual([row.as_of for row in kept], ["2026-10-03"])


class SummaryTests(unittest.TestCase):
    def test_counts_partition_the_rows(self) -> None:
        rows = build_alert_rows(
            [_report("2026-10-01"), _report("2026-10-02", perishable_lost=["news"])]
        )
        summary = summarise(rows)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["critical"] + summary["warn"], summary["alerting"])
        self.assertEqual(summary["alerting"] + summary["clean"], summary["runs"])

    def test_empty_summary_is_safe(self) -> None:
        summary = summarise([])
        self.assertEqual(summary["runs"], 0)
        self.assertEqual(summary["newest"], "")


class TickerDetailTests(unittest.TestCase):
    def test_blank_ticker_returns_nothing(self) -> None:
        rows = build_alert_rows([_report("2026-10-01")])
        self.assertEqual(ticker_explanations(rows, [_report("2026-10-01")], ""), [])

    def test_explanation_carries_kind_and_action(self) -> None:
        reports = [_report("2026-10-01")]
        rows = build_alert_rows(reports)
        detail = ticker_explanations(rows, reports, "MSFT")
        self.assertEqual(len(detail), 1)
        self.assertTrue(detail[0]["kind"])
        self.assertTrue(detail[0]["reason"])


class PayloadTests(unittest.TestCase):
    def _write(self, directory: str, reports: list[dict]) -> Path:
        path = Path(directory) / "collection_report.jsonl"
        path.write_text(
            "".join(json.dumps(report) + "\n" for report in reports), encoding="utf-8"
        )
        return path

    def test_payload_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, [_report("2026-10-01"), _report("2026-10-02")])
            payload = alert_history(path=path)
        for key in ("summary", "rows", "filters", "source", "delivery_caveat"):
            self.assertIn(key, payload)
        self.assertEqual(payload["total_runs_on_disk"], 2)

    def test_limit_caps_rows_but_reports_the_true_total(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(directory, [_report(f"2026-10-0{n}") for n in range(1, 6)])
            payload = alert_history(limit=2, path=path)
        self.assertEqual(payload["returned"], 2)
        self.assertEqual(payload["total_runs_on_disk"], 5)

    def test_missing_ledger_is_reported_not_raised(self) -> None:
        with TemporaryDirectory() as directory:
            payload = alert_history(path=Path(directory) / "absent.jsonl")
        self.assertFalse(payload["source_exists"])
        self.assertEqual(payload["rows"], [])

    def test_truncated_line_does_not_lose_the_rest(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "collection_report.jsonl"
            path.write_text(
                json.dumps(_report("2026-10-01")) + "\n{broken\n"
                + json.dumps(_report("2026-10-02")) + "\n",
                encoding="utf-8",
            )
            payload = alert_history(path=path)
        self.assertEqual(payload["total_runs_on_disk"], 2)

    def test_caveat_states_rows_are_not_proof_of_delivery(self) -> None:
        # The honesty constraint belongs in the payload, so any consumer
        # inherits it rather than relying on this one UI to say it.
        with TemporaryDirectory() as directory:
            path = self._write(directory, [_report("2026-10-01")])
            payload = alert_history(path=path)
        self.assertIn("not proof", payload["delivery_caveat"])


if __name__ == "__main__":
    unittest.main()
