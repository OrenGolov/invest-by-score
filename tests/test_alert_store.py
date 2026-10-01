"""A8 store tests — append-only, stable ids, and the failures that must not happen.

THE PROPERTY THE OPERATOR ACTUALLY ASKED FOR: "Duplicate emails for the same event
should be prevented." That reduces to one question — does the same finding observed
on two days share an id? It does only because the id excludes every timestamp, and
that exclusion is the trap this codebase has now measured three times:

    A1  four consecutive days of an identical unavailable forecast -> 4 digests
    A7  a suppression key containing as_of -> unique every session, suppresses nothing
    A8  an id over the whole record -> would prevent no duplicate at all

So `test_the_same_finding_on_two_days_shares_an_id` is the load-bearing test here.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from core.alert_priority import grade
from core.alert_store import (
    ALERT_ID_FIELDS,
    ALERT_ID_FORBIDDEN,
    CHANNEL_DASHBOARD,
    CHANNEL_EMAIL,
    DELIVERY_FAILED,
    DELIVERY_SENT,
    DELIVERY_SKIPPED,
    AlertStoreError,
    alert_id,
    append_alert,
    build_record,
    counts_by_priority,
    delivered_ids,
    event_types_seen,
    load_alerts,
    load_deliveries,
    query,
    record_delivery,
    tickers_seen,
)
from core.config import ALERT_PRIORITIES


def detector_alert(
    ticker="NVDA",
    state="CONFIRMED",
    severity="warn",
    as_of="2026-10-01",
    name="regime_change",
    horizon="20d",
):
    return {
        "alert": name,
        "ticker": ticker,
        "horizon": horizon,
        "as_of": as_of,
        "kind": state,
        "severity": severity,
        "reason": f"{ticker} reached {state} on {as_of}",
    }


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._dir.name)
        self.store = base / "alerts.jsonl"
        self.deliveries = base / "deliveries.jsonl"

    def tearDown(self):
        self._dir.cleanup()

    def record(self, alert=None, *, title="Test Alert", at=None, **kwargs):
        alert = alert if alert is not None else detector_alert()
        row = build_record(alert, grade(alert), title=title, detected_at=at, **kwargs)
        append_alert(row, self.store)
        return row


class TheIdIsTimelessTests(StoreTestCase):
    """The load-bearing property: dedup is only possible if the id excludes time."""

    def test_the_same_finding_on_two_days_shares_an_id(self):
        day_one = self.record(
            detector_alert(as_of="2026-09-30"), at="2026-09-30T22:00:00+00:00"
        )
        day_two = self.record(
            detector_alert(as_of="2026-10-01"), at="2026-10-01T22:00:00+00:00"
        )
        self.assertEqual(
            day_one["alert_id"],
            day_two["alert_id"],
            "an unchanged finding must share its id, or every day re-emails it",
        )

    def test_a_changed_state_is_a_new_finding(self):
        confirmed = self.record(detector_alert(state="CONFIRMED"))
        vanished = self.record(detector_alert(state="DISAPPEARED"))
        self.assertNotEqual(confirmed["alert_id"], vanished["alert_id"])

    def test_different_tickers_never_collide(self):
        nvda = self.record(detector_alert(ticker="NVDA"))
        msft = self.record(detector_alert(ticker="MSFT"))
        self.assertNotEqual(nvda["alert_id"], msft["alert_id"])

    def test_different_horizons_never_collide(self):
        short = self.record(detector_alert(horizon="5d"))
        long = self.record(detector_alert(horizon="60d"))
        self.assertNotEqual(short["alert_id"], long["alert_id"])

    def test_different_detectors_never_collide(self):
        regime = self.record(detector_alert(name="regime_change"))
        thesis = self.record(detector_alert(name="thesis_break"))
        self.assertNotEqual(regime["alert_id"], thesis["alert_id"])

    def test_no_forbidden_field_is_part_of_the_identity(self):
        # Declared as data so the rule is auditable without tracing code.
        for field in ALERT_ID_FORBIDDEN:
            with self.subTest(field=field):
                self.assertNotIn(field, ALERT_ID_FIELDS)

    def test_the_reason_text_does_not_change_the_id(self):
        # Free text is evidence, not identity. A reworded reason is the same
        # finding, and letting it change the id would re-email on a cosmetic edit.
        first = alert_id({"alert": "a", "ticker": "T", "horizon": "20d",
                          "state": "CONFIRMED", "reason": "one wording"})
        second = alert_id({"alert": "a", "ticker": "T", "horizon": "20d",
                           "state": "CONFIRMED", "reason": "a different wording"})
        self.assertEqual(first, second)

    def test_an_id_needs_a_mapping(self):
        with self.assertRaises(AlertStoreError):
            alert_id(["not", "a", "mapping"])


class NothingIsEverDeletedTests(StoreTestCase):
    """The operator was explicit, and W6 provenance depends on it."""

    def test_every_appended_row_is_readable(self):
        for index in range(12):
            self.record(
                detector_alert(ticker=f"T{index:02d}"),
                at=f"2026-10-01T{index + 6:02d}:00:00+00:00",
            )
        self.assertEqual(len(load_alerts(self.store)), 12)

    def test_re_observing_the_same_finding_appends_rather_than_replaces(self):
        self.record(at="2026-09-30T22:00:00+00:00")
        self.record(at="2026-10-01T22:00:00+00:00")
        rows = load_alerts(self.store)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["alert_id"], rows[1]["alert_id"])

    def test_a_truncated_final_line_does_not_lose_history(self):
        # The realistic corruption: power loss mid-append. Discarding the whole
        # history for one bad line would lose every earlier finding.
        for index in range(5):
            self.record(
                detector_alert(ticker=f"T{index}"),
                at=f"2026-10-01T{index + 6:02d}:00:00+00:00",
            )
        with self.store.open("a", encoding="utf-8") as handle:
            handle.write('{"alert": "truncated", "ticker": "XX"')
        self.assertEqual(len(load_alerts(self.store)), 5)

    def test_a_missing_store_reads_as_empty_not_as_an_error(self):
        self.assertEqual(load_alerts(self.store.with_name("absent.jsonl")), [])

    def test_a_stored_row_carries_the_evidence_for_its_own_priority(self):
        # Nobody should have to re-run the grader to learn why a band was given.
        row = self.record()
        self.assertTrue(row["priority_reason"])
        self.assertTrue(row["action_reason"])
        self.assertIn("kind", row["detail"])


class TheTimestampIsValidatedTests(StoreTestCase):
    """CAUGHT BY PROBE: 'T24:00:00' was stored and sorted as newest forever."""

    def test_an_impossible_hour_is_refused(self):
        with self.assertRaises(AlertStoreError) as caught:
            self.record(at="2026-10-01T24:00:00+00:00")
        self.assertIn("ISO-8601", str(caught.exception))

    def test_a_naive_timestamp_is_refused(self):
        # Alerts are stored in UTC so they sort across a DST boundary.
        with self.assertRaises(AlertStoreError) as caught:
            self.record(at="2026-10-01T22:00:00")
        self.assertIn("timezone", str(caught.exception))

    def test_nonsense_is_refused(self):
        with self.assertRaises(AlertStoreError):
            self.record(at="yesterday evening")

    def test_a_valid_offset_is_accepted(self):
        row = self.record(at="2026-10-01T22:00:00+03:00")
        self.assertEqual(row["detected_at"], "2026-10-01T22:00:00+03:00")

    def test_an_absent_timestamp_defaults_to_now_in_utc(self):
        row = self.record()
        self.assertTrue(row["detected_at"].endswith("+00:00"))


class DeliveryIsSeparateFromDetectionTests(StoreTestCase):
    def test_a_failed_send_does_not_mark_the_alert_delivered(self):
        # THE IMPORTANT ONE: if FAILED counted as delivered, one SMTP outage
        # would permanently silence every alert it touched.
        row = self.record()
        record_delivery(
            row,
            channel=CHANNEL_EMAIL,
            status=DELIVERY_FAILED,
            detail="host refused",
            path=self.deliveries,
        )
        self.assertEqual(delivered_ids(path=self.deliveries), set())

    def test_a_skipped_send_does_not_mark_the_alert_delivered(self):
        row = self.record()
        record_delivery(
            row, channel=CHANNEL_EMAIL, status=DELIVERY_SKIPPED,
            path=self.deliveries,
        )
        self.assertEqual(delivered_ids(path=self.deliveries), set())

    def test_a_sent_delivery_is_recorded(self):
        row = self.record()
        record_delivery(
            row, channel=CHANNEL_EMAIL, status=DELIVERY_SENT, path=self.deliveries
        )
        self.assertEqual(delivered_ids(path=self.deliveries), {row["alert_id"]})

    def test_delivery_can_be_filtered_by_channel(self):
        row = self.record()
        record_delivery(
            row, channel=CHANNEL_DASHBOARD, status=DELIVERY_SENT,
            path=self.deliveries,
        )
        self.assertEqual(
            delivered_ids(CHANNEL_DASHBOARD, path=self.deliveries),
            {row["alert_id"]},
        )
        self.assertEqual(delivered_ids(CHANNEL_EMAIL, path=self.deliveries), set())

    def test_a_failed_attempt_is_still_recorded_for_audit(self):
        # "We chose not to send" and "the host refused" are different facts from
        # "it was sent", and only a record of the attempt distinguishes them.
        row = self.record()
        record_delivery(
            row, channel=CHANNEL_EMAIL, status=DELIVERY_FAILED,
            detail="timeout", path=self.deliveries,
        )
        rows = load_deliveries(self.deliveries)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], DELIVERY_FAILED)
        self.assertEqual(rows[0]["detail"], "timeout")

    def test_an_unknown_channel_is_refused(self):
        with self.assertRaises(AlertStoreError):
            record_delivery(
                self.record(), channel="carrier_pigeon", status=DELIVERY_SENT,
                path=self.deliveries,
            )

    def test_an_unknown_status_is_refused(self):
        with self.assertRaises(AlertStoreError):
            record_delivery(
                self.record(), channel=CHANNEL_EMAIL, status="PROBABLY",
                path=self.deliveries,
            )


class DashboardQueryTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.rows = []
        for index, (ticker, state, severity, name) in enumerate(
            [
                ("NVDA", "DISAPPEARED", "warn", "forecast_change"),
                ("NVDA", "CONFIRMED", "warn", "regime_change"),
                ("MSFT", "APPEARED", "info", "forecast_change"),
                ("AMD", "PENDING", None, "regime_change"),
                ("KO", "LOW_IMPACT", "info", "event_impact"),
                ("TSLA", "NOT_EVALUATED", None, "thesis_break"),
            ]
        ):
            self.rows.append(
                self.record(
                    detector_alert(ticker=ticker, state=state, severity=severity,
                                   name=name),
                    title=f"{ticker} {state}",
                    at=f"2026-10-0{index + 1}T12:00:00+00:00",
                )
            )

    def test_newest_first_by_default(self):
        stamps = [r["detected_at"] for r in query(path=self.store)]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_oldest_first_on_request(self):
        stamps = [r["detected_at"] for r in query(path=self.store, newest_first=False)]
        self.assertEqual(stamps, sorted(stamps))

    def test_the_ticker_filter_is_case_insensitive(self):
        # The operator types `nvda`; the store holds `NVDA`. An exact match
        # would silently return nothing.
        self.assertEqual(len(query(path=self.store, ticker="nvda")), 2)
        self.assertEqual(len(query(path=self.store, ticker="NVDA")), 2)
        self.assertEqual(len(query(path=self.store, ticker=" nvda ")), 2)

    def test_filtering_by_priority(self):
        urgent = query(path=self.store, priority="Urgent")
        self.assertTrue(urgent)
        self.assertTrue(all(r["priority"] == "Urgent" for r in urgent))

    def test_filtering_by_several_priorities(self):
        rows = query(path=self.store, priority=["Urgent", "Very High"])
        self.assertTrue(all(r["priority"] in ("Urgent", "Very High") for r in rows))

    def test_an_unknown_priority_filter_is_refused(self):
        with self.assertRaises(AlertStoreError):
            query(path=self.store, priority="Catastrophic")

    def test_filtering_by_event_type(self):
        rows = query(path=self.store, event_type="forecast_change")
        self.assertEqual(len(rows), 2)

    def test_the_end_date_includes_the_whole_day(self):
        # A bare date compares as midnight, so without the boundary fix
        # end="2026-10-01" would exclude everything that day.
        rows = query(path=self.store, start="2026-10-01", end="2026-10-01")
        self.assertEqual(len(rows), 1)

    def test_a_local_date_filter_finds_an_evening_alert(self):
        """CAUGHT LIVE at UTC+3, and it made the feed look empty.

        At 02:15 local the alerts generated minutes earlier carried
        detected_at 2026-10-01T23:12Z. Filtering the operator's "today"
        (2026-10-02) against the raw UTC string returned ZERO of 12 rows -- on
        the very day the feed was populated. Comparing a local date against a
        UTC timestamp is off by the offset, every day, and silently.
        """
        evening = self.record(
            detector_alert(ticker="LATE"), at="2026-10-01T23:12:00+00:00"
        )
        # UTC+3: 23:12Z on the 1st is 02:12 local on the 2nd.
        found = query(path=self.store, start="2026-10-02", end="2026-10-02",
                      utc_offset_minutes=180)
        self.assertIn(
            evening["alert_id"], [row["alert_id"] for row in found],
            "an alert from the viewer's today must appear under their date",
        )

    def test_the_same_filter_in_utc_excludes_it(self):
        # The complement: without an offset the bound is read as UTC, which is
        # the right default for a caller that did not say which zone it meant.
        evening = self.record(
            detector_alert(ticker="LATE"), at="2026-10-01T23:12:00+00:00"
        )
        found = query(path=self.store, start="2026-10-02", end="2026-10-02")
        self.assertNotIn(evening["alert_id"], [r["alert_id"] for r in found])

    def test_a_negative_offset_shifts_the_other_way(self):
        # New York at UTC-4: 01:30Z on the 2nd is still the evening of the 1st.
        early = self.record(
            detector_alert(ticker="NYC"), at="2026-10-02T01:30:00+00:00"
        )
        found = query(path=self.store, start="2026-10-01", end="2026-10-01",
                      utc_offset_minutes=-240)
        self.assertIn(early["alert_id"], [r["alert_id"] for r in found])

    def test_an_unparseable_stamp_is_excluded_from_a_dated_query(self):
        # An unparseable stamp has no position in time; placing it at either
        # boundary would make it appear in half of all ranges.
        rows = list(load_alerts(self.store))
        rows.append({"alert_id": "broken", "ticker": "XX",
                     "detected_at": "not a time", "priority": "Low"})
        found = query(rows, start="2026-10-01", end="2026-12-31")
        self.assertNotIn("broken", [r.get("alert_id") for r in found])

    def test_a_full_timestamp_bound_still_works(self):
        # An hour-wide window, so it cannot collide with the setUp fixtures
        # (which sit at 12:00 on 2026-10-01..06).
        self.record(detector_alert(ticker="ZZ"), at="2026-10-09T15:30:00+00:00")
        found = query(path=self.store, start="2026-10-09T15:00:00+00:00",
                      end="2026-10-09T16:00:00+00:00")
        self.assertEqual([r["ticker"] for r in found], ["ZZ"])

    def test_a_date_range_selects_the_span(self):
        rows = query(path=self.store, start="2026-10-02", end="2026-10-04")
        self.assertEqual(len(rows), 3)

    def test_search_covers_the_fields_a_reader_would_type(self):
        self.assertTrue(query(path=self.store, search="disappeared"))
        self.assertTrue(query(path=self.store, search="NVDA"))
        self.assertTrue(query(path=self.store, search="Urgent"))

    def test_search_is_case_insensitive(self):
        self.assertEqual(
            len(query(path=self.store, search="nvda")),
            len(query(path=self.store, search="NVDA")),
        )

    def test_an_absent_filter_does_not_filter(self):
        # Absent must mean "do not filter", never "match nothing".
        self.assertEqual(len(query(path=self.store)), len(self.rows))
        self.assertEqual(len(query(path=self.store, ticker=None)), len(self.rows))
        self.assertEqual(len(query(path=self.store, search="")), len(self.rows))

    def test_filters_combine(self):
        rows = query(path=self.store, ticker="nvda", priority="Urgent")
        self.assertEqual(len(rows), 1)

    def test_a_limit_pages_the_feed(self):
        self.assertEqual(len(query(path=self.store, limit=2)), 2)

    def test_a_limit_below_one_is_refused(self):
        with self.assertRaises(AlertStoreError):
            query(path=self.store, limit=0)

    def test_the_ungraded_alert_is_still_in_the_feed(self):
        # An ungraded alert means a detector is blind. It must appear, and it
        # must not be given a band.
        rows = query(path=self.store)
        ungraded = [r for r in rows if r["priority"] is None]
        self.assertTrue(ungraded)

    def test_counts_report_the_ungraded_rather_than_hiding_them(self):
        counts = counts_by_priority(self.store)
        self.assertIn("(ungraded)", counts)
        self.assertEqual(counts["(ungraded)"], 1)
        self.assertEqual(sum(counts.values()), len(self.rows))
        for band in ALERT_PRIORITIES:
            self.assertIn(band, counts)

    def test_the_filter_vocabularies_come_from_the_data(self):
        self.assertEqual(
            tickers_seen(self.store), ["AMD", "KO", "MSFT", "NVDA", "TSLA"]
        )
        self.assertEqual(
            event_types_seen(self.store),
            ["event_impact", "forecast_change", "regime_change", "thesis_break"],
        )


class BuildRecordRefusesWhatItCannotStoreTests(StoreTestCase):
    def test_an_unknown_priority_is_refused(self):
        alert = detector_alert()
        with self.assertRaises(AlertStoreError):
            build_record(alert, {"priority": "Catastrophic"}, title="x")

    def test_a_non_mapping_alert_is_refused(self):
        with self.assertRaises(AlertStoreError):
            build_record(["nope"], grade(detector_alert()), title="x")

    def test_a_non_mapping_grading_is_refused(self):
        with self.assertRaises(AlertStoreError):
            build_record(detector_alert(), ["nope"], title="x")

    def test_a_row_is_json_serialisable(self):
        # The store writes JSONL; an unserialisable row would be lost at write
        # time with only a log line to show for it.
        row = self.record()
        self.assertEqual(json.loads(json.dumps(row, default=str))["ticker"], "NVDA")


if __name__ == "__main__":
    unittest.main()
