"""The alert runner: the quota defect, and the ledger it reads instead.

THE DEFECT THESE EXIST TO PREVENT RECURRING. Triggering the real scheduled task
over 77 tickers logged **62 HTTP 429s**: `detect_news` called
`build_news_snapshot` per ticker, one provider request each, and the collector had
already spent ~40 of the free tier's 100 that day.

That is worse than a wasted run. News is the one PERISHABLE source -- gone after
NEWS_LOOKBACK_DAYS = 7 -- and collection had just been moved to 14:30 precisely to
protect it from daytime consumption. An alert task at 22:15 spending 77 requests
would consume the next day's budget and recreate, from the other direction, the
exact five-night data loss it was meant to fix.

So the runner reads what the collector already captured into the W6 ledger and
makes NO provider call at all.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNNER = REPO_ROOT / "scripts" / "run_alerts.py"


class TheRunnerNeverSpendsTheNewsQuotaTests(unittest.TestCase):
    """A source-level assertion, because the cost is invisible at runtime."""

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_the_runner_does_not_call_the_fetching_entry_point(self):
        # `build_news_snapshot` fetches. One call per ticker over 77 tickers is
        # 77 requests against a 100/day tier the collector already drew on.
        self.assertNotIn(
            "build_news_snapshot(",
            self.source,
            "the runner must not fetch news; it reads the captured ledger",
        )

    def test_the_runner_reads_the_raw_ledger(self):
        self.assertIn("load_raw_records", self.source)

    def test_the_runner_uses_the_shared_enrichment(self):
        # Copying the enrichment pipeline into the runner would be a W5
        # violation: two implementations of one measurement, free to drift.
        self.assertIn("enrich_captured_news", self.source)

    def test_the_reader_walks_every_provider_in_the_order(self):
        """CAUGHT ON THE FIRST RUN WITH A FINNHUB KEY.

        The collector had just written 50 articles per ticker into
        data/raw/finnhub_news/, and the runner reported "News Unavailable" for
        those same tickers because this read only the NewsAPI store.

        The failure mode is the dangerous kind: it does not look like a bug. The
        alert carries a plausible NOT_EVALUATED reason, which reads as "the
        provider could not be reached" rather than "I looked in the wrong
        drawer" -- so every alert would have been blind for as long as Finnhub
        was the active provider.
        """
        self.assertIn("NEWS_PROVIDER_ORDER", self.source)
        self.assertIn("for source_id in NEWS_PROVIDER_ORDER", self.source)

    def test_the_reader_does_not_hardcode_one_provider(self):
        # A hardcoded store is how the reader fell behind the fetcher.
        import re

        body = self.source[
            self.source.index("def _captured_news("):
            self.source.index("def detect_news(")
        ]
        self.assertFalse(
            re.search(r"load_raw_records\(\s*NEWS_SOURCE_ID", body),
            "the ledger reader must not name a single provider",
        )

    def test_an_uncaptured_ticker_is_not_reported_as_quiet(self):
        # The collector rotates 40 of 75 eligible tickers per run, so "not
        # captured today" means "not looked at". Reporting NO_EVENT would claim
        # calm for two thirds of the portfolio every day.
        self.assertIn("UNAVAILABLE", self.source)
        self.assertIn("rotates", self.source)


class TheLedgerReplayPathTests(unittest.TestCase):
    """`enrich_captured_news` must agree with the live path on everything."""

    @staticmethod
    def record(
        headline,
        *,
        ticker="AVGO",
        summary="",
        published="2026-10-01T18:00:00Z",
        record_id="r1",
        source="Reuters",
        quality=0.9,
    ):
        return {
            "source_record_id": record_id,
            "headline": headline,
            "summary": summary,
            "ticker": ticker,
            "company_name": "Broadcom",
            "published_time": published,
            "source_name": source,
            "source_quality": quality,
            "url": f"https://example.invalid/{record_id}",
        }

    def enrich(self, records, ticker="AVGO", as_of="2026-10-01"):
        from core.news_adapter import enrich_captured_news

        return enrich_captured_news({"records": records}, ticker, as_of)

    def test_it_makes_no_provider_call(self):
        # THE DECISIVE PROPERTY. Patching urlopen to raise makes a regression
        # fail loudly rather than silently costing quota.
        with mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("fetched!")
        ):
            snapshot = self.enrich(
                [self.record("Broadcom reports quarterly earnings")]
            )
        self.assertEqual(snapshot["status"], "OK")

    def test_it_produces_the_payload_shape_a9_reads(self):
        snapshot = self.enrich([self.record("Broadcom reports quarterly earnings")])
        self.assertIn("articles", snapshot)
        article = snapshot["articles"][0]
        for field in ("relevance", "category", "tone", "included_in_aggregation"):
            with self.subTest(field=field):
                self.assertIn(field, article)

    def test_it_marks_itself_as_a_replay(self):
        # So a reader can tell a replayed snapshot from a live fetch.
        self.assertTrue(self.enrich([self.record("x")])["replayed_from_ledger"])

    def test_the_point_in_time_policy_still_applies_on_replay(self):
        # Replaying a ledger must not relax the policy that governed its
        # capture: a record published after as_of is still rejected.
        snapshot = self.enrich(
            [self.record("Broadcom earnings", published="2026-12-25T10:00:00Z")],
            as_of="2026-10-01",
        )
        self.assertEqual(snapshot["status"], "INVALID")
        self.assertIn("point-in-time", snapshot["reason"])

    def test_no_records_is_unavailable_not_empty_success(self):
        self.assertEqual(self.enrich([])["status"].upper(), "UNAVAILABLE")

    def test_an_unparseable_as_of_is_refused(self):
        snapshot = self.enrich([self.record("x")], as_of="not a date")
        self.assertEqual(snapshot["status"].upper(), "UNAVAILABLE")

    def test_a_duplicate_headline_is_excluded_with_its_reason(self):
        snapshot = self.enrich(
            [
                self.record("Broadcom reports quarterly earnings", record_id="a"),
                self.record("Broadcom reports quarterly earnings", record_id="b"),
            ]
        )
        reasons = [a["exclusion_reason"] for a in snapshot["articles"]]
        self.assertIn("duplicate_headline", reasons)

    def test_an_off_entity_article_leaves_nothing_to_aggregate(self):
        snapshot = self.enrich(
            [self.record("Some unrelated firm reports earnings", ticker="ZZZZ")]
        )
        self.assertEqual(snapshot["status"], "INCOMPLETE")

    def test_an_excluded_article_stays_in_the_evidence(self):
        # Evidence considered and rejected is different from evidence that
        # never existed, so the row survives with its reason.
        snapshot = self.enrich(
            [self.record("Some unrelated firm reports earnings", ticker="ZZZZ")]
        )
        self.assertEqual(len(snapshot["articles"]), 1)
        self.assertFalse(snapshot["articles"][0]["included_in_aggregation"])
        self.assertTrue(snapshot["articles"][0]["exclusion_reason"])


class TheLedgerLookupUsesDeclaredNamesTests(unittest.TestCase):
    """The bug class that has now bitten twice, two commits apart.

    A detector compares today against the prior it finds in the ledger. If the
    lookup name is wrong the prior is always None, every alert reports
    NOT_EVALUATED FOREVER, and the failure looks exactly like "no change yet" --
    it is silent.

    FIRST INSTANCE: the regime detector read `detail["after"]`, a key A4 never
    writes. SECOND INSTANCE: the lookup guessed `f"{detector}_change"` then
    `f"{detector}_event"`, which matches A4 (`regime_change`) and A9
    (`news_event`) but not A5 (`thesis_break`).

    So the names are DATA now, and these tests assert the mapping is complete
    and that it matches what each detector actually stamps.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_alerts_mod", RUNNER)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_every_detector_has_a_declared_record_name(self):
        self.assertEqual(
            set(self.module.DETECTOR_RECORD_NAMES),
            set(self.module.DETECTORS),
            "a detector with no declared record name gets a None prior and "
            "reports NOT_EVALUATED forever, silently",
        )

    def test_every_detector_has_a_function(self):
        self.assertEqual(
            set(self.module.DETECTOR_FUNCTIONS), set(self.module.DETECTORS)
        )

    def test_the_declared_names_are_distinct(self):
        names = list(self.module.DETECTOR_RECORD_NAMES.values())
        self.assertEqual(len(set(names)), len(names))

    def test_the_names_match_what_the_detectors_actually_stamp(self):
        """The assertion that would have caught both instances.

        Each A-module hard-codes its own `alert` field; this reads it from the
        source rather than trusting the mapping.
        """
        expected = {
            "regime": ("core/regime_alert.py", "regime_change"),
            "news": ("core/news_event_alert.py", "news_event"),
            "thesis": ("core/thesis_alert.py", "thesis_break"),
        }
        for detector, (module_path, name) in expected.items():
            with self.subTest(detector=detector):
                source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
                self.assertIn(
                    f'"alert": "{name}"',
                    source,
                    f"{module_path} does not stamp alert={name!r}",
                )
                self.assertEqual(
                    self.module.DETECTOR_RECORD_NAMES[detector], name
                )

    def test_the_lookup_no_longer_guesses_the_name(self):
        """Asserted against CODE, not prose.

        The comment explaining this bug necessarily quotes the old guessed
        names, so a whole-file search matches them and proves nothing. The
        check is that the `_previous_state` CALL uses the declared mapping.
        """
        source = RUNNER.read_text(encoding="utf-8")
        calls = [
            line.strip()
            for line in source.splitlines()
            if "_previous_state(ticker" in line
            and not line.strip().startswith("def ")
        ]
        self.assertTrue(calls, "the ledger lookup call vanished")
        for call in calls:
            with self.subTest(call=call):
                self.assertIn("DETECTOR_RECORD_NAMES[detector]", call)
                self.assertNotIn("f\"{detector}", call)


class TheDefaultSweepCostsNoProviderRequestsTests(unittest.TestCase):
    """What a NIGHTLY run does, which is deliberately not every detector."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_alerts_mod2", RUNNER)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_thesis_is_not_in_the_default_sweep(self):
        # It calls orchestrate_score, which fetches news: one request per
        # ticker, which over 77 tickers is the quota defect already fixed once.
        self.assertNotIn("thesis", self.module.DEFAULT_DETECTORS)

    def test_the_default_detectors_are_all_known(self):
        for detector in self.module.DEFAULT_DETECTORS:
            with self.subTest(detector=detector):
                self.assertIn(detector, self.module.DETECTORS)

    def test_the_zero_cost_detectors_are_the_default(self):
        self.assertEqual(
            set(self.module.DEFAULT_DETECTORS), {"regime", "news"}
        )

    def test_the_help_text_states_the_cost(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("one news request per ticker", source)


class TheRunnerReportsHonestlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_a_suppressed_finding_is_still_stored(self):
        # "Nothing fired" and "it fired and we chose not to show it" are
        # different facts, and only one can be audited after a loss.
        self.assertIn("A SUPPRESSED ALERT IS STILL STORED", self.source)

    def test_one_ticker_failing_does_not_abort_the_run(self):
        self.assertIn("FAIL-SOFT PER TICKER", self.source)

    def test_the_previous_state_comes_from_the_ledger(self):
        # Recomputing it would derive a value from today's data and label it
        # yesterday's -- the staleness trap run_forecasts measured.
        self.assertIn("_previous_state", self.source)
        self.assertIn("rather than recomputed", self.source)

    def test_the_prior_regime_label_is_read_from_the_field_a4_writes(self):
        # A4 records `previous`/`current`. Reading `after` -- a key it never
        # writes -- made every regime alert report NOT_EVALUATED forever.
        self.assertIn('detail.get("current")', self.source)
        self.assertNotIn('detail.get("after")', self.source)


if __name__ == "__main__":
    unittest.main()
