"""The Monitoring tab: the API it reads, and the markup it renders.

The operator specified a dedicated nav tab, a newest-first feed, collapsed cards
that expand, colour-coded priorities, four filters plus search, local timezone
rendering, and mobile support. These tests assert each of those against the real
page and the real endpoint rather than against a description of them.

THE DEFECT THESE EXIST TO PREVENT RECURRING: filtering by date returned ZERO of 12
rows on the day the feed was populated, because the operator is at UTC+3 and the
store is UTC. A local date compared against a UTC timestamp is off by the offset,
every day, and silently.
"""

from __future__ import annotations

import pathlib
import re
import unittest
from html.parser import HTMLParser

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "index.html"


class Balance(HTMLParser):
    """Every opened tag closes, in order. A broken feed container hides rows."""

    VOID = {"meta", "br", "hr", "img", "input", "link", "source"}
    IMPLICIT = {"option", "li", "dd", "dt"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        if tag in self.IMPLICIT and self.stack and self.stack[-1] == tag:
            self.stack.pop()
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
            return
        if self.stack[-1] != tag:
            if self.stack[-1] in self.IMPLICIT:
                self.stack.pop()
                if self.stack and self.stack[-1] == tag:
                    self.stack.pop()
                    return
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            return
        self.stack.pop()


class ThePageCarriesWhatWasSpecifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_there_is_a_dedicated_monitoring_tab(self):
        self.assertIn('data-page="monitoring"', self.html)
        self.assertIn("Monitoring", self.html)

    def test_the_score_page_is_still_reachable(self):
        # Adding a tab must not remove the thing the page already did.
        self.assertIn('data-page="score"', self.html)
        self.assertIn('id="page-score"', self.html)

    def test_every_specified_filter_is_present(self):
        for element in ("fSearch", "fTicker", "fType", "fStart", "fEnd"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', self.html)

    def test_priority_filtering_is_present(self):
        self.assertIn('id="priorityChips"', self.html)

    def test_there_is_a_way_to_clear_the_filters(self):
        self.assertIn('id="fClear"', self.html)

    def test_every_priority_has_its_own_colour_class(self):
        for cls in ("p-urgent", "p-veryhigh", "p-high", "p-medium", "p-low"):
            with self.subTest(cls=cls):
                self.assertIn(f".{cls}", self.html)

    def test_an_ungraded_alert_has_its_own_marker(self):
        # It means a detector could not decide, which must not render as Low.
        self.assertIn(".p-none", self.html)

    def test_cards_expand_and_collapse(self):
        self.assertIn("classList.toggle('open')", self.html)
        self.assertIn(".alert-card.open .alert-body", self.html)

    def test_timestamps_render_in_the_viewers_timezone(self):
        self.assertIn("toLocaleString", self.html)

    def test_the_browsers_offset_is_sent_to_the_api(self):
        # The fix for the zero-rows defect. Without this the date filter
        # compares the viewer's local date against a UTC timestamp.
        self.assertIn("getTimezoneOffset", self.html)
        self.assertIn("params.set('tz'", self.html)

    def test_the_layout_adapts_to_a_phone(self):
        self.assertRegex(self.html, r"@media \(max-width: 6\d\dpx\)")

    def test_alert_text_is_escaped_before_reaching_the_dom(self):
        # Alert detail carries provider headlines: untrusted input.
        self.assertIn("function esc(", self.html)
        self.assertIn("replace(/</g, '&lt;')", self.html)

    def test_the_markup_is_balanced(self):
        parser = Balance()
        parser.feed(self.html)
        self.assertEqual(parser.errors, [])
        self.assertEqual(parser.stack, [])

    def test_the_card_renderer_escapes_every_field_it_interpolates(self):
        """No raw interpolation into HTML.

        Scans the card builder for `+ row.` or `+ detail.` without `esc(`, which
        is how an unescaped headline would reach the DOM.
        """
        start = self.html.index("function alertCard(")
        end = self.html.index("function renderAlerts(")
        body = self.html[start:end]
        raw = re.findall(r"\+\s*(?:row|detail|summary)\.[A-Za-z_]+", body)
        self.assertEqual(
            raw, [], f"unescaped interpolation into markup: {raw}"
        )


class TheApiContractTests(unittest.TestCase):
    """The endpoint's shape, exercised through the handler's own query path."""

    def setUp(self):
        import tempfile

        from core.alert_priority import grade
        from core.alert_store import append_alert, build_record

        self._dir = tempfile.TemporaryDirectory()
        self.store = pathlib.Path(self._dir.name) / "alerts.jsonl"
        for ticker, state, severity, at in (
            ("NVDA", "DISAPPEARED", "warn", "2026-10-01T23:12:00+00:00"),
            ("MSFT", "CONFIRMED", "warn", "2026-10-01T22:00:00+00:00"),
            ("KO", "LOW_IMPACT", "info", "2026-09-28T14:00:00+00:00"),
        ):
            alert = {
                "alert": "regime_change",
                "ticker": ticker,
                "horizon": "20d",
                "as_of": "2026-10-01",
                "kind": state,
                "severity": severity,
                "reason": f"{ticker} reached {state}",
            }
            append_alert(
                build_record(alert, grade(alert), title=f"{ticker} {state}",
                             detected_at=at),
                self.store,
            )

    def tearDown(self):
        self._dir.cleanup()

    def test_the_feed_is_newest_first(self):
        from core.alert_store import query

        stamps = [row["detected_at"] for row in query(path=self.store)]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_the_filter_vocabularies_come_from_the_data(self):
        # A filter must never offer a value that returns nothing.
        from core.alert_store import event_types_seen, tickers_seen

        self.assertEqual(tickers_seen(self.store), ["KO", "MSFT", "NVDA"])
        self.assertEqual(event_types_seen(self.store), ["regime_change"])

    def test_a_local_date_filter_finds_an_evening_alert(self):
        """The regression this suite exists for.

        NVDA's alert is 23:12Z and MSFT's 22:00Z on 2026-10-01, which at UTC+3
        are 02:12 and 01:00 local on the 2nd. BOTH belong to the operator's
        "today"; KO (14:00Z on 09-28) does not. Before the fix this query
        returned nothing at all.
        """
        from core.alert_store import query

        found = query(path=self.store, start="2026-10-02", end="2026-10-02",
                      utc_offset_minutes=180)
        self.assertEqual([row["ticker"] for row in found], ["NVDA", "MSFT"])

    def test_the_same_filter_read_as_utc_finds_nothing(self):
        # The defect, preserved: every row is 2026-10-01 or earlier in UTC.
        from core.alert_store import query

        self.assertEqual(
            query(path=self.store, start="2026-10-02", end="2026-10-02"), []
        )

    def test_the_counts_include_the_ungraded(self):
        from core.alert_store import counts_by_priority

        counts = counts_by_priority(self.store)
        self.assertIn("(ungraded)", counts)

    def test_the_handler_declares_the_alerts_route(self):
        source = (REPO_ROOT / "web_app.py").read_text(encoding="utf-8")
        self.assertIn('parsed.path == "/api/alerts"', source)

    def test_the_handler_passes_the_offset_through(self):
        source = (REPO_ROOT / "web_app.py").read_text(encoding="utf-8")
        self.assertIn("utc_offset_minutes=offset", source)

    def test_the_handler_returns_the_filter_vocabularies(self):
        source = (REPO_ROOT / "web_app.py").read_text(encoding="utf-8")
        for call in ("tickers_seen()", "event_types_seen()", "counts_by_priority()"):
            with self.subTest(call=call):
                self.assertIn(call, source)

    def test_a_bad_limit_is_rejected_rather_than_coerced(self):
        source = (REPO_ROOT / "web_app.py").read_text(encoding="utf-8")
        self.assertIn("limit must be an integer", source)

    def test_the_route_is_registered_before_the_static_fallback(self):
        # A route declared after the catch-all would never be reached.
        source = (REPO_ROOT / "web_app.py").read_text(encoding="utf-8")
        alerts_at = source.index('parsed.path == "/api/alerts"')
        static_at = source.index('parsed.path in ("/", "/index.html")')
        self.assertLess(alerts_at, static_at)


class TheLedgersStayOutOfTheRepositoryTests(unittest.TestCase):
    """A tracked operational ledger is the checkout-dependence bug class."""

    def test_the_alert_ledgers_are_gitignored(self):
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for path in ("data/alerts.jsonl", "data/alert_deliveries.jsonl"):
            with self.subTest(path=path):
                self.assertIn(path, ignore)


if __name__ == "__main__":
    unittest.main()
