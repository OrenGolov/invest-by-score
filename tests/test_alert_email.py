"""A8 email tests — the operator's template, and the things that must never leak.

The operator specified the format exactly, so most of these assert the spec:

    Subject:  <Priority>: <Ticker> - <Short, Direct Description>
    Body:     Action, Relevance Date, then Who / What / When / Why it Matters

Three tests are not about formatting and matter more:

* `test_no_credential_appears_in_any_rendered_output` — a password rendered into a
  message body would be mailed to a third party by the feature's own design.
* `test_a_missing_credential_degrades_rather_than_raising` — the dashboard receives
  every alert regardless, so an unset password must cost the email channel alone.
* `test_a_hostile_title_cannot_inject_markup` — alert text includes provider
  headlines, which are untrusted input rendered into HTML.
"""

from __future__ import annotations

import os
import unittest
from html.parser import HTMLParser
from unittest import mock

from core.alert_email import (
    MARK_IMMEDIATE,
    MARK_MONITORING,
    AlertEmailError,
    build_digest,
    build_message,
    colour_of,
    credentials,
    render_html,
    render_text,
    routing,
    send,
    subject_for,
)
from core.alert_priority import grade
from core.alert_store import build_record
from core.config import (
    ALERT_EMAIL_MAX_WIDTH_PX,
    ALERT_EMAIL_PASSWORD_ENV,
    ALERT_EMAIL_TO_ENV,
    ALERT_EMAIL_USER_ENV,
    ALERT_PRIORITIES,
)


class Balance(HTMLParser):
    """Every opened tag is closed, in order. A broken table hides rows in a client."""

    VOID = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
        else:
            self.stack.pop()

    @classmethod
    def problems(cls, markup: str) -> list[str]:
        parser = cls()
        parser.feed(markup)
        return parser.errors + ([f"unclosed {parser.stack}"] if parser.stack else [])


def record(
    *,
    ticker="AVGO",
    state="HIGH_IMPACT",
    severity="warn",
    title="Earnings Results",
    name="event_impact",
    at="2026-10-01T20:15:00+00:00",
    summary=None,
    **detail,
):
    alert = {
        "alert": name,
        "ticker": ticker,
        "horizon": "20d",
        "as_of": "2026-10-01",
        "verdict": state,
        "severity": severity,
        "reason": f"{ticker} reported {state}",
        **detail,
    }
    return build_record(
        alert, grade(alert), title=title, detected_at=at, summary=summary
    )


NO_CREDENTIALS = {
    ALERT_EMAIL_USER_ENV: "",
    ALERT_EMAIL_PASSWORD_ENV: "",
    ALERT_EMAIL_TO_ENV: "",
}


class SubjectFormatTests(unittest.TestCase):
    def test_the_subject_matches_the_specified_format(self):
        self.assertEqual(
            subject_for(record()), "Very High: AVGO - Earnings Results"
        )

    def test_the_operators_own_examples_render(self):
        # Straight from the specification.
        self.assertEqual(
            subject_for(record(ticker="MU", title="Investor Call")),
            "Very High: MU - Investor Call",
        )
        self.assertEqual(
            subject_for(record(ticker="NVDA", title="New AI Partnership")),
            "Very High: NVDA - New AI Partnership",
        )

    def test_an_ungraded_alert_is_labelled_rather_than_left_blank(self):
        row = record(state="NO_EVENT", severity=None)
        self.assertIn("Unprioritised", subject_for(row))

    def test_a_portfolio_wide_alert_has_a_subject(self):
        row = record(ticker=None)
        self.assertIn("PORTFOLIO", subject_for(row))


class PlainTextBodyTests(unittest.TestCase):
    def setUp(self):
        self.row = record(
            summary={
                "who": "Broadcom (AVGO)",
                "what": "Reported quarterly earnings above analyst expectations.",
                "when": "2026-10-01 after market close.",
                "why": [
                    "Revenue exceeded estimates by 8%.",
                    "AI-related revenue grew significantly.",
                    "Management raised full-year guidance.",
                ],
            },
        )
        self.row["monitor_date"] = "2026-10-29"
        self.text = render_text(self.row)

    def test_every_specified_section_is_present(self):
        for needle in ("Action: ", "Who:", "What:", "When:", "Why it Matters:"):
            with self.subTest(section=needle):
                self.assertIn(needle, self.text)

    def test_the_time_sensitive_marker_is_present_for_an_immediate_band(self):
        self.assertIn(MARK_IMMEDIATE, self.text)

    def test_the_monitoring_marker_is_present_when_a_date_is_given(self):
        self.assertIn(MARK_MONITORING, self.text)
        self.assertIn("2026-10-29", self.text)

    def test_a_digested_band_carries_no_immediate_marker(self):
        quiet = record(state="LOW_IMPACT", severity="info", title="Dividend")
        self.assertNotIn(MARK_IMMEDIATE, render_text(quiet))

    def test_the_operators_bullets_are_preserved_verbatim(self):
        self.assertIn("Revenue exceeded estimates by 8%.", self.text)
        self.assertIn("  - ", self.text)

    def test_the_not_advice_notice_is_present(self):
        # Rule 5: the system is analysis-only until the release gates pass.
        self.assertIn("not financial advice", self.text)

    def test_a_plain_text_part_exists_at_all(self):
        # An HTML-only message renders empty in a client with HTML disabled, and
        # the text part is what a phone's lock-screen preview shows.
        self.assertTrue(self.text.strip())


class MeasurementRenderingTests(unittest.TestCase):
    def test_an_unmeasured_value_is_not_coalesced_to_zero(self):
        # The project's shape rule. A1 MEASURED that coercing an absent value
        # reports a -0.56 crash that never happened.
        text = render_text(record(previous=None, current=0.0612))
        self.assertIn("(not measured)", text)
        self.assertNotIn("Previous: 0.0000", text)

    def test_a_count_renders_as_an_integer(self):
        # CAUGHT IN A RENDERED SAMPLE: 94 comparable events printed as
        # "94.0000", which claims a precision the number does not have.
        text = render_text(record(analogs=94))
        self.assertIn("Comparable events: 94", text)
        self.assertNotIn("94.0000", text)

    def test_a_rate_keeps_four_decimals(self):
        text = render_text(record(median_abs_move=0.0497))
        self.assertIn("0.0497", text)

    def test_a_boolean_does_not_render_as_a_number(self):
        # bool IS an int in Python, so an unguarded numeric branch prints 1.0000.
        text = render_text(record(threshold=True))
        self.assertIn("Threshold: yes", text)

    def test_a_genuine_zero_is_shown_as_a_measurement(self):
        # 0.0 was MEASURED; it is not the same as absent.
        text = render_text(record(delta=0.0))
        self.assertIn("Change: 0.0000", text)
        self.assertNotIn("Change: —", text)


class HtmlBodyTests(unittest.TestCase):
    def test_the_markup_is_balanced(self):
        self.assertEqual(Balance.problems(render_html(record())), [])

    def test_the_layout_is_capped_for_a_phone(self):
        markup = render_html(record())
        self.assertIn(f"max-width:{ALERT_EMAIL_MAX_WIDTH_PX}px", markup)
        self.assertIn('name="viewport"', markup)

    def test_a_hostile_title_cannot_inject_markup(self):
        # Alert text includes provider headlines: untrusted input rendered into
        # HTML that is then mailed.
        markup = render_html(record(title='<script>alert("x")</script>'))
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_an_ampersand_in_a_ticker_is_escaped(self):
        self.assertIn("A&amp;B", render_html(record(ticker="A&B")))

    def test_a_hostile_reason_cannot_inject_markup(self):
        row = record()
        row["reason"] = '<img src=x onerror="steal()">'
        self.assertNotIn("<img", render_html(row))

    def test_each_band_renders_its_own_colour(self):
        seen = set()
        for state, severity in (
            ("HIGH_IMPACT", "warn"),
            ("LOW_IMPACT", "info"),
            ("UNKNOWN_IMPACT", "info"),
        ):
            row = record(state=state, severity=severity)
            markup = render_html(row)
            self.assertIn("border-radius", markup)
            seen.add(row["priority"])
        self.assertGreater(len(seen), 1)

    def test_the_colour_names_match_the_dashboard(self):
        # An email and the Monitoring tab must not disagree about what Urgent
        # looks like.
        for band in ALERT_PRIORITIES:
            with self.subTest(band=band):
                self.assertTrue(colour_of(band))
        self.assertEqual(colour_of(None), "grey")


class RoutingTests(unittest.TestCase):
    def test_the_three_most_urgent_bands_are_immediate(self):
        for band in ("Urgent", "Very High", "High"):
            with self.subTest(band=band):
                self.assertEqual(routing(band)[0], "immediate")

    def test_the_two_lowest_bands_are_digested(self):
        for band in ("Medium", "Low"):
            with self.subTest(band=band):
                self.assertEqual(routing(band)[0], "digest")

    def test_every_band_has_a_route(self):
        # A band with no route would be email-silent, which the operator
        # explicitly refused.
        for band in ALERT_PRIORITIES:
            with self.subTest(band=band):
                self.assertIn(routing(band)[0], ("immediate", "digest"))

    def test_an_ungraded_alert_is_digested_not_dropped(self):
        route, reason = routing(None)
        self.assertEqual(route, "digest")
        self.assertIn("blind detector", reason)

    def test_an_unknown_band_is_refused(self):
        with self.assertRaises(AlertEmailError):
            routing("Catastrophic")


class MessageAssemblyTests(unittest.TestCase):
    def test_a_message_carries_both_a_text_and_an_html_part(self):
        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        types = {
            part.get_content_type()
            for part in message.walk()
            if not part.is_multipart()
        }
        self.assertIn("text/plain", types)
        self.assertIn("text/html", types)

    def test_the_alert_id_travels_as_a_header(self):
        # So a client seeing the same finding twice can thread them.
        row = record()
        message = build_message(row, sender="a@b.c", recipient="d@e.f")
        self.assertEqual(message["X-Alert-Id"], row["alert_id"])

    def test_an_empty_address_is_refused(self):
        for sender, recipient in (("", "d@e.f"), ("a@b.c", "")):
            with self.subTest(sender=sender, recipient=recipient):
                with self.assertRaises(AlertEmailError):
                    build_message(record(), sender=sender, recipient=recipient)


class DigestTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            record(ticker="KO", state="LOW_IMPACT", severity="info",
                   title="Dividend Declaration", at="2026-10-01T18:00:00+00:00"),
            record(ticker="AMD", state="UNKNOWN_IMPACT", severity="info",
                   title="Unclassified Event", at="2026-10-01T19:00:00+00:00"),
            record(ticker="NVDA", state="LOW_IMPACT", severity="info",
                   title="Analyst Note", at="2026-10-01T17:00:00+00:00"),
        ]
        self.digest = build_digest(
            self.rows, sender="a@b.c", recipient="d@e.f", as_of="2026-10-01"
        )

    def test_the_subject_states_the_count_and_the_tally(self):
        subject = self.digest["Subject"]
        self.assertIn("3 alerts", subject)
        self.assertIn("2026-10-01", subject)

    def test_the_most_urgent_alert_leads(self):
        body = self.digest.get_body(preferencelist=("plain",)).get_content()
        headers = [line for line in body.splitlines() if line.startswith("--- ")]
        self.assertIn("Medium", headers[0])

    def test_the_digest_html_is_balanced(self):
        markup = self.digest.get_body(preferencelist=("html",)).get_content()
        self.assertEqual(Balance.problems(markup), [])

    def test_every_alert_appears(self):
        body = self.digest.get_body(preferencelist=("plain",)).get_content()
        for row in self.rows:
            with self.subTest(ticker=row["ticker"]):
                self.assertIn(row["ticker"], body)

    def test_the_digest_carries_the_not_advice_notice(self):
        body = self.digest.get_body(preferencelist=("plain",)).get_content()
        self.assertIn("not financial advice", body)

    def test_an_empty_digest_is_refused(self):
        # Mailing "nothing happened" every evening trains the reader to ignore
        # the channel.
        with self.assertRaises(AlertEmailError):
            build_digest([], sender="a@b.c", recipient="d@e.f", as_of="2026-10-01")


class CredentialHandlingTests(unittest.TestCase):
    def test_no_credential_appears_in_any_rendered_output(self):
        # A password rendered into a body would be mailed out by design.
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "someone@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "sixteencharsecret",
                ALERT_EMAIL_TO_ENV: "someone@example.com",
            },
        ):
            row = record()
            blob = render_text(row) + render_html(row)
            digest = build_digest(
                [row], sender="a@b.c", recipient="d@e.f", as_of="2026-10-01"
            )
            blob += digest.get_body(preferencelist=("plain",)).get_content()
            blob += digest.get_body(preferencelist=("html",)).get_content()
        self.assertNotIn("sixteencharsecret", blob)
        self.assertNotIn("password", blob.lower())

    def test_a_missing_credential_degrades_rather_than_raising(self):
        with mock.patch.dict(os.environ, NO_CREDENTIALS):
            user, password, recipient, reason = credentials()
            self.assertIsNone(user)
            self.assertIsNone(password)
            self.assertIsNone(recipient)
            self.assertIn(ALERT_EMAIL_USER_ENV, reason)

    def test_send_returns_false_rather_than_raising_without_a_credential(self):
        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        with mock.patch.dict(os.environ, NO_CREDENTIALS):
            sent, detail = send(message)
        self.assertFalse(sent)
        self.assertIn("disabled", detail)

    def test_the_reason_tells_the_operator_the_dashboard_still_works(self):
        with mock.patch.dict(os.environ, NO_CREDENTIALS):
            _, _, _, reason = credentials()
        self.assertIn("Monitoring tab", reason)

    def test_the_recipient_defaults_to_the_sender(self):
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "me@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "secret",
                ALERT_EMAIL_TO_ENV: "",
            },
        ):
            _, _, recipient, _ = credentials()
        self.assertEqual(recipient, "me@example.com")

    def test_a_dry_run_renders_without_sending(self):
        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "me@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "secret",
            },
        ):
            with mock.patch("smtplib.SMTP") as smtp:
                sent, detail = send(message, dry_run=True)
        self.assertFalse(sent)
        self.assertIn("dry run", detail)
        smtp.assert_not_called()

    def test_an_authentication_failure_names_the_app_password(self):
        # Gmail rejects an account password here, and the generic message would
        # send the operator looking at the wrong thing.
        import smtplib

        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "me@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "wrong",
            },
        ):
            with mock.patch("smtplib.SMTP") as smtp:
                smtp.return_value.__enter__.return_value.login.side_effect = (
                    smtplib.SMTPAuthenticationError(535, b"denied")
                )
                sent, detail = send(message)
        self.assertFalse(sent)
        self.assertIn("App Password", detail)

    def test_a_transient_failure_does_not_raise(self):
        # Losing the run over an SMTP timeout would turn one outage into a lost
        # day; the finding is already stored and visible.
        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "me@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "secret",
            },
        ):
            with mock.patch("smtplib.SMTP", side_effect=OSError("timed out")):
                sent, detail = send(message)
        self.assertFalse(sent)
        self.assertIn("timed out", detail)

    def test_a_successful_send_uses_starttls_before_logging_in(self):
        # Port 587 is the submission port; logging in before STARTTLS would send
        # the password in clear text.
        message = build_message(record(), sender="a@b.c", recipient="d@e.f")
        with mock.patch.dict(
            os.environ,
            {
                ALERT_EMAIL_USER_ENV: "me@example.com",
                ALERT_EMAIL_PASSWORD_ENV: "secret",
            },
        ):
            with mock.patch("smtplib.SMTP") as smtp:
                client = smtp.return_value.__enter__.return_value
                calls: list[str] = []
                client.starttls.side_effect = lambda *a, **k: calls.append("starttls")
                client.login.side_effect = lambda *a, **k: calls.append("login")
                sent, detail = send(message)
        self.assertTrue(sent)
        self.assertEqual(calls, ["starttls", "login"])


if __name__ == "__main__":
    unittest.main()
