"""Tests for alert delivery.

The transport is substituted everywhere via ``smtp_factory``; no test
reaches the network. The real credential is exercised only by
``scripts/verify_alert_email.py``, which is a setup verifier and not a gate
for exactly that reason.
"""

from __future__ import annotations

import smtplib
import unittest
from unittest import mock

from core.alert_delivery import (
    DeliveryResult,
    build_message,
    credential_problems,
    normalise_app_password,
    render_delivery,
    resolve_recipient,
    send_alert_email,
)
from core.config import (
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_FAILED,
    ALERT_DELIVERY_OK,
    ALERT_DELIVERY_STATUSES,
    ALERT_DELIVERY_UNCONFIGURED,
    ALERT_EMAIL_PASSWORD_ENV,
    ALERT_EMAIL_TO_ENV,
    ALERT_EMAIL_USER_ENV,
)

GOOD_USER = "operator@gmail.com"
GOOD_PASSWORD = "cygkigirbvipnqtg"


class FakeSMTP:
    """A stand-in transport that records what it was asked to do."""

    def __init__(self, host, port, timeout=None, *, fail=None, logins=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._fail = fail
        self.started_tls = False
        self.logged_in_as = None
        self.sent = []
        self.ehlo_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ehlo(self):
        self.ehlo_calls += 1

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        if isinstance(self._fail, smtplib.SMTPAuthenticationError):
            raise self._fail
        self.logged_in_as = (user, password)

    def send_message(self, message):
        if isinstance(self._fail, (OSError, smtplib.SMTPException)):
            raise self._fail
        self.sent.append(message)


def factory_for(**kwargs):
    created = []

    def make(host, port, timeout=None):
        server = FakeSMTP(host, port, timeout, **kwargs)
        created.append(server)
        return server

    make.created = created
    return make


class NormalisePasswordTests(unittest.TestCase):
    def test_display_spaces_are_stripped(self):
        self.assertEqual(
            normalise_app_password("cygk igir bvip nqtg"), GOOD_PASSWORD
        )

    def test_trailing_newline_from_a_paste_is_stripped(self):
        self.assertEqual(
            normalise_app_password(" cygk igir bvip nqtg \n"), GOOD_PASSWORD
        )

    def test_none_and_empty_are_empty(self):
        self.assertEqual(normalise_app_password(None), "")
        self.assertEqual(normalise_app_password(""), "")

    def test_an_already_clean_password_is_unchanged(self):
        self.assertEqual(normalise_app_password(GOOD_PASSWORD), GOOD_PASSWORD)


class CredentialProblemTests(unittest.TestCase):
    def test_a_plausible_pair_has_no_problems(self):
        self.assertEqual(credential_problems(GOOD_USER, GOOD_PASSWORD), [])

    def test_spaces_in_the_password_are_not_a_problem(self):
        self.assertEqual(
            credential_problems(GOOD_USER, "cygk igir bvip nqtg"), []
        )

    def test_missing_user_is_named(self):
        problems = credential_problems("", GOOD_PASSWORD)
        self.assertTrue(any(ALERT_EMAIL_USER_ENV in p for p in problems))

    def test_missing_password_is_named(self):
        problems = credential_problems(GOOD_USER, "")
        self.assertTrue(any(ALERT_EMAIL_PASSWORD_ENV in p for p in problems))

    def test_both_missing_reports_both(self):
        self.assertEqual(len(credential_problems("", "")), 2)

    def test_a_user_without_an_at_sign_is_rejected(self):
        problems = credential_problems("operator", GOOD_PASSWORD)
        self.assertTrue(any("not an email address" in p for p in problems))

    def test_a_short_password_reports_its_length(self):
        problems = credential_problems(GOOD_USER, "abc")
        self.assertTrue(any("3 characters" in p for p in problems))

    def test_a_sixteen_character_account_password_is_still_refused(self):
        """Length alone is too weak a check.

        MEASURED while building this module: "my-real-password" is exactly
        16 characters and passed a length-only guard, so an account
        password would reach Gmail and come back as an unexplained
        auth_failed.
        """
        problems = credential_problems(GOOD_USER, "my-real-password")
        self.assertTrue(any("lowercase letters" in p for p in problems))

    def test_sixteen_digits_are_refused(self):
        problems = credential_problems(GOOD_USER, "1234567890123456")
        self.assertTrue(any("lowercase letters" in p for p in problems))

    def test_an_uppercase_password_is_refused(self):
        problems = credential_problems(GOOD_USER, GOOD_PASSWORD.upper())
        self.assertTrue(any("lowercase letters" in p for p in problems))


class RecipientTests(unittest.TestCase):
    def test_an_explicit_recipient_wins(self):
        self.assertEqual(
            resolve_recipient(GOOD_USER, "someone@else.com"),
            "someone@else.com",
        )

    def test_it_falls_back_to_the_sending_account(self):
        self.assertEqual(resolve_recipient(GOOD_USER), GOOD_USER)

    def test_the_to_variable_is_used_when_no_explicit_recipient(self):
        with mock.patch.dict(
            "os.environ", {ALERT_EMAIL_TO_ENV: "env@target.com"}
        ):
            self.assertEqual(resolve_recipient(GOOD_USER), "env@target.com")

    def test_nothing_anywhere_is_empty(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_recipient(None), "")


class DeliveryResultTests(unittest.TestCase):
    def test_unconfigured_is_not_ok(self):
        result = DeliveryResult(
            status=ALERT_DELIVERY_UNCONFIGURED, reason="nothing is set"
        )
        self.assertFalse(result.ok)

    def test_only_delivered_is_ok(self):
        for status in ALERT_DELIVERY_STATUSES:
            result = DeliveryResult(status=status, reason="r")
            self.assertEqual(result.ok, status == ALERT_DELIVERY_OK)

    def test_an_unknown_status_is_refused(self):
        with self.assertRaises(ValueError):
            DeliveryResult(status="probably_sent", reason="r")

    def test_a_failure_without_a_reason_is_refused(self):
        """A failure nobody can act on is not a usable report."""
        with self.assertRaises(ValueError):
            DeliveryResult(status=ALERT_DELIVERY_FAILED)

    def test_success_needs_no_reason(self):
        self.assertTrue(DeliveryResult(status=ALERT_DELIVERY_OK).ok)


class SendAlertEmailTests(unittest.TestCase):
    def test_a_configured_send_reaches_the_transport(self):
        factory = factory_for()
        result = send_alert_email(
            "subject",
            "body",
            user=GOOD_USER,
            password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertTrue(result.ok)
        server = factory.created[0]
        self.assertEqual(len(server.sent), 1)
        self.assertEqual(server.sent[0]["Subject"], "subject")

    def test_starttls_is_always_negotiated(self):
        """An app password must never cross the wire in clear text."""
        factory = factory_for()
        send_alert_email(
            "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertTrue(factory.created[0].started_tls)

    def test_the_password_is_stripped_before_login(self):
        factory = factory_for()
        send_alert_email(
            "s", "b", user=GOOD_USER, password="cygk igir bvip nqtg",
            smtp_factory=factory,
        )
        self.assertEqual(
            factory.created[0].logged_in_as, (GOOD_USER, GOOD_PASSWORD)
        )

    def test_an_unset_credential_reports_unconfigured_and_sends_nothing(self):
        factory = factory_for()
        result = send_alert_email(
            "s", "b", user="", password="", smtp_factory=factory
        )
        self.assertEqual(result.status, ALERT_DELIVERY_UNCONFIGURED)
        self.assertFalse(result.ok)
        self.assertEqual(factory.created, [])

    def test_unconfigured_carries_the_problems_it_found(self):
        result = send_alert_email("s", "b", user="", password="")
        self.assertEqual(len(result.problems), 2)

    def test_a_refused_credential_is_auth_failed_not_failed(self):
        """The two route to different fixes and must stay distinct."""
        factory = factory_for(
            fail=smtplib.SMTPAuthenticationError(535, b"bad creds")
        )
        result = send_alert_email(
            "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertEqual(result.status, ALERT_DELIVERY_AUTH_FAILED)
        self.assertIn("app password", result.reason)

    def test_an_unreachable_server_is_failed(self):
        factory = factory_for(fail=OSError("connection refused"))
        result = send_alert_email(
            "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertEqual(result.status, ALERT_DELIVERY_FAILED)

    def test_an_smtp_exception_is_failed(self):
        factory = factory_for(fail=smtplib.SMTPDataError(451, b"try later"))
        result = send_alert_email(
            "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertEqual(result.status, ALERT_DELIVERY_FAILED)

    def test_nothing_raises_into_the_caller(self):
        """A notification outage must never abort a collection run.

        The daily collector captures perishable data: news is gone after
        seven days, so an email failure that propagated would turn a
        channel outage into a permanent hole in the ledger.
        """
        for failure in (
            OSError("down"),
            smtplib.SMTPAuthenticationError(535, b"no"),
            smtplib.SMTPServerDisconnected("bye"),
        ):
            with self.subTest(failure=type(failure).__name__):
                result = send_alert_email(
                    "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
                    smtp_factory=factory_for(fail=failure),
                )
                self.assertIn(result.status, ALERT_DELIVERY_STATUSES)
                self.assertFalse(result.ok)

    def test_an_empty_subject_is_refused_before_connecting(self):
        factory = factory_for()
        result = send_alert_email(
            "   ", "b", user=GOOD_USER, password=GOOD_PASSWORD,
            smtp_factory=factory,
        )
        self.assertEqual(result.status, ALERT_DELIVERY_FAILED)
        self.assertEqual(factory.created, [])

    def test_the_credential_is_read_from_the_environment_when_not_passed(self):
        factory = factory_for()
        with mock.patch.dict(
            "os.environ",
            {
                ALERT_EMAIL_USER_ENV: GOOD_USER,
                ALERT_EMAIL_PASSWORD_ENV: GOOD_PASSWORD,
            },
        ):
            result = send_alert_email("s", "b", smtp_factory=factory)
        self.assertTrue(result.ok)

    def test_the_recipient_defaults_to_the_sender(self):
        factory = factory_for()
        with mock.patch.dict("os.environ", {}, clear=True):
            result = send_alert_email(
                "s", "b", user=GOOD_USER, password=GOOD_PASSWORD,
                smtp_factory=factory,
            )
        self.assertEqual(result.recipient, GOOD_USER)


class MessageTests(unittest.TestCase):
    def test_a_date_header_is_set_explicitly(self):
        """A catch-up run must not look current.

        StartWhenAvailable means a run missed overnight fires late; without
        an explicit Date some clients stamp receipt time instead.
        """
        message = build_message(GOOD_USER, GOOD_USER, "s", "b")
        self.assertTrue(message["Date"])

    def test_the_body_survives_intact(self):
        message = build_message(GOOD_USER, GOOD_USER, "s", "line1\nline2")
        self.assertIn("line1", message.get_content())

    def test_the_version_is_recorded_on_the_message(self):
        message = build_message(GOOD_USER, GOOD_USER, "s", "b")
        self.assertTrue(message["X-Alert-Version"])


class RenderTests(unittest.TestCase):
    def test_a_delivered_alert_renders_its_recipient(self):
        line = render_delivery(
            DeliveryResult(
                status=ALERT_DELIVERY_OK, recipient=GOOD_USER, subject="s"
            )
        )
        self.assertIn(GOOD_USER, line)

    def test_a_failure_renders_its_reason(self):
        line = render_delivery(
            DeliveryResult(
                status=ALERT_DELIVERY_AUTH_FAILED, reason="refused"
            )
        )
        self.assertIn("refused", line)


if __name__ == "__main__":
    unittest.main()
