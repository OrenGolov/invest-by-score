"""Tests for Telegram delivery.

The transport is substituted everywhere via ``poster``; no test reaches
the network. The real credential is exercised only by
``scripts/verify_telegram.py``, which is a setup verifier rather than a
gate for exactly that reason.
"""

from __future__ import annotations

import unittest
from unittest import mock

from core.config import (
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_FAILED,
    ALERT_DELIVERY_OK,
    ALERT_DELIVERY_STATUSES,
    ALERT_DELIVERY_UNCONFIGURED,
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_DELIVERY_STATUSES,
    TELEGRAM_MAX_MESSAGE_CHARS,
)
from core.telegram_delivery import (
    TelegramResult,
    credential_problems,
    discover_chat_id,
    render_result,
    send_telegram_alert,
    truncate,
)

GOOD_TOKEN = "8012345678:AAHxyz_abcdefghijklmnopqrstuvwxyz123"
GOOD_CHAT = "123456789"


def poster_for(payload, status=200, raises=None):
    """A substitute transport that records what it was asked to send."""
    calls = []

    def post(url, body, timeout):
        calls.append({"url": url, "body": body, "timeout": timeout})
        if raises is not None:
            raise raises
        return status, payload

    post.calls = calls
    return post


class StatusVocabularyTests(unittest.TestCase):
    def test_both_channels_share_one_vocabulary(self):
        """A caller asking 'did it send?' must not learn two answers."""
        self.assertEqual(TELEGRAM_DELIVERY_STATUSES, ALERT_DELIVERY_STATUSES)

    def test_unconfigured_is_not_ok(self):
        result = TelegramResult(
            status=ALERT_DELIVERY_UNCONFIGURED, reason="nothing set"
        )
        self.assertFalse(result.ok)

    def test_only_delivered_is_ok(self):
        for status in TELEGRAM_DELIVERY_STATUSES:
            with self.subTest(status=status):
                result = TelegramResult(status=status, reason="r")
                self.assertEqual(result.ok, status == ALERT_DELIVERY_OK)

    def test_an_unknown_status_is_refused(self):
        with self.assertRaises(ValueError):
            TelegramResult(status="probably_sent", reason="r")

    def test_a_failure_without_a_reason_is_refused(self):
        with self.assertRaises(ValueError):
            TelegramResult(status=ALERT_DELIVERY_FAILED)


class CredentialTests(unittest.TestCase):
    def test_a_plausible_pair_has_no_problems(self):
        self.assertEqual(credential_problems(GOOD_TOKEN, GOOD_CHAT), [])

    def test_a_negative_chat_id_is_valid(self):
        """A group chat id is negative."""
        self.assertEqual(credential_problems(GOOD_TOKEN, "-1001234567"), [])

    def test_missing_token_is_named(self):
        problems = credential_problems("", GOOD_CHAT)
        self.assertTrue(any(TELEGRAM_BOT_TOKEN_ENV in p for p in problems))

    def test_missing_chat_id_points_at_the_discovery_step(self):
        problems = credential_problems(GOOD_TOKEN, "")
        self.assertTrue(any("--discover" in p for p in problems))

    def test_a_short_token_reports_its_length(self):
        problems = credential_problems("123:abc", GOOD_CHAT)
        self.assertTrue(any("7 characters" in p for p in problems))

    def test_a_token_of_the_wrong_shape_is_refused(self):
        """Length alone is too weak; the shape is '<digits>:<secret>'."""
        problems = credential_problems("x" * 50, GOOD_CHAT)
        self.assertTrue(any("shaped" in p for p in problems))

    def test_a_non_numeric_chat_id_is_refused(self):
        problems = credential_problems(GOOD_TOKEN, "@oren")
        self.assertTrue(any("numeric" in p for p in problems))


class TruncationTests(unittest.TestCase):
    def test_a_short_message_is_untouched(self):
        text, truncated = truncate("hello")
        self.assertEqual(text, "hello")
        self.assertFalse(truncated)

    def test_a_long_message_fits_the_hard_limit(self):
        """Telegram REFUSES an over-long message rather than trimming it."""
        text, truncated = truncate("x" * 9000)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text), TELEGRAM_MAX_MESSAGE_CHARS)

    def test_truncation_is_visible_in_the_message(self):
        """A reader who cannot see the cut will act on a partial alert."""
        text, _ = truncate("x" * 9000)
        self.assertIn("truncated", text)

    def test_a_message_at_exactly_the_limit_is_not_truncated(self):
        text, truncated = truncate("x" * TELEGRAM_MAX_MESSAGE_CHARS)
        self.assertFalse(truncated)
        self.assertEqual(len(text), TELEGRAM_MAX_MESSAGE_CHARS)


class SendTests(unittest.TestCase):
    def test_a_configured_send_reaches_the_transport(self):
        post = poster_for({"ok": True})
        result = send_telegram_alert(
            "score moved", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertTrue(result.ok)
        self.assertEqual(post.calls[0]["body"]["text"], "score moved")
        self.assertEqual(post.calls[0]["body"]["chat_id"], GOOD_CHAT)

    def test_the_token_is_in_the_url_not_the_body(self):
        """Telegram authenticates by path; the body carries no secret."""
        post = poster_for({"ok": True})
        send_telegram_alert("x", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post)
        self.assertIn(GOOD_TOKEN, post.calls[0]["url"])
        self.assertNotIn(GOOD_TOKEN, str(post.calls[0]["body"]))

    def test_the_api_is_always_https(self):
        post = poster_for({"ok": True})
        send_telegram_alert("x", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post)
        self.assertTrue(post.calls[0]["url"].startswith("https://"))

    def test_an_unset_credential_sends_nothing(self):
        post = poster_for({"ok": True})
        result = send_telegram_alert("x", token="", chat_id="", poster=post)
        self.assertEqual(result.status, ALERT_DELIVERY_UNCONFIGURED)
        self.assertEqual(post.calls, [])

    def test_a_rejected_token_is_auth_failed(self):
        post = poster_for(
            {"ok": False, "description": "Unauthorized"}, status=401
        )
        result = send_telegram_alert(
            "x", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertEqual(result.status, ALERT_DELIVERY_AUTH_FAILED)
        self.assertIn("BotFather", result.reason)

    def test_chat_not_found_explains_that_a_bot_cannot_open_a_chat(self):
        """The single most likely setup error, named precisely."""
        post = poster_for(
            {"ok": False, "description": "Bad Request: chat not found"},
            status=400,
        )
        result = send_telegram_alert(
            "x", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertEqual(result.status, ALERT_DELIVERY_AUTH_FAILED)
        self.assertIn("send your bot a message first", result.reason)

    def test_a_rate_limit_is_failed_not_auth_failed(self):
        """One is fixed by waiting, the other by changing a credential."""
        post = poster_for(
            {"ok": False, "description": "Too Many Requests"}, status=429
        )
        result = send_telegram_alert(
            "x", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertEqual(result.status, ALERT_DELIVERY_FAILED)

    def test_nothing_raises_into_the_caller(self):
        """A notification outage must never abort a collection run."""
        for failure in (OSError("down"), ValueError("bad json"), TimeoutError()):
            with self.subTest(failure=type(failure).__name__):
                result = send_telegram_alert(
                    "x", token=GOOD_TOKEN, chat_id=GOOD_CHAT,
                    poster=poster_for({}, raises=failure),
                )
                self.assertIn(result.status, TELEGRAM_DELIVERY_STATUSES)
                self.assertFalse(result.ok)

    def test_an_empty_alert_is_refused_before_sending(self):
        post = poster_for({"ok": True})
        result = send_telegram_alert(
            "   ", token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertEqual(result.status, ALERT_DELIVERY_FAILED)
        self.assertEqual(post.calls, [])

    def test_a_long_alert_is_sent_truncated_rather_than_dropped(self):
        post = poster_for({"ok": True})
        result = send_telegram_alert(
            "x" * 9000, token=GOOD_TOKEN, chat_id=GOOD_CHAT, poster=post
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.truncated)
        self.assertLessEqual(
            len(post.calls[0]["body"]["text"]), TELEGRAM_MAX_MESSAGE_CHARS
        )

    def test_the_credential_is_read_from_the_environment(self):
        post = poster_for({"ok": True})
        with mock.patch.dict(
            "os.environ",
            {TELEGRAM_BOT_TOKEN_ENV: GOOD_TOKEN, TELEGRAM_CHAT_ID_ENV: GOOD_CHAT},
        ):
            result = send_telegram_alert("x", poster=post)
        self.assertTrue(result.ok)


class DiscoveryTests(unittest.TestCase):
    def test_a_chat_id_is_found_in_the_updates(self):
        post = poster_for({
            "ok": True,
            "result": [{"message": {"chat": {"id": 987654321, "username": "oren"}}}],
        })
        chat_id, explanation = discover_chat_id(GOOD_TOKEN, poster=post)
        self.assertEqual(chat_id, "987654321")
        self.assertIn("oren", explanation)

    def test_the_newest_chat_wins(self):
        post = poster_for({
            "ok": True,
            "result": [
                {"message": {"chat": {"id": 1, "first_name": "old"}}},
                {"message": {"chat": {"id": 2, "first_name": "new"}}},
            ],
        })
        chat_id, _ = discover_chat_id(GOOD_TOKEN, poster=post)
        self.assertEqual(chat_id, "2")

    def test_no_messages_explains_that_the_operator_must_write_first(self):
        post = poster_for({"ok": True, "result": []})
        chat_id, explanation = discover_chat_id(GOOD_TOKEN, poster=post)
        self.assertEqual(chat_id, "")
        self.assertIn("send it any", explanation)

    def test_an_unset_token_is_reported_not_raised(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            chat_id, explanation = discover_chat_id()
        self.assertEqual(chat_id, "")
        self.assertIn(TELEGRAM_BOT_TOKEN_ENV, explanation)

    def test_discovery_never_raises(self):
        chat_id, explanation = discover_chat_id(
            GOOD_TOKEN, poster=poster_for({}, raises=OSError("down"))
        )
        self.assertEqual(chat_id, "")
        self.assertIn("OSError", explanation)


class RenderTests(unittest.TestCase):
    def test_a_delivered_alert_names_its_chat(self):
        line = render_result(
            TelegramResult(status=ALERT_DELIVERY_OK, chat_id=GOOD_CHAT)
        )
        self.assertIn(GOOD_CHAT, line)

    def test_truncation_is_visible_in_the_rendered_line(self):
        line = render_result(
            TelegramResult(
                status=ALERT_DELIVERY_OK, chat_id=GOOD_CHAT, truncated=True
            )
        )
        self.assertIn("truncated", line)

    def test_a_failure_renders_its_reason(self):
        line = render_result(
            TelegramResult(status=ALERT_DELIVERY_FAILED, reason="rate limited")
        )
        self.assertIn("rate limited", line)


if __name__ == "__main__":
    unittest.main()
