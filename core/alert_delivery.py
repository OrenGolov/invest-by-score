"""Alert delivery — the transport the seven alert evaluators never had.

MEASURED 2026-10-04: this repo shipped seven alert modules and no way to
send one. A7's `alert_suppression.py` uses the word "delivered" nine times,
but it means DECIDED: it ranks severity, applies a cooldown, and returns a
verdict about whether a reader *should* see an alert. Nothing then showed
it to anybody. `grep -rn "smtplib|sendmail|telegram|webhook"` over the
whole tree returned zero hits, and each of the seven evaluators was
imported by exactly two files — its own CI gate and `core/config.py`.

A gate that governs delivery, in a system with no delivery, is a decision
written to nowhere.

**What this module guarantees.** `send_alert_email` returns one of four
declared statuses and NEVER raises. Three of the four are failures, and
they are kept distinct because they route the operator to different fixes:

    unconfigured  no credential is set        -> run the setup
    auth_failed   the credential was REFUSED  -> re-issue the app password
    failed        the server was unreachable  -> retry; nothing is wrong
    delivered     the server accepted it

**Why "unconfigured" is a status and not an error.** The same reasoning as
the news contract's no-key path: a missing provider is a state, never a
neutral score. A fresh clone has no credential, and a module that raised
there would make every unconfigured machine look broken.

**Why nothing raises.** The daily collector captures PERISHABLE data —
news is gone after 7 days. If a failed email aborted that run, an outage
in the notification channel would become a permanent hole in the ledger.
Delivery is strictly less important than collection and must fail softly.

**Why the password spaces are stripped.** Gmail displays an app password
in four groups of four. Pasting it verbatim is the single most likely
setup error, it is silently correctable, and the alternative is an
`auth_failed` the operator cannot explain.

**What this module does NOT do.** It does not decide whether to alert —
that is A7's job, and this module is the thing A7 was always deciding
about. It takes a rendered subject and body and transmits them.

**SMTP IS BLOCKED ON THE OPERATOR NETWORK — MEASURED 2026-10-04.** The
transport below is correct and tested, but it cannot deliver from that
machine. The measurement, in order:

    tcp connect smtp.gmail.com:587   OPEN  (142.251.127.109)
    tcp connect smtp.gmail.com:465   OPEN
    tcp connect smtp.gmail.com:25    OPEN
    SMTP greeting banner on 587      NONE within 15s
    SMTP greeting banner on 465      b''   (closed immediately)
    HTTPS to gmail.googleapis.com    200   (port 443 is unimpeded)
    Windows proxy configured         no (ProxyEnable=0)

The TCP handshake completes and the session is then swallowed. A refused
connection would be a firewall; a completed connect with no banner is
outbound SMTP INTERCEPTION, routine on a corporate network — that host is
an enterprise-managed machine.

Note what this rules out: the credential is never reached, so the timeout
says NOTHING about whether it is valid. `failed` and `auth_failed` being
distinct statuses is what makes that readable rather than a guess, and it
is the first thing this module's status split actually bought.

**The consequence for channel choice.** Any SMTP channel is dead on that
network, whichever provider it points at. A working channel must ride
port 443 — the Gmail REST API, or Telegram's bot API, both HTTPS.
Telegram is the better of the two: no OAuth flow, and a real phone push.
That is why the transport here sits behind `smtp_factory` rather than
calling `smtplib` inline — the next channel substitutes for it.
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate

from core.config import (
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_FAILED,
    ALERT_DELIVERY_OK,
    ALERT_DELIVERY_STATUSES,
    ALERT_DELIVERY_UNCONFIGURED,
    ALERT_DELIVERY_VERSION,
    ALERT_EMAIL_PASSWORD_ENV,
    ALERT_EMAIL_PASSWORD_LENGTH,
    ALERT_EMAIL_TO_ENV,
    ALERT_EMAIL_USER_ENV,
    ALERT_SMTP_HOST,
    ALERT_SMTP_PORT,
    ALERT_SMTP_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class DeliveryResult:
    """The outcome of one delivery attempt.

    ``status`` is always one of ALERT_DELIVERY_STATUSES. ``reason`` explains
    a non-OK status in terms of what the operator should DO about it.
    """

    status: str
    reason: str = ""
    recipient: str = ""
    subject: str = ""
    version: str = ALERT_DELIVERY_VERSION
    problems: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in ALERT_DELIVERY_STATUSES:
            raise ValueError(
                f"unknown delivery status {self.status!r}; expected one of "
                f"{ALERT_DELIVERY_STATUSES}"
            )
        if self.status != ALERT_DELIVERY_OK and not self.reason:
            raise ValueError(
                f"status {self.status!r} is a failure and must carry a reason; "
                f"a failure without a reason cannot be acted on"
            )

    @property
    def ok(self) -> bool:
        """True ONLY when the server accepted the message.

        Deliberately not ``status != failed``: an unconfigured channel has
        sent nothing, and treating that as success is how a silent
        notification gap survives a passing test suite.
        """
        return self.status == ALERT_DELIVERY_OK


# A Gmail app password is 16 lowercase letters, displayed in four groups of
# four. Anchored with fullmatch at the call site.
APP_PASSWORD_PATTERN = re.compile(r"[a-z]{16}")


def normalise_app_password(raw: str | None) -> str:
    """Strip the display spaces from a Gmail app password.

    Gmail shows "abcd efgh ijkl mnop"; the credential is the 16 characters
    without spaces. Any whitespace is removed, not just the three gaps, so
    a trailing newline from a copy-paste is handled too.
    """
    if not raw:
        return ""
    return re.sub(r"\s+", "", raw)


def credential_problems(user: str | None, password: str | None) -> list[str]:
    """What is wrong with this credential pair, in operator terms.

    Returns an empty list when the pair is plausible. "Plausible" is not
    "accepted" — only the server can say that, which is why
    `send_alert_email` still reports `auth_failed` separately.
    """
    problems: list[str] = []
    if not (user or "").strip():
        problems.append(f"{ALERT_EMAIL_USER_ENV} is not set")
    elif "@" not in (user or ""):
        problems.append(
            f"{ALERT_EMAIL_USER_ENV}={user!r} is not an email address"
        )

    cleaned = normalise_app_password(password)
    if not cleaned:
        problems.append(f"{ALERT_EMAIL_PASSWORD_ENV} is not set")
    elif len(cleaned) != ALERT_EMAIL_PASSWORD_LENGTH:
        problems.append(
            f"{ALERT_EMAIL_PASSWORD_ENV} is {len(cleaned)} characters, not "
            f"{ALERT_EMAIL_PASSWORD_LENGTH} — Gmail refuses an ordinary "
            f"account password here; generate an APP password at "
            f"myaccount.google.com/apppasswords"
        )
    elif not APP_PASSWORD_PATTERN.fullmatch(cleaned):
        # Length alone is too weak a check, MEASURED: "my-real-password" is
        # exactly 16 characters and sailed through a length-only guard. A
        # Gmail app password is 16 LOWERCASE LETTERS, so punctuation,
        # digits or capitals mean an account password was pasted — which
        # Gmail rejects, producing an auth_failed the operator cannot
        # explain from the error text.
        problems.append(
            f"{ALERT_EMAIL_PASSWORD_ENV} is 16 characters but not 16 "
            f"lowercase letters — an app password looks like "
            f"'abcd efgh ijkl mnop'; this looks like an account password, "
            f"which Gmail refuses for SMTP"
        )
    return problems


def resolve_recipient(user: str | None, explicit: str | None = None) -> str:
    """Who the alert goes to.

    Defaults to the sending account: a one-operator system mailing itself
    is the common case, and requiring a second variable for it would make
    the channel harder to set up than it needs to be.
    """
    for candidate in (explicit, os.getenv(ALERT_EMAIL_TO_ENV), user):
        if (candidate or "").strip():
            return (candidate or "").strip()
    return ""


def build_message(
    sender: str, recipient: str, subject: str, body: str
) -> EmailMessage:
    """Render one alert as an email.

    A Date header is set explicitly: without it some clients stamp the
    message with receipt time, which would make a late alert from a
    catch-up run look current.
    """
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message["X-Alert-Version"] = ALERT_DELIVERY_VERSION
    message.set_content(body)
    return message


def send_alert_email(
    subject: str,
    body: str,
    *,
    recipient: str | None = None,
    user: str | None = None,
    password: str | None = None,
    host: str = ALERT_SMTP_HOST,
    port: int = ALERT_SMTP_PORT,
    timeout: float = ALERT_SMTP_TIMEOUT_SECONDS,
    smtp_factory=None,
) -> DeliveryResult:
    """Send one alert. Never raises; always returns a declared status.

    ``smtp_factory`` exists so the transport can be substituted in tests
    without reaching the network. Production passes nothing and gets
    `smtplib.SMTP`.
    """
    if not subject.strip():
        return DeliveryResult(
            status=ALERT_DELIVERY_FAILED,
            reason="an alert with no subject is unreadable in an inbox list",
        )

    user = user if user is not None else os.getenv(ALERT_EMAIL_USER_ENV)
    password = (
        password
        if password is not None
        else os.getenv(ALERT_EMAIL_PASSWORD_ENV)
    )

    problems = credential_problems(user, password)
    if problems:
        return DeliveryResult(
            status=ALERT_DELIVERY_UNCONFIGURED,
            reason="; ".join(problems),
            subject=subject,
            problems=tuple(problems),
        )

    sender = (user or "").strip()
    target = resolve_recipient(sender, recipient)
    if not target:
        return DeliveryResult(
            status=ALERT_DELIVERY_UNCONFIGURED,
            reason=(
                f"no recipient: set {ALERT_EMAIL_TO_ENV} or "
                f"{ALERT_EMAIL_USER_ENV}"
            ),
            subject=subject,
        )

    message = build_message(sender, target, subject, body)
    secret = normalise_app_password(password)
    factory = smtp_factory or smtplib.SMTP

    try:
        with factory(host, port, timeout=timeout) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(sender, secret)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        return DeliveryResult(
            status=ALERT_DELIVERY_AUTH_FAILED,
            reason=(
                f"the server REFUSED the credential ({exc.smtp_code}): "
                f"2-Step Verification must be on and the password must be a "
                f"16-character app password, not the account password"
            ),
            recipient=target,
            subject=subject,
        )
    except (OSError, smtplib.SMTPException) as exc:
        return DeliveryResult(
            status=ALERT_DELIVERY_FAILED,
            reason=f"{type(exc).__name__}: {exc}",
            recipient=target,
            subject=subject,
        )

    return DeliveryResult(
        status=ALERT_DELIVERY_OK, recipient=target, subject=subject
    )


def render_delivery(result: DeliveryResult) -> str:
    """One operator-readable line for a log or a console."""
    if result.ok:
        return f"[OK] alert delivered to {result.recipient}: {result.subject}"
    return f"[{result.status.upper()}] {result.reason}"
