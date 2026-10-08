"""Telegram delivery — a phone push that works on the operator's network.

**Why a second channel exists.** MEASURED 2026-10-04, outbound SMTP is
intercepted on the operator's network: all three SMTP ports accepted a TCP
connection and then sent no greeting, while HTTPS to Google returned 200.
No SMTP channel can deliver from that machine whatever the provider (open
item 11).

**AND TELEGRAM IS BLOCKED THERE TOO — measured after building this.** A
reachability probe looked fine (tcp 443 OPEN, the host answered), but the
probe was too weak: it only proved SOMETHING answered. Against the real
API:

    POST /bot<token>/getMe        HTTP 503, an HTML login page
    api.telegram.org cert issuer  palo-decrypt.scp.co.il
    gmail.googleapis.com issuer   Google Trust Services (WR2)

A certificate issued by an appliance rather than a public CA means TLS is
terminated and inspected locally — a Palo Alto device blocking the host by
category. Google is allowed, so this is messaging-app filtering, not a
general egress block (open item 24).

The consequence for the operator: this channel's CODE is sound and tested,
but it must be RUN from a network they control (home wifi, phone hotspot).
The token is untested rather than rejected, because the request never
reached Telegram — which is exactly why `auth_failed` and `failed` stay
distinct, and why `diagnose_interception` names the appliance instead of
reporting a credential problem.

**Why this answers "can we make a phone app?"** The scoring engine needs
pandas, numpy, sklearn and pyarrow against a 143MB local store, so it
cannot run ON a phone. Any app would be a thin client talking to a server
that still has to run somewhere — a hosting problem wearing an app
costume. What the operator actually wants is to *know when something
happens* while away from the laptop, and a push channel delivers exactly
that with no server, no data store and no app to install.

**What it does NOT give you.** A push cannot be queried. You receive what
the last scheduled run computed; you cannot pull a fresh score on demand.
If on-demand scoring from the phone is ever needed, that is a hosting
decision (open item 23), not a notification one.

**The status vocabulary is shared with the email channel on purpose.**
`delivered` / `unconfigured` / `auth_failed` / `failed` mean the same four
things in both, enforced by a config validator — a caller asking "did it
send?" must not have to learn two answers.

**Nothing raises**, for the reason the email channel does not: the daily
collector captures perishable data, so a notification outage must never
abort a collection run.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from core.config import (
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_FAILED,
    ALERT_DELIVERY_OK,
    ALERT_DELIVERY_UNCONFIGURED,
    TELEGRAM_API_BASE,
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_DELIVERY_STATUSES,
    TELEGRAM_DELIVERY_VERSION,
    TELEGRAM_MAX_MESSAGE_CHARS,
    TELEGRAM_TIMEOUT_SECONDS,
    TELEGRAM_TOKEN_MIN_LENGTH,
    TELEGRAM_TRUNCATION_NOTICE,
)

# "<digits>:<secret>" — BotFather's shape. Anchored at the call site.
TOKEN_PATTERN = re.compile(r"\d+:[A-Za-z0-9_-]{30,}")


@dataclass(frozen=True)
class TelegramResult:
    """The outcome of one delivery attempt.

    Deliberately the same shape and the same four statuses as
    `core.alert_delivery.DeliveryResult`.
    """

    status: str
    reason: str = ""
    chat_id: str = ""
    version: str = TELEGRAM_DELIVERY_VERSION
    truncated: bool = False
    problems: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in TELEGRAM_DELIVERY_STATUSES:
            raise ValueError(
                f"unknown delivery status {self.status!r}; expected one of "
                f"{TELEGRAM_DELIVERY_STATUSES}"
            )
        if self.status != ALERT_DELIVERY_OK and not self.reason:
            raise ValueError(
                f"status {self.status!r} is a failure and must carry a "
                f"reason; a failure without a reason cannot be acted on"
            )

    @property
    def ok(self) -> bool:
        """True ONLY when Telegram accepted the message.

        Not `status != failed`: an unconfigured channel has sent nothing,
        and calling that success is how a silent notification gap survives.
        """
        return self.status == ALERT_DELIVERY_OK


def credential_problems(token: str | None, chat_id: str | None) -> list[str]:
    """What is wrong with this credential pair, in operator terms."""
    problems: list[str] = []
    cleaned = str(token or "").strip()
    if not cleaned:
        problems.append(f"{TELEGRAM_BOT_TOKEN_ENV} is not set")
    elif len(cleaned) < TELEGRAM_TOKEN_MIN_LENGTH:
        problems.append(
            f"{TELEGRAM_BOT_TOKEN_ENV} is {len(cleaned)} characters, which is "
            f"too short for a bot token — @BotFather issues one shaped "
            f"'<digits>:<secret>'"
        )
    elif not TOKEN_PATTERN.fullmatch(cleaned):
        problems.append(
            f"{TELEGRAM_BOT_TOKEN_ENV} is not shaped like a bot token "
            f"('<digits>:<secret>') — check for a truncated paste"
        )

    target = str(chat_id or "").strip()
    if not target:
        problems.append(
            f"{TELEGRAM_CHAT_ID_ENV} is not set — run "
            f"scripts/verify_telegram.py --discover after messaging the bot"
        )
    elif not re.fullmatch(r"-?\d+", target):
        problems.append(
            f"{TELEGRAM_CHAT_ID_ENV}={target!r} is not numeric; a chat id is "
            f"an integer (negative for a group)"
        )
    return problems


def truncate(text: str, limit: int = TELEGRAM_MAX_MESSAGE_CHARS) -> tuple[str, bool]:
    """Fit a message inside Telegram's hard limit, visibly.

    Telegram REFUSES a message above the limit rather than trimming it, so
    a long alert would be dropped entirely. Truncating silently is nearly
    as bad: a reader who cannot see that the message was cut will act on a
    partial one. So the notice is part of the budget, not added after it.
    """
    if len(text) <= limit:
        return text, False
    keep = max(0, limit - len(TELEGRAM_TRUNCATION_NOTICE))
    return text[:keep] + TELEGRAM_TRUNCATION_NOTICE, True


def _api_url(token: str, method: str) -> str:
    return f"{TELEGRAM_API_BASE}/bot{token}/{method}"


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    """POST JSON and return (http_status, parsed_body).

    A Telegram error arrives as an HTTP error WITH a JSON body explaining
    it, so the body is parsed on both paths — discarding it would throw
    away the only description of what went wrong.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read() or b""
            return exc.code, json.loads(raw)
        except Exception:  # noqa: BLE001 - the body may not be JSON
            # A NON-JSON BODY MEANS SOMETHING OTHER THAN TELEGRAM ANSWERED.
            # MEASURED 2026-10-04 on the operator's network: a request to
            # api.telegram.org returned HTTP 503 carrying an HTML captive
            # portal login page, and the TLS certificate was issued by
            # `palo-decrypt.scp.co.il` rather than by Telegram's CA — a
            # Palo Alto decryption appliance blocking the host by category.
            # Reporting that as "the provider refused" would send the
            # operator to re-issue a token that was never delivered.
            body = raw[:400].decode("utf-8", "replace")
            if body.lstrip().lower().startswith(("<!doctype", "<html")):
                return exc.code, {
                    "ok": False,
                    "description": (
                        f"HTTP {exc.code} with an HTML body, not a Telegram "
                        f"response — the request was INTERCEPTED by a "
                        f"network appliance and never reached Telegram"
                    ),
                    "intercepted": True,
                }
            return exc.code, {}


def send_telegram_alert(
    text: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    timeout: float = TELEGRAM_TIMEOUT_SECONDS,
    poster=None,
) -> TelegramResult:
    """Send one alert to the operator's phone. Never raises.

    ``poster`` substitutes the transport in tests so nothing reaches the
    network.
    """
    if not str(text or "").strip():
        return TelegramResult(
            status=ALERT_DELIVERY_FAILED,
            reason="an empty alert says nothing and trains the reader to ignore the channel",
        )

    token = token if token is not None else os.getenv(TELEGRAM_BOT_TOKEN_ENV)
    chat_id = chat_id if chat_id is not None else os.getenv(TELEGRAM_CHAT_ID_ENV)

    problems = credential_problems(token, chat_id)
    if problems:
        return TelegramResult(
            status=ALERT_DELIVERY_UNCONFIGURED,
            reason="; ".join(problems),
            problems=tuple(problems),
        )

    token = str(token).strip()
    target = str(chat_id).strip()
    body, was_truncated = truncate(text)
    post = poster or _post

    try:
        status_code, payload = post(
            _api_url(token, "sendMessage"),
            {"chat_id": target, "text": body, "disable_web_page_preview": True},
            timeout,
        )
    except Exception as exc:  # noqa: BLE001 - delivery must never raise
        return TelegramResult(
            status=ALERT_DELIVERY_FAILED,
            reason=f"{type(exc).__name__}: {exc}",
            chat_id=target,
            truncated=was_truncated,
        )

    if payload.get("ok") is True:
        return TelegramResult(
            status=ALERT_DELIVERY_OK, chat_id=target, truncated=was_truncated
        )

    description = str(payload.get("description") or f"HTTP {status_code}")

    # 401 is a rejected token; 400 on the chat id is a rejected RECIPIENT.
    # Both are credential problems the operator fixes by changing something,
    # which is what separates them from a transport failure they wait out.
    if status_code in (401, 403) or "unauthorized" in description.lower():
        return TelegramResult(
            status=ALERT_DELIVERY_AUTH_FAILED,
            reason=(
                f"Telegram REFUSED the credential: {description}. Re-issue "
                f"the token with @BotFather, or confirm you have sent the "
                f"bot at least one message (a bot cannot open a chat)"
            ),
            chat_id=target,
            truncated=was_truncated,
        )
    if status_code == 400 and "chat not found" in description.lower():
        return TelegramResult(
            status=ALERT_DELIVERY_AUTH_FAILED,
            reason=(
                f"chat {target} not found: send your bot a message first — a "
                f"bot cannot start a conversation, only reply to one"
            ),
            chat_id=target,
            truncated=was_truncated,
        )

    return TelegramResult(
        status=ALERT_DELIVERY_FAILED,
        reason=description,
        chat_id=target,
        truncated=was_truncated,
    )


def discover_chat_id(
    token: str | None = None,
    timeout: float = TELEGRAM_TIMEOUT_SECONDS,
    poster=None,
) -> tuple[str, str]:
    """Find the chat id from the bot's pending updates.

    Returns ``(chat_id, explanation)``; the id is empty when none could be
    found. A bot cannot open a conversation, so the operator must message
    it once first — and that message is exactly what carries the id.
    """
    token = token if token is not None else os.getenv(TELEGRAM_BOT_TOKEN_ENV)
    if not str(token or "").strip():
        return "", f"{TELEGRAM_BOT_TOKEN_ENV} is not set"

    post = poster or _post
    try:
        _status, payload = post(_api_url(str(token).strip(), "getUpdates"), {}, timeout)
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"

    if payload.get("ok") is not True:
        return "", str(payload.get("description") or "the provider refused getUpdates")

    for update in reversed(payload.get("result") or []):
        chat = ((update.get("message") or {}).get("chat")) or {}
        if chat.get("id") is not None:
            name = chat.get("username") or chat.get("first_name") or "this chat"
            return str(chat["id"]), f"found {name}"

    return "", (
        "no messages yet — open your bot in Telegram and send it any "
        "message, then run this again"
    )


def diagnose_interception(host: str = "api.telegram.org", timeout: float = 12.0) -> list[str]:
    """Is something between us and Telegram? Name it if so.

    MEASURED 2026-10-04 on the operator's network, this is the difference
    between "retry later" and "this channel will never work here":

        api.telegram.org  cert issuer  palo-decrypt.scp.co.il
        gmail.googleapis.com  issuer   Google Trust Services (WR2)

    A certificate issued by an appliance rather than by a public CA means
    TLS is being terminated and inspected locally. That is how a corporate
    firewall blocks a host by category, and no credential can fix it.
    """
    import socket
    import ssl

    lines: list[str] = []
    try:
        with socket.create_connection((host, 443), timeout=timeout) as raw:
            context = ssl.create_default_context()
            with context.wrap_socket(raw, server_hostname=host) as secure:
                certificate = secure.getpeercert() or {}
        issuer = dict(
            item[0] for item in certificate.get("issuer", []) if item
        )
        name = issuer.get("commonName") or issuer.get("organizationName") or "?"
        lines.append(f"- {host} TLS certificate issued by: {name}")
        if any(
            marker in str(name).lower()
            for marker in ("decrypt", "palo", "proxy", "firewall", "zscaler", "fortinet")
        ):
            lines.append("")
            lines.append("  DIAGNOSIS: TLS is being INTERCEPTED by a network")
            lines.append("  appliance, which is how a corporate firewall blocks a")
            lines.append("  host by category. The request never reached Telegram,")
            lines.append("  so the token is untested rather than rejected.")
            lines.append("")
            lines.append("  No credential change will fix this. Try the same")
            lines.append("  command from a network you control (home wifi, or a")
            lines.append("  phone hotspot) — the channel itself is sound.")
        else:
            lines.append(
                "- the certificate looks genuine, so this was most likely a "
                "transient failure; retry before changing anything"
            )
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        lines.append(f"- could not inspect the certificate ({type(exc).__name__}: {exc})")
    return lines


def render_result(result: TelegramResult) -> str:
    """One operator-readable line."""
    if result.ok:
        suffix = " (truncated)" if result.truncated else ""
        return f"[OK] telegram alert delivered to {result.chat_id}{suffix}"
    return f"[{result.status.upper()}] {result.reason}"
