"""A8 email rendering and sending — the operator's template, mobile first.

The operator specified the format precisely:

    Subject:  <Priority>: <Ticker> - <Short, Direct Description>
    Body:     Action, Relevance Date, then Who / What / When / Why it Matters

This module renders that and sends it. Three decisions in here are measurements
rather than preferences.

**THE CREDENTIAL IS NEVER IN THE REPOSITORY.** Host and port are configuration;
the password is read from the environment at send time. A password committed once
is in the git history permanently, and the operator's keys have already been
handled that way elsewhere in this project.

**A MISSING CREDENTIAL DEGRADES EMAIL AND NOTHING ELSE.** ``ALERT_EMAIL_REQUIRED``
is False and a config validator enforces it. The dashboard receives every alert
regardless, so an unset password costs the email channel and never the run — the
same fail-soft-per-source reasoning ``daily_collect`` uses for providers.

**WHY THE PLAIN-TEXT PART IS NOT OPTIONAL.** A multipart/alternative message with
only HTML renders as an empty body in any client with images or HTML disabled, and
as raw markup in a few mobile notification previews. Both parts carry the same
facts; the text part is what a phone's lock-screen preview actually shows.

**WHY `Why it Matters` CARRIES THE GRADER'S REASON VERBATIM.** Every A1-A6 module
returns a measured ``reason`` and the grader returns a ``priority_reason``.
Paraphrasing them here would create a second, unmeasured explanation of the same
finding, and the two would drift. The email quotes them.
"""

from __future__ import annotations

import html
import logging
import os
import smtplib
from collections.abc import Mapping, Sequence
from email.message import EmailMessage
from email.utils import formatdate
from typing import Any

from core.config import (
    ALERT_EMAIL_DIGEST_PRIORITIES,
    ALERT_EMAIL_HOST,
    ALERT_EMAIL_IMMEDIATE_PRIORITIES,
    ALERT_EMAIL_MAX_WIDTH_PX,
    ALERT_EMAIL_PASSWORD_ENV,
    ALERT_EMAIL_PORT,
    ALERT_EMAIL_SUBJECT_TEMPLATE,
    ALERT_EMAIL_TO_ENV,
    ALERT_EMAIL_USE_STARTTLS,
    ALERT_EMAIL_USER_ENV,
    ALERT_PRIORITIES,
    ALERT_PRIORITY_COLOURS,
)

LOGGER = logging.getLogger("alert_email")

# The marker the operator specified for a time-sensitive finding, and the one for
# a future checkpoint.
MARK_IMMEDIATE = "\U0001F534 IMPORTANT - Immediate Review Required"
MARK_MONITORING = "\U0001F7E2 Active Monitoring Date"

# Hex for each band, matching the dashboard's colour names so an email and the
# Monitoring tab never disagree about what Urgent looks like.
_PRIORITY_HEX: dict[str, str] = {
    "Urgent": "#d93025",
    "Very High": "#e8710a",
    "High": "#f9ab00",
    "Medium": "#1a73e8",
    "Low": "#9aa0a6",
}

# Priorities the operator wants in the inbox the moment they are found.
IMMEDIATE: frozenset[str] = frozenset(ALERT_EMAIL_IMMEDIATE_PRIORITIES)
DIGESTED: frozenset[str] = frozenset(ALERT_EMAIL_DIGEST_PRIORITIES)


class AlertEmailError(ValueError):
    """Raised when a message cannot be built or addressed honestly."""


def credentials() -> tuple[str | None, str | None, str | None, str]:
    """``(user, password, recipient, reason)`` read from the environment.

    Returns None values with a stated reason rather than raising, because a
    missing credential must degrade the email channel and not the run.
    """
    user = os.environ.get(ALERT_EMAIL_USER_ENV) or None
    password = os.environ.get(ALERT_EMAIL_PASSWORD_ENV) or None
    # The recipient defaults to the sender: a one-reader system mailing itself is
    # the ordinary case, and requiring a third variable adds a way to misconfigure
    # it silently.
    recipient = os.environ.get(ALERT_EMAIL_TO_ENV) or user

    missing = [
        name
        for name, value in (
            (ALERT_EMAIL_USER_ENV, user),
            (ALERT_EMAIL_PASSWORD_ENV, password),
        )
        if not value
    ]
    if missing:
        return None, None, None, (
            f"{', '.join(missing)} is not set, so email is disabled; every "
            f"alert is still recorded and visible in the Monitoring tab"
        )
    return user, password, recipient, "credentials present"


def routing(priority: str | None) -> tuple[str, str]:
    """``(route, reason)`` for one priority: immediate, digest, or neither.

    An UNGRADED alert goes in the digest rather than the inbox. It means a
    detector could not reach a verdict, which the operator must see but which is
    not an event to interrupt them for -- and it must not be dropped, because a
    blind detector is exactly what silence would hide.
    """
    if priority is None:
        return "digest", (
            "the alert carries no priority because its detector reached no "
            "verdict; it is reported in the digest rather than dropped, since "
            "silence would hide a blind detector"
        )
    if priority not in ALERT_PRIORITIES:
        raise AlertEmailError(
            f"unknown priority {priority!r}; routing it would send an alert to "
            f"a channel nobody chose"
        )
    if priority in IMMEDIATE:
        return "immediate", f"{priority} is an immediate-email band"
    if priority in DIGESTED:
        return "digest", f"{priority} collects into the daily digest"
    raise AlertEmailError(
        f"{priority!r} is a known band with no email route, so an alert in it "
        f"would be email-silent; every band must be routed"
    )


def subject_for(record: Mapping[str, Any]) -> str:
    """``<Priority>: <Ticker> - <Short, Direct Description>``."""
    priority = record.get("priority") or "Unprioritised"
    ticker = record.get("ticker") or "PORTFOLIO"
    title = record.get("title") or record.get("state") or "Alert"
    return ALERT_EMAIL_SUBJECT_TEMPLATE.format(
        priority=priority, ticker=ticker, title=title
    )


def _summary_lines(record: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """The Who / What / When / Why it Matters block, as (heading, bullets).

    Built from the stored record. A supplied ``summary`` wins for any section it
    provides; everything else is derived from the detector's own fields, never
    invented.
    """
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    ticker = record.get("ticker") or "the portfolio"
    detector = str(record.get("alert") or "a detector").replace("_", " ")
    state = record.get("state") or "an unnamed state"

    who = summary.get("who") or f"{ticker}"
    what = summary.get("what") or (
        f"The {detector} detector reported {state}."
    )
    when = summary.get("when") or (
        f"Observed for {record.get('as_of') or 'an unstated date'}; "
        f"detected {record.get('detected_at') or 'at an unstated time'}."
    )

    why = summary.get("why")
    if isinstance(why, str):
        why_bullets = [why]
    elif isinstance(why, Sequence):
        why_bullets = [str(item) for item in why]
    else:
        # The MEASURED reasons, quoted rather than paraphrased. Two separate
        # facts: what the detector found, and why that earned this band.
        why_bullets = []
        if record.get("reason"):
            why_bullets.append(str(record["reason"]))
        if record.get("priority_reason"):
            why_bullets.append(f"Priority: {record['priority_reason']}")
        if record.get("action_reason"):
            why_bullets.append(f"Action: {record['action_reason']}")
        if not why_bullets:
            why_bullets = ["No reason was recorded with this alert."]

    return [
        ("Who", [str(who)]),
        ("What", [str(what)]),
        ("When", [str(when)]),
        ("Why it Matters", why_bullets),
    ]


def _relevance(record: Mapping[str, Any]) -> list[str]:
    """The Relevance Date block: when this matters, and what to watch next."""
    lines: list[str] = []
    priority = record.get("priority")
    if priority in IMMEDIATE:
        lines.append(MARK_IMMEDIATE)
    relevant_from = record.get("as_of") or record.get("detected_at")
    if relevant_from:
        lines.append(f"Relevant from: {relevant_from}")
    monitor = record.get("monitor_date")
    if monitor:
        lines.append(f"{MARK_MONITORING}: {monitor}")
    horizon = record.get("horizon")
    if horizon:
        lines.append(f"Evaluation horizon: {horizon}")
    return lines


def render_text(record: Mapping[str, Any]) -> str:
    """The plain-text part. This is what a phone's preview actually shows."""
    out: list[str] = [subject_for(record), ""]

    out.append(f"Action: {record.get('action') or 'Review'}")
    out.append("")

    relevance = _relevance(record)
    if relevance:
        out.extend(relevance)
        out.append("")

    for heading, bullets in _summary_lines(record):
        out.append(f"{heading}:")
        for bullet in bullets:
            out.append(f"  - {bullet}")
        out.append("")

    numbers = _numbers(record)
    if numbers:
        out.append("Measurements:")
        for label, value in numbers:
            out.append(f"  {label}: {value}")
        out.append("")

    out.append(
        "This is an analysis alert. It is not financial advice and not a trade "
        "instruction."
    )
    return "\n".join(out).rstrip() + "\n"


def _numbers(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The numeric evidence, when the detector measured any.

    A value is shown IFF it was measured. The project's shape rule: a key exists
    only when there was a measurement, so `None` is printed as an em dash rather
    than coalesced to 0.0 -- that coercion is what A1 measured turning a vanished
    forecast into a -0.56 crash.
    """
    detail = record.get("detail") if isinstance(record.get("detail"), Mapping) else {}
    rows: list[tuple[str, str]] = []
    for key, label in (
        ("previous", "Previous"),
        ("current", "Current"),
        ("delta", "Change"),
        ("threshold", "Threshold"),
        ("confirming_sessions", "Confirming sessions"),
        ("median_abs_move", "Median absolute move"),
        ("analogs", "Comparable events"),
    ):
        if key not in detail:
            continue
        value = detail[key]
        if value is None:
            rows.append((label, "—  (not measured)"))
        elif isinstance(value, bool):
            # Checked before the numeric branch: bool IS an int in Python, so a
            # True would otherwise render as "1.0000".
            rows.append((label, "yes" if value else "no"))
        elif isinstance(value, int):
            # A COUNT IS NOT A MEASUREMENT TO FOUR DECIMALS. Caught in the
            # rendered sample: 94 comparable events printed as "94.0000", which
            # reads as a precision the number does not have.
            rows.append((label, f"{value:,}"))
        elif isinstance(value, float):
            rows.append((label, f"{value:.4f}"))
        else:
            rows.append((label, str(value)))
    return rows


def render_html(record: Mapping[str, Any]) -> str:
    """The HTML part, single column and capped for a phone screen."""
    priority = record.get("priority")
    colour = _PRIORITY_HEX.get(str(priority), "#5f6368")
    esc = html.escape

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "</head>",
        '<body style="margin:0;padding:16px;background:#f1f3f4;'
        'font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
        'color:#202124;">',
        f'<div style="max-width:{ALERT_EMAIL_MAX_WIDTH_PX}px;margin:0 auto;'
        'background:#ffffff;border-radius:12px;overflow:hidden;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.12);">',
        f'<div style="background:{colour};color:#ffffff;padding:14px 18px;'
        'font-size:15px;font-weight:bold;">'
        f'{esc(str(priority or "Unprioritised"))} &middot; '
        f'{esc(str(record.get("ticker") or "PORTFOLIO"))}</div>',
        '<div style="padding:18px;">',
        f'<h1 style="margin:0 0 14px;font-size:19px;line-height:1.3;">'
        f'{esc(str(record.get("title") or record.get("state") or "Alert"))}</h1>',
        '<p style="margin:0 0 16px;font-size:16px;">'
        '<strong>Action:</strong> '
        f'{esc(str(record.get("action") or "Review"))}</p>',
    ]

    relevance = _relevance(record)
    if relevance:
        parts.append(
            '<div style="background:#f8f9fa;border-left:4px solid '
            f'{colour};padding:10px 14px;margin:0 0 16px;font-size:14px;">'
        )
        parts.append(
            "<br>".join(esc(line) for line in relevance)
        )
        parts.append("</div>")

    for heading, bullets in _summary_lines(record):
        parts.append(
            '<p style="margin:0 0 4px;font-size:13px;font-weight:bold;'
            'text-transform:uppercase;color:#5f6368;letter-spacing:.04em;">'
            f"{esc(heading)}</p>"
        )
        if len(bullets) == 1:
            parts.append(
                '<p style="margin:0 0 14px;font-size:15px;line-height:1.5;">'
                f"{esc(bullets[0])}</p>"
            )
        else:
            parts.append(
                '<ul style="margin:0 0 14px;padding-left:20px;font-size:15px;'
                'line-height:1.5;">'
            )
            parts.extend(f"<li>{esc(b)}</li>" for b in bullets)
            parts.append("</ul>")

    numbers = _numbers(record)
    if numbers:
        parts.append(
            '<table role="presentation" style="width:100%;border-collapse:'
            'collapse;margin:0 0 14px;font-size:14px;">'
        )
        for label, value in numbers:
            parts.append(
                '<tr><td style="padding:6px 0;color:#5f6368;'
                f'border-bottom:1px solid #e8eaed;">{esc(label)}</td>'
                '<td style="padding:6px 0;text-align:right;font-weight:bold;'
                f'border-bottom:1px solid #e8eaed;">{esc(value)}</td></tr>'
            )
        parts.append("</table>")

    parts.extend(
        [
            '<p style="margin:14px 0 0;font-size:12px;color:#5f6368;'
            'line-height:1.5;">This is an analysis alert. It is not financial '
            "advice and not a trade instruction.</p>",
            "</div></div></body></html>",
        ]
    )
    return "".join(parts)


def build_message(
    record: Mapping[str, Any],
    *,
    sender: str,
    recipient: str,
) -> EmailMessage:
    """One addressed, multipart message for a single alert."""
    if not sender or not recipient:
        raise AlertEmailError(
            "a message needs both a sender and a recipient; sending to an "
            "empty address fails silently at some providers"
        )
    message = EmailMessage()
    message["Subject"] = subject_for(record)
    message["From"] = sender
    message["To"] = recipient
    message["Date"] = formatdate(localtime=True)
    identifier = record.get("alert_id")
    if identifier:
        # A stable header, so a mail client that sees the same finding twice can
        # thread them rather than showing two unrelated messages.
        message["X-Alert-Id"] = str(identifier)
    message.set_content(render_text(record))
    message.add_alternative(render_html(record), subtype="html")
    return message


def build_digest(
    records: Sequence[Mapping[str, Any]],
    *,
    sender: str,
    recipient: str,
    as_of: str,
) -> EmailMessage:
    """One message carrying every digested alert, most urgent first."""
    if not records:
        raise AlertEmailError("a digest needs at least one alert")

    ordered = sorted(
        records,
        key=lambda r: (
            -(r.get("priority_rank") or 0),
            str(r.get("detected_at") or ""),
        ),
    )
    bands: dict[str, int] = {}
    for row in ordered:
        key = str(row.get("priority") or "Unprioritised")
        bands[key] = bands.get(key, 0) + 1
    tally = ", ".join(f"{count} {band}" for band, count in bands.items())

    message = EmailMessage()
    message["Subject"] = f"Daily alert digest {as_of}: {len(ordered)} alerts ({tally})"
    message["From"] = sender
    message["To"] = recipient
    message["Date"] = formatdate(localtime=True)

    text = [message["Subject"], "", ""]
    for row in ordered:
        text.append(f"--- {subject_for(row)}")
        text.append(f"    Action: {row.get('action') or 'Review'}")
        if row.get("reason"):
            text.append(f"    {row['reason']}")
        text.append("")
    text.append(
        "These are analysis alerts. They are not financial advice and not trade "
        "instructions."
    )
    message.set_content("\n".join(text))

    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "</head>",
        '<body style="margin:0;padding:16px;background:#f1f3f4;'
        'font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
        'color:#202124;">',
        f'<div style="max-width:{ALERT_EMAIL_MAX_WIDTH_PX}px;margin:0 auto;">',
        f'<h1 style="font-size:18px;margin:0 0 4px;">Daily alert digest</h1>',
        f'<p style="font-size:13px;color:#5f6368;margin:0 0 16px;">'
        f"{esc(as_of)} &middot; {len(ordered)} alerts &middot; "
        f"{esc(tally)}</p>",
    ]
    for row in ordered:
        colour = _PRIORITY_HEX.get(str(row.get("priority")), "#5f6368")
        parts.append(
            '<div style="background:#ffffff;border-radius:10px;padding:14px;'
            f'margin:0 0 10px;border-left:4px solid {colour};">'
        )
        parts.append(
            f'<div style="font-size:12px;font-weight:bold;color:{colour};">'
            f'{esc(str(row.get("priority") or "Unprioritised"))} &middot; '
            f'{esc(str(row.get("ticker") or "PORTFOLIO"))}</div>'
        )
        parts.append(
            '<div style="font-size:15px;font-weight:bold;margin:4px 0;">'
            f'{esc(str(row.get("title") or row.get("state") or "Alert"))}</div>'
        )
        parts.append(
            '<div style="font-size:13px;margin:0 0 6px;">'
            f'<strong>Action:</strong> {esc(str(row.get("action") or "Review"))}'
            "</div>"
        )
        if row.get("reason"):
            parts.append(
                '<div style="font-size:13px;color:#3c4043;line-height:1.5;">'
                f'{esc(str(row["reason"]))}</div>'
            )
        parts.append("</div>")
    parts.extend(
        [
            '<p style="font-size:12px;color:#5f6368;line-height:1.5;">These are '
            "analysis alerts. They are not financial advice and not trade "
            "instructions.</p>",
            "</div></body></html>",
        ]
    )
    message.add_alternative("".join(parts), subtype="html")
    return message


def send(message: EmailMessage, *, dry_run: bool = False) -> tuple[bool, str]:
    """Send one message. Returns ``(sent, detail)`` and never raises.

    A send failure is reported, not thrown: the finding is already stored and
    visible, and losing the run over a transient SMTP error would turn one
    outage into a lost day.
    """
    user, password, _, reason = credentials()
    if not user or not password:
        return False, reason
    if dry_run:
        return False, "dry run: the message was rendered but not sent"

    try:
        with smtplib.SMTP(ALERT_EMAIL_HOST, ALERT_EMAIL_PORT, timeout=30) as smtp:
            smtp.ehlo()
            if ALERT_EMAIL_USE_STARTTLS:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(user, password)
            smtp.send_message(message)
        return True, f"sent to {message['To']}"
    except smtplib.SMTPAuthenticationError as exc:
        # NAMED SEPARATELY because the fix is specific and the generic message
        # would send the operator looking at the wrong thing: Gmail rejects an
        # account password here and requires an App Password.
        return False, (
            f"authentication refused ({exc.smtp_code}); Gmail requires an App "
            f"Password rather than the account password"
        )
    except (smtplib.SMTPException, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def colour_of(priority: str | None) -> str:
    """The dashboard colour name for a band, for parity between the surfaces."""
    if priority is None:
        return "grey"
    return ALERT_PRIORITY_COLOURS.get(priority, "grey")
