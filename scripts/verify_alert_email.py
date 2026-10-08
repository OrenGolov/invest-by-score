"""Is the alert email channel set up, and does it actually deliver?

Run this after setting ALERT_EMAIL_USER and ALERT_EMAIL_PASSWORD. It
answers four questions a failed 22:00 alert cannot separate on its own:

    1. are the variables set in THIS process?
    2. are they SHAPED like a Gmail account and a 16-character app password?
    3. does Gmail ACCEPT the credential?
    4. does a real message arrive in the inbox?

Only question 4 proves the channel. A credential that is set but wrong
looks identical to one that works until the first real alert fires
unattended, which is exactly the failure this script exists to prevent.

NAMED `verify_*`, NOT `check_*`: CI auto-discovers every
`scripts/check_*.py` as a governance gate, and this one exits 1 whenever
the credential is absent. As a gate it would fail every clone that has not
been set up, which is a setup state rather than a code defect. It also
sends real mail, which a gate must never do.

    python scripts/verify_alert_email.py
    python scripts/verify_alert_email.py --to someone@else.com
    python scripts/verify_alert_email.py --dry-run   # stop before sending
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.alert_delivery import (  # noqa: E402
    credential_problems,
    normalise_app_password,
    resolve_recipient,
    send_alert_email,
)
from core.config import (  # noqa: E402
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_UNCONFIGURED,
    ALERT_DELIVERY_VERSION,
    ALERT_EMAIL_PASSWORD_ENV,
    ALERT_EMAIL_USER_ENV,
    ALERT_SMTP_HOST,
    ALERT_SMTP_PORT,
)

SETUP_HELP = f"""
      Set them for your account (permanent), in PowerShell:

        [Environment]::SetEnvironmentVariable(
            "{ALERT_EMAIL_USER_ENV}","you@gmail.com","User")
        [Environment]::SetEnvironmentVariable(
            "{ALERT_EMAIL_PASSWORD_ENV}","<16-char app password>","User")

      Then open a NEW terminal — a variable set this way is not visible to
      the terminal that set it.

      The app password comes from myaccount.google.com/apppasswords, which
      is only visible once 2-Step Verification is on. It is NOT your
      account password; Gmail refuses that for SMTP.
"""


def diagnose_transport(timeout: float = 12.0) -> list[str]:
    """Why did the send fail? Separate a firewall from an interceptor.

    MEASURED 2026-10-04 on the operator machine, this distinction is the
    whole point: all three SMTP ports accepted a TCP connection and then
    sent no greeting, while HTTPS to Google returned 200. A refused
    connection is a firewall; a completed connection with no banner is
    outbound SMTP interception, and only the second one means "no SMTP
    channel will ever work here".
    """
    import socket
    import urllib.request

    lines: list[str] = []

    try:
        sock = socket.create_connection(
            (ALERT_SMTP_HOST, ALERT_SMTP_PORT), timeout=timeout
        )
    except OSError as exc:
        lines.append(f"- tcp {ALERT_SMTP_HOST}:{ALERT_SMTP_PORT} REFUSED ({exc})")
        lines.append("- a refused connection is a firewall rule; ask whoever")
        lines.append("  runs the network to permit outbound 587, or use a")
        lines.append("  channel that rides port 443 instead.")
        return lines

    lines.append(f"- tcp {ALERT_SMTP_HOST}:{ALERT_SMTP_PORT} connected")
    sock.settimeout(timeout)
    try:
        banner = sock.recv(200)
    except OSError:
        banner = b""
    finally:
        sock.close()

    if banner.startswith(b"220"):
        lines.append(f"- greeting received: {banner[:60]!r}")
        lines.append("- SMTP is reachable, so this was a transient failure;")
        lines.append("  retry before changing anything.")
        return lines

    lines.append(
        f"- NO SMTP greeting (got {banner!r}) though the handshake succeeded"
    )

    try:
        status = urllib.request.urlopen(
            "https://gmail.googleapis.com/$discovery/rest?version=v1",
            timeout=timeout,
        ).status
        lines.append(f"- but HTTPS to Google returned {status}: port 443 is fine")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        lines.append(f"- HTTPS to Google also failed ({type(exc).__name__})")

    lines.append("")
    lines.append("  DIAGNOSIS: outbound SMTP is being INTERCEPTED, not blocked.")
    lines.append("  A connection that completes and then goes silent is the")
    lines.append("  signature of a network that terminates SMTP sessions —")
    lines.append("  routine on a corporate/managed network.")
    lines.append("")
    lines.append("  No SMTP channel will work here, whatever the provider.")
    lines.append("  A working channel must ride port 443: Telegram's bot API")
    lines.append("  (no OAuth, real phone push) or the Gmail REST API.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--to", default=None, help="recipient; defaults to the sending account"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check the credential shape and stop before sending",
    )
    args = parser.parse_args()

    print(f"alert email channel check  [{ALERT_DELIVERY_VERSION}]")
    print(f"  endpoint : {ALERT_SMTP_HOST}:{ALERT_SMTP_PORT} (STARTTLS)")
    print()

    # -- 1. are they set here? ----------------------------------------
    user = os.getenv(ALERT_EMAIL_USER_ENV)
    password = os.getenv(ALERT_EMAIL_PASSWORD_ENV)

    if not user and not password:
        print(f"  [X] neither {ALERT_EMAIL_USER_ENV} nor "
              f"{ALERT_EMAIL_PASSWORD_ENV} is set in this process.")
        print(SETUP_HELP)
        return 1

    # -- 2. are they shaped right? ------------------------------------
    problems = credential_problems(user, password)
    if problems:
        print("  [X] the credential cannot be used as it stands:")
        for problem in problems:
            print(f"      - {problem}")
        print(SETUP_HELP)
        return 1

    cleaned = normalise_app_password(password)
    print(f"  [OK] {ALERT_EMAIL_USER_ENV} = {user}")
    print(f"  [OK] {ALERT_EMAIL_PASSWORD_ENV} = "
          f"{cleaned[:2]}...{cleaned[-2:]} ({len(cleaned)} chars, app-password shape)")

    recipient = resolve_recipient(user, args.to)
    print(f"  [OK] recipient = {recipient}")

    if args.dry_run:
        print()
        print("  --dry-run: stopping before the send. The credential is")
        print("  SHAPED correctly, which is not the same as ACCEPTED — only")
        print("  a real send proves the channel.")
        return 0

    # -- 3 + 4. does it actually deliver? -----------------------------
    print()
    print("  sending a real test message...")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    result = send_alert_email(
        subject=f"invest-by-score: alert channel verified {stamp}",
        body=(
            "This is the setup verification message from\n"
            "scripts/verify_alert_email.py.\n\n"
            f"Sent  : {stamp}\n"
            f"From  : {user}\n"
            f"To    : {recipient}\n"
            f"Via   : {ALERT_SMTP_HOST}:{ALERT_SMTP_PORT} (STARTTLS)\n"
            f"Module: {ALERT_DELIVERY_VERSION}\n\n"
            "If you are reading this in your inbox, the alert channel can\n"
            "deliver. Note what this does NOT prove: no alert evaluator is\n"
            "wired to this transport yet, so nothing will arrive on its own\n"
            "until one is.\n"
        ),
        recipient=recipient,
    )

    if result.status == ALERT_DELIVERY_UNCONFIGURED:
        print(f"  [X] unconfigured: {result.reason}")
        print(SETUP_HELP)
        return 1

    if result.status == ALERT_DELIVERY_AUTH_FAILED:
        print("  [X] Gmail REFUSED the credential.")
        print(f"      {result.reason}")
        print()
        print("      Common causes:")
        print("        - the app password was revoked or regenerated")
        print("        - 2-Step Verification was turned off")
        print("        - the account password was used instead of an app one")
        return 1

    if not result.ok:
        print(f"  [X] the send failed: {result.reason}")
        print()
        print("      This is a transport failure, not a credential one: the")
        print("      credential was never reached, so this says NOTHING")
        print("      about whether it is valid.")
        print()
        print("      Probing why...")
        for line in diagnose_transport():
            print(f"      {line}")
        return 1

    print(f"  [OK] Gmail accepted the credential")
    print(f"  [OK] message accepted for delivery to {result.recipient}")
    print()
    print(f"  Check that inbox for the subject line ending {stamp}.")
    print("  An accepted message is not a received one: Gmail can still")
    print("  file it as spam, and only the inbox settles that.")
    print()
    print("  WHAT THIS DOES NOT PROVE. The channel works; nothing uses it")
    print("  yet. MEASURED 2026-10-04, all seven alert evaluators are")
    print("  imported only by their own CI gates, so no alert reaches this")
    print("  transport on the live path. Wiring one is a separate task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
