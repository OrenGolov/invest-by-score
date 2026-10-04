"""Is the Telegram alert channel set up, and does it reach your phone?

**BOTH CHANNELS ARE BLOCKED ON THE AMAN NETWORK — MEASURED 2026-10-04.**
SMTP is intercepted (every port accepts a connection, then goes silent),
and api.telegram.org is intercepted too: the TLS certificate is issued by
`palo-decrypt.scp.co.il`, a Palo Alto decryption appliance, and requests
return HTTP 503 carrying a captive-portal login page. Google hosts are
allowed (genuine Google Trust Services certificate), so this is
category-based blocking of messaging apps rather than a general egress
block. See open items 11 and 24.

The channel code is sound and fully tested; it has to be RUN from a
network the operator controls — home wifi, or a phone hotspot.

SETUP, about two minutes, on the phone:

    1. open Telegram, search @BotFather
    2. send /newbot, pick a display name and a username ending in 'bot'
    3. BotFather replies with a token like 8012345678:AAH...
    4. open your new bot and SEND IT ANY MESSAGE
       (a bot cannot start a conversation, only reply to one)

Then:

    [Environment]::SetEnvironmentVariable(
        "ALERT_TELEGRAM_TOKEN","<token>","User")

    python scripts/verify_telegram.py --discover   # finds the chat id
    [Environment]::SetEnvironmentVariable(
        "ALERT_TELEGRAM_CHAT_ID","<id>","User")

    python scripts/verify_telegram.py              # sends a real message

NAMED `verify_*`, NOT `check_*`: CI auto-discovers every
`scripts/check_*.py` as a governance gate, and this one exits 1 whenever
the credential is absent — a setup state rather than a code defect. It also
sends real messages, which a gate must never do.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ALERT_DELIVERY_AUTH_FAILED,
    ALERT_DELIVERY_UNCONFIGURED,
    TELEGRAM_API_BASE,
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_DELIVERY_VERSION,
)
from core.telegram_delivery import (  # noqa: E402
    credential_problems,
    diagnose_interception,
    discover_chat_id,
    send_telegram_alert,
)

SETUP_HELP = f"""
      On your phone, about two minutes:

        1. open Telegram, search @BotFather
        2. send /newbot, pick a name and a username ending in 'bot'
        3. copy the token it replies with
        4. open your new bot and SEND IT ANY MESSAGE

      Step 4 is not optional: a bot cannot open a conversation, only
      reply to one, and that first message is what carries your chat id.

      Then, in PowerShell:

        [Environment]::SetEnvironmentVariable(
            "{TELEGRAM_BOT_TOKEN_ENV}","<token>","User")

      Open a NEW terminal, then:

        python scripts/verify_telegram.py --discover
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="find the chat id from the bot's pending messages and stop",
    )
    parser.add_argument(
        "--message", default=None, help="send this text instead of the default"
    )
    args = parser.parse_args()

    print(f"telegram alert channel check  [{TELEGRAM_DELIVERY_VERSION}]")
    print(f"  endpoint : {TELEGRAM_API_BASE} (https, port 443)")
    print()

    token = os.getenv(TELEGRAM_BOT_TOKEN_ENV)

    # -- discovery mode: the step between a token and a chat id ---------
    if args.discover:
        if not token:
            print(f"  [X] {TELEGRAM_BOT_TOKEN_ENV} is not set in this process.")
            print(SETUP_HELP)
            return 1
        print(f"  [OK] {TELEGRAM_BOT_TOKEN_ENV} is set ({token[:10]}...)")
        print("  looking for a message you sent the bot...")
        chat_id, explanation = discover_chat_id(token)
        if not chat_id:
            print(f"  [X] no chat id found: {explanation}")
            print()
            if "INTERCEPTED" in explanation or "HTML" in explanation:
                for line in diagnose_interception():
                    print(f"      {line}")
                return 1
            print("      A bot cannot start a conversation. Open your bot in")
            print("      Telegram, send it any message, then run this again.")
            print()
            print("      If you HAVE messaged it, probing the connection:")
            for line in diagnose_interception():
                print(f"      {line}")
            return 1
        print(f"  [OK] chat id {chat_id} ({explanation})")
        print()
        print("  Set it, then open a NEW terminal:")
        print(f'    [Environment]::SetEnvironmentVariable(')
        print(f'        "{TELEGRAM_CHAT_ID_ENV}","{chat_id}","User")')
        return 0

    chat_id = os.getenv(TELEGRAM_CHAT_ID_ENV)

    if not token and not chat_id:
        print(f"  [X] neither {TELEGRAM_BOT_TOKEN_ENV} nor "
              f"{TELEGRAM_CHAT_ID_ENV} is set in this process.")
        print(SETUP_HELP)
        return 1

    problems = credential_problems(token, chat_id)
    if problems:
        print("  [X] the credential cannot be used as it stands:")
        for problem in problems:
            print(f"      - {problem}")
        print(SETUP_HELP)
        return 1

    print(f"  [OK] {TELEGRAM_BOT_TOKEN_ENV} = {str(token)[:10]}... "
          f"({len(str(token))} chars, bot-token shape)")
    print(f"  [OK] {TELEGRAM_CHAT_ID_ENV} = {chat_id}")
    print()
    print("  sending a real message to your phone...")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    text = args.message or (
        f"invest-by-score: alert channel verified\n\n"
        f"Sent  : {stamp}\n"
        f"Module: {TELEGRAM_DELIVERY_VERSION}\n\n"
        f"If this is on your phone, monitoring works without the laptop "
        f"being in front of you.\n\n"
        f"What this does NOT yet prove: no alert evaluator is wired to this "
        f"transport, so nothing arrives on its own until one is."
    )

    result = send_telegram_alert(text)

    if result.status == ALERT_DELIVERY_UNCONFIGURED:
        print(f"  [X] unconfigured: {result.reason}")
        print(SETUP_HELP)
        return 1

    if result.status == ALERT_DELIVERY_AUTH_FAILED:
        print("  [X] Telegram REFUSED the credential.")
        print(f"      {result.reason}")
        return 1

    if not result.ok:
        print(f"  [X] the send failed: {result.reason}")
        print()
        print("      This is a transport failure, not a credential one — the")
        print("      usual fix is to retry rather than change anything.")
        return 1

    print(f"  [OK] Telegram accepted the message for chat {result.chat_id}")
    if result.truncated:
        print("  [!!] the message was TRUNCATED to fit Telegram's 4096 limit")
    print()
    print("  Check your phone. It should arrive within a second or two.")
    print()
    print("  WHAT THIS DOES NOT PROVE. The channel works; nothing uses it")
    print("  yet. MEASURED 2026-10-04, all seven alert evaluators are")
    print("  imported only by their own CI gates (open item 12), so no")
    print("  alert reaches this transport on the live path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
