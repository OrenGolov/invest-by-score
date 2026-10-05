"""Check the last collection run and alert if the operator needs to know.

This is the "monitoring and alerting for collection failures" half of
scheduling. The collector itself reports to a log; a log nobody reads is
not monitoring, and MEASURED 2026-10-04 the 2026-09-27 -> 2026-10-04 gap
lost a week of news with nobody noticing.

Run it right after the collector, from the same scheduled task:

    python scripts/daily_collect.py ; python scripts/monitor_collection.py

What it does NOT do: decide policy. `core.collection_monitor` decides
whether a run is worth an alert (perishable loss and silence yes; quota
no, because a quota day is the expected steady state on a 100/day tier).
This script delivers that verdict and nothing more.

DELIVERY IS BEST-EFFORT BY DESIGN. Both channels are tried, and a channel
that is unconfigured or blocked is reported rather than raised — exit code
reflects whether the COLLECTION is healthy, not whether the notification
arrived. A monitor that fails because email is unset would convert a
notification gap into a monitoring outage.

    python scripts/monitor_collection.py            # check and alert
    python scripts/monitor_collection.py --dry-run  # decide, send nothing
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.collection_monitor import (  # noqa: E402
    evaluate_collection,
    load_reports,
    render_verdict,
)
from core.config import COLLECT_REPORT_PATH, COLLECTION_MONITOR_VERSION  # noqa: E402


def deliver(subject: str, body: str) -> list[str]:
    """Send one alert through every configured channel.

    Returns a human-readable line per channel. Never raises: a channel failure
    is information, not a reason to abort monitoring.

    EMAIL ONLY on this branch. The Telegram channel lives on `alert-delivery`
    and is not ported: the operator chose Gmail, and MEASURED today the block
    that motivated Telegram is gone -- `smtp.gmail.com:587` completes a full
    STARTTLS handshake with AUTH advertised, and `api.telegram.org` presents a
    genuine certificate rather than the interception appliance that branch
    found. If SMTP is ever swallowed again, the module is there to port.
    """
    outcomes: list[str] = []

    try:
        from email.message import EmailMessage

        from core.alert_email import credentials, send

        user, password, recipient, reason = credentials()
        if not user or not password or not recipient:
            # NOT AN ERROR. An unconfigured channel is a stated condition, and
            # the reason names the variable to set.
            outcomes.append(f"email    [UNCONFIGURED] {reason}")
        else:
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = user
            message["To"] = recipient
            message.set_content(body)
            sent, detail = send(message)
            outcomes.append(
                f"email    [{'SENT' if sent else 'FAILED'}] {detail}"
            )
    except Exception as exc:  # noqa: BLE001 - a channel must not break the monitor
        outcomes.append(f"email    [ERROR] {type(exc).__name__}: {exc}")

    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide whether to alert, but send nothing",
    )
    parser.add_argument(
        "--as-of", default=None, help="treat this ISO date as today (for testing)"
    )
    args = parser.parse_args()

    today = date.fromisoformat(args.as_of) if args.as_of else date.today()

    print(f"collection monitor  [{COLLECTION_MONITOR_VERSION}]")
    reports = load_reports(REPO_ROOT / COLLECT_REPORT_PATH)
    print(f"  reports on disk : {len(reports)}")
    print(f"  as of           : {today}")

    # THE LAST RUN IS THE ONE FOR TODAY, not merely the newest line. A
    # collector that stopped days ago leaves a newest line that looks like a
    # completed run; treating it as today's would report health while the
    # job was dead.
    todays = [r for r in reports if str(r.get("as_of", ""))[:10] == today.isoformat()]
    latest = todays[-1] if todays else None
    history = [r for r in reports if r is not latest]

    if latest is None:
        print(f"  today's run     : ABSENT")
    else:
        print(f"  today's run     : {latest.get('status', '?')}")

    verdict = evaluate_collection(latest, history, today=today)
    print()
    print(f"  {render_verdict(verdict)}")

    if not verdict.should_alert:
        print()
        print("  Nothing to send. Collection is healthy, or its failure is")
        print("  the expected kind (see COLLECTION_ALERT_ON_QUOTA).")
        return 0

    print()
    for reason in verdict.reasons:
        print(f"  - {reason}")

    if args.dry_run:
        print()
        print("  --dry-run: would have sent:")
        print(f"    subject: {verdict.subject}")
        for line in verdict.body.splitlines():
            print(f"    | {line}")
        return 1

    print()
    for outcome in deliver(verdict.subject, verdict.body):
        print(f"  {outcome}")

    # Exit code reflects COLLECTION health, not delivery success. A blocked
    # channel must not make a healthy collection look broken, nor a failed
    # collection look fine.
    return 1


if __name__ == "__main__":
    sys.exit(main())
