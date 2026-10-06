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
    python scripts/monitor_collection.py --explain-tickers        # per-ticker
    python scripts/monitor_collection.py --explain-tickers --json # machine-readable

**WHY --explain-tickers EXISTS.** The daily digest that reaches the inbox is
produced OUTSIDE this repository -- MEASURED 2026-10-06, no file on this machine
and no commit in any branch contains its strings ("Unprioritised", "Action:
Review", "not trade instructions"), and the scheduled task runs only
`daily_collect.py` and this script. That digest sent 77 alerts which all said the
same wrong thing: that each ticker "was not looked at today" because of a
rotation of "75 per run", when the batch was 40, the provider quota was spent,
and AMZN -- named as not looked at -- was the one ticker that HAD been fetched.

This mode prints the accurate per-ticker reason and action, so whatever composes
that email has one command to call and one JSON shape to read instead of
re-deriving an explanation from prose. Until it does, this is also how a human
gets the truthful answer for a given ticker.
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
    explain_ticker_news_gap,
    load_reports,
    render_verdict,
)
from core.config import COLLECT_REPORT_PATH, COLLECTION_MONITOR_VERSION  # noqa: E402


def deliver(subject: str, body: str) -> list[str]:
    """Send one alert through every configured channel.

    Returns a human-readable line per channel. Never raises: a channel
    failure is information, not a reason to abort monitoring.
    """
    outcomes: list[str] = []

    try:
        from core.alert_delivery import render_delivery, send_alert_email

        outcomes.append("email    " + render_delivery(
            send_alert_email(subject=subject, body=body)
        ))
    except Exception as exc:  # noqa: BLE001 - a channel must not break the monitor
        outcomes.append(f"email    [ERROR] {type(exc).__name__}: {exc}")

    try:
        from core.telegram_delivery import render_result, send_telegram_alert

        outcomes.append("telegram " + render_result(
            send_telegram_alert(f"{subject}\n\n{body}")
        ))
    except Exception as exc:  # noqa: BLE001
        outcomes.append(f"telegram [ERROR] {type(exc).__name__}: {exc}")

    return outcomes


def _explain_tickers(report, today, explicit: str, as_json: bool) -> int:
    """Print the real reason each ticker has no news, and what to do about it.

    Grouped by KIND rather than listed per ticker: 77 separate messages saying
    the same thing is what made the original digest unreadable, and the whole
    point of classifying failures is that one sentence can now cover every
    ticker that shares a cause.
    """
    import json as _json

    if explicit:
        tickers = [t.strip().upper() for t in explicit.split(",") if t.strip()]
    else:
        from fetch_data import PORTFOLIO_TICKERS

        tickers = sorted({str(t).upper() for t in PORTFOLIO_TICKERS if str(t).strip()})

    explained = {
        ticker: explain_ticker_news_gap(ticker, today.isoformat(), report)
        for ticker in tickers
    }

    if as_json:
        print(_json.dumps({
            "as_of": today.isoformat(),
            "version": COLLECTION_MONITOR_VERSION,
            "tickers": explained,
        }, indent=2, sort_keys=True))
        return 0

    grouped: dict[str, list[str]] = {}
    for ticker, detail in explained.items():
        grouped.setdefault(detail["kind"], []).append(ticker)

    print()
    print(f"  per-ticker news status for {today.isoformat()}")
    print()
    for kind, names in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sample = explained[names[0]]
        print(f"  {kind}  ({len(names)} ticker(s))")
        print(f"    {', '.join(sorted(names))}")
        # The reason names its own ticker, so it is shown with that ticker
        # substituted out -- the sentence is about the CAUSE, not the symbol.
        print(f"    why   : {sample['reason'].replace(names[0], '<ticker>', 1)}")
        print(f"    action: {sample['action']}")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decide whether to alert, but send nothing",
    )
    parser.add_argument(
        "--explain-tickers",
        action="store_true",
        help=(
            "print why each ticker has or has not got news today, with the "
            "action for each, instead of evaluating the alert"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="with --explain-tickers, emit JSON for a downstream sender",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help=(
            "with --explain-tickers, limit to these comma-separated symbols "
            "(default: the whole portfolio)"
        ),
    )
    parser.add_argument(
        "--as-of", default=None, help="treat this ISO date as today (for testing)"
    )
    args = parser.parse_args()

    today = date.fromisoformat(args.as_of) if args.as_of else date.today()

    # --json promises PARSEABLE stdout, so the human banner would corrupt it:
    # a consumer piping this into a parser must not have to strip four header
    # lines it never asked for. They go to stderr, where a human still sees
    # them and `| jq` does not.
    banner = sys.stderr if (args.explain_tickers and args.json) else sys.stdout
    print(f"collection monitor  [{COLLECTION_MONITOR_VERSION}]", file=banner)
    reports = load_reports(REPO_ROOT / COLLECT_REPORT_PATH)
    print(f"  reports on disk : {len(reports)}", file=banner)
    print(f"  as of           : {today}", file=banner)

    # THE LAST RUN IS THE ONE FOR TODAY, not merely the newest line. A
    # collector that stopped days ago leaves a newest line that looks like a
    # completed run; treating it as today's would report health while the
    # job was dead.
    todays = [r for r in reports if str(r.get("as_of", ""))[:10] == today.isoformat()]
    latest = todays[-1] if todays else None
    history = [r for r in reports if r is not latest]

    if latest is None:
        print(f"  today's run     : ABSENT", file=banner)
    else:
        print(f"  today's run     : {latest.get('status', '?')}", file=banner)

    if args.explain_tickers:
        return _explain_tickers(latest, today, args.tickers, args.json)

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
