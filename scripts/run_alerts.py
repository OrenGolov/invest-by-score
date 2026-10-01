"""Run every detector over the portfolio, store the findings, deliver them.

This is the step that turns seven built-but-unwired detectors into a monitoring
system. MEASURED before it existed: no production module imported any of A1-A7,
so the whole alert family was exercised only by gates and tests.

    detect      run each detector per ticker against yesterday and today
    grade       assign one of the operator's five priorities, and an action
    suppress    A7 decides whether the reader sees it again
    store       append to data/alerts.jsonl -- nothing is ever deleted
    deliver     Urgent/Very High/High email now; Medium/Low batch to a digest

**THE PREVIOUS OBSERVATION COMES FROM THE LEDGER, NOT FROM A RE-RUN.** Every
detector compares two states, so the runner needs yesterday's. Recomputing it
would re-derive a value from today's data and call it yesterday's -- which is
exactly the staleness trap `run_forecasts` measured, where anchoring to whenever
a ticker last produced a memory left the chart state a median 144 days behind. The
previous state is read from the stored alert whose id matches.

**ONE TICKER'S FAILURE MUST NOT COST THE OTHERS THEIR RUN.** Same reasoning as
`daily_collect`'s fail-soft-per-source: a dead provider for NVDA is not a reason
to lose the regime change on MSFT. Each ticker is wrapped, and the exit code
reflects whether anything was LOST, not whether everything succeeded.

**A DETECTOR THAT CANNOT RUN REPORTS THAT IT CANNOT RUN.** It never reports calm.
NOT_EVALUATED findings are stored and digested rather than dropped, because a
blind detector is the one thing silence would hide.

Usage:
    python scripts/run_alerts.py                    # detect, store, deliver
    python scripts/run_alerts.py --dry-run          # detect and show, send nothing
    python scripts/run_alerts.py --no-email         # store only
    python scripts/run_alerts.py --tickers NVDA,MSFT
    python scripts/run_alerts.py --digest-only      # just send today's digest
    python scripts/run_alerts.py --detectors regime,news
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.alert_email import (  # noqa: E402
    build_digest,
    build_message,
    credentials,
    routing,
    send,
)
from core.alert_priority import grade  # noqa: E402
from core.alert_store import (  # noqa: E402
    CHANNEL_DASHBOARD,
    CHANNEL_DIGEST,
    CHANNEL_EMAIL,
    DELIVERY_DUPLICATE,
    DELIVERY_FAILED,
    DELIVERY_SENT,
    DELIVERY_SKIPPED,
    append_alert,
    build_record,
    delivered_ids,
    load_alerts,
    query,
    record_delivery,
)
from core.alert_suppression import disposition  # noqa: E402
from core.config import (  # noqa: E402
    ALERT_PRIORITIES,
    SUPPRESS_DELIVER,
    SUPPRESS_GATED,
)

LOGGER = logging.getLogger("run_alerts")

# The detectors this runner can drive. Named so `--detectors` reads clearly and
# so adding one is a deliberate edit here rather than a discovery that silently
# changes what a scheduled run does.
DETECTORS: tuple[str, ...] = ("regime", "news")


def _tickers(explicit: str) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    from fetch_data import PORTFOLIO_TICKERS

    return sorted({str(t).upper() for t in PORTFOLIO_TICKERS if str(t).strip()})


def _previous_state(ticker: str, detector: str, rows: list[dict]) -> dict | None:
    """The last stored finding for this ticker and detector, or None.

    Read from the LEDGER rather than recomputed. Recomputing would derive a value
    from today's data and label it yesterday's -- the staleness trap measured in
    `run_forecasts`, where a memory-anchored chart state ran a median 144 days
    behind and retrieved a different history entirely.
    """
    matches = [
        row
        for row in rows
        if str(row.get("ticker") or "").upper() == ticker
        and row.get("alert") == detector
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: str(row.get("detected_at") or ""))
    return matches[-1]


def _sessions_since(previous: dict | None, now: datetime) -> int | None:
    """Approximate sessions since a stored finding, or None when unknown.

    None does NOT suppress: A7 is explicit that suppressing on an unknown age
    hides an alert on a guess. Business days are counted rather than calendar
    days because the cooldown is measured in SESSIONS.
    """
    if not previous:
        return None
    stamp = previous.get("detected_at")
    if not stamp:
        return None
    try:
        then = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        return None
    days = (now.date() - then.date()).days
    if days < 0:
        return None
    # Five sessions per seven calendar days. Deliberately approximate, and
    # deliberately NOT a holiday calendar: the cooldown is 15 sessions, so a
    # one-session error cannot change a delivery decision.
    return int(round(days * 5 / 7))


def detect_regime(ticker: str, as_of: str, previous: dict | None) -> dict | None:
    """A4: has the regime changed, and has the change earned an alert?"""
    from core.regime_agent import classify_regime
    from core.regime_alert import regime_change_alert
    from fetch_data import fetch_price_history

    frame = fetch_price_history(ticker, period="5y", interval="1d")
    if frame is None or getattr(frame, "empty", True):
        return None
    current = classify_regime(frame)
    current["ticker"] = ticker

    # The PRIOR label is the one the ledger stored, not a recomputed one. When
    # there is no stored finding the detector reports NOT_EVALUATED, which is
    # correct: a first observation is not a change.
    #
    # THE FIELD IS `current`, NOT `after`. CAUGHT ON THE SECOND LIVE RUN: the
    # first version read `detail["after"]`, a key A4 never writes, so the prior
    # label was always None and every regime alert reported NOT_EVALUATED
    # FOREVER -- the detector could never make a comparison at all. A4 records
    # the two labels as `previous` and `current`.
    prior = None
    if previous:
        detail = previous.get("detail") or {}
        label = detail.get("current")
        if label:
            prior = {"label": label, "computable": True}

    alert = regime_change_alert(prior, current)
    alert["ticker"] = ticker
    alert["as_of"] = as_of
    return alert


def detect_news(ticker: str, as_of: str, previous: dict | None) -> dict | None:
    """A9: did a material news event occur, graded by its type's history?"""
    from core.event_memory import load_memory_objects
    from core.news_adapter import build_news_snapshot
    from core.news_event_alert import news_event_alert

    snapshot = build_news_snapshot(ticker, as_of)
    memories = load_memory_objects()
    return news_event_alert(snapshot, ticker, memories=memories, as_of=as_of)


DETECTOR_FUNCTIONS = {
    "regime": detect_regime,
    "news": detect_news,
}


def _title_for(alert: dict) -> str:
    """The short, direct description the subject line needs."""
    if alert.get("title"):
        return str(alert["title"])
    detector = str(alert.get("alert") or "alert").replace("_", " ").title()
    state = str(alert.get("kind") or alert.get("verdict") or "").replace("_", " ")
    return f"{detector}: {state.title()}" if state else detector


def _summary_for(alert: dict) -> dict | None:
    """The Who/What/When/Why block, when the detector can supply one."""
    if alert.get("alert") == "news_event":
        from core.news_event_alert import summary_for

        return summary_for(alert)
    return None


def run_detection(
    tickers: list[str],
    detectors: list[str],
    as_of: str,
    *,
    store_path: Path | None = None,
) -> dict:
    """Detect, grade, suppress and store. Returns a report; never raises."""
    now = datetime.now(timezone.utc)
    existing = load_alerts(store_path)

    stored: list[dict] = []
    failures: list[str] = []
    counts = {"detected": 0, "fired": 0, "suppressed": 0, "ungraded": 0}

    for ticker in tickers:
        for detector in detectors:
            function = DETECTOR_FUNCTIONS.get(detector)
            if function is None:
                continue
            previous = _previous_state(ticker, f"{detector}_change", existing)
            if previous is None:
                previous = _previous_state(ticker, f"{detector}_event", existing)
            try:
                alert = function(ticker, as_of, previous)
            except Exception as exc:
                # FAIL-SOFT PER TICKER. One dead provider must not cost the rest
                # of the portfolio its run.
                LOGGER.warning("%s/%s failed: %s", ticker, detector, exc)
                failures.append(f"{ticker}/{detector}: {exc}")
                continue
            if alert is None:
                continue
            counts["detected"] += 1

            graded = grade(alert)
            if graded["priority"] is None:
                counts["ungraded"] += 1

            # A7 decides whether the reader sees this again. An ungraded alert
            # still goes through: "the detector is blind" is a finding.
            try:
                verdict = disposition(
                    alert,
                    previous=None,
                    sessions_since=_sessions_since(previous, now),
                )
            except Exception as exc:
                LOGGER.warning("suppression failed for %s/%s: %s", ticker, detector, exc)
                verdict = {"disposition": SUPPRESS_DELIVER, "reason": str(exc)}

            if verdict.get("disposition") not in (
                SUPPRESS_DELIVER,
                SUPPRESS_GATED,
                "NOT_EVALUATED",
            ):
                counts["suppressed"] += 1
                # A SUPPRESSED ALERT IS STILL STORED. "Nothing fired" and "it
                # fired and we chose not to show it" are different facts, and
                # only one of them can be audited after a loss.
                record = build_record(
                    alert,
                    graded,
                    title=_title_for(alert),
                    summary=_summary_for(alert),
                    disposition=verdict,
                )
                append_alert(record, store_path)
                continue

            counts["fired"] += 1
            record = build_record(
                alert,
                graded,
                title=_title_for(alert),
                summary=_summary_for(alert),
                disposition=verdict,
            )
            append_alert(record, store_path)
            record_delivery(
                record, channel=CHANNEL_DASHBOARD, status=DELIVERY_SENT,
                detail="stored for the Monitoring tab",
            )
            stored.append(record)

    return {
        "as_of": as_of,
        "ran_at": now.isoformat(),
        "tickers": len(tickers),
        "detectors": detectors,
        "counts": counts,
        "failures": failures,
        "delivered": stored,
    }


def deliver(records: list[dict], *, dry_run: bool, as_of: str) -> dict:
    """Email what should be emailed. Never raises; a failure is reported.

    Immediate bands go out one message each; digested bands collect into one.
    An alert already SENT under its id is not re-sent -- that is the duplicate
    prevention the operator asked for, and it works because the id excludes
    every timestamp.
    """
    user, password, recipient, reason = credentials()
    already = delivered_ids(CHANNEL_EMAIL)
    already |= delivered_ids(CHANNEL_DIGEST)

    report = {
        "immediate_sent": 0,
        "immediate_failed": 0,
        "digested": 0,
        "duplicates": 0,
        "skipped": 0,
        "digest_sent": False,
        "reason": reason,
    }

    immediate: list[dict] = []
    digested: list[dict] = []
    for record in records:
        identifier = str(record.get("alert_id") or "")
        if identifier and identifier in already:
            report["duplicates"] += 1
            record_delivery(
                record, channel=CHANNEL_EMAIL, status=DELIVERY_DUPLICATE,
                detail="already emailed under this id",
            )
            continue
        route, _ = routing(record.get("priority"))
        (immediate if route == "immediate" else digested).append(record)

    if not user or not password:
        # Email degrades alone. Every record is already stored and visible.
        for record in immediate + digested:
            record_delivery(
                record, channel=CHANNEL_EMAIL, status=DELIVERY_SKIPPED,
                detail=reason,
            )
        report["skipped"] = len(immediate) + len(digested)
        return report

    for record in immediate:
        message = build_message(record, sender=user, recipient=recipient)
        sent, detail = send(message, dry_run=dry_run)
        record_delivery(
            record,
            channel=CHANNEL_EMAIL,
            status=DELIVERY_SENT if sent else (
                DELIVERY_SKIPPED if dry_run else DELIVERY_FAILED
            ),
            detail=detail,
        )
        report["immediate_sent" if sent else "immediate_failed"] += 1

    if digested:
        message = build_digest(
            digested, sender=user, recipient=recipient, as_of=as_of
        )
        sent, detail = send(message, dry_run=dry_run)
        report["digest_sent"] = sent
        report["digested"] = len(digested)
        for record in digested:
            record_delivery(
                record,
                channel=CHANNEL_DIGEST,
                status=DELIVERY_SENT if sent else (
                    DELIVERY_SKIPPED if dry_run else DELIVERY_FAILED
                ),
                detail=detail,
            )

    return report


def send_digest_only(as_of: str, *, dry_run: bool) -> dict:
    """Send one digest for everything stored today that has not been emailed."""
    start = f"{as_of}T00:00:00+00:00"
    end = as_of
    todays = query(start=start, end=end)
    pending = [
        row
        for row in todays
        if routing(row.get("priority"))[0] == "digest"
    ]
    if not pending:
        return {"digested": 0, "digest_sent": False, "reason": "nothing to digest"}
    return deliver(pending, dry_run=dry_run, as_of=as_of)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--detectors", default=",".join(DETECTORS),
                        help=f"comma-separated; known: {', '.join(DETECTORS)}")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true",
                        help="detect and render, but send no email")
    parser.add_argument("--no-email", action="store_true",
                        help="store only; do not attempt delivery")
    parser.add_argument("--digest-only", action="store_true",
                        help="send today's digest and nothing else")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of tickers, for a quick check")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.digest_only:
        report = send_digest_only(args.as_of, dry_run=args.dry_run)
        print(f"digest [{args.as_of}]: {report}")
        return 0

    requested = [d.strip() for d in args.detectors.split(",") if d.strip()]
    unknown = [d for d in requested if d not in DETECTORS]
    if unknown:
        print(f"unknown detector(s): {', '.join(unknown)}")
        print(f"known: {', '.join(DETECTORS)}")
        return 2

    tickers = _tickers(args.tickers)
    if args.limit > 0:
        tickers = tickers[: args.limit]

    result = run_detection(tickers, requested, args.as_of)
    counts = result["counts"]

    print(f"alerts [{result['as_of']}] over {result['tickers']} tickers")
    print(f"  detected   {counts['detected']}")
    print(f"  delivered  {counts['fired']}")
    print(f"  suppressed {counts['suppressed']}  (stored, not shown again)")
    print(f"  ungraded   {counts['ungraded']}  (a detector could not decide)")

    by_band: dict[str, int] = {}
    for record in result["delivered"]:
        key = str(record.get("priority") or "(ungraded)")
        by_band[key] = by_band.get(key, 0) + 1
    if by_band:
        print("  by priority:")
        for band in list(ALERT_PRIORITIES) + ["(ungraded)"]:
            if band in by_band:
                print(f"      {band:12} {by_band[band]}")

    if result["failures"]:
        print(f"  failures   {len(result['failures'])}")
        for line in result["failures"][:10]:
            print(f"      {line}")

    if args.no_email:
        print("  email: not attempted (--no-email)")
        return 0

    delivery = deliver(
        result["delivered"], dry_run=args.dry_run, as_of=args.as_of
    )
    print("  email:")
    print(f"      immediate sent   {delivery['immediate_sent']}")
    if delivery["immediate_failed"]:
        print(f"      immediate failed {delivery['immediate_failed']}")
    print(f"      digested         {delivery['digested']}"
          f"  (one message, sent={delivery['digest_sent']})")
    if delivery["duplicates"]:
        print(f"      duplicates       {delivery['duplicates']}  (already emailed)")
    if delivery["skipped"]:
        print(f"      skipped          {delivery['skipped']}")
        print(f"      {delivery['reason']}")

    # A ticker that failed is a LOSS; an alert that was suppressed is not.
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
