"""Gate: an irreplaceable local store must not vanish unnoticed.

WHY THIS EXISTS. Twice in one session I destroyed a gitignored store with an
`rm -f data/*.jsonl`-shaped command while resetting test data:

    data/collection_report.jsonl   deleted 2026-10-02, rebuilt from collect.log
    data/event_memory.jsonl        deleted 2026-10-02, 2,921 memories

Neither is tracked by git, so neither could be restored by checkout. The second
loss was invisible for a day: nothing failed, no test broke, and the news
detector simply began grading every event as UNKNOWN_IMPACT -- which reads
exactly like "we have no history yet" rather than "the history was deleted".

THAT SILENCE IS THE DEFECT THIS GATE CLOSES. A store that is expensive or
impossible to rebuild must announce its own absence.

**IT IS A NOTICE, NOT A FAILURE, ON A FRESH CLONE.** A new checkout legitimately
has none of these files, and failing there would make the gate fire for everyone
who ever clones the repository -- the checkout-dependence trap this project has
hit five times. So absence is only an ERROR when the repository shows evidence
the store once existed here: a sibling ledger, a log, or a recorded run.

**REBUILDABLE AND IRREPLACEABLE ARE DIFFERENT.** Each store declares which it is,
and the message says how to rebuild the ones that can be. A store with no rebuild
path is the one worth stopping for.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Each store: what it holds, whether it can be rebuilt, and how.
#
# `witness` is a file whose presence proves this machine HAS run the thing that
# writes the store. Without a witness, absence is just a fresh clone.
STORES: tuple[dict, ...] = (
    {
        "path": "data/event_memory.jsonl",
        "holds": "remembered events and their realized outcomes",
        "witness": "data/raw/yahoo_finance_chart",
        "rebuild": (
            "python scripts/build_event_memory.py --mode backfill --years 5"
        ),
        "note": (
            "the backfill re-derives earnings events from price history, which "
            "is itself re-fetchable; `observed` memories from news are NOT "
            "rebuildable once the 7-day news window closes"
        ),
    },
    {
        "path": "data/collection_report.jsonl",
        "holds": "one row per collector run, which proves collection is alive",
        "witness": "data/collect.log",
        "rebuild": "parse data/collect.log (the same task writes both)",
        "note": (
            "check_data_coverage reads this to tell a real outage from a "
            "machine that has never collected; without it that gate passes "
            "while blind"
        ),
    },
    {
        "path": "data/forecast_ledger.jsonl",
        "holds": "recorded forecasts and their closures",
        # The witness is the ledger's OWN sibling, not a training artifact.
        #
        # CAUGHT ON THE GATE'S FIRST RUN: witnessing on data/training_runs.jsonl
        # reported this store as LOST, when in fact `run_forecasts.py` has never
        # been run on this machine at all -- it was never created, which is a
        # different fact and a different fix. A witness must prove the writing
        # task RAN, not merely that some neighbouring task did.
        "witness": "data/forecast_report.json",
        "rebuild": "python scripts/run_forecasts.py --record",
        "note": (
            "a forecast that was never recorded cannot be scored later, so the "
            "denominator is lost rather than the numerator"
        ),
    },
    {
        "path": "data/alerts.jsonl",
        "holds": "every alert ever raised, for the Monitoring tab",
        "witness": "data/alert_deliveries.jsonl",
        "rebuild": "python scripts/run_alerts.py (re-detects today only)",
        "note": (
            "append-only by contract and the operator asked that nothing ever "
            "be deleted; a re-run recovers today, never the history"
        ),
    },
)


def _exists(relative: str) -> bool:
    return (REPO_ROOT / relative).exists()


def main() -> int:
    missing_with_witness: list[dict] = []
    missing_fresh: list[dict] = []
    present: list[dict] = []

    for store in STORES:
        if _exists(store["path"]):
            present.append(store)
        elif _exists(store["witness"]):
            missing_with_witness.append(store)
        else:
            missing_fresh.append(store)

    if missing_with_witness:
        print("IRREPLACEABLE STORE GATE: FAILED")
        for store in missing_with_witness:
            print()
            print(f"  {store['path']} is MISSING")
            print(f"    holds   : {store['holds']}")
            print(
                f"    evidence: {store['witness']} exists, so this machine HAS "
                f"run the task that writes it"
            )
            print(f"    rebuild : {store['rebuild']}")
            print(f"    note    : {store['note']}")
        print()
        print("  A gitignored store cannot be restored by git. If this was a")
        print("  deliberate reset, rebuild it with the command above; if it was")
        print("  an accident, that is what this gate exists to surface.")
        return 1

    print("irreplaceable store gate: OK")
    print(f"  stores watched                    {len(STORES)}")
    print(f"  present                           {len(present)}")
    if missing_fresh:
        print(f"  absent, no witness                {len(missing_fresh)}")
        for store in missing_fresh:
            print(
                f"      {store['path']} — not judged: "
                f"{store['witness']} is absent too, so this looks like a fresh "
                f"clone rather than a loss"
            )
    print("  twice deleted by an rm during a test reset; neither was")
    print("  recoverable from git, and one went unnoticed for a day")
    return 0


if __name__ == "__main__":
    sys.exit(main())
