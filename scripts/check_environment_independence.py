"""D5 governance gate — a gate must not measure the machine it runs on.

Exits 1 when a gate's verdict changes with gitignored data present or absent.

THE BUG CLASS THIS CATCHES HAS OCCURRED FOUR TIMES:

    9e7852d  a gate needed gitignored data; passed locally, failed on every push
    2aa15f2  the same bug INVERTED; passed in CI, failed locally
    755e94b  an assertion keyed to whether the working tree was clean
    a99b1d7  a ratchet whose threshold depended on the interpreter and OS

Each cost a red build and a diagnosis. Each was found by a human noticing, not by
anything in the repository. The rule was written down after the first one and
restated after the third, and it still recurred — which is the argument for
checking it mechanically.

HOW. Run every gate twice: once as the repository stands, and once with the
gitignored data stores moved aside. A gate whose exit status differs is measuring
the checkout rather than the repository.

WHAT IT CANNOT CATCH, stated so nobody trusts it further than it goes: the fourth
instance (a99b1d7) turned on the Python version and OS, which this cannot vary. A
green run here does not prove a gate is CI-safe; it proves only that gitignored
data does not decide its answer.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []

# The gitignored stores a gate might read. Moving these aside simulates a fresh
# clone without needing one.
GITIGNORED_STORES = (
    "data/event_memory.jsonl",
    "data/decision_audit.jsonl",
    "data/paper_orders.jsonl",
    "data/collection_report.jsonl",
)

# Gates excluded from the sweep, each with the reason.
SKIP = {
    # Re-runs the whole suite; the cost is minutes per invocation and it reads no
    # gitignored store.
    "check_reproducibility.py": "trains models; minutes per run",
    # This gate. Running it inside itself would recurse.
    "check_environment_independence.py": "would recurse",
    # Legitimately reports on the collector's local history, which IS a property
    # of the machine. It already refuses to judge a machine that has never run
    # the collector.
    "check_data_coverage.py": "deliberately machine-specific, and says so",
}


def run(gate: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(gate)],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=600,
    )
    return result.returncode


gates = sorted(
    path
    for path in (REPO_ROOT / "scripts").glob("check_*.py")
    if path.name not in SKIP
)

if not gates:
    print("D5 ENVIRONMENT INDEPENDENCE GATE: FAILED")
    print("  no gates were found; 'nothing to check' must not read as 'all clear'")
    sys.exit(1)

present = {gate.name: run(gate) for gate in gates}

# Move the gitignored stores aside, run again, and restore no matter what.
moved: list[tuple[Path, Path]] = []
holding = Path(tempfile.mkdtemp(prefix="d5-stores-"))
try:
    for relative in GITIGNORED_STORES:
        source = REPO_ROOT / relative
        if source.exists():
            destination = holding / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))

    absent = {gate.name: run(gate) for gate in gates}
finally:
    for source, destination in moved:
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(source))
    shutil.rmtree(holding, ignore_errors=True)

# Every store must be back. A gate that measures the machine is a nuisance; a
# check that DELETES a data store is a catastrophe.
for relative in GITIGNORED_STORES:
    source = REPO_ROOT / relative
    if any(source == original for original, _ in moved) and not source.exists():
        failures.append(
            f"{relative} was moved aside and not restored — this check must "
            f"never cost data"
        )

differing = sorted(
    name for name in present if present[name] != absent.get(name)
)
for name in differing:
    failures.append(
        f"{name} exits {present[name]} with the gitignored stores present and "
        f"{absent[name]} with them absent. It is measuring the checkout rather "
        f"than the repository — the bug class that has cost four red builds. "
        f"Assert an invariant of the TRACKED contents, or tolerate the absence"
    )

if moved:
    note = f"{len(moved)} gitignored store(s) moved aside and restored"
else:
    # SKIP, not fail. On a fresh clone — which is exactly what CI is — there are
    # no gitignored stores to move aside, so the two runs are identical by
    # construction and this check cannot discriminate. Failing there would break
    # every CI build for the absence of data CI is never supposed to have, which
    # is the same environment-dependence error this gate exists to catch.
    #
    # It runs meaningfully on a DEVELOPER's machine, which is where the data
    # lives and where all four historical instances were introduced.
    print("D5 environment independence gate: SKIPPED")
    print("  no gitignored stores are present, so the two runs are identical by")
    print("  construction. This check discriminates only where the data lives —")
    print("  a developer's machine — which is where all four instances began.")
    sys.exit(0)


if failures:
    print("D5 ENVIRONMENT INDEPENDENCE GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("D5 environment independence gate: OK")
print(f"  gates swept                       {len(gates)}")
print(f"  verdicts that changed             0")
print(f"  {note}")
print(f"  does NOT prove CI-safety          the fourth instance turned on the")
print(f"                                    Python version, which this cannot vary")
sys.exit(0)
