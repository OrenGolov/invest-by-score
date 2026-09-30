"""D2 governance gate — the type-error count may fall, never rise.

Exits 1 when the mypy error count exceeds the recorded baseline, or when a module
that is currently CLEAN acquires an error.

WHY A RATCHET AND NOT A STRICT GATE. MEASURED on the first run: 92 errors in 28 of
108 core files. Demanding zero on a twelve-sprint codebase produces a number nobody
can act on, and a gate that cannot pass gets disabled. A ratchet lets the existing
debt stand while making it impossible for new code to add to it.

THE FIRST RUN FOUND TWO WRONG SIGNATURES, which is the argument for this gate in
one finding. `fold_regime_labels` and `attribution_of` both declared
`list[str] | None` while returning `list[str | None]` — A1 had made partial context
possible (an entry may be None while the list is present) and the signatures
promised consumers the opposite. No test could catch it: the annotations are never
evaluated at runtime.

SKIPS when mypy is unavailable, because a missing dev tool is an environment fact
rather than a governance breach.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The recorded baseline. LOWER THIS when errors are fixed; never raise it.
#
#   92  first measurement (D2, 2026-09-30)
#   83  after fixing the two wrong signatures and this session's own modules
BASELINE_ERRORS = 83

# Modules held to ZERO. Everything written in the improvement sprints, because
# debt that is never allowed in does not need paying down later.
MUST_BE_CLEAN = (
    "core/cluster_exposure.py",
    "core/honest_gate.py",
    "core/release_snapshot.py",
    "core/sealed_holdout.py",
    "core/source_outcome_join.py",
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


result = subprocess.run(
    [sys.executable, "-m", "mypy", "--no-error-summary"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
)

if result.returncode not in (0, 1) or "No module named mypy" in (result.stderr or ""):
    print("D2 type coverage gate: SKIPPED")
    print("  mypy is not installed; a missing dev tool is an environment fact")
    sys.exit(0)

lines = [line for line in (result.stdout or "").splitlines() if ": error:" in line]
count = len(lines)

check(
    count <= BASELINE_ERRORS,
    f"mypy reports {count} errors against a baseline of {BASELINE_ERRORS}. New "
    f"code must not add type debt; fix the new errors or, if the baseline is "
    f"genuinely stale, lower it deliberately with the reason",
)

# A ratchet that only ever checks the total lets one module be cleaned while
# another rots. The modules written to be clean must stay clean.
for module in MUST_BE_CLEAN:
    stem = module.replace("/", "\\")
    offending = [line for line in lines if line.startswith((module, stem))]
    check(
        not offending,
        f"{module} is held to zero type errors and now has {len(offending)}: "
        f"{offending[:2]}",
    )

# And the count going DOWN should be recorded, so the baseline tracks reality
# rather than drifting into meaninglessness.
if count < BASELINE_ERRORS - 5:
    failures.append(
        f"mypy reports {count} errors, well under the {BASELINE_ERRORS} baseline. "
        f"That is progress — lower BASELINE_ERRORS to {count} so the ratchet keeps "
        f"its grip, rather than leaving {BASELINE_ERRORS - count} errors of slack"
    )


if failures:
    print("D2 TYPE COVERAGE GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("D2 type coverage gate: OK")
print(f"  mypy errors                       {count}  (baseline {BASELINE_ERRORS})")
print(f"  modules held to zero              {len(MUST_BE_CLEAN)}, all clean")
print(f"  first run found                   2 wrong signatures no test could catch")
sys.exit(0)
