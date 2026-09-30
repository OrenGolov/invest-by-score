"""D3 coverage ratchet — coverage may rise, never fall.

DELIBERATELY NOT NAMED `check_*.py`. CI discovers every `scripts/check_*.py` and
runs them all, and this one re-runs the entire test suite: MEASURED at 353 s, which
would more than double the 126 s gate sweep by repeating work CI already does. It
runs as its own CI step instead, after the suite.

Exits 1 when total coverage drops below the recorded floor.

WHY THIS EXISTS. MEASURED before D3: 89 of 90 core modules were IMPORTED by some
test, and that number was the only evidence of coverage there was. Import is not
coverage — a module can be imported and have half its branches never run — so
which code was actually exercised was unknown.

MEASURED after: 93.8% of 15,715 statements, with the floor set BELOW it so the gate
ratchets rather than blocks.

A FINDING WORTH KEEPING. The raw number was 90%, and the only modules below 80%
were the ten config parts (71–78%). Their uncovered lines are `raise ValueError`
branches inside IMPORT-TIME VALIDATORS — reachable only when a contract is
VIOLATED, which nothing at runtime does, and which the mutation tests exercise by
editing the file and re-importing. Excluding those branches takes `_sizing` from
71% to 100% and the project from 90% to 93.8%, which is the honest figure: the
uncovered remainder is code that runs, not code that guards.

SKIPS when pytest-cov is unavailable: a missing dev tool is an environment fact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The recorded floor. RAISE THIS when coverage improves; never lower it without
# stating why.
#
#   90.0%  first measurement (D3, 2026-09-30), raw
#   93.8%  with contract-violation branches excluded — the honest figure
#   90.0%  the floor, deliberately below the measurement so it ratchets
COVERAGE_FLOOR = 90.0

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


report = REPO_ROOT / "data" / ".cache" / "coverage.json"
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "-p",
        "no:cacheprovider",
        "--cov=core",
        f"--cov-report=json:{report}",
        "--cov-report=",
        "-x",
    ],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
)

output = (result.stdout or "") + (result.stderr or "")
if "unrecognized arguments" in output or "No module named pytest_cov" in output:
    print("D3 coverage gate: SKIPPED")
    print("  pytest-cov is not installed; a missing dev tool is an environment fact")
    sys.exit(0)

if not report.exists():
    print("D3 COVERAGE RATCHET: FAILED")
    print("  the coverage report was not produced; the suite may have errored")
    print(f"  {output[-500:]}")
    sys.exit(1)

data = json.loads(report.read_text(encoding="utf-8"))
percent = float(data["totals"]["percent_covered"])
statements = int(data["totals"]["num_statements"])

check(
    percent >= COVERAGE_FLOOR,
    f"coverage is {percent:.1f}%, below the {COVERAGE_FLOOR}% floor. New code "
    f"must be tested to at least the standard the rest of the project holds; "
    f"if the floor is genuinely wrong, change it deliberately with the reason",
)

# A floor far below the measurement is not a ratchet, it is decoration.
check(
    percent - COVERAGE_FLOOR < 6.0,
    f"coverage is {percent:.1f}% against a {COVERAGE_FLOOR}% floor — "
    f"{percent - COVERAGE_FLOOR:.1f} points of slack. Raise the floor so the "
    f"gate keeps its grip",
)

# And the suite itself must have passed: coverage of a failing suite measures
# nothing.
check(
    result.returncode == 0,
    f"the suite did not pass while measuring coverage (exit {result.returncode}); "
    f"coverage of a failing suite measures nothing",
)


if failures:
    print("D3 COVERAGE RATCHET: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("D3 coverage ratchet: OK")
print(f"  statements measured               {statements:,}")
print(f"  coverage                          {percent:.1f}%  (floor {COVERAGE_FLOOR}%)")
print(f"  before D3                         unknown — 89/90 modules were merely")
print(f"                                    IMPORTED by a test, which is not coverage")
sys.exit(0)
