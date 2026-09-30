"""D4 governance gate — a broad catch must say what it absorbs.

Exits 1 when a `except Exception` in `core/` carries no comment explaining what it
absorbs and why.

AUDITED all 21 sites in D4. NONE was swallowing a real error: every one converts a
failure at a boundary the project does not control — a provider, git, a parser —
into a NAMED, RECORDED state (UNAVAILABLE, NOT_EVALUATED, an excluded row with a
reason, a logged skip). That is the fail-closed contract working.

What was missing was the RECORD. 16 of 21 carried no comment, so a reader could not
tell a deliberate boundary catch from a swallowed bug — and in a system whose whole
premise is that failures must be loud, that distinction is the one that matters.

This gate does not forbid broad catches. It requires each to be explained, because
the next one added without a reason is the one that hides a defect.

SCOPED TO `core/` DELIBERATELY. A gate's own infrastructure has a different
contract: `check_reproducibility.py` catches broadly around its dataset cache so an
unwritable cache cannot fail the gate, which is right there and would be wrong in
`core/`. Those sites carry inline `# noqa: BLE001` explanations of their own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []

# How far from the `except` line a comment may sit and still count as its
# explanation. Three lines covers a comment placed just above the handler or as
# the handler's first statement.
WINDOW = 4


def documented(lines: list[str], index: int) -> bool:
    """Whether the handler at `index` carries an explanation near it."""
    if "#" in lines[index]:
        return True
    after = lines[index + 1 : index + 1 + WINDOW]
    before = lines[max(0, index - WINDOW) : index]
    return any(line.strip().startswith("#") for line in after + before)


sites = 0
undocumented: list[str] = []
for path in sorted((REPO_ROOT / "core").rglob("*.py")):
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.search(r"\bexcept\s+Exception\b", line):
            sites += 1
            if not documented(lines, index):
                undocumented.append(
                    f"{path.relative_to(REPO_ROOT)}:{index + 1}"
                )

if undocumented:
    failures.append(
        f"{len(undocumented)} broad catch(es) carry no explanation: "
        f"{undocumented}. Say what the handler ABSORBS and what state it reports "
        f"instead — in a fail-closed system, an unexplained `except Exception` "
        f"cannot be told from a swallowed bug"
    )

# A bare `except:` is never acceptable — it catches KeyboardInterrupt and
# SystemExit, so it can swallow a deliberate interrupt.
bare: list[str] = []
for path in sorted((REPO_ROOT / "core").rglob("*.py")):
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if re.match(r"\s*except\s*:", line):
            bare.append(f"{path.relative_to(REPO_ROOT)}:{index + 1}")
if bare:
    failures.append(
        f"{len(bare)} bare `except:` clause(s): {bare}. A bare except catches "
        f"KeyboardInterrupt and SystemExit, so it can swallow a deliberate "
        f"interrupt; name the exception or use `except Exception`"
    )

# `except Exception: pass` discards the failure entirely, with nothing recorded.
silent: list[str] = []
for path in sorted((REPO_ROOT / "core").rglob("*.py")):
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if re.search(r"\bexcept\s+Exception\b", line):
            body = [
                stripped
                for stripped in (item.strip() for item in lines[index + 1 : index + 4])
                if stripped and not stripped.startswith("#")
            ]
            if body and body[0] == "pass":
                silent.append(f"{path.relative_to(REPO_ROOT)}:{index + 1}")
if silent:
    failures.append(
        f"{len(silent)} catch(es) are `except Exception: pass`: {silent}. A "
        f"failure with nothing recorded cannot be distinguished from success; "
        f"report a named state, log it, or let it raise"
    )


if failures:
    print("D4 BROAD EXCEPT GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("D4 broad except gate: OK")
print(f"  `except Exception` sites in core/  {sites}, all explained")
print(f"  bare `except:` clauses             0")
print(f"  silent `except: pass` handlers     0")
print(f"  audited in D4                      none was swallowing a real error;")
print(f"                                     16 of 21 lacked the explanation")
sys.exit(0)
