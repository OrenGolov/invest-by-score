"""C1 governance gate — the config split must stay a pure refactor.

Exits 1 when any of these fails.

C1 split a single 9,783-line `core/config.py` — 20% of all core code, 1,092
constants, 76 import-time validators — into a chained package. The whole value of
that change depends on it having altered NOTHING: every constant keeps its value,
every validator still runs, and `from core.config import ANYTHING` resolves as
before.

TWO THINGS THIS CATCHES, both of which happened while building it:

1. A STAR IMPORT DROPS UNDERSCORE NAMES AT EVERY LINK. The first chained version
   lost 63 of the 76 validators that way, and a later one lost 2 more when a new
   part was imported but left out of the re-export walk. They have already RUN by
   import time, so nothing failed — they were simply unreachable by name, and one
   test asserts a validator raises on a bad weight set.

2. A part growing back past the point of being navigable. `_forecasting` was 3,631
   lines after the first split, which moved the problem rather than solving it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import core.config as config  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


PACKAGE = REPO_ROOT / "core" / "config"

# The largest a single part may be before it stops being navigable. The original
# file was 9,783 lines; the point of the split is that no part approaches that.
MAX_PART_LINES = 2200


# --- 1. THE PACKAGE REPLACED THE FLAT FILE ---------------------------------------

check(
    PACKAGE.is_dir(),
    "core/config is not a package; the C1 split is not in place",
)
check(
    not (REPO_ROOT / "core" / "config.py").exists(),
    "core/config.py still exists beside the package. Python prefers the package, "
    "so the flat file would be dead code that silently drifts from what runs",
)
check(
    config.__file__.endswith(str(Path("core") / "config" / "__init__.py")),
    f"core.config resolved to {config.__file__}, not the package __init__",
)


# --- 2. EVERY VALIDATOR IS REACHABLE BY NAME -------------------------------------
#
# The failure that happened twice. A star import drops underscore names, so a part
# added to the chain but omitted from the re-export walk loses its validators
# silently — they still RUN, they are just invisible.

declared: set[str] = set()
for part in sorted(p for p in PACKAGE.glob("_*.py") if p.stem != "__init__"):
    declared |= set(
        re.findall(r"^def (_validate\w+)", part.read_text(encoding="utf-8"), re.M)
    )
reachable = {name for name in dir(config) if name.startswith("_validate")}

check(
    bool(declared),
    "no validators were found in the package parts; the scan is broken",
)
missing = sorted(declared - reachable)
check(
    not missing,
    f"{len(missing)} validator(s) are defined in a part but unreachable from "
    f"core.config: {missing}. A star import drops underscore names at every "
    f"link, so a part added to the chain must also be added to the re-export walk",
)
check(
    len(reachable) >= 76,
    f"{len(reachable)} validators are reachable, expected at least 76; the "
    f"original flat module defined 76",
)


# --- 3. EVERY PART IS IN THE CHAIN -----------------------------------------------
#
# A part nobody imports is dead text: its constants would be absent and its
# validators would never run, so a contract breach it guards would go unnoticed.

# `_*.py` also matches `__init__.py`, which is the re-exporter rather than a
# part. A first version of this gate flagged __init__ for not naming itself.
parts = sorted(
    path.stem for path in PACKAGE.glob("_*.py") if path.stem != "__init__"
)
init_text = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
for part in parts:
    check(
        part in init_text,
        f"{part} is a config part but is not named in __init__.py, so its "
        f"validators never run and its constants are unreachable",
    )

chained = []
for part in parts:
    text = (PACKAGE / f"{part}.py").read_text(encoding="utf-8")
    imports = re.findall(r"from core\.config\.(\w+) import \*", text)
    chained.append((part, imports))

roots = [part for part, imports in chained if not imports]
check(
    len(roots) == 1,
    f"{len(roots)} parts import nothing ({roots}); the chain must have exactly "
    f"one root, or a part's constants are missing from the namespace",
)


# --- 4. THE CONSTANTS ARE UNCHANGED ----------------------------------------------
#
# Compared against the flat module as it stood before the split, read straight out
# of git rather than from a number recorded in a comment.

data_names = sorted(
    name
    for name in dir(config)
    if not name.startswith("__")
    and not callable(getattr(config, name))
    and not isinstance(getattr(config, name), types.ModuleType)
)
check(
    len(data_names) >= 1092,
    f"{len(data_names)} data constants are exported, expected at least 1,092; "
    f"the split must not have dropped any",
)

before = subprocess.run(
    ["git", "log", "--format=%H", "-1", "--", "core/config.py"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
)
commit = before.stdout.strip()
if commit:
    # BYTES, not text. `text=True` decodes with the locale codec, which on
    # Windows mangles UTF-8 — a first version of this gate reported
    # JOINT_REJECTED_RULES and SEQUENCE_ARCH_REGISTRY as "changed" because their
    # em-dashes came back as replacement characters. The split was byte-faithful;
    # the comparison was not.
    flat = subprocess.run(
        ["git", "show", f"{commit}:core/config.py"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if flat.returncode == 0 and flat.stdout:
        namespace: dict = {}
        try:
            exec(
                compile(
                    flat.stdout.decode("utf-8"), "config_before_split", "exec"
                ),
                namespace,
            )
        except Exception as error:  # noqa: BLE001 - reported, not raised
            failures.append(
                f"the pre-split config could not be executed for comparison: "
                f"{type(error).__name__}: {error}"
            )
        else:

            def canonical(value):
                try:
                    json.dumps(value)
                    return value
                except TypeError:
                    return repr(value)

            differing = [
                name
                for name in data_names
                if name in namespace
                and canonical(namespace[name]) != canonical(getattr(config, name))
            ]
            check(
                not differing,
                f"{len(differing)} constant(s) changed value across the split: "
                f"{differing[:8]}. C1 is a PURE refactor and must alter nothing",
            )
            dropped = sorted(
                name
                for name in namespace
                if not name.startswith("__") and not hasattr(config, name)
            )
            check(
                not dropped,
                f"{len(dropped)} name(s) exported before the split are now "
                f"missing: {dropped[:8]}",
            )


# --- 5. NO PART GROWS BACK ------------------------------------------------------
#
# `_forecasting` was 3,631 lines after the first split, which moved the
# navigability problem rather than solving it. This is the ratchet.

oversized = {}
for part in parts:
    count = len((PACKAGE / f"{part}.py").read_text(encoding="utf-8").splitlines())
    if count > MAX_PART_LINES:
        oversized[part] = count
check(
    not oversized,
    f"{oversized} exceed {MAX_PART_LINES} lines. The original file was 9,783 "
    f"lines and the point of C1 is that no part approaches it; split the part "
    f"rather than raising this bound",
)


# --- 6. THE COMMENTS SURVIVED ---------------------------------------------------
#
# They were 38% of the original text and carry the measurement behind each
# threshold. A "tidy-up" that trimmed them would leave numbers nobody can
# re-derive, and that loss would not fail any other check.

total = 0
comments = 0
for part in parts:
    for line in (PACKAGE / f"{part}.py").read_text(encoding="utf-8").splitlines():
        total += 1
        if line.strip().startswith("#"):
            comments += 1
share = comments / total if total else 0.0
check(
    share >= 0.30,
    f"comments are {share:.0%} of the config package, below 30%. They were 38% "
    f"before the split and carry the measurement that justifies each threshold; "
    f"losing them leaves numbers nobody can re-derive",
)


if failures:
    print("C1 CONFIG PACKAGE GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("C1 config package gate: OK")
print(f"  parts                             {len(parts)}")
print(f"  largest part                      "
      f"{max(len((PACKAGE / f'{p}.py').read_text(encoding='utf-8').splitlines()) for p in parts)}"
      f" lines  (was 9,783)")
print(f"  data constants exported           {len(data_names)}")
print(f"  validators reachable by name      {len(reachable)}")
print(f"  comment share                     {share:.0%}  (was 38%)")
sys.exit(0)
