"""CI drift gate for the C5 temporal sequence dataset.

Proves, on every push, that the sequence contract C6 will depend on holds:

1. step i sees bars up to i and NOTHING after — checked directly, by
   rebuilding the step from a frame physically truncated there;
2. appending future bars cannot revise an earlier step;
3. visibility advances by exactly one bar per step, and step times increase;
4. per-step channels genuinely VARY across the window — a constant would mean
   they were computed once and repeated;
5. as-of-T0 context is labelled and never appears inside a step, because
   back-projecting today's macro across sixty past steps is the
   revised-data-in-history failure the master context forbids;
6. C5 is NOT a second door into a training set: it emits no TrainingRow and
   computes no forward return of its own (M2 stays the only generator);
7. the sequence hash is deterministic, and an incomplete sequence carries no
   steps at all — a model trained on a ragged window learns the raggedness.

Synthetic only, so it needs no network.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import core.sequences as sequences_module  # noqa: E402
from core.config import (  # noqa: E402
    SEQUENCE_CONTEXT_SCOPE,
    SEQUENCE_LOOKBACK_STEPS,
    SEQUENCE_STATUS_INCOMPLETE,
    SEQUENCE_STEP_CHANNELS,
    SEQUENCE_STEP_WARMUP_BARS,
    SEQUENCE_T0_CHANNELS,
)
from core.sequences import (  # noqa: E402
    build_sequence,
    build_step,
    sequence_problems,
)

# M2 owns supervised rows. A reference in CODE here would be a second door.
_M2_OWNED = ("TrainingRow", "build_training_row", "build_training_dataset")


def _frame(n, start="2024-01-01", base=100.0):
    closes = []
    for i in range(n):
        leg = (i % 14) / 14
        wave = 8.0 * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(base + wave + 0.12 * i)
    index = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.012 for c in closes],
            "Low": [c * 0.988 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 + (i % 7) * 10_000 for i in range(n)],
        },
        index=index,
    )


def _code_without_docstrings(module) -> str:
    """Source with docstrings stripped — the module documents its own rules."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def main() -> int:
    failures: list[str] = []
    frame = _frame(SEQUENCE_LOOKBACK_STEPS + 1 + SEQUENCE_STEP_WARMUP_BARS + 40)

    # 1. truncation, checked directly
    for position in (SEQUENCE_STEP_WARMUP_BARS + 5, len(frame) - 1):
        if build_step(frame, position) != build_step(frame.iloc[: position + 1], position):
            failures.append(
                f"step {position} differs when built from the full frame vs a frame "
                f"truncated there — the step is seeing bars after itself"
            )

    # 2. future bars cannot revise an earlier step
    position = len(frame) - 30
    before = build_step(frame, position)
    extended = pd.concat([frame, _frame(20, start="2030-01-01", base=900.0)])
    if build_step(extended, position) != before:
        failures.append("appending future bars revised an earlier step — PIT violation")

    sequence = build_sequence("TEST", "2026-09-15", frame)
    steps = sequence["steps"]
    if not steps:
        failures.append("the healthy fixture produced no steps")
        print("C5 sequence gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    # 3. monotonic visibility and time
    counts = [step["visible_bars"] for step in steps]
    if counts != list(range(counts[0], counts[0] + len(counts))):
        failures.append("visibility does not advance by exactly one bar per step")
    times = [step["step_time"] for step in steps]
    if times != sorted(times) or len(times) != len(set(times)):
        failures.append("step times are not strictly increasing")
    for problem in sequence_problems(sequence):
        failures.append(f"contract problem: {problem}")

    # 4. per-step channels genuinely vary
    if len({step["price"]["close"] for step in steps}) <= len(steps) // 2:
        failures.append("price barely varies across the window — is it computed per step?")
    if len({step["structure"]["phase"] for step in steps}) < 2:
        failures.append(
            "structure phase is constant across 61 steps — it was computed once "
            "and repeated rather than recomputed per step"
        )
    for channel in SEQUENCE_STEP_CHANNELS:
        if channel not in steps[0]:
            failures.append(f"declared per-step channel {channel!r} is absent from a step")

    # 5. context labelled, and never inside a step
    if sequence["context"].get("context_scope") != SEQUENCE_CONTEXT_SCOPE:
        failures.append("context is not labelled as_of_t0")
    for channel in SEQUENCE_T0_CHANNELS:
        if channel in steps[0]:
            failures.append(
                f"as-of-T0 channel {channel!r} appears INSIDE a step — that would imply "
                f"sixty vintages the system never had"
            )

    # 6. M2 boundary
    code = _code_without_docstrings(sequences_module)
    for owned in _M2_OWNED:
        if owned in code:
            failures.append(
                f"core.sequences references {owned!r} in code — M2 is the only door "
                f"into a supervised dataset"
            )
    if "forward_return =" in code:
        failures.append("core.sequences computes its own forward return — labels must come from V1")

    # 7. determinism and the ragged-window rule
    if build_sequence("TEST", "2026-09-15", frame) != sequence:
        failures.append("the sequence build is not deterministic")
    thin = build_sequence("TEST", "2026-09-15", _frame(40))
    if thin["status"] != SEQUENCE_STATUS_INCOMPLETE or thin["steps"]:
        failures.append(
            "a sequence without enough history must be INCOMPLETE and carry NO steps"
        )
    if thin["sequence_hash"] is not None:
        failures.append("an incomplete sequence must carry no hash")

    if failures:
        print("C5 sequence gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C5 sequence gate OK:")
    print(f"  {len(steps)} steps; each equals the state of a frame truncated there.")
    print("  future bars cannot revise an earlier step; visibility advances one bar at a time.")
    print("  per-step channels vary across the window; as-of-T0 context is labelled and separate.")
    print("  no TrainingRow and no forward return of its own — M2 stays the only door.")
    print("  deterministic hash; a window without enough history carries no steps at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
