"""CI drift gate for the C7 chart reaction memory.

Proves, on every push, that the memory an analog lookup will trust holds:

1. every declared horizon answers — 1h through 60d;
2. the 1h channel is never INVENTED: no hourly frame means UNAVAILABLE, and an
   event predating the hourly window is refused rather than silently matching
   the first bar (the live bug: `index >= target` is true for every bar, so
   argmax picked bar 0 and reported an unrelated hour months later);
3. an absent INTRADAY horizon does not degrade the memory, because for any
   event old enough to have a 60d reaction the hourly data cannot exist;
   an absent DAILY horizon does;
4. the before-picture EXCLUDES the event bar — including it would let the
   reaction describe its own setup;
5. E4's reactions are consumed, not recomputed (W5);
6. retrieval matches STRUCTURE first: same numbers with a different phase is
   not an analog;
7. a thin analog set refuses a median rather than smoothing an anecdote into a
   base rate;
8. the memory hash is deterministic and reaction-sensitive.

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

import core.reaction_memory as memory_module  # noqa: E402
from core.config import (  # noqa: E402
    REACTION_HORIZONS,
    REACTION_INTRADAY_HORIZONS,
    REACTION_MIN_ANALOGS,
    REACTION_PRE_EVENT_SESSIONS,
    REACTION_REQUIRED_HORIZONS,
    REACTION_STATUS_INCOMPLETE,
    REACTION_STATUS_OK,
    REACTION_STATUS_UNAVAILABLE,
)
from core.reaction_memory import (  # noqa: E402
    analog_response,
    before_picture,
    before_similarity,
    build_reaction_memory,
    find_reaction_analogs,
    hourly_reaction,
    memory_problems,
)

EVENT = "2026-06-01"

# E4 owns these. A `def` here would be a second implementation.
_E4_OWNED = ("def run_event_study(", "def _abnormal_return(", "def _baseline_stats(")


def _daily(n=200, start="2025-09-01"):
    closes = []
    for i in range(n):
        leg = (i % 14) / 14
        wave = 7.0 * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(100.0 + wave + 0.15 * i)
    index = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.012 for c in closes],
            "Low": [c * 0.988 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n,
        },
        index=index,
    )


def _hourly(n=40, start="2026-06-01 13:30:00"):
    closes = [100.0 + 0.5 * i for i in range(n)]
    index = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes],
            "Close": closes,
            "Volume": [10_000] * n,
        },
        index=index,
    )


def _study(horizons=("intraday", "1d", "5d", "20d", "60d"), value=0.02):
    return {
        "status": "OK",
        "reactions": {
            h: {"horizon": h, "stock_return": value, "abnormal_return": value / 2}
            for h in horizons
        },
    }


def _code_without_docstrings(module) -> str:
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

    memory = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study())

    # 1. every horizon answers
    for horizon in REACTION_HORIZONS:
        if horizon not in memory["reactions"]:
            failures.append(f"horizon {horizon!r} absent — C7 requires every horizon to answer")
    for problem in memory_problems(memory):
        failures.append(f"contract problem: {problem}")

    # 2. 1h is never invented
    if hourly_reaction(None, EVENT)["status"] != REACTION_STATUS_UNAVAILABLE:
        failures.append("a missing hourly frame did not read UNAVAILABLE")
    stale = hourly_reaction(_hourly(start="2026-09-01 13:30:00"), "2026-03-02")
    if stale["status"] != REACTION_STATUS_UNAVAILABLE:
        failures.append(
            "an event predating the hourly window produced a reaction — argmax "
            "matched the first bar and measured an unrelated hour months later"
        )
    live = hourly_reaction(_hourly(), EVENT + " 13:30:00")
    if live["status"] != REACTION_STATUS_OK or live["stock_return"] is None:
        failures.append("a recent event failed to measure an hourly reaction")

    # 3. intraday absence is tolerated; daily absence is not
    if memory["reactions"]["1h"]["status"] != REACTION_STATUS_UNAVAILABLE:
        failures.append("1h should be UNAVAILABLE without an hourly frame")
    if memory["status"] != REACTION_STATUS_OK:
        failures.append(
            f"a memory missing only intraday horizons must stay OK (got {memory['status']}) — "
            f"that data cannot exist for an older event"
        )
    thin = build_reaction_memory("TEST", EVENT, _daily(), event_study=_study(horizons=("1d", "5d")))
    if thin["status"] != REACTION_STATUS_INCOMPLETE:
        failures.append("a memory missing a REQUIRED daily horizon must be INCOMPLETE")
    for horizon in REACTION_INTRADAY_HORIZONS:
        if horizon in REACTION_REQUIRED_HORIZONS:
            failures.append(f"{horizon!r} is both intraday-optional and required")

    # 4. the before-picture excludes the event bar
    before = before_picture(_daily(), EVENT)
    if before["status"] != REACTION_STATUS_OK:
        failures.append("the before-picture could not be built from ample history")
    elif pd.Timestamp(before["last_bar"]) >= pd.Timestamp(EVENT):
        failures.append(
            "the before-picture includes the event bar — the reaction would be "
            "describing its own setup"
        )
    if before.get("bars", 0) > REACTION_PRE_EVENT_SESSIONS:
        failures.append("the before-window exceeded its declared bound")
    if not before.get("phase"):
        failures.append("the before-picture carries no structural phase (C4 not consumed)")

    # 5. W5: E4 consumed, not recomputed
    code = _code_without_docstrings(memory_module)
    for owned in _E4_OWNED:
        if owned in code:
            failures.append(
                f"core.reaction_memory defines {owned.strip('def (')!r} — that is "
                f"owned by E4's event study and must be consumed (W5)"
            )
    if build_reaction_memory("TEST", EVENT, _daily(), event_study=_study(value=0.077))[
        "reactions"]["5d"]["stock_return"] != 0.077:
        failures.append("E4's measured reaction was not carried through unchanged")

    # 6. structure-first retrieval
    setup = {"phase": "breakout", "consolidation_range": 0.05,
             "volatility_regime_ratio": 1.1, "gap_pct": 0.0}
    same_phase_drifted = {**setup, "consolidation_range": 0.07}
    other_phase_identical = {**setup, "phase": "failed_breakout"}
    if before_similarity(setup, same_phase_drifted) <= before_similarity(setup, other_phase_identical):
        failures.append(
            "a phase mismatch costs no more than numeric drift — two charts with "
            "the same numbers and different phases are not analogs"
        )

    def _mem(phase, value):
        return {
            "ticker": "T", "event_time": EVENT, "status": REACTION_STATUS_OK,
            "before": {"status": REACTION_STATUS_OK, "phase": phase,
                       "consolidation_range": 0.05, "volatility_regime_ratio": 1.1,
                       "gap_pct": 0.0},
            "reactions": {"20d": {"status": REACTION_STATUS_OK, "stock_return": value}},
        }

    analogs = find_reaction_analogs(setup, [_mem("breakout", 0.05), _mem("consolidation", -0.03)])
    if len(analogs) != 1 or analogs[0]["before"]["phase"] != "breakout":
        failures.append("retrieval did not filter on the structural phase")

    # 7. a thin analog set refuses a median
    few = [{"reactions": {"20d": {"status": REACTION_STATUS_OK, "stock_return": 0.02}}}] * (
        REACTION_MIN_ANALOGS - 1
    )
    thin_response = analog_response(few, "20d")
    if thin_response["sufficient"] or thin_response["median_response"] is not None:
        failures.append(
            f"{REACTION_MIN_ANALOGS - 1} observation(s) produced a median — that is "
            f"an anecdote dressed as a base rate"
        )
    enough = [{"reactions": {"20d": {"status": REACTION_STATUS_OK, "stock_return": v}}}
              for v in (0.02, 0.05, 0.08, -0.01)]
    if not analog_response(enough, "20d")["sufficient"]:
        failures.append("a sufficient analog set was refused a median")

    # 8. hashing
    if memory["memory_hash"] != build_reaction_memory(
        "TEST", EVENT, _daily(), event_study=_study())["memory_hash"]:
        failures.append("the memory hash is not deterministic")
    if memory["memory_hash"] == build_reaction_memory(
        "TEST", EVENT, _daily(), event_study=_study(value=0.09))["memory_hash"]:
        failures.append("a different reaction produced the same hash")

    if failures:
        print("C7 reaction-memory gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C7 reaction-memory gate OK:")
    print(f"  all {len(REACTION_HORIZONS)} horizons answer: {', '.join(REACTION_HORIZONS)}.")
    print("  1h is never invented; an event predating the hourly window is refused.")
    print("  intraday absence tolerated, daily absence is INCOMPLETE.")
    print("  the before-picture is C4-structural and excludes the event bar.")
    print("  E4 reactions consumed not recomputed; retrieval matches phase first.")
    print(f"  fewer than {REACTION_MIN_ANALOGS} analogs refuses a median.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
