"""CI drift gate for the C4 deterministic chart structure.

Proves, on every push, that the structure contract holds:

1. no pivot is claimed in the unconfirmable tail — a fractal swing needs k
   bars AFTER it, so scanning to the final bar would let future bars decide a
   past label (the C1 still-forming-bar leak, in swing form);
2. the unconfirmed tail is REPORTED, so a consumer can see that the most
   recent action is not yet structural;
3. a confirmed pivot is settled — appending future bars cannot revise it;
4. phase precedence is the declared one, and every resolution lands in
   STRUCTURE_PHASES;
5. `undefined` always explains itself, or it is indistinguishable from a
   failure to compute;
6. C2's features are CONSUMED, not recomputed (W5);
7. the description is deterministic.

Synthetic only, so it runs in well under a second and needs no network.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import core.chart_structure as structure_module  # noqa: E402
from core.chart_structure import (  # noqa: E402
    describe_structure,
    resolve_phase,
    structure_problems,
    swing_points,
)
from core.config import (  # noqa: E402
    CHART_STRUCTURE_MIN_BARS,
    CHART_SWING_FRACTAL_K,
    STRUCTURE_PHASE_BREAKOUT,
    STRUCTURE_PHASE_CONSOLIDATION,
    STRUCTURE_PHASE_FAILED_BREAKOUT,
    STRUCTURE_PHASE_REVERSAL,
    STRUCTURE_PHASE_UNDEFINED,
    STRUCTURE_PHASES,
    STRUCTURE_SWING_LABELS,
    STRUCTURE_SWING_UPTREND,
)

# These belong to C2's chart_feature_agent. A `def` here would be a second
# implementation of an already-canonical quantity.
_C2_OWNED = ("def breakout_state(", "def gap_pct(", "def volatility_regime_ratio(")


def _frame(closes, start="2026-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(closes),
        },
        index=index,
    )


def _zigzag(n, amplitude=10.0, period=10, drift=0.0):
    closes = []
    for i in range(n):
        leg = (i % period) / period
        wave = amplitude * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(100.0 + wave + drift * i)
    return closes


def main() -> int:
    failures: list[str] = []

    frame = _frame(_zigzag(160, drift=0.2))
    highs, lows = swing_points(frame)
    last_confirmable = len(frame) - 1 - CHART_SWING_FRACTAL_K

    # 1. nothing claimed in the unconfirmable tail
    for pivot in highs + lows:
        if pivot["index"] > last_confirmable:
            failures.append(
                f"a pivot was claimed at index {pivot['index']} — beyond the last "
                f"confirmable bar {last_confirmable}; future bars would be deciding "
                f"a past label"
            )
            break
        if pivot["index"] < CHART_SWING_FRACTAL_K:
            failures.append(
                f"a pivot was claimed at index {pivot['index']} before its window opens"
            )
            break

    # 2. the tail is reported
    described = describe_structure(frame)
    if described["unconfirmed_tail_bars"] != CHART_SWING_FRACTAL_K:
        failures.append(
            "a described structure must declare its unconfirmed tail so the most "
            "recent action is visibly not yet structural"
        )
    for problem in structure_problems(described):
        failures.append(f"contract problem: {problem}")

    # 3. a confirmed pivot is settled
    closes = _zigzag(160, drift=0.2)
    before = {p["index"]: p["price"] for p in swing_points(_frame(closes))[0]}
    after = {p["index"]: p["price"] for p in swing_points(_frame(closes + [900.0] * 10))[0]}
    for index, price in before.items():
        if index <= len(closes) - 1 - 2 * CHART_SWING_FRACTAL_K and after.get(index) != price:
            failures.append(
                f"pivot at index {index} was revised by future bars — a confirmed "
                f"pivot must be settled"
            )
            break

    # 4. declared precedence
    precedence = (
        (("failed_breakout_up", "reversal_up", True, STRUCTURE_SWING_UPTREND), STRUCTURE_PHASE_FAILED_BREAKOUT),
        (("breakout_up", "reversal_up", True, STRUCTURE_SWING_UPTREND), STRUCTURE_PHASE_BREAKOUT),
        (("none", "reversal_down", True, STRUCTURE_SWING_UPTREND), STRUCTURE_PHASE_REVERSAL),
        (("none", "none", True, STRUCTURE_SWING_UPTREND), STRUCTURE_PHASE_CONSOLIDATION),
    )
    for args, expected in precedence:
        actual = resolve_phase(*args)
        if actual != expected:
            failures.append(f"precedence broke: resolve_phase{args} gave {actual!r}, expected {expected!r}")

    for breakout in ("none", "breakout_up", "failed_breakout_down"):
        for reversal in ("none", "reversal_up"):
            for consolidating in (True, False):
                for swing in STRUCTURE_SWING_LABELS + (None,):
                    if resolve_phase(breakout, reversal, consolidating, swing) not in STRUCTURE_PHASES:
                        failures.append("resolve_phase produced a phase outside STRUCTURE_PHASES")
                        break

    # 5. undefined explains itself
    thin = describe_structure(_frame([100.0 + i for i in range(30)]))
    if thin["phase"] != STRUCTURE_PHASE_UNDEFINED or not thin["reason"]:
        failures.append(
            f"thin history ({CHART_STRUCTURE_MIN_BARS} required) must be undefined WITH a reason"
        )
    straight = describe_structure(_frame([100.0 + i * 0.8 for i in range(150)] ))
    if straight["phase"] == STRUCTURE_PHASE_UNDEFINED and not straight["reason"]:
        failures.append("an undefined phase was produced with no explanation")

    # 6. W5: C2 features consumed, not recomputed
    source = inspect.getsource(structure_module)
    for owned in _C2_OWNED:
        if owned in source:
            failures.append(
                f"core.chart_structure defines {owned.strip('def (')!r} — that quantity "
                f"is owned by C2's chart_feature_agent and must be consumed (W5)"
            )

    # 7. determinism
    if describe_structure(frame) != describe_structure(frame):
        failures.append("the structure description is not deterministic")

    if failures:
        print("C4 chart-structure gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C4 chart-structure gate OK:")
    print(f"  no pivot claimed in the {CHART_SWING_FRACTAL_K}-bar unconfirmable tail; the tail is reported.")
    print("  a confirmed pivot is settled — future bars cannot revise it.")
    print(f"  phase precedence declared and total over {len(STRUCTURE_PHASES)} phases.")
    print("  undefined always explains itself; thin history never gets a confident label.")
    print("  C2's breakout/gap/volatility features are consumed, not recomputed (W5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
