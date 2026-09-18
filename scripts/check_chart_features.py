"""CI drift gate for the C2 price/volume feature expansion.

Proves, on every push, that the feature contract C5/C6 will depend on holds:

1. every C2 feature is REGISTERED in M1 — an unregistered feature cannot
   enter a production model, and the producer must exist on disk;
2. insufficient history yields None, never a neutral zero that would be
   indistinguishable from a real reading once it reaches a training row;
3. a withheld feature is NAMED in `insufficient_history`, so a consumer can
   tell "no data" from "computed, and the answer is zero";
4. relative strength refuses to answer without an INJECTED benchmark — C2
   never selects or fabricates one (that is C3's contract);
5. a failed breakout stays distinguishable from a holding one;
6. the existing market features were NOT re-owned — no second implementation
   of an already-canonical quantity (W5);
7. the surface is deterministic.

Synthetic only, so it runs in well under a second and needs no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from core.chart_features import (  # noqa: E402
    breakout_state,
    compute_chart_features,
    relative_strength,
)
from core.config import (  # noqa: E402
    CHART_BREAKOUT_FAILED_UP,
    CHART_BREAKOUT_UP,
    CHART_FEATURE_MIN_HISTORY,
)
from core.feature_registry import build_default_registry, producer_problems  # noqa: E402

# Already canonical elsewhere; C2 must not re-own them.
_MARKET_OWNED = (
    "change_1d", "change_5d", "change_20d", "change_60d",
    "rsi", "volatility", "atr_14", "volume_ratio_20d", "trend_slope_60d",
)


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


def main() -> int:
    failures: list[str] = []
    registry = build_default_registry()._features

    # 1. registration + real producer
    for name in CHART_FEATURE_MIN_HISTORY:
        if name not in registry:
            failures.append(f"C2 feature {name!r} is not registered — a model could not consume it")
            continue
        spec = registry[name]
        if spec.owner != "chart_feature_agent":
            failures.append(f"{name!r} declares owner {spec.owner!r}, expected chart_feature_agent")
        if spec.minimum_history != CHART_FEATURE_MIN_HISTORY[name]:
            failures.append(
                f"{name!r} registry minimum_history={spec.minimum_history} disagrees with the "
                f"producer's {CHART_FEATURE_MIN_HISTORY[name]}"
            )
        if spec.null_policy != "exclude":
            failures.append(
                f"{name!r} null_policy={spec.null_policy!r} — a withheld feature must be "
                f"excluded, never replaced by a neutral default"
            )
    for problem in producer_problems("chart_feature_agent"):
        failures.append(f"producer problem: {problem}")

    # 2 + 3. thin history withholds, and says so
    thin = compute_chart_features(_frame([100.0, 101.0, 102.0]))
    for name in ("drawdown_60d", "recovery_speed_60d", "breakout_state_60d"):
        if thin["features"][name] is not None:
            failures.append(f"{name!r} fabricated a value from three bars")
        if name not in thin["insufficient_history"]:
            failures.append(f"{name!r} was withheld without being named in insufficient_history")

    # 4. no benchmark, no answer
    rich = _frame([100.0 + i for i in range(120)])
    if relative_strength(rich, None) is not None:
        failures.append(
            "relative strength answered without a benchmark — C2 must never select "
            "or fabricate one (C3 owns benchmark selection)"
        )
    matched = relative_strength(rich, _frame([100.0 + i for i in range(120)]))
    if matched is None or abs(matched) > 1e-6:
        failures.append(
            f"a stock matching its benchmark must read ~0 relative strength (got {matched})"
        )

    # 5. failed vs holding breakout
    holding = breakout_state(_frame([100.0] * 60 + [120.0] * 5))
    failed = breakout_state(
        _frame([100.0 + (i % 5) for i in range(60)] + [120.0, 118.0, 110.0, 103.0, 102.0])
    )
    if holding != CHART_BREAKOUT_UP:
        failures.append(f"a holding break must read {CHART_BREAKOUT_UP} (got {holding})")
    if failed != CHART_BREAKOUT_FAILED_UP:
        failures.append(
            f"a break that closed back inside must read {CHART_BREAKOUT_FAILED_UP} "
            f"(got {failed}) — the distinction this feature exists for"
        )

    # 6. W5: no re-owned features
    for name in _MARKET_OWNED:
        if name in registry and registry[name].owner != "market_data_agent":
            failures.append(
                f"{name!r} was re-owned by {registry[name].owner!r} — a second "
                f"implementation of a canonical feature (W5 violation)"
            )

    # 7. determinism
    frame = _frame([100.0 + (i % 13) for i in range(150)])
    if compute_chart_features(frame) != compute_chart_features(frame):
        failures.append("the chart feature surface is not deterministic")

    if failures:
        print("C2 chart-feature gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("C2 chart-feature gate OK:")
    print(f"  all {len(CHART_FEATURE_MIN_HISTORY)} C2 features registered to a producer that exists.")
    print("  thin history withholds rather than zero-fills, and names what it withheld.")
    print("  relative strength refuses an absent benchmark; a matched benchmark reads ~0.")
    print("  failed breakouts stay distinguishable from holding ones.")
    print(f"  the {len(_MARKET_OWNED)} pre-existing market features were not re-owned (W5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
