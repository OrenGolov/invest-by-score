"""CI drift gate for the E5 confounder / attribution engine.

Proves, on every push, that the decomposition stays honest:

1. it is ADDITIVE — stock = market + sector_excess + stock_specific — across
   many random shapes, not just one hand-picked example;
2. the sector component is EXCESS over market, so market beta is never
   counted twice and the error never lands in the residual;
3. a residual must clear BOTH bars: share of movement AND magnitude in
   baseline sigma. A large share of a tiny move is noise;
4. overlapping events confound by default;
5. missing market context yields `inconclusive`, never a flattering
   `event_associated`;
6. no verdict ever claims causation.

Synthetic only; runs in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.attribution import (  # noqa: E402
    attribute,
    attribution_report,
    decompose,
    find_overlapping_events,
)
from core.config import (  # noqa: E402
    ATTRIBUTION_CONFOUNDED,
    ATTRIBUTION_EVENT_ASSOCIATED,
    ATTRIBUTION_INCONCLUSIVE,
    ATTRIBUTION_MIN_RESIDUAL_SHARE,
    ATTRIBUTION_MIN_RESIDUAL_SIGMA,
    ATTRIBUTION_UNEXPLAINED,
)
from core.event_contract import Event  # noqa: E402


def main() -> int:
    failures: list[str] = []

    # 1. Additivity across many shapes, not one example.
    rng = np.random.default_rng(11)
    non_additive = 0
    for _ in range(200):
        stock, market, sector = rng.normal(0.0, 0.05, 3)
        if not decompose(stock, market, sector, "5d").is_additive():
            non_additive += 1
    if non_additive:
        failures.append(
            f"{non_additive} of 200 random decompositions are not additive — "
            f"the residual is absorbing arithmetic error"
        )

    # 2. The sector component must be EXCESS over market.
    decomposition = decompose(0.050, 0.020, 0.035, "5d")
    if abs(decomposition.sector_excess - (0.035 - 0.020)) > 1e-9:
        failures.append(
            "the sector component is not excess over market — a sector ETF "
            "already contains market beta, so the raw return double-counts it"
        )
    naive_residual = 0.050 - 0.020 - 0.035
    if abs(decomposition.stock_specific - naive_residual) < 1e-9:
        failures.append(
            "the residual equals the double-counted value, so market beta is "
            "being subtracted twice"
        )

    # A missing market means no decomposition at all.
    no_market = decompose(0.050, None, None, "5d")
    if no_market.components_available or no_market.stock_specific is not None:
        failures.append(
            "a decomposition was produced without a market return — the residual "
            "would be the raw return under a different name"
        )

    # 3. Both bars, independently.
    market_driven = attribute(
        decompose(0.021, 0.020, 0.020, "5d", baseline_volatility=0.015, sessions=5)
    )
    if market_driven.verdict != ATTRIBUTION_CONFOUNDED:
        failures.append(
            f"a market-driven move was judged {market_driven.verdict!r}, expected "
            f"confounded"
        )

    strong = attribute(
        decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015, sessions=5)
    )
    if strong.verdict != ATTRIBUTION_EVENT_ASSOCIATED:
        failures.append(
            f"a large company-specific move was judged {strong.verdict!r} — the "
            f"rule can never pass, which is indistinguishable from broken"
        )

    tiny = attribute(
        decompose(0.0012, 0.0001, 0.0002, "5d", baseline_volatility=0.015, sessions=5)
    )
    if tiny.verdict != ATTRIBUTION_UNEXPLAINED:
        failures.append(
            f"a large share of a TINY move was judged {tiny.verdict!r} — share "
            f"alone must not be enough"
        )

    no_baseline = attribute(decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=None))
    if no_baseline.verdict == ATTRIBUTION_EVENT_ASSOCIATED:
        failures.append(
            "a residual was called event-associated without a baseline volatility "
            "to size it against"
        )

    # 4. Overlapping events confound.
    confounded = attribute(
        decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015, sessions=5),
        overlapping_events=["evt-2"],
    )
    if confounded.verdict != ATTRIBUTION_CONFOUNDED:
        failures.append(
            "an overlapping event did not confound the attribution — that is the "
            "most common confounder in an event study"
        )
    if "evt-2" not in confounded.overlapping_events:
        failures.append("the confounding event was not named in the result")

    first = Event(
        entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
        source="s", evidence=[{"source_record_id": "r1"}],
    )
    near = Event(
        entity="NVDA", published_time="2026-01-08 00:00:00", event_type="guidance",
        source="s", evidence=[{"source_record_id": "r2"}],
    )
    far = Event(
        entity="NVDA", published_time="2026-09-08 00:00:00", event_type="guidance",
        source="s", evidence=[{"source_record_id": "r3"}],
    )
    other_company = Event(
        entity="AMD", published_time="2026-01-06 00:00:00", event_type="earnings",
        source="s", evidence=[{"source_record_id": "r4"}],
    )
    overlaps = find_overlapping_events(first, [first, near, far, other_company], sessions=20)
    if near.event_id not in overlaps:
        failures.append("an event inside the window was not detected as overlapping")
    if far.event_id in overlaps:
        failures.append("an event outside the window was treated as overlapping")
    if other_company.event_id in overlaps:
        failures.append("another company's news was treated as a confounder")
    if first.event_id in overlaps:
        failures.append("an event was treated as confounding itself")

    # 5. Missing context is inconclusive.
    inconclusive = attribute(decompose(0.090, None, None, "5d"))
    if inconclusive.verdict != ATTRIBUTION_INCONCLUSIVE:
        failures.append(
            f"a decomposition with no market context was judged "
            f"{inconclusive.verdict!r}, expected inconclusive"
        )
    if inconclusive.is_event_associated():
        failures.append("an unattributable movement was called event-associated")

    zero = attribute(decompose(0.0, 0.0, 0.0, "5d", baseline_volatility=0.015))
    if zero.verdict != ATTRIBUTION_INCONCLUSIVE:
        failures.append("a zero movement was attributed to something")

    # 6. Never claims causation.
    for attribution, label in (
        (strong, "event_associated"), (market_driven, "confounded"),
        (inconclusive, "inconclusive"),
    ):
        if "NOT event-caused" not in attribution.causality_note:
            failures.append(f"the {label} verdict does not disclaim causation")
        if "caused" in attribution.reason:
            failures.append(f"the {label} reason claims causation")
    if "NOT event-caused" not in attribution_report([]).get("causality_note", ""):
        failures.append("the attribution report does not carry the causality note")

    report = attribution_report([strong, market_driven, tiny])
    if report["event_associated"] != 1:
        failures.append(f"the report counted {report['event_associated']} associations, expected 1")

    if failures:
        print("E5 attribution gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E5 attribution gate OK:")
    print("  additive over 200 random shapes; sector component is excess over market.")
    print(
        f"  a residual needs >={ATTRIBUTION_MIN_RESIDUAL_SHARE:.0%} share AND "
        f">={ATTRIBUTION_MIN_RESIDUAL_SIGMA} baseline sigma."
    )
    print("  overlapping events confound; missing market context is inconclusive.")
    print("  no verdict claims causation — association only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
