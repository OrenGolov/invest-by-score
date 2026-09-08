"""Versioned execution cost model (Sprint V2).

The cost table is a versioned governance artifact — parameters live here,
never inline in the engine. Every execution pays:

- half the spread of its liquidity bucket (full spread round trip),
- square-root market-impact slippage: bps = impact_coefficient * sqrt(participation),
- commission in bps.

Fail-closed: unknown liquidity pays the widest bucket; unknown
participation is treated as the worst case (1.0 = the order is the whole
day's volume).
"""

from __future__ import annotations

import math

COST_TABLE_VERSION = "backtest-cost-table-v1"

COST_TABLE_V1 = {
    "spread_bps": {"micro": 25.0, "small": 12.0, "mid": 6.0, "large": 2.0},
    "liquidity_buckets_dollars": {
        "micro": 1_000_000.0,
        "small": 10_000_000.0,
        "mid": 100_000_000.0,
    },
    "impact_coefficient_bps": 50.0,
    "commission_bps": 0.5,
}

SIDES = ("buy", "sell")


def liquidity_bucket(avg_dollar_volume: float | None, table: dict | None = None) -> str:
    """Bucket a 20-session average dollar volume; unknown -> widest spread."""
    table = table if table is not None else COST_TABLE_V1
    thresholds = table["liquidity_buckets_dollars"]
    if avg_dollar_volume is None:
        return "micro"
    if avg_dollar_volume < thresholds["micro"]:
        return "micro"
    if avg_dollar_volume < thresholds["small"]:
        return "small"
    if avg_dollar_volume < thresholds["mid"]:
        return "mid"
    return "large"


def total_side_cost_bps(
    participation: float | None,
    avg_dollar_volume: float | None,
    table: dict | None = None,
) -> dict:
    """Per-side execution cost in bps, decomposed for the run record.

    participation = order notional / avg dollar volume (worst case 1.0 when
    unknown). Square-root law: impact_bps = coefficient * sqrt(participation).
    """
    table = table if table is not None else COST_TABLE_V1
    bucket = liquidity_bucket(avg_dollar_volume, table)
    half_spread_bps = table["spread_bps"][bucket] / 2.0
    participation_value = 1.0 if participation is None else max(0.0, float(participation))
    impact_bps = table["impact_coefficient_bps"] * math.sqrt(participation_value)
    commission_bps = table["commission_bps"]
    return {
        "bucket": bucket,
        "half_spread_bps": round(half_spread_bps, 4),
        "impact_bps": round(impact_bps, 4),
        "commission_bps": commission_bps,
        "total_bps": round(half_spread_bps + impact_bps + commission_bps, 4),
    }


def execution_price(
    open_price: float,
    side: str,
    participation: float | None,
    avg_dollar_volume: float | None,
    table: dict | None = None,
) -> float:
    """Executed price for one side: open adjusted by the side's cost in bps.

    Buys pay the costs (price moves against them); sells receive less.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if open_price <= 0:
        raise ValueError(f"open_price must be positive, got {open_price!r}")
    costs = total_side_cost_bps(participation, avg_dollar_volume, table)
    bps = costs["total_bps"]
    if side == "buy":
        return round(open_price * (1.0 + bps / 10_000.0), 6)
    return round(open_price * (1.0 - bps / 10_000.0), 6)
