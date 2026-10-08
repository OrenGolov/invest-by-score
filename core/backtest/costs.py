"""Versioned execution cost model (Sprint V2/V3).

Transaction costs and slippage assumptions are FIRST-CLASS, VERSIONED
ARTIFACTS — parameters live in the cost table, never inline in the engine,
and every execution emits an itemized cost record (see
`execution_cost_record`) so no cost is ever hidden inside a price.

Per-side execution cost (V2 model):

    total_bps = clamp(
        half_spread(bucket)
        + impact_coefficient_bps * vol_factor * sqrt(participation)
        + commission_bps,
        min_total_side_cost_bps, max_total_side_cost_bps)

    vol_factor = clamp(daily_vol / impact_vol_baseline_daily,
                       impact_vol_factor_min, impact_vol_factor_max)

- `bucket` comes from the 20-session average dollar volume (micro / small /
  mid / large). Fail-closed: unknown liquidity pays the widest spread.
- `participation` = order notional / average dollar volume. Fail-closed:
  unknown participation is treated as the worst case (1.0 = the order is
  the whole day's volume).
- `daily_vol` = realized daily volatility of the trailing window. The
  square-root law scales impact with volatility; when volatility is
  unavailable the neutral factor 1.0 is used and the record says so —
  the engine always passes a computed value for live replays.
- The clamp makes degenerate outcomes impossible: no zero-cost fill, no
  absurd-cost fill.

COST_TABLE_V1 is frozen for reference (no volatility term, no clamp);
COST_TABLE_V2 is the current default. `COST_TABLE_VERSION` always names the
default table, and every table carries its own `cost_table_version` key so
a run manifest states exactly which assumptions produced its numbers.
"""

from __future__ import annotations

import math

COST_TABLE_VERSION = "backtest-cost-table-v2"

# Frozen V1 table (pre-V3): no volatility term, no floor/cap. Kept so a run
# can be replayed under the historical assumptions; do not extend it.
COST_TABLE_V1 = {
    "cost_table_version": "backtest-cost-table-v1",
    "spread_bps": {"micro": 25.0, "small": 12.0, "mid": 6.0, "large": 2.0},
    "liquidity_buckets_dollars": {
        "micro": 1_000_000.0,
        "small": 10_000_000.0,
        "mid": 100_000_000.0,
    },
    "impact_coefficient_bps": 50.0,
    "commission_bps": 0.5,
}

COST_TABLE_V2 = {
    "cost_table_version": "backtest-cost-table-v2",
    "spread_bps": {"micro": 25.0, "small": 12.0, "mid": 6.0, "large": 2.0},
    "liquidity_buckets_dollars": {
        "micro": 1_000_000.0,
        "small": 10_000_000.0,
        "mid": 100_000_000.0,
    },
    "impact_coefficient_bps": 50.0,
    "impact_vol_baseline_daily": 0.02,
    "impact_vol_factor_min": 0.25,
    "impact_vol_factor_max": 4.0,
    "min_total_side_cost_bps": 1.0,
    "max_total_side_cost_bps": 250.0,
    "commission_bps": 0.5,
}

SIDES = ("buy", "sell")


def _validate_table(table: dict) -> None:
    if table.get("cost_table_version") not in ("backtest-cost-table-v1", "backtest-cost-table-v2"):
        raise ValueError(
            f"cost table must carry a known cost_table_version, got "
            f"{table.get('cost_table_version')!r}"
        )
    if "impact_vol_baseline_daily" in table and table["impact_vol_baseline_daily"] <= 0:
        raise ValueError("impact_vol_baseline_daily must be positive")
    if "min_total_side_cost_bps" in table and "max_total_side_cost_bps" in table:
        if table["min_total_side_cost_bps"] > table["max_total_side_cost_bps"]:
            raise ValueError("min_total_side_cost_bps must be <= max_total_side_cost_bps")



def liquidity_bucket(avg_dollar_volume: float | None, table: dict | None = None) -> str:
    """Bucket a 20-session average dollar volume; unknown -> widest spread."""
    table = table if table is not None else COST_TABLE_V2
    _validate_table(table)
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


def realized_vol_daily(frame, position: int, window: int = 20) -> float | None:
    """Realized daily volatility over the `window` returns ending at `position`.

    Close-to-close simple returns, population std (ddof=0). None until at
    least two returns exist — the documented neutral fallback is vol_factor
    1.0, never a fabricated volatility.
    """
    if position < window or position + 1 > len(frame):
        return None
    closes = [float(value) for value in frame["Close"].iloc[position - window:position + 1]]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return round(math.sqrt(variance), 6)


def total_side_cost_bps(
    participation: float | None,
    avg_dollar_volume: float | None,
    daily_vol: float | None = None,
    table: dict | None = None,
) -> dict:
    """Per-side execution cost in bps, fully itemized and clamped.

    participation = order notional / avg dollar volume (worst case 1.0 when
    unknown). Square-root law: impact_bps = coefficient * vol_factor *
    sqrt(participation). The total is clamped to the table's floor/cap and
    the record states whether the clamp engaged.
    """
    table = table if table is not None else COST_TABLE_V2
    _validate_table(table)
    bucket = liquidity_bucket(avg_dollar_volume, table)
    half_spread_bps = table["spread_bps"][bucket] / 2.0
    participation_value = 1.0 if participation is None else max(0.0, float(participation))

    vol_baseline = table.get("impact_vol_baseline_daily")
    vol_factor = 1.0
    if daily_vol is not None and vol_baseline:
        vol_factor = max(
            table.get("impact_vol_factor_min", 0.0),
            min(table.get("impact_vol_factor_max", float("inf")), float(daily_vol) / vol_baseline),
        )
    impact_bps = table["impact_coefficient_bps"] * vol_factor * math.sqrt(participation_value)
    commission_bps = table["commission_bps"]

    raw_total = half_spread_bps + impact_bps + commission_bps
    floor = table.get("min_total_side_cost_bps")
    cap = table.get("max_total_side_cost_bps")
    total = raw_total
    clamped = False
    if floor is not None and total < floor:
        total = floor
        clamped = True
    if cap is not None and total > cap:
        total = cap
        clamped = True
    record = {
        "bucket": bucket,
        "half_spread_bps": round(half_spread_bps, 4),
        "impact_bps": round(impact_bps, 4),
        "commission_bps": commission_bps,
        "raw_total_bps": round(raw_total, 4),
        "total_bps": round(total, 4),
        "clamped": clamped,
    }
    if vol_baseline:
        # The volatility term only exists in tables that declare it (v2+).
        # A V1 table has no volatility assumption, so its records honestly
        # carry no vol fields rather than a fake neutral.
        record["vol_factor"] = round(vol_factor, 4)
        record["vol_available"] = daily_vol is not None
    return record


def execution_price(
    open_price: float,
    side: str,
    participation: float | None,
    avg_dollar_volume: float | None,
    daily_vol: float | None = None,
    table: dict | None = None,
) -> float:
    """Executed price for one side: open adjusted by the side's cost in bps.

    Buys pay the costs (price moves against them); sells receive less.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if open_price <= 0:
        raise ValueError(f"open_price must be positive, got {open_price!r}")
    costs = total_side_cost_bps(participation, avg_dollar_volume, daily_vol, table)
    bps = costs["total_bps"]
    if side == "buy":
        return round(open_price * (1.0 + bps / 10_000.0), 6)
    return round(open_price * (1.0 - bps / 10_000.0), 6)


def execution_cost_record(
    side: str,
    open_price: float,
    participation: float | None,
    avg_dollar_volume: float | None,
    daily_vol: float | None,
    order_notional: float,
    table: dict | None = None,
) -> dict:
    """Itemized, explicit cost record for one execution (Sprint V3).

    The executed price and the full cost decomposition travel together so a
    reviewer can see exactly which assumption produced which number.
    `cost_notional` is the order notional times the total cost bps — the
    money actually paid to the market.
    """
    costs = total_side_cost_bps(participation, avg_dollar_volume, daily_vol, table)
    executed_price = execution_price(
        open_price, side, participation, avg_dollar_volume, daily_vol, table
    )
    return {
        "side": side,
        "open_price": round(float(open_price), 6),
        "executed_price": executed_price,
        "order_notional": round(float(order_notional), 2),
        "participation": round(float(participation), 6) if participation is not None else None,
        "avg_dollar_volume": round(float(avg_dollar_volume), 2) if avg_dollar_volume is not None else None,
        "daily_vol": daily_vol,
        **costs,
        "cost_notional": round(float(order_notional) * costs["total_bps"] / 10_000.0, 2),
    }
