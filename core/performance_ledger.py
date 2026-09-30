"""Forecast performance ledger (Sprint L2) — sliced, with its coverage.

Performance by ticker, sector, regime, horizon, event type, source, model,
confidence bucket and volatility regime.

**Four dimensions had to be captured at RECORD time, not derived later.** A
ledger row carried only two of the nine (ticker, horizon); sector and the
confidence bucket are safely derivable because their inputs are immutable. But
regime, event type, source and volatility regime are POINT-IN-TIME facts:

- regime and volatility can be recomputed from bars, yet the forecast was
  *made* under a specific classifier reading. Recomputing later risks a
  different answer if the classifier moved, silently re-attributing past
  performance to a regime nobody forecast under;
- event type depends on the event set at `as_of`, and news is gone after
  `NEWS_LOOKBACK_DAYS`;
- source — the observed/inferred mix of the analogs — depends on a retrieval
  against a store that grows. MEASURED across 8 probes, comparing retrieval
  against half the store versus the full store, **7 of 8 returned a different
  analog count** (14→19, 14→20, 16→24, 0→1, 3→4, 9→16, 0→1).

So "what evidence did this forecast rest on?" is answerable only because it
was written down when the forecast was made.

**Breakdowns are MARGINAL, never a cross-tab.** MEASURED: the nine dimensions
cross-multiply to 4,065,600 cells against 77,616 forecasts a year — 0.0191 per
cell. A full cross-tabulation would be empty almost everywhere while looking
thorough, so this module slices one dimension at a time.

**Every cell faces F4's floors.** A cell below `PERFORMANCE_MIN_CELL` reports
its count and refuses a metric. A performance number from three forecasts is
the F4 stress cell wearing an analytics hat.

**Every cell carries L1's coverage.** Scored / refused counts travel with each
per-cell error, for the same measured reason they travel with the global one:
refusing remains the cheapest way to look accurate.
"""

from __future__ import annotations

import logging

from core.config import (
    CLOSURE_CLOSED,
    PERF_CONFIDENCE_BUCKET,
    PERF_EVENT_TYPE,
    PERF_HORIZON,
    PERF_MODEL,
    PERF_REGIME,
    PERF_SECTOR,
    PERF_SOURCE,
    PERF_TICKER,
    PERF_VOLATILITY_REGIME,
    PERFORMANCE_CONFIDENCE_BUCKETS,
    PERFORMANCE_CONTRACT_VERSION,
    PERFORMANCE_CROSS_TABULATE,
    PERFORMANCE_DIMENSIONS,
    PERFORMANCE_LEDGER_VERSION,
    PERFORMANCE_MIN_CELL,
    PERFORMANCE_SPARSITY_EVIDENCE,
    PERFORMANCE_VOLATILITY_BANDS,
    SCORE_METHOD_COVERAGE,
    SCORE_METHOD_DIRECTION,
    SCORE_METHOD_NOT_SCORED,
)

LOGGER = logging.getLogger("core.performance_ledger")

UNKNOWN = "unknown"


class PerformanceLedgerError(ValueError):
    """Raised when a performance request violates the L2 contract."""


def _band(value, bands) -> str:
    """The band a value falls in, or `unknown` when there is no value.

    `unknown` is a real level, not a default: a forecast whose confidence was
    never recorded is a different fact from one recorded as low, and bucketing
    it into LOW would invent evidence.
    """
    if value is None:
        return UNKNOWN
    name = bands[0][0]
    for label, floor in bands:
        if float(value) >= floor:
            name = label
    return name


def dimension_value(row: dict, dimension: str) -> str:
    """The level of one dimension for one closed forecast.

    Captured dimensions are READ, never recomputed — that is the whole point
    of writing them at record time.
    """
    if dimension not in PERFORMANCE_DIMENSIONS:
        raise PerformanceLedgerError(
            f"unknown dimension {dimension!r} "
            f"(known: {list(PERFORMANCE_DIMENSIONS)})"
        )

    if dimension == PERF_TICKER:
        return str(row.get("ticker") or UNKNOWN).upper()
    if dimension == PERF_HORIZON:
        return str(row.get("horizon") or UNKNOWN)
    if dimension == PERF_REGIME:
        return str(row.get("regime") or UNKNOWN)
    if dimension == PERF_EVENT_TYPE:
        return str(row.get("event_type") or UNKNOWN)
    if dimension == PERF_VOLATILITY_REGIME:
        return _band(row.get("volatility"), PERFORMANCE_VOLATILITY_BANDS)
    if dimension == PERF_CONFIDENCE_BUCKET:
        return _band(row.get("confidence"), PERFORMANCE_CONFIDENCE_BUCKETS)

    if dimension == PERF_SOURCE:
        share = row.get("observed_share")
        if share is None:
            return UNKNOWN
        # A binary split: the evidence was predominantly sourced, or it was
        # predominantly dated by inference (precision 0.65). A finer scale
        # would imply the share itself was measured more precisely than it is.
        return "observed" if float(share) >= 0.5 else "inferred"

    if dimension == PERF_SECTOR:
        # Derived, and safe to derive: a ticker's sector mapping does not
        # change retroactively the way a retrieval does.
        from core.market_context import sector_for

        return str(sector_for(str(row.get("ticker") or "")) or UNKNOWN)

    if dimension == PERF_MODEL:
        versions = row.get("forecast_version") or {}
        return str(versions.get("snapshot") or versions.get("event") or UNKNOWN)

    return UNKNOWN


def cell_metrics(rows: list[dict]) -> dict:
    """Metrics for one cell, or a refusal with its count.

    Below `PERFORMANCE_MIN_CELL` scored forecasts the cell reports what it has
    and publishes no metric — the same floor F4 applies to a conditional
    slice, because a performance number from three forecasts is the same
    stress cell in different clothing.
    """
    total = len(rows)
    scored = [r for r in rows if (r.get("score") or {}).get("scored")]
    refused = [
        r for r in rows
        if (r.get("score") or {}).get("method") == SCORE_METHOD_NOT_SCORED
    ]

    # Intervals now carry the Brier of their midpoint, so they contribute a
    # usable point error alongside POINT claims.
    briers = [
        float(r["score"]["brier"]) for r in scored
        if r["score"].get("brier") is not None
    ]
    # COVERAGE, computed across the CELL rather than per forecast. An
    # interval claims the RATE at which such setups rise; a single binary
    # outcome can never fall inside a probability range, so asking per
    # forecast returns False every time and measures nothing. The answerable
    # question is whether the cell's realised rate landed inside.
    interval_rows = [
        r for r in scored if r["score"].get("method") == SCORE_METHOD_COVERAGE
    ]
    hits = [
        bool(r["score"]["hit"]) for r in scored
        if r["score"].get("method") == SCORE_METHOD_DIRECTION
    ]

    cell = {
        "forecasts": total,
        "scored": len(scored),
        "refused": len(refused),
        "scored_share": round(len(scored) / total, 6) if total else 0.0,
        "sufficient": len(scored) >= PERFORMANCE_MIN_CELL,
    }

    if not cell["sufficient"]:
        cell["reason"] = (
            f"{len(scored)} scored forecast(s), below the {PERFORMANCE_MIN_CELL} "
            f"floor; a performance number from this few is the F4 stress cell "
            f"in analytics clothing"
        )
        return cell

    cell["reason"] = ""
    # THE SHAPE RULE, inherited: a metric key exists only where it was
    # computed. A `mean_brier: None` would coalesce to 0.0 in a consumer and
    # render a cell that scored nothing as perfectly accurate.
    if briers:
        cell["mean_brier"] = round(sum(briers) / len(briers), 6)
        cell["brier_count"] = len(briers)
    if interval_rows:
        realised = sum(
            float(r["score"]["actual"]) for r in interval_rows
        ) / len(interval_rows)
        # The widest claim the cell made, so "inside" means inside every
        # interval the cell actually published.
        lower = max(float(r["score"]["lower"]) for r in interval_rows)
        upper = min(float(r["score"]["upper"]) for r in interval_rows)
        cell["realised_rate"] = round(realised, 6)
        cell["interval_lower"] = round(lower, 6)
        cell["interval_upper"] = round(upper, 6)
        cell["coverage_rate"] = 1.0 if lower <= realised <= upper else 0.0
        cell["coverage_count"] = len(interval_rows)
        cell["coverage_is_group_property"] = True
    if hits:
        cell["direction_accuracy"] = round(sum(hits) / len(hits), 6)
        cell["direction_count"] = len(hits)
    return cell


def breakdown(closed_rows: list[dict], dimension: str) -> dict:
    """Performance sliced by ONE dimension.

    Marginal by construction. MEASURED, the nine dimensions cross-multiply to
    4,065,600 cells against 77,616 forecasts a year, so a cross-tab is empty
    almost everywhere while appearing thorough.
    """
    if dimension not in PERFORMANCE_DIMENSIONS:
        raise PerformanceLedgerError(
            f"unknown dimension {dimension!r} "
            f"(known: {list(PERFORMANCE_DIMENSIONS)})"
        )

    grouped: dict[str, list[dict]] = {}
    for row in closed_rows or []:
        if row.get("state") != CLOSURE_CLOSED:
            continue
        grouped.setdefault(dimension_value(row, dimension), []).append(row)

    cells = {level: cell_metrics(rows) for level, rows in sorted(grouped.items())}
    sufficient = [name for name, cell in cells.items() if cell["sufficient"]]

    return {
        "dimension": dimension,
        "levels": sorted(cells),
        "cells": cells,
        "sufficient_levels": sorted(sufficient),
        "thin_levels": sorted(set(cells) - set(sufficient)),
        "forecasts": sum(c["forecasts"] for c in cells.values()),
        "scored": sum(c["scored"] for c in cells.values()),
        "refused": sum(c["refused"] for c in cells.values()),
        "min_cell": PERFORMANCE_MIN_CELL,
        "cross_tabulated": PERFORMANCE_CROSS_TABULATE,
        "ledger_version": PERFORMANCE_LEDGER_VERSION,
        "contract_version": PERFORMANCE_CONTRACT_VERSION,
    }


def performance_report(closed_rows: list[dict]) -> dict:
    """Every declared dimension, each sliced marginally."""
    return {
        "dimension_order": list(PERFORMANCE_DIMENSIONS),
        "breakdowns": {
            dimension: breakdown(closed_rows, dimension)
            for dimension in PERFORMANCE_DIMENSIONS
        },
        "forecasts": sum(
            1 for r in (closed_rows or []) if r.get("state") == CLOSURE_CLOSED
        ),
        "cross_tabulated": PERFORMANCE_CROSS_TABULATE,
        "sparsity_evidence": PERFORMANCE_SPARSITY_EVIDENCE,
        "ledger_version": PERFORMANCE_LEDGER_VERSION,
        "contract_version": PERFORMANCE_CONTRACT_VERSION,
    }


def performance_problems(report: dict) -> list[str]:
    """Validate a performance report against the L2 contract."""
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a dict"]

    for field in ("ledger_version", "contract_version", "sparsity_evidence"):
        if not report.get(field):
            problems.append(f"report field {field!r} missing/empty")
    if report.get("cross_tabulated"):
        problems.append(
            "the report claims to cross-tabulate. " + PERFORMANCE_SPARSITY_EVIDENCE
        )

    breakdowns = report.get("breakdowns") or {}
    if list(breakdowns) != list(PERFORMANCE_DIMENSIONS):
        problems.append(
            "not every declared dimension is reported in order — a missing "
            "breakdown reads as an oversight where an empty one reads as a fact"
        )

    for name, table in breakdowns.items():
        for level, cell in (table.get("cells") or {}).items():
            where = f"{name}[{level}]"
            if not cell.get("sufficient"):
                for metric in ("mean_brier", "coverage_rate", "direction_accuracy"):
                    if metric in cell:
                        problems.append(
                            f"{where}: a thin cell published {metric!r} over "
                            f"{cell.get('scored')} scored forecast(s), below "
                            f"the {PERFORMANCE_MIN_CELL} floor"
                        )
                if not cell.get("reason"):
                    problems.append(f"{where}: a thin cell does not explain itself")
            # Coverage travels with every per-cell error.
            has_metric = any(
                key in cell
                for key in ("mean_brier", "coverage_rate", "direction_accuracy")
            )
            if has_metric:
                for required in ("scored", "refused", "forecasts", "scored_share"):
                    if cell.get(required) is None:
                        problems.append(
                            f"{where}: a metric is published without "
                            f"{required!r} — refusing remains the cheapest way "
                            f"to look accurate"
                        )
            if cell.get("scored", 0) > cell.get("forecasts", 0):
                problems.append(f"{where}: scored exceeds the cell's forecasts")

    return problems


def render_breakdown(table: dict) -> list[dict]:
    """One reading row per level, thinnest facts included. Never blank."""
    rows: list[dict] = []
    for level in table.get("levels") or ():
        cell = (table.get("cells") or {}).get(level) or {}
        rows.append(
            {
                "level": level,
                "forecasts": cell.get("forecasts"),
                "scored": cell.get("scored"),
                "refused": cell.get("refused"),
                "sufficient": cell.get("sufficient"),
                "mean_brier": cell.get("mean_brier"),
                "coverage_rate": cell.get("coverage_rate"),
                "direction_accuracy": cell.get("direction_accuracy"),
                "reason": cell.get("reason") or "",
            }
        )
    return rows
