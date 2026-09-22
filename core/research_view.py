"""D1 the combined research view — score, forecast, confidence, regime, risk.

The sprint goal is "a research workstation, not a black box". The failure mode
of a dashboard is not that it shows too little; it is that it renders an empty
system as a confident one.

**THE DECIDING MEASUREMENT, taken on live data before any view was built.** A
forecast snapshot for NVDA at 2026-09-22, at every horizon the roadmap names:

    horizon   PRESENT   ABSENT   REFUSED
    1d              5        2         8
    5d              5        2         8
    20d             5        2         8
    60d             5        2         8

The five PRESENT fields are ticker, as_of, forecast_version, horizon and
warnings — metadata only. Every field a reader would act on (probability_up,
confidence, regime, event_context, evidence) is REFUSED, and expected_return
is ABSENT because no trained model exists.

So this view's first duty is arithmetic honesty: a view over 5 of 15 fields
says so, in its header, before anything else. Rendered as blanks, the same
data would look like a working system having a quiet day.

**Three cell states, not two.** PRESENT / ABSENT / REFUSED are the snapshot's
own vocabulary and the view reuses them rather than inventing a second set.
ABSENT means no producer exists; REFUSED means a producer ran and declined
here. Collapsing them into "no data" would undo the distinction F3 through F7
were built to preserve — and they are genuinely different: ABSENT is a
standing property of the system, REFUSED is a fact about this request.

**The return column is shown, and every cell reads ABSENT.** The roadmap asks
for "return% and probability" per horizon. Omitting the column would hide that
the field was requested and could not be produced; filling it would fabricate
the quantity. The column exists, it refuses, and it names why.

**The view renders; it decides nothing.** The refusal is R7's and the veto is
W2's. Both are displayed, neither is recomputed, and the view never forms a
third opinion — a dashboard that reached its own verdict would be a second
decision-maker nobody governs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.config import (
    RESEARCH_COVERAGE_WARN,
    RESEARCH_PANEL_CONFIDENCE,
    RESEARCH_PANEL_FORECAST,
    RESEARCH_PANEL_QUALITY,
    RESEARCH_PANEL_REGIME,
    RESEARCH_PANEL_RISK,
    RESEARCH_PANELS,
    RESEARCH_RETURN_IS_ABSENT,
    RESEARCH_SHOW_RETURN_COLUMN,
    RESEARCH_VIEW_BLOCKS_TRADES,
    RESEARCH_VIEW_HORIZONS,
    RESEARCH_VIEW_VERSION,
    SNAPSHOT_UNAVAILABLE_FIELDS,
    VIEW_CELL_ABSENT,
    VIEW_CELL_PRESENT,
    VIEW_CELL_REFUSED,
    VIEW_CELL_STATES,
)


class ResearchViewError(ValueError):
    """Raised when a research view request is structurally invalid."""


def _cell(state: str, value: Any = None, reason: str = "", **detail) -> dict:
    """One displayed value, carrying why it is what it is.

    A cell never holds a value in a non-PRESENT state: that is the shape rule
    the whole codebase follows, applied at the render boundary where it
    matters most, because 0.0 and "no measurement" look identical on screen.
    """
    if state not in VIEW_CELL_STATES:
        raise ResearchViewError(f"unknown cell state {state!r}")
    if state != VIEW_CELL_PRESENT and value is not None:
        raise ResearchViewError(
            f"a {state} cell carried the value {value!r}; an unavailable "
            f"quantity rendered as a number is the failure this view exists "
            f"to prevent"
        )
    if state != VIEW_CELL_PRESENT and not str(reason or "").strip():
        raise ResearchViewError(f"a {state} cell carries no reason")
    payload = {"state": state, "value": value, "reason": reason}
    payload.update(detail)
    return payload


def cell_from_field(field: Mapping[str, Any] | None, name: str = "") -> dict:
    """Translate one snapshot field into a view cell, states preserved."""
    if field is None:
        return _cell(
            VIEW_CELL_REFUSED,
            reason=f"no {name or 'field'} was supplied to the view",
        )
    if not isinstance(field, Mapping):
        raise ResearchViewError(f"{name}: snapshot field is not a mapping")
    status = field.get("status")
    if status == VIEW_CELL_PRESENT:
        return _cell(VIEW_CELL_PRESENT, value=field.get("value"),
                     reason=str(field.get("reason") or ""))
    if status not in VIEW_CELL_STATES:
        raise ResearchViewError(f"{name}: unknown snapshot status {status!r}")
    reason = str(field.get("reason") or "").strip()
    if not reason:
        reason = SNAPSHOT_UNAVAILABLE_FIELDS.get(
            name, f"the snapshot reported {status} without a reason"
        )
    return _cell(status, reason=reason)


def forecast_row(snapshot: Mapping[str, Any], horizon: str) -> dict:
    """One horizon's row: return%, P(up), interval, confidence.

    The return cell is ABSENT by construction, not by accident — see
    RESEARCH_RETURN_IS_ABSENT.
    """
    if not isinstance(snapshot, Mapping):
        raise ResearchViewError(f"{horizon}: snapshot is not a mapping")
    fields = snapshot.get("fields") or {}

    row: dict[str, Any] = {"horizon": horizon}
    if RESEARCH_SHOW_RETURN_COLUMN:
        row["expected_return"] = cell_from_field(
            fields.get("expected_return"), "expected_return"
        )
    row["probability_up"] = cell_from_field(
        fields.get("probability_up"), "probability_up"
    )
    row["prediction_interval"] = cell_from_field(
        fields.get("prediction_interval"), "prediction_interval"
    )
    row["confidence"] = cell_from_field(fields.get("confidence"), "confidence")
    return row


def _coverage(cells: Sequence[Mapping[str, Any]]) -> dict:
    """How much of this view is actually populated."""
    total = len(cells)
    present = sum(1 for c in cells if c.get("state") == VIEW_CELL_PRESENT)
    absent = sum(1 for c in cells if c.get("state") == VIEW_CELL_ABSENT)
    refused = sum(1 for c in cells if c.get("state") == VIEW_CELL_REFUSED)
    share = round(present / total, 6) if total else None
    return {
        "cells": total,
        "present": present,
        "absent": absent,
        "refused": refused,
        "share_present": share,
        "below_threshold": bool(share is not None and share < RESEARCH_COVERAGE_WARN),
        "threshold": RESEARCH_COVERAGE_WARN,
    }


def build_research_view(
    ticker: str,
    as_of: Any,
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    quality: Mapping[str, Any] | None = None,
    regime: Mapping[str, Any] | None = None,
    risk: Mapping[str, Any] | None = None,
    portfolio: Mapping[str, Any] | None = None,
    horizons: Sequence[str] = RESEARCH_VIEW_HORIZONS,
) -> dict:
    """Assemble the five panels the sprint requires, shown together.

    `snapshots` maps horizon to an F-sprint forecast snapshot. Everything is
    COMPOSED; nothing is recomputed, so the view cannot disagree with the
    machinery it displays.
    """
    if not str(ticker or "").strip():
        raise ResearchViewError("a ticker is required")
    if not isinstance(snapshots, Mapping):
        raise ResearchViewError("a mapping of horizon to snapshot is required")

    rows = []
    for horizon in horizons:
        snapshot = snapshots.get(horizon)
        if snapshot is None:
            # A MISSING HORIZON IS NOT AN EMPTY ONE. The row still appears, so
            # a reader sees that the horizon was asked for and not answered.
            rows.append(
                {
                    "horizon": horizon,
                    **(
                        {
                            "expected_return": _cell(
                                VIEW_CELL_REFUSED,
                                reason=f"no snapshot was supplied for {horizon}",
                            )
                        }
                        if RESEARCH_SHOW_RETURN_COLUMN
                        else {}
                    ),
                    "probability_up": _cell(
                        VIEW_CELL_REFUSED,
                        reason=f"no snapshot was supplied for {horizon}",
                    ),
                    "prediction_interval": _cell(
                        VIEW_CELL_REFUSED,
                        reason=f"no snapshot was supplied for {horizon}",
                    ),
                    "confidence": _cell(
                        VIEW_CELL_REFUSED,
                        reason=f"no snapshot was supplied for {horizon}",
                    ),
                }
            )
            continue
        rows.append(forecast_row(snapshot, horizon))

    panels = {
        RESEARCH_PANEL_QUALITY: _quality_panel(quality),
        RESEARCH_PANEL_FORECAST: {"horizons": list(horizons), "rows": rows},
        RESEARCH_PANEL_CONFIDENCE: _confidence_panel(rows),
        RESEARCH_PANEL_REGIME: _regime_panel(regime, snapshots, horizons),
        RESEARCH_PANEL_RISK: _risk_panel(risk, portfolio),
    }

    cells = list(_all_cells(panels))
    coverage = _coverage(cells)

    return {
        "version": RESEARCH_VIEW_VERSION,
        "ticker": str(ticker).upper(),
        "as_of": str(as_of),
        "panel_order": list(RESEARCH_PANELS),
        "panels": panels,
        "coverage": coverage,
        "headline": _headline(coverage),
        "blocks_trades": RESEARCH_VIEW_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _all_cells(panels: Mapping[str, Any]):
    """Every cell in the view, wherever it sits."""
    for panel in panels.values():
        if not isinstance(panel, Mapping):
            continue
        for key, value in panel.items():
            if isinstance(value, Mapping) and "state" in value:
                yield value
            elif key == "rows" and isinstance(value, list):
                for row in value:
                    for cell in row.values():
                        if isinstance(cell, Mapping) and "state" in cell:
                            yield cell


def _headline(coverage: Mapping[str, Any]) -> str:
    """What a reader must know before reading anything else."""
    if coverage["cells"] == 0:
        return "this view has no cells to show"
    if coverage["below_threshold"]:
        return (
            f"{coverage['present']} of {coverage['cells']} values are "
            f"available ({coverage['share_present']:.0%}); "
            f"{coverage['absent']} have no producer and "
            f"{coverage['refused']} were declined for this request. Read the "
            f"gaps before the numbers"
        )
    return (
        f"{coverage['present']} of {coverage['cells']} values are available "
        f"({coverage['share_present']:.0%})"
    )


def _quality_panel(quality: Mapping[str, Any] | None) -> dict:
    """Investment Quality, as the score engine reported it."""
    if quality is None:
        return {
            "score": _cell(
                VIEW_CELL_REFUSED, reason="no score was supplied to the view"
            ),
            "action": _cell(
                VIEW_CELL_REFUSED, reason="no score was supplied to the view"
            ),
        }
    if not isinstance(quality, Mapping):
        raise ResearchViewError("the quality panel input is not a mapping")
    score = quality.get("score")
    action = quality.get("action")
    return {
        "score": (
            _cell(VIEW_CELL_PRESENT, value=float(score),
                  reason=str(quality.get("score_reason") or ""))
            if score is not None
            else _cell(VIEW_CELL_REFUSED, reason="the score engine published no score")
        ),
        "action": (
            _cell(VIEW_CELL_PRESENT, value=str(action),
                  reason=str(quality.get("action_reason") or ""))
            if action
            else _cell(VIEW_CELL_REFUSED, reason="no recommended action was published")
        ),
    }


def _confidence_panel(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Confidence per horizon, read off the same rows the forecast shows.

    Not recomputed: a confidence that disagreed with the forecast table beside
    it would be two numbers for one quantity.
    """
    per_horizon = {}
    for row in rows:
        cell = row.get("confidence")
        if isinstance(cell, Mapping):
            per_horizon[row["horizon"]] = cell
    measured = [
        c for c in per_horizon.values() if c.get("state") == VIEW_CELL_PRESENT
    ]
    return {
        "per_horizon": per_horizon,
        "measured_horizons": len(measured),
        "note": (
            "confidence is how reliable the forecast is, never how bullish. "
            "MEASURED, P(up) 0.500 from 1,000 observations is a "
            "HIGH-confidence statement and P(up) 1.000 from 2 is a near-zero "
            "one"
        ),
    }


def _regime_panel(
    regime: Mapping[str, Any] | None,
    snapshots: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[str],
) -> dict:
    """The regime in force, from the snapshot that carries it."""
    if regime is not None:
        if not isinstance(regime, Mapping):
            raise ResearchViewError("the regime panel input is not a mapping")
        label = regime.get("label")
        return {
            "regime": (
                _cell(VIEW_CELL_PRESENT, value=str(label),
                      reason=str(regime.get("reason") or ""))
                if label
                else _cell(
                    VIEW_CELL_REFUSED,
                    reason=str(regime.get("reason") or "no regime was established"),
                )
            )
        }
    for horizon in horizons:
        snapshot = snapshots.get(horizon)
        if isinstance(snapshot, Mapping):
            field = (snapshot.get("fields") or {}).get("regime")
            if field is not None:
                return {"regime": cell_from_field(field, "regime")}
    return {
        "regime": _cell(
            VIEW_CELL_REFUSED, reason="no regime was supplied to the view"
        )
    }


def _risk_panel(
    risk: Mapping[str, Any] | None, portfolio: Mapping[str, Any] | None
) -> dict:
    """W2's per-ticker veto and R7's portfolio decision, side by side.

    Both are DISPLAYED, neither recomputed. They answer different questions —
    W2 about this ticker's evidence, R7 about whether the book can absorb the
    trade — and a view that merged them would hide which one refused.
    """
    panel: dict[str, Any] = {}
    if risk is None:
        panel["veto"] = _cell(
            VIEW_CELL_REFUSED, reason="no risk policy evaluation was supplied"
        )
    elif not isinstance(risk, Mapping):
        raise ResearchViewError("the risk panel input is not a mapping")
    else:
        panel["veto"] = _cell(
            VIEW_CELL_PRESENT,
            value=bool(risk.get("veto")),
            reason=(
                f"W2 per-ticker evidence policy; triggered: "
                f"{list(risk.get('veto_rule_ids') or []) or 'none'}"
            ),
        )
        panel["veto_rule_ids"] = list(risk.get("veto_rule_ids") or [])

    if portfolio is None:
        panel["portfolio_verdict"] = _cell(
            VIEW_CELL_REFUSED, reason="no portfolio decision was supplied"
        )
    elif not isinstance(portfolio, Mapping):
        raise ResearchViewError("the portfolio panel input is not a mapping")
    else:
        verdict = portfolio.get("verdict")
        panel["portfolio_verdict"] = (
            _cell(
                VIEW_CELL_PRESENT,
                value=str(verdict),
                reason=str(portfolio.get("reason") or ""),
            )
            if verdict
            else _cell(
                VIEW_CELL_REFUSED, reason="the portfolio decision published no verdict"
            )
        )
        panel["portfolio_veto_rule_ids"] = list(
            portfolio.get("veto_rule_ids") or []
        )
    panel["note"] = (
        "W2 asks whether this ticker's EVIDENCE is sound; R7 asks whether the "
        "BOOK can absorb the trade. Both are shown because they refuse for "
        "different reasons"
    )
    return panel


_NOTE = (
    "A research view renders; it decides nothing. MEASURED on live data, 5 of "
    "15 snapshot fields are PRESENT at every horizon - the rest have no "
    "producer or were declined - so the view leads with what is missing "
    "rather than rendering absence as a blank."
)


def view_problems(view: Mapping[str, Any]) -> list[str]:
    """Contract check on a research view. Empty means clean."""
    problems: list[str] = []
    if not isinstance(view, Mapping):
        return ["view is not a mapping"]

    if view.get("blocks_trades"):
        problems.append(
            "the research view claims to block trades — it renders, and the "
            "refusal belongs to R7 and W2"
        )

    panels = view.get("panels") or {}
    for required in RESEARCH_PANELS:
        if required not in panels:
            problems.append(
                f"panel {required!r} is missing — the sprint requires score, "
                f"forecast, confidence, regime and risk shown TOGETHER"
            )

    forecast = panels.get(RESEARCH_PANEL_FORECAST) or {}
    rows = forecast.get("rows") or []
    shown = {row.get("horizon") for row in rows if isinstance(row, Mapping)}
    for horizon in RESEARCH_VIEW_HORIZONS:
        if horizon not in shown:
            problems.append(
                f"horizon {horizon!r} was requested and does not appear — a "
                f"dropped horizon is indistinguishable from one never asked for"
            )

    for row in rows:
        if not isinstance(row, Mapping):
            problems.append("a forecast row is not a mapping")
            continue
        if RESEARCH_SHOW_RETURN_COLUMN and "expected_return" not in row:
            problems.append(
                f"{row.get('horizon')}: the return column is missing, hiding "
                f"that the field was requested and could not be produced"
            )
        return_cell = row.get("expected_return")
        if isinstance(return_cell, Mapping) and RESEARCH_RETURN_IS_ABSENT:
            if return_cell.get("state") == VIEW_CELL_PRESENT:
                problems.append(
                    f"{row.get('horizon')}: an expected return was rendered as "
                    f"PRESENT — no trained model exists to produce one"
                )

    # THE SHAPE RULE AT THE RENDER BOUNDARY: a value exists IFF PRESENT.
    for cell in _all_cells(panels):
        state = cell.get("state")
        if state not in VIEW_CELL_STATES:
            problems.append(f"unknown cell state {state!r}")
            continue
        if state == VIEW_CELL_PRESENT:
            if cell.get("value") is None:
                problems.append("a PRESENT cell carries no value")
        else:
            if cell.get("value") is not None:
                problems.append(
                    f"a {state} cell carries the value {cell.get('value')!r} — "
                    f"an unavailable quantity rendered as a number is exactly "
                    f"what this view exists to prevent"
                )
            if not str(cell.get("reason") or "").strip():
                problems.append(f"a {state} cell carries no reason")

    coverage = view.get("coverage") or {}
    if coverage:
        counted = (
            int(coverage.get("present", 0))
            + int(coverage.get("absent", 0))
            + int(coverage.get("refused", 0))
        )
        if counted != int(coverage.get("cells", -1)):
            problems.append(
                f"coverage counts {counted} cells against a declared "
                f"{coverage.get('cells')}"
            )
        if coverage.get("below_threshold") and not str(
            view.get("headline") or ""
        ).strip():
            problems.append(
                "coverage is below the threshold and the view has no headline "
                "— a mostly-empty view that does not say so reads as a "
                "working system having a quiet day"
            )
    return problems


def render_view(view: Mapping[str, Any]) -> list[str]:
    """Human-readable lines: the headline first, then each panel."""
    lines = [
        f"  {view.get('ticker')} @ {view.get('as_of')}",
        f"  {view.get('headline')}",
    ]
    panels = view.get("panels") or {}

    quality = panels.get(RESEARCH_PANEL_QUALITY) or {}
    score = quality.get("score") or {}
    lines.append(
        f"  quality      {_shown(score)}   action {_shown(quality.get('action'))}"
    )

    regime = (panels.get(RESEARCH_PANEL_REGIME) or {}).get("regime") or {}
    risk = panels.get(RESEARCH_PANEL_RISK) or {}
    lines.append(
        f"  regime       {_shown(regime)}   "
        f"W2 veto {_shown(risk.get('veto'))}   "
        f"R7 {_shown(risk.get('portfolio_verdict'))}"
    )

    forecast = panels.get(RESEARCH_PANEL_FORECAST) or {}
    lines.append(f"  {'horizon':8s} {'return%':12s} {'P(up)':12s} {'confidence':12s}")
    for row in forecast.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"  {str(row.get('horizon')):8s} "
            f"{_shown(row.get('expected_return')):12s} "
            f"{_shown(row.get('probability_up')):12s} "
            f"{_shown(row.get('confidence')):12s}"
        )
    return lines


def _shown(cell: Mapping[str, Any] | None) -> str:
    """A cell as one short token: its value, or the state that replaced it.

    A structured value is reduced to the scalar a reader needs. The confidence
    field, for instance, carries F7's whole assessment — nine factors, their
    statuses and the aggregation evidence — and printing that into a table
    column produces a wall of text where a number belongs. The detail is not
    lost: it stays in the cell for the provenance surface to expose.
    """
    if not isinstance(cell, Mapping):
        return "—"
    if cell.get("state") != VIEW_CELL_PRESENT:
        return str(cell.get("state"))
    return _scalar(cell.get("value"))


def _scalar(value: Any) -> str:
    """The one token that stands for a value in a table cell."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Mapping):
        # Prefer the quantity a reader is looking for, then its band/label.
        for key in ("confidence", "band", "value", "label", "verdict"):
            if key in value and not isinstance(value[key], (Mapping, list)):
                return _scalar(value[key])
        return "{...}"
    if isinstance(value, (list, tuple)):
        return f"[{len(value)}]"
    text = str(value)
    return text if len(text) <= 24 else text[:21] + "..."
