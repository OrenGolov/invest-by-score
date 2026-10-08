"""Joint forecast (Sprint F3) — every target, every horizon, one object.

F3's requirement: report return and P(up) across all horizons together, with
calibrated uncertainty. `build_joint_forecast` composes F1's six targets with
F2's six horizons into a 36-cell grid, each cell carrying its own status,
uncertainty and reason.

**F3 defines and validates the joint contract; it fits no models.** None exist:
the measured incumbent is a momentum baseline at 0.575 directional accuracy. So
every cell that would need a model refuses honestly, and the object is useful
today as a contract rather than as a prediction.

**The shape rule that carries the design.** A cell carries a `value` key IF AND
ONLY IF its status is OK — the key is ABSENT otherwise, not `None`. The
dashboard's existing idiom is `Number(x ?? 0)`, so a null P(up) would coalesce
to `0.0%` and render as CERTAIN DOWN: the most dangerous possible misreading,
produced by code that looks defensive. With the key absent, `cell.value ?? 0` is
`undefined` in JS and `cell["value"]` is a KeyError in Python. The failure is
loud at the boundary instead of plausible on the screen. For the same reason a
non-OK cell carries `interval: None` AND `dispersion: None` — refusing a point
estimate in one field while supplying one in another is incoherent.

**Point-in-time correctness is structural.** This module never fetches. Labels
arrive as a parameter, so there is no provider to read past `as_of` with.

**Width is the uncertainty guard, not fold count.** MEASURED: an empirical
interval NARROWS as folds decrease (width 0.089 at 2 folds vs 0.238 at 30 for
the same true spread), and `prediction_interval([0.55]*3)` returns width 0.0 at
`folds: 3`. A fold-count floor passes its own check while publishing perfect
certainty, so `DEGENERATE_INTERVAL` keys on width.

**Four coherence rules were REJECTED**, each by measurement, and the reasoning
lives in `JOINT_REJECTED_RULES` so nobody re-adds one believing it was merely
overlooked. The most seductive was `adverse_excursion <= min(0, expected_return)`:
a gap-up produces a POSITIVE adverse excursion, and 9 of 300 real 20d labels
violate it. A rule that calls ground truth malformed is the wrong rule.

**F3 reports contract defects rather than papering over them.** Building a joint
forecast surfaces `findings` — real inconsistencies between F1, F2 and V1 that
predate this sprint. Hiding them inside a tidy object would make the tidiness a
lie.
"""

from __future__ import annotations

import logging

from core.config import (
    CELL_STATUS_DEGENERATE_INTERVAL,
    CELL_STATUS_LABEL_UNBACKED,
    CELL_STATUS_NEEDS_BENCHMARK,
    CELL_STATUS_NO_MODEL,
    CELL_STATUS_OK,
    CELL_STATUS_PENDING,
    CELL_STATUS_UNAVAILABLE,
    CELL_STATUS_UNCALIBRATED,
    FORECAST_HORIZONS,
    FORECAST_TARGETS,
    JOINT_CELL_PRECEDENCE,
    JOINT_COHERENCE_FLAGGED,
    JOINT_COHERENCE_NOT_EVALUATED,
    JOINT_COHERENCE_OK,
    JOINT_COHERENCE_VIOLATED,
    JOINT_FORECAST_CONTRACT_VERSION,
    JOINT_FORECAST_PIPELINE_VERSION,
    JOINT_MIN_INTERVAL_WIDTH,
    JOINT_REJECTED_RULES,
    JOINT_STATUS_NO_MODEL,
    JOINT_STATUS_OK,
    JOINT_STATUS_PARTIAL,
    JOINT_STATUS_UNAVAILABLE,
)
from core.forecast_horizons import (
    HORIZON_STATUS_PENDING,
    HORIZON_STATUS_SCORABLE,
    horizon_contract,
    horizon_readiness,
)
from core.forecast_targets import (
    bounds_problems,
    label_field_for,
    realized_value,
    target_contract,
)

LOGGER = logging.getLogger("core.forecast_joint")

MODEL_STATUS_NONE = "NO_MODEL"
MODEL_STATUS_BASELINE = "BASELINE"
MODEL_STATUS_MODEL = "MODEL"


class JointForecastError(ValueError):
    """Raised when a joint-forecast request violates the F3 contract."""


def _interval_width(interval: dict | None) -> float | None:
    if not interval:
        return None
    low, high = interval.get("lower"), interval.get("upper")
    if low is None or high is None:
        return None
    return float(high) - float(low)


def resolve_cell_status(
    target: str,
    horizon: str,
    *,
    has_labels: bool,
    readiness: str,
    label_backed: bool,
    benchmark: str | None,
    calibration_map=None,
    interval: dict | None = None,
    has_model: bool = False,
    has_value: bool = False,
) -> tuple[str, str]:
    """The single status for one cell, by DECLARED precedence.

    Returns `(status, reason)`. Precedence is total and ordered in
    `JOINT_CELL_PRECEDENCE`, so a cell is never ambiguous about why it is
    empty: the most fundamental obstacle wins, and the reason names it.
    """
    contract = target_contract(target)

    if not has_labels:
        return CELL_STATUS_UNAVAILABLE, "no label set was supplied"

    if readiness == HORIZON_STATUS_PENDING:
        return (
            CELL_STATUS_PENDING,
            f"the {horizon} window has not closed; an outcome here would be fabricated",
        )
    if readiness != HORIZON_STATUS_SCORABLE:
        return CELL_STATUS_UNAVAILABLE, f"the label set reports no outcome for {horizon}"

    if not label_backed:
        return (
            CELL_STATUS_LABEL_UNBACKED,
            f"no realized {contract['label_field']!r} exists at {horizon}, so this "
            f"target could never be scored there",
        )

    if contract["requires_benchmark"] and not benchmark:
        return (
            CELL_STATUS_NEEDS_BENCHMARK,
            f"{target} compares the stock to a benchmark and none was supplied",
        )

    width = _interval_width(interval)
    if width is not None and width < JOINT_MIN_INTERVAL_WIDTH:
        return (
            CELL_STATUS_DEGENERATE_INTERVAL,
            f"the prediction interval has width {width:g} — that is arithmetic, "
            f"not uncertainty, and would publish perfect certainty",
        )

    if contract["requires_calibration"] and calibration_map is None:
        return (
            CELL_STATUS_UNCALIBRATED,
            f"{target} is a probability and no fitted calibration map was supplied; "
            f"a raw score is not a likelihood",
        )

    if not has_model:
        return (
            CELL_STATUS_NO_MODEL,
            "no trained forecasting model exists for this target and horizon",
        )

    # A model can exist and still have produced nothing for THIS cell. Without
    # this step an unsupplied value reached OK carrying `value: None` — exactly
    # the null the shape rule exists to prevent, arriving through the front door.
    if not has_value:
        return (
            CELL_STATUS_NO_MODEL,
            f"a model is registered but produced no value for {target} at {horizon}",
        )

    return CELL_STATUS_OK, ""


def build_cell(
    target: str,
    horizon: str,
    *,
    labels: dict | None,
    readiness: str,
    benchmark: str | None = None,
    calibration_map=None,
    interval: dict | None = None,
    dispersion: dict | None = None,
    value: float | None = None,
    has_model: bool = False,
) -> dict:
    """One (target, horizon) cell.

    A `value` key appears IF AND ONLY IF the status is OK. See the module
    docstring: `value: None` would coalesce to 0.0 in the dashboard and render
    a refused probability as certain-down.
    """
    contract = target_contract(target)
    has_labels = bool(labels)
    label_backed = (
        has_labels and realized_value(target, labels, horizon) is not None
    )

    status, reason = resolve_cell_status(
        target, horizon,
        has_labels=has_labels,
        readiness=readiness,
        label_backed=label_backed,
        benchmark=benchmark,
        calibration_map=calibration_map,
        interval=interval,
        has_model=has_model,
        has_value=value is not None,
    )

    cell = {
        "target": target,
        "horizon": horizon,
        "status": status,
        "kind": contract["kind"],
        "unit": contract["unit"],
        "question": contract["question"],
        "requires_calibration": contract["requires_calibration"],
        "label_field": contract["label_field"],
        "label_backed": label_backed,
        # A refused cell supplies NO uncertainty either: an interval beside a
        # withheld point estimate is a point estimate by another name.
        "interval": None,
        "dispersion": None,
        "reason": reason,
    }

    if status == CELL_STATUS_OK:
        problems = bounds_problems(target, value)
        if problems:
            raise JointForecastError(
                f"{target}@{horizon}: an OK cell violates its own bounds: "
                + "; ".join(problems)
            )
        cell["value"] = value
        cell["interval"] = interval
        cell["dispersion"] = dispersion
    return cell


def _model_block(model_resolution: dict | None) -> dict:
    """Model identity, never a bare string.

    An absent model is a STATUS with a reason, so a reader can tell "no model
    was trained" from "a model ran and produced nothing".
    """
    if not model_resolution:
        return {
            "status": MODEL_STATUS_NONE,
            "model_version": None,
            "entry_hash": None,
            "reason": (
                "no trained forecasting model is registered; the measured incumbent "
                "is a rule-based baseline, which cannot produce a multi-horizon "
                "forecast"
            ),
        }
    return {
        "status": str(model_resolution.get("status", MODEL_STATUS_NONE)),
        "model_version": model_resolution.get("model_version"),
        "entry_hash": model_resolution.get("entry_hash"),
        "reason": str(model_resolution.get("reason", "")),
    }


def contract_findings(labels: dict | None) -> list[str]:
    """Contract defects F3 can SEE, reported rather than hidden.

    These predate F3 and are not F3's to silently fix. A joint object that
    rendered tidily over them would make the tidiness a lie.
    """
    findings: list[str] = []

    # 1. A realized adverse excursion can violate its own F1 bound.
    horizons = (labels or {}).get("horizons") or {}
    for name, record in horizons.items():
        if not isinstance(record, dict):
            continue
        excursion = record.get("adverse_excursion")
        if excursion is None:
            continue
        problems = bounds_problems("adverse_excursion", excursion)
        if problems:
            findings.append(
                f"realized adverse_excursion at {name} is {excursion}, which violates "
                f"its own F1 bound ({problems[0]}). A gap-up leaves every low above "
                f"the entry close, and labels.py applies no floor at zero — so ground "
                f"truth fails the contract it is scored against"
            )

    # 2. A target that shares another's label field WITHOUT declaring itself
    # unavailable would silently be scored against the wrong quantity. Checked
    # by BEHAVIOUR, not by field name: probability_outperform still declares
    # `forward_return` for shape uniformity, but `label_unavailable` stops it
    # resolving, so comparing names alone would report a defect that is fixed.
    probe = {
        "horizons": {
            "20d": {"status": "OK", "forward_return": 0.01, "label_up": True,
                    "adverse_excursion": -0.01, "realized_vol": 0.01}
        }
    }
    relative = realized_value("probability_outperform", probe, "20d")
    if relative is not None and relative == realized_value("expected_return", probe, "20d"):
        findings.append(
            "probability_outperform resolves to the same realized value as "
            "expected_return, so the relative target is scored against the raw "
            "stock return with no benchmark subtracted"
        )

    return findings


def build_joint_forecast(
    ticker: str,
    as_of,
    labels: dict | None = None,
    benchmark: str | None = None,
    model_resolution: dict | None = None,
    calibration_maps: dict | None = None,
    intervals: dict | None = None,
    dispersions: dict | None = None,
    values: dict | None = None,
) -> dict:
    """The joint forecast for one ticker at one as_of.

    `labels` arrives as a parameter and nothing is fetched, so this module has
    no way to read past `as_of`. `calibration_maps`, `intervals`, `dispersions`
    and `values` are keyed by `(target, horizon)` tuples; all are optional and
    absent entries simply leave their cell refused.
    """
    if not str(ticker or "").strip():
        raise JointForecastError("a ticker is required")

    import pandas as pd

    target_stamp = pd.Timestamp(as_of)
    calibration_maps = calibration_maps or {}
    intervals = intervals or {}
    dispersions = dispersions or {}
    values = values or {}
    model = _model_block(model_resolution)
    has_model = model["status"] in (MODEL_STATUS_MODEL, MODEL_STATUS_BASELINE)

    rows: dict[str, dict] = {}
    emitted = 0
    for horizon in FORECAST_HORIZONS:
        readiness = horizon_readiness(labels or {}, horizon)
        contract = horizon_contract(horizon)
        record = ((labels or {}).get("horizons") or {}).get(horizon) or {}

        cells: dict[str, dict] = {}
        for target in FORECAST_TARGETS:
            key = (target, horizon)
            cell = build_cell(
                target, horizon,
                labels=labels,
                readiness=readiness["status"],
                benchmark=benchmark,
                calibration_map=calibration_maps.get(key),
                interval=intervals.get(key),
                dispersion=dispersions.get(key),
                value=values.get(key),
                has_model=has_model,
            )
            if cell["status"] == CELL_STATUS_OK:
                emitted += 1
            cells[target] = cell

        rows[horizon] = {
            "horizon": horizon,
            "sessions": contract["sessions"],
            "calendar_days": contract["calendar_days"],
            "is_long": contract["is_long"],
            "readiness": readiness["status"],
            "readiness_reason": readiness["reason"],
            # This horizon's OWN exit bar: different horizons become knowable at
            # different times, and one shared timestamp would hide that.
            "knowable_at": record.get("exit_bar"),
            "entry_bar": record.get("entry_bar"),
            "cells": cells,
        }

    if not labels:
        status = JOINT_STATUS_UNAVAILABLE
        reason = "no label set was supplied, so no cell can be placed"
    elif not has_model:
        status = JOINT_STATUS_NO_MODEL
        reason = model["reason"]
    elif emitted == len(FORECAST_HORIZONS) * len(FORECAST_TARGETS):
        status = JOINT_STATUS_OK
        reason = ""
    else:
        status = JOINT_STATUS_PARTIAL
        reason = f"{emitted} of {len(FORECAST_HORIZONS) * len(FORECAST_TARGETS)} cells carry a value"

    return {
        "ticker": str(ticker).upper(),
        "as_of": target_stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_version": JOINT_FORECAST_CONTRACT_VERSION,
        "pipeline_version": JOINT_FORECAST_PIPELINE_VERSION,
        "label_version": (labels or {}).get("label_version"),
        "status": status,
        "reason": reason,
        "emitted_values": emitted,
        # Explicit, because "120d" sorts before "1d" and a consumer iterating
        # dict keys would read the grid out of time order.
        "horizon_order": list(FORECAST_HORIZONS),
        "targets": list(FORECAST_TARGETS),
        "benchmark": benchmark,
        "model": model,
        "rows": rows,
        "coherence": evaluate_coherence(rows),
        "findings": contract_findings(labels),
        "rejected_rules": dict(JOINT_REJECTED_RULES),
    }


def evaluate_coherence(rows: dict) -> dict:
    """Cross-horizon consistency, over the cells that actually carry values.

    Deliberately thin. Four tempting rules were REJECTED by measurement (see
    `JOINT_REJECTED_RULES`); what remains are checks that cannot fire on
    legitimate market behaviour.
    """
    emitted = [
        (horizon, target, cell)
        for horizon, row in rows.items()
        for target, cell in (row.get("cells") or {}).items()
        if cell.get("status") == CELL_STATUS_OK
    ]

    checked = [
        "probability_in_unit_interval",
        "value_within_declared_bounds",
    ]
    not_checked = [
        {"rule": name, "why": why} for name, why in JOINT_REJECTED_RULES.items()
    ]

    if not emitted:
        return {
            "status": JOINT_COHERENCE_NOT_EVALUATED,
            "violations": [],
            "flags": [],
            "checked": [],
            "not_checked": not_checked,
            "reason": "no cell carries a value, so there is nothing to be consistent about",
        }

    violations: list[dict] = []
    for horizon, target, cell in emitted:
        problems = bounds_problems(target, cell.get("value"))
        if problems:
            violations.append({
                "rule": "value_within_declared_bounds",
                "horizons": [horizon],
                "detail": f"{target}@{horizon}: {problems[0]}",
            })

    status = JOINT_COHERENCE_VIOLATED if violations else JOINT_COHERENCE_OK
    return {
        "status": status,
        "violations": violations,
        "flags": [],
        "checked": checked,
        "not_checked": not_checked,
        "reason": "",
    }


def joint_problems(forecast: dict) -> list[str]:
    """Validate a built joint forecast against the F3 contract."""
    problems: list[str] = []
    if not isinstance(forecast, dict):
        return ["joint forecast must be a dict"]

    for field in ("ticker", "as_of", "status", "contract_version", "horizon_order"):
        if not forecast.get(field):
            problems.append(f"joint forecast field {field!r} missing/empty")
    if problems:
        return problems

    if forecast["horizon_order"] != list(FORECAST_HORIZONS):
        problems.append(
            "horizon_order is not the declared shortest-first order — a consumer "
            "reading it would render the grid out of time order"
        )

    rows = forecast.get("rows") or {}
    for horizon in FORECAST_HORIZONS:
        if horizon not in rows:
            problems.append(f"horizon {horizon!r} absent — the grid must be complete")
            continue
        cells = rows[horizon].get("cells") or {}
        for target in FORECAST_TARGETS:
            if target not in cells:
                problems.append(f"cell {target}@{horizon} absent — the grid must be complete")
                continue
            cell = cells[target]
            status = cell.get("status")
            if status not in JOINT_CELL_PRECEDENCE:
                problems.append(f"{target}@{horizon}: undeclared status {status!r}")
            # THE shape rule.
            if status == CELL_STATUS_OK and "value" not in cell:
                problems.append(f"{target}@{horizon}: an OK cell carries no value")
            if status != CELL_STATUS_OK and "value" in cell:
                problems.append(
                    f"{target}@{horizon}: a {status} cell carries a value key — it would "
                    f"coalesce to 0 in a consumer and render as a real reading"
                )
            if status != CELL_STATUS_OK and not cell.get("reason"):
                problems.append(f"{target}@{horizon}: a refused cell must explain itself")
            if status != CELL_STATUS_OK and (cell.get("interval") or cell.get("dispersion")):
                problems.append(
                    f"{target}@{horizon}: a refused cell supplies uncertainty — an "
                    f"interval beside a withheld estimate is an estimate by another name"
                )

    if forecast.get("status") != JOINT_STATUS_OK and not forecast.get("reason"):
        problems.append("a non-OK joint forecast must explain itself")

    model = forecast.get("model") or {}
    if not model.get("status"):
        problems.append("the model block carries no status")
    if model.get("status") == MODEL_STATUS_NONE and not model.get("reason"):
        problems.append("an absent model must say why")

    emitted = sum(
        1
        for row in rows.values()
        for cell in (row.get("cells") or {}).values()
        if cell.get("status") == CELL_STATUS_OK
    )
    if emitted != forecast.get("emitted_values"):
        problems.append(
            f"emitted_values says {forecast.get('emitted_values')} but {emitted} cells are OK"
        )
    return problems


def render_rows(forecast: dict, targets: tuple[str, ...] = ("expected_return", "probability_up")) -> list[dict]:
    """The roadmap's reading order: one line per horizon, shortest first.

    Returns display-ready rows. A refused cell yields None for that column and
    carries its reason, so a renderer shows why a number is missing instead of
    showing a zero.
    """
    lines: list[dict] = []
    for horizon in forecast.get("horizon_order") or []:
        row = (forecast.get("rows") or {}).get(horizon) or {}
        cells = row.get("cells") or {}
        line = {"horizon": horizon, "readiness": row.get("readiness")}
        for target in targets:
            cell = cells.get(target) or {}
            # `.get("value")` is safe HERE precisely because the key is absent
            # on a refused cell: there is no null to coalesce.
            line[target] = cell.get("value")
            line[f"{target}_status"] = cell.get("status")
            line[f"{target}_reason"] = cell.get("reason")
        lines.append(line)
    return lines
