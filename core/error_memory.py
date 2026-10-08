"""Error memory (Sprint L3) — a persistent bias, not the worst cell.

"Store forecast, actual, error, context; identify systematic errors."

**The first clause was already done.** A closed L1 ledger row carries the
forecast, the actual outcome, the error and the context. This module reads
that store rather than duplicating it (W5). The real work is the second
clause, and it is not the breakdown table L2 already produces.

**A systematic error is a BIAS, not a large error.** MEASURED over 200
forecasts:

    case                          brier    bias      t
    noisy but unbiased           0.2466  -0.0459   -1.3
    systematically over-bullish  0.3300  +0.2850   +8.1

Brier barely separates them; the signed bias does. Systematic means a
direction that *persists*, not an error that is big.

**The worst cell in a table is not a finding.** L2 reports roughly 36 cells.
MEASURED with no real effect anywhere — every cell drawn from identical skill
— the true mean Brier is 0.2722 while the worst cell per report averages
0.3712. It looks 36% worse than the truth, every single time.

**THE DECIDING MEASUREMENT: scanning many cells manufactures findings.** With
no systematic error present:

    t>1.96 (p<0.05)    2.77 false alarms/report, 94% of reports flag something
    t>2.58 (p<0.01)    1.15 false alarms/report, 68% flag something
    t>3.29 (p<0.001)   0.39 false alarms/report, 34% flag something

At the conventional threshold a perfectly clean system reports ~2.8
systematic errors on 94% of runs, which trains an operator to ignore the
alert. So the threshold is strict *because* many cells are scanned, and the
comparison count travels with every finding.

**Threshold and sample floor were chosen together.** Detection of a real 0.25
bias falls as the threshold rises (40% → 7% at n=12), so no threshold rescues
a thin cell: the floor buys detection, the threshold buys silence. At n=30
with t>3.29 the pair gives 0.26 false alarms per report and 31% detection,
rising to 95% by n=100.

**Power is reported, not assumed.** At n=30 a real bias is found 31% of the
time, so the absence of a finding is not evidence of no bias — it is most
often a statement about sample size, and every report says so.
"""

from __future__ import annotations

import logging
import math

from core.config import (
    CLOSURE_CLOSED,
    ERROR_DIRECTION_OVERCONFIDENT,
    ERROR_DIRECTION_UNDERCONFIDENT,
    ERROR_MEMORY_CONTRACT_VERSION,
    ERROR_MEMORY_MIN_BIAS,
    ERROR_MEMORY_MIN_SAMPLES,
    ERROR_MEMORY_POWER_EVIDENCE,
    ERROR_MEMORY_REPORT_COMPARISONS,
    ERROR_MEMORY_REPORT_POWER,
    ERROR_MEMORY_T_THRESHOLD,
    ERROR_MEMORY_VERSION,
    ERROR_VERDICT_NEGLIGIBLE,
    ERROR_VERDICT_NO_BIAS_DETECTED,
    ERROR_VERDICT_NOT_ENOUGH_DATA,
    ERROR_VERDICT_SYSTEMATIC,
    PERFORMANCE_DIMENSIONS,
)

LOGGER = logging.getLogger("core.error_memory")

# MEASURED detection rates at the shipped threshold, by sample size. Declared
# as data so a report can state its own power rather than implying certainty.
_POWER_TABLE: tuple[tuple[int, float], ...] = (
    (30, 0.31),
    (50, 0.58),
    (100, 0.95),
    (200, 1.00),
)


class ErrorMemoryError(ValueError):
    """Raised when an error-memory request violates the L3 contract."""


def error_records(closed_rows: list[dict]) -> list[dict]:
    """The forecast/actual/error/context tuples L3 reasons over.

    Read from the L1 ledger, not stored again. A row that was never scored
    carries no error to learn from and is excluded — but the CALLER still
    sees it in the L1 coverage numbers, so nothing disappears silently.
    """
    records: list[dict] = []
    for row in closed_rows or []:
        if row.get("state") != CLOSURE_CLOSED:
            continue
        score = row.get("score") or {}
        if not score.get("scored"):
            continue
        predicted = score.get("predicted")
        if predicted is None:
            predicted = score.get("midpoint")
        actual = score.get("actual")
        if predicted is None or actual is None:
            continue
        records.append(
            {
                "forecast_id": row.get("forecast_id"),
                "ticker": row.get("ticker"),
                "as_of": row.get("as_of"),
                "horizon": row.get("horizon"),
                "predicted": float(predicted),
                "actual": float(actual),
                "error": round(float(predicted) - float(actual), 6),
                "brier": score.get("brier"),
                "regime": row.get("regime"),
                "event_type": row.get("event_type"),
                "volatility": row.get("volatility"),
                "observed_share": row.get("observed_share"),
                "confidence": row.get("confidence"),
            }
        )
    return records


def bias_test(errors: list[float]) -> dict:
    """Is the error signed in a consistent direction, or just noisy?

    Returns the mean signed error and its t-statistic. A large Brier with a
    zero bias is a noisy forecaster; a modest Brier with a persistent bias is
    a broken one, and only the second is fixable by correction.
    """
    values = [float(e) for e in (errors or []) if e is not None]
    count = len(values)
    if count < 2:
        return {
            "samples": count, "bias": None, "t_statistic": None,
            "standard_error": None,
        }

    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    standard_error = math.sqrt(variance / count) if variance > 0 else 0.0

    # ZERO VARIANCE IS THE STRONGEST EVIDENCE, NOT THE WEAKEST. A first
    # version returned t=0.0 whenever the errors were identical, which
    # inverted the truth: 48 identical errors of +0.25 is the most consistent
    # bias possible, while adding one part per million of jitter produced
    # t=34,000,000. Both readings were wrong for the same reason — a
    # degenerate standard error is not a measurement of uncertainty.
    #
    # When every error is identical, the verdict rests on the BIAS alone: a
    # non-zero constant error is systematic by definition, and a constant of
    # exactly zero is a perfect forecaster. The t is reported as infinite
    # rather than fabricated, and `classify_bias` still applies the magnitude
    # floor, so a tiny constant bias remains NEGLIGIBLE.
    if standard_error > 0:
        t_statistic = mean / standard_error
    elif mean == 0.0:
        t_statistic = 0.0
    else:
        t_statistic = math.inf if mean > 0 else -math.inf
    return {
        "samples": count,
        "bias": round(mean, 6),
        "t_statistic": (
            t_statistic if math.isinf(t_statistic) else round(t_statistic, 4)
        ),
        "standard_error": round(standard_error, 6),
    }


def detection_power(samples: int) -> float:
    """MEASURED detection of a real 0.25 bias at this sample size.

    Reported so that "no finding" is read as what it is: most often a
    statement about sample size rather than about the forecaster.
    """
    power = 0.0
    for floor, value in _POWER_TABLE:
        if samples >= floor:
            power = value
    return power


def classify_bias(test: dict) -> tuple[str, str]:
    """The strongest verdict this cell's evidence supports.

    NOT_ENOUGH_DATA is deliberately distinct from NO_BIAS_DETECTED: "we could
    not test" and "we tested and found nothing" are different facts with
    different fixes.
    """
    samples = int(test.get("samples") or 0)
    if samples < ERROR_MEMORY_MIN_SAMPLES:
        return (
            ERROR_VERDICT_NOT_ENOUGH_DATA,
            (
                f"{samples} scored forecast(s), below the "
                f"{ERROR_MEMORY_MIN_SAMPLES} floor; MEASURED, a real 0.25 bias "
                f"is detected only 7% of the time at n=12, so a test here "
                f"would be close to blind"
            ),
        )

    bias = test.get("bias")
    t_statistic = test.get("t_statistic")
    if bias is None or t_statistic is None:
        return ERROR_VERDICT_NOT_ENOUGH_DATA, "the bias could not be computed"

    if abs(float(t_statistic)) < ERROR_MEMORY_T_THRESHOLD:
        return (
            ERROR_VERDICT_NO_BIAS_DETECTED,
            (
                f"bias {float(bias):+.4f} at t={float(t_statistic):+.2f}, below "
                f"the {ERROR_MEMORY_T_THRESHOLD} threshold. That threshold is "
                f"strict because ~36 cells are scanned: at t>1.96 a clean "
                f"system flags something on 94% of runs"
            ),
        )

    if abs(float(bias)) < ERROR_MEMORY_MIN_BIAS:
        return (
            ERROR_VERDICT_NEGLIGIBLE,
            (
                f"bias {float(bias):+.4f} is statistically clear "
                f"(t={float(t_statistic):+.2f}) but smaller than the "
                f"{ERROR_MEMORY_MIN_BIAS} floor; with enough observations a "
                f"trivial bias becomes significant without becoming important"
            ),
        )

    return ERROR_VERDICT_SYSTEMATIC, ""


def _direction(bias: float) -> str:
    """Which way the bias runs, named rather than signed."""
    return (
        ERROR_DIRECTION_OVERCONFIDENT
        if bias > 0
        else ERROR_DIRECTION_UNDERCONFIDENT
    )


def cell_finding(records: list[dict], comparisons: int = 0) -> dict:
    """One cell's bias verdict, with its power and comparison count."""
    errors = [r["error"] for r in records or []]
    test = bias_test(errors)
    verdict, reason = classify_bias(test)

    finding = {
        "verdict": verdict,
        "samples": test["samples"],
        "reason": reason,
        "power": round(detection_power(test["samples"]), 4),
        "power_note": ERROR_MEMORY_POWER_EVIDENCE if ERROR_MEMORY_REPORT_POWER else "",
        "comparisons": comparisons if ERROR_MEMORY_REPORT_COMPARISONS else None,
    }

    # THE SHAPE RULE: bias figures appear only where the test actually ran.
    # A `bias: None` on an untested cell would coalesce to 0.0 in a consumer
    # and render an unexamined slice as perfectly unbiased.
    if verdict != ERROR_VERDICT_NOT_ENOUGH_DATA:
        finding["bias"] = test["bias"]
        finding["t_statistic"] = test["t_statistic"]
        finding["standard_error"] = test["standard_error"]
    if verdict == ERROR_VERDICT_SYSTEMATIC:
        finding["direction"] = _direction(float(test["bias"]))
        finding["mean_predicted"] = round(
            sum(r["predicted"] for r in records) / len(records), 6
        )
        finding["mean_actual"] = round(
            sum(r["actual"] for r in records) / len(records), 6
        )
    return finding


def scan_dimension(records: list[dict], dimension: str) -> dict:
    """Test every level of one dimension for a persistent bias."""
    if dimension not in PERFORMANCE_DIMENSIONS:
        raise ErrorMemoryError(
            f"unknown dimension {dimension!r} "
            f"(known: {list(PERFORMANCE_DIMENSIONS)})"
        )

    from core.performance_ledger import dimension_value

    grouped: dict[str, list[dict]] = {}
    for record in records or []:
        grouped.setdefault(dimension_value(record, dimension), []).append(record)

    comparisons = len(grouped)
    cells = {
        level: cell_finding(rows, comparisons)
        for level, rows in sorted(grouped.items())
    }
    systematic = [
        level for level, cell in cells.items()
        if cell["verdict"] == ERROR_VERDICT_SYSTEMATIC
    ]
    return {
        "dimension": dimension,
        "levels": sorted(cells),
        "cells": cells,
        "systematic": sorted(systematic),
        "comparisons": comparisons,
        "memory_version": ERROR_MEMORY_VERSION,
    }


def error_memory_report(closed_rows: list[dict]) -> dict:
    """Scan every dimension for systematic error, honestly."""
    records = error_records(closed_rows)

    scans = {
        dimension: scan_dimension(records, dimension)
        for dimension in PERFORMANCE_DIMENSIONS
    }
    total_comparisons = sum(scan["comparisons"] for scan in scans.values())

    # Every finding carries the FULL scan width, not just its own dimension's.
    # A finding's credibility depends on how many chances it had across the
    # whole report, not within one column.
    findings: list[dict] = []
    for dimension, scan in scans.items():
        for level in scan["systematic"]:
            cell = dict(scan["cells"][level])
            cell["comparisons"] = total_comparisons
            findings.append({"dimension": dimension, "level": level, **cell})

    overall = cell_finding(records, total_comparisons)

    return {
        "records": len(records),
        "dimension_order": list(PERFORMANCE_DIMENSIONS),
        "scans": scans,
        "systematic_findings": findings,
        "overall": overall,
        "comparisons": total_comparisons,
        "t_threshold": ERROR_MEMORY_T_THRESHOLD,
        "min_samples": ERROR_MEMORY_MIN_SAMPLES,
        "power": round(detection_power(len(records)), 4),
        "power_note": ERROR_MEMORY_POWER_EVIDENCE,
        "absence_note": (
            "no systematic finding is NOT evidence of no bias — MEASURED, a "
            "real 0.25 bias is detected 31% of the time at n=30"
        ),
        "memory_version": ERROR_MEMORY_VERSION,
        "contract_version": ERROR_MEMORY_CONTRACT_VERSION,
    }


def error_memory_problems(report: dict) -> list[str]:
    """Validate an error-memory report against the L3 contract."""
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a dict"]

    for field in ("memory_version", "contract_version", "power_note",
                  "absence_note"):
        if not report.get(field):
            problems.append(f"report field {field!r} missing/empty")

    if report.get("t_threshold") is not None:
        if float(report["t_threshold"]) < 2.58:
            problems.append(
                f"the t-threshold fell to {report['t_threshold']} — MEASURED, "
                f"t>1.96 over ~36 cells yields 2.77 false alarms per report "
                f"and flags something on 94% of clean runs"
            )
    if (report.get("min_samples") or 0) < ERROR_MEMORY_MIN_SAMPLES:
        problems.append(
            f"the sample floor fell to {report.get('min_samples')} — at n=12 a "
            f"real 0.25 bias is detected 7% of the time"
        )

    scans = report.get("scans") or {}
    if list(scans) != list(PERFORMANCE_DIMENSIONS):
        problems.append(
            "not every dimension was scanned, in order — a missing scan reads "
            "as an oversight where an empty one reads as a fact"
        )

    for dimension, scan in scans.items():
        for level, cell in (scan.get("cells") or {}).items():
            where = f"{dimension}[{level}]"
            verdict = cell.get("verdict")
            if verdict == ERROR_VERDICT_NOT_ENOUGH_DATA:
                for key in ("bias", "t_statistic", "direction"):
                    if key in cell:
                        problems.append(
                            f"{where}: an untested cell carries {key!r} — it "
                            f"would coalesce to 0.0 in a consumer and render "
                            f"an unexamined slice as perfectly unbiased"
                        )
                if not cell.get("reason"):
                    problems.append(f"{where}: an untested cell does not say why")
            if verdict == ERROR_VERDICT_SYSTEMATIC:
                if cell.get("direction") not in (
                    ERROR_DIRECTION_OVERCONFIDENT, ERROR_DIRECTION_UNDERCONFIDENT
                ):
                    problems.append(
                        f"{where}: a systematic finding names no direction — "
                        f"a signed number means nothing without it"
                    )
                if abs(float(cell.get("t_statistic") or 0)) < ERROR_MEMORY_T_THRESHOLD:
                    problems.append(
                        f"{where}: reported SYSTEMATIC at t="
                        f"{cell.get('t_statistic')}, below the threshold"
                    )
                if int(cell.get("samples") or 0) < ERROR_MEMORY_MIN_SAMPLES:
                    problems.append(
                        f"{where}: reported SYSTEMATIC on "
                        f"{cell.get('samples')} samples, below the floor"
                    )
                if not cell.get("comparisons"):
                    problems.append(
                        f"{where}: a finding carries no comparison count — a "
                        f"reader cannot weigh it without knowing how many "
                        f"chances it had to appear"
                    )

    for finding in report.get("systematic_findings") or []:
        if finding.get("comparisons") != report.get("comparisons"):
            problems.append(
                f"{finding.get('dimension')}[{finding.get('level')}]: carries "
                f"its own dimension's comparison count rather than the full "
                f"scan width"
            )
    return problems


def render_findings(report: dict) -> list[dict]:
    """One reading row per systematic finding. Empty is a real answer."""
    rows: list[dict] = []
    for finding in report.get("systematic_findings") or []:
        rows.append(
            {
                "dimension": finding.get("dimension"),
                "level": finding.get("level"),
                "direction": finding.get("direction"),
                "bias": finding.get("bias"),
                "t_statistic": finding.get("t_statistic"),
                "samples": finding.get("samples"),
                "mean_predicted": finding.get("mean_predicted"),
                "mean_actual": finding.get("mean_actual"),
                "comparisons": finding.get("comparisons"),
            }
        )
    return rows
