"""CI drift gate for L3 error memory.

The load-bearing property is SILENCE ON A CLEAN SYSTEM. An error detector that
finds something every run is worse than none: it trains the operator to
ignore it, and the one real finding arrives among the noise.

1.  A CLEAN SYSTEM PRODUCES NO FINDINGS. MEASURED, at the conventional
    p<0.05 over ~36 cells a perfectly clean system reports 2.77 systematic
    errors and flags something on 94% of runs;
2.  ...and a REAL bias is still found, so the strictness is a threshold and
    not a gag;
3.  systematic means BIAS, not a large error. MEASURED, a noisy-but-unbiased
    forecaster and a systematically over-bullish one score Brier 0.2466 vs
    0.3300 — barely separable — while their biases are -0.046 and +0.285;
4.  the t-threshold stays strict BECAUSE many cells are scanned, and the
    comparison count travels with every finding;
5.  the sample floor holds. MEASURED, detection of a real 0.25 bias is 7% at
    n=12 and 31% at n=30, so below the floor the test is close to blind;
6.  NOT_ENOUGH_DATA stays distinct from NO_BIAS_DETECTED — "we could not
    test" and "we tested and found nothing" have different fixes;
7.  an untested cell carries NO bias figures, or a `bias: None` coalesces to
    0.0 and renders an unexamined slice as perfectly unbiased;
8.  a finding names its DIRECTION; a signed number means nothing without it;
9.  power is reported, so "no finding" is read as a statement about sample
    size rather than about the forecaster;
10. L3 READS the L1 ledger rather than storing forecast/actual/error again.

Synthetic and deterministic.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CLOSURE_CLOSED,
    ERROR_DIRECTION_OVERCONFIDENT,
    ERROR_DIRECTION_UNDERCONFIDENT,
    ERROR_MEMORY_MIN_BIAS,
    ERROR_MEMORY_MIN_SAMPLES,
    ERROR_MEMORY_POWER_EVIDENCE,
    ERROR_MEMORY_T_THRESHOLD,
    ERROR_VERDICT_NEGLIGIBLE,
    ERROR_VERDICT_NO_BIAS_DETECTED,
    ERROR_VERDICT_NOT_ENOUGH_DATA,
    PERFORMANCE_DIMENSIONS,
    SCORE_METHOD_BRIER,
)
from core.error_memory import (  # noqa: E402
    ErrorMemoryError,
    bias_test,
    cell_finding,
    classify_bias,
    detection_power,
    error_memory_problems,
    error_memory_report,
    error_records,
    render_findings,
    scan_dimension,
)


def _rows(count, predicted, true_rate, seed, **context):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        actual = 1.0 if rng.random() < true_rate else 0.0
        row = {
            "state": CLOSURE_CLOSED, "forecast_id": f"f{seed}-{index}",
            "ticker": "NVDA", "as_of": "2024-06-15", "horizon": "20d",
            "regime": "bullish", "event_type": "earnings",
            "volatility": 0.22, "observed_share": 1.0, "confidence": 0.8,
            "score": {
                "method": SCORE_METHOD_BRIER, "scored": True,
                "predicted": float(predicted), "actual": actual,
                "brier": (predicted - actual) ** 2,
            },
        }
        row.update(context)
        rows.append(row)
    return rows


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    # A CLEAN system, scanned across every dimension, must stay silent.
    clean_reports = 0
    noisy_reports = 0
    for seed in range(12):
        rng = random.Random(1000 + seed)
        clean = []
        for index in range(300):
            clean.extend(
                _rows(
                    1, 0.55, 0.55, seed * 1000 + index,
                    regime=rng.choice(["bullish", "risk_off", "range"]),
                    horizon=rng.choice(["1d", "5d", "20d", "60d"]),
                    ticker=rng.choice(["NVDA", "AAPL", "MSFT", "AMD"]),
                )
            )
        report = error_memory_report(clean)
        if report["systematic_findings"]:
            noisy_reports += 1
        else:
            clean_reports += 1
        for problem in error_memory_problems(report):
            failures.append(f"clean report problem: {problem}")

    if clean_reports < 9:
        failures.append(
            f"only {clean_reports}/12 clean systems produced NO finding — "
            f"MEASURED, at t>1.96 a clean system flags something on 94% of "
            f"runs, which trains an operator to ignore the alert"
        )

    # ---------------------------------------------------------------- 2
    # ...but a REAL bias is still caught.
    biased = (
        _rows(200, 0.55, 0.55, 7, regime="bullish")
        + _rows(80, 0.80, 0.45, 8, regime="stress")
    )
    biased_report = error_memory_report(biased)
    stress = [
        f for f in biased_report["systematic_findings"]
        if f["dimension"] == "regime" and f["level"] == "stress"
    ]
    if not stress:
        failures.append(
            "a forecaster over-predicting by 0.35 in one regime was not "
            "flagged — the threshold has become a gag rather than a filter"
        )
    else:
        finding = stress[0]
        if finding.get("direction") != ERROR_DIRECTION_OVERCONFIDENT:
            failures.append(
                f"an over-predicting bias was labelled "
                f"{finding.get('direction')!r}"
            )
        if not finding.get("comparisons"):
            failures.append("a finding carries no comparison count")
        if finding.get("comparisons") != biased_report["comparisons"]:
            failures.append(
                "a finding carries its own dimension's comparison count rather "
                "than the full scan width — its credibility depends on how "
                "many chances it had across the whole report"
            )
    for problem in error_memory_problems(biased_report):
        failures.append(f"biased report problem: {problem}")

    # An UNDER-predicting bias is named correctly too.
    under = (
        _rows(200, 0.55, 0.55, 11, regime="bullish")
        + _rows(80, 0.25, 0.70, 12, regime="stress")
    )
    under_findings = [
        f for f in error_memory_report(under)["systematic_findings"]
        if f["level"] == "stress"
    ]
    if under_findings and under_findings[0].get("direction") != (
        ERROR_DIRECTION_UNDERCONFIDENT
    ):
        failures.append(
            f"an under-predicting bias was labelled "
            f"{under_findings[0].get('direction')!r}"
        )

    # ---------------------------------------------------------------- 3
    # Systematic means BIAS, not a large error.
    noisy = bias_test([0.4, -0.4, 0.35, -0.38, 0.42, -0.41] * 8)
    skewed = bias_test([0.25] * 48)
    if abs(noisy["t_statistic"]) >= ERROR_MEMORY_T_THRESHOLD:
        failures.append(
            f"a large-but-unbiased error set scored t={noisy['t_statistic']} — "
            f"systematic means a direction that persists, not an error that "
            f"is big"
        )
    if abs(skewed["t_statistic"]) <= ERROR_MEMORY_T_THRESHOLD:
        failures.append(
            f"a consistently one-sided error set scored only "
            f"t={skewed['t_statistic']}"
        )

    # ---------------------------------------------------------------- 4 + 5
    if ERROR_MEMORY_T_THRESHOLD < 2.58:
        failures.append(
            f"the t-threshold fell to {ERROR_MEMORY_T_THRESHOLD} — MEASURED, "
            f"t>1.96 over ~36 cells yields 2.77 false alarms per report"
        )
    if ERROR_MEMORY_MIN_SAMPLES < 30:
        failures.append(
            f"the sample floor fell to {ERROR_MEMORY_MIN_SAMPLES} — MEASURED, "
            f"a real 0.25 bias is detected 7% of the time at n=12"
        )
    thin = classify_bias(bias_test([0.3] * (ERROR_MEMORY_MIN_SAMPLES - 1)))
    if thin[0] != ERROR_VERDICT_NOT_ENOUGH_DATA:
        failures.append(
            f"a cell one short of the floor was tested anyway (got {thin[0]})"
        )

    # ---------------------------------------------------------------- 6 + 7
    untested = cell_finding(
        [{"error": 0.3, "predicted": 0.8, "actual": 0.5} for _ in range(5)]
    )
    if untested["verdict"] != ERROR_VERDICT_NOT_ENOUGH_DATA:
        failures.append("a 5-record cell was tested")
    for key in ("bias", "t_statistic", "direction"):
        if key in untested:
            failures.append(
                f"an untested cell carries {key!r} — it would coalesce to 0.0 "
                f"in a consumer and render an unexamined slice as perfectly "
                f"unbiased"
            )
    if not untested.get("reason"):
        failures.append("an untested cell does not say why")
    if ERROR_VERDICT_NOT_ENOUGH_DATA == ERROR_VERDICT_NO_BIAS_DETECTED:
        failures.append(
            "'we could not test' collapsed into 'we tested and found nothing'"
        )

    # A significant but trivial bias is NEGLIGIBLE, not SYSTEMATIC.
    tiny = classify_bias(
        {"samples": 400, "bias": ERROR_MEMORY_MIN_BIAS / 2,
         "t_statistic": 9.0, "standard_error": 0.001}
    )
    if tiny[0] != ERROR_VERDICT_NEGLIGIBLE:
        failures.append(
            f"a bias of {ERROR_MEMORY_MIN_BIAS / 2} at t=9 was reported "
            f"{tiny[0]} — with enough observations a trivial bias becomes "
            f"significant without becoming important"
        )

    # ---------------------------------------------------------------- 9
    if not ERROR_MEMORY_POWER_EVIDENCE or "MEASURED" not in ERROR_MEMORY_POWER_EVIDENCE:
        failures.append("the power evidence lost its measurement")
    if detection_power(ERROR_MEMORY_MIN_SAMPLES) >= 0.9:
        failures.append(
            "detection power at the sample floor is reported as >=90% — "
            "MEASURED it is 31% at n=30, and overstating it would make "
            "'no finding' read as 'no bias'"
        )
    if detection_power(200) < 0.9:
        failures.append("detection power at n=200 is understated")
    if not biased_report.get("absence_note"):
        failures.append(
            "the report does not say that no finding is not evidence of no bias"
        )

    # ---------------------------------------------------------------- 10
    from core import error_memory

    source = Path(error_memory.__file__).read_text(encoding="utf-8")
    for forbidden in ("def record_", "open(", "write_text"):
        if forbidden in source:
            failures.append(
                f"error_memory appears to WRITE a store ({forbidden!r}) — L1 "
                f"already stores forecast, actual, error and context, and a "
                f"second copy is the split-brain W5 forbids"
            )
    if "error_records" not in source:
        failures.append("error_memory no longer reads the L1 ledger")

    # Unscored rows contribute nothing but are not silently invented either.
    unscored = [
        {"state": CLOSURE_CLOSED, "score": {"scored": False, "method": "not_scored"}}
    ]
    if error_records(unscored):
        failures.append("an unscored row produced an error record")

    if list((biased_report.get("scans") or {})) != list(PERFORMANCE_DIMENSIONS):
        failures.append("not every dimension was scanned, in order")
    try:
        scan_dimension([], "not_a_dimension")
        failures.append("an unknown dimension was accepted")
    except ErrorMemoryError:
        pass

    for row in render_findings(biased_report):
        if not row.get("direction"):
            failures.append(f"{row['dimension']}[{row['level']}]: renders no direction")

    if failures:
        print("L3 error-memory gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("L3 error-memory gate OK:")
    print(
        f"  {clean_reports}/12 clean systems produced NO finding "
        f"(t>{ERROR_MEMORY_T_THRESHOLD}, floor {ERROR_MEMORY_MIN_SAMPLES})."
    )
    print("  a real 0.35 regime bias is still caught, named and directed.")
    print("  systematic means BIAS: a large unbiased error set stays silent.")
    print(
        f"  power is reported honestly: {detection_power(ERROR_MEMORY_MIN_SAMPLES):.0%} "
        f"at the floor, {detection_power(100):.0%} at n=100."
    )
    print("  L3 READS the L1 ledger; it never stores a second copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
