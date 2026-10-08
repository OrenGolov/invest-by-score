"""L3 error-memory tests.

The behaviour under test is SILENCE on a clean system. A detector that finds
something every run trains the operator to ignore it, and the one real
finding then arrives among the noise.
"""

from __future__ import annotations

import math
import random
import unittest

from core.config import (
    CLOSURE_CLOSED,
    ERROR_DIRECTION_OVERCONFIDENT,
    ERROR_DIRECTION_UNDERCONFIDENT,
    ERROR_MEMORY_MIN_BIAS,
    ERROR_MEMORY_MIN_SAMPLES,
    ERROR_MEMORY_POWER_EVIDENCE,
    ERROR_MEMORY_T_THRESHOLD,
    ERROR_MEMORY_VERDICTS,
    ERROR_VERDICT_NEGLIGIBLE,
    ERROR_VERDICT_NO_BIAS_DETECTED,
    ERROR_VERDICT_NOT_ENOUGH_DATA,
    ERROR_VERDICT_SYSTEMATIC,
    PERFORMANCE_DIMENSIONS,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_NOT_SCORED,
)
from core.error_memory import (
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


def rows(count, predicted, true_rate, seed, **context):
    rng = random.Random(seed)
    built = []
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
        built.append(row)
    return built


class SilenceOnCleanSystemTests(unittest.TestCase):
    """MEASURED: at t>1.96 a clean system flags something on 94% of runs."""

    def test_a_clean_system_is_usually_silent(self):
        silent = 0
        for seed in range(12):
            rng = random.Random(500 + seed)
            clean = []
            for index in range(300):
                clean.extend(
                    rows(
                        1, 0.55, 0.55, seed * 977 + index,
                        regime=rng.choice(["bullish", "risk_off", "range"]),
                        horizon=rng.choice(["1d", "5d", "20d", "60d"]),
                        ticker=rng.choice(["NVDA", "AAPL", "MSFT"]),
                    )
                )
            if not error_memory_report(clean)["systematic_findings"]:
                silent += 1
        self.assertGreaterEqual(silent, 9, f"only {silent}/12 clean runs silent")

    def test_the_threshold_is_strict_because_many_cells_are_scanned(self):
        self.assertGreaterEqual(ERROR_MEMORY_T_THRESHOLD, 2.58)

    def test_the_sample_floor_supports_a_test(self):
        # MEASURED: a real 0.25 bias is detected 7% of the time at n=12.
        self.assertGreaterEqual(ERROR_MEMORY_MIN_SAMPLES, 30)


class RealBiasIsCaughtTests(unittest.TestCase):
    """The strictness must be a filter, not a gag."""

    def setUp(self):
        self.report = error_memory_report(
            rows(200, 0.55, 0.55, 7, regime="bullish")
            + rows(80, 0.80, 0.45, 8, regime="stress")
        )

    def test_a_regime_bias_is_found(self):
        stress = [
            f for f in self.report["systematic_findings"]
            if f["dimension"] == "regime" and f["level"] == "stress"
        ]
        self.assertTrue(stress, "a 0.35 over-prediction was not flagged")

    def test_the_finding_names_its_direction(self):
        stress = [
            f for f in self.report["systematic_findings"] if f["level"] == "stress"
        ][0]
        self.assertEqual(stress["direction"], ERROR_DIRECTION_OVERCONFIDENT)

    def test_an_under_prediction_is_named_correctly(self):
        report = error_memory_report(
            rows(200, 0.55, 0.55, 11, regime="bullish")
            + rows(80, 0.25, 0.70, 12, regime="stress")
        )
        stress = [f for f in report["systematic_findings"] if f["level"] == "stress"]
        self.assertTrue(stress)
        self.assertEqual(stress[0]["direction"], ERROR_DIRECTION_UNDERCONFIDENT)

    def test_a_finding_carries_the_full_scan_width(self):
        # Credibility depends on how many chances it had across the whole
        # report, not within one column.
        for finding in self.report["systematic_findings"]:
            self.assertEqual(finding["comparisons"], self.report["comparisons"])

    def test_the_report_is_contract_clean(self):
        self.assertEqual(error_memory_problems(self.report), [])


class BiasNotMagnitudeTests(unittest.TestCase):
    """Systematic means a direction that persists, not an error that is big."""

    def test_a_large_but_unbiased_error_set_is_silent(self):
        test = bias_test([0.4, -0.4, 0.35, -0.38, 0.42, -0.41] * 8)
        self.assertLess(abs(test["t_statistic"]), ERROR_MEMORY_T_THRESHOLD)

    def test_a_consistently_one_sided_set_is_flagged(self):
        test = bias_test([0.25, 0.24, 0.26] * 16)
        self.assertGreater(abs(test["t_statistic"]), ERROR_MEMORY_T_THRESHOLD)

    def test_zero_variance_is_the_strongest_evidence_not_the_weakest(self):
        # A first version returned t=0.0 here, inverting the truth: 48
        # identical errors is the most consistent bias possible.
        test = bias_test([0.25] * 48)
        self.assertTrue(math.isinf(test["t_statistic"]))
        self.assertEqual(classify_bias(test)[0], ERROR_VERDICT_SYSTEMATIC)

    def test_a_constant_zero_error_is_a_perfect_forecaster(self):
        test = bias_test([0.0] * 48)
        self.assertEqual(test["t_statistic"], 0.0)
        self.assertEqual(classify_bias(test)[0], ERROR_VERDICT_NO_BIAS_DETECTED)

    def test_a_tiny_constant_bias_is_negligible_not_systematic(self):
        # Infinite t, but the magnitude floor still applies.
        test = bias_test([ERROR_MEMORY_MIN_BIAS / 5] * 48)
        self.assertTrue(math.isinf(test["t_statistic"]))
        self.assertEqual(classify_bias(test)[0], ERROR_VERDICT_NEGLIGIBLE)

    def test_a_single_error_cannot_be_tested(self):
        self.assertIsNone(bias_test([0.3])["bias"])
        self.assertIsNone(bias_test([])["t_statistic"])


class VerdictTests(unittest.TestCase):
    def test_not_enough_data_is_distinct_from_no_bias(self):
        self.assertNotEqual(
            ERROR_VERDICT_NOT_ENOUGH_DATA, ERROR_VERDICT_NO_BIAS_DETECTED
        )

    def test_a_thin_cell_is_not_tested(self):
        verdict, reason = classify_bias(
            bias_test([0.3] * (ERROR_MEMORY_MIN_SAMPLES - 1))
        )
        self.assertEqual(verdict, ERROR_VERDICT_NOT_ENOUGH_DATA)
        self.assertTrue(reason)

    def test_an_untested_cell_carries_no_bias_figures(self):
        # A `bias: None` would coalesce to 0.0 and render an unexamined
        # slice as perfectly unbiased.
        finding = cell_finding(
            [{"error": 0.3, "predicted": 0.8, "actual": 0.5} for _ in range(5)]
        )
        self.assertEqual(finding["verdict"], ERROR_VERDICT_NOT_ENOUGH_DATA)
        for key in ("bias", "t_statistic", "direction"):
            self.assertNotIn(key, finding, key)

    def test_a_tested_cell_carries_its_figures(self):
        finding = cell_finding(
            [{"error": 0.0, "predicted": 0.5, "actual": 0.5} for _ in range(40)]
        )
        self.assertIn("bias", finding)
        self.assertIn("t_statistic", finding)

    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(ERROR_MEMORY_VERDICTS[0], ERROR_VERDICT_NOT_ENOUGH_DATA)
        self.assertEqual(ERROR_MEMORY_VERDICTS[-1], ERROR_VERDICT_SYSTEMATIC)


class PowerTests(unittest.TestCase):
    """No finding is not evidence of no bias."""

    def test_power_at_the_floor_is_reported_honestly(self):
        self.assertLess(detection_power(ERROR_MEMORY_MIN_SAMPLES), 0.5)

    def test_power_rises_with_samples(self):
        values = [detection_power(n) for n in (30, 50, 100, 200)]
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(values[-1], 0.9)

    def test_power_below_the_floor_is_zero(self):
        self.assertEqual(detection_power(10), 0.0)

    def test_the_power_evidence_is_measured(self):
        self.assertIn("MEASURED", ERROR_MEMORY_POWER_EVIDENCE)

    def test_the_report_states_that_absence_is_not_evidence(self):
        report = error_memory_report(rows(60, 0.55, 0.55, 3))
        self.assertIn("not evidence", report["absence_note"].lower())


class ReadsTheLedgerTests(unittest.TestCase):
    """L1 already stores forecast, actual, error and context (W5)."""

    def test_records_are_read_from_closed_rows(self):
        records = error_records(rows(10, 0.6, 0.5, 4))
        self.assertEqual(len(records), 10)
        for record in records:
            self.assertIn("error", record)
            self.assertIn("regime", record)

    def test_an_unscored_row_yields_no_record(self):
        row = rows(1, 0.6, 0.5, 5)[0]
        row["score"] = {"method": SCORE_METHOD_NOT_SCORED, "scored": False}
        self.assertEqual(error_records([row]), [])

    def test_an_open_row_yields_no_record(self):
        row = rows(1, 0.6, 0.5, 6)[0]
        row["state"] = "OPEN"
        self.assertEqual(error_records([row]), [])

    def test_an_interval_midpoint_is_used_when_there_is_no_point(self):
        row = rows(1, 0.6, 0.5, 9)[0]
        row["score"] = {
            "method": "coverage", "scored": True, "midpoint": 0.65,
            "actual": 1.0, "brier": 0.1225,
        }
        records = error_records([row])
        self.assertEqual(records[0]["predicted"], 0.65)


class ScanTests(unittest.TestCase):
    def test_every_dimension_is_scanned_in_order(self):
        report = error_memory_report(rows(40, 0.6, 0.5, 2))
        self.assertEqual(list(report["scans"]), list(PERFORMANCE_DIMENSIONS))

    def test_an_unknown_dimension_is_refused(self):
        with self.assertRaises(ErrorMemoryError):
            scan_dimension([], "not_a_dimension")

    def test_an_empty_report_is_contract_clean(self):
        report = error_memory_report([])
        self.assertEqual(report["records"], 0)
        self.assertEqual(error_memory_problems(report), [])

    def test_render_returns_nothing_when_nothing_is_found(self):
        # An empty finding list is a real answer, not a missing one.
        self.assertEqual(render_findings(error_memory_report(rows(60, 0.55, 0.55, 1))), [])


if __name__ == "__main__":
    unittest.main()
