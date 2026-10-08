"""L2 performance-ledger tests.

The behaviour under test: point-in-time dimensions are captured rather than
recomputed, breakdowns are marginal, and a thin cell publishes its count and
no metric.
"""

from __future__ import annotations

import unittest

from core.config import (
    CLOSURE_CLOSED,
    CLOSURE_OPEN,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    PERF_CONFIDENCE_BUCKET,
    PERF_EVENT_TYPE,
    PERF_HORIZON,
    PERF_MODEL,
    PERF_REGIME,
    PERF_SECTOR,
    PERF_SOURCE,
    PERF_TICKER,
    PERF_VOLATILITY_REGIME,
    PERFORMANCE_CAPTURED_AT_RECORD,
    PERFORMANCE_CONFIDENCE_BUCKETS,
    PERFORMANCE_CROSS_TABULATE,
    PERFORMANCE_DERIVED,
    PERFORMANCE_DIMENSIONS,
    PERFORMANCE_MIN_CELL,
    PERFORMANCE_SPARSITY_EVIDENCE,
    PERFORMANCE_VOLATILITY_BANDS,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_NOT_SCORED,
)
from core.performance_ledger import (
    UNKNOWN,
    PerformanceLedgerError,
    breakdown,
    cell_metrics,
    dimension_value,
    performance_problems,
    performance_report,
    render_breakdown,
)

METRICS = ("mean_brier", "coverage_rate", "direction_accuracy")


def scored(predicted, actual, **extra):
    row = {
        "state": CLOSURE_CLOSED, "ticker": "NVDA", "horizon": "20d",
        "regime": "bullish", "event_type": "earnings",
        "observed_share": 1.0, "volatility": 0.22, "confidence": 0.8,
        "forecast_version": {"snapshot": "forecast-snapshot-v1"},
        "score": {
            "method": SCORE_METHOD_BRIER, "scored": True,
            "predicted": float(predicted), "actual": float(actual),
            "brier": float((predicted - actual) ** 2),
        },
    }
    row.update(extra)
    return row


def refused(**extra):
    row = scored(0.5, 1.0)
    row["score"] = {
        "method": SCORE_METHOD_NOT_SCORED, "scored": False, "reason": "no claim",
    }
    row.update(extra)
    return row


def covered(**extra):
    row = scored(0.5, 1.0)
    row["score"] = {
        "method": "coverage", "scored": True,
        "actual": 1.0, "lower": 0.3, "upper": 1.0, "midpoint": 0.65,
        "brier": 0.1225, "coverage_is_group_property": True,
    }
    row.update(extra)
    return row


class CapturedDimensionTests(unittest.TestCase):
    """Point-in-time facts are read off the row, never recomputed."""

    def test_the_perishable_four_are_captured(self):
        for dimension in (PERF_REGIME, PERF_EVENT_TYPE, PERF_SOURCE,
                          PERF_VOLATILITY_REGIME):
            self.assertIn(dimension, PERFORMANCE_CAPTURED_AT_RECORD, dimension)

    def test_a_planted_regime_wins_over_any_recomputation(self):
        # MEASURED: a later recomputation depends on whichever classifier
        # version runs at closing time.
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, regime="stress"), PERF_REGIME),
            "stress",
        )

    def test_the_event_type_is_read_from_the_row(self):
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, event_type="m_and_a"), PERF_EVENT_TYPE),
            "m_and_a",
        )

    def test_an_unrecorded_dimension_reads_unknown(self):
        bare = {"state": CLOSURE_CLOSED, "ticker": "NVDA"}
        for dimension in PERFORMANCE_CAPTURED_AT_RECORD:
            self.assertEqual(dimension_value(bare, dimension), UNKNOWN, dimension)

    def test_captured_and_derived_do_not_overlap(self):
        self.assertFalse(
            set(PERFORMANCE_CAPTURED_AT_RECORD) & set(PERFORMANCE_DERIVED)
        )

    def test_the_source_split_is_binary(self):
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, observed_share=1.0), PERF_SOURCE),
            "observed",
        )
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, observed_share=0.0), PERF_SOURCE),
            "inferred",
        )
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, observed_share=None), PERF_SOURCE),
            UNKNOWN,
        )


class DerivedDimensionTests(unittest.TestCase):
    def test_the_sector_comes_from_the_ticker(self):
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, ticker="NVDA"), PERF_SECTOR),
            "Information Technology",
        )

    def test_a_fund_has_no_sector(self):
        self.assertEqual(
            dimension_value(scored(0.6, 1.0, ticker="VOO"), PERF_SECTOR), UNKNOWN
        )

    def test_confidence_falls_into_a_bucket(self):
        for value, expected in ((0.0, "NONE"), (0.3, "LOW"), (0.6, "MODERATE"),
                                (0.9, "HIGH")):
            self.assertEqual(
                dimension_value(scored(0.6, 1.0, confidence=value),
                                PERF_CONFIDENCE_BUCKET),
                expected,
                value,
            )

    def test_volatility_falls_into_a_band(self):
        for value, expected in ((0.05, "CALM"), (0.22, "NORMAL"), (0.5, "TURBULENT")):
            self.assertEqual(
                dimension_value(scored(0.6, 1.0, volatility=value),
                                PERF_VOLATILITY_REGIME),
                expected,
                value,
            )

    def test_bands_ascend_and_start_at_zero(self):
        for bands in (PERFORMANCE_CONFIDENCE_BUCKETS, PERFORMANCE_VOLATILITY_BANDS):
            values = [value for _name, value in bands]
            self.assertEqual(values, sorted(values))
            self.assertEqual(values[0], 0.0)


class ThinCellTests(unittest.TestCase):
    """A performance number from three forecasts is the F4 stress cell."""

    def test_a_thin_cell_publishes_no_metric(self):
        cell = cell_metrics([scored(0.95, 0.0) for _ in range(3)])
        self.assertFalse(cell["sufficient"])
        for metric in METRICS:
            self.assertNotIn(metric, cell, metric)

    def test_a_thin_cell_reports_how_thin_it_was(self):
        cell = cell_metrics([scored(0.95, 0.0) for _ in range(3)])
        self.assertEqual(cell["scored"], 3)
        self.assertTrue(cell["reason"])

    def test_the_worst_thin_cell_is_still_silenced(self):
        # The eye-catching, least-earned case.
        table = breakdown(
            [scored(0.6, 1.0) for _ in range(20)]
            + [scored(0.95, 0.0, ticker="TINY") for _ in range(3)],
            PERF_TICKER,
        )
        tiny = table["cells"]["TINY"]
        for metric in METRICS:
            self.assertNotIn(metric, tiny, metric)
        self.assertIn("TINY", table["thin_levels"])

    def test_a_sufficient_cell_still_publishes(self):
        cell = cell_metrics(
            [scored(0.6, 1.0) for _ in range(PERFORMANCE_MIN_CELL + 2)]
        )
        self.assertTrue(cell["sufficient"])
        self.assertIn("mean_brier", cell)

    def test_the_floor_matches_f4s_interval_floor(self):
        # One sample-size policy governs a forecast and its evaluation.
        self.assertEqual(PERFORMANCE_MIN_CELL, CONDITIONAL_MIN_SAMPLES_INTERVAL)


class ShapeRuleTests(unittest.TestCase):
    def test_a_cell_with_no_briers_omits_the_key(self):
        # present-but-None coalesces to 0.0 and renders an unscored cell as
        # perfectly accurate. Built from rows that carry NO brier at all.
        rows = [covered() for _ in range(PERFORMANCE_MIN_CELL + 2)]
        for r in rows:
            r["score"].pop("brier")
        cell = cell_metrics(rows)
        self.assertNotIn("mean_brier", cell)
        self.assertIn("coverage_rate", cell)

    def test_coverage_is_computed_across_the_cell(self):
        # MEASURED: per-forecast coverage was False 42/42, because a binary
        # outcome cannot land inside a probability range.
        cell = cell_metrics([covered() for _ in range(PERFORMANCE_MIN_CELL + 2)])
        self.assertTrue(cell["coverage_is_group_property"])
        self.assertIsNotNone(cell["realised_rate"])

    def test_coverage_can_fail(self):
        rows = [covered() for _ in range(PERFORMANCE_MIN_CELL + 2)]
        for r in rows:
            r["score"].update({"actual": 0.0, "lower": 0.6, "upper": 0.9})
        self.assertEqual(cell_metrics(rows)["coverage_rate"], 0.0)

    def test_no_published_metric_is_none(self):
        cell = cell_metrics([scored(0.6, 1.0) for _ in range(20)])
        for metric in METRICS:
            if metric in cell:
                self.assertIsNotNone(cell[metric], metric)

    def test_coverage_travels_with_every_metric(self):
        cell = cell_metrics(
            [scored(0.6, 1.0) for _ in range(12)] + [refused() for _ in range(5)]
        )
        self.assertIn("mean_brier", cell)
        for required in ("scored", "refused", "forecasts", "scored_share"):
            self.assertIsNotNone(cell[required], required)

    def test_refusals_are_counted_inside_a_cell(self):
        cell = cell_metrics(
            [scored(0.6, 1.0) for _ in range(10)] + [refused() for _ in range(5)]
        )
        self.assertEqual(cell["refused"], 5)
        self.assertEqual(cell["forecasts"], 15)


class BreakdownTests(unittest.TestCase):
    def test_breakdowns_are_marginal(self):
        self.assertFalse(PERFORMANCE_CROSS_TABULATE)
        table = breakdown([scored(0.6, 1.0) for _ in range(20)], PERF_REGIME)
        for level in table["levels"]:
            self.assertIsInstance(level, str)

    def test_the_sparsity_evidence_is_measured(self):
        self.assertIn("MEASURED", PERFORMANCE_SPARSITY_EVIDENCE)

    def test_an_unknown_dimension_is_refused(self):
        with self.assertRaises(PerformanceLedgerError):
            breakdown([], "not_a_dimension")
        with self.assertRaises(PerformanceLedgerError):
            dimension_value({}, "not_a_dimension")

    def test_only_closed_rows_are_counted(self):
        rows = [scored(0.6, 1.0) for _ in range(10)]
        rows.append({**scored(0.6, 1.0), "state": CLOSURE_OPEN})
        self.assertEqual(breakdown(rows, PERF_TICKER)["forecasts"], 10)

    def test_levels_split_by_their_dimension(self):
        rows = (
            [scored(0.6, 1.0, regime="bullish") for _ in range(10)]
            + [scored(0.6, 1.0, regime="stress") for _ in range(10)]
        )
        table = breakdown(rows, PERF_REGIME)
        self.assertEqual(table["levels"], ["bullish", "stress"])

    def test_an_empty_breakdown_is_contract_clean(self):
        table = breakdown([], PERF_TICKER)
        self.assertEqual(table["levels"], [])
        self.assertEqual(table["forecasts"], 0)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = performance_report([scored(0.6, 1.0) for _ in range(20)])

    def test_every_dimension_is_reported_in_order(self):
        self.assertEqual(
            list(self.report["breakdowns"]), list(PERFORMANCE_DIMENSIONS)
        )

    def test_the_sprint_names_nine_dimensions(self):
        self.assertEqual(len(PERFORMANCE_DIMENSIONS), 9)

    def test_a_healthy_report_raises_nothing(self):
        self.assertEqual(performance_problems(self.report), [])

    def test_an_empty_report_raises_nothing(self):
        self.assertEqual(performance_problems(performance_report([])), [])

    def test_a_cross_tab_claim_is_reported(self):
        broken = dict(self.report)
        broken["cross_tabulated"] = True
        self.assertTrue(any("cross-tab" in p for p in performance_problems(broken)))

    def test_a_thin_cell_publishing_a_metric_is_reported(self):
        broken = performance_report([scored(0.9, 0.0) for _ in range(3)])
        cell = broken["breakdowns"][PERF_TICKER]["cells"]["NVDA"]
        cell["mean_brier"] = 0.81
        self.assertTrue(any("below the" in p for p in performance_problems(broken)))

    def test_a_missing_dimension_is_reported(self):
        broken = dict(self.report)
        broken["breakdowns"] = dict(broken["breakdowns"])
        broken["breakdowns"].pop(PERF_MODEL)
        self.assertTrue(
            any("every declared dimension" in p for p in performance_problems(broken))
        )


class RenderTests(unittest.TestCase):
    def test_render_never_leaves_a_thin_cell_blank(self):
        table = breakdown(
            [scored(0.6, 1.0) for _ in range(20)]
            + [scored(0.9, 0.0, ticker="TINY") for _ in range(2)],
            PERF_TICKER,
        )
        for row in render_breakdown(table):
            if not row["sufficient"]:
                self.assertTrue(row["reason"], row["level"])

    def test_render_covers_every_level(self):
        table = breakdown(
            [scored(0.6, 1.0, horizon=h) for h in ("1d", "5d", "20d")] * 10,
            PERF_HORIZON,
        )
        self.assertEqual(
            sorted(r["level"] for r in render_breakdown(table)),
            ["1d", "20d", "5d"],
        )


if __name__ == "__main__":
    unittest.main()
