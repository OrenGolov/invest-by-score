"""L1 outcome-closure tests.

The behaviour under test is that a forecast is scored ONCE, by the measure its
claim tier earned, and that an error figure never travels without the coverage
it came from.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.config import (
    CLOSURE_CLOSEABLE_STATES,
    CLOSURE_CLOSED,
    CLOSURE_EXPIRED_NO_DATA,
    CLOSURE_MATURED,
    CLOSURE_MIN_FOR_CALIBRATION,
    CLOSURE_MIN_SCORED_SHARE,
    CLOSURE_OPEN,
    CLOSURE_SCORE_METHODS,
    CLOSURE_SCORE_REFUSALS,
    CLOSURE_STATES,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_COVERAGE,
    SCORE_METHOD_DIRECTION,
    SCORE_METHOD_NOT_SCORED,
)
from core.outcome_closure import (
    OutcomeClosureError,
    build_ledger_row,
    close_forecast,
    closure_problems,
    closure_report,
    closure_state,
    current_ledger,
    evaluate_calibration,
    forecast_id,
    ledger_row_problems,
    load_ledger,
    realised_outcome,
    record_forecast,
    score_forecast,
)

UP = {"forward_return": 0.04, "direction_up": True}
DOWN = {"forward_return": -0.04, "direction_up": False}

MATURED_LABELS = {
    "horizons": {"20d": {"forward_return": 0.04, "entry_bar": "a", "exit_bar": "b"}},
    "matured_horizons": ["20d"],
    "pending_horizons": [],
    "label_version": "outcome-label-v2",
}
PENDING_LABELS = {
    "horizons": {"20d": {"status": "PENDING"}},
    "matured_horizons": [],
    "pending_horizons": ["20d"],
}
NO_DATA_LABELS = {
    "horizons": {"20d": {"forward_return": None}},
    "matured_horizons": ["20d"],
    "pending_horizons": [],
}


def row(claim, **extra):
    payload = dict(
        ticker="NVDA", as_of="2024-06-15", horizon="20d",
        target="probability_up", claim=claim, samples=45,
    )
    payload.update(extra)
    return build_ledger_row(**payload)


def closed_point(predicted, actual):
    return {
        "claim": "POINT", "state": CLOSURE_CLOSED,
        "score": {
            "method": SCORE_METHOD_BRIER, "scored": True,
            "predicted": float(predicted), "actual": float(actual),
            "brier": float((predicted - actual) ** 2),
        },
    }


def refused_row():
    return {
        "claim": "INSUFFICIENT", "state": CLOSURE_CLOSED,
        "score": {"method": SCORE_METHOD_NOT_SCORED, "scored": False,
                  "reason": "no claim"},
    }


class ScoringPerTierTests(unittest.TestCase):
    """One universal error function cannot serve four kinds of claim."""

    def test_a_point_claim_is_scored_by_brier(self):
        result = score_forecast(row("POINT", value=0.9), DOWN)
        self.assertEqual(result["method"], SCORE_METHOD_BRIER)
        self.assertAlmostEqual(result["brier"], 0.81, places=9)

    def test_an_interval_claim_is_scored_by_coverage(self):
        # Brier on a range is undefined; coverage is what the interval claimed.
        result = score_forecast(
            row("INTERVAL", interval={"lower": 0.3, "upper": 0.8}), UP
        )
        self.assertEqual(result["method"], SCORE_METHOD_COVERAGE)
        self.assertNotIn("brier", result)
        self.assertIn("covered", result)

    def test_coverage_records_a_miss(self):
        result = score_forecast(
            row("INTERVAL", interval={"lower": 0.3, "upper": 0.8}), DOWN
        )
        self.assertTrue(result["scored"])
        self.assertFalse(result["covered"])

    def test_a_directional_claim_is_scored_by_hit(self):
        hit = score_forecast(row("DIRECTIONAL", direction="HIGHER"), UP)
        miss = score_forecast(row("DIRECTIONAL", direction="HIGHER"), DOWN)
        self.assertEqual(hit["method"], SCORE_METHOD_DIRECTION)
        self.assertTrue(hit["hit"])
        self.assertTrue(miss["scored"])
        self.assertFalse(miss["hit"])

    def test_a_refusal_is_not_scored(self):
        # Zero error rewards silence; maximum error punishes honesty.
        result = score_forecast(row("INSUFFICIENT"), UP)
        self.assertEqual(result["method"], SCORE_METHOD_NOT_SCORED)
        self.assertFalse(result["scored"])
        self.assertFalse(CLOSURE_SCORE_REFUSALS)

    def test_every_tier_has_a_declared_method(self):
        self.assertEqual(
            set(CLOSURE_SCORE_METHODS),
            {"POINT", "INTERVAL", "DIRECTIONAL", "INSUFFICIENT"},
        )

    def test_a_claim_missing_its_payload_is_not_scored(self):
        self.assertFalse(
            score_forecast({"claim": "POINT", "score_method": SCORE_METHOD_BRIER}, UP)["scored"]
        )
        self.assertFalse(
            score_forecast({"claim": "INTERVAL", "score_method": SCORE_METHOD_COVERAGE}, UP)["scored"]
        )

    def test_no_realised_direction_means_no_score(self):
        self.assertFalse(
            score_forecast(row("POINT", value=0.6), {"direction_up": None})["scored"]
        )


class GamingTests(unittest.TestCase):
    """MEASURED: refusing the hardest cases improves Brier 11x."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.predictions = np.clip(rng.normal(0.58, 0.12, 200), 0.02, 0.98)
        self.actuals = (rng.random(200) < self.predictions).astype(float)
        self.order = np.argsort(np.abs(self.predictions - 0.5))[::-1]

    def _report(self, keep_hardest):
        keep = set(self.order[:keep_hardest].tolist())
        return closure_report([
            closed_point(self.predictions[i], self.actuals[i]) if i in keep
            else refused_row()
            for i in range(200)
        ])

    def test_refusing_lowers_the_apparent_error(self):
        # The fixture that justifies publishing coverage.
        self.assertLess(self._report(2)["mean_brier"], self._report(200)["mean_brier"])

    def test_a_heavily_refused_report_is_marked_unreliable(self):
        gamed = self._report(2)
        self.assertFalse(gamed["reliable"])
        self.assertTrue(gamed["reliability_reason"])

    def test_a_fully_scored_report_is_reliable(self):
        self.assertTrue(self._report(200)["reliable"])

    def test_coverage_travels_with_the_error(self):
        report = self._report(2)
        for name in ("scored", "refused", "pending", "scored_share"):
            self.assertIsNotNone(report.get(name), name)

    def test_an_error_without_its_coverage_is_a_contract_problem(self):
        report = self._report(200)
        report.pop("scored")
        self.assertTrue(closure_problems(report))

    def test_the_reliability_floor_is_inside_the_unit_range(self):
        self.assertGreater(CLOSURE_MIN_SCORED_SHARE, 0.0)
        self.assertLessEqual(CLOSURE_MIN_SCORED_SHARE, 1.0)


class ClosureStateTests(unittest.TestCase):
    def test_an_unmatured_horizon_stays_open(self):
        state, reason = closure_state(row("POINT", value=0.6), PENDING_LABELS)
        self.assertEqual(state, CLOSURE_OPEN)
        self.assertTrue(reason)

    def test_a_matured_horizon_is_closeable(self):
        state, _reason = closure_state(row("POINT", value=0.6), MATURED_LABELS)
        self.assertEqual(state, CLOSURE_MATURED)
        self.assertIn(state, CLOSURE_CLOSEABLE_STATES)

    def test_an_elapsed_window_with_no_outcome_is_expired_not_open(self):
        # "we could not score it" and "it has not happened yet" differ.
        state, reason = closure_state(row("POINT", value=0.6), NO_DATA_LABELS)
        self.assertEqual(state, CLOSURE_EXPIRED_NO_DATA)
        self.assertIn("never existed", reason)

    def test_no_labels_means_open(self):
        state, _reason = closure_state(row("POINT", value=0.6), None)
        self.assertEqual(state, CLOSURE_OPEN)

    def test_every_state_is_declared(self):
        for state in (CLOSURE_OPEN, CLOSURE_MATURED, CLOSURE_CLOSED,
                      CLOSURE_EXPIRED_NO_DATA):
            self.assertIn(state, CLOSURE_STATES)


class IdempotenceTests(unittest.TestCase):
    """A CLOSED forecast is terminal."""

    def test_a_closed_forecast_cannot_be_closed_again(self):
        self.assertNotIn(CLOSURE_CLOSED, CLOSURE_CLOSEABLE_STATES)

    def test_re_closing_produces_no_fresh_score(self):
        first = close_forecast(row("POINT", value=0.6), MATURED_LABELS)
        self.assertEqual(first["state"], CLOSURE_CLOSED)
        self.assertTrue(first["score"]["scored"])
        second = close_forecast(first, MATURED_LABELS)
        self.assertEqual(second["state"], CLOSURE_CLOSED)
        self.assertIsNone(second["score"])

    def test_closing_yields_the_realised_outcome(self):
        closed = close_forecast(row("POINT", value=0.6), MATURED_LABELS)
        self.assertEqual(closed["outcome"]["forward_return"], 0.04)
        self.assertTrue(closed["outcome"]["direction_up"])

    def test_realised_outcome_reads_the_label_set(self):
        outcome = realised_outcome(MATURED_LABELS, "20d")
        self.assertEqual(outcome["label_version"], "outcome-label-v2")
        self.assertTrue(outcome["direction_up"])


class LedgerTests(unittest.TestCase):
    def test_the_forecast_id_is_deterministic(self):
        self.assertEqual(
            forecast_id("NVDA", "2024-06-15", "20d", "probability_up"),
            forecast_id("nvda", "2024-06-15", "20d", "probability_up"),
        )

    def test_different_horizons_are_different_forecasts(self):
        self.assertNotEqual(
            forecast_id("NVDA", "2024-06-15", "20d", "probability_up"),
            forecast_id("NVDA", "2024-06-15", "60d", "probability_up"),
        )

    def test_recording_deduplicates(self):
        # A re-run must not inflate the denominator.
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "l.jsonl"
            entry = row("POINT", value=0.62)
            record_forecast(entry, ledger)
            record_forecast(entry, ledger)
            record_forecast(entry, ledger)
            self.assertEqual(len(load_ledger(ledger)), 1)

    def test_a_missing_ledger_reads_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_ledger(Path(folder) / "absent.jsonl"), [])

    def test_a_malformed_ledger_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "l.jsonl"
            ledger.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(OutcomeClosureError):
                load_ledger(ledger)

    def test_an_unknown_claim_tier_is_refused(self):
        with self.assertRaises(OutcomeClosureError):
            build_ledger_row(
                ticker="NVDA", as_of="x", horizon="20d", target="t", claim="NOPE"
            )

    def test_a_ticker_and_horizon_are_required(self):
        with self.assertRaises(OutcomeClosureError):
            build_ledger_row(
                ticker="", as_of="x", horizon="20d", target="t", claim="POINT",
                value=0.5,
            )
        with self.assertRaises(OutcomeClosureError):
            build_ledger_row(
                ticker="NVDA", as_of="x", horizon="", target="t", claim="POINT",
                value=0.5,
            )


class ShapeRuleTests(unittest.TestCase):
    """A refusal carries no value, at BOTH guard layers."""

    def test_the_builder_refuses_a_value_on_a_refusal(self):
        with self.assertRaises(OutcomeClosureError):
            build_ledger_row(
                ticker="NVDA", as_of="x", horizon="20d",
                target="probability_up", claim="INSUFFICIENT", value=0.5,
            )

    def test_validation_catches_a_hand_assembled_refusal_value(self):
        # A row read back from disk, or built by an older version, bypasses
        # the builder entirely.
        problems = ledger_row_problems({
            "forecast_id": "m", "ticker": "NVDA", "as_of": "x",
            "horizon": "20d", "target": "t", "claim": "INSUFFICIENT",
            "score_method": SCORE_METHOD_NOT_SCORED, "state": CLOSURE_OPEN,
            "value": 0.5,
        })
        self.assertTrue(any("confident answer" in p for p in problems))

    def test_a_point_row_without_a_value_is_reported(self):
        problems = ledger_row_problems({
            "forecast_id": "m", "ticker": "NVDA", "as_of": "x",
            "horizon": "20d", "target": "t", "claim": "POINT",
            "score_method": SCORE_METHOD_BRIER, "state": CLOSURE_OPEN,
        })
        self.assertTrue(any("no value" in p for p in problems))

    def test_a_mismatched_score_method_is_reported(self):
        problems = ledger_row_problems({
            "forecast_id": "m", "ticker": "NVDA", "as_of": "x",
            "horizon": "20d", "target": "t", "claim": "POINT",
            "score_method": SCORE_METHOD_COVERAGE, "state": CLOSURE_OPEN,
            "value": 0.5,
        })
        self.assertTrue(any("does not match" in p for p in problems))

    def test_a_healthy_row_raises_nothing(self):
        self.assertEqual(ledger_row_problems(row("POINT", value=0.6)), [])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.predictions = np.clip(rng.normal(0.55, 0.15, 200), 0.02, 0.98)
        self.actuals = (rng.random(200) < self.predictions).astype(float)

    def _rows(self, count):
        return [
            closed_point(self.predictions[i], self.actuals[i])
            for i in range(count)
        ]

    def test_a_thin_sample_is_refused(self):
        result = evaluate_calibration(self._rows(5))
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertEqual(result["required"], CLOSURE_MIN_FOR_CALIBRATION)

    def test_a_full_sample_is_measured(self):
        result = evaluate_calibration(self._rows(200))
        self.assertEqual(result["status"], "MEASURED")
        self.assertIsNotNone(result["brier"])
        self.assertIsNotNone(result["expected_calibration_error"])

    def test_only_point_claims_enter_calibration(self):
        # An interval or a refusal supplies no probability to map.
        mixed = self._rows(200) + [refused_row()] * 50
        self.assertEqual(evaluate_calibration(mixed)["observations"], 200)

    def test_calibration_readiness_is_reported(self):
        self.assertFalse(closure_report(self._rows(5))["calibration_ready"])
        self.assertTrue(closure_report(self._rows(200))["calibration_ready"])


class ReportTests(unittest.TestCase):
    def test_an_empty_report_is_contract_clean(self):
        report = closure_report([])
        self.assertEqual(report["total"], 0)
        self.assertEqual(closure_problems(report), [])

    def test_counts_never_exceed_the_total(self):
        report = closure_report([closed_point(0.6, 1.0), refused_row()])
        self.assertLessEqual(
            report["scored"] + report["refused"] + report["pending"],
            report["total"],
        )

    def test_expired_rows_are_counted_separately(self):
        report = closure_report([{"state": CLOSURE_EXPIRED_NO_DATA, "claim": "POINT"}])
        self.assertEqual(report["expired_no_data"], 1)

    def test_a_mean_brier_over_zero_observations_is_reported(self):
        report = closure_report([refused_row()])
        self.assertIsNone(report["mean_brier"])
        self.assertEqual(closure_problems(report), [])


class SupersedeOnReadTests(unittest.TestCase):
    """The ledger is append-only; the READER collapses to the latest state.

    Closing a forecast writes a NEW row rather than rewriting the original —
    that is what keeps "what did we believe at the time?" answerable. A reader
    counting raw rows double-counts every re-run: MEASURED, three passes over
    12 forecasts produced 36 rows and a scoreboard claiming 8 scored from 4.
    """

    def _ledger(self, folder):
        ledger = Path(folder) / "l.jsonl"
        opened = row("POINT", value=0.62)
        record_forecast(opened, ledger)
        closed = close_forecast(opened, MATURED_LABELS)
        with ledger.open("a", encoding="utf-8") as handle:
            line = json.dumps(closed, sort_keys=True, default=str)
            handle.write(line + chr(10))
            handle.write(line + chr(10))
        return ledger, opened, closed

    def test_the_raw_ledger_keeps_every_row(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger, _opened, _closed = self._ledger(folder)
            self.assertEqual(len(load_ledger(ledger)), 3)

    def test_the_collapsed_view_holds_one_row_per_forecast(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger, _opened, _closed = self._ledger(folder)
            self.assertEqual(len(current_ledger(ledger)), 1)

    def test_closed_supersedes_open_whatever_the_file_order(self):
        # A re-record after a close appends an OPEN row LAST; a naive
        # "last row wins" reader would resurrect it and score twice.
        with tempfile.TemporaryDirectory() as folder:
            ledger, opened, _closed = self._ledger(folder)
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(opened, sort_keys=True, default=str) + chr(10)
                )
            collapsed = current_ledger(ledger)
            self.assertEqual(len(collapsed), 1)
            self.assertEqual(collapsed[0]["state"], CLOSURE_CLOSED)

    def test_an_empty_ledger_collapses_to_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(current_ledger(Path(folder) / "absent.jsonl"), [])

    def test_rows_without_an_id_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "l.jsonl"
            ledger.write_text(
                json.dumps({"ticker": "NVDA"}) + chr(10), encoding="utf-8"
            )
            self.assertEqual(current_ledger(ledger), [])


if __name__ == "__main__":
    unittest.main()
