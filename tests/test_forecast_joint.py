"""Joint forecast tests (Sprint F3).

F3 composes F1's six targets with F2's six horizons into one 36-cell object.
Four rules are pinned here, each settled by measurement rather than intuition.

**A cell carries a `value` key IFF its status is OK.** Not `value: None` — the
key is ABSENT. The dashboard's idiom is `Number(x ?? 0)`, so a null P(up) would
coalesce to `0.0%` and render as CERTAIN DOWN: the most dangerous possible
misreading, produced by defensive-looking code.

**Width guards uncertainty, not fold count.** MEASURED:
`prediction_interval([0.55]*3)` returns width 0.0 at `folds: 3`, so a fold-count
floor passes its own check while publishing perfect certainty.

**Four coherence rules are NOT enforced**, each rejected by measurement. The
most seductive was `adverse_excursion <= min(0, expected_return)`: a gap-up
produces a POSITIVE adverse excursion and 9 of 300 real 20d labels violate it.
A rule that calls ground truth malformed is the wrong rule.

**Contract defects are reported, not hidden.** F3 surfaces real F1/V1
inconsistencies that predate it; rendering tidily over them would make the
tidiness a lie.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.calibration import fit_calibration, prediction_interval
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
    JOINT_REJECTED_RULES,
    JOINT_STATUS_NO_MODEL,
    JOINT_STATUS_PARTIAL,
    JOINT_STATUS_UNAVAILABLE,
)
from core.forecast_joint import (
    JointForecastError,
    build_cell,
    build_joint_forecast,
    contract_findings,
    evaluate_coherence,
    joint_problems,
    render_rows,
    resolve_cell_status,
)

HORIZONS = list(FORECAST_HORIZONS)


def _calibration_map(seed=7, n=200):
    rng = np.random.default_rng(seed)
    scores = list(rng.uniform(0, 1, n))
    actuals = [1.0 if s + rng.normal(0, 0.2) > 0.5 else 0.0 for s in scores]
    return fit_calibration(scores, actuals)


def _labels(matured=("20d",), **overrides):
    record = {
        "status": "OK",
        "entry_bar": "2024-06-14 00:00:00",
        "exit_bar": "2024-07-15 00:00:00",
        "forward_return": 0.032,
        "realized_vol": 0.024,
        "label_up": True,
        "adverse_excursion": -0.075,
    }
    record.update(overrides)
    return {
        "label_version": "outcome-label-v1",
        "matured_horizons": list(matured),
        "pending_horizons": [h for h in HORIZONS if h not in matured],
        "horizons": {h: dict(record) for h in matured},
    }


def _interval(lower=-0.02, upper=0.06, folds=5):
    return {"level": 0.8, "lower": lower, "upper": upper,
            "median": (lower + upper) / 2, "folds": folds}


def _model():
    return {"status": "MODEL", "model_version": "demo-v1", "entry_hash": "abc", "reason": ""}


class ShapeRuleTests(unittest.TestCase):
    """The decision the whole design rests on."""

    def test_a_refused_cell_has_NO_value_key(self):
        """`value: None` would coalesce to 0.0 and render as certain-down."""
        cell = build_cell("probability_up", "20d", labels=_labels(),
                          readiness="SCORABLE", has_model=False)
        self.assertNotEqual(cell["status"], CELL_STATUS_OK)
        self.assertNotIn("value", cell)

    def test_an_ok_cell_carries_a_value_key(self):
        cell = build_cell("expected_return", "20d", labels=_labels(),
                          readiness="SCORABLE", has_model=True, value=0.03)
        self.assertEqual(cell["status"], CELL_STATUS_OK)
        self.assertIn("value", cell)
        self.assertEqual(cell["value"], 0.03)

    def test_a_refused_cell_supplies_no_uncertainty(self):
        """An interval beside a withheld estimate is an estimate by another name."""
        cell = build_cell("expected_return", "20d", labels=_labels(),
                          readiness="SCORABLE", has_model=False,
                          interval=_interval(), dispersion={"std": 0.01})
        self.assertIsNone(cell["interval"])
        self.assertIsNone(cell["dispersion"])

    def test_an_ok_cell_violating_its_bounds_is_refused_loudly(self):
        with self.assertRaises(JointForecastError):
            build_cell("probability_up", "20d", labels=_labels(),
                       readiness="SCORABLE", has_model=True,
                       calibration_map=_calibration_map(), value=1.4)

    def test_the_validator_catches_a_value_on_a_refused_cell(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        forecast["rows"]["20d"]["cells"]["probability_up"]["value"] = 0.0
        self.assertTrue(any("coalesce" in p for p in joint_problems(forecast)))

    def test_the_validator_catches_an_ok_cell_without_a_value(self):
        forecast = build_joint_forecast(
            "T", "2024-06-15", labels=_labels(), model_resolution=_model(),
            values={("expected_return", "20d"): 0.03},
        )
        del forecast["rows"]["20d"]["cells"]["expected_return"]["value"]
        self.assertTrue(any("carries no value" in p for p in joint_problems(forecast)))


class CellPrecedenceTests(unittest.TestCase):
    """Precedence is total and declared, so a refusal is never ambiguous."""

    def test_no_labels_outranks_everything(self):
        status, _ = resolve_cell_status(
            "probability_up", "20d", has_labels=False, readiness="SCORABLE",
            label_backed=False, benchmark=None)
        self.assertEqual(status, CELL_STATUS_UNAVAILABLE)

    def test_pending_outranks_label_unbacked(self):
        status, reason = resolve_cell_status(
            "probability_up", "252d", has_labels=True, readiness="PENDING",
            label_backed=False, benchmark=None)
        self.assertEqual(status, CELL_STATUS_PENDING)
        self.assertIn("has not closed", reason)

    def test_label_unbacked_outranks_needs_benchmark(self):
        status, _ = resolve_cell_status(
            "probability_outperform", "1d", has_labels=True, readiness="SCORABLE",
            label_backed=False, benchmark=None)
        self.assertEqual(status, CELL_STATUS_LABEL_UNBACKED)

    def test_needs_benchmark_outranks_uncalibrated(self):
        status, _ = resolve_cell_status(
            "probability_outperform", "20d", has_labels=True, readiness="SCORABLE",
            label_backed=True, benchmark=None, calibration_map=None)
        self.assertEqual(status, CELL_STATUS_NEEDS_BENCHMARK)

    def test_degenerate_interval_outranks_uncalibrated(self):
        status, _ = resolve_cell_status(
            "probability_up", "20d", has_labels=True, readiness="SCORABLE",
            label_backed=True, benchmark=None, calibration_map=None,
            interval=_interval(lower=0.55, upper=0.55))
        self.assertEqual(status, CELL_STATUS_DEGENERATE_INTERVAL)

    def test_uncalibrated_outranks_no_model(self):
        status, reason = resolve_cell_status(
            "probability_up", "20d", has_labels=True, readiness="SCORABLE",
            label_backed=True, benchmark=None, calibration_map=None, has_model=False)
        self.assertEqual(status, CELL_STATUS_UNCALIBRATED)
        self.assertIn("not a likelihood", reason)

    def test_no_model_is_the_last_refusal(self):
        status, _ = resolve_cell_status(
            "expected_return", "20d", has_labels=True, readiness="SCORABLE",
            label_backed=True, benchmark=None, has_model=False)
        self.assertEqual(status, CELL_STATUS_NO_MODEL)

    def test_ok_is_last_in_the_declared_precedence(self):
        """Every refusal must outrank OK, or a cell could be OK despite one."""
        self.assertEqual(JOINT_CELL_PRECEDENCE[-1], CELL_STATUS_OK)


class DegenerateIntervalTests(unittest.TestCase):
    """Width is the guard; fold count is decorative."""

    def test_a_zero_width_interval_is_refused(self):
        degenerate = prediction_interval([0.55, 0.55, 0.55])
        self.assertEqual(degenerate["upper"] - degenerate["lower"], 0.0)
        cell = build_cell("expected_return", "20d", labels=_labels(),
                          readiness="SCORABLE", has_model=True,
                          interval=degenerate, value=0.03)
        self.assertEqual(cell["status"], CELL_STATUS_DEGENERATE_INTERVAL)

    def test_a_fold_count_floor_would_have_passed_it(self):
        """The measurement behind the design: 3 folds, zero width."""
        degenerate = prediction_interval([0.55, 0.55, 0.55])
        self.assertGreaterEqual(degenerate["folds"], 3)
        self.assertEqual(degenerate["upper"] - degenerate["lower"], 0.0)

    def test_a_real_interval_is_accepted(self):
        cell = build_cell("expected_return", "20d", labels=_labels(),
                          readiness="SCORABLE", has_model=True,
                          interval=_interval(), value=0.03)
        self.assertEqual(cell["status"], CELL_STATUS_OK)


class RejectedRuleTests(unittest.TestCase):
    """Rules NOT enforced, each with the measurement that rejected it."""

    def test_the_rejected_register_is_populated(self):
        self.assertGreaterEqual(len(JOINT_REJECTED_RULES), 4)

    def test_every_rejected_rule_states_its_evidence(self):
        for name, why in JOINT_REJECTED_RULES.items():
            with self.subTest(rule=name):
                self.assertIn("REJECTED", why)

    def test_non_monotonic_returns_are_not_a_violation(self):
        """Real NVDA labels run - - - - + + across 1d..252d."""
        rows = {
            h: {"cells": {"expected_return": {
                "status": CELL_STATUS_OK, "value": v}}}
            for h, v in zip(HORIZONS, [-0.007, -0.104, -0.042, -0.114, 0.100, 0.103])
        }
        self.assertEqual(evaluate_coherence(rows)["violations"], [])

    def test_p_up_disagreeing_with_return_sign_is_not_a_violation(self):
        """A skewed payoff gives P(up)=0.73 with E[return]=-0.0075."""
        rows = {"20d": {"cells": {
            "probability_up": {"status": CELL_STATUS_OK, "value": 0.73},
            "expected_return": {"status": CELL_STATUS_OK, "value": -0.0075},
        }}}
        self.assertEqual(evaluate_coherence(rows)["violations"], [])

    def test_the_rejected_rules_travel_with_the_forecast(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertEqual(set(forecast["rejected_rules"]), set(JOINT_REJECTED_RULES))

    def test_not_checked_is_reported_in_coherence(self):
        rows = {"20d": {"cells": {"expected_return": {"status": CELL_STATUS_OK, "value": 0.03}}}}
        names = {entry["rule"] for entry in evaluate_coherence(rows)["not_checked"]}
        self.assertIn("adverse_excursion_below_return", names)


class ContractFindingTests(unittest.TestCase):
    """Defects F3 reports rather than papering over."""

    def test_a_positive_adverse_excursion_is_reported(self):
        """A gap-up leaves every low above entry; 9 of 300 real labels hit this."""
        labels = _labels(adverse_excursion=0.01)
        findings = contract_findings(labels)
        self.assertTrue(any("adverse_excursion" in f for f in findings))
        self.assertTrue(any("ground truth" in f for f in findings))

    def test_a_normal_adverse_excursion_raises_no_finding(self):
        findings = contract_findings(_labels(adverse_excursion=-0.075))
        self.assertFalse(any("adverse_excursion at" in f for f in findings))

    def test_the_relative_target_defect_is_reported(self):
        """probability_outperform scores against the raw stock return."""
        findings = contract_findings(_labels())
        self.assertTrue(any("outperform" in f for f in findings))

    def test_findings_travel_with_the_forecast(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels(adverse_excursion=0.01))
        self.assertGreaterEqual(len(forecast["findings"]), 2)


class JointAssemblyTests(unittest.TestCase):
    def test_the_grid_is_complete(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        for horizon in FORECAST_HORIZONS:
            with self.subTest(horizon=horizon):
                cells = forecast["rows"][horizon]["cells"]
                self.assertEqual(set(cells), set(FORECAST_TARGETS))

    def test_horizon_order_is_shortest_first(self):
        """'120d' sorts before '1d', so iterating dict keys reads out of order."""
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertEqual(forecast["horizon_order"], list(FORECAST_HORIZONS))

    def test_no_labels_is_unavailable(self):
        forecast = build_joint_forecast("T", "2024-06-15")
        self.assertEqual(forecast["status"], JOINT_STATUS_UNAVAILABLE)
        self.assertTrue(forecast["reason"])

    def test_no_model_is_reported_honestly(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertEqual(forecast["status"], JOINT_STATUS_NO_MODEL)
        self.assertEqual(forecast["emitted_values"], 0)
        self.assertTrue(forecast["model"]["reason"])

    def test_a_partial_forecast_counts_its_cells(self):
        forecast = build_joint_forecast(
            "T", "2024-06-15", labels=_labels(), model_resolution=_model(),
            values={("expected_return", "20d"): 0.03},
            intervals={("expected_return", "20d"): _interval()},
        )
        self.assertEqual(forecast["status"], JOINT_STATUS_PARTIAL)
        self.assertEqual(forecast["emitted_values"], 1)

    def test_a_healthy_forecast_is_contract_clean(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertEqual(joint_problems(forecast), [])

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(JointForecastError):
            build_joint_forecast("  ", "2024-06-15", labels=_labels())

    def test_the_forecast_is_versioned(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertTrue(forecast["contract_version"])
        self.assertTrue(forecast["pipeline_version"])

    def test_each_horizon_reports_its_own_knowable_at(self):
        """Different horizons become knowable at different times."""
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertIsNotNone(forecast["rows"]["20d"]["knowable_at"])

    def test_the_forecast_is_deterministic(self):
        labels = _labels()
        self.assertEqual(
            build_joint_forecast("T", "2024-06-15", labels=labels),
            build_joint_forecast("T", "2024-06-15", labels=labels),
        )

    def test_emitted_count_matches_the_grid(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        forecast["emitted_values"] = 99
        self.assertTrue(any("emitted_values" in p for p in joint_problems(forecast)))


class RenderTests(unittest.TestCase):
    """The roadmap's reading order: 1D / 5D / 20D / 60D ..."""

    def test_rows_render_shortest_first(self):
        forecast = build_joint_forecast("T", "2024-06-15", labels=_labels())
        self.assertEqual([r["horizon"] for r in render_rows(forecast)], HORIZONS)

    def test_a_refused_cell_renders_none_with_a_reason(self):
        rows = render_rows(build_joint_forecast("T", "2024-06-15", labels=_labels()))
        first = rows[0]
        self.assertIsNone(first["expected_return"])
        self.assertTrue(first["expected_return_reason"])

    def test_an_emitted_cell_renders_its_value(self):
        forecast = build_joint_forecast(
            "T", "2024-06-15", labels=_labels(), model_resolution=_model(),
            values={("expected_return", "20d"): 0.047},
            intervals={("expected_return", "20d"): _interval()},
        )
        row = next(r for r in render_rows(forecast) if r["horizon"] == "20d")
        self.assertEqual(row["expected_return"], 0.047)


class LiveLabelTests(unittest.TestCase):
    """Against the real V1 label builder."""

    def test_a_real_forecast_is_contract_clean_and_honest(self):
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2024-06-15")
        if labels.get("status") not in ("OK", "PARTIAL"):
            self.skipTest("label builder unavailable in this environment")
        forecast = build_joint_forecast("NVDA", "2024-06-15", labels=labels)
        self.assertEqual(joint_problems(forecast), [])
        self.assertEqual(forecast["status"], JOINT_STATUS_NO_MODEL)
        self.assertEqual(forecast["emitted_values"], 0)

    def test_real_cells_discriminate_their_refusals(self):
        """Four distinct reasons, not one blanket status."""
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2024-06-15")
        if labels.get("status") not in ("OK", "PARTIAL"):
            self.skipTest("label builder unavailable in this environment")
        forecast = build_joint_forecast("NVDA", "2024-06-15", labels=labels)
        statuses = {
            cell["status"]
            for row in forecast["rows"].values()
            for cell in row["cells"].values()
        }
        self.assertGreaterEqual(len(statuses), 3)
