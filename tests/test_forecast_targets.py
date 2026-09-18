"""Forecast target tests (Sprint F1).

F1 defines the six things the engine may be asked to predict and, for each, what
the answer means. Three rules are pinned here.

**A probability only exists through a fitted calibration.** Targets whose kind is
probability or distribution must route through M6's `calibrated_probability`,
which refuses without a map. A raw model score is not a probability, and this is
where that stops being a comment.

**Every target resolves to a realized label.** A forecast that cannot be
compared to what actually happened is an opinion, so each target names a field
the V1 label builder produces.

**Bounds are declared and checked.** An adverse excursion is the worst drawdown
INSIDE the window and is never positive; a volatility is a dispersion and is
never negative. A value violating its own contract is refused, not rendered.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.calibration import UncalibratedProbabilityError, fit_calibration
from core.config import (
    FORECAST_KIND_DISTRIBUTION,
    FORECAST_KIND_PROBABILITY,
    FORECAST_TARGET_CONTRACTS,
    FORECAST_TARGET_DIRECTION,
    FORECAST_TARGET_DISTRIBUTION,
    FORECAST_TARGET_DOWNSIDE,
    FORECAST_TARGET_RELATIVE,
    FORECAST_TARGET_RETURN,
    FORECAST_TARGET_VOLATILITY,
    FORECAST_TARGETS,
)
from core.forecast_targets import (
    TARGET_STATUS_OK,
    TARGET_STATUS_UNAVAILABLE,
    ForecastTargetError,
    bounds_problems,
    build_target_request,
    is_probability_target,
    known_targets,
    label_field_for,
    realized_value,
    request_problems,
    requires_benchmark,
    resolve_probability,
    target_contract,
    target_coverage,
)


def _calibration_map(seed=7, n=200):
    rng = np.random.default_rng(seed)
    scores = list(rng.uniform(0, 1, n))
    actuals = [1.0 if s + rng.normal(0, 0.2) > 0.5 else 0.0 for s in scores]
    return fit_calibration(scores, actuals)


def _labels(status="OK", **overrides):
    horizon = {
        "status": status,
        "label_up": True,
        "forward_return": 0.032214,
        "adverse_excursion": -0.075004,
        "realized_vol": 0.024811,
    }
    horizon.update(overrides)
    return {"status": "OK", "horizons": {"20d": horizon}}


class TargetVocabularyTests(unittest.TestCase):
    def test_every_f1_target_is_declared(self):
        for target in (
            FORECAST_TARGET_DIRECTION, FORECAST_TARGET_RETURN,
            FORECAST_TARGET_DISTRIBUTION, FORECAST_TARGET_DOWNSIDE,
            FORECAST_TARGET_VOLATILITY, FORECAST_TARGET_RELATIVE,
        ):
            with self.subTest(target=target):
                self.assertIn(target, known_targets())

    def test_an_unknown_target_is_refused_not_defaulted(self):
        """A typo silently becoming expected_return would score a model against
        a question nobody asked."""
        with self.assertRaises(ForecastTargetError):
            target_contract("probability_of_profit")

    def test_every_target_names_a_realized_label_field(self):
        for target in FORECAST_TARGETS:
            with self.subTest(target=target):
                self.assertTrue(label_field_for(target))

    def test_every_target_states_the_question_it_answers(self):
        for target in FORECAST_TARGETS:
            with self.subTest(target=target):
                self.assertTrue(target_contract(target)["question"])

    def test_probability_targets_are_identified_as_such(self):
        self.assertTrue(is_probability_target(FORECAST_TARGET_DIRECTION))
        self.assertTrue(is_probability_target(FORECAST_TARGET_DISTRIBUTION))
        self.assertTrue(is_probability_target(FORECAST_TARGET_RELATIVE))
        self.assertFalse(is_probability_target(FORECAST_TARGET_RETURN))
        self.assertFalse(is_probability_target(FORECAST_TARGET_VOLATILITY))

    def test_only_the_relative_target_needs_a_benchmark(self):
        self.assertTrue(requires_benchmark(FORECAST_TARGET_RELATIVE))
        for target in set(FORECAST_TARGETS) - {FORECAST_TARGET_RELATIVE}:
            with self.subTest(target=target):
                self.assertFalse(requires_benchmark(target))

    def test_probability_and_calibration_flags_never_disagree(self):
        """A probability that skips calibration is the M6 failure mode."""
        for target, contract in FORECAST_TARGET_CONTRACTS.items():
            with self.subTest(target=target):
                is_probability = contract["kind"] in (
                    FORECAST_KIND_PROBABILITY, FORECAST_KIND_DISTRIBUTION
                )
                self.assertEqual(is_probability, contract["requires_calibration"])


class CalibrationGuardTests(unittest.TestCase):
    """The rule that keeps a raw score from being shown as a likelihood."""

    def test_an_uncalibrated_probability_is_refused(self):
        with self.assertRaises(UncalibratedProbabilityError):
            resolve_probability(FORECAST_TARGET_DIRECTION, 0.8, None)

    def test_a_fitted_map_produces_a_probability(self):
        value = resolve_probability(FORECAST_TARGET_DIRECTION, 0.8, _calibration_map())
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_calibrating_a_return_target_is_refused(self):
        """It would present a return as a likelihood."""
        with self.assertRaises(ForecastTargetError):
            resolve_probability(FORECAST_TARGET_RETURN, 0.05, _calibration_map())

    def test_every_probability_target_can_be_calibrated(self):
        cmap = _calibration_map()
        for target in FORECAST_TARGETS:
            if not is_probability_target(target):
                continue
            with self.subTest(target=target):
                self.assertIsNotNone(resolve_probability(target, 0.6, cmap))


class BoundsTests(unittest.TestCase):
    def test_a_probability_above_one_is_refused(self):
        self.assertTrue(bounds_problems(FORECAST_TARGET_DIRECTION, 1.4))

    def test_a_negative_probability_is_refused(self):
        self.assertTrue(bounds_problems(FORECAST_TARGET_DIRECTION, -0.1))

    def test_a_positive_adverse_excursion_is_refused(self):
        """It is the worst drawdown INSIDE the window; it cannot be a gain."""
        self.assertTrue(bounds_problems(FORECAST_TARGET_DOWNSIDE, 0.05))

    def test_a_negative_volatility_is_refused(self):
        """A dispersion cannot be below zero."""
        self.assertTrue(bounds_problems(FORECAST_TARGET_VOLATILITY, -0.01))

    def test_a_return_below_total_loss_is_refused(self):
        self.assertTrue(bounds_problems(FORECAST_TARGET_RETURN, -1.5))

    def test_valid_values_raise_no_problems(self):
        self.assertEqual(bounds_problems(FORECAST_TARGET_DIRECTION, 0.63), [])
        self.assertEqual(bounds_problems(FORECAST_TARGET_DOWNSIDE, -0.08), [])
        self.assertEqual(bounds_problems(FORECAST_TARGET_VOLATILITY, 0.02), [])

    def test_none_is_not_a_bounds_violation(self):
        """An absent forecast is absent, not malformed."""
        self.assertEqual(bounds_problems(FORECAST_TARGET_DIRECTION, None), [])

    def test_nan_is_refused(self):
        self.assertTrue(bounds_problems(FORECAST_TARGET_RETURN, float("nan")))


class RealizedValueTests(unittest.TestCase):
    def test_each_target_reads_its_declared_label_field(self):
        labels = _labels()
        self.assertEqual(realized_value(FORECAST_TARGET_RETURN, labels, "20d"), 0.032214)
        self.assertEqual(realized_value(FORECAST_TARGET_DOWNSIDE, labels, "20d"), -0.075004)
        self.assertEqual(realized_value(FORECAST_TARGET_VOLATILITY, labels, "20d"), 0.024811)

    def test_a_boolean_label_becomes_a_numeric_outcome(self):
        self.assertEqual(realized_value(FORECAST_TARGET_DIRECTION, _labels(), "20d"), 1.0)
        down = _labels(label_up=False)
        self.assertEqual(realized_value(FORECAST_TARGET_DIRECTION, down, "20d"), 0.0)

    def test_an_unmatured_horizon_yields_nothing(self):
        """A forecast judged against an unfinished window is judged against noise."""
        pending = _labels(status="PENDING")
        self.assertIsNone(realized_value(FORECAST_TARGET_RETURN, pending, "20d"))

    def test_an_absent_horizon_yields_nothing(self):
        self.assertIsNone(realized_value(FORECAST_TARGET_RETURN, _labels(), "120d"))

    def test_empty_labels_yield_nothing(self):
        self.assertIsNone(realized_value(FORECAST_TARGET_RETURN, {}, "20d"))


class TargetRequestTests(unittest.TestCase):
    def test_a_plain_request_is_ok_and_contract_clean(self):
        request = build_target_request(FORECAST_TARGET_RETURN, "20d")
        self.assertEqual(request["status"], TARGET_STATUS_OK)
        self.assertEqual(request_problems(request), [])

    def test_the_relative_target_without_a_benchmark_is_unavailable(self):
        """Scoring it against nothing would silently reduce it to probability_up."""
        request = build_target_request(FORECAST_TARGET_RELATIVE, "20d")
        self.assertEqual(request["status"], TARGET_STATUS_UNAVAILABLE)
        self.assertIn("benchmark", request["reason"])

    def test_the_relative_target_with_a_benchmark_is_ok(self):
        request = build_target_request(FORECAST_TARGET_RELATIVE, "20d", benchmark="SPY")
        self.assertEqual(request["status"], TARGET_STATUS_OK)
        self.assertEqual(request_problems(request), [])

    def test_a_distribution_without_an_interval_is_unavailable(self):
        """P(return within WHAT?) is not a question."""
        request = build_target_request(FORECAST_TARGET_DISTRIBUTION, "20d")
        self.assertEqual(request["status"], TARGET_STATUS_UNAVAILABLE)
        self.assertIn("interval", request["reason"])

    def test_a_distribution_with_an_interval_is_ok(self):
        request = build_target_request(
            FORECAST_TARGET_DISTRIBUTION, "20d", interval=(0.0, 0.05)
        )
        self.assertEqual(request["status"], TARGET_STATUS_OK)
        self.assertEqual(request_problems(request), [])

    def test_an_empty_interval_is_refused(self):
        """P(return in an empty set) is zero by construction and says nothing."""
        with self.assertRaises(ForecastTargetError):
            build_target_request(FORECAST_TARGET_DISTRIBUTION, "20d", interval=(0.05, 0.05))

    def test_a_missing_horizon_is_refused(self):
        with self.assertRaises(ForecastTargetError):
            build_target_request(FORECAST_TARGET_RETURN, "")

    def test_a_request_disagreeing_with_its_contract_is_caught(self):
        request = build_target_request(FORECAST_TARGET_DIRECTION, "20d")
        request["requires_calibration"] = False
        self.assertTrue(any("calibration" in p for p in request_problems(request)))

    def test_a_non_ok_request_must_explain_itself(self):
        request = build_target_request(FORECAST_TARGET_RELATIVE, "20d")
        request["reason"] = ""
        self.assertTrue(any("explain" in p for p in request_problems(request)))


class CoverageTests(unittest.TestCase):
    def test_a_matured_horizon_supports_every_target(self):
        coverage = target_coverage(_labels(), "20d")
        self.assertEqual(coverage["unscorable_targets"], [])

    def test_an_unmatured_horizon_supports_nothing(self):
        """Six Nones are less useful than saying the horizon is not ready."""
        coverage = target_coverage(_labels(status="PENDING"), "20d")
        self.assertEqual(coverage["scorable_targets"], [])

    def test_coverage_is_reported_per_target(self):
        coverage = target_coverage(_labels(), "20d")
        for target in FORECAST_TARGETS:
            with self.subTest(target=target):
                self.assertIn(target, coverage["covered"])


class LiveLabelTests(unittest.TestCase):
    """Against the real V1 label builder, not a fixture."""

    def test_every_target_scores_against_real_labels(self):
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2026-06-15")
        if labels.get("status") != "OK":
            self.skipTest("label builder unavailable in this environment")
        coverage = target_coverage(labels, "20d")
        self.assertEqual(
            coverage["unscorable_targets"], [],
            "a target with no realized counterpart cannot be validated",
        )
