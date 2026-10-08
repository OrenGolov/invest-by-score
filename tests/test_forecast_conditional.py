"""F4 conditional-forecast tests.

The behaviour under test is SELECTION: that the module picks a different
estimator and a different claim strength per cell, based on the evidence that
cell holds. A test suite that only checked "it returns a number" would pass
against a global rule, which is the design F4 exists to reject.
"""

from __future__ import annotations

import unittest

from core.config import (
    COND_STATUS_EMPTY_SLICE,
    COND_STATUS_INSUFFICIENT,
    COND_STATUS_LABEL_UNBACKED,
    COND_STATUS_NO_CONDITION,
    COND_STATUS_OK,
    COND_STATUS_PENDING,
    COND_STATUS_UNAVAILABLE,
    CONDITIONAL_CELL_PRECEDENCE,
    CONDITIONAL_CLAIM_DIRECTIONAL,
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_INTERVAL,
    CONDITIONAL_CLAIM_POINT,
    CONDITIONAL_CLAIM_PRECEDENCE,
    CONDITIONAL_MAX_INTERVAL_WIDTH,
    CONDITIONAL_MIN_SAMPLES_DIRECTIONAL,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    FORECAST_HORIZONS,
    FORECAST_TARGETS,
    REGIME_LABELS,
)
from core.forecast_conditional import (
    DIRECTION_HIGHER,
    DIRECTION_INDISTINGUISHABLE,
    DIRECTION_LOWER,
    ConditionalForecastError,
    build_cell,
    build_conditional_forecast,
    condition_counts,
    conditional_problems,
    direction_of,
    excludes,
    interval_width,
    render_rows,
    select_claim,
    slice_observations,
    wilson_interval,
)
from core.forecast_horizons import HORIZON_STATUS_PENDING


def observations(counts):
    """An observation set with a given {regime: (trials, successes)}."""
    built = []
    for regime, (trials, successes) in counts.items():
        for index in range(trials):
            built.append(
                {
                    "as_of": f"2024-01-{(index % 28) + 1:02d}",
                    "conditions": {"market_regime": regime},
                    "outcomes": {"probability_up": 1.0 if index < successes else 0.0},
                }
            )
    return built


class WilsonIntervalTests(unittest.TestCase):
    def test_no_observations_yields_no_interval(self):
        # [0, 1] is the absence of an answer, not a very uncertain one.
        self.assertIsNone(wilson_interval(0, 0))

    def test_unanimous_sample_does_not_collapse_to_zero_width(self):
        # The normal approximation gives width 0 at p=1; Wilson does not.
        # This is the whole reason Wilson was chosen.
        band = wilson_interval(2, 2)
        self.assertGreater(band["width"], 0.6)
        self.assertEqual(band["upper"], 1.0)
        self.assertGreater(band["lower"], 0.0)

    def test_width_shrinks_with_sample_size(self):
        widths = [wilson_interval(int(n * 0.2), n)["width"] for n in (5, 10, 40, 140)]
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_interval_stays_inside_the_unit_range(self):
        for trials in (1, 3, 9, 50):
            for successes in (0, trials):
                band = wilson_interval(successes, trials)
                self.assertGreaterEqual(band["lower"], 0.0)
                self.assertLessEqual(band["upper"], 1.0)

    def test_successes_above_trials_is_rejected(self):
        with self.assertRaises(ConditionalForecastError):
            wilson_interval(5, 2)

    def test_interval_carries_its_own_sample_size(self):
        # The N must travel with the estimate; a bare interval cannot be
        # audited for the tier it was allowed to support.
        band = wilson_interval(7, 32)
        self.assertEqual(band["samples"], 32)
        self.assertEqual(band["successes"], 7)


class SelectClaimTests(unittest.TestCase):
    """The selection mechanism — the load-bearing behaviour of F4."""

    BASE = 12 / 140

    def claim(self, trials, successes, base=None):
        return select_claim(
            trials,
            wilson_interval(successes, trials),
            self.BASE if base is None else base,
        )[0]

    def test_the_measured_slices_do_not_all_land_in_one_tier(self):
        # A global rule would put all five in the same bucket. This is the
        # assertion that distinguishes F4's design from the thing it rejects.
        tiers = {
            name: self.claim(trials, successes)
            for name, trials, successes in (
                ("bullish", 140, 12),
                ("risk_off", 32, 7),
                ("range", 9, 3),
                ("bearish", 5, 1),
                ("stress", 2, 2),
            )
        }
        self.assertGreater(len(set(tiers.values())), 1, tiers)

    def test_well_sampled_slice_earns_a_point_estimate(self):
        self.assertEqual(self.claim(140, 12), CONDITIONAL_CLAIM_POINT)

    def test_two_observation_slice_is_refused(self):
        # The roadmap's own example. It reads P=1.00 and must not publish.
        self.assertEqual(self.claim(2, 2), CONDITIONAL_CLAIM_INSUFFICIENT)

    def test_refusal_reason_names_the_sample_size(self):
        _, reason = select_claim(2, wilson_interval(2, 2), self.BASE)
        self.assertIn("2", reason)

    def test_empty_slice_is_refused_with_its_own_reason(self):
        claim, reason = select_claim(0, None, self.BASE)
        self.assertEqual(claim, CONDITIONAL_CLAIM_INSUFFICIENT)
        self.assertIn("never occurred", reason)

    def test_directional_needs_more_evidence_than_interval(self):
        # The deliberate inversion. A tight interval at N=10 cleanly excludes
        # the base rate, and must STILL be refused a directional claim: a 15pt
        # effect is detected only 24% of the time at that sample size.
        self.assertNotEqual(
            self.claim(10, 10, base=0.05), CONDITIONAL_CLAIM_DIRECTIONAL
        )
        self.assertLess(
            CONDITIONAL_MIN_SAMPLES_INTERVAL, CONDITIONAL_MIN_SAMPLES_DIRECTIONAL
        )
        self.assertLess(
            CONDITIONAL_MIN_SAMPLES_DIRECTIONAL, CONDITIONAL_MIN_SAMPLES_POINT
        )

    def test_directional_fires_when_the_interval_separates(self):
        self.assertEqual(self.claim(32, 7), CONDITIONAL_CLAIM_DIRECTIONAL)

    def test_directional_does_not_fire_when_the_base_rate_is_inside(self):
        # Same sample, base rate now sits inside the interval.
        self.assertNotEqual(self.claim(32, 7, base=0.22), CONDITIONAL_CLAIM_DIRECTIONAL)

    def test_a_narrow_unanimous_tiny_sample_is_still_refused(self):
        # MEASURED: 5/5 gives width 0.434, NARROWER than a 9-observation cell
        # at 0.525. Width alone would admit it; the sample floor must not.
        self.assertLess(
            interval_width(wilson_interval(5, 5)),
            interval_width(wilson_interval(3, 9)),
        )
        self.assertEqual(self.claim(5, 5, base=0.20), CONDITIONAL_CLAIM_INSUFFICIENT)

    def test_a_well_sampled_but_too_wide_cell_drops_a_tier(self):
        # Clearing the point floor is not sufficient; the width must also be
        # usable. A near-50/50 rate at a modest N can clear one and not the
        # other.
        claim, _ = select_claim(9, wilson_interval(4, 9), 0.20)
        self.assertEqual(claim, CONDITIONAL_CLAIM_INSUFFICIENT)

    def test_interval_tier_is_reachable(self):
        # A slice above the interval floor whose width is inside the cap.
        claim, _ = select_claim(20, wilson_interval(4, 20), 0.20)
        self.assertEqual(claim, CONDITIONAL_CLAIM_INTERVAL)

    def test_every_claim_is_a_declared_tier(self):
        for trials in range(0, 60):
            successes = max(0, trials // 3)
            claim, reason = select_claim(
                trials, wilson_interval(successes, trials), self.BASE
            )
            self.assertIn(claim, CONDITIONAL_CLAIM_PRECEDENCE)
            self.assertTrue(reason, f"claim at N={trials} carries no reason")

    def test_claim_strength_never_decreases_with_more_of_the_same_evidence(self):
        # Monotonicity: holding the rate fixed, more observations must never
        # produce a WEAKER claim. A non-monotonic ladder would mean a cell
        # could be punished for gathering data.
        rank = {name: i for i, name in enumerate(CONDITIONAL_CLAIM_PRECEDENCE)}
        previous = -1
        for trials in (8, 20, 40, 80, 140, 300):
            claim = self.claim(trials, round(trials * 0.086))
            self.assertGreaterEqual(rank[claim], previous, f"weakened at N={trials}")
            previous = max(previous, rank[claim])


class DirectionTests(unittest.TestCase):
    def test_excludes_is_strict_not_numeric_difference(self):
        # Every finite sample differs numerically from the base rate; the test
        # must ask whether the difference is distinguishable.
        band = wilson_interval(7, 32)
        self.assertFalse(excludes(band, 7 / 32))
        self.assertTrue(excludes(band, 12 / 140))

    def test_direction_names_the_side(self):
        band = wilson_interval(7, 32)
        self.assertEqual(direction_of(band, 0.01), DIRECTION_HIGHER)
        self.assertEqual(direction_of(band, 0.99), DIRECTION_LOWER)
        self.assertEqual(direction_of(band, 7 / 32), DIRECTION_INDISTINGUISHABLE)

    def test_no_base_rate_means_no_direction(self):
        self.assertFalse(excludes(wilson_interval(7, 32), None))
        self.assertEqual(
            direction_of(wilson_interval(7, 32), None), DIRECTION_INDISTINGUISHABLE
        )


class SliceTests(unittest.TestCase):
    def test_slices_only_matching_observations(self):
        sample = observations({"bullish": (10, 5), "stress": (3, 3)})
        self.assertEqual(len(slice_observations(sample, "bullish")), 10)
        self.assertEqual(len(slice_observations(sample, "stress")), 3)
        self.assertEqual(len(slice_observations(sample, "bearish")), 0)

    def test_an_unresolved_condition_is_dropped_not_counted_as_other(self):
        # "the regime was not bullish" and "the regime could not be
        # established" are different facts. Folding the second into the first
        # would inflate every other slice with unresolved sessions.
        sample = observations({"bullish": (4, 2)})
        sample.append({"conditions": {}, "outcomes": {"probability_up": 1.0}})
        sample.append({"conditions": {"market_regime": None}, "outcomes": {}})
        self.assertEqual(len(slice_observations(sample, "bullish")), 4)
        counts = condition_counts(sample)
        self.assertEqual(counts["bullish"], 4)
        self.assertEqual(sum(counts.values()), 4)

    def test_counts_report_every_regime_including_zeroes(self):
        # A missing row reads as an oversight; a zero reads as a fact.
        counts = condition_counts(observations({"bullish": (3, 1)}))
        for label in REGIME_LABELS:
            self.assertIn(label, counts)
        self.assertEqual(counts["stress"], 0)


class BuildCellTests(unittest.TestCase):
    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ConditionalForecastError):
            build_cell("not_a_target", "20d", "bullish", observations=[])

    def test_unknown_horizon_is_rejected(self):
        with self.assertRaises(ConditionalForecastError):
            build_cell("probability_up", "7d", "bullish", observations=[])

    def test_the_shape_rule_value_exists_iff_claim_is_point(self):
        sample = observations(
            {"bullish": (140, 12), "risk_off": (32, 7), "stress": (2, 2)}
        )
        for regime in ("bullish", "risk_off", "stress"):
            cell = build_cell(
                "probability_up", "20d", regime,
                observations=sample, base_rate=12 / 140,
            )
            if cell["claim"] == CONDITIONAL_CLAIM_POINT:
                self.assertIn("value", cell, regime)
            else:
                self.assertNotIn("value", cell, regime)

    def test_a_refused_cell_supplies_no_uncertainty(self):
        # An interval beside a withheld point estimate is a point estimate by
        # another name.
        cell = build_cell(
            "probability_up", "20d", "stress",
            observations=observations({"bullish": (50, 20), "stress": (2, 2)}),
            base_rate=0.4,
        )
        self.assertEqual(cell["status"], COND_STATUS_INSUFFICIENT)
        self.assertIsNone(cell["interval"])
        self.assertIsNone(cell["direction"])
        self.assertTrue(cell["reason"])

    def test_a_refused_cell_still_reports_how_thin_it_was(self):
        cell = build_cell(
            "probability_up", "20d", "stress",
            observations=observations({"bullish": (50, 20), "stress": (2, 2)}),
            base_rate=0.4,
        )
        self.assertEqual(cell["samples"], 2)

    def test_empty_slice_is_distinct_from_insufficient(self):
        # Different facts with different fixes: widen the window vs wait.
        empty = build_cell(
            "probability_up", "20d", "stress",
            observations=observations({"bullish": (50, 20)}), base_rate=0.4,
        )
        thin = build_cell(
            "probability_up", "20d", "stress",
            observations=observations({"bullish": (50, 20), "stress": (2, 2)}),
            base_rate=0.4,
        )
        self.assertEqual(empty["status"], COND_STATUS_EMPTY_SLICE)
        self.assertEqual(thin["status"], COND_STATUS_INSUFFICIENT)
        self.assertNotEqual(empty["reason"], thin["reason"])

    def test_no_observations_reads_unavailable(self):
        cell = build_cell("probability_up", "20d", "bullish", observations=None)
        self.assertEqual(cell["status"], COND_STATUS_UNAVAILABLE)
        self.assertNotIn("value", cell)

    def test_a_pending_horizon_refuses_before_counting_anything(self):
        cell = build_cell(
            "probability_up", "20d", "bullish",
            observations=observations({"bullish": (140, 12)}),
            readiness=HORIZON_STATUS_PENDING,
        )
        self.assertEqual(cell["status"], COND_STATUS_PENDING)
        self.assertNotIn("value", cell)
        self.assertIn("fabricated", cell["reason"])

    def test_an_unresolvable_condition_refuses(self):
        cell = build_cell(
            "probability_up", "20d", "bullish",
            observations=observations({"bullish": (140, 12)}),
            condition_resolved=False,
        )
        self.assertEqual(cell["status"], COND_STATUS_NO_CONDITION)
        self.assertNotIn("value", cell)

    def test_an_unbacked_label_refuses(self):
        cell = build_cell(
            "probability_up", "20d", "bullish",
            observations=observations({"bullish": (140, 12)}),
            label_backed=False,
        )
        self.assertEqual(cell["status"], COND_STATUS_LABEL_UNBACKED)
        self.assertNotIn("value", cell)

    def test_precedence_the_most_fundamental_obstacle_wins(self):
        # Every obstacle at once: the reason must name the FIRST one, so a
        # reader is never told about a thin sample when the real problem is
        # that the window has not closed.
        cell = build_cell(
            "probability_up", "20d", "stress",
            observations=observations({"stress": (2, 2)}),
            readiness=HORIZON_STATUS_PENDING,
            condition_resolved=False,
            label_backed=False,
        )
        self.assertEqual(cell["status"], COND_STATUS_PENDING)

    def test_a_point_cell_carries_its_interval_too(self):
        cell = build_cell(
            "probability_up", "20d", "bullish",
            observations=observations({"bullish": (140, 12)}), base_rate=0.5,
        )
        self.assertEqual(cell["claim"], CONDITIONAL_CLAIM_POINT)
        self.assertIsNotNone(cell["interval"])
        self.assertAlmostEqual(cell["value"], 12 / 140, places=5)

    def test_an_observation_with_no_outcome_leaves_the_denominator(self):
        # Counting it as a miss would bias every rate toward zero in exactly
        # the thin slices that can least afford it.
        sample = observations({"bullish": (10, 5)})
        sample.append({"conditions": {"market_regime": "bullish"}, "outcomes": {}})
        sample.append(
            {"conditions": {"market_regime": "bullish"},
             "outcomes": {"probability_up": None}}
        )
        cell = build_cell(
            "probability_up", "20d", "bullish", observations=sample, base_rate=0.5,
        )
        self.assertEqual(cell["samples"], 10)


class GridTests(unittest.TestCase):
    def setUp(self):
        self.sample = observations(
            {
                "bullish": (140, 12),
                "risk_off": (32, 7),
                "range": (9, 3),
                "bearish": (5, 1),
                "stress": (2, 2),
            }
        )
        self.forecast = build_conditional_forecast(
            "SPY", "2026-06-15",
            observations=self.sample,
            targets=("probability_up",),
            horizons=("20d",),
            base_rates={"probability_up": 12 / 140},
        )

    def test_a_ticker_is_required(self):
        with self.assertRaises(ConditionalForecastError):
            build_conditional_forecast("", "2026-06-15")

    def test_the_grid_is_contract_clean(self):
        self.assertEqual(conditional_problems(self.forecast), [])

    def test_every_condition_appears_in_every_row(self):
        cells = self.forecast["rows"]["20d"]["cells"]["probability_up"]
        for label in REGIME_LABELS:
            self.assertIn(label, cells)

    def test_the_grid_reports_its_multiplicity_exposure(self):
        # 180 cells from one history yields ~9 spurious findings at 5%.
        self.assertEqual(self.forecast["comparisons"], 1 * 1 * len(REGIME_LABELS))
        for cell in self.forecast["rows"]["20d"]["cells"]["probability_up"].values():
            if cell["claim"] == CONDITIONAL_CLAIM_DIRECTIONAL:
                self.assertTrue(cell["comparisons"])

    def test_the_full_default_grid_is_contract_clean(self):
        full = build_conditional_forecast("SPY", "2026-06-15", observations=self.sample)
        self.assertEqual(conditional_problems(full), [])
        self.assertEqual(
            len(full["rows"]), len(FORECAST_HORIZONS)
        )
        for row in full["rows"].values():
            self.assertEqual(set(row["cells"]), set(FORECAST_TARGETS))

    def test_an_observation_less_grid_emits_nothing_and_says_why(self):
        empty = build_conditional_forecast(
            "SPY", "2026-06-15", observations=None,
            targets=("probability_up",), horizons=("20d",),
        )
        self.assertEqual(empty["emitted_points"], 0)
        self.assertEqual(empty["emitted_intervals"], 0)
        self.assertEqual(empty["emitted_directions"], 0)
        self.assertEqual(conditional_problems(empty), [])
        for cell in empty["rows"]["20d"]["cells"]["probability_up"].values():
            self.assertEqual(cell["status"], COND_STATUS_UNAVAILABLE)
            self.assertTrue(cell["reason"])

    def test_the_governance_note_survives(self):
        # A favourable stress cell must never read as permission to trade.
        self.assertIn("risk_policy", self.forecast["governance_note"])

    def test_render_rows_never_leaves_a_blank(self):
        for row in render_rows(self.forecast):
            for regime, rendered in row["conditions"].items():
                self.assertTrue(rendered["reason"] or rendered["value"] is not None)
                self.assertIn(rendered["claim"], CONDITIONAL_CLAIM_PRECEDENCE)


class ContractProblemTests(unittest.TestCase):
    """conditional_problems must actually catch a malformed grid."""

    def _grid(self, mutate):
        forecast = build_conditional_forecast(
            "SPY", "2026-06-15",
            observations=observations({"bullish": (140, 12), "stress": (2, 2)}),
            targets=("probability_up",), horizons=("20d",),
            base_rates={"probability_up": 12 / 140},
        )
        mutate(forecast["rows"]["20d"]["cells"]["probability_up"])
        return conditional_problems(forecast)

    def test_a_value_on_a_refused_cell_is_reported(self):
        problems = self._grid(lambda cells: cells["stress"].update({"value": 1.0}))
        self.assertTrue(any("coalesce" in p for p in problems), problems)

    def test_a_missing_value_on_a_point_cell_is_reported(self):
        problems = self._grid(lambda cells: cells["bullish"].pop("value"))
        self.assertTrue(any("carries no value" in p for p in problems), problems)

    def test_an_interval_on_a_refused_cell_is_reported(self):
        problems = self._grid(
            lambda cells: cells["stress"].update({"interval": wilson_interval(2, 2)})
        )
        self.assertTrue(any("another name" in p for p in problems), problems)

    def test_an_under_sampled_point_claim_is_reported(self):
        def mutate(cells):
            cells["stress"].update(
                {"status": COND_STATUS_OK,
                 "claim": CONDITIONAL_CLAIM_POINT,
                 "value": 1.0}
            )

        problems = self._grid(mutate)
        self.assertTrue(any("below the measured floor" in p for p in problems), problems)

    def test_an_over_wide_published_interval_is_reported(self):
        def mutate(cells):
            cells["bullish"]["interval"] = wilson_interval(2, 2)

        problems = self._grid(mutate)
        self.assertTrue(any("wide" in p for p in problems), problems)

    def test_a_directionless_directional_claim_is_reported(self):
        def mutate(cells):
            cells["bullish"].update(
                {"claim": CONDITIONAL_CLAIM_DIRECTIONAL, "direction": None}
            )
            cells["bullish"].pop("value")

        problems = self._grid(mutate)
        self.assertTrue(any("names no direction" in p for p in problems), problems)

    def test_a_healthy_grid_raises_nothing(self):
        self.assertEqual(self._grid(lambda cells: None), [])


class ConfigContractTests(unittest.TestCase):
    def test_cell_precedence_is_total_and_ok_is_last(self):
        self.assertEqual(
            len(set(CONDITIONAL_CELL_PRECEDENCE)), len(CONDITIONAL_CELL_PRECEDENCE)
        )
        self.assertEqual(CONDITIONAL_CELL_PRECEDENCE[-1], COND_STATUS_OK)

    def test_claim_precedence_runs_weakest_to_strongest(self):
        self.assertEqual(CONDITIONAL_CLAIM_PRECEDENCE[0], CONDITIONAL_CLAIM_INSUFFICIENT)
        self.assertEqual(CONDITIONAL_CLAIM_PRECEDENCE[-1], CONDITIONAL_CLAIM_POINT)

    def test_the_width_cap_lies_inside_the_unit_range(self):
        self.assertGreater(CONDITIONAL_MAX_INTERVAL_WIDTH, 0.0)
        self.assertLess(CONDITIONAL_MAX_INTERVAL_WIDTH, 1.0)


if __name__ == "__main__":
    unittest.main()
