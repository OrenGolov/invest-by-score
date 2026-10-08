"""L4 source-reliability tests.

The behaviour under test is RESTRAINT. The task says one global score is not
assumed sufficient — it does not say conditioning is free, and MEASURED it is
not: per-cell estimation is five times worse than a global rate on thin
evidence even when quality genuinely varies.
"""

from __future__ import annotations

import random
import unittest

from core.config import (
    SOURCE_BACKING_CELL,
    SOURCE_BACKING_GLOBAL,
    SOURCE_BACKING_NONE,
    SOURCE_BACKING_PRIOR,
    SOURCE_RELIABILITY_ABSENT_IS_ZERO,
    SOURCE_RELIABILITY_BACKINGS,
    SOURCE_RELIABILITY_DIMENSIONS,
    SOURCE_RELIABILITY_MIN_CELL,
    SOURCE_RELIABILITY_SHRINKAGE_K,
)
from core.source_reliability import (
    CONDITIONING_DIMENSIONS,
    SourceReliabilityError,
    conditioning_gain,
    observation_records,
    reliability_problems,
    reliability_report,
    render_scores,
    score_source,
    shrunk_rate,
    tally,
    wilson_interval,
)


def observations(source, rate, count, seed, **context):
    rng = random.Random(seed)
    return [
        {"source": source, "hit": 1 if rng.random() < rate else 0, **context}
        for _ in range(count)
    ]


class ShrinkageTests(unittest.TestCase):
    """The estimator degrades to the global score rather than inventing one."""

    def test_an_empty_cell_is_the_prior(self):
        self.assertAlmostEqual(shrunk_rate(0, 0, 0.75), 0.75)

    def test_abundant_evidence_washes_the_prior_out(self):
        self.assertGreater(shrunk_rate(900, 1000, 0.50), 0.85)

    def test_thin_evidence_stays_near_the_prior(self):
        # 2 of 2 is a 100% raw rate; shrinkage must not report a perfect source.
        self.assertLess(shrunk_rate(2, 2, 0.60), 0.70)

    def test_shrinkage_is_monotone_in_the_observed_rate(self):
        low = shrunk_rate(10, 100, 0.5)
        high = shrunk_rate(90, 100, 0.5)
        self.assertLess(low, high)

    def test_a_negative_k_is_refused(self):
        with self.assertRaises(SourceReliabilityError):
            shrunk_rate(1, 2, 0.5, k=-1)

    def test_the_shipped_k_is_real_shrinkage(self):
        self.assertGreaterEqual(SOURCE_RELIABILITY_SHRINKAGE_K, 1)
        self.assertLessEqual(SOURCE_RELIABILITY_SHRINKAGE_K, 100)


class BackingTests(unittest.TestCase):
    """Four different epistemic states, never collapsed."""

    def test_no_evidence_carries_no_score(self):
        # The shape rule: a score key exists IFF it was measured, because
        # Number(x ?? 0) renders a missing score as 0.0 — the same number as
        # an outlet measured to be worthless.
        result = score_source([], "Unknown")
        self.assertIsNone(result["score"])
        self.assertEqual(result["backing"], SOURCE_BACKING_NONE)

    def test_a_registry_prior_is_named_as_asserted(self):
        result = score_source([], "Known", registry_prior=0.75)
        self.assertEqual(result["backing"], SOURCE_BACKING_PRIOR)
        self.assertAlmostEqual(result["score"], 0.75)

    def test_a_thin_cell_is_global_backed(self):
        thin = observations("T", 0.8, 10, 1, event_type="earnings")
        self.assertEqual(
            score_source(thin, "T", event_type="earnings")["backing"],
            SOURCE_BACKING_GLOBAL,
        )

    def test_a_thick_cell_earns_conditional(self):
        thick = observations("K", 0.8, 60, 2, event_type="earnings")
        result = score_source(thick, "K", event_type="earnings")
        self.assertEqual(result["backing"], SOURCE_BACKING_CELL)
        self.assertTrue(result["cell"])

    def test_the_backings_run_weakest_to_strongest(self):
        self.assertEqual(SOURCE_RELIABILITY_BACKINGS[0], SOURCE_BACKING_NONE)
        self.assertEqual(SOURCE_RELIABILITY_BACKINGS[-1], SOURCE_BACKING_CELL)

    def test_a_prior_and_a_measured_global_rate_stay_distinct(self):
        self.assertNotEqual(SOURCE_BACKING_PRIOR, SOURCE_BACKING_GLOBAL)

    def test_absence_is_never_zero(self):
        self.assertFalse(SOURCE_RELIABILITY_ABSENT_IS_ZERO)


class ConditioningIsEarnedTests(unittest.TestCase):
    """MEASURED: a flat noise band claimed conditioning on 7 of 30 clean runs."""

    def test_a_clean_system_rarely_claims_conditioning(self):
        claims = 0
        for seed in range(40):
            rng = random.Random(700 + seed)
            clean = []
            for event_type in ("earnings", "regulation", "product_launch"):
                clean.extend(
                    {
                        "source": "X",
                        "hit": 1 if rng.random() < 0.6 else 0,
                        "event_type": event_type,
                    }
                    for _ in range(50)
                )
            if conditioning_gain(clean, "event_type")["explains"]:
                claims += 1
        self.assertLessEqual(claims, 6, f"{claims}/40 clean runs claimed conditioning")

    def test_a_real_dependence_is_still_found(self):
        rng = random.Random(99)
        real = []
        for event_type, rate in (
            ("earnings", 0.75),
            ("regulation", 0.45),
            ("product_launch", 0.60),
        ):
            real.extend(
                {
                    "source": "X",
                    "hit": 1 if rng.random() < rate else 0,
                    "event_type": event_type,
                }
                for _ in range(300)
            )
        self.assertTrue(conditioning_gain(real, "event_type")["explains"])

    def test_a_single_cell_cannot_show_dependence(self):
        single = observations("X", 0.6, 100, 3, event_type="earnings")
        gain = conditioning_gain(single, "event_type")
        self.assertFalse(gain["explains"])
        self.assertIn("two cells", gain["reason"])

    def test_an_unknown_dimension_is_refused(self):
        with self.assertRaises(SourceReliabilityError):
            conditioning_gain([], "not_a_dimension")

    def test_the_reason_explains_a_negative_verdict(self):
        rng = random.Random(5)
        flat = []
        for event_type in ("a", "b"):
            flat.extend(
                {
                    "source": "X",
                    "hit": 1 if rng.random() < 0.6 else 0,
                    "event_type": event_type,
                }
                for _ in range(200)
            )
        gain = conditioning_gain(flat, "event_type")
        if not gain["explains"]:
            self.assertIn("noise", gain["reason"])


class RecordTests(unittest.TestCase):
    def test_a_row_without_an_outlet_is_dropped(self):
        self.assertEqual(observation_records([{"hit": 1}]), [])

    def test_a_row_without_a_scored_outcome_is_dropped(self):
        # Counting it as a miss would penalise a source for the system's own
        # missing join.
        self.assertEqual(observation_records([{"source": "A"}]), [])
        self.assertEqual(observation_records([{"source": "A", "hit": None}]), [])

    def test_a_scored_row_is_kept(self):
        records = observation_records([{"source": "A", "hit": True}])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["hit"], 1)

    def test_tally_counts_per_cell(self):
        counts = tally(
            observation_records(observations("A", 1.0, 5, 1, event_type="earnings")),
            ("event_type",),
        )
        self.assertEqual(counts[("A", "earnings")], {"hits": 5, "total": 5})

    def test_an_empty_source_name_is_refused(self):
        with self.assertRaises(SourceReliabilityError):
            score_source([], "   ")


class IntervalTests(unittest.TestCase):
    def test_a_small_sample_yields_a_wide_interval(self):
        low, high = wilson_interval(3, 5)
        self.assertGreater(high - low, 0.5)

    def test_a_large_sample_yields_a_narrow_interval(self):
        low, high = wilson_interval(300, 500)
        self.assertLess(high - low, 0.1)

    def test_an_empty_sample_has_no_interval(self):
        self.assertIsNone(wilson_interval(0, 0))


class ReportTests(unittest.TestCase):
    def test_an_empty_report_is_contract_clean_and_names_the_gap(self):
        report = reliability_report([])
        self.assertEqual(reliability_problems(report), [])
        self.assertIn("MEASURED", report["join_gap"])

    def test_a_populated_report_is_contract_clean(self):
        report = reliability_report(
            observations("A", 0.7, 60, 11, event_type="earnings")
            + observations("B", 0.5, 60, 12, event_type="earnings")
        )
        self.assertEqual(reliability_problems(report), [])
        self.assertEqual(report["sources"], 2)

    def test_render_returns_nothing_when_there_is_nothing(self):
        self.assertEqual(render_scores(reliability_report([])), [])

    def test_the_report_states_that_absence_is_not_worthlessness(self):
        self.assertIn("different facts", reliability_report([])["absence_note"])

    def test_the_dimensions_are_the_tasks_dimensions(self):
        self.assertEqual(SOURCE_RELIABILITY_DIMENSIONS[0], "source")
        self.assertEqual(
            set(CONDITIONING_DIMENSIONS), {"event_type", "sector", "horizon"}
        )

    def test_a_conditional_score_below_the_floor_is_a_problem(self):
        forged = {
            "scores": {
                "X": {
                    "backing": SOURCE_BACKING_CELL,
                    "cell": {"event_type": "earnings"},
                    "cell_observations": SOURCE_RELIABILITY_MIN_CELL - 1,
                    "score": 0.9,
                }
            }
        }
        self.assertTrue(reliability_problems(forged))

    def test_a_no_evidence_score_carrying_a_value_is_a_problem(self):
        forged = {"scores": {"X": {"backing": SOURCE_BACKING_NONE, "score": 0.0}}}
        self.assertTrue(reliability_problems(forged))


if __name__ == "__main__":
    unittest.main()
