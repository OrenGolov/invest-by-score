"""X4 event robustness tests.

The measurements these tests defend:

* There are 0 event memories on disk and the raw store holds only price and
  fundamentals — X4's blocker is MISSING DATA, not X3's dropped join key.
* A raw drop cannot be the test: an ordinary event in a small book drops
  accuracy 0.1100 purely because it is a fifth of the observations, and the null
  drop scales with share (p95/share ≈ 0.20 from 3 to 20 events).
* The 0.35 ratio bar sits between a uniform maximum of 0.322 and a carrier
  minimum of 0.356 — but only once three-item books are excluded, because with
  them the null reaches 0.350 and touches the bar.
* A source supplying every third event scores 0.425 on the source axis and only
  0.340 on the event axis, so both axes must be tested.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import (
    EVENT_CARRIED,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_ROBUST,
    EVENT_ROBUSTNESS_AXES,
    EVENT_ROBUSTNESS_AXIS_EVENT,
    EVENT_ROBUSTNESS_AXIS_SOURCE,
    EVENT_ROBUSTNESS_MAX_RATIO,
    EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS,
    EVENT_ROBUSTNESS_MIN_ITEMS,
    EVENT_ROBUSTNESS_NOT_EVALUATED,
    EVENT_ROBUSTNESS_VERSION,
)
from core.event_robustness import (
    ER_REASON_CARRIED,
    ER_REASON_NO_ATTRIBUTION,
    ER_REASON_TOO_FEW_ITEMS,
    EventRobustnessError,
    attribution_of,
    carrier_ratio,
    evaluate_event_robustness,
    event_robustness_problems,
    group_by_item,
    render_event_robustness,
)


def uniform_book(events=8, n=30, seed=11):
    """An edge spread evenly over many events and sources."""
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    ids: list[str] = []
    sources: list[str] = []
    for index in range(events):
        outcome = rng.normal(0, 0.05, n)
        predictions.extend(outcome * 0.35 + rng.normal(0, 0.03, n))
        actuals.extend(outcome)
        ids.extend([f"ev{index}"] * n)
        sources.extend([f"src{index % 3}"] * n)
    return predictions, actuals, ids, sources


def source_carried_book(seed=77):
    """One SOURCE carries the edge across several distinct events.

    The case that justifies testing both axes: no single event looks guilty.
    """
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    actuals: list[float] = []
    ids: list[str] = []
    sources: list[str] = []
    for index in range(6):
        n = 40
        outcome = rng.normal(0, 0.05, n)
        carried = index % 3 == 0
        predicted = (
            outcome * 0.95 + rng.normal(0, 0.004, n)
            if carried
            else rng.normal(0, 0.05, n)
        )
        predictions.extend(predicted)
        actuals.extend(outcome)
        ids.extend([f"ev{index}"] * n)
        sources.extend(["src_hot" if carried else f"src{index}"] * n)
    return predictions, actuals, ids, sources


class AttributionTests(unittest.TestCase):
    def test_a_fold_without_attribution_yields_none(self):
        """THE BLOCKING MEASUREMENT: shipped folds carry no events."""
        fold = {"predictions": [1.0], "actuals": [1.0]}
        self.assertIsNone(attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT))
        self.assertIsNone(attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE))

    def test_attribution_is_never_synthesised(self):
        """One synthetic id would leave leave-one-out nothing to leave."""
        fold = {
            "predictions": [1.0] * 5,
            "actuals": [1.0] * 5,
            "validation_start_time": "2025-07-21 13:30:00",
        }
        self.assertIsNone(attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT))

    def test_present_attribution_is_read(self):
        self.assertEqual(
            attribution_of({"events": ["a", "b"]}, EVENT_ROBUSTNESS_AXIS_EVENT),
            ["a", "b"],
        )
        self.assertEqual(
            attribution_of({"sources": ["r"]}, EVENT_ROBUSTNESS_AXIS_SOURCE), ["r"]
        )

    def test_unknown_axis_raises(self):
        with self.assertRaises(EventRobustnessError):
            attribution_of({"events": ["a"]}, "vibes")

    def test_a_string_is_not_an_attribution_sequence(self):
        with self.assertRaises(EventRobustnessError):
            attribution_of({"events": "ev0"}, EVENT_ROBUSTNESS_AXIS_EVENT)

    def test_absent_fold_is_none(self):
        self.assertIsNone(attribution_of(None, EVENT_ROBUSTNESS_AXIS_EVENT))


class GroupingTests(unittest.TestCase):
    def test_cells_carry_share_and_accuracy(self):
        cells = group_by_item([1.0, -1.0], [1.0, 1.0], ["a", "b"])
        self.assertAlmostEqual(cells["a"]["share"], 0.5)
        self.assertAlmostEqual(cells["a"]["directional_accuracy"], 1.0)
        self.assertAlmostEqual(cells["b"]["directional_accuracy"], 0.0)

    def test_a_thin_item_is_marked_unusable(self):
        cells = group_by_item([1.0], [1.0], ["a"])
        self.assertFalse(cells["a"]["usable"])

    def test_a_missing_attribution_raises(self):
        with self.assertRaises(EventRobustnessError):
            group_by_item([1.0], [1.0], [None])

    def test_an_empty_attribution_raises(self):
        with self.assertRaises(EventRobustnessError):
            group_by_item([1.0], [1.0], [""])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(EventRobustnessError):
            group_by_item([1.0, 2.0], [1.0], ["a", "b"])

    def test_an_empty_book_raises(self):
        with self.assertRaises(EventRobustnessError):
            group_by_item([], [], [])


class CarrierRatioTests(unittest.TestCase):
    def test_too_few_items_raises(self):
        with self.assertRaises(EventRobustnessError):
            carrier_ratio([1.0] * 4, [1.0] * 4, ["a", "a", "b", "b"])

    def test_the_ratio_is_drop_over_share(self):
        predictions, actuals, ids, _ = source_carried_book()
        result = carrier_ratio(predictions, actuals, ids)
        expected = result["worst_drop"] / result["carrier_share"]
        self.assertAlmostEqual(result["worst_ratio"], expected)

    def test_shares_sum_to_one(self):
        predictions, actuals, ids, _ = uniform_book()
        result = carrier_ratio(predictions, actuals, ids)
        total = sum(cell["share"] for cell in result["cells"].values())
        self.assertAlmostEqual(total, 1.0)


class EvaluateTests(unittest.TestCase):
    def test_a_uniform_book_is_robust(self):
        """The gate must be able to PASS, or refusing proves nothing."""
        report = evaluate_event_robustness(*uniform_book())
        self.assertEqual(report["verdict"], EVENT_ROBUST)
        self.assertLessEqual(report["worst_ratio"], EVENT_ROBUSTNESS_MAX_RATIO)
        self.assertEqual(event_robustness_problems(report), [])

    def test_a_source_carrier_is_caught(self):
        """THE DECIDING CASE for testing both axes."""
        report = evaluate_event_robustness(*source_carried_book())
        self.assertEqual(report["verdict"], EVENT_CARRIED)
        self.assertEqual(report["reason_code"], ER_REASON_CARRIED)
        self.assertEqual(report["worst_axis"], EVENT_ROBUSTNESS_AXIS_SOURCE)
        self.assertEqual(report["carrier"], "src_hot")

    def test_the_event_axis_alone_would_miss_that_carrier(self):
        """MEASURED: event ratio 0.340 is UNDER the 0.35 bar."""
        predictions, actuals, ids, sources = source_carried_book()
        events_only = evaluate_event_robustness(predictions, actuals, ids, None)
        self.assertEqual(events_only["verdict"], EVENT_ROBUST)
        both = evaluate_event_robustness(predictions, actuals, ids, sources)
        self.assertEqual(both["verdict"], EVENT_CARRIED)

    def test_missing_attribution_is_not_evaluated(self):
        predictions, actuals, _, _ = uniform_book()
        report = evaluate_event_robustness(predictions, actuals, None, None)
        self.assertEqual(report["verdict"], EVENT_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], ER_REASON_NO_ATTRIBUTION)

    def test_missing_attribution_names_both_fixes(self):
        """Ingesting news alone still leaves nothing joinable."""
        predictions, actuals, _, _ = uniform_book()
        report = evaluate_event_robustness(predictions, actuals, None, None)
        self.assertIn("ingest news", report["reason"])
        self.assertIn("onto each validation observation", report["reason"])

    def test_missing_attribution_invents_no_numbers(self):
        predictions, actuals, _, _ = uniform_book()
        report = evaluate_event_robustness(predictions, actuals, None, None)
        self.assertIsNone(report.get("worst_ratio"))
        self.assertIsNone(report.get("pooled_accuracy"))

    def test_too_few_items_is_not_evaluated(self):
        report = evaluate_event_robustness(
            [1.0] * 4, [1.0] * 4, ["a", "a", "b", "b"], None
        )
        self.assertEqual(report["verdict"], EVENT_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], ER_REASON_TOO_FEW_ITEMS)

    def test_one_usable_axis_is_enough_to_decide(self):
        predictions, actuals, ids, _ = source_carried_book()
        report = evaluate_event_robustness(predictions, actuals, ids, None)
        self.assertNotEqual(report["verdict"], EVENT_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["axes_tested"], [EVENT_ROBUSTNESS_AXIS_EVENT])


class RawDropTests(unittest.TestCase):
    """Why the raw drop cannot be the test."""

    def test_an_ordinary_event_in_a_small_book_drops_accuracy_materially(self):
        """MEASURED 0.1100 — larger than many real effects, and innocent."""
        predictions, actuals, ids, _ = uniform_book(events=5, n=20, seed=3)
        result = carrier_ratio(predictions, actuals, ids)
        self.assertGreater(result["carrier_share"], 0.15)
        # The drop is material, yet the RATIO stays within the bar.
        self.assertLessEqual(result["worst_ratio"], EVENT_ROBUSTNESS_MAX_RATIO)

    def test_the_ratio_is_scale_free_across_book_sizes(self):
        """A fixed drop bar would treat these very differently; the ratio
        does not."""
        ratios = []
        for events in (4, 8, 16):
            predictions, actuals, ids, _ = uniform_book(events=events, n=30, seed=21)
            ratios.append(carrier_ratio(predictions, actuals, ids)["worst_ratio"])
        for ratio in ratios:
            self.assertLessEqual(ratio, EVENT_ROBUSTNESS_MAX_RATIO)


class ThresholdTests(unittest.TestCase):
    def test_the_bar_sits_in_the_measured_gap(self):
        """Uniform maximum 0.322 at four-plus items, carrier minimum 0.356."""
        self.assertGreater(EVENT_ROBUSTNESS_MAX_RATIO, 0.322)
        self.assertLess(EVENT_ROBUSTNESS_MAX_RATIO, 0.356)

    def test_both_axes_are_tested(self):
        self.assertIn(EVENT_ROBUSTNESS_AXIS_EVENT, EVENT_ROBUSTNESS_AXES)
        self.assertIn(EVENT_ROBUSTNESS_AXIS_SOURCE, EVENT_ROBUSTNESS_AXES)

    def test_the_item_floor_agrees_with_e6(self):
        self.assertEqual(EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS, EVENT_MEMORY_MIN_ANALOGS)

    def test_at_least_four_items_are_required(self):
        """MEASURED: with three, the uniform null reaches 0.350 and touches
        the 0.35 bar, because removing one of three deletes a third of the
        evidence."""
        self.assertGreaterEqual(EVENT_ROBUSTNESS_MIN_ITEMS, 4)

    def test_a_three_item_book_is_refused(self):
        with self.assertRaises(EventRobustnessError):
            carrier_ratio(
                [1.0] * 6, [1.0] * 6, ["a", "a", "b", "b", "c", "c"]
            )


class ContractTests(unittest.TestCase):
    def test_clean_report_has_no_problems(self):
        self.assertEqual(
            event_robustness_problems(evaluate_event_robustness(*uniform_book())), []
        )

    def test_robust_above_the_bar_is_a_problem(self):
        report = dict(evaluate_event_robustness(*uniform_book()))
        report["worst_ratio"] = 0.9
        self.assertTrue(any("above the" in p for p in event_robustness_problems(report)))

    def test_robust_without_a_tested_axis_is_a_problem(self):
        report = dict(evaluate_event_robustness(*uniform_book()))
        report["axes_tested"] = []
        self.assertTrue(
            any("without testing any axis" in p for p in event_robustness_problems(report))
        )

    def test_carried_without_a_carrier_is_a_problem(self):
        report = dict(evaluate_event_robustness(*source_carried_book()))
        report["carrier"] = None
        self.assertTrue(
            any("must name the event or source" in p for p in event_robustness_problems(report))
        )

    def test_carried_without_an_axis_is_a_problem(self):
        report = dict(evaluate_event_robustness(*source_carried_book()))
        report["worst_axis"] = None
        self.assertTrue(
            any("WHICH AXIS" in p for p in event_robustness_problems(report))
        )

    def test_uses_raw_drop_is_a_problem(self):
        report = dict(evaluate_event_robustness(*uniform_book()))
        report["uses_raw_drop"] = True
        self.assertTrue(any("raw drop" in p for p in event_robustness_problems(report)))

    def test_infers_attribution_is_a_problem(self):
        report = dict(evaluate_event_robustness(*uniform_book()))
        report["infers_attribution"] = True
        self.assertTrue(
            any("never be assigned" in p for p in event_robustness_problems(report))
        )

    def test_blocks_trades_is_a_problem(self):
        report = dict(evaluate_event_robustness(*uniform_book()))
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in event_robustness_problems(report)))

    def test_not_evaluated_without_a_reason_code_is_a_problem(self):
        predictions, actuals, _, _ = uniform_book()
        report = dict(evaluate_event_robustness(predictions, actuals, None, None))
        report["reason_code"] = None
        self.assertTrue(any("must name WHY" in p for p in event_robustness_problems(report)))

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(event_robustness_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        report = evaluate_event_robustness(*uniform_book())
        self.assertEqual(report["version"], EVENT_ROBUSTNESS_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_worst_axis(self):
        report = evaluate_event_robustness(*source_carried_book())
        self.assertIn("worst", "\n".join(render_event_robustness(report)))

    def test_render_shows_both_axes(self):
        text = "\n".join(render_event_robustness(evaluate_event_robustness(*uniform_book())))
        self.assertIn("event", text)
        self.assertIn("source", text)

    def test_render_shows_absent_rather_than_zero(self):
        predictions, actuals, _, _ = uniform_book()
        report = evaluate_event_robustness(predictions, actuals, None, None)
        self.assertIn("ABSENT", "\n".join(render_event_robustness(report)))

    def test_render_returns_lines_not_a_blob(self):
        lines = render_event_robustness(evaluate_event_robustness(*uniform_book()))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedDataTests(unittest.TestCase):
    """The blocking measurement, against what the repo actually ships."""

    def test_collected_memories_never_reach_a_fold(self):
        """The blocker is attribution, not the size of the memory store.

        `data/event_memory.jsonl` is GITIGNORED, so asserting it is empty
        tests the checkout rather than the repository: this assertion passed in
        CI (fresh clone, 0 rows) and FAILED locally (2084 rows). The invariant
        X4 actually rests on is that however many memories exist, none of them
        carry per-observation attribution into a training fold - which is
        measured below from the TRACKED `data/training_runs.jsonl`.
        """
        from core.event_memory import load_memories
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        # However many memories were collected, the folds carry none of them.
        collected = len(load_memories())
        attributed = [
            run["estimator"]
            for run in runs
            if attribution_of(run["folds"][0], EVENT_ROBUSTNESS_AXIS_EVENT) is not None
            or attribution_of(run["folds"][0], EVENT_ROBUSTNESS_AXIS_SOURCE) is not None
        ]
        self.assertEqual(
            attributed,
            [],
            f"{collected} memories exist and {attributed} now carry attribution; "
            f"the robustness test is runnable and X4's blocker is stale",
        )

    def test_no_shipped_fold_carries_event_attribution(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        for run in runs:
            fold = run["folds"][0]
            with self.subTest(estimator=run["estimator"]):
                self.assertIsNone(attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT))
                self.assertIsNone(attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE))

    def test_a_shipped_run_is_not_evaluated(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        fold = runs[0]["folds"][0]
        report = evaluate_event_robustness(
            fold["predictions"],
            fold["actuals"],
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_EVENT),
            attribution_of(fold, EVENT_ROBUSTNESS_AXIS_SOURCE),
        )
        self.assertEqual(report["verdict"], EVENT_ROBUSTNESS_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], ER_REASON_NO_ATTRIBUTION)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
