"""X5 feature ablation tests.

The measurements these tests defend:

* All 16 features the pipeline produces are price/volume derivatives, so four
  of the five named groups (news, macro, sentiment, fundamentals) have ZERO
  features and cannot be ablated at all.
* An absent group is ABSENT, never NO_VALUE — saying "removing news changed
  nothing" about a group that was never present presents an untested source as
  a tested one.
* The one group that CAN be ablated is HARMFUL: removing all 16 technical
  features leaves `historical_mean`, which is better by 0.01028 rmse.
"""

from __future__ import annotations

import unittest

from core.config import (
    ABLATION_ABSENT,
    ABLATION_ADDS_VALUE,
    ABLATION_GROUPS,
    ABLATION_GROUP_FUNDAMENTAL,
    ABLATION_GROUP_MACRO,
    ABLATION_GROUP_NEWS,
    ABLATION_GROUP_SENTIMENT,
    ABLATION_GROUP_TECHNICAL,
    ABLATION_HARMFUL,
    ABLATION_METRIC,
    ABLATION_MIN_IMPROVEMENT,
    ABLATION_NOT_EVALUATED,
    ABLATION_NO_VALUE,
    FEATURE_ABLATION_VERSION,
    REGIME_SPECIALIZATION_MARGIN,
)
from core.feature_ablation import (
    FeatureAblationError,
    ablate_group,
    ablation_problems,
    evaluate_ablation,
    evaluate_ablation as _evaluate,  # noqa: F401  (kept for readability below)
    group_features,
    group_of,
    incremental_value,
    render_ablation,
)

SHIPPED_FEATURES = [
    "change_1d",
    "change_5d",
    "change_20d",
    "change_60d",
    "ma_50",
    "ma_100",
    "ma_150",
    "ma_200",
    "price_vs_ma_50",
    "price_vs_ma_100",
    "price_vs_ma_150",
    "price_vs_ma_200",
    "rsi",
    "trend_vs_20d_mean",
    "volatility",
    "volume_ratio_20d",
]


class GroupOfTests(unittest.TestCase):
    def test_every_shipped_feature_is_technical(self):
        """THE DECIDING MEASUREMENT: all 16 are price/volume derivatives."""
        for feature in SHIPPED_FEATURES:
            with self.subTest(feature=feature):
                self.assertEqual(group_of(feature), ABLATION_GROUP_TECHNICAL)

    def test_price_vs_ma_is_not_claimed_by_the_ma_prefix(self):
        """Most-specific-first matters: 'ma_' would otherwise swallow it."""
        self.assertEqual(group_of("price_vs_ma_50"), ABLATION_GROUP_TECHNICAL)

    def test_each_named_group_has_a_prefix(self):
        for feature, group in (
            ("news_polarity_5d", ABLATION_GROUP_NEWS),
            ("macro_cpi_yoy", ABLATION_GROUP_MACRO),
            ("sentiment_score", ABLATION_GROUP_SENTIMENT),
            ("fundamental_pe", ABLATION_GROUP_FUNDAMENTAL),
            ("valuation_pb", ABLATION_GROUP_FUNDAMENTAL),
        ):
            with self.subTest(feature=feature):
                self.assertEqual(group_of(feature), group)

    def test_macro_is_not_swallowed_by_the_technical_ma_prefix(self):
        """`macro_` and `ma_` are close enough that ORDER is load-bearing.

        MEASURED: with `macro_` added to the technical prefixes AND technicals
        checked first, `macro_cpi_yoy` is filed as technical and the macro
        group then reports ABSENT — an untested source presented as absent.
        """
        self.assertEqual(group_of("macro_cpi_yoy"), ABLATION_GROUP_MACRO)
        self.assertEqual(group_of("ma_50"), ABLATION_GROUP_TECHNICAL)

    def test_no_feature_is_claimed_by_two_groups(self):
        """A feature in two groups would be removed by both ablations."""
        probes = [
            "price_vs_ma_50",
            "ma_50",
            "macro_cpi_yoy",
            "news_polarity_5d",
            "sentiment_score",
            "fundamental_pe",
            "valuation_pb",
        ]
        for probe in probes:
            with self.subTest(probe=probe):
                owner = group_of(probe)
                self.assertIsNotNone(owner)
                grouped = group_features([probe])
                owning = [g for g, names in grouped.items() if names]
                self.assertEqual(owning, [owner])

    def test_an_unrecognised_feature_is_none_not_a_guess(self):
        """Sweeping it into a group would make that ablation remove something
        the group does not own."""
        self.assertIsNone(group_of("mystery_signal"))

    def test_an_empty_name_raises(self):
        with self.assertRaises(FeatureAblationError):
            group_of("")


class GroupFeaturesTests(unittest.TestCase):
    def test_every_declared_group_appears_even_when_empty(self):
        """An absent group has to be VISIBLE to be reported as ABSENT."""
        grouped = group_features(SHIPPED_FEATURES)
        self.assertEqual(sorted(grouped), sorted(ABLATION_GROUPS))

    def test_four_groups_are_empty_on_the_shipped_set(self):
        grouped = group_features(SHIPPED_FEATURES)
        empty = [g for g, names in grouped.items() if not names]
        self.assertEqual(
            sorted(empty),
            sorted(
                [
                    ABLATION_GROUP_NEWS,
                    ABLATION_GROUP_MACRO,
                    ABLATION_GROUP_SENTIMENT,
                    ABLATION_GROUP_FUNDAMENTAL,
                ]
            ),
        )

    def test_technicals_hold_all_sixteen(self):
        grouped = group_features(SHIPPED_FEATURES)
        self.assertEqual(len(grouped[ABLATION_GROUP_TECHNICAL]), 16)

    def test_an_unplaceable_feature_raises(self):
        with self.assertRaises(FeatureAblationError):
            group_features(SHIPPED_FEATURES + ["mystery_signal"])


class IncrementalValueTests(unittest.TestCase):
    def test_removing_a_useful_group_worsens_the_metric(self):
        self.assertAlmostEqual(incremental_value(0.10, 0.15), 0.05)

    def test_removing_a_harmful_group_improves_it(self):
        self.assertAlmostEqual(incremental_value(0.13192, 0.12164), -0.01028, places=5)

    def test_a_missing_side_is_none_not_zero(self):
        """An unmeasured side is not a zero difference."""
        self.assertIsNone(incremental_value(0.10, None))
        self.assertIsNone(incremental_value(None, 0.10))


class AblateGroupTests(unittest.TestCase):
    def test_a_group_with_no_features_is_absent(self):
        report = ablate_group(ABLATION_GROUP_NEWS, [])
        self.assertEqual(report["verdict"], ABLATION_ABSENT)
        self.assertIsNone(report["value"])

    def test_absent_says_never_tested_not_worthless(self):
        report = ablate_group(ABLATION_GROUP_NEWS, [])
        self.assertIn("NEVER TESTED", report["reason"])

    def test_a_harmful_group_is_reported_as_harmful(self):
        """THE MEASURED CASE: technicals at -0.01028."""
        report = ablate_group(
            ABLATION_GROUP_TECHNICAL,
            SHIPPED_FEATURES,
            with_group=0.13192,
            without_group=0.12164,
        )
        self.assertEqual(report["verdict"], ABLATION_HARMFUL)
        self.assertLess(report["value"], 0)

    def test_a_valuable_group_adds_value(self):
        """The gate must be able to report value, or it proves nothing."""
        report = ablate_group(
            ABLATION_GROUP_TECHNICAL,
            SHIPPED_FEATURES,
            with_group=0.10,
            without_group=0.20,
        )
        self.assertEqual(report["verdict"], ABLATION_ADDS_VALUE)

    def test_a_change_inside_the_margin_is_no_value(self):
        report = ablate_group(
            ABLATION_GROUP_TECHNICAL,
            SHIPPED_FEATURES,
            with_group=0.1000,
            without_group=0.1001,
        )
        self.assertEqual(report["verdict"], ABLATION_NO_VALUE)

    def test_a_present_group_without_scores_is_not_evaluated(self):
        report = ablate_group(ABLATION_GROUP_TECHNICAL, SHIPPED_FEATURES)
        self.assertEqual(report["verdict"], ABLATION_NOT_EVALUATED)

    def test_not_evaluated_is_distinct_from_absent(self):
        present = ablate_group(ABLATION_GROUP_TECHNICAL, SHIPPED_FEATURES)
        missing = ablate_group(ABLATION_GROUP_NEWS, [])
        self.assertNotEqual(present["verdict"], missing["verdict"])

    def test_an_unknown_group_raises(self):
        with self.assertRaises(FeatureAblationError):
            ablate_group("vibes", ["x"])


class EvaluateTests(unittest.TestCase):
    def shipped(self):
        return evaluate_ablation(
            SHIPPED_FEATURES,
            {ABLATION_GROUP_TECHNICAL: {"with": 0.13192, "without": 0.12164}},
            estimator="random_forest",
        )

    def test_the_shipped_set_cannot_prove_which_sources_matter(self):
        report = self.shipped()
        self.assertEqual(report["verdict"], ABLATION_NOT_EVALUATED)
        self.assertEqual(len(report["absent_groups"]), 4)

    def test_the_one_testable_group_is_harmful(self):
        report = self.shipped()
        self.assertEqual(report["harmful_groups"], [ABLATION_GROUP_TECHNICAL])

    def test_absent_groups_are_named(self):
        report = self.shipped()
        for group in (
            ABLATION_GROUP_NEWS,
            ABLATION_GROUP_MACRO,
            ABLATION_GROUP_SENTIMENT,
            ABLATION_GROUP_FUNDAMENTAL,
        ):
            self.assertIn(group, report["absent_groups"])

    def test_every_group_is_reported_even_when_absent(self):
        report = self.shipped()
        self.assertEqual(sorted(report["groups"]), sorted(ABLATION_GROUPS))

    def test_a_complete_feature_set_can_reach_a_real_verdict(self):
        """With every group present, the suite is no longer NOT_EVALUATED."""
        features = SHIPPED_FEATURES + [
            "news_polarity_5d",
            "macro_cpi_yoy",
            "sentiment_score",
            "fundamental_pe",
        ]
        scores = {
            ABLATION_GROUP_TECHNICAL: {"with": 0.10, "without": 0.20},
            ABLATION_GROUP_NEWS: {"with": 0.10, "without": 0.12},
            ABLATION_GROUP_MACRO: {"with": 0.10, "without": 0.11},
            ABLATION_GROUP_SENTIMENT: {"with": 0.10, "without": 0.1001},
            ABLATION_GROUP_FUNDAMENTAL: {"with": 0.10, "without": 0.13},
        }
        report = evaluate_ablation(features, scores)
        self.assertEqual(report["verdict"], ABLATION_ADDS_VALUE)
        self.assertEqual(report["absent_groups"], [])
        self.assertEqual(ablation_problems(report), [])

    def test_absent_probabilities_are_not_evaluated(self):
        report = evaluate_ablation(None)
        self.assertEqual(report["verdict"], ABLATION_NOT_EVALUATED)


class ThresholdTests(unittest.TestCase):
    def test_the_margin_is_l7s_measured_one(self):
        self.assertEqual(ABLATION_MIN_IMPROVEMENT, REGIME_SPECIALIZATION_MARGIN)

    def test_all_five_roadmap_groups_are_tested(self):
        for group in (
            ABLATION_GROUP_TECHNICAL,
            ABLATION_GROUP_NEWS,
            ABLATION_GROUP_MACRO,
            ABLATION_GROUP_SENTIMENT,
            ABLATION_GROUP_FUNDAMENTAL,
        ):
            self.assertIn(group, ABLATION_GROUPS)

    def test_the_metric_is_an_error_metric(self):
        self.assertEqual(ABLATION_METRIC, "rmse")


class ContractTests(unittest.TestCase):
    def clean(self):
        features = SHIPPED_FEATURES + [
            "news_polarity_5d",
            "macro_cpi_yoy",
            "sentiment_score",
            "fundamental_pe",
        ]
        scores = {g: {"with": 0.10, "without": 0.20} for g in ABLATION_GROUPS}
        return evaluate_ablation(features, scores)

    def test_clean_report_has_no_problems(self):
        self.assertEqual(ablation_problems(self.clean()), [])

    def test_absent_means_no_value_is_a_problem(self):
        report = dict(self.clean())
        report["absent_means_no_value"] = True
        self.assertTrue(
            any("untested source" in p for p in ablation_problems(report))
        )

    def test_an_absent_group_carrying_a_value_is_a_problem(self):
        report = dict(evaluate_ablation(SHIPPED_FEATURES))
        groups = dict(report["groups"])
        entry = dict(groups[ABLATION_GROUP_NEWS])
        entry["value"] = 0.0
        groups[ABLATION_GROUP_NEWS] = entry
        report["groups"] = groups
        self.assertTrue(
            any("never tested cannot have" in p for p in ablation_problems(report))
        )

    def test_concluding_value_while_groups_were_untested_is_a_problem(self):
        report = dict(evaluate_ablation(SHIPPED_FEATURES))
        report["verdict"] = ABLATION_ADDS_VALUE
        self.assertTrue(
            any("never tested" in p for p in ablation_problems(report))
        )

    def test_a_missing_roadmap_group_is_a_problem(self):
        report = dict(self.clean())
        groups = dict(report["groups"])
        groups.pop(ABLATION_GROUP_NEWS)
        report["groups"] = groups
        self.assertTrue(any("roadmap names" in p for p in ablation_problems(report)))

    def test_blocks_trades_is_a_problem(self):
        report = dict(self.clean())
        report["blocks_trades"] = True
        self.assertTrue(any("promotes" in p for p in ablation_problems(report)))

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(ablation_problems(["x"]), ["report is not a mapping"])

    def test_version_is_carried(self):
        self.assertEqual(self.clean()["version"], FEATURE_ABLATION_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_untested_groups(self):
        text = "\n".join(render_ablation(evaluate_ablation(SHIPPED_FEATURES)))
        self.assertIn("NEVER TESTED", text)

    def test_render_shows_absent_rather_than_zero(self):
        text = "\n".join(render_ablation(evaluate_ablation(SHIPPED_FEATURES)))
        self.assertIn("ABSENT", text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_ablation(evaluate_ablation(SHIPPED_FEATURES))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class ShippedRunTests(unittest.TestCase):
    """The measurement itself, against the runs the repo ships."""

    def test_the_shipped_feature_set_is_entirely_technical(self):
        from core.training import load_training_runs

        runs = load_training_runs()
        if not runs:
            self.skipTest("no training runs on disk")
        grouped = group_features(runs[0]["feature_names"])
        self.assertEqual(len(grouped[ABLATION_GROUP_TECHNICAL]), 16)
        for group in (
            ABLATION_GROUP_NEWS,
            ABLATION_GROUP_MACRO,
            ABLATION_GROUP_SENTIMENT,
            ABLATION_GROUP_FUNDAMENTAL,
        ):
            with self.subTest(group=group):
                self.assertEqual(grouped[group], [])

    def test_removing_all_technicals_improves_the_metric(self):
        """MEASURED: historical_mean (no features) beats the best learned model."""
        from core.training import load_training_runs

        runs = {r["estimator"]: r for r in load_training_runs()}
        if "historical_mean" not in runs:
            self.skipTest("no shipped training runs")
        without = runs["historical_mean"]["metrics"]["rmse"]
        with_group = min(
            r["metrics"]["rmse"]
            for name, r in runs.items()
            if name != "historical_mean"
        )
        self.assertLess(without, with_group)
        report = ablate_group(
            ABLATION_GROUP_TECHNICAL,
            runs["historical_mean"]["feature_names"],
            with_group=with_group,
            without_group=without,
        )
        self.assertEqual(report["verdict"], ABLATION_HARMFUL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
