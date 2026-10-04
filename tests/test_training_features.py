"""Tests for training feature routing (Priority 2).

Pure: every score result is a stub, so nothing fetches, scores or touches a
network. The routing decision is what is under test, not the producers.
"""

from __future__ import annotations

import unittest

from core.config import (
    TRAINING_INCLUDE_CONTEXTUAL_FEATURES,
    TRAINING_INCLUDE_FUNDAMENTAL_FEATURES,
)
from core.training_features import (
    CONTEXTUAL_FEATURE_NAMES,
    FUNDAMENTAL_FEATURE_NAMES,
    FUNDAMENTAL_NEUTRAL_VECTOR,
    ROUTABLE_FEATURE_NAMES,
    STATE_ABSENT,
    STATE_LIVE,
    STATE_NEUTRAL_DEFAULTS,
    FeatureRoutingReport,
    fundamental_feature_state,
    merged_feature_surface,
    optional_feature_names,
    render_routing,
    zero_variance_features,
)


class _Result:
    """A score result stub exposing only what routing reads."""

    def __init__(self, fundamental=None, as_of="2026-09-15"):
        self.fundamental_features = fundamental
        self.as_of = as_of


def _contract(name, value, source_id="news_news"):
    return {
        "name": name,
        "value": value,
        "as_of": "2026-09-15",
        "published_time": "2026-09-15",
        "source_id": source_id,
    }


class FundamentalStateTests(unittest.TestCase):
    def test_the_measured_neutral_vector_is_recognised(self):
        """The exact values measured with no ALPHAVANTAGE_API_KEY."""
        self.assertEqual(
            fundamental_feature_state(dict(FUNDAMENTAL_NEUTRAL_VECTOR)),
            STATE_NEUTRAL_DEFAULTS,
        )

    def test_one_differing_value_makes_it_live(self):
        values = dict(FUNDAMENTAL_NEUTRAL_VECTOR)
        values["revenue_growth"] = 7.3
        self.assertEqual(fundamental_feature_state(values), STATE_LIVE)

    def test_empty_and_none_are_absent(self):
        self.assertEqual(fundamental_feature_state({}), STATE_ABSENT)
        self.assertEqual(fundamental_feature_state(None), STATE_ABSENT)

    def test_non_numeric_values_are_absent(self):
        self.assertEqual(
            fundamental_feature_state({"revenue_growth": "n/a"}), STATE_ABSENT
        )

    def test_absent_and_neutral_are_distinct_states(self):
        """"No fundamentals" and "fundamentals that are constants" differ.

        The first is honest absence; the second is absence wearing a
        number, and only the second can mislead a reader.
        """
        self.assertNotEqual(STATE_ABSENT, STATE_NEUTRAL_DEFAULTS)


class ZeroVarianceTests(unittest.TestCase):
    def test_a_constant_column_is_reported(self):
        rows = [{"a": 1.0, "b": 5.0}, {"a": 2.0, "b": 5.0}, {"a": 3.0, "b": 5.0}]
        self.assertEqual(zero_variance_features(rows), ["b"])

    def test_a_single_row_yields_no_verdict(self):
        """One row cannot establish variance either way."""
        self.assertEqual(zero_variance_features([{"a": 1.0}]), [])

    def test_all_varying_columns_report_nothing(self):
        rows = [{"a": 1.0}, {"a": 2.0}]
        self.assertEqual(zero_variance_features(rows), [])

    def test_the_measured_fundamental_vector_is_all_constant(self):
        """The case this check exists for.

        MEASURED across AAPL/MSFT/NVDA/KO/LLY, all five fundamental
        features were identical, so every one is a constant column.
        """
        rows = [dict(FUNDAMENTAL_NEUTRAL_VECTOR) for _ in range(5)]
        self.assertEqual(
            zero_variance_features(rows), sorted(FUNDAMENTAL_NEUTRAL_VECTOR)
        )


class MergeTests(unittest.TestCase):
    def test_the_chart_surface_is_preserved(self):
        chart = {"rsi": _contract("rsi", 55.0, "chart_features")}
        merged, _ = merged_feature_surface(chart, _Result())
        self.assertIn("rsi", merged)

    def test_the_chart_surface_wins_a_name_collision(self):
        """The dataset must agree with the decision that produced it.

        Silently replacing a CONSUMED contract with a contextual one of the
        same name would make the two disagree.
        """
        chart = {"sentiment_score": _contract("sentiment_score", 1.0, "chart_features")}
        merged, report = merged_feature_surface(chart, _Result())
        self.assertEqual(merged["sentiment_score"]["value"], 1.0)
        if "sentiment_score" in report.skipped:
            self.assertIn("chart surface", report.skipped["sentiment_score"])

    def test_neutral_fundamentals_are_refused_with_a_reason(self):
        merged, report = merged_feature_surface(
            {}, _Result(fundamental=dict(FUNDAMENTAL_NEUTRAL_VECTOR))
        )
        for name in FUNDAMENTAL_FEATURE_NAMES:
            self.assertNotIn(name, merged)
        self.assertIn("fundamental_features", report.skipped)
        self.assertIn("neutral_defaults", report.skipped["fundamental_features"])

    def test_absent_fundamentals_are_also_refused(self):
        merged, report = merged_feature_surface({}, _Result(fundamental=None))
        self.assertEqual(merged, {})
        self.assertIn("fundamental_features", report.skipped)

    def test_the_report_counts_what_it_routed(self):
        chart = {"rsi": _contract("rsi", 55.0, "chart_features")}
        _merged, report = merged_feature_surface(chart, _Result())
        self.assertEqual(report.chart_features, ("rsi",))

    def test_all_features_unions_the_three_groups(self):
        report = FeatureRoutingReport(
            chart_features=("rsi",),
            contextual_features=("sentiment_score",),
            fundamental_features=("revenue_growth",),
        )
        self.assertEqual(
            report.all_features, ("revenue_growth", "rsi", "sentiment_score")
        )


class OptionalFeatureTests(unittest.TestCase):
    def test_chart_features_are_not_optional(self):
        """A row without its chart state is not a row."""
        surface = {"rsi": _contract("rsi", 55.0, "chart_features")}
        self.assertNotIn("rsi", optional_feature_names(surface))

    def test_a_non_chart_source_is_optional(self):
        surface = {"news_sentiment_score": _contract("news_sentiment_score", 0.3)}
        self.assertIn("news_sentiment_score", optional_feature_names(surface))

    def test_every_fundamental_name_is_optional(self):
        optional = optional_feature_names({})
        for name in FUNDAMENTAL_FEATURE_NAMES:
            with self.subTest(feature=name):
                self.assertIn(name, optional)


class RoutableNameTests(unittest.TestCase):
    def test_every_routable_name_is_in_the_default_registry(self):
        """Routing widens what may be ASKED for, never what bypasses M1."""
        from core.feature_registry import build_default_registry

        registry = build_default_registry()
        for name in ROUTABLE_FEATURE_NAMES:
            with self.subTest(feature=name):
                self.assertTrue(registry.is_registered(name))

    def test_routable_covers_both_contextual_and_fundamental(self):
        for name in (*CONTEXTUAL_FEATURE_NAMES, *FUNDAMENTAL_FEATURE_NAMES):
            with self.subTest(feature=name):
                self.assertIn(name, ROUTABLE_FEATURE_NAMES)

    def test_the_four_missing_x5_groups_are_all_covered(self):
        """X5 named news, macro, sentiment and fundamentals."""
        self.assertIn("news_sentiment_score", ROUTABLE_FEATURE_NAMES)
        self.assertIn("macro_regime_score", ROUTABLE_FEATURE_NAMES)
        self.assertIn("sentiment_score", ROUTABLE_FEATURE_NAMES)
        self.assertTrue(
            set(FUNDAMENTAL_FEATURE_NAMES) <= set(ROUTABLE_FEATURE_NAMES)
        )


class ConfigPostureTests(unittest.TestCase):
    def test_contextual_routing_is_on(self):
        """Off restores the X5 blocker this work exists to fix."""
        self.assertTrue(TRAINING_INCLUDE_CONTEXTUAL_FEATURES)

    def test_fundamental_routing_is_off_until_the_data_is_real(self):
        """Not a preference: the values are measured constants today."""
        self.assertFalse(TRAINING_INCLUDE_FUNDAMENTAL_FEATURES)


class RenderTests(unittest.TestCase):
    def test_render_names_the_skipped_group(self):
        _merged, report = merged_feature_surface(
            {}, _Result(fundamental=dict(FUNDAMENTAL_NEUTRAL_VECTOR))
        )
        text = "\n".join(render_routing(report))
        self.assertIn("SKIPPED", text)

    def test_render_shows_all_three_groups(self):
        text = "\n".join(render_routing(FeatureRoutingReport()))
        for label in ("chart", "contextual", "fundamental"):
            self.assertIn(label, text)


if __name__ == "__main__":
    unittest.main()


class EstimatorSelectionTests(unittest.TestCase):
    """Per-family selection, and the gate it must not weaken."""

    def test_a_baseline_does_not_receive_contextual_features(self):
        """MEASURED: contextual features declare compatibility with
        ['linear','logistic','tree','boosting'] and NOT 'baseline_mean'.

        A feature-free baseline consuming a news score would stop being a
        reference point, which is the whole purpose of a baseline.
        """
        from core.feature_registry import build_default_registry
        from core.training import estimator_feature_names

        class _DS:
            feature_names = ["rsi", "regime_probability_proxy"]

        selected = estimator_feature_names(
            _DS(), "historical_mean", build_default_registry()
        )
        self.assertIn("rsi", selected)
        self.assertNotIn("regime_probability_proxy", selected)

    def test_a_learned_family_does_receive_them(self):
        from core.feature_registry import build_default_registry
        from core.training import estimator_feature_names

        class _DS:
            feature_names = ["rsi", "regime_probability_proxy"]

        selected = estimator_feature_names(
            _DS(), "ridge", build_default_registry()
        )
        self.assertIn("regime_probability_proxy", selected)

    def test_an_unregistered_feature_is_KEPT_so_the_gate_can_abort(self):
        """The regression this filter caused once, locked down.

        A first version filtered on ANY registry problem, which silently
        dropped an unregistered feature and defeated the M1 gate that
        exists to refuse one. Unregistered is a governance failure;
        family-incompatible is a legitimate selection. Only the second may
        be filtered.
        """
        from core.feature_registry import build_default_registry
        from core.training import estimator_feature_names

        class _DS:
            feature_names = ["rsi", "totally_unregistered_feature"]

        selected = estimator_feature_names(
            _DS(), "ridge", build_default_registry()
        )
        self.assertIn("totally_unregistered_feature", selected)
