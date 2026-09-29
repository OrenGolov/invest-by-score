"""A4 tests — non-price features reach the training pipeline, or are refused honestly.

MEASURED before A4: 40 features registered, 16 used in training, 24 never trained
on — and the 24 were every fundamental, news, sentiment and macro feature. The
project's central claim, that this evidence predicts returns, had never been
tested because the features were never in a training row.

The gap was NOT that the features did not exist. `contextual_feature_surface`
already built registry-conformant, fail-closed contracts for the auditor; the
dataset builder simply never saw it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.contract_verification import (
    _FUNDAMENTAL_FEATURE_INPUTS,
    _FUNDAMENTAL_LAUNDERED_DEFAULTS,
    contextual_feature_surface,
    fundamental_feature_problems,
    fundamental_feature_surface,
)
from core.feature_registry import build_default_registry


def result(**kwargs):
    base = {
        "as_of": "2026-09-21 00:00:00",
        "news_snapshot": {},
        "macro_snapshot": {},
        "market_regime_snapshot": {},
        "sentiment_snapshot": {},
        "fundamental_features": {},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def fundamentals(*, status="live_provider", metrics=None, scores=None):
    return {
        "source_status": status,
        "source_contract": {
            "source_id": "alpha_vantage_overview",
            "as_of": "2026-09-21 13:30:00",
            # The REGISTERED version. A drifted one is refused by the M1 gate,
            # which is what caught this test's first invented value.
            "calculation_version": "fundamental-feature-v1",
        },
        "valuation_metrics": metrics or {},
        **(scores or {}),
    }


class TheContextualSurfaceReachesTrainingTests(unittest.TestCase):
    """The wiring A4 added: the builder now merges what the auditor checked."""

    def test_the_dataset_builder_merges_the_contextual_surface(self):
        import inspect

        from core import training_dataset

        source = inspect.getsource(training_dataset.build_training_dataset)
        self.assertIn("contextual_feature_surface", source)
        self.assertIn("fundamental_feature_surface", source)

    def test_a_live_regime_contributes_a_feature(self):
        surface = contextual_feature_surface(
            result(
                market_regime_snapshot={
                    "status": "OK",
                    "probability_proxy": 0.655,
                    "as_of": "2026-09-21 00:00:00",
                    "source_id": "yahoo_finance_chart",
                    "calculation_version": "regime-classifier-v1",
                    "published_time": "2026-09-18 13:30:00",
                }
            )
        )
        self.assertIn("regime_probability_proxy", surface)
        self.assertAlmostEqual(surface["regime_probability_proxy"]["value"], 0.655)

    def test_an_unavailable_agent_contributes_nothing(self):
        # Fail-closed: a provider outage must not become a neutral feature.
        for status in ("UNAVAILABLE", "INVALID", "STALE", "INCOMPLETE"):
            with self.subTest(status=status):
                surface = contextual_feature_surface(
                    result(
                        news_snapshot={"status": status, "sentiment_score": 0.4}
                    )
                )
                self.assertNotIn("news_sentiment_score", surface)

    def test_an_ok_agent_with_no_value_contributes_nothing(self):
        surface = contextual_feature_surface(
            result(news_snapshot={"status": "OK", "sentiment_score": None})
        )
        self.assertEqual(surface, {})


class ImputationIsRefusedTests(unittest.TestCase):
    """The defect that made this more than a wiring change.

    MEASURED on VOO: `source_status` is "live_provider" and every raw input is
    None, yet the derived scores are 5.0 / 5.0 / 7.0 / 3.0 / 10.0 — the default
    ladder in `_build_fundamental_features`. Those values are CONSTANT for every
    ticker without fundamentals, so a model would learn "this row is an ETF" and
    X5 would call the family incremental on a survivorship marker.
    """

    def test_absent_raw_inputs_yield_no_features(self):
        surface = fundamental_feature_surface(
            result(
                fundamental_features=fundamentals(
                    metrics={key: None for _, key in _FUNDAMENTAL_FEATURE_INPUTS},
                    scores={
                        "revenue_growth": 5.0,
                        "margin_quality": 5.0,
                        "free_cash_flow_quality": 3.0,
                        "balance_sheet_quality": 10.0,
                        "valuation_quality": 7.0,
                    },
                )
            )
        )
        self.assertEqual(surface, {})

    def test_a_present_raw_input_yields_its_feature(self):
        surface = fundamental_feature_surface(
            result(
                fundamental_features=fundamentals(
                    metrics={"revenue_growth": 0.24},
                    scores={"revenue_growth": 8.5},
                )
            )
        )
        self.assertIn("revenue_growth", surface)
        self.assertEqual(surface["revenue_growth"]["value"], 8.5)

    def test_a_laundered_default_is_refused(self):
        # price_to_book defaults to 4.0 and is written back as if observed, so an
        # input equal to the default is indistinguishable from an absent one.
        self.assertIn("price_to_book", _FUNDAMENTAL_LAUNDERED_DEFAULTS)
        surface = fundamental_feature_surface(
            result(
                fundamental_features=fundamentals(
                    metrics={"price_to_book": 4.0},
                    scores={"valuation_quality": 7.0},
                )
            )
        )
        self.assertNotIn("valuation_quality", surface)

    def test_a_real_price_to_book_is_admitted(self):
        # The guard is narrow: it refuses the placeholder, not the metric.
        surface = fundamental_feature_surface(
            result(
                fundamental_features=fundamentals(
                    metrics={"price_to_book": 11.2},
                    scores={"valuation_quality": 2.0},
                )
            )
        )
        self.assertIn("valuation_quality", surface)

    def test_a_non_live_source_yields_nothing(self):
        for status in ("provider_key_required", "unknown", ""):
            with self.subTest(status=status):
                surface = fundamental_feature_surface(
                    result(
                        fundamental_features=fundamentals(
                            status=status,
                            metrics={"revenue_growth": 0.24},
                            scores={"revenue_growth": 8.5},
                        )
                    )
                )
                self.assertEqual(surface, {})

    def test_a_missing_fundamental_block_is_not_fatal(self):
        self.assertEqual(fundamental_feature_surface(result()), {})
        self.assertEqual(fundamental_feature_surface(SimpleNamespace()), {})


class TheSurfaceIsRegistryConformantTests(unittest.TestCase):
    def test_a_fundamental_surface_passes_the_registry_gate(self):
        live = result(
            fundamental_features=fundamentals(
                metrics={"revenue_growth": 0.24},
                scores={"revenue_growth": 8.5},
            )
        )
        self.assertEqual(fundamental_feature_problems(live), [])

    def test_an_empty_surface_conforms(self):
        self.assertEqual(fundamental_feature_problems(result()), [])

    def test_every_emitted_feature_is_registered(self):
        registry = build_default_registry()
        for name, _ in _FUNDAMENTAL_FEATURE_INPUTS:
            with self.subTest(name=name):
                self.assertIsNotNone(registry.get(name))

    def test_a_contract_carries_what_the_row_builder_needs(self):
        surface = fundamental_feature_surface(
            result(
                fundamental_features=fundamentals(
                    metrics={"revenue_growth": 0.24},
                    scores={"revenue_growth": 8.5},
                )
            )
        )
        contract = surface["revenue_growth"]
        for field in ("name", "value", "as_of", "source_id", "published_time"):
            with self.subTest(field=field):
                self.assertIsNotNone(contract.get(field))


class TheGapWasWiringNotAbsenceTests(unittest.TestCase):
    """40 registered, 16 trained on. The producers existed all along."""

    def test_the_registry_holds_far_more_than_training_used(self):
        registry = build_default_registry()
        self.assertGreaterEqual(len(registry.feature_names()), 40)

    def test_the_non_price_families_are_registered(self):
        registry = build_default_registry()
        for name in (
            "revenue_growth",
            "valuation_quality",
            "margin_quality",
            "free_cash_flow_quality",
            "balance_sheet_quality",
            "news_sentiment_score",
            "sentiment_score",
            "macro_regime_score",
            "regime_probability_proxy",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(registry.get(name))


if __name__ == "__main__":
    unittest.main()
