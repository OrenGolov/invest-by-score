"""F6 forecast-decomposition tests.

The behaviour under test is restraint: the decomposition reports WHAT EVIDENCE
ENTERED, never how much each component contributed, because the components
overlap and do not sum to the forecast.
"""

from __future__ import annotations

import unittest

from core.config import (
    CONDITIONAL_MIN_SAMPLES_POINT,
    DECOMP_EFFECT_NARROWED,
    DECOMP_EFFECT_NO_EFFECT,
    DECOMP_EFFECT_UNMEASURED,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_DISCLAIMER,
    DECOMPOSITION_EFFECTS,
    DECOMPOSITION_OVERLAP_EVIDENCE,
    DECOMPOSITION_WIRED_COMPONENTS,
)
from core.event_memory import EventMemory
from core.forecast_conditional import CONDITIONAL_CLAIM_INSUFFICIENT
from core.forecast_decomposition import (
    DECOMPOSED_OBJECT,
    DecompositionError,
    _component,
    decompose_conditional_cell,
    decompose_event_forecast,
    decomposition_problems,
    marginal_effect,
    render_components,
)
from core.forecast_event import build_event_forecast

MANY = CONDITIONAL_MIN_SAMPLES_POINT + 5

EVENT = {
    "event_id": "probe", "entity": "NVDA", "event_type": "earnings",
    "direction": "positive", "published_time": "2026-01-05 00:00:00",
    "effective_time": "2026-01-05 00:00:00",
}


def _snapshot(**overrides):
    payload = {
        "close": 120.0, "rsi": 55.0, "volatility": 0.22, "volume_ratio_20d": 1.1,
        "atr_14": 2.5, "trend_slope_60d": 0.3, "trend_vs_20d_mean": 0.02,
        "market_regime": "bullish", "change_5d": 0.01, "change_20d": 0.05,
        "change_60d": 0.12, "price_vs_ma_50": 1.03, "price_vs_ma_200": 1.15,
    }
    payload.update(overrides)
    return payload


def _memory(index, ticker="AMD"):
    return EventMemory(
        event_id=f"m{index}", ticker=ticker,
        published_time="2025-06-01 00:00:00", event_type="earnings",
        direction="positive", provenance="observed",
        chart_state=_snapshot(close=300.0, atr_14=7.5, rsi=55.0 + (index % 3) * 0.3),
        response={"20d": {"abnormal_return": 0.03 if index % 3 else -0.02,
                          "stock_return": 0.04}},
        attribution={"20d": "event_associated"},
    )


def _forecast(pool=None, regime="bullish"):
    return build_event_forecast(
        EVENT, "2026-01-05", chart_state=_snapshot(), regime=regime,
        memories=[_memory(i, ticker=f"T{i}") for i in range(MANY)]
        if pool is None else pool,
        base_rate=0.55,
    )


class NonAdditiveTests(unittest.TestCase):
    """The central claim: the parts do not sum to the whole."""

    def test_the_decomposition_is_not_additive(self):
        self.assertFalse(DECOMPOSITION_ADDITIVE)
        self.assertIs(decompose_event_forecast(_forecast())["additive"], False)

    def test_the_overlap_evidence_carries_its_measurement(self):
        # Otherwise a future reader cannot tell a finding from an assertion.
        self.assertIn("MEASURED", DECOMPOSITION_OVERLAP_EVIDENCE)
        self.assertIn(
            "MEASURED",
            decompose_event_forecast(_forecast())["overlap_evidence"],
        )

    def test_the_disclaimer_denies_causation_and_addition(self):
        for phrase in ("not what caused", "do not sum"):
            self.assertIn(phrase, DECOMPOSITION_DISCLAIMER)

    def test_no_effect_is_named_contribution_or_cause(self):
        for forbidden in ("CONTRIBUTION", "CONTRIBUTED", "CAUSED"):
            self.assertNotIn(forbidden, DECOMPOSITION_EFFECTS)

    def test_no_component_reports_a_contribution_field(self):
        decomposition = decompose_event_forecast(_forecast())
        for name, row in decomposition["components"].items():
            for key in row:
                self.assertNotIn("contribution", str(key).lower(), name)
            for key in (row.get("measured") or {}):
                self.assertNotIn("contribution", str(key).lower(), name)

    def test_a_measured_effect_declares_itself_non_additive(self):
        self.assertIs(marginal_effect(26, 45, 30, 60)["additive"], False)


class NotWiredIsNotZeroTests(unittest.TestCase):
    """An unmeasured component must never read as a measured zero."""

    def setUp(self):
        self.decomposition = decompose_event_forecast(_forecast())
        self.unwired = set(DECOMPOSITION_COMPONENTS) - set(
            DECOMPOSITION_WIRED_COMPONENTS
        )

    def test_there_is_at_least_one_unwired_component(self):
        self.assertTrue(self.unwired)

    def test_unwired_components_report_not_wired(self):
        for name in self.unwired:
            self.assertEqual(
                self.decomposition["components"][name]["status"],
                DECOMP_STATUS_NOT_WIRED,
                name,
            )

    def test_unwired_components_carry_no_numbers(self):
        for name in self.unwired:
            row = self.decomposition["components"][name]
            for key in ("rate", "rate_difference", "samples", "measured"):
                self.assertNotIn(key, row, f"{name}.{key}")

    def test_unwired_components_say_they_are_unmeasured_not_zero(self):
        for name in self.unwired:
            self.assertIn(
                "not zero", self.decomposition["components"][name]["reason"], name
            )

    def test_a_blocked_status_discards_smuggled_detail(self):
        # The guard itself: a caller passing detail into a non-PRESENT row
        # must not be able to make an unmeasured component look measured.
        for status in (DECOMP_STATUS_NOT_WIRED, DECOMP_STATUS_ABSENT):
            row = _component(
                "macro", status, reason="probe", effect=DECOMP_EFFECT_NARROWED,
                rate=0.0, samples=0,
            )
            self.assertIsNone(row["effect"], status)
            self.assertNotIn("rate", row, status)
            self.assertNotIn("samples", row, status)

    def test_a_present_status_keeps_its_detail(self):
        row = _component(
            "technical", DECOMP_STATUS_PRESENT, reason="r",
            effect=DECOMP_EFFECT_NARROWED, selected=12,
        )
        self.assertEqual(row["effect"], DECOMP_EFFECT_NARROWED)
        self.assertEqual(row["selected"], 12)


class MarginalEffectTests(unittest.TestCase):
    def test_a_narrowed_slice_larger_than_its_base_is_refused(self):
        with self.assertRaises(DecompositionError):
            marginal_effect(5, 80, 5, 40)

    def test_negative_samples_are_refused(self):
        with self.assertRaises(DecompositionError):
            marginal_effect(1, -1, 1, 10)

    def test_a_thin_slice_publishes_no_rate(self):
        thin = marginal_effect(2, 2, 30, 60)
        self.assertEqual(thin["claim"], CONDITIONAL_CLAIM_INSUFFICIENT)
        self.assertIsNone(thin["rate"])
        self.assertIsNone(thin["interval"])

    def test_a_well_sampled_slice_publishes_a_rate(self):
        healthy = marginal_effect(26, 45, 30, 60)
        self.assertIsNotNone(healthy["rate"])
        self.assertIsNotNone(healthy["interval"])

    def test_the_difference_is_reported_as_a_rate_difference(self):
        # Never as a share or a contribution.
        effect = marginal_effect(30, 50, 25, 100)
        self.assertIn("rate_difference", effect)
        self.assertNotIn("contribution", effect)
        self.assertAlmostEqual(effect["rate_difference"], 0.35, places=6)

    def test_the_excluded_count_is_reported(self):
        self.assertEqual(marginal_effect(5, 20, 30, 60)["excluded"], 40)


class EventDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.decomposition = decompose_event_forecast(_forecast())

    def test_a_non_dict_is_refused(self):
        with self.assertRaises(DecompositionError):
            decompose_event_forecast("not a forecast")

    def test_it_names_the_forecast_as_its_object(self):
        # So it cannot be mistaken for W1's ensemble breakdown of the SCORE.
        self.assertEqual(self.decomposition["decomposed_object"], "forecast")
        self.assertEqual(DECOMPOSED_OBJECT, "forecast")

    def test_every_component_is_reported_in_order(self):
        self.assertEqual(
            list(self.decomposition["components"]), list(DECOMPOSITION_COMPONENTS)
        )

    def test_the_wired_components_are_present(self):
        for name in DECOMPOSITION_WIRED_COMPONENTS:
            self.assertEqual(
                self.decomposition["components"][name]["status"],
                DECOMP_STATUS_PRESENT,
                name,
            )

    def test_a_filter_that_excluded_nothing_reports_no_effect(self):
        # Every fixture memory matches, so the chart filter cut nothing.
        # Claiming NARROWED here would overstate what the filter did.
        row = self.decomposition["components"]["technical"]
        self.assertEqual(row["selected"], row["pool"])
        self.assertEqual(row["effect"], DECOMP_EFFECT_NO_EFFECT)

    def test_a_filter_that_excluded_memories_reports_narrowed(self):
        pool = [_memory(i, ticker=f"T{i}") for i in range(MANY)]
        for index in range(20):
            memory = _memory(900 + index, ticker=f"Z{index}")
            memory.chart_state = _snapshot(
                rsi=12.0, volatility=0.9, change_20d=-0.4, change_60d=-0.5,
                price_vs_ma_50=-0.3, price_vs_ma_200=-0.4,
                market_regime="bearish",
            )
            pool.append(memory)
        row = decompose_event_forecast(_forecast(pool=pool))["components"]["technical"]
        self.assertLess(row["selected"], row["pool"])
        self.assertEqual(row["effect"], DECOMP_EFFECT_NARROWED)

    def test_the_technical_row_states_that_the_filters_overlap(self):
        # The reason a reader must not add the components together.
        self.assertIn("overlap", self.decomposition["components"]["technical"]["reason"])

    def test_the_regime_is_reported_not_filtered(self):
        row = self.decomposition["components"]["regime"]
        self.assertEqual(row["effect"], DECOMP_EFFECT_NO_EFFECT)
        self.assertFalse(row["filtered"])

    def test_the_analog_component_carries_the_value(self):
        row = self.decomposition["components"]["historical_analog"]
        self.assertEqual(row["status"], DECOMP_STATUS_PRESENT)
        self.assertTrue(row["samples"])

    def test_the_decomposition_is_contract_clean(self):
        self.assertEqual(decomposition_problems(self.decomposition), [])

    def test_a_refused_forecast_invents_nothing(self):
        decomposition = decompose_event_forecast(_forecast(pool=[]))
        self.assertEqual(decomposition["present"], [])
        self.assertNotIn("headline_value", decomposition)
        self.assertEqual(decomposition_problems(decomposition), [])

    def test_render_never_leaves_a_component_blank(self):
        for row in render_components(decompose_event_forecast(_forecast(pool=[]))):
            self.assertTrue(row["reason"], row["component"])


class ConditionalDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.cell = {
            "condition_value": "bullish", "samples": 40, "successes": 26,
            "base_rate": 0.5, "claim": "POINT", "horizon": "20d", "value": 0.65,
            "interval": {"lower": 0.5, "upper": 0.78},
        }
        self.decomposition = decompose_conditional_cell(
            self.cell, total_observations=100
        )

    def test_a_non_dict_is_refused(self):
        with self.assertRaises(DecompositionError):
            decompose_conditional_cell(None)

    def test_the_regime_is_the_filter_here(self):
        row = self.decomposition["components"]["regime"]
        self.assertEqual(row["status"], DECOMP_STATUS_PRESENT)
        self.assertTrue(row["filtered"])

    def test_components_f4_does_not_use_are_absent_not_present(self):
        # Padding them would be the false certainty this sprint avoids.
        for name in ("technical", "news_event"):
            self.assertEqual(
                self.decomposition["components"][name]["status"],
                DECOMP_STATUS_ABSENT,
                name,
            )

    def test_it_is_contract_clean(self):
        self.assertEqual(decomposition_problems(self.decomposition), [])

    def test_an_empty_slice_reports_absent(self):
        empty = decompose_conditional_cell(
            {"condition_value": "stress", "samples": 0, "reason": "thin"},
            total_observations=100,
        )
        self.assertEqual(
            empty["components"]["historical_analog"]["status"], DECOMP_STATUS_ABSENT
        )
        self.assertEqual(decomposition_problems(empty), [])


class ContractProblemTests(unittest.TestCase):
    """decomposition_problems must actually catch a malformed decomposition."""

    def _mutated(self, mutate):
        decomposition = decompose_event_forecast(_forecast())
        mutate(decomposition)
        return decomposition_problems(decomposition)

    def test_an_additive_claim_is_reported(self):
        problems = self._mutated(lambda d: d.update({"additive": True}))
        self.assertTrue(any("additive" in p for p in problems))

    def test_a_wrong_object_is_reported(self):
        problems = self._mutated(lambda d: d.update({"decomposed_object": "score"}))
        self.assertTrue(any("FORECAST" in p for p in problems))

    def test_a_missing_component_is_reported(self):
        problems = self._mutated(lambda d: d["components"].pop("macro"))
        self.assertTrue(any("every declared component" in p for p in problems))

    def test_an_effect_on_a_refused_component_is_reported(self):
        problems = self._mutated(
            lambda d: d["components"]["macro"].update({"effect": "NARROWED"})
        )
        self.assertTrue(any("reports an effect" in p for p in problems))

    def test_a_number_on_an_unwired_component_is_reported(self):
        problems = self._mutated(
            lambda d: d["components"]["macro"].update({"rate": 0.0})
        )
        self.assertTrue(any("never measured" in p for p in problems))

    def test_a_healthy_decomposition_raises_nothing(self):
        self.assertEqual(self._mutated(lambda d: None), [])


if __name__ == "__main__":
    unittest.main()
