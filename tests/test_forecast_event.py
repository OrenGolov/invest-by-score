"""F5 event-conditioned forecast tests.

The behaviour under test is the PIPELINE: that each stage reports its own
status, that the first failure stops the rest, and that the published claim
never exceeds what the retrieved analogs independently support.
"""

from __future__ import annotations

import unittest

from core.config import (
    CONDITIONAL_MIN_SAMPLES_POINT,
    EVENT_FORECAST_EFFECTIVE_PER_TICKER,
    EVENT_FORECAST_HORIZONS,
    EVENT_FORECAST_MIN_DISTINCT_FOR_POINT,
    EVENT_FORECAST_OK,
    EVENT_FORECAST_REFUSED,
    EVENT_FORECAST_RETRIEVAL,
    EVENT_FORECAST_STAGE_CHART,
    EVENT_FORECAST_STAGE_EVENT,
    EVENT_FORECAST_STAGE_FORECAST,
    EVENT_FORECAST_STAGE_MATCHES,
    EVENT_FORECAST_STAGE_REGIME,
    EVENT_FORECAST_STAGE_REPRESENTATION,
    EVENT_FORECAST_STAGES,
    EVENT_MEMORY_RESPONSE_HORIZONS,
    EVENT_MEMORY_SIMILARITY_FIELDS,
    EVENT_STAGE_BLOCKED,
    EVENT_STAGE_DEGRADED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_OK,
    FORECAST_HORIZONS,
)
from core.event_memory import EventMemory
from core.forecast_conditional import (
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_POINT,
)
from core.forecast_event import (
    EventForecastError,
    associated_share,
    build_event_forecast,
    event_forecast_problems,
    regime_agreement,
    render_pipeline,
    represent_event,
    representation_problems,
    same_ticker_share,
)

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


def _memory(index, ticker="AMD", associated=True, price=300.0, regime="bullish"):
    return EventMemory(
        event_id=f"m{index}", ticker=ticker,
        published_time="2025-06-01 00:00:00", event_type="earnings",
        direction="positive",
        chart_state=_snapshot(close=price, atr_14=price / 40, market_regime=regime,
                              rsi=55.0 + (index % 3) * 0.3),
        response={"20d": {"abnormal_return": 0.03 if index % 3 else -0.02,
                          "stock_return": 0.04}},
        attribution={"20d": "event_associated" if associated else "confounded"},
    )


def _run(memories, regime="bullish", chart=None, event=EVENT, **kwargs):
    return build_event_forecast(
        event, "2026-01-05",
        chart_state=_snapshot() if chart is None else chart,
        regime=regime, memories=memories, base_rate=0.55, **kwargs,
    )


MANY = CONDITIONAL_MIN_SAMPLES_POINT + 5


class RepresentationTests(unittest.TestCase):
    def test_a_missing_event_is_refused(self):
        with self.assertRaises(EventForecastError):
            represent_event(None)

    def test_representation_reads_dicts_and_objects_alike(self):
        from types import SimpleNamespace

        as_object = SimpleNamespace(**EVENT)
        self.assertEqual(represent_event(EVENT), represent_event(as_object))

    def test_an_event_without_a_type_cannot_be_matched(self):
        problems = representation_problems(represent_event({**EVENT, "event_type": ""}))
        self.assertTrue(any("type" in p for p in problems))

    def test_an_event_without_a_time_cannot_be_matched(self):
        problems = representation_problems(
            represent_event({**EVENT, "published_time": "", "effective_time": ""})
        )
        self.assertTrue(any("time" in p for p in problems))

    def test_a_complete_event_raises_nothing(self):
        self.assertEqual(representation_problems(represent_event(EVENT)), [])


class PipelineShapeTests(unittest.TestCase):
    def test_every_declared_stage_is_reported_in_order(self):
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(8)])
        self.assertEqual(list(forecast["stages"]), list(EVENT_FORECAST_STAGES))

    def test_the_forecast_is_the_last_stage(self):
        self.assertEqual(EVENT_FORECAST_STAGES[-1], EVENT_FORECAST_STAGE_FORECAST)

    def test_retrieval_precedes_the_forecast(self):
        self.assertLess(
            EVENT_FORECAST_STAGES.index(EVENT_FORECAST_STAGE_MATCHES),
            EVENT_FORECAST_STAGES.index(EVENT_FORECAST_STAGE_FORECAST),
        )

    def test_an_unknown_horizon_is_refused(self):
        with self.assertRaises(EventForecastError):
            build_event_forecast(EVENT, "2026-01-05", horizon="7d")

    def test_horizons_are_the_intersection_of_memory_and_forecast(self):
        for horizon in EVENT_FORECAST_HORIZONS:
            self.assertIn(horizon, EVENT_MEMORY_RESPONSE_HORIZONS)
            self.assertIn(horizon, FORECAST_HORIZONS)

    def test_render_never_leaves_a_stage_blank(self):
        for row in render_pipeline(_run([])):
            if row["status"] != EVENT_STAGE_OK:
                self.assertTrue(row["reason"], row["stage"])


class FirstFailureStopsThePipelineTests(unittest.TestCase):
    def test_a_missing_event_blocks_everything_after_it(self):
        forecast = build_event_forecast(None, "2026-01-05")
        self.assertEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_EVENT)
        for name in EVENT_FORECAST_STAGES[1:]:
            self.assertEqual(forecast["stages"][name]["status"], EVENT_STAGE_BLOCKED)

    def test_a_typeless_event_fails_at_representation_not_retrieval(self):
        forecast = _run(
            [_memory(i, ticker=f"T{i}") for i in range(8)],
            event={**EVENT, "event_type": ""},
        )
        self.assertEqual(
            forecast["failed_stage"], EVENT_FORECAST_STAGE_REPRESENTATION
        )
        self.assertEqual(
            forecast["stages"][EVENT_FORECAST_STAGE_MATCHES]["status"],
            EVENT_STAGE_BLOCKED,
        )

    def test_the_reported_obstacle_is_the_most_upstream_one(self):
        # Everything wrong at once: the reader must learn about the event, not
        # about the empty store downstream of it.
        forecast = build_event_forecast(None, "2026-01-05", memories=[], regime="bogus")
        self.assertEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_EVENT)

    def test_no_chart_state_is_a_retrieval_failure(self):
        forecast = build_event_forecast(
            EVENT, "2026-01-05", chart_state=None, regime="bullish",
            memories=[_memory(i, ticker=f"T{i}") for i in range(8)],
        )
        self.assertEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_MATCHES)


class EmptyStoreTests(unittest.TestCase):
    """A fresh clone has no memories. That is the ordinary state, not an edge."""

    def test_an_empty_store_refuses_at_retrieval(self):
        forecast = _run([])
        self.assertEqual(forecast["status"], EVENT_FORECAST_REFUSED)
        self.assertEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_MATCHES)

    def test_an_empty_store_says_it_is_empty(self):
        # "no analogs" and "nothing has ever been recorded" have different fixes.
        reason = _run([])["stages"][EVENT_FORECAST_STAGE_MATCHES]["reason"]
        self.assertIn("empty", reason)

    def test_an_empty_store_emits_no_value_and_no_interval(self):
        forecast = _run([])
        self.assertNotIn("value", forecast)
        self.assertIsNone(forecast["interval"])

    def test_an_empty_store_grid_is_contract_clean(self):
        self.assertEqual(event_forecast_problems(_run([])), [])


class IndependenceTests(unittest.TestCase):
    """Analogs from one ticker are one situation, resampled."""

    def test_many_analogs_from_one_ticker_cannot_earn_a_point_claim(self):
        forecast = _run([_memory(i, ticker="NVDA") for i in range(MANY)])
        self.assertNotEqual(forecast["claim"], CONDITIONAL_CLAIM_POINT)
        self.assertNotIn("value", forecast)

    def test_the_effective_sample_is_capped_per_ticker(self):
        forecast = _run([_memory(i, ticker="NVDA") for i in range(MANY)])
        stage = forecast["stages"][EVENT_FORECAST_STAGE_FORECAST]
        self.assertEqual(
            stage["effective_samples"], EVENT_FORECAST_EFFECTIVE_PER_TICKER
        )
        self.assertEqual(stage["distinct_tickers"], 1)

    def test_a_capped_refusal_publishes_no_interval(self):
        # Retrieval SUCCEEDED here and an interval genuinely exists, so this is
        # the one refusal path where a leak would be both possible and real.
        forecast = _run([_memory(i, ticker="NVDA") for i in range(MANY)])
        self.assertEqual(forecast["status"], EVENT_FORECAST_REFUSED)
        self.assertIsNone(forecast["interval"])

    def test_diverse_analogs_still_reach_the_point_tier(self):
        # The cap is a floor on honesty, not a blanket refusal.
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(MANY)])
        self.assertEqual(forecast["claim"], CONDITIONAL_CLAIM_POINT)
        self.assertIn("value", forecast)
        self.assertEqual(forecast["status"], EVENT_FORECAST_OK)

    def test_one_ticker_alone_cannot_reach_the_point_floor(self):
        self.assertLessEqual(
            EVENT_FORECAST_EFFECTIVE_PER_TICKER
            * EVENT_FORECAST_MIN_DISTINCT_FOR_POINT,
            CONDITIONAL_MIN_SAMPLES_POINT,
        )
        self.assertLess(
            EVENT_FORECAST_EFFECTIVE_PER_TICKER, CONDITIONAL_MIN_SAMPLES_POINT
        )

    def test_same_ticker_share_is_measured(self):
        analogs = [{"memory": _memory(i, ticker="NVDA")} for i in range(3)]
        analogs += [{"memory": _memory(i + 3, ticker="AMD")} for i in range(1)]
        self.assertAlmostEqual(same_ticker_share(analogs, "NVDA"), 0.75, places=4)
        self.assertEqual(same_ticker_share([], "NVDA"), 0.0)

    def test_a_concentrated_analog_set_is_flagged_at_retrieval(self):
        forecast = _run([_memory(i, ticker="NVDA") for i in range(MANY)])
        stage = forecast["stages"][EVENT_FORECAST_STAGE_MATCHES]
        self.assertEqual(stage["status"], EVENT_STAGE_DEGRADED)
        self.assertIn("same ticker", stage["reason"])


class AttributionTests(unittest.TestCase):
    def test_a_mostly_confounded_set_is_flagged(self):
        forecast = _run(
            [_memory(i, ticker=f"T{i}", associated=(i < 2)) for i in range(10)]
        )
        stage = forecast["stages"][EVENT_FORECAST_STAGE_MATCHES]
        self.assertEqual(stage["status"], EVENT_STAGE_DEGRADED)
        self.assertIn("confounded", stage["reason"])

    def test_associated_share_is_measured(self):
        analogs = [
            {"memory": _memory(i, ticker=f"T{i}", associated=(i < 3))}
            for i in range(6)
        ]
        self.assertAlmostEqual(associated_share(analogs, "20d"), 0.5, places=4)
        self.assertEqual(associated_share([], "20d"), 0.0)


class RegimeTests(unittest.TestCase):
    def test_regime_is_reported_not_filtered(self):
        # market_regime is already an E6 similarity field; filtering again
        # would weight the same evidence twice.
        self.assertIn("market_regime", EVENT_MEMORY_SIMILARITY_FIELDS)
        stage = _run([_memory(i, ticker=f"T{i}") for i in range(8)])["stages"][
            EVENT_FORECAST_STAGE_REGIME
        ]
        self.assertFalse(stage["filtered"])
        self.assertIsNotNone(stage["agreement"])

    def test_an_ungoverned_regime_label_is_refused(self):
        forecast = _run(
            [_memory(i, ticker=f"T{i}") for i in range(8)], regime="euphoric"
        )
        self.assertEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_REGIME)
        self.assertEqual(
            forecast["stages"][EVENT_FORECAST_STAGE_FORECAST]["status"],
            EVENT_STAGE_BLOCKED,
        )

    def test_no_regime_degrades_rather_than_fails(self):
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(8)], regime=None)
        stage = forecast["stages"][EVENT_FORECAST_STAGE_REGIME]
        self.assertEqual(stage["status"], EVENT_STAGE_DEGRADED)
        self.assertNotEqual(forecast["failed_stage"], EVENT_FORECAST_STAGE_REGIME)

    def test_regime_agreement_is_measured(self):
        analogs = [{"memory": _memory(i, ticker=f"T{i}", regime="bullish")}
                   for i in range(3)]
        analogs += [{"memory": _memory(9, ticker="ZZ", regime="bearish")}]
        self.assertAlmostEqual(regime_agreement(analogs, "bullish"), 0.75, places=4)
        self.assertIsNone(regime_agreement(analogs, None))
        self.assertIsNone(regime_agreement([], "bullish"))


class ShapeRuleTests(unittest.TestCase):
    def test_a_value_exists_iff_the_claim_is_point(self):
        for pool in (
            [],
            [_memory(i, ticker=f"T{i}") for i in range(3)],
            [_memory(i, ticker="NVDA") for i in range(MANY)],
            [_memory(i, ticker=f"T{i}") for i in range(MANY)],
        ):
            forecast = _run(pool)
            if forecast["claim"] == CONDITIONAL_CLAIM_POINT:
                self.assertIn("value", forecast)
            else:
                self.assertNotIn("value", forecast)

    def test_a_refusal_never_carries_an_interval(self):
        for pool in (
            [],
            [_memory(i, ticker=f"T{i}") for i in range(3)],
            [_memory(i, ticker="NVDA") for i in range(MANY)],
        ):
            forecast = _run(pool)
            self.assertEqual(forecast["status"], EVENT_FORECAST_REFUSED)
            self.assertIsNone(forecast["interval"])

    def test_every_refusal_names_the_stage_that_failed(self):
        for pool in ([], [_memory(i, ticker="NVDA") for i in range(MANY)]):
            self.assertTrue(_run(pool)["failed_stage"])

    def test_an_ok_forecast_carries_its_interval(self):
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(MANY)])
        self.assertEqual(forecast["status"], EVENT_FORECAST_OK)
        self.assertIsNotNone(forecast["interval"])


class ContractProblemTests(unittest.TestCase):
    def test_a_healthy_pipeline_raises_nothing(self):
        for pool in (
            [],
            [_memory(i, ticker=f"T{i}") for i in range(3)],
            [_memory(i, ticker="NVDA") for i in range(MANY)],
            [_memory(i, ticker=f"T{i}") for i in range(MANY)],
        ):
            self.assertEqual(event_forecast_problems(_run(pool)), [])

    def test_a_value_on_a_refused_forecast_is_reported(self):
        forecast = _run([])
        forecast["value"] = 1.0
        self.assertTrue(
            any("coalesce" in p or "value" in p
                for p in event_forecast_problems(forecast))
        )

    def test_an_interval_on_a_refused_forecast_is_reported(self):
        forecast = _run([])
        forecast["interval"] = {"lower": 0.1, "upper": 0.9}
        self.assertTrue(
            any("another name" in p for p in event_forecast_problems(forecast))
        )

    def test_a_stage_that_ran_after_a_failure_is_reported(self):
        forecast = _run([])
        forecast["stages"][EVENT_FORECAST_STAGE_REGIME]["status"] = EVENT_STAGE_OK
        self.assertTrue(
            any("downstream symptom" in p for p in event_forecast_problems(forecast))
        )

    def test_a_silent_failed_stage_is_reported(self):
        forecast = _run([])
        forecast["stages"][EVENT_FORECAST_STAGE_MATCHES]["reason"] = ""
        self.assertTrue(
            any("does not say why" in p for p in event_forecast_problems(forecast))
        )

    def test_an_ok_forecast_on_zero_analogs_is_reported(self):
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(MANY)])
        forecast["stages"][EVENT_FORECAST_STAGE_MATCHES]["analogs"] = 0
        self.assertTrue(
            any("zero analogs" in p for p in event_forecast_problems(forecast))
        )


class CompositionTests(unittest.TestCase):
    """F5 must not grow a third retrieval or a second sample-size policy."""

    def test_retrieval_is_declared_as_e6(self):
        self.assertEqual(EVENT_FORECAST_RETRIEVAL, "event_memory.find_analogs")

    def test_f5_calls_e6s_find_analogs(self):
        import core.event_memory as e6
        import core.forecast_event as f5

        self.assertIs(f5.find_analogs, e6.find_analogs)

    def test_the_claim_matches_what_f4_would_give_for_the_same_evidence(self):
        from core.forecast_conditional import select_claim, wilson_interval

        for count in (3, 8, 20, MANY):
            forecast = _run([_memory(i, ticker=f"D{i}") for i in range(count)])
            stage = forecast["stages"][EVENT_FORECAST_STAGE_FORECAST]
            if not forecast["samples"]:
                continue
            expected, _ = select_claim(
                forecast["samples"],
                wilson_interval(stage.get("successes", 0), forecast["samples"]),
                0.55,
            )
            self.assertEqual(forecast["claim"], expected, f"at N={count}")

    def test_the_disclaimer_survives(self):
        forecast = _run([_memory(i, ticker=f"T{i}") for i in range(MANY)])
        self.assertIn("not a", forecast["disclaimer"])


if __name__ == "__main__":
    unittest.main()
