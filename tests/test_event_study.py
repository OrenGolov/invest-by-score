"""Event study engine tests (Sprint E4).

Per event: pre-event baseline -> stock reaction -> benchmark reaction ->
sector reaction -> abnormal return -> volatility response -> volume response,
across intraday / 1D / 5D / 20D / 60D.

Three properties decide whether the measurements can be trusted:

- the baseline ENDS BEFORE the event, with a gap, so pre-announcement drift
  cannot be absorbed into "normal" and shrink the abnormal return;
- an unmatured or unavailable window is ABSENT, never 0.0 — a zero would
  enter training as a measured absence of reaction;
- the model is named on every result, because "abnormal return" means
  nothing without saying what normal was assumed to be.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.config import (
    EVENT_STUDY_BASELINE_GAP_SESSIONS,
    EVENT_STUDY_HORIZONS,
    EVENT_STUDY_MIN_BASELINE_SESSIONS,
    EVENT_STUDY_MODEL_MARKET_ADJUSTED,
    EVENT_STUDY_MODEL_MEAN_ADJUSTED,
    EVENT_STUDY_VERSION,
)
from core.event_contract import Event
from core.event_study import (
    EventStudyError,
    observations_from_studies,
    run_event_study,
    study_event,
    study_report,
)


def _frame(bars: int = 400, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    """A synthetic daily frame with a known, controllable shape."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.015, bars)
    closes = 100.0 * np.cumprod(1.0 + returns)
    index = pd.bdate_range("2024-01-01", periods=bars)
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, bars).astype(float),
        },
        index=index,
    )


class EventStudyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _frame()
        self.benchmark = _frame(seed=1)
        self.sector = _frame(seed=2)
        # Well inside the history: enough baseline behind, 60d ahead.
        self.event_time = self.frame.index[200].strftime("%Y-%m-%d %H:%M:%S")

    def _study(self, **kwargs):
        payload = dict(
            ticker="NVDA", event_time=self.event_time, price_frame=self.frame,
            benchmark_frame=self.benchmark, sector_frame=self.sector,
            benchmark="VOO", sector="SOXX",
        )
        payload.update(kwargs)
        return run_event_study(**payload)


class TestFullChain(EventStudyTestCase):
    """pre-event baseline -> ... -> volume response, every stage present."""

    def test_a_study_succeeds_on_ample_history(self) -> None:
        self.assertEqual(self._study().status, "OK")

    def test_every_horizon_is_measured(self) -> None:
        reactions = self._study().reactions
        self.assertEqual(set(reactions), set(EVENT_STUDY_HORIZONS))

    def test_every_stage_of_the_chain_is_reported(self) -> None:
        reaction = self._study().reactions["20d"]
        for stage in (
            "stock_return", "benchmark_return", "sector_return",
            "abnormal_return", "sector_abnormal_return",
            "volatility_ratio", "volume_ratio",
        ):
            with self.subTest(stage=stage):
                self.assertIsNotNone(reaction[stage])

    def test_the_baseline_is_reported(self) -> None:
        baseline = self._study().baseline
        for key in ("sessions", "start_bar", "end_bar", "mean_daily_return",
                    "daily_volatility", "mean_volume"):
            with self.subTest(key=key):
                self.assertIn(key, baseline)

    def test_abnormal_return_is_stock_minus_benchmark(self) -> None:
        reaction = self._study().reactions["5d"]
        self.assertAlmostEqual(
            reaction["abnormal_return"],
            reaction["stock_return"] - reaction["benchmark_return"],
            places=6,
        )

    def test_sector_abnormal_is_stock_minus_sector(self) -> None:
        reaction = self._study().reactions["5d"]
        self.assertAlmostEqual(
            reaction["sector_abnormal_return"],
            reaction["stock_return"] - reaction["sector_return"],
            places=6,
        )

    def test_the_study_version_is_stamped(self) -> None:
        self.assertEqual(self._study().study_version, EVENT_STUDY_VERSION)


class TestBaselineExcludesTheEvent(EventStudyTestCase):
    """The baseline must never contain the reaction it measures."""

    def test_the_baseline_ends_before_the_event(self) -> None:
        study = self._study()
        self.assertLess(study.baseline["end_bar"], study.entry_bar)

    def test_the_gap_is_applied_and_recorded(self) -> None:
        study = self._study()
        self.assertEqual(study.baseline["gap_sessions"], EVENT_STUDY_BASELINE_GAP_SESSIONS)
        entry_position = list(self.frame.index).index(pd.Timestamp(study.entry_bar))
        end_position = list(self.frame.index).index(pd.Timestamp(study.baseline["end_bar"]))
        self.assertGreaterEqual(entry_position - end_position, EVENT_STUDY_BASELINE_GAP_SESSIONS)

    def test_the_baseline_explains_why_the_gap_exists(self) -> None:
        self.assertIn("drift", self._study().baseline["note"])


class TestRefusals(EventStudyTestCase):
    def test_a_thin_baseline_is_refused(self) -> None:
        """There is no honest description of normal behaviour."""
        early = self.frame.index[5].strftime("%Y-%m-%d %H:%M:%S")
        study = self._study(event_time=early)
        self.assertEqual(study.status, "INSUFFICIENT_BASELINE")
        self.assertEqual(study.reactions, {})

    def test_the_baseline_threshold_is_named_in_the_reason(self) -> None:
        early = self.frame.index[5].strftime("%Y-%m-%d %H:%M:%S")
        self.assertIn(
            str(EVENT_STUDY_MIN_BASELINE_SESSIONS), self._study(event_time=early).reason
        )

    def test_an_event_predating_the_history_is_unavailable(self) -> None:
        study = self._study(event_time="1990-01-01")
        self.assertEqual(study.status, "UNAVAILABLE")

    def test_an_empty_frame_is_unavailable(self) -> None:
        study = self._study(price_frame=pd.DataFrame())
        self.assertEqual(study.status, "UNAVAILABLE")

    def test_an_unknown_model_raises(self) -> None:
        with self.assertRaises(EventStudyError):
            self._study(model="telepathy")


class TestUnmaturedIsAbsentNotZero(EventStudyTestCase):
    """A 0.0 would enter training as a measured absence of reaction."""

    def test_a_recent_event_measures_only_what_elapsed(self) -> None:
        recent = self.frame.index[-3].strftime("%Y-%m-%d %H:%M:%S")
        reactions = self._study(event_time=recent).reactions
        self.assertIn("intraday", reactions)
        self.assertNotIn("60d", reactions)

    def test_unmatured_horizons_are_absent_rather_than_zero(self) -> None:
        recent = self.frame.index[-3].strftime("%Y-%m-%d %H:%M:%S")
        study = self._study(event_time=recent)
        for horizon in ("20d", "60d"):
            with self.subTest(horizon=horizon):
                self.assertIsNone(study.abnormal_return_for(horizon))

    def test_an_event_on_the_final_bar_reports_unmatured_or_intraday_only(self) -> None:
        last = self.frame.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        study = self._study(event_time=last)
        self.assertNotIn("1d", study.reactions)


class TestMissingBenchmark(EventStudyTestCase):
    def test_no_benchmark_means_no_abnormal_return(self) -> None:
        """Never a stock return relabelled as abnormal."""
        study = self._study(benchmark_frame=None)
        self.assertIsNotNone(study.reactions["5d"]["stock_return"])
        self.assertIsNone(study.reactions["5d"]["abnormal_return"])

    def test_a_non_overlapping_benchmark_yields_no_abnormal_return(self) -> None:
        """A benchmark whose history starts after the event cannot align."""
        later = _frame(bars=50)
        later.index = pd.bdate_range("2030-01-01", periods=50)
        study = self._study(benchmark_frame=later)
        self.assertIsNone(study.reactions["5d"]["abnormal_return"])

    def test_the_mean_adjusted_model_works_without_a_benchmark(self) -> None:
        study = self._study(
            benchmark_frame=None, model=EVENT_STUDY_MODEL_MEAN_ADJUSTED
        )
        self.assertIsNotNone(study.reactions["5d"]["abnormal_return"])

    def test_the_model_is_named_on_the_result(self) -> None:
        """'Abnormal return' means nothing without saying what normal was."""
        self.assertEqual(self._study().model, EVENT_STUDY_MODEL_MARKET_ADJUSTED)
        self.assertEqual(
            self._study(model=EVENT_STUDY_MODEL_MEAN_ADJUSTED).model,
            EVENT_STUDY_MODEL_MEAN_ADJUSTED,
        )


class TestResponses(EventStudyTestCase):
    def test_a_volatile_window_raises_the_volatility_ratio(self) -> None:
        frame = _frame()
        # Inject a violent stretch right after the event.
        event_position = 200
        multiplier = np.ones(len(frame))
        multiplier[event_position + 1:event_position + 6] = [1.08, 0.93, 1.09, 0.92, 1.07]
        frame["Close"] = frame["Close"] * np.cumprod(multiplier)
        study = self._study(price_frame=frame)
        self.assertGreater(study.reactions["5d"]["volatility_ratio"], 1.5)

    def test_a_volume_spike_raises_the_volume_ratio(self) -> None:
        frame = _frame()
        frame.iloc[201:206, frame.columns.get_loc("Volume")] *= 10
        study = self._study(price_frame=frame)
        self.assertGreater(study.reactions["5d"]["volume_ratio"], 2.0)

    def test_intraday_uses_open_to_close(self) -> None:
        study = self._study()
        entry = pd.Timestamp(study.entry_bar)
        row = self.frame.loc[entry]
        self.assertAlmostEqual(
            study.reactions["intraday"]["stock_return"],
            row["Close"] / row["Open"] - 1.0,
            places=6,
        )


class TestEventIntegration(EventStudyTestCase):
    def _event(self, **overrides) -> Event:
        payload = dict(
            entity="NVDA", published_time=self.event_time, event_type="earnings",
            source="news", actor="Jensen Huang",
            evidence=[{"source_record_id": "r1"}],
        )
        payload.update(overrides)
        return Event(**payload)

    def test_a_canonical_event_can_be_studied(self) -> None:
        study = study_event(self._event(), self.frame, self.benchmark)
        self.assertEqual(study.status, "OK")
        self.assertEqual(study.ticker, "NVDA")

    def test_the_study_anchors_on_publication_not_effect(self) -> None:
        """The market reacts when it is told."""
        backdated = self._event(effective_time="2024-01-05 00:00:00")
        study = study_event(backdated, self.frame, self.benchmark)
        self.assertLessEqual(pd.Timestamp(study.entry_bar), pd.Timestamp(self.event_time))

    def test_the_event_id_travels_with_the_study(self) -> None:
        event = self._event()
        self.assertEqual(study_event(event, self.frame).event_id, event.event_id)

    def test_studies_produce_actor_observations_with_real_returns(self) -> None:
        """The join E3 was waiting for."""
        events = [self._event(evidence=[{"source_record_id": f"r{i}"}],
                              published_time=self.frame.index[150 + i].strftime("%Y-%m-%d %H:%M:%S"))
                  for i in range(5)]
        studies = [study_event(e, self.frame, self.benchmark) for e in events]
        observations = observations_from_studies(events, studies, horizon="20d")
        self.assertEqual(len(observations), 5)
        for observation in observations:
            with self.subTest(time=observation.published_time):
                self.assertIsNotNone(observation.abnormal_return)

    def test_anonymous_events_produce_no_observations(self) -> None:
        event = self._event(actor="")
        studies = [study_event(event, self.frame, self.benchmark)]
        self.assertEqual(observations_from_studies([event], studies), [])

    def test_an_unmeasured_study_produces_no_observation(self) -> None:
        """Never a fabricated reaction."""
        event = self._event(published_time=self.frame.index[5].strftime("%Y-%m-%d %H:%M:%S"))
        studies = [study_event(event, self.frame, self.benchmark)]
        self.assertEqual(observations_from_studies([event], studies), [])


class TestDisclaimerAndReport(EventStudyTestCase):
    def test_every_result_disclaims_causality(self) -> None:
        self.assertIn("association only", self._study().disclaimer)

    def test_the_report_counts_by_status(self) -> None:
        studies = [
            self._study(),
            self._study(event_time=self.frame.index[5].strftime("%Y-%m-%d %H:%M:%S")),
        ]
        report = study_report(studies)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["measured"], 1)
        self.assertEqual(report["by_status"]["INSUFFICIENT_BASELINE"], 1)

    def test_the_report_carries_the_disclaimer(self) -> None:
        self.assertIn("association only", study_report([])["disclaimer"])


class TimezoneAwareEventTimeTests(unittest.TestCase):
    """A real news timestamp is tz-aware; a price index is not.

    FOUND IN PRODUCTION, not by a fixture: the first live news event the
    daily collector produced raised
    `TypeError: Invalid comparison between dtype=datetime64[ms] and Timestamp`
    inside `_entry_position`. Every synthetic fixture and the price-derived
    backfill supply NAIVE timestamps, so nothing exercised this path until
    NEWS_PROVIDER_API_KEY was set and real articles arrived.
    """

    def _frame(self, sessions: int = 260) -> pd.DataFrame:
        index = pd.bdate_range("2025-01-02", periods=sessions)
        closes = np.linspace(100.0, 160.0, sessions)
        return pd.DataFrame(
            {
                "Open": closes, "High": closes * 1.01,
                "Low": closes * 0.99, "Close": closes,
                "Volume": np.full(sessions, 1_000_000.0),
            },
            index=index,
        )

    def test_a_utc_zulu_timestamp_does_not_raise(self):
        frame = self._frame()
        middle = frame.index[len(frame) // 2]
        zulu = middle.strftime("%Y-%m-%dT21:33:00Z")
        result = run_event_study("NVDA", zulu, frame, event_id="tz")
        self.assertIsNotNone(result.status)

    def test_aware_and_naive_forms_select_the_same_bar(self):
        frame = self._frame()
        middle = frame.index[len(frame) // 2]
        aware = run_event_study(
            "NVDA", middle.strftime("%Y-%m-%dT00:00:00Z"), frame, event_id="a"
        )
        naive = run_event_study(
            "NVDA", middle.strftime("%Y-%m-%d %H:%M:%S"), frame, event_id="b"
        )
        self.assertEqual(aware.entry_bar, naive.entry_bar)

    def test_an_offset_timestamp_is_accepted(self):
        frame = self._frame()
        middle = frame.index[len(frame) // 2]
        result = run_event_study(
            "NVDA", middle.strftime("%Y-%m-%dT16:00:00+03:00"), frame, event_id="o"
        )
        self.assertIsNotNone(result.status)


if __name__ == "__main__":
    unittest.main()
