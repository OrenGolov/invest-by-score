"""Confounder / attribution engine tests (Sprint E5).

The E5 rule, pinned: "Never automatically claim causality — decompose
observed movement into market component + sector component + stock-specific
component + event-associated residual."

Three properties decide whether the decomposition is trustworthy:

- it is ADDITIVE, and the sector component is EXCESS over market. Using the
  raw sector return would count market beta twice (a sector ETF contains it)
  and push the error into the residual — the one number nobody would notice
  was wrong;
- a residual must clear TWO bars, share and magnitude. A residual that is
  90% of a 0.1% move is noise, not a finding;
- no verdict ever claims causation, and missing context yields
  `inconclusive` rather than a flattering `event_associated`.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.attribution import (
    AttributionError,
    attribute,
    attribute_study,
    attribution_report,
    decompose,
    find_overlapping_events,
)
from core.config import (
    ATTRIBUTION_CONFOUNDED,
    ATTRIBUTION_EVENT_ASSOCIATED,
    ATTRIBUTION_INCONCLUSIVE,
    ATTRIBUTION_MIN_RESIDUAL_SHARE,
    ATTRIBUTION_MIN_RESIDUAL_SIGMA,
    ATTRIBUTION_UNEXPLAINED,
    EVENT_ATTRIBUTION_VERSION,
)
from core.event_contract import Event
from core.event_study import study_event


def _frame(bars: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, bars))
    return pd.DataFrame(
        {
            "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, bars).astype(float),
        },
        index=pd.bdate_range("2024-01-01", periods=bars),
    )


class TestAdditiveDecomposition(unittest.TestCase):
    """stock = market + sector_excess + stock_specific, exactly."""

    def test_components_sum_to_the_observed_movement(self) -> None:
        decomposition = decompose(0.050, 0.020, 0.035, "5d")
        self.assertTrue(decomposition.is_additive())
        self.assertAlmostEqual(
            decomposition.market
            + decomposition.sector_excess
            + decomposition.stock_specific,
            decomposition.stock_return,
            places=8,
        )

    def test_the_sector_component_is_excess_over_market(self) -> None:
        """Raw sector would count market beta twice — an ETF contains it."""
        decomposition = decompose(0.050, 0.020, 0.035, "5d")
        self.assertAlmostEqual(decomposition.sector_excess, 0.035 - 0.020, places=8)

    def test_double_counting_would_change_the_residual(self) -> None:
        """Demonstrates the error the excess formulation avoids."""
        decomposition = decompose(0.050, 0.020, 0.035, "5d")
        naive_residual = 0.050 - 0.020 - 0.035
        self.assertNotAlmostEqual(decomposition.stock_specific, naive_residual, places=6)

    def test_additivity_holds_across_many_shapes(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(50):
            stock, market, sector = rng.normal(0, 0.05, 3)
            with self.subTest(stock=round(stock, 4)):
                self.assertTrue(decompose(stock, market, sector, "5d").is_additive())

    def test_no_sector_leaves_a_zero_sector_component(self) -> None:
        decomposition = decompose(0.050, 0.020, None, "5d")
        self.assertEqual(decomposition.sector_excess, 0.0)
        self.assertTrue(decomposition.is_additive())

    def test_no_market_means_no_decomposition(self) -> None:
        """The residual would be the raw return under a different name."""
        decomposition = decompose(0.050, None, None, "5d")
        self.assertFalse(decomposition.components_available)
        self.assertIsNone(decomposition.stock_specific)
        self.assertFalse(decomposition.is_additive())


class TestResidualMeasures(unittest.TestCase):
    def test_share_uses_absolute_magnitudes(self) -> None:
        """A +3% market against a -3% specific move is a large decomposition."""
        decomposition = decompose(0.0, 0.03, 0.03, "5d")
        self.assertGreater(decomposition.residual_share, 0.0)

    def test_sigma_scales_with_the_window(self) -> None:
        one_day = decompose(0.05, 0.01, None, "1d", baseline_volatility=0.015, sessions=1)
        twenty = decompose(0.05, 0.01, None, "20d", baseline_volatility=0.015, sessions=20)
        self.assertGreater(one_day.residual_sigma, twenty.residual_sigma)

    def test_sigma_is_absent_without_a_baseline(self) -> None:
        self.assertIsNone(decompose(0.05, 0.01, None, "5d").residual_sigma)

    def test_a_zero_movement_has_no_share(self) -> None:
        self.assertIsNone(decompose(0.0, 0.0, 0.0, "5d").residual_share)


class TestVerdicts(unittest.TestCase):
    def _decompose(self, stock, market, sector=None, volatility=0.015, sessions=5):
        return decompose(
            stock, market, sector, "5d",
            baseline_volatility=volatility, sessions=sessions,
        )

    def test_a_market_driven_move_is_confounded(self) -> None:
        verdict = attribute(self._decompose(0.021, 0.020, 0.020))
        self.assertEqual(verdict.verdict, ATTRIBUTION_CONFOUNDED)
        self.assertIn("common-factor", verdict.reason)

    def test_a_large_specific_move_is_event_associated(self) -> None:
        verdict = attribute(self._decompose(0.090, 0.010, 0.012))
        self.assertEqual(verdict.verdict, ATTRIBUTION_EVENT_ASSOCIATED)
        self.assertTrue(verdict.is_event_associated())

    def test_a_large_share_of_a_tiny_move_is_noise(self) -> None:
        """The second bar: share alone is not enough."""
        verdict = attribute(self._decompose(0.0012, 0.0001, 0.0002))
        self.assertEqual(verdict.verdict, ATTRIBUTION_UNEXPLAINED)
        self.assertIn("noise", verdict.reason)

    def test_without_a_baseline_a_dominant_residual_stays_unexplained(self) -> None:
        verdict = attribute(self._decompose(0.090, 0.010, 0.012, volatility=None))
        self.assertEqual(verdict.verdict, ATTRIBUTION_UNEXPLAINED)

    def test_missing_market_context_is_inconclusive(self) -> None:
        """Never a flattering event_associated."""
        verdict = attribute(decompose(0.090, None, None, "5d"))
        self.assertEqual(verdict.verdict, ATTRIBUTION_INCONCLUSIVE)
        self.assertFalse(verdict.is_event_associated())

    def test_a_zero_movement_is_inconclusive(self) -> None:
        verdict = attribute(decompose(0.0, 0.0, 0.0, "5d", baseline_volatility=0.015))
        self.assertEqual(verdict.verdict, ATTRIBUTION_INCONCLUSIVE)

    def test_the_share_threshold_is_the_documented_one(self) -> None:
        just_below = attribute(self._decompose(
            0.020, 0.0105, 0.0105, volatility=0.001
        ))
        self.assertLess(
            just_below.decomposition.residual_share, ATTRIBUTION_MIN_RESIDUAL_SHARE
        )
        self.assertEqual(just_below.verdict, ATTRIBUTION_CONFOUNDED)


class TestOverlappingEvents(unittest.TestCase):
    """The most common confounder, and the easiest to forget."""

    def _strong(self):
        return decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015, sessions=5)

    def test_an_overlapping_event_confounds_the_attribution(self) -> None:
        verdict = attribute(self._strong(), overlapping_events=["evt-2"])
        self.assertEqual(verdict.verdict, ATTRIBUTION_CONFOUNDED)
        self.assertIn("no single event can claim", verdict.reason)

    def test_the_overlapping_events_are_named(self) -> None:
        verdict = attribute(self._strong(), overlapping_events=["evt-2", "evt-3"])
        self.assertEqual(verdict.overlapping_events, ["evt-2", "evt-3"])

    def test_no_overlap_allows_association(self) -> None:
        self.assertTrue(attribute(self._strong(), overlapping_events=[]).is_event_associated())

    def test_overlap_detection_finds_events_in_the_window(self) -> None:
        first = Event(
            entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
            source="s", evidence=[{"source_record_id": "r1"}],
        )
        second = Event(
            entity="NVDA", published_time="2026-01-08 00:00:00", event_type="guidance",
            source="s", evidence=[{"source_record_id": "r2"}],
        )
        overlaps = find_overlapping_events(first, [first, second], sessions=20)
        self.assertEqual(overlaps, [second.event_id])

    def test_an_event_outside_the_window_does_not_confound(self) -> None:
        first = Event(
            entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
            source="s", evidence=[{"source_record_id": "r1"}],
        )
        far = Event(
            entity="NVDA", published_time="2026-06-05 00:00:00", event_type="guidance",
            source="s", evidence=[{"source_record_id": "r2"}],
        )
        self.assertEqual(find_overlapping_events(first, [first, far], sessions=5), [])

    def test_another_companys_news_is_not_a_confounder(self) -> None:
        first = Event(
            entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
            source="s", evidence=[{"source_record_id": "r1"}],
        )
        other = Event(
            entity="AMD", published_time="2026-01-06 00:00:00", event_type="earnings",
            source="s", evidence=[{"source_record_id": "r2"}],
        )
        self.assertEqual(find_overlapping_events(first, [first, other], sessions=20), [])

    def test_an_event_does_not_confound_itself(self) -> None:
        event = Event(
            entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
            source="s", evidence=[{"source_record_id": "r1"}],
        )
        self.assertEqual(find_overlapping_events(event, [event], sessions=20), [])


class TestNeverClaimsCausality(unittest.TestCase):
    def test_every_attribution_carries_the_causality_note(self) -> None:
        verdict = attribute(decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015))
        self.assertIn("NOT event-caused", verdict.causality_note)

    def test_the_strongest_verdict_is_still_only_association(self) -> None:
        verdict = attribute(
            decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015, sessions=5)
        )
        self.assertEqual(verdict.verdict, ATTRIBUTION_EVENT_ASSOCIATED)
        self.assertNotIn("caused", verdict.reason)

    def test_the_report_carries_the_note(self) -> None:
        self.assertIn("NOT event-caused", attribution_report([])["causality_note"])

    def test_the_version_is_stamped(self) -> None:
        verdict = attribute(decompose(0.05, 0.01, None, "5d"))
        self.assertEqual(verdict.attribution_version, EVENT_ATTRIBUTION_VERSION)


class TestStudyIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _frame()
        self.benchmark = _frame(seed=1)
        self.event = Event(
            entity="NVDA",
            published_time=self.frame.index[200].strftime("%Y-%m-%d %H:%M:%S"),
            event_type="earnings", source="news", actor="Jensen Huang",
            evidence=[{"source_record_id": "r1"}],
        )

    def test_every_studied_horizon_is_attributed(self) -> None:
        study = study_event(self.event, self.frame, self.benchmark)
        attributions = attribute_study(study)
        self.assertEqual(set(attributions), set(study.reactions))

    def test_attribution_reuses_the_studys_baseline_volatility(self) -> None:
        """The sigma test must use the same definition of normal."""
        study = study_event(self.event, self.frame, self.benchmark)
        attributions = attribute_study(study)
        self.assertIsNotNone(attributions["20d"].decomposition.residual_sigma)

    def test_a_study_without_reactions_raises(self) -> None:
        early = Event(
            entity="NVDA",
            published_time=self.frame.index[5].strftime("%Y-%m-%d %H:%M:%S"),
            event_type="earnings", source="news",
            evidence=[{"source_record_id": "r1"}],
        )
        study = study_event(early, self.frame, self.benchmark)
        with self.assertRaises(AttributionError):
            attribute_study(study)

    def test_every_decomposition_from_a_study_is_additive(self) -> None:
        study = study_event(self.event, self.frame, self.benchmark)
        for horizon, attribution in attribute_study(study).items():
            with self.subTest(horizon=horizon):
                self.assertTrue(attribution.decomposition.is_additive())


class TestReport(unittest.TestCase):
    def test_the_report_counts_by_verdict(self) -> None:
        attributions = [
            attribute(decompose(0.021, 0.020, 0.020, "5d", baseline_volatility=0.015)),
            attribute(decompose(0.090, 0.010, 0.012, "5d", baseline_volatility=0.015, sessions=5)),
        ]
        report = attribution_report(attributions)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["event_associated"], 1)
        self.assertEqual(report["by_verdict"][ATTRIBUTION_CONFOUNDED], 1)

    def test_the_report_states_its_thresholds(self) -> None:
        report = attribution_report([])
        self.assertEqual(report["min_residual_share"], ATTRIBUTION_MIN_RESIDUAL_SHARE)
        self.assertEqual(report["min_residual_sigma"], ATTRIBUTION_MIN_RESIDUAL_SIGMA)

    def test_an_empty_batch_does_not_divide_by_zero(self) -> None:
        self.assertEqual(attribution_report([])["association_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
