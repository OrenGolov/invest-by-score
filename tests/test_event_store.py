"""Tests for the event-memory store producer and its provenance contract.

The store is what F5 retrieves from. The behaviour under test is that an
INFERRED memory can never be mistaken for an observed one, because MEASURED,
roughly one inferred "earnings" event in three is not an earnings event.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (
    EVENT_MEMORY_INFERENCE_METHODS,
    EVENT_MEMORY_INFERENCE_PRECISION,
    EVENT_MEMORY_MIN_INFERENCE_PRECISION,
    EVENT_MEMORY_PROVENANCES,
    EVENT_MEMORY_REQUIRE_PROVENANCE,
    MEMORY_PROVENANCE_INFERRED,
    MEMORY_PROVENANCE_OBSERVED,
)
from core.event_memory import (
    EventMemory,
    EventMemoryError,
    load_memories,
    memory_problems,
    remember,
)
from scripts.build_event_memory import (
    MIN_OPENING_GAP,
    MIN_VOLUME_RATIO,
    QUARTER_SESSIONS,
    TRAILING_BUFFER_SESSIONS,
    cadence_candidates,
    is_triple_witching,
)

CADENCE = "quarterly_volume_cadence"


def _memory(**overrides) -> EventMemory:
    payload = dict(
        event_id="e1", ticker="NVDA", published_time="2026-01-05 00:00:00",
        event_type="earnings", direction="positive",
        chart_state={"rsi": 55.0, "market_regime": "bullish"},
        response={"20d": {"abnormal_return": 0.03, "stock_return": 0.05}},
        provenance=MEMORY_PROVENANCE_OBSERVED,
    )
    payload.update(overrides)
    return EventMemory(**payload)


def _frame(sessions: int = 760, seed: int = 7, gap: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=sessions)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, sessions))
    volume = rng.normal(1_000_000, 60_000, sessions).clip(200_000)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, sessions))
    for position in range(40, sessions, QUARTER_SESSIONS):
        volume[position] *= 4.0
        open_[position] = close[position - 1] * (1.0 + gap)
    return pd.DataFrame(
        {"Open": open_, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": volume},
        index=index,
    )


class ProvenanceRequiredTests(unittest.TestCase):
    def test_provenance_is_required_at_the_door(self):
        self.assertTrue(EVENT_MEMORY_REQUIRE_PROVENANCE)
        problems = memory_problems(_memory(provenance=""))
        self.assertTrue(any("provenance is required" in p for p in problems))

    def test_the_dataclass_does_not_default_to_observed(self):
        # The single most dangerous edit: it would launder every unlabelled
        # memory into sourced evidence, and the requirement check could never
        # fire because the field would no longer be empty.
        self.assertEqual(
            EventMemory.__dataclass_fields__["provenance"].default, ""
        )
        self.assertEqual(
            EventMemory.__dataclass_fields__["inference_method"].default, ""
        )

    def test_a_healthy_observed_memory_is_accepted(self):
        self.assertEqual(memory_problems(_memory()), [])

    def test_an_unknown_provenance_is_refused(self):
        problems = memory_problems(_memory(provenance="fabricated"))
        self.assertTrue(any("unknown provenance" in p for p in problems))

    def test_the_write_path_refuses_an_unlabelled_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "m.jsonl"
            with self.assertRaises(EventMemoryError):
                remember(_memory(provenance=""), store)
            self.assertEqual(load_memories(store), [])

    def test_the_vocabulary_is_exactly_observed_and_inferred(self):
        self.assertEqual(
            set(EVENT_MEMORY_PROVENANCES),
            {MEMORY_PROVENANCE_OBSERVED, MEMORY_PROVENANCE_INFERRED},
        )


class InferenceMethodTests(unittest.TestCase):
    def test_an_inferred_memory_must_name_its_method(self):
        problems = memory_problems(
            _memory(provenance=MEMORY_PROVENANCE_INFERRED, inference_method="")
        )
        self.assertTrue(any("must name the method" in p for p in problems))

    def test_an_undeclared_method_is_refused(self):
        problems = memory_problems(
            _memory(provenance=MEMORY_PROVENANCE_INFERRED, inference_method="vibes")
        )
        self.assertTrue(any("unknown inference method" in p for p in problems))

    def test_an_observed_memory_may_not_name_a_method(self):
        # It was one or the other; claiming both hides which.
        problems = memory_problems(
            _memory(provenance=MEMORY_PROVENANCE_OBSERVED, inference_method=CADENCE)
        )
        self.assertTrue(any("either observed or inferred" in p for p in problems))

    def test_a_well_formed_inferred_memory_is_accepted(self):
        self.assertEqual(
            memory_problems(
                _memory(provenance=MEMORY_PROVENANCE_INFERRED,
                        inference_method=CADENCE)
            ),
            [],
        )

    def test_every_method_declares_a_measured_precision(self):
        for method in EVENT_MEMORY_INFERENCE_METHODS:
            self.assertIn(method, EVENT_MEMORY_INFERENCE_PRECISION)
            self.assertIn("MEASURED", EVENT_MEMORY_INFERENCE_METHODS[method])

    def test_every_precision_clears_the_floor(self):
        for method, precision in EVENT_MEMORY_INFERENCE_PRECISION.items():
            self.assertGreaterEqual(precision, EVENT_MEMORY_MIN_INFERENCE_PRECISION)
            self.assertLessEqual(precision, 1.0)

    def test_is_inferred_reports_the_provenance(self):
        self.assertFalse(_memory().is_inferred())
        self.assertTrue(
            _memory(provenance=MEMORY_PROVENANCE_INFERRED,
                    inference_method=CADENCE).is_inferred()
        )

    def test_provenance_survives_a_round_trip_to_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "m.jsonl"
            remember(
                _memory(provenance=MEMORY_PROVENANCE_INFERRED,
                        inference_method=CADENCE),
                store,
            )
            row = load_memories(store)[0]
            self.assertEqual(row["provenance"], MEMORY_PROVENANCE_INFERRED)
            self.assertEqual(row["inference_method"], CADENCE)


class TripleWitchingTests(unittest.TestCase):
    """MEASURED: 31.7% of unfiltered picks were options expiry, not earnings."""

    def test_quarterly_third_fridays_are_recognised(self):
        for day in ("2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20"):
            self.assertTrue(is_triple_witching(pd.Timestamp(day).date()), day)

    def test_an_ordinary_friday_is_not(self):
        self.assertFalse(is_triple_witching(pd.Timestamp("2024-09-13").date()))

    def test_a_non_quarterly_third_friday_is_not(self):
        # Flagging every month would discard legitimate earnings dates.
        self.assertFalse(is_triple_witching(pd.Timestamp("2024-08-16").date()))

    def test_a_witching_day_is_excluded_even_at_enormous_volume(self):
        frame = _frame()
        positions = [
            p for p, stamp in enumerate(frame.index)
            if is_triple_witching(pd.Timestamp(stamp).date())
        ]
        self.assertTrue(positions, "fixture contains no witching day")
        target = positions[len(positions) // 2]
        frame.iloc[target, frame.columns.get_loc("Volume")] *= 50.0
        frame.iloc[target, frame.columns.get_loc("Open")] = (
            float(frame["Close"].iloc[target - 1]) * 1.20
        )
        self.assertNotIn(target, cadence_candidates(frame))


class CadenceFilterTests(unittest.TestCase):
    def test_candidates_are_found_on_a_planted_cadence(self):
        self.assertTrue(cadence_candidates(_frame()))

    def test_candidate_spacing_is_roughly_quarterly(self):
        gaps = np.diff(cadence_candidates(_frame()))
        self.assertTrue(len(gaps))
        self.assertGreaterEqual(float(np.median(gaps)), 40.0)
        self.assertLessEqual(float(np.median(gaps)), 90.0)

    def test_bars_that_never_gap_yield_no_candidates(self):
        # A purely mechanical volume event is not an earnings reaction.
        frame = _frame()
        frame["Open"] = frame["Close"].shift().fillna(frame["Close"])
        self.assertEqual(cadence_candidates(frame), [])

    def test_the_gap_requirement_is_still_in_force(self):
        self.assertGreater(MIN_OPENING_GAP, 0.0)

    def test_the_volume_floor_excludes_ordinary_sessions(self):
        self.assertGreaterEqual(MIN_VOLUME_RATIO, 1.5)

    def test_an_empty_or_volumeless_frame_is_handled(self):
        self.assertEqual(cadence_candidates(None), [])
        self.assertEqual(cadence_candidates(pd.DataFrame()), [])
        self.assertEqual(
            cadence_candidates(pd.DataFrame({"Close": [1.0, 2.0]})), []
        )

    def test_the_trailing_buffer_covers_the_longest_horizon(self):
        # The longest recorded response is 60d; a shorter buffer would write
        # a memory whose windows have not elapsed.
        self.assertGreaterEqual(TRAILING_BUFFER_SESSIONS, 60)


class DeduplicationTests(unittest.TestCase):
    def test_re_recording_does_not_inflate_the_store(self):
        # A re-run of the builder must not quietly make every base rate wrong.
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "m.jsonl"
            memory = _memory(
                event_id="dupe", provenance=MEMORY_PROVENANCE_INFERRED,
                inference_method=CADENCE,
            )
            remember(memory, store)
            remember(memory, store)
            remember(memory, store)
            self.assertEqual(len(load_memories(store)), 1)


class BuilderModeTests(unittest.TestCase):
    """Each mode writes exactly one provenance. Otherwise it is decorative."""

    def setUp(self):
        self.source = (
            Path(__file__).resolve().parent.parent
            / "scripts" / "build_event_memory.py"
        ).read_text(encoding="utf-8")
        self.backfill = self.source[
            self.source.index("def backfill("):self.source.index("def forward(")
        ]
        self.forward = self.source[
            self.source.index("def forward("):self.source.index("def _portfolio_tickers(")
        ]

    def test_the_backfill_writes_only_inferred(self):
        self.assertIn("MEMORY_PROVENANCE_INFERRED", self.backfill)
        self.assertNotIn("MEMORY_PROVENANCE_OBSERVED", self.backfill)

    def test_the_forward_builder_writes_only_observed(self):
        self.assertIn("MEMORY_PROVENANCE_OBSERVED", self.forward)
        self.assertNotIn("MEMORY_PROVENANCE_INFERRED", self.forward)

    def test_the_backfill_records_one_event_type(self):
        # The volume cadence supports 'earnings' and nothing else; deriving
        # ten taxonomy buckets from one signal would be fabrication.
        self.assertIn('self.event_type = "earnings"', self.source)

    def test_the_backfill_honours_the_trailing_buffer(self):
        self.assertIn("position >= usable_end", self.source)
        self.assertIn("windows_not_elapsed", self.source)


if __name__ == "__main__":
    unittest.main()
