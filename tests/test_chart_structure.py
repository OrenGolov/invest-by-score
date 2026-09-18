"""Deterministic chart structure tests (Sprint C4).

Three rules are pinned here.

**A fractal pivot needs bars AFTER it.** A swing high is the maximum of a window
CENTRED on the bar, so it cannot be confirmed until k further bars exist.
Scanning to the final bar would let future bars decide a past label — the same
class of leak C1's still-forming bar was.

**Every label is recomputable.** No pattern names, no screenshot interpretation.
A reader must be able to check the arithmetic by hand, or the label is not
evidence.

**Undefined is a legitimate answer, but it must explain itself.** A broadening
range is genuinely neither trending nor consolidating; an unexplained
`undefined` is indistinguishable from a failure to compute.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    CHART_STRUCTURE_MIN_BARS,
    CHART_SWING_FRACTAL_K,
    STRUCTURE_PHASE_BREAKOUT,
    STRUCTURE_PHASE_CONSOLIDATION,
    STRUCTURE_PHASE_FAILED_BREAKOUT,
    STRUCTURE_PHASE_REVERSAL,
    STRUCTURE_PHASE_TRENDING,
    STRUCTURE_PHASE_UNDEFINED,
    STRUCTURE_PHASES,
    STRUCTURE_SWING_CONTRACTING,
    STRUCTURE_SWING_DOWNTREND,
    STRUCTURE_SWING_EXPANDING,
    STRUCTURE_SWING_LABELS,
    STRUCTURE_SWING_UPTREND,
)
from core.chart_structure import (
    ChartStructureError,
    consolidation_range,
    describe_structure,
    is_consolidating,
    resolve_phase,
    reversal_state,
    structure_problems,
    swing_points,
    swing_structure,
)


def _frame(closes, highs=None, lows=None, start="2026-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs if highs is not None else [c * 1.01 for c in closes],
            "Low": lows if lows is not None else [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(closes),
        },
        index=index,
    )


def _zigzag(n, amplitude=10.0, period=10, drift=0.0):
    """A deterministic saw-tooth, so swings are placed by construction."""
    closes = []
    for i in range(n):
        leg = (i % period) / period
        wave = amplitude * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(100.0 + wave + drift * i)
    return closes


def _swings(highs, lows):
    """Build swing dicts directly, for structure labelling in isolation."""
    return (
        [{"index": i, "time": str(i), "price": p} for i, p in enumerate(highs)],
        [{"index": i, "time": str(i), "price": p} for i, p in enumerate(lows)],
    )


class PivotConfirmationTests(unittest.TestCase):
    """The C4 leak: a centred window needs bars that have not happened yet."""

    def test_no_pivot_is_claimed_in_the_unconfirmable_tail(self):
        frame = _frame(_zigzag(120))
        highs, lows = swing_points(frame)
        last_confirmable = len(frame) - 1 - CHART_SWING_FRACTAL_K
        for pivot in highs + lows:
            with self.subTest(index=pivot["index"]):
                self.assertLessEqual(pivot["index"], last_confirmable)

    def test_no_pivot_is_claimed_before_the_window_opens(self):
        frame = _frame(_zigzag(120))
        highs, lows = swing_points(frame)
        for pivot in highs + lows:
            with self.subTest(index=pivot["index"]):
                self.assertGreaterEqual(pivot["index"], CHART_SWING_FRACTAL_K)

    def test_the_unconfirmed_tail_is_reported_not_hidden(self):
        """A consumer must see that the latest action is not yet structural."""
        structure = describe_structure(_frame(_zigzag(120)))
        self.assertEqual(structure["unconfirmed_tail_bars"], CHART_SWING_FRACTAL_K)

    def test_appending_future_bars_cannot_change_a_confirmed_pivot(self):
        """PIT: a pivot already confirmed is settled, whatever happens next."""
        closes = _zigzag(120)
        early = swing_points(_frame(closes))[0]
        extended = swing_points(_frame(closes + [500.0] * 10))[0]
        settled = [p for p in early if p["index"] <= len(closes) - 1 - 2 * CHART_SWING_FRACTAL_K]
        extended_by_index = {p["index"]: p["price"] for p in extended}
        for pivot in settled:
            with self.subTest(index=pivot["index"]):
                self.assertEqual(extended_by_index.get(pivot["index"]), pivot["price"])

    def test_a_frame_shorter_than_the_window_yields_no_pivots(self):
        highs, lows = swing_points(_frame([100.0] * (CHART_SWING_FRACTAL_K * 2)))
        self.assertEqual(highs, [])
        self.assertEqual(lows, [])


class SwingStructureTests(unittest.TestCase):
    def test_higher_high_and_higher_low_is_an_uptrend(self):
        highs, lows = _swings([100.0, 110.0], [90.0, 95.0])
        self.assertEqual(swing_structure(highs, lows), STRUCTURE_SWING_UPTREND)

    def test_lower_high_and_lower_low_is_a_downtrend(self):
        highs, lows = _swings([110.0, 100.0], [95.0, 90.0])
        self.assertEqual(swing_structure(highs, lows), STRUCTURE_SWING_DOWNTREND)

    def test_higher_high_and_lower_low_is_broadening(self):
        highs, lows = _swings([100.0, 110.0], [95.0, 90.0])
        self.assertEqual(swing_structure(highs, lows), STRUCTURE_SWING_EXPANDING)

    def test_lower_high_and_higher_low_is_narrowing(self):
        highs, lows = _swings([110.0, 100.0], [90.0, 95.0])
        self.assertEqual(swing_structure(highs, lows), STRUCTURE_SWING_CONTRACTING)

    def test_movement_inside_the_noise_band_is_not_structural(self):
        """A one-cent difference must not become a structural break."""
        highs, lows = _swings([100.0, 100.05], [90.0, 90.02])
        self.assertNotIn(
            swing_structure(highs, lows),
            (STRUCTURE_SWING_UPTREND, STRUCTURE_SWING_DOWNTREND),
        )

    def test_too_few_swings_yields_no_label(self):
        highs, lows = _swings([100.0], [90.0])
        self.assertIsNone(swing_structure(highs, lows))

    def test_every_label_is_a_declared_one(self):
        highs, lows = _swings([100.0, 110.0], [90.0, 95.0])
        self.assertIn(swing_structure(highs, lows), STRUCTURE_SWING_LABELS)


class ConsolidationTests(unittest.TestCase):
    def test_a_tight_range_is_consolidating(self):
        self.assertTrue(is_consolidating(_frame([100.0 + (i % 2) * 0.2 for i in range(40)])))

    def test_a_wide_trend_is_not_consolidating(self):
        self.assertFalse(is_consolidating(_frame([100.0 + i for i in range(40)])))

    def test_the_range_is_reported_alongside_the_verdict(self):
        """The number must be checkable, not just the boolean."""
        self.assertIsNotNone(consolidation_range(_frame([100.0] * 40)))

    def test_short_history_yields_no_verdict(self):
        self.assertIsNone(is_consolidating(_frame([100.0, 101.0])))


class ReversalTests(unittest.TestCase):
    def test_a_fall_then_a_rise_is_a_reversal_up(self):
        closes = [100.0 - i for i in range(11)] + [90.0 + i * 1.5 for i in range(11)]
        self.assertEqual(reversal_state(_frame(closes)), "reversal_up")

    def test_a_rise_then_a_fall_is_a_reversal_down(self):
        closes = [100.0 + i for i in range(11)] + [110.0 - i * 1.5 for i in range(11)]
        self.assertEqual(reversal_state(_frame(closes)), "reversal_down")

    def test_a_drift_that_changes_sign_is_not_a_reversal(self):
        """Both legs must be real moves, or every wobble is a reversal."""
        closes = [100.0 + (i % 3) * 0.05 for i in range(30)]
        self.assertEqual(reversal_state(_frame(closes)), "none")

    def test_a_steady_trend_is_not_a_reversal(self):
        self.assertEqual(reversal_state(_frame([100.0 + i for i in range(30)])), "none")

    def test_short_history_yields_no_verdict(self):
        self.assertIsNone(reversal_state(_frame([100.0, 101.0])))


class PhasePrecedenceTests(unittest.TestCase):
    """Precedence is declared, so it can be argued with rather than discovered."""

    def test_a_failed_breakout_outranks_a_breakout(self):
        phase = resolve_phase("failed_breakout_up", "reversal_up", True, STRUCTURE_SWING_UPTREND)
        self.assertEqual(phase, STRUCTURE_PHASE_FAILED_BREAKOUT)

    def test_a_breakout_outranks_a_reversal(self):
        phase = resolve_phase("breakout_up", "reversal_up", True, STRUCTURE_SWING_UPTREND)
        self.assertEqual(phase, STRUCTURE_PHASE_BREAKOUT)

    def test_a_reversal_outranks_consolidation(self):
        phase = resolve_phase("none", "reversal_down", True, STRUCTURE_SWING_UPTREND)
        self.assertEqual(phase, STRUCTURE_PHASE_REVERSAL)

    def test_consolidation_outranks_trending(self):
        phase = resolve_phase("none", "none", True, STRUCTURE_SWING_UPTREND)
        self.assertEqual(phase, STRUCTURE_PHASE_CONSOLIDATION)

    def test_trending_is_what_remains(self):
        phase = resolve_phase("none", "none", False, STRUCTURE_SWING_UPTREND)
        self.assertEqual(phase, STRUCTURE_PHASE_TRENDING)

    def test_a_broadening_range_is_neither_trending_nor_consolidating(self):
        phase = resolve_phase("none", "none", False, STRUCTURE_SWING_EXPANDING)
        self.assertEqual(phase, STRUCTURE_PHASE_UNDEFINED)

    def test_every_resolution_is_a_declared_phase(self):
        for breakout in ("none", "breakout_up", "failed_breakout_down"):
            for reversal in ("none", "reversal_up"):
                for consolidating in (True, False):
                    for swing in STRUCTURE_SWING_LABELS + (None,):
                        with self.subTest(b=breakout, r=reversal, c=consolidating, s=swing):
                            self.assertIn(
                                resolve_phase(breakout, reversal, consolidating, swing),
                                STRUCTURE_PHASES,
                            )


class StructureDescriptionTests(unittest.TestCase):
    def test_a_monotonic_rise_reads_breakout_not_merely_trending(self):
        """A chart making new highs every bar IS breaking out; breakout is the
        sharper statement, so precedence correctly puts it above trending."""
        structure = describe_structure(_frame([100.0 + i * 0.8 for i in range(150)]))
        self.assertEqual(structure["phase"], STRUCTURE_PHASE_BREAKOUT)

    def test_a_perfectly_straight_line_establishes_no_swing_structure(self):
        """A line has no fractal pivots, so there are no swings to compare.

        Reporting `undefined` with a reason is the honest answer; inventing a
        swing structure from a chart that has none would be the pattern-matching
        C4 exists to avoid.
        """
        closes = [100.0 + i * 0.8 for i in range(140)] + [212.0 - i * 1.2 for i in range(12)]
        structure = describe_structure(_frame(closes))
        self.assertIsNone(structure["swing_structure"])
        self.assertEqual(structure["phase"], STRUCTURE_PHASE_UNDEFINED)
        self.assertIn("unestablished", structure["reason"])

    def test_a_zigzagging_uptrend_reads_trending(self):
        """A real trend has swings; trending is what remains after the sharper
        phases are ruled out."""
        closes = _zigzag(160, amplitude=6.0, period=12, drift=0.35)
        structure = describe_structure(_frame(closes))
        self.assertIn(
            structure["phase"],
            (STRUCTURE_PHASE_TRENDING, STRUCTURE_PHASE_BREAKOUT),
        )
        self.assertIsNotNone(structure["swing_structure"])

    def test_a_flat_chart_reads_consolidation(self):
        structure = describe_structure(_frame([100.0 + (i % 3) * 0.2 for i in range(150)]))
        self.assertEqual(structure["phase"], STRUCTURE_PHASE_CONSOLIDATION)

    def test_thin_history_is_undefined_with_a_reason(self):
        """A confident label from too little history is worse than none."""
        structure = describe_structure(_frame([100.0 + i for i in range(30)]))
        self.assertEqual(structure["phase"], STRUCTURE_PHASE_UNDEFINED)
        self.assertIn(str(CHART_STRUCTURE_MIN_BARS), structure["reason"])

    def test_an_undefined_phase_always_explains_itself(self):
        """Otherwise it is indistinguishable from a failure to compute."""
        for closes in (
            [100.0 + i for i in range(30)],                       # too thin
            _zigzag(150, amplitude=30.0, period=15),              # no sharp phase
        ):
            structure = describe_structure(_frame(closes))
            if structure["phase"] == STRUCTURE_PHASE_UNDEFINED:
                with self.subTest(bars=len(closes)):
                    self.assertTrue(structure["reason"])

    def test_the_description_is_deterministic(self):
        frame = _frame(_zigzag(160, drift=0.2))
        self.assertEqual(describe_structure(frame), describe_structure(frame))

    def test_the_description_is_versioned(self):
        structure = describe_structure(_frame(_zigzag(150)))
        self.assertTrue(structure["calculation_version"])
        self.assertTrue(structure["pipeline_version"])

    def test_a_frame_without_close_is_refused(self):
        with self.assertRaises(ChartStructureError):
            describe_structure(pd.DataFrame({"Open": [1.0]}))

    def test_a_missing_frame_is_refused(self):
        with self.assertRaises(ChartStructureError):
            describe_structure(None)

    def test_a_healthy_description_is_contract_clean(self):
        self.assertEqual(structure_problems(describe_structure(_frame(_zigzag(160)))), [])

    def test_c2_features_are_consumed_not_recomputed(self):
        """W5: breakout, gap and volatility regime stay owned by C2."""
        import inspect

        import core.chart_structure as module

        source = inspect.getsource(module)
        for forbidden in ("def breakout_state(", "def gap_pct(", "def volatility_regime_ratio("):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, source)


class ContractValidationTests(unittest.TestCase):
    def test_an_unexplained_undefined_phase_is_a_problem(self):
        structure = describe_structure(_frame(_zigzag(160)))
        structure["phase"] = STRUCTURE_PHASE_UNDEFINED
        structure["reason"] = ""
        self.assertTrue(any("explain" in problem for problem in structure_problems(structure)))

    def test_a_described_structure_must_declare_its_unconfirmed_tail(self):
        structure = describe_structure(_frame(_zigzag(160)))
        structure["unconfirmed_tail_bars"] = 0
        self.assertTrue(any("unconfirmed tail" in problem for problem in structure_problems(structure)))

    def test_consolidation_phase_must_agree_with_the_range(self):
        structure = describe_structure(_frame(_zigzag(160)))
        structure["phase"] = STRUCTURE_PHASE_CONSOLIDATION
        structure["consolidating"] = False
        self.assertTrue(any("otherwise" in problem for problem in structure_problems(structure)))
