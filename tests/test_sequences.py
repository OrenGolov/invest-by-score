"""Temporal sequence dataset tests (Sprint C5).

The binding rule: **step i sees bars up to i, and nothing after.**

If step 5 could see bar 40, a sequence model would learn from information that
did not exist at step 5, and nothing in the output would reveal it — the
features would look perfectly ordinary. So truncation is pinned directly (a
step's state must equal the state of a frame truncated at that step) rather
than only through a proxy like bar counts.

Two further rules:

**Per-step and as-of-T0 channels are distinguishable.** Price, volume, C2
features and C4 structure vary across the window. Market/macro/sentiment/event/
fundamental state is attached once at T0 and labelled `as_of_t0`, because
back-projecting today's macro reading across sixty past steps is the
revised-data-in-history failure the master context forbids.

**C5 is not a second door into a training set.** M2 remains the only
`TrainingRow` generator; C5 produces none and takes its labels from the same V1
builder, so the two can never disagree about an outcome.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    SEQUENCE_CONTEXT_SCOPE,
    SEQUENCE_LOOKBACK_STEPS,
    SEQUENCE_MIN_STEP_COVERAGE,
    SEQUENCE_STATUS_INCOMPLETE,
    SEQUENCE_STATUS_OK,
    SEQUENCE_STATUS_UNAVAILABLE,
    SEQUENCE_STEP_CHANNELS,
    SEQUENCE_STEP_WARMUP_BARS,
    SEQUENCE_T0_CHANNELS,
)
from core.sequences import (
    SequenceError,
    build_sequence,
    build_step,
    sequence_hash,
    sequence_problems,
)


def _frame(n, start="2024-01-01", base=100.0):
    """A deterministic wave with drift, so structure genuinely varies."""
    closes = []
    for i in range(n):
        leg = (i % 14) / 14
        wave = 8.0 * (leg if leg < 0.5 else 1.0 - leg) * 2
        closes.append(base + wave + 0.12 * i)
    index = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.012 for c in closes],
            "Low": [c * 0.988 for c in closes],
            "Close": closes,
            "Volume": [1_000_000 + (i % 7) * 10_000 for i in range(n)],
        },
        index=index,
    )


def _enough():
    return _frame(SEQUENCE_LOOKBACK_STEPS + 1 + SEQUENCE_STEP_WARMUP_BARS + 40)


def _ok(status="OK", **payload):
    return {"status": status, **payload}


class StepTruncationTests(unittest.TestCase):
    """The leak C5 exists to prevent."""

    def test_a_step_equals_the_state_of_a_frame_truncated_there(self):
        """The strongest statement of the rule, checked directly.

        Not a proxy: the step's own state must be byte-identical to building
        that state from a frame that physically ends at the step.
        """
        frame = _enough()
        for position in (SEQUENCE_STEP_WARMUP_BARS + 5, len(frame) - 1):
            with self.subTest(position=position):
                from_full = build_step(frame, position)
                from_truncated = build_step(frame.iloc[: position + 1], position)
                self.assertEqual(from_full, from_truncated)

    def test_appending_future_bars_cannot_change_an_earlier_step(self):
        frame = _enough()
        position = len(frame) - 30
        before = build_step(frame, position)
        extended = pd.concat([frame, _frame(20, start="2030-01-01", base=900.0)])
        self.assertEqual(build_step(extended, position), before)

    def test_visibility_advances_by_exactly_one_bar_per_step(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        counts = [step["visible_bars"] for step in sequence["steps"]]
        self.assertEqual(counts, list(range(counts[0], counts[0] + len(counts))))

    def test_step_times_are_strictly_increasing(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        times = [step["step_time"] for step in sequence["steps"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), len(set(times)))

    def test_a_step_reports_its_own_bar_not_the_last_one(self):
        frame = _enough()
        position = len(frame) - 25
        step = build_step(frame, position)
        self.assertAlmostEqual(step["price"]["close"], float(frame["Close"].iloc[position]), places=6)
        self.assertNotAlmostEqual(step["price"]["close"], float(frame["Close"].iloc[-1]), places=6)

    def test_a_position_outside_the_frame_is_refused(self):
        with self.assertRaises(SequenceError):
            build_step(_frame(50), 99)


class PerStepVariationTests(unittest.TestCase):
    """Per-step channels must genuinely vary, or they are not per-step."""

    def setUp(self):
        self.sequence = build_sequence("TEST", "2026-09-15", _enough())

    def test_price_varies_across_the_window(self):
        closes = [step["price"]["close"] for step in self.sequence["steps"]]
        self.assertGreater(len(set(closes)), len(closes) // 2)

    def test_structure_is_recomputed_not_repeated(self):
        """A constant phase for 61 steps would mean it was computed once."""
        phases = {step["structure"]["phase"] for step in self.sequence["steps"]}
        self.assertGreater(len(phases), 1)

    def test_every_declared_step_channel_is_present(self):
        step = self.sequence["steps"][0]
        for channel in SEQUENCE_STEP_CHANNELS:
            with self.subTest(channel=channel):
                self.assertIn(channel, step)


class T0ContextTests(unittest.TestCase):
    """Context is attached once and labelled, never back-projected."""

    def test_context_is_labelled_as_of_t0(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertEqual(sequence["context"]["context_scope"], SEQUENCE_CONTEXT_SCOPE)

    def test_no_context_channel_appears_inside_a_step(self):
        """If macro appeared per-step it would imply sixty vintages we never had."""
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        step = sequence["steps"][0]
        for channel in SEQUENCE_T0_CHANNELS:
            with self.subTest(channel=channel):
                self.assertNotIn(channel, step)

    def test_an_absent_channel_reads_unavailable_not_missing(self):
        """An absent key and a failed provider look identical otherwise."""
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertEqual(sequence["context"]["macro"]["status"], SEQUENCE_STATUS_UNAVAILABLE)

    def test_a_supplied_channel_carries_its_status(self):
        sequence = build_sequence(
            "TEST", "2026-09-15", _enough(),
            macro_snapshot=_ok(regime="risk_on", regime_score=0.8, series_values={"vix": 17.0}),
        )
        self.assertEqual(sequence["context"]["macro"]["status"], SEQUENCE_STATUS_OK)
        self.assertEqual(sequence["context"]["macro"]["value"]["regime"], "risk_on")

    def test_degraded_context_makes_the_sequence_incomplete(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertEqual(sequence["status"], SEQUENCE_STATUS_INCOMPLETE)
        self.assertIn("macro", sequence["reason"])

    def test_a_fully_supplied_context_reads_ok(self):
        sequence = build_sequence(
            "TEST", "2026-09-15", _enough(),
            market_context=_ok(benchmark="sp500", benchmark_symbol="SPY", returns={}),
            macro_snapshot=_ok(regime="risk_on", regime_score=0.8, series_values={}),
            news_snapshot=_ok(event_count=0, direction="neutral"),
            sentiment_snapshot=_ok(sentiment_score=None, derivation="none"),
            fundamentals=_ok(valuation_metrics={}),
        )
        self.assertEqual(sequence["status"], SEQUENCE_STATUS_OK)


class SequenceAssemblyTests(unittest.TestCase):
    def test_a_sequence_carries_lookback_plus_one_steps(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertEqual(sequence["step_count"], SEQUENCE_LOOKBACK_STEPS + 1)

    def test_thin_history_yields_no_steps_with_a_reason(self):
        sequence = build_sequence("TEST", "2026-09-15", _frame(40))
        self.assertEqual(sequence["status"], SEQUENCE_STATUS_INCOMPLETE)
        self.assertEqual(sequence["steps"], [])
        self.assertIn("warm-up", sequence["reason"])

    def test_the_warmup_keeps_early_steps_as_deep_as_late_ones(self):
        """Without it the first step would be computed from far fewer bars."""
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        first = sequence["steps"][0]["visible_bars"]
        self.assertGreaterEqual(first, SEQUENCE_STEP_WARMUP_BARS)

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(SequenceError):
            build_sequence("  ", "2026-09-15", _enough())

    def test_a_frame_without_close_is_refused(self):
        with self.assertRaises(SequenceError):
            build_sequence("TEST", "2026-09-15", pd.DataFrame({"Open": [1.0]}))

    def test_a_missing_frame_is_refused(self):
        with self.assertRaises(SequenceError):
            build_sequence("TEST", "2026-09-15", None)

    def test_a_too_short_lookback_is_refused(self):
        with self.assertRaises(SequenceError):
            build_sequence("TEST", "2026-09-15", _enough(), lookback=1)

    def test_the_sequence_is_versioned(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertTrue(sequence["schema_version"])
        self.assertTrue(sequence["calculation_version"])
        self.assertTrue(sequence["pipeline_version"])


class SequenceHashTests(unittest.TestCase):
    def test_the_same_inputs_produce_the_same_hash(self):
        frame = _enough()
        first = build_sequence("TEST", "2026-09-15", frame)
        second = build_sequence("TEST", "2026-09-15", frame)
        self.assertEqual(first["sequence_hash"], second["sequence_hash"])
        self.assertEqual(first, second)

    def test_a_different_ticker_changes_the_hash(self):
        frame = _enough()
        self.assertNotEqual(
            build_sequence("AAA", "2026-09-15", frame)["sequence_hash"],
            build_sequence("BBB", "2026-09-15", frame)["sequence_hash"],
        )

    def test_different_labels_change_the_hash(self):
        steps = [{"step_time": "t", "visible_bars": 1}]
        self.assertNotEqual(
            sequence_hash(steps, "T", "2026-09-15", {"label_version": "v1", "record_hash": "a"}),
            sequence_hash(steps, "T", "2026-09-15", {"label_version": "v1", "record_hash": "b"}),
        )

    def test_an_incomplete_sequence_carries_no_hash(self):
        """Nothing to identify: there are no steps."""
        self.assertIsNone(build_sequence("TEST", "2026-09-15", _frame(40))["sequence_hash"])


class ContractValidationTests(unittest.TestCase):
    def test_a_healthy_sequence_is_contract_clean(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        self.assertEqual(sequence_problems(sequence), [])

    def test_a_step_that_jumps_ahead_is_caught(self):
        """The check that would catch a truncation bug."""
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        sequence["steps"][10]["visible_bars"] += 25
        self.assertTrue(any("seeing the future" in problem for problem in sequence_problems(sequence)))

    def test_out_of_order_steps_are_caught(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        sequence["steps"][5]["step_time"] = "1999-01-01 00:00:00"
        self.assertTrue(any("strictly after" in problem for problem in sequence_problems(sequence)))

    def test_unlabelled_context_is_caught(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        sequence["context"]["context_scope"] = "per_step"
        self.assertTrue(any("as_of_t0" in problem for problem in sequence_problems(sequence)))

    def test_an_ok_sequence_must_carry_a_hash(self):
        sequence = build_sequence("TEST", "2026-09-15", _enough())
        sequence["status"] = SEQUENCE_STATUS_OK
        sequence["sequence_hash"] = None
        self.assertTrue(any("must carry a hash" in problem for problem in sequence_problems(sequence)))


class M2BoundaryTests(unittest.TestCase):
    """C5 must not become a second door into a training set."""

    @staticmethod
    def _code_without_docstrings() -> str:
        """Module source with docstrings stripped.

        The module DOCUMENTS that it builds no TrainingRow, so a naive grep
        matches its own explanation. Only executable code is evidence here.
        """
        import ast
        import inspect

        import core.sequences as module

        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    def test_c5_produces_no_training_rows(self):
        code = self._code_without_docstrings()
        for forbidden in ("TrainingRow", "build_training_row", "build_training_dataset"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, code)

    def test_c5_computes_no_forward_return_of_its_own(self):
        """Labels are passed in, from the same V1 builder M2 uses."""
        code = self._code_without_docstrings()
        self.assertNotIn("def build_outcome_labels", code)
        self.assertNotIn("forward_return =", code)

    def test_supplied_labels_reach_the_sequence_unchanged(self):
        labels = {"label_version": "v1", "record_hash": "abc", "forward_return": 0.05}
        sequence = build_sequence("TEST", "2026-09-15", _enough(), labels=labels)
        self.assertEqual(sequence["labels"], labels)
