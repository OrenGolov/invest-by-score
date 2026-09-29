"""B1 tests — the paper engine is reachable from a decision.

MEASURED before B1: `core/paper_engine.py` was 259 tested lines imported by
`tests/test_paper_engine.py` and NOTHING ELSE, while the system published a
governance mode meaning "paper-trading ready". The W-sprint principle is that
every agent is born wired; this one was not.

The modes already agreed — the orchestrator emits `mode == "PAPER"` and the engine
gates on exactly that — so the gap was only the missing call. These tests pin the
call, both postures, and the fact that a broken paper log cannot invalidate a
decision.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import core.paper_engine as paper_engine
from core.agent_contracts import OrchestrationDecision
from core.config import PAPER_ENGINE_WIRED, PAPER_ORDER_NOTIONAL, PAPER_TRADABLE_MODE
from core.orchestrator import _submit_paper_intent
from core.paper_engine import STATUS_ACCEPTED, STATUS_REJECTED, submit_order_intent


def decision(mode: str, *, ticker="NVDA", as_of="2026-09-21 00:00:00"):
    return SimpleNamespace(
        ticker=ticker,
        as_of=as_of,
        mode=mode,
        action=mode,
        score=8.1,
        confidence=0.9,
        veto_reasons=[] if mode == PAPER_TRADABLE_MODE else ["score_below_threshold"],
    )


class TemporaryOrderStore:
    """Redirect the order ledger, so tests never append to the real one."""

    def __enter__(self):
        self._directory = tempfile.TemporaryDirectory()
        self._original = paper_engine.PAPER_ORDER_LOG_PATH
        paper_engine.PAPER_ORDER_LOG_PATH = (
            pathlib.Path(self._directory.name) / "paper_orders.jsonl"
        )
        return paper_engine.PAPER_ORDER_LOG_PATH

    def __exit__(self, *exc):
        paper_engine.PAPER_ORDER_LOG_PATH = self._original
        self._directory.cleanup()
        return False


class TheModesAgreeTests(unittest.TestCase):
    """The reason B1 was a wiring change and not a translation."""

    def test_the_engine_gates_on_the_mode_the_orchestrator_emits(self):
        # score_engine's governance block publishes "PAPER_TRADING_READY" and
        # config carries "PAPER_ONLY", but the DECISION path emits "PAPER" and
        # the engine gates on "PAPER". A mismatch would mean the engine is wired
        # and can never accept anything.
        self.assertEqual(PAPER_TRADABLE_MODE, "PAPER")

    def test_the_engine_is_declared_wired(self):
        self.assertTrue(PAPER_ENGINE_WIRED)

    def test_the_decision_object_carries_a_paper_order_field(self):
        self.assertIn("paper_order", OrchestrationDecision.__dataclass_fields__)

    def test_the_field_defaults_to_none(self):
        built = OrchestrationDecision(
            ticker="X", as_of="t", mode="NO_TRADE", action="NO_TRADE",
            score=0.0, confidence=0.0,
        )
        self.assertIsNone(built.paper_order)


class BothPosturesAreRecordedTests(unittest.TestCase):
    """A paper log that stays silent cannot say why nothing happened."""

    def test_a_paper_posture_is_accepted_with_a_notional(self):
        with TemporaryOrderStore():
            record = submit_order_intent(decision(PAPER_TRADABLE_MODE))
            self.assertEqual(record["status"], STATUS_ACCEPTED)
            self.assertEqual(record["notional"], PAPER_ORDER_NOTIONAL)

    def test_a_refused_posture_is_recorded_not_skipped(self):
        for mode in ("ANALYSIS_ONLY", "NO_TRADE"):
            with self.subTest(mode=mode), TemporaryOrderStore():
                record = submit_order_intent(decision(mode))
                self.assertEqual(record["status"], STATUS_REJECTED)
                self.assertEqual(record["notional"], 0.0)
                self.assertEqual(record["decision_mode"], mode)

    def test_a_refusal_names_the_governing_mode(self):
        with TemporaryOrderStore():
            record = submit_order_intent(decision("NO_TRADE"))
            self.assertIn("NO_TRADE", str(record))

    def test_every_intent_is_paper_only(self):
        for mode in (PAPER_TRADABLE_MODE, "ANALYSIS_ONLY", "NO_TRADE"):
            with self.subTest(mode=mode), TemporaryOrderStore():
                self.assertTrue(submit_order_intent(decision(mode))["paper_only"])

    def test_resubmitting_is_idempotent(self):
        with TemporaryOrderStore():
            first = submit_order_intent(decision(PAPER_TRADABLE_MODE))
            second = submit_order_intent(decision(PAPER_TRADABLE_MODE))
            self.assertEqual(first["order_id"], second["order_id"])
            self.assertTrue(second.get("duplicate"))


class TheWiringIsNonFatalTests(unittest.TestCase):
    """A decision is a research artifact, not a consequence of its paper log."""

    def test_a_failing_engine_yields_an_unrecorded_marker(self):
        # A decision must not fail because a downstream log could not be
        # written — but the failure must be VISIBLE, not silent, or a broken log
        # looks identical to "no order was ever wanted".
        broken = SimpleNamespace(ticker="", as_of="", mode="PAPER")
        result = _submit_paper_intent(broken)
        self.assertEqual(result["status"], "UNRECORDED")
        self.assertIn("could not be recorded", result["reason"])

    def test_a_valid_decision_yields_a_real_intent(self):
        with TemporaryOrderStore():
            result = _submit_paper_intent(decision("ANALYSIS_ONLY"))
            self.assertEqual(result["status"], STATUS_REJECTED)

    def test_the_helper_never_raises(self):
        for bad in (
            SimpleNamespace(),
            SimpleNamespace(ticker="X"),
            SimpleNamespace(ticker="X", as_of="t", mode=None),
        ):
            with self.subTest(decision=bad), TemporaryOrderStore():
                self.assertIsInstance(_submit_paper_intent(bad), dict)


class TheOrchestratorCallsItTests(unittest.TestCase):
    """The specific gap: nothing called the engine."""

    def test_the_orchestrator_submits_a_paper_intent(self):
        import inspect

        from core import orchestrator

        source = inspect.getsource(orchestrator.orchestrate_score)
        self.assertIn("_submit_paper_intent", source)

    def test_the_submission_is_gated_on_the_flag(self):
        import inspect

        from core import orchestrator

        source = inspect.getsource(orchestrator.orchestrate_score)
        self.assertIn("PAPER_ENGINE_WIRED", source)

    def test_the_engine_is_imported_lazily(self):
        # Inside the helper, so importing the orchestrator does not pull the
        # order ledger into every consumer.
        import inspect

        from core import orchestrator

        source = inspect.getsource(orchestrator._submit_paper_intent)
        self.assertIn("from core.paper_engine import", source)


class NoLiveOrderPathExistsTests(unittest.TestCase):
    """Wiring paper must not have opened a live path."""

    def test_submitting_a_live_order_is_refused(self):
        with self.assertRaises(Exception):
            paper_engine.submit_live_order()


if __name__ == "__main__":
    unittest.main()
