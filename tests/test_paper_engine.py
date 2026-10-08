"""Paper-trading order engine tests (Sprint V6) — simulation only.

The V6 acceptance criteria, pinned:

- duplicate intent submission yields one order (idempotent order ids);
- a vetoed decision produces an intent-shaped rejection, never a fill;
- the live branch raises unconditionally;
- NO_TRADE decisions are logged too — the paper log shows why nothing
  happened;
- fills price through the V3 cost table, never an inline slippage number.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from core.agent_contracts import OrchestrationDecision
from core.backtest.costs import COST_TABLE_V1, COST_TABLE_V2, execution_cost_record
from core.config import (
    EXECUTION_MODE,
    EXECUTION_MODE_LIVE_DISABLED,
    PAPER_ORDER_NOTIONAL,
)
from core.paper_engine import (
    STATUS_ACCEPTED,
    STATUS_FILLED,
    STATUS_REJECTED,
    PaperEngineError,
    get_order,
    get_orders,
    simulate_fill,
    submit_live_order,
    submit_order_intent,
)


def _decision(mode: str = "PAPER", veto_reasons=None, ticker: str = "NVDA") -> OrchestrationDecision:
    """A decision in the given posture — the engine's only input."""
    return OrchestrationDecision(
        ticker=ticker,
        as_of="2026-09-16 00:00:00",
        mode=mode,
        action=mode,
        score=8.2,
        confidence=0.77,
        veto_reasons=list(veto_reasons or []),
        replay_hash="deadbeef",
    )


class PaperEngineTestCase(unittest.TestCase):
    """Each test gets an isolated append-only log."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._log = pathlib.Path(self._tmp.name) / "paper_orders.jsonl"
        patcher = patch("core.paper_engine.PAPER_ORDER_LOG_PATH", self._log)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _lines(self) -> list[dict]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestOrderIntent(PaperEngineTestCase):
    def test_paper_decision_is_accepted(self) -> None:
        intent = submit_order_intent(_decision("PAPER"))
        self.assertEqual(intent["status"], STATUS_ACCEPTED)
        self.assertEqual(intent["notional"], PAPER_ORDER_NOTIONAL)
        self.assertTrue(intent["paper_only"])
        self.assertEqual(intent["governing_rule_ids"], [])

    def test_no_trade_decision_is_logged_as_rejection(self) -> None:
        """The paper log must show why nothing happened."""
        intent = submit_order_intent(_decision("NO_TRADE", ["market_regime_stress"]))
        self.assertEqual(intent["status"], STATUS_REJECTED)
        self.assertEqual(intent["notional"], 0.0)
        self.assertEqual(intent["governing_rule_ids"], ["market_regime_stress"])
        self.assertIn("NO_TRADE", intent["rejection_reason"])
        # Logged, not silently dropped.
        self.assertEqual(len(self._lines()), 1)

    def test_analysis_only_decision_is_rejected(self) -> None:
        intent = submit_order_intent(_decision("ANALYSIS_ONLY"))
        self.assertEqual(intent["status"], STATUS_REJECTED)
        self.assertEqual(intent["notional"], 0.0)

    def test_auditor_veto_rule_ids_are_carried(self) -> None:
        intent = submit_order_intent(_decision("NO_TRADE", ["auditor_veto", "score_below_threshold"]))
        self.assertEqual(intent["governing_rule_ids"], ["auditor_veto", "score_below_threshold"])

    def test_invalid_side_raises(self) -> None:
        with self.assertRaises(ValueError):
            submit_order_intent(_decision(), side="hold")

    def test_decision_without_ticker_raises(self) -> None:
        decision = _decision()
        decision.ticker = ""
        with self.assertRaises(PaperEngineError):
            submit_order_intent(decision)


class TestIdempotency(PaperEngineTestCase):
    def test_duplicate_intent_yields_one_order(self) -> None:
        """V6 acceptance: duplicate intent submission yields one order."""
        first = submit_order_intent(_decision())
        second = submit_order_intent(_decision())
        self.assertEqual(first["order_id"], second["order_id"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(self._lines()), 1)

    def test_order_id_is_deterministic_across_processes(self) -> None:
        """The id is a pure hash of the intent, not a counter or uuid."""
        first = submit_order_intent(_decision())
        self._log.unlink()
        second = submit_order_intent(_decision())
        self.assertEqual(first["order_id"], second["order_id"])

    def test_different_side_is_a_different_order(self) -> None:
        buy = submit_order_intent(_decision(), side="buy")
        sell = submit_order_intent(_decision(), side="sell")
        self.assertNotEqual(buy["order_id"], sell["order_id"])
        self.assertEqual(len(self._lines()), 2)

    def test_different_as_of_is_a_different_order(self) -> None:
        first = submit_order_intent(_decision())
        later = _decision()
        later.as_of = "2026-09-17 00:00:00"
        second = submit_order_intent(later)
        self.assertNotEqual(first["order_id"], second["order_id"])

    def test_duplicate_fill_yields_one_fill(self) -> None:
        intent = submit_order_intent(_decision())
        first = simulate_fill(intent, next_bar_open=100.0)
        second = simulate_fill(intent, next_bar_open=100.0)
        self.assertEqual(first["order_id"], second["order_id"])
        self.assertTrue(second["duplicate"])
        # one intent + one fill
        self.assertEqual(len(self._lines()), 2)


class TestFillSimulation(PaperEngineTestCase):
    def test_rejected_intent_can_never_fill(self) -> None:
        """V6 acceptance: a vetoed decision never produces a fill."""
        intent = submit_order_intent(_decision("NO_TRADE", ["market_regime_stress"]))
        with self.assertRaises(PaperEngineError) as ctx:
            simulate_fill(intent, next_bar_open=100.0)
        self.assertIn("can never fill", str(ctx.exception))
        # No fill record was appended.
        self.assertEqual(len(self._lines()), 1)

    def test_buy_pays_the_spread(self) -> None:
        intent = submit_order_intent(_decision(), side="buy")
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertGreater(fill["executed_price"], 100.0)
        self.assertEqual(fill["status"], STATUS_FILLED)

    def test_sell_receives_less(self) -> None:
        intent = submit_order_intent(_decision(), side="sell")
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertLess(fill["executed_price"], 100.0)

    def test_fill_price_comes_from_the_v3_cost_table(self) -> None:
        """The engine must not inline its own slippage number."""
        intent = submit_order_intent(_decision())
        fill = simulate_fill(
            intent, next_bar_open=100.0, avg_dollar_volume=50_000_000.0, daily_vol=0.02
        )
        expected = execution_cost_record(
            side="buy",
            open_price=100.0,
            participation=min(1.0, PAPER_ORDER_NOTIONAL / 50_000_000.0),
            avg_dollar_volume=50_000_000.0,
            daily_vol=0.02,
            order_notional=PAPER_ORDER_NOTIONAL,
        )
        self.assertEqual(fill["executed_price"], expected["executed_price"])
        self.assertEqual(fill["cost_record"]["total_bps"], expected["total_bps"])

    def test_cost_table_is_pinned_in_the_record(self) -> None:
        intent = submit_order_intent(_decision())
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertEqual(fill["cost_table_version"], COST_TABLE_V2["cost_table_version"])

    def test_historical_cost_table_is_honoured(self) -> None:
        """A run can be replayed under the frozen V1 assumptions."""
        intent = submit_order_intent(_decision())
        fill = simulate_fill(intent, next_bar_open=100.0, cost_table=COST_TABLE_V1)
        self.assertNotIn("vol_factor", fill["cost_record"])

    def test_unknown_liquidity_pays_the_widest_spread(self) -> None:
        """Fail-closed: no avg dollar volume means worst-case participation."""
        intent = submit_order_intent(_decision())
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertEqual(fill["cost_record"]["bucket"], "micro")

    def test_non_positive_open_raises(self) -> None:
        intent = submit_order_intent(_decision())
        with self.assertRaises(PaperEngineError):
            simulate_fill(intent, next_bar_open=0.0)

    def test_quantity_reconciles_with_notional_and_price(self) -> None:
        intent = submit_order_intent(_decision())
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertAlmostEqual(
            fill["quantity"] * fill["executed_price"], fill["notional"], places=2
        )

    def test_fill_links_back_to_its_intent(self) -> None:
        intent = submit_order_intent(_decision())
        fill = simulate_fill(intent, next_bar_open=100.0)
        self.assertEqual(fill["intent_order_id"], intent["order_id"])


class TestNoLivePath(PaperEngineTestCase):
    def test_live_order_raises_unconditionally(self) -> None:
        """V6 acceptance: the live branch raises, it is not a config flag away."""
        with self.assertRaises(NotImplementedError):
            submit_live_order()
        with self.assertRaises(NotImplementedError):
            submit_live_order("NVDA", side="buy", quantity=100)

    def test_execution_mode_is_permanently_disabled(self) -> None:
        self.assertEqual(EXECUTION_MODE, EXECUTION_MODE_LIVE_DISABLED)

    def test_live_approved_has_no_construction_path(self) -> None:
        """LIVE_APPROVED is a schema value only — nothing assigns it."""
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        assignments = []
        for path in sorted(repo_root.glob("core/*.py")) + sorted(repo_root.glob("agents/*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#") or "EXECUTION_MODE_LIVE_APPROVED =" in stripped:
                    continue
                if "EXECUTION_MODE = " in stripped and "LIVE_APPROVED" in stripped:
                    assignments.append(f"{path.name}:{number}")
        self.assertEqual(assignments, [], f"live posture is constructible at {assignments}")


class TestAppendOnlyStore(PaperEngineTestCase):
    def test_records_are_never_mutated(self) -> None:
        intent = submit_order_intent(_decision())
        before = self._log.read_text(encoding="utf-8")
        simulate_fill(intent, next_bar_open=100.0)
        after = self._log.read_text(encoding="utf-8")
        self.assertTrue(after.startswith(before), "existing lines were rewritten")

    def test_every_record_is_paper_only(self) -> None:
        submit_order_intent(_decision("PAPER"))
        submit_order_intent(_decision("NO_TRADE", ticker="TSLA"))
        self.assertTrue(all(record["paper_only"] for record in self._lines()))

    def test_get_orders_filters_by_ticker(self) -> None:
        submit_order_intent(_decision(ticker="NVDA"))
        submit_order_intent(_decision(ticker="TSLA"))
        self.assertEqual(len(get_orders("NVDA")), 1)
        self.assertEqual(len(get_orders()), 2)

    def test_get_order_returns_none_for_unknown_id(self) -> None:
        self.assertIsNone(get_order("nonexistent"))

    def test_malformed_line_raises_loudly(self) -> None:
        """Integrity is loud, never skipped."""
        submit_order_intent(_decision())
        with self._log.open("a", encoding="utf-8") as handle:
            handle.write("{not valid json\n")
        with self.assertRaises(PaperEngineError):
            get_orders()


if __name__ == "__main__":
    unittest.main()
