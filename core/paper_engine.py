"""Paper-trading order engine (Sprint V6) — simulation only, governance intact.

This closes the third of Sprint V ("Labels / Backtest / Paper"). The engine
simulates the decision -> order -> fill loop so later sprints have real trade
evidence to close outcomes against (L1). It executes nothing real: there is
no broker adapter, no credential surface, and no code path that constructs a
live order.

The loop:

    OrchestrationDecision
        |
        v
    submit_order_intent()
        |
        +-- mode == PAPER  -> intent ACCEPTED
        +-- anything else  -> intent REJECTED (governing rule ids recorded)
        |
        v
    simulate_fill()  (next bar open +/- V3 slippage)
        |
        v
    data/paper_orders.jsonl  (append-only)

Binding rules, all test-enforced:

- **Governance is upstream and absolute.** Only `mode == "PAPER"` produces a
  tradable intent. An ANALYSIS_ONLY or NO_TRADE decision produces an
  intent-shaped REJECTED record carrying the governing veto rule ids — never
  a fill. The engine re-derives nothing and cannot overturn a veto; it reads
  the decision it is given.
- **Idempotent order ids.** `order_id` is a deterministic SHA-256 of
  (ticker, as_of, side, kind). Submitting the same intent twice yields one
  order: the second call returns the stored record with `duplicate: True`
  and appends nothing.
- **No live path.** `submit_live_order` raises `NotImplementedError`
  unconditionally. It is not gated on a config flag — enabling live trading
  requires writing code that does not exist, which is the design.
- **Costs come from the V3 table.** Fill prices call
  `core.backtest.costs.execution_cost_record`; the engine never inlines a
  spread or slippage number of its own.
- **Append-only storage.** `data/paper_orders.jsonl`, one JSON line per
  record, `paper_only: True` by construction. Nothing is ever mutated.

Pure and deterministic apart from the append itself: the same decision and
the same bar produce the same order id, the same executed price, and the
same record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.backtest.costs import COST_TABLE_VERSION, execution_cost_record
from core.config import (
    EXECUTION_MODE,
    EXECUTION_MODE_LIVE_APPROVED,
    PAPER_ENGINE_VERSION,
    PAPER_ORDER_NOTIONAL,
    PAPER_ORDER_SCHEMA_VERSION,
    PAPER_TRADABLE_MODE,
)

PAPER_ORDER_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_orders.jsonl"

SIDES = ("buy", "sell")

STATUS_ACCEPTED = "ACCEPTED"
STATUS_REJECTED = "REJECTED"
STATUS_FILLED = "FILLED"


class PaperEngineError(RuntimeError):
    """Raised when the paper engine is asked to do something it must not do."""


def _order_id(ticker: str, as_of: str, side: str, kind: str = "intent") -> str:
    """Deterministic order id — the idempotency key.

    Same (ticker, as_of, side, kind) always hashes to the same id, so a retry
    after a crash cannot create a second order.
    """
    canonical = json.dumps(
        {"ticker": str(ticker).upper(), "as_of": str(as_of), "side": side, "kind": kind},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _read_orders() -> list[dict[str, Any]]:
    """Read the append-only order log. Malformed lines raise (integrity is loud)."""
    if not PAPER_ORDER_LOG_PATH.exists():
        return []
    orders: list[dict[str, Any]] = []
    with PAPER_ORDER_LOG_PATH.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                orders.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise PaperEngineError(
                    f"{PAPER_ORDER_LOG_PATH.name} line {number} is not valid JSON: {exc}"
                ) from exc
    return orders


def _append_order(record: dict[str, Any]) -> dict[str, Any]:
    PAPER_ORDER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_ORDER_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def get_order(order_id: str) -> dict[str, Any] | None:
    """Return the newest record for an order id, or None."""
    matches = [order for order in _read_orders() if order.get("order_id") == order_id]
    return matches[-1] if matches else None


def get_orders(ticker: str | None = None) -> list[dict[str, Any]]:
    """Read the paper order log, optionally filtered by ticker."""
    orders = _read_orders()
    if ticker is None:
        return orders
    target = str(ticker).upper()
    return [order for order in orders if str(order.get("ticker", "")).upper() == target]


def submit_order_intent(decision, side: str = "buy") -> dict[str, Any]:
    """Turn a decision into a paper order intent.

    Only `mode == "PAPER"` yields an ACCEPTED intent. Every other posture
    yields a REJECTED record naming the governing rule ids — the paper log
    must show why nothing happened, not stay silent.

    Idempotent: resubmitting the same intent returns the stored record with
    `duplicate: True` and appends nothing.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    ticker = str(getattr(decision, "ticker", "")).upper()
    as_of = str(getattr(decision, "as_of", ""))
    mode = str(getattr(decision, "mode", ""))
    if not ticker or not as_of:
        raise PaperEngineError("decision must carry both ticker and as_of")

    order_id = _order_id(ticker, as_of, side)
    existing = get_order(order_id)
    if existing is not None:
        return {**existing, "duplicate": True}

    veto_reasons = list(getattr(decision, "veto_reasons", []) or [])
    accepted = mode == PAPER_TRADABLE_MODE
    record = {
        "order_id": order_id,
        "schema_version": PAPER_ORDER_SCHEMA_VERSION,
        "engine_version": PAPER_ENGINE_VERSION,
        "paper_only": True,
        "ticker": ticker,
        "as_of": as_of,
        "intent_time": as_of,
        "side": side,
        "status": STATUS_ACCEPTED if accepted else STATUS_REJECTED,
        "decision_mode": mode,
        "decision_action": str(getattr(decision, "action", "")),
        "score": float(getattr(decision, "score", 0.0) or 0.0),
        "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
        "notional": PAPER_ORDER_NOTIONAL if accepted else 0.0,
        "governing_rule_ids": veto_reasons,
        "replay_hash": str(getattr(decision, "replay_hash", "")),
        "rejection_reason": (
            "" if accepted
            else f"decision mode {mode!r} is not {PAPER_TRADABLE_MODE!r}"
        ),
    }
    return _append_order(record)


def simulate_fill(
    intent: dict[str, Any],
    next_bar_open: float,
    avg_dollar_volume: float | None = None,
    daily_vol: float | None = None,
    cost_table: dict | None = None,
) -> dict[str, Any]:
    """Simulate a fill at the next bar's open, adjusted by the V3 cost model.

    A REJECTED intent can never fill — that is the governance guarantee, so
    it raises rather than returning a degraded record. Fills are idempotent
    on the intent's order id.
    """
    if intent.get("status") == STATUS_REJECTED:
        raise PaperEngineError(
            f"order {intent.get('order_id')} was rejected "
            f"({intent.get('rejection_reason')}) and can never fill"
        )
    if intent.get("status") == STATUS_FILLED:
        return {**intent, "duplicate": True}
    if next_bar_open is None or float(next_bar_open) <= 0:
        raise PaperEngineError(f"next_bar_open must be positive, got {next_bar_open!r}")

    fill_id = _order_id(intent["ticker"], intent["as_of"], intent["side"], kind="fill")
    existing = get_order(fill_id)
    if existing is not None:
        return {**existing, "duplicate": True}

    notional = float(intent.get("notional", PAPER_ORDER_NOTIONAL))
    participation = (
        None if not avg_dollar_volume
        else min(1.0, notional / float(avg_dollar_volume))
    )
    costs = execution_cost_record(
        side=intent["side"],
        open_price=float(next_bar_open),
        participation=participation,
        avg_dollar_volume=avg_dollar_volume,
        daily_vol=daily_vol,
        order_notional=notional,
        table=cost_table,
    )
    record = {
        "order_id": fill_id,
        "intent_order_id": intent["order_id"],
        "schema_version": PAPER_ORDER_SCHEMA_VERSION,
        "engine_version": PAPER_ENGINE_VERSION,
        "paper_only": True,
        "ticker": intent["ticker"],
        "as_of": intent["as_of"],
        "side": intent["side"],
        "status": STATUS_FILLED,
        "executed_price": costs["executed_price"],
        "quantity": round(notional / costs["executed_price"], 6),
        "notional": notional,
        "cost_table_version": COST_TABLE_VERSION,
        "cost_record": costs,
    }
    return _append_order(record)


def submit_live_order(*args: Any, **kwargs: Any):
    """The live branch. Raises unconditionally — there is no live path.

    This is not a feature flag. `EXECUTION_MODE` is permanently
    `LIVE_DISABLED` and `LIVE_APPROVED` has no construction path anywhere in
    the codebase; enabling live execution means writing an adapter that does
    not exist, deliberately.
    """
    raise NotImplementedError(
        f"live execution does not exist in this system (EXECUTION_MODE={EXECUTION_MODE!r}). "
        f"{EXECUTION_MODE_LIVE_APPROVED!r} is a schema value with no construction path; "
        f"the paper engine simulates fills and never places an order."
    )
