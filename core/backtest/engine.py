"""Walk-forward validation engine (Sprint V2).

The harness replays the LIVE scoring path (`core.score_engine.build_score`)
over history through an offline injection seam, trades the resulting signal
under a versioned strategy with the versioned cost table, evaluates against
V1 labels, and emits per-fold + aggregate metrics under a mandatory
manifest.

SIMULATION ONLY — nothing in this module is a production trading path. The
production posture remains fail-closed ANALYSIS_ONLY; the harness exists to
make the signal historically measurable.

Temporal hygiene:

- Folding: anchored train [0, t1] -> embargo (>= max label horizon, i.e. 60
  sessions) -> validation [t1 + embargo + 1, t1 + embargo + fold_sessions]
  -> advance. The embargo covers both features (PIT eligibility, already
  enforced by the agents) and labels (V1 boundary rule).
- Holdout: the final `BACKTEST_HOLDOUT_SESSIONS` sessions are excluded from
  every fold and evaluated once with the frozen configuration.
- Decisions at bar t act at bar t+1 open with costs (never the same bar's
  close — the classic look-ahead).
- Label alignment: every label used or injected is verified against the
  canonical V1 recomputation; a one-bar-early (or otherwise shifted) label
  fails the hash comparison and the whole run is rejected via
  `BacktestLeakageError` before any metric is produced.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd

from core.backtest.costs import (
    COST_TABLE_V2,
    COST_TABLE_VERSION,
    execution_cost_record,
    realized_vol_daily,
)
from core.backtest.manifest import build_manifest, validate_manifest
from core.backtest.metrics import compute_metrics
from core.config import (
    BACKTEST_AVG_DOLLAR_VOLUME_WINDOW,
    BACKTEST_EMBARGO_SESSIONS,
    BACKTEST_ENTER_SCORE,
    BACKTEST_EXIT_SCORE,
    BACKTEST_FOLD_SESSIONS,
    BACKTEST_HOLDOUT_SESSIONS,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_STRATEGY_VERSION,
    BACKTEST_TRADE_NOTIONAL,
    CURRENT_SCORE_VERSION,
    ENSEMBLE_VERSION,
    LABEL_HORIZON_SESSIONS,
    LONG_TERM_SCORE_VERSION,
    MARKET_FEATURE_VERSION,
    OUTCOME_LABEL_VERSION,
)
from core.labels import build_outcome_labels
from core.score_engine import build_score
from fetch_data import TickerFetchError


class BacktestLeakageError(RuntimeError):
    """Raised when injected labels fail canonical hash verification."""


@contextmanager
def offline_replay_seam(history_by_ticker: dict):
    """Inject cached history frames and force provider-gated domains offline.

    The seam is what lets the LIVE scoring path replay offline and
    deterministically: price history for the market-data agent, the regime
    agent, and the label builder is served from the preloaded frames;
    news/sentiment/macro run as their explicit no-key UNAVAILABLE contracts;
    fundamentals take their documented offline fallback path.
    """

    def _provider(ticker, *args, **kwargs):
        key = str(ticker).upper()
        if key not in history_by_ticker:
            raise TickerFetchError(f"{ticker}: no cached frame for offline replay")
        return history_by_ticker[key].copy()

    with patch("agents.market_data_agent.fetch_price_history", _provider), \
            patch("core.regime_agent.fetch_price_history", _provider), \
            patch("core.labels.fetch_price_history", _provider), \
            patch.dict(os.environ, {"NEWS_PROVIDER_API_KEY": "", "FRED_API_KEY": ""}):
        yield


def build_walk_forward_folds(
    session_count: int,
    fold_sessions: int | None = None,
    embargo_sessions: int | None = None,
    holdout_sessions: int | None = None,
) -> dict:
    """Anchored walk-forward folds with a strict embargo and a tail holdout.

    Fold i: train [0, t1_i], embargo sessions (t1_i, t1_i + embargo],
    validation [t1_i + embargo + 1, t1_i + embargo + fold_sessions].
    Validation windows never touch the holdout tail. Raises ValueError when
    the embargo is shorter than the longest label horizon (spec: >= 60).
    """
    fold_sessions = fold_sessions if fold_sessions is not None else BACKTEST_FOLD_SESSIONS
    embargo = embargo_sessions if embargo_sessions is not None else BACKTEST_EMBARGO_SESSIONS
    holdout = holdout_sessions if holdout_sessions is not None else BACKTEST_HOLDOUT_SESSIONS

    max_horizon = max(LABEL_HORIZON_SESSIONS.values())
    if embargo < max_horizon:
        raise ValueError(
            f"embargo ({embargo}) must be >= the max label horizon ({max_horizon})"
        )
    if fold_sessions < 1 or holdout < 1:
        raise ValueError("fold_sessions and holdout_sessions must be positive")
    if session_count < fold_sessions + embargo + fold_sessions + holdout:
        raise ValueError(
            f"history of {session_count} sessions is too short for fold_sessions="
            f"{fold_sessions}, embargo={embargo}, holdout={holdout}"
        )

    holdout_start = session_count - holdout
    folds = []
    fold_id = 0
    while True:
        train_end = (fold_id + 1) * fold_sessions - 1
        validation_start = train_end + embargo + 1
        validation_end = validation_start + fold_sessions - 1
        if validation_end >= holdout_start:
            break
        folds.append({
            "fold_id": fold_id,
            "train": [0, train_end],
            "embargo_sessions": embargo,
            "validation": [validation_start, validation_end],
        })
        fold_id += 1
    if not folds:
        raise ValueError("no walk-forward fold fits the requested geometry")
    return {
        "folds": folds,
        "holdout": [holdout_start, session_count - 1],
        "embargo_sessions": embargo,
        "fold_sessions": fold_sessions,
        "holdout_sessions": holdout,
    }


def _avg_dollar_volume(frame: pd.DataFrame, position: int) -> float | None:
    """Average dollar volume over the window ending at `position`."""
    window = BACKTEST_AVG_DOLLAR_VOLUME_WINDOW
    if position + 1 < window:
        return None
    window_frame = frame.iloc[position + 1 - window:position + 1]
    return float((window_frame["Close"] * window_frame["Volume"]).mean())


def _verify_injected_labels(
    injected: dict,
    canonical: dict,
    as_of: str,
) -> None:
    """Reject any injected label that fails canonical hash verification."""
    injected_for_decision = injected.get(as_of)
    if injected_for_decision is None:
        raise BacktestLeakageError(
            f"leaked_labels_detected: no injected labels for decision {as_of}"
        )
    for horizon in canonical["matured_horizons"]:
        canonical_record = canonical["horizons"][horizon]
        injected_record = injected_for_decision.get(horizon)
        if injected_record is None:
            raise BacktestLeakageError(
                f"leaked_labels_detected: decision {as_of} horizon {horizon} missing "
                f"from injected labels"
            )
        if injected_record.get("record_hash") != canonical_record["record_hash"]:
            raise BacktestLeakageError(
                f"leaked_labels_detected: decision {as_of} horizon {horizon} "
                f"record_hash mismatch — the injected label is not the canonical "
                f"V1 label (e.g. shifted one bar early)"
            )


def _replay_window(
    ticker: str,
    frame: pd.DataFrame,
    start: int,
    end: int,
    initial_capital: float,
    trade_notional: float,
    injected_labels: dict | None,
    cost_table: dict,
) -> dict:
    """Replay decisions over [start, end] (inclusive session positions).

    The caller must hold the offline seam open. Decisions at bar t act at
    bar t+1 open with costs; the final bar's signal cannot execute inside
    the window and is dropped (documented boundary behavior).
    """
    cash = float(initial_capital)
    shares = 0
    position = 0
    equity_curve: list[float] = []
    daily_returns: list[float] = []
    position_flags: list[int] = []
    trades: list[dict] = []
    decisions: list[dict] = []
    executions: list[dict] = []
    open_trade: dict | None = None
    rejected_count = 0
    position_changes = 0
    label_evaluated = 0
    label_hits = 0
    total_cost_notional = 0.0
    total_cost_bps_weighted = 0.0
    prev_equity: float | None = None

    for t in range(start, end + 1):
        as_of = frame.index[t].strftime("%Y-%m-%d %H:%M:%S")
        score_result = build_score(ticker, as_of, persist_audit=False)
        score = float(score_result.score)
        action = str(score_result.action)
        is_rejected = action == "NO_TRADE"
        if is_rejected:
            rejected_count += 1

        if is_rejected or score < BACKTEST_EXIT_SCORE:
            target = 0
        elif score >= BACKTEST_ENTER_SCORE:
            target = 1
        else:
            target = position  # hold zone
        decision_count = len(decisions) + 1
        decisions.append({
            "as_of": as_of,
            "score": score,
            "action": action,
            "target_position": target,
            "rejected": is_rejected,
        })

        if target != position and t + 1 <= end:
            side = "buy" if target == 1 else "sell"
            avg_dollar_volume = _avg_dollar_volume(frame, t)
            participation = (trade_notional / avg_dollar_volume) if avg_dollar_volume else None
            # V3: realized daily volatility is an explicit impact input — the
            # square-root law scales with it; None falls back to the neutral
            # vol factor, and the execution record states which path ran.
            daily_vol = realized_vol_daily(frame, t)
            cost_record = execution_cost_record(
                side=side,
                open_price=float(frame["Open"].iloc[t + 1]),
                participation=participation,
                avg_dollar_volume=avg_dollar_volume,
                daily_vol=daily_vol,
                order_notional=trade_notional,
                table=cost_table,
            )
            price = cost_record["executed_price"]
            executions.append({"decision_bar": as_of, **cost_record})
            total_cost_notional += cost_record["cost_notional"]
            total_cost_bps_weighted += cost_record["total_bps"]
            position_changes += 1
            if side == "buy":
                shares = int(trade_notional / price)
                cash -= shares * price
                open_trade = {
                    "entry_bar": frame.index[t + 1].strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": price,
                    "shares": shares,
                    "entry_costs": {
                        key: cost_record[key]
                        for key in ("bucket", "half_spread_bps", "impact_bps", "vol_factor",
                                    "commission_bps", "total_bps", "clamped", "cost_notional")
                        if key in cost_record
                    },
                }
                position = 1
            else:
                cash += shares * price
                trades.append({
                    **open_trade,
                    "exit_bar": frame.index[t + 1].strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_price": price,
                    "return": round(price / open_trade["entry_price"] - 1.0, 6),
                    "exit_costs": {
                        key: cost_record[key]
                        for key in ("bucket", "half_spread_bps", "impact_bps", "vol_factor",
                                    "commission_bps", "total_bps", "clamped", "cost_notional")
                        if key in cost_record
                    },
                })
                open_trade = None
                position = 0

        equity = cash + shares * float(frame["Close"].iloc[t])
        equity_curve.append(round(equity, 2))
        position_flags.append(1 if shares else 0)
        if prev_equity not in (None, 0.0):
            daily_returns.append(equity / prev_equity - 1.0)
        prev_equity = equity

        # Labels: canonical V1 recomputation (the seam serves the cached frame).
        canonical = build_outcome_labels(ticker, as_of)
        if injected_labels is not None:
            _verify_injected_labels(injected_labels, canonical, as_of)
        record_20d = canonical["horizons"].get("20d")
        if record_20d is not None:
            label_evaluated += 1
            decision_long = 1 if target == 1 else 0
            if (record_20d["label_up"] and decision_long == 1) or (
                not record_20d["label_up"] and decision_long == 0
            ):
                label_hits += 1

    metrics = compute_metrics(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        position_flags=position_flags,
        trade_returns=[trade["return"] for trade in trades],
        decision_count=len(decisions),
        rejected_count=rejected_count,
        position_changes=position_changes,
        sessions=end - start + 1,
    )
    execution_count = len(executions)
    return {
        "sessions": end - start + 1,
        "decisions": decisions,
        "rejected_count": rejected_count,
        "position_changes": position_changes,
        "trades": trades,
        "executions": executions,
        "execution_count": execution_count,
        "total_cost_notional": round(total_cost_notional, 2),
        "avg_execution_cost_bps": round(total_cost_bps_weighted / execution_count, 4) if execution_count else None,
        "cost_drag": round(total_cost_notional / float(initial_capital), 6) if initial_capital else None,
        "daily_returns": daily_returns,
        "position_flags": position_flags,
        "equity_curve": equity_curve,
        "final_equity": equity_curve[-1] if equity_curve else round(float(initial_capital), 2),
        "label_evaluated": label_evaluated,
        "label_hits": label_hits,
        "label_hit_rate": round(label_hits / label_evaluated, 6) if label_evaluated else None,
        "metrics": metrics,
    }


def run_walk_forward_backtest(
    ticker: str,
    frame: pd.DataFrame,
    fold_sessions: int | None = None,
    embargo_sessions: int | None = None,
    holdout_sessions: int | None = None,
    injected_labels: dict | None = None,
    cost_table: dict | None = None,
    initial_capital: float | None = None,
    trade_notional: float | None = None,
) -> dict:
    """Run the full walk-forward validation for one ticker over one frame.

    Steps: build folds (embargo >= max label horizon) -> manifest -> replay
    each validation window over the live scoring path under the offline seam
    -> replay the never-touched holdout once with the frozen configuration
    -> per-fold + aggregate metrics. Injected labels are verified against
    the canonical V1 recomputation; any mismatch aborts the run with
    `BacktestLeakageError` before metrics exist.
    """
    cost_table = cost_table if cost_table is not None else COST_TABLE_V2
    initial_capital = initial_capital if initial_capital is not None else BACKTEST_INITIAL_CAPITAL
    trade_notional = trade_notional if trade_notional is not None else BACKTEST_TRADE_NOTIONAL

    geometry = build_walk_forward_folds(
        len(frame), fold_sessions, embargo_sessions, holdout_sessions
    )
    versions = {
        "market_feature": MARKET_FEATURE_VERSION,
        "ensemble": ENSEMBLE_VERSION,
        "current_score": CURRENT_SCORE_VERSION,
        "long_term_score": LONG_TERM_SCORE_VERSION,
        "outcome_label": OUTCOME_LABEL_VERSION,
        "cost_table": cost_table["cost_table_version"],
        "strategy": BACKTEST_STRATEGY_VERSION,
        "metrics": "backtest-metrics-v1",
    }
    config_snapshot = {
        "ticker": str(ticker).upper(),
        "session_count": len(frame),
        "embargo_sessions": geometry["embargo_sessions"],
        "fold_sessions": geometry["fold_sessions"],
        "holdout_sessions": geometry["holdout_sessions"],
        "enter_score": BACKTEST_ENTER_SCORE,
        "exit_score": BACKTEST_EXIT_SCORE,
        "initial_capital": initial_capital,
        "trade_notional": trade_notional,
        "avg_dollar_volume_window": BACKTEST_AVG_DOLLAR_VOLUME_WINDOW,
        "cost_table": cost_table,
    }
    provider_overrides = {
        "price_history": "cached_frame_injection",
        "news": "forced_unavailable",
        "sentiment": "forced_unavailable",
        "macro": "forced_unavailable",
        "fundamentals": "offline_fallback",
    }
    manifest = build_manifest(ticker, frame, config_snapshot, versions, provider_overrides)
    manifest_issues = validate_manifest(manifest)

    with offline_replay_seam({str(ticker).upper(): frame}):
        fold_results = []
        for fold in geometry["folds"]:
            replay = _replay_window(
                ticker, frame,
                fold["validation"][0], fold["validation"][1],
                initial_capital, trade_notional, injected_labels, cost_table,
            )
            fold_results.append({**fold, "evaluation": "validation_fold", **replay})
        holdout_start, holdout_end = geometry["holdout"]
        holdout_replay = _replay_window(
            ticker, frame, holdout_start, holdout_end,
            initial_capital, trade_notional, injected_labels, cost_table,
        )

    pooled_returns: list[float] = []
    pooled_trades: list[float] = []
    pooled_flags: list[int] = []
    decision_count = 0
    rejected_count = 0
    position_changes = 0
    sessions = 0
    label_evaluated = 0
    label_hits = 0
    total_cost_notional = 0.0
    total_cost_bps_weighted = 0.0
    execution_count = 0
    for fold in fold_results:
        pooled_returns.extend(fold["daily_returns"])
        pooled_trades.extend(trade["return"] for trade in fold["trades"])
        pooled_flags.extend(fold["position_flags"])
        decision_count += len(fold["decisions"])
        rejected_count += fold["rejected_count"]
        position_changes += fold["position_changes"]
        sessions += fold["sessions"]
        label_evaluated += fold["label_evaluated"]
        label_hits += fold["label_hits"]
        total_cost_notional += fold["total_cost_notional"]
        total_cost_bps_weighted += (fold["avg_execution_cost_bps"] or 0.0) * fold["execution_count"]
        execution_count += fold["execution_count"]
    pooled_equity: list[float] = []
    equity = float(initial_capital)
    for value in pooled_returns:
        equity *= (1.0 + value)
        pooled_equity.append(round(equity, 2))
    aggregate_metrics = compute_metrics(
        daily_returns=pooled_returns,
        equity_curve=pooled_equity,
        position_flags=pooled_flags,
        trade_returns=pooled_trades,
        decision_count=decision_count,
        rejected_count=rejected_count,
        position_changes=position_changes,
        sessions=sessions,
    )
    aggregate = {
        "fold_count": len(fold_results),
        "sessions": sessions,
        "decision_count": decision_count,
        "rejected_count": rejected_count,
        "trade_count": len(pooled_trades),
        "execution_count": execution_count,
        "total_cost_notional": round(total_cost_notional, 2),
        "avg_execution_cost_bps": round(total_cost_bps_weighted / execution_count, 4) if execution_count else None,
        "cost_drag": round(total_cost_notional / float(initial_capital), 6) if initial_capital else None,
        "label_evaluated": label_evaluated,
        "label_hits": label_hits,
        "label_hit_rate": round(label_hits / label_evaluated, 6) if label_evaluated else None,
        "final_equity": pooled_equity[-1] if pooled_equity else round(float(initial_capital), 2),
        "metrics": aggregate_metrics,
    }
    return {
        "manifest": manifest,
        "manifest_issues": manifest_issues,
        "label_alignment": "verified",
        "geometry": {
            "embargo_sessions": geometry["embargo_sessions"],
            "fold_sessions": geometry["fold_sessions"],
            "holdout_sessions": geometry["holdout_sessions"],
            "holdout": geometry["holdout"],
        },
        "folds": fold_results,
        "aggregate": aggregate,
        "holdout": {**holdout_replay, "evaluation": "holdout_once"},
    }

