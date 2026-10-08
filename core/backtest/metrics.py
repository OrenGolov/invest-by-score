"""Pure, deterministic performance metrics (Sprint V2).

Every function is a pure function of its inputs — no wall-clock, no
randomness, no I/O — so identical runs produce identical metrics. None
conventions (documented per function) keep results JSON-safe; a metric that
is undefined for the given inputs is None, never a fabricated number.
"""

from __future__ import annotations

import math

TRADING_SESSIONS_PER_YEAR = 252


def max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.0 when flat)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        if value <= 0:
            continue
        peak = max(peak, value)
        drawdown = (peak - value) / peak
        worst = max(worst, drawdown)
    return round(worst, 6)


def cagr(equity_curve: list[float], sessions_per_year: int = TRADING_SESSIONS_PER_YEAR) -> float | None:
    """Compound annual growth rate from the equity curve; None if undefined."""
    if len(equity_curve) < 2:
        return None
    start, end = equity_curve[0], equity_curve[-1]
    if start <= 0 or end <= 0:
        return None
    sessions = len(equity_curve) - 1
    if sessions <= 0:
        return None
    return round((end / start) ** (sessions_per_year / sessions) - 1.0, 6)


def sharpe_ratio(
    daily_returns: list[float],
    annual_risk_free: float = 0.0,
    sessions_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> float | None:
    """Annualized Sharpe; None when dispersion is zero or fewer than 2 points."""
    if len(daily_returns) < 2:
        return None
    daily_rf = annual_risk_free / sessions_per_year
    excess = [value - daily_rf for value in daily_returns]
    mean = sum(excess) / len(excess)
    variance = sum((value - mean) ** 2 for value in excess) / len(excess)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return round(mean / std * math.sqrt(sessions_per_year), 6)


def sortino_ratio(
    daily_returns: list[float],
    annual_risk_free: float = 0.0,
    sessions_per_year: int = TRADING_SESSIONS_PER_YEAR,
) -> float | None:
    """Annualized Sortino; None when downside deviation is zero."""
    if len(daily_returns) < 2:
        return None
    daily_rf = annual_risk_free / sessions_per_year
    excess = [value - daily_rf for value in daily_returns]
    mean = sum(excess) / len(excess)
    downside = [min(value, 0.0) for value in excess]
    downside_variance = sum(value ** 2 for value in downside) / len(downside)
    downside_deviation = math.sqrt(downside_variance)
    if downside_deviation == 0:
        return None
    return round(mean / downside_deviation * math.sqrt(sessions_per_year), 6)


def calmar_ratio(cagr_value: float | None, max_dd: float) -> float | None:
    """CAGR over max drawdown; None when either side is undefined/zero."""
    if cagr_value is None or max_dd <= 0:
        return None
    return round(cagr_value / max_dd, 6)


def win_rate(trade_returns: list[float]) -> float | None:
    """Fraction of round trips with a positive return; None without trades."""
    if not trade_returns:
        return None
    wins = sum(1 for value in trade_returns if value > 0)
    return round(wins / len(trade_returns), 6)


def profit_factor(trade_returns: list[float]) -> float | None:
    """Gross profit / |gross loss|; None when no losses or no trades."""
    if not trade_returns:
        return None
    gross_profit = sum(value for value in trade_returns if value > 0)
    gross_loss = sum(value for value in trade_returns if value < 0)
    if gross_loss == 0:
        return None
    return round(gross_profit / abs(gross_loss), 6)


def exposure(position_flags: list[int]) -> float:
    """Fraction of sessions holding a position (0.0 when empty)."""
    if not position_flags:
        return 0.0
    return round(sum(1 for flag in position_flags if flag) / len(position_flags), 6)


def turnover(position_changes: int, sessions: int) -> float:
    """Executions per session (0.0 when no sessions)."""
    if sessions <= 0:
        return 0.0
    return round(position_changes / sessions, 6)


def compute_metrics(
    daily_returns: list[float],
    equity_curve: list[float],
    position_flags: list[int],
    trade_returns: list[float],
    decision_count: int,
    rejected_count: int,
    position_changes: int,
    sessions: int,
    annual_risk_free: float = 0.0,
) -> dict:
    """Assemble the full metric block for a window (per fold or aggregate)."""
    cagr_value = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    return {
        "metrics_version": "backtest-metrics-v1",
        "cagr": cagr_value,
        "sharpe": sharpe_ratio(daily_returns, annual_risk_free),
        "sortino": sortino_ratio(daily_returns, annual_risk_free),
        "calmar": calmar_ratio(cagr_value, max_dd),
        "max_drawdown": max_dd,
        "win_rate": win_rate(trade_returns),
        "profit_factor": profit_factor(trade_returns),
        "exposure": exposure(position_flags),
        "turnover": turnover(position_changes, sessions),
        "rejection_rate": round(rejected_count / decision_count, 6) if decision_count else None,
        "decision_count": decision_count,
        "rejected_count": rejected_count,
        "trade_count": len(trade_returns),
        "sessions": sessions,
        "annual_risk_free": annual_risk_free,
    }
