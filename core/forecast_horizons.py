"""Forecast horizons (Sprint F2) — how far ahead, and when it can be judged.

F2 fixes the horizon set the engine answers over: 1d, 5d, 20d, 60d, 120d, 252d.
Every one resolves to a V1 label horizon, so a forecast at any of them can be
scored against a realized outcome.

**The horizons ARE the label horizons.** Keeping a separate forecast-horizon
table would let the two drift, and a forecast horizon with no matching label
could never be validated. So `FORECAST_HORIZONS` is derived from
`LABEL_HORIZON_SESSIONS`, and the F2 minimum is pinned against it at import: a
narrower product fails to start rather than shipping quietly.

**Adding 252d moved two things nobody would have guessed.** The label builder's
calendar coverage was 130 days, sized for a 60-session maximum — a 252-session
window spans roughly a year, so long labels would have reported pending forever
because their future bars were never fetched. And the backtest embargo must
cover the longest horizon or a validation fold sees bars that shaped a training
row's outcome; an existing import-time guard caught that immediately. Both are
now DERIVED from the horizon set, so the next horizon change cannot reintroduce
either problem.

**A horizon's maturity is a fact about time, not a status to guess.**
`horizon_readiness` reports which horizons can be scored at a given as_of and
which are still pending, because a 252d forecast made last month is not wrong —
it is unfinished, and those are different.

**Long horizons are flagged, not discouraged.** Beyond
`FORECAST_LONG_HORIZON_SESSIONS` a horizon needs materially more history to
validate, and a thin dataset shows up there first. Marking them lets a consumer
see which part of a joint forecast rests on less evidence.
"""

from __future__ import annotations

import logging

from core.config import (
    FORECAST_HORIZON_CALENDAR_DAYS,
    FORECAST_HORIZON_CONTRACT_VERSION,
    FORECAST_HORIZONS,
    FORECAST_LONG_HORIZON_SESSIONS,
    FORECAST_REQUIRED_HORIZONS,
    LABEL_HORIZON_SESSIONS,
)

LOGGER = logging.getLogger("core.forecast_horizons")

HORIZON_STATUS_SCORABLE = "SCORABLE"
HORIZON_STATUS_PENDING = "PENDING"
HORIZON_STATUS_UNAVAILABLE = "UNAVAILABLE"


class ForecastHorizonError(ValueError):
    """Raised when a horizon request violates the F2 contract."""


def known_horizons() -> tuple[str, ...]:
    """Every horizon the engine answers over, shortest first."""
    return tuple(FORECAST_HORIZONS)


def horizon_sessions(horizon: str) -> int:
    """Trading sessions in a horizon.

    Raises on an unknown name rather than defaulting: a typo silently becoming
    20d would score a forecast against a window nobody asked for.
    """
    sessions = LABEL_HORIZON_SESSIONS.get(horizon)
    if sessions is None:
        raise ForecastHorizonError(
            f"unknown horizon {horizon!r} (known: {list(FORECAST_HORIZONS)})"
        )
    return int(sessions)


def horizon_calendar_days(horizon: str) -> int:
    """Roughly how many calendar days a horizon spans."""
    horizon_sessions(horizon)  # validates
    return int(FORECAST_HORIZON_CALENDAR_DAYS[horizon])


def is_long_horizon(horizon: str) -> bool:
    """Whether this horizon needs materially more history to validate."""
    return horizon_sessions(horizon) > FORECAST_LONG_HORIZON_SESSIONS


def horizon_contract(horizon: str) -> dict:
    """The full contract for one horizon."""
    sessions = horizon_sessions(horizon)
    return {
        "horizon": horizon,
        "sessions": sessions,
        "calendar_days": horizon_calendar_days(horizon),
        "is_long": is_long_horizon(horizon),
        "label_horizon": horizon,
        "contract_version": FORECAST_HORIZON_CONTRACT_VERSION,
    }


def horizon_readiness(labels: dict, horizon: str) -> dict:
    """Whether a horizon can be scored yet, and why not if it cannot.

    A 252d forecast made last month is not wrong — it is unfinished. Conflating
    "not yet knowable" with "no data" would make a young long-horizon forecast
    look like a failure.
    """
    contract = horizon_contract(horizon)
    matured = set((labels or {}).get("matured_horizons") or [])
    pending = set((labels or {}).get("pending_horizons") or [])

    if horizon in matured:
        status, reason = HORIZON_STATUS_SCORABLE, ""
    elif horizon in pending:
        status = HORIZON_STATUS_PENDING
        reason = (
            f"{horizon} needs {contract['sessions']} forward sessions "
            f"(~{contract['calendar_days']} calendar days); the window has not closed"
        )
    else:
        status = HORIZON_STATUS_UNAVAILABLE
        reason = f"the label set reports no outcome for {horizon}"

    return {**contract, "status": status, "reason": reason}


def readiness_report(labels: dict) -> dict:
    """Readiness across every declared horizon.

    Reported as a whole because a joint forecast (F3) is only as scorable as its
    individual horizons, and a caller needs to see which parts are evidence and
    which are still open.
    """
    per_horizon = {name: horizon_readiness(labels, name) for name in FORECAST_HORIZONS}
    scorable = [n for n, r in per_horizon.items() if r["status"] == HORIZON_STATUS_SCORABLE]
    pending = [n for n, r in per_horizon.items() if r["status"] == HORIZON_STATUS_PENDING]
    unavailable = [n for n, r in per_horizon.items() if r["status"] == HORIZON_STATUS_UNAVAILABLE]
    return {
        "horizons": per_horizon,
        # Ordered, not sorted alphabetically: "120d" would sort before "1d".
        "scorable": [n for n in FORECAST_HORIZONS if n in scorable],
        "pending": [n for n in FORECAST_HORIZONS if n in pending],
        "unavailable": [n for n in FORECAST_HORIZONS if n in unavailable],
        "long_horizons": [n for n in FORECAST_HORIZONS if is_long_horizon(n)],
        "contract_version": FORECAST_HORIZON_CONTRACT_VERSION,
    }


def required_history_sessions(horizon: str, warmup_sessions: int = 0) -> int:
    """Sessions of data needed to produce AND score one forecast at a horizon.

    The forward window is the part people forget: a 252d label needs a year of
    bars AFTER the prediction time, so a dataset that merely reaches as_of
    cannot score it at all.
    """
    return int(warmup_sessions) + horizon_sessions(horizon)


def horizon_problems(report: dict) -> list[str]:
    """Validate a readiness report against the F2 contract."""
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a dict"]
    if not report.get("contract_version"):
        problems.append("report carries no contract_version")

    horizons = report.get("horizons") or {}
    for name in FORECAST_REQUIRED_HORIZONS:
        if name not in horizons:
            problems.append(
                f"required horizon {name!r} absent — F2 declares it, so a report "
                f"without it is a silently narrower product"
            )
    for name, entry in horizons.items():
        if entry.get("status") != HORIZON_STATUS_SCORABLE and not entry.get("reason"):
            problems.append(f"{name}: a non-scorable horizon must explain itself")
        if entry.get("status") == HORIZON_STATUS_SCORABLE and entry.get("reason"):
            problems.append(f"{name}: a scorable horizon should carry no reason")

    ordered = [n for n in FORECAST_HORIZONS if n in horizons]
    if list(horizons) != ordered:
        problems.append("horizons are not in shortest-first order")
    return problems
