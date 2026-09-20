"""Event study engine (Sprint E4).

For every event, measured across intraday / 1D / 5D / 20D / 60D:

    pre-event baseline
        -> stock reaction
        -> benchmark reaction
        -> sector reaction
        -> abnormal return
        -> volatility response
        -> volume response

This is the module that finally supplies the abnormal returns E3 has been
refusing to invent, and that E5 will decompose and E6 will remember.

Design decisions worth stating:

- **The baseline ends BEFORE the event, with a gap.** Information leaks into
  prices ahead of an announcement. Including the two sessions before
  publication in "normal behaviour" would fold part of the reaction into the
  baseline and shrink the measured abnormal return toward zero — the study
  would quietly understate exactly what it exists to measure.
- **The event anchors on PUBLICATION, not effective time.** The market
  reacts when it is told, which is what E1 kept these timestamps distinct
  for.
- **Abnormal return is market-adjusted (beta = 1), and says so.** A full
  market model estimating beta from the baseline is the natural v2. Calling
  this "the abnormal return" without naming the model would let a reader
  assume a sophistication that is not there, so `model` travels with every
  result.
- **Association, never causation.** Every result carries the disclaimer. An
  abnormal return following an event is not evidence the event caused it
  (master context §37), and E5 exists precisely because this module cannot
  make that claim.
- **A thin baseline is a refusal.** Below `EVENT_STUDY_MIN_BASELINE_SESSIONS`
  there is no honest description of normal behaviour, so the study returns
  an explicit `insufficient_baseline` status rather than a number.
- **An unmatured horizon is absent, not zero.** A 60D window that has not
  elapsed yields no 60D result; reporting 0.0 would enter training as a real
  measurement of no reaction.

Pure and deterministic: no wall-clock, no network, no randomness. All
timestamps come from the caller and all prices from supplied frames.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from core.timeframes import as_naive_timestamp

from core.config import (
    EVENT_STUDY_BASELINE_GAP_SESSIONS,
    EVENT_STUDY_BASELINE_SESSIONS,
    EVENT_STUDY_HORIZONS,
    EVENT_STUDY_MIN_BASELINE_SESSIONS,
    EVENT_STUDY_MODELS,
    EVENT_STUDY_MODEL_MARKET_ADJUSTED,
    EVENT_STUDY_VERSION,
    LABEL_HORIZON_SESSIONS,
)

_CAUSALITY_DISCLAIMER = (
    "association only — an abnormal return following an event is not evidence "
    "the event caused it (master context section 37)"
)


class EventStudyError(ValueError):
    """Raised when a study cannot be performed as requested."""


@dataclass
class HorizonReaction:
    """One horizon's measured reaction."""

    horizon: str
    sessions: int
    stock_return: float
    benchmark_return: float | None = None
    sector_return: float | None = None
    abnormal_return: float | None = None
    sector_abnormal_return: float | None = None
    volatility_ratio: float | None = None
    volume_ratio: float | None = None
    exit_bar: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventStudyResult:
    """A complete study for one event."""

    ticker: str
    event_time: str
    status: str
    model: str = EVENT_STUDY_MODEL_MARKET_ADJUSTED
    entry_bar: str = ""
    entry_close: float = 0.0
    baseline: dict[str, Any] = field(default_factory=dict)
    reactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark: str | None = None
    sector: str | None = None
    event_id: str | None = None
    reason: str = ""
    study_version: str = EVENT_STUDY_VERSION
    disclaimer: str = _CAUSALITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def abnormal_return_for(self, horizon: str) -> float | None:
        """The abnormal return at one horizon, or None if unmeasured."""
        return (self.reactions.get(horizon) or {}).get("abnormal_return")

    def is_measured(self) -> bool:
        return self.status == "OK" and bool(self.reactions)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorted frame with a DatetimeIndex."""
    if frame is None or frame.empty:
        raise EventStudyError("price frame is empty")
    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(normalized.index)
    return normalized.sort_index()


def _entry_position(frame: pd.DataFrame, event_time: pd.Timestamp) -> int | None:
    """Index of the last bar at or before the event.

    The event is anchored on PUBLICATION: the market reacts when it is told,
    which is why E1 keeps published_time and effective_time distinct.
    """
    eligible = frame.index[frame.index <= event_time]
    return len(eligible) - 1 if len(eligible) else None


def _window_return(frame: pd.DataFrame, start: int, end: int) -> float | None:
    """Close-to-close return between two positions, or None if out of range."""
    if start < 0 or end >= len(frame) or end <= start:
        return None
    first = float(frame["Close"].iloc[start])
    last = float(frame["Close"].iloc[end])
    if first <= 0:
        return None
    return last / first - 1.0


def _baseline_window(frame: pd.DataFrame, entry: int) -> tuple[int, int] | None:
    """The estimation window, ending BEFORE the event with a gap.

    Information leaks ahead of an announcement, so the sessions immediately
    before publication are excluded: folding them into "normal" would shrink
    the abnormal return toward zero.
    """
    end = entry - EVENT_STUDY_BASELINE_GAP_SESSIONS
    start = end - EVENT_STUDY_BASELINE_SESSIONS
    start = max(0, start)
    if end - start < EVENT_STUDY_MIN_BASELINE_SESSIONS:
        return None
    return start, end


def _baseline_stats(frame: pd.DataFrame, start: int, end: int) -> dict[str, Any]:
    """Normal behaviour: mean daily return, volatility and mean volume."""
    closes = [float(v) for v in frame["Close"].iloc[start:end + 1]]
    returns = [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    volumes = [float(v) for v in frame["Volume"].iloc[start:end + 1] if float(v) > 0]
    return {
        "sessions": end - start,
        "start_bar": frame.index[start].strftime("%Y-%m-%d %H:%M:%S"),
        "end_bar": frame.index[end].strftime("%Y-%m-%d %H:%M:%S"),
        "mean_daily_return": round(statistics.fmean(returns), 8) if returns else None,
        "daily_volatility": (
            round(statistics.pstdev(returns), 8) if len(returns) > 1 else None
        ),
        "mean_volume": round(statistics.fmean(volumes), 2) if volumes else None,
        "gap_sessions": EVENT_STUDY_BASELINE_GAP_SESSIONS,
        "note": (
            "the baseline ends before the event with a gap, so pre-announcement "
            "drift cannot be absorbed into 'normal'"
        ),
    }


def _realized_volatility(frame: pd.DataFrame, start: int, end: int) -> float | None:
    closes = [float(v) for v in frame["Close"].iloc[start:end + 1]]
    returns = [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    return round(statistics.pstdev(returns), 8) if len(returns) > 1 else None


def _mean_volume(frame: pd.DataFrame, start: int, end: int) -> float | None:
    volumes = [float(v) for v in frame["Volume"].iloc[start + 1:end + 1] if float(v) > 0]
    return round(statistics.fmean(volumes), 2) if volumes else None


def _aligned_return(
    frame: pd.DataFrame | None,
    event_time: pd.Timestamp,
    sessions: int,
) -> float | None:
    """The same-window return for a benchmark or sector frame.

    Aligned by TIMESTAMP rather than by position: a benchmark with a
    different history length would otherwise be measured over a different
    calendar window, and the difference would land in the abnormal return.
    """
    if frame is None:
        return None
    try:
        aligned = _normalize(frame)
    except EventStudyError:
        return None
    entry = _entry_position(aligned, event_time)
    if entry is None:
        return None
    if sessions == 0:
        return _intraday_return(aligned, entry)
    return _window_return(aligned, entry, entry + sessions)


def _intraday_return(frame: pd.DataFrame, entry: int) -> float | None:
    """Open-to-close on the event session — what daily bars honestly support."""
    if entry < 0 or entry >= len(frame):
        return None
    open_price = float(frame["Open"].iloc[entry])
    close_price = float(frame["Close"].iloc[entry])
    if open_price <= 0:
        return None
    return close_price / open_price - 1.0


def run_event_study(
    ticker: str,
    event_time: str,
    price_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    sector_frame: pd.DataFrame | None = None,
    benchmark: str | None = None,
    sector: str | None = None,
    event_id: str | None = None,
    model: str = EVENT_STUDY_MODEL_MARKET_ADJUSTED,
) -> EventStudyResult:
    """Measure one event's reaction across every horizon.

    Returns a result with an explicit status rather than raising for ordinary
    conditions: an event too early in the history, or too recent for its
    windows to have elapsed, is a normal outcome that must be recorded.
    """
    if model not in EVENT_STUDY_MODELS:
        raise EventStudyError(
            f"unknown model {model!r} (known: {sorted(EVENT_STUDY_MODELS)})"
        )

    ticker = str(ticker).upper()
    try:
        frame = _normalize(price_frame)
    except EventStudyError as exc:
        return EventStudyResult(
            ticker=ticker, event_time=str(event_time), status="UNAVAILABLE",
            model=model, event_id=event_id, reason=str(exc),
        )

    # Normalised to tz-NAIVE before any comparison. Price bars are tz-naive,
    # while a real news timestamp arrives tz-aware UTC
    # ("2026-09-16T21:33:00Z"), and pandas refuses to compare the two --
    # `_entry_position` raised TypeError on the first live news event the
    # collector produced. Synthetic fixtures and the price-derived backfill
    # never exposed it, because both supply naive timestamps.
    stamp = as_naive_timestamp(event_time)
    entry = _entry_position(frame, stamp)
    if entry is None:
        return EventStudyResult(
            ticker=ticker, event_time=str(event_time), status="UNAVAILABLE",
            model=model, event_id=event_id,
            reason=(
                f"no price bar at or before {event_time} — the event predates "
                f"the available history"
            ),
        )

    window = _baseline_window(frame, entry)
    if window is None:
        return EventStudyResult(
            ticker=ticker, event_time=str(event_time), status="INSUFFICIENT_BASELINE",
            model=model, event_id=event_id,
            entry_bar=frame.index[entry].strftime("%Y-%m-%d %H:%M:%S"),
            reason=(
                f"fewer than {EVENT_STUDY_MIN_BASELINE_SESSIONS} sessions before "
                f"the event (with a {EVENT_STUDY_BASELINE_GAP_SESSIONS}-session "
                f"gap) — there is no honest description of normal behaviour"
            ),
        )

    baseline_start, baseline_end = window
    baseline = _baseline_stats(frame, baseline_start, baseline_end)
    baseline_volatility = baseline["daily_volatility"]
    baseline_volume = baseline["mean_volume"]

    reactions: dict[str, dict[str, Any]] = {}
    for horizon in EVENT_STUDY_HORIZONS:
        sessions = 0 if horizon == "intraday" else LABEL_HORIZON_SESSIONS[horizon]

        if sessions == 0:
            stock_return = _intraday_return(frame, entry)
            exit_position = entry
        else:
            exit_position = entry + sessions
            stock_return = _window_return(frame, entry, exit_position)

        if stock_return is None:
            # An unmatured or unavailable window is ABSENT, never zero: a 0.0
            # would enter training as a measured absence of reaction.
            continue

        benchmark_return = _aligned_return(benchmark_frame, stamp, sessions)
        sector_return = _aligned_return(sector_frame, stamp, sessions)

        if model == EVENT_STUDY_MODEL_MARKET_ADJUSTED:
            abnormal = (
                round(stock_return - benchmark_return, 8)
                if benchmark_return is not None else None
            )
        else:
            mean_daily = baseline["mean_daily_return"]
            abnormal = (
                round(stock_return - mean_daily * max(sessions, 1), 8)
                if mean_daily is not None else None
            )

        sector_abnormal = (
            round(stock_return - sector_return, 8)
            if sector_return is not None else None
        )

        volatility_ratio = None
        if sessions > 0 and baseline_volatility:
            realized = _realized_volatility(frame, entry, exit_position)
            if realized is not None:
                volatility_ratio = round(realized / baseline_volatility, 6)

        volume_ratio = None
        if baseline_volume:
            realized_volume = _mean_volume(frame, entry, max(exit_position, entry + 1))
            if realized_volume is not None:
                volume_ratio = round(realized_volume / baseline_volume, 6)

        reactions[horizon] = HorizonReaction(
            horizon=horizon,
            sessions=sessions,
            stock_return=round(stock_return, 8),
            benchmark_return=(
                round(benchmark_return, 8) if benchmark_return is not None else None
            ),
            sector_return=round(sector_return, 8) if sector_return is not None else None,
            abnormal_return=abnormal,
            sector_abnormal_return=sector_abnormal,
            volatility_ratio=volatility_ratio,
            volume_ratio=volume_ratio,
            exit_bar=frame.index[min(exit_position, len(frame) - 1)].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        ).to_dict()

    if not reactions:
        return EventStudyResult(
            ticker=ticker, event_time=str(event_time), status="UNMATURED",
            model=model, event_id=event_id, baseline=baseline,
            entry_bar=frame.index[entry].strftime("%Y-%m-%d %H:%M:%S"),
            entry_close=round(float(frame["Close"].iloc[entry]), 6),
            reason=(
                "no horizon window has elapsed since the event — every reaction "
                "is unmeasured, which is not the same as no reaction"
            ),
        )

    return EventStudyResult(
        ticker=ticker,
        event_time=str(event_time),
        status="OK",
        model=model,
        entry_bar=frame.index[entry].strftime("%Y-%m-%d %H:%M:%S"),
        entry_close=round(float(frame["Close"].iloc[entry]), 6),
        baseline=baseline,
        reactions=reactions,
        benchmark=benchmark,
        sector=sector,
        event_id=event_id,
    )


def study_event(
    event,
    price_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    sector_frame: pd.DataFrame | None = None,
    **kwargs,
) -> EventStudyResult:
    """Run a study for a canonical E1 event.

    Anchors on `published_time`: the market reacts when it is told, not when
    the effect dated from.
    """
    return run_event_study(
        ticker=getattr(event, "entity", ""),
        event_time=getattr(event, "published_time", ""),
        price_frame=price_frame,
        benchmark_frame=benchmark_frame,
        sector_frame=sector_frame,
        event_id=getattr(event, "event_id", None),
        **kwargs,
    )


def observations_from_studies(
    events,
    studies,
    horizon: str = "20d",
) -> list[Any]:
    """Turn studied events into E3 actor observations with real reactions.

    This is the join E3 was waiting for: an actor's track record needs
    measured abnormal returns, and until E4 existed there were none. Events
    without a named actor, or whose study did not measure the horizon, are
    skipped rather than recorded with a fabricated reaction.
    """
    from core.actor_intelligence import ActorObservation

    by_id = {study.event_id: study for study in studies if study.event_id}
    observations: list[ActorObservation] = []
    for event in events:
        actor = str(getattr(event, "actor", "") or "").strip()
        if not actor:
            continue
        study = by_id.get(getattr(event, "event_id", None))
        if study is None or not study.is_measured():
            continue
        abnormal = study.abnormal_return_for(horizon)
        if abnormal is None:
            continue
        observations.append(ActorObservation(
            actor=actor,
            event_type=str(getattr(event, "event_type", "other")),
            ticker=str(getattr(event, "entity", "")).upper(),
            published_time=str(getattr(event, "published_time", "")),
            novelty=float(getattr(event, "novelty", 0.0) or 0.0),
            abnormal_return=abnormal,
        ))
    return observations


def study_report(studies: list[EventStudyResult]) -> dict[str, Any]:
    """Coverage summary over a batch of studies."""
    by_status: dict[str, int] = {}
    for study in studies:
        by_status[study.status] = by_status.get(study.status, 0) + 1
    measured = [s for s in studies if s.is_measured()]
    return {
        "study_version": EVENT_STUDY_VERSION,
        "total": len(studies),
        "measured": len(measured),
        "by_status": dict(sorted(by_status.items())),
        "horizons": list(EVENT_STUDY_HORIZONS),
        "baseline_sessions": EVENT_STUDY_BASELINE_SESSIONS,
        "baseline_gap_sessions": EVENT_STUDY_BASELINE_GAP_SESSIONS,
        "disclaimer": _CAUSALITY_DISCLAIMER,
    }
