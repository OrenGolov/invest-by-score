"""Multi-timeframe representation (Sprint C1) — one as_of, five bar clocks.

Pipeline (each stage a pure function; `build_timeframe_snapshot` wires them):

    FETCH -> PIT BAR-ELIGIBILITY -> STATE PER TIMEFRAME -> ALIGNMENT -> OUTPUT

C1's requirement is that intraday, daily, weekly, monthly and yearly state all
answer for the SAME instant. The difficulty is not fetching five series; it is
that each series runs on its own clock, and a coarse bar is dishonest about
when its information existed.

**The bar-close rule.** A provider labels a bar at period START. Yahoo returns
the week of Sep 14 as `2026-09-14`; at `as_of = 2026-09-15` that bar's Close is
Friday's close — future data wearing a past timestamp. A naive `index <= as_of`
filter keeps it, and every downstream feature inherits the leak. So a bar is
eligible only once it has CLOSED:

    bar_open + close_after <= as_of

The same test removes the partial trailing bar providers append to weekly and
monthly series (a `2026-09-17` row inside a monthly series is not a month).
Excluded bars are COUNTED into `excluded_unclosed_bars` and
`excluded_future_bars`, never silently dropped — a timeframe that quietly
shrinks is indistinguishable from one with no data.

**Yearly is resampled, not fetched.** The provider has no 1y interval.
Inventing one would fork price truth (W5: one canonical implementation), so
yearly is derived by resampling the same monthly bars that feed the monthly
timeframe, after those bars pass the eligibility test.

**Fail-closed, per timeframe.** A timeframe with too few eligible bars is
INCOMPLETE with `trend` explicitly None; a failed fetch is UNAVAILABLE; a
malformed payload is INVALID. The snapshot's own status is the WORST of its
parts, because a multi-timeframe view is only as trustworthy as the weakest
clock in it.

**This is a representation, not a score.** It carries no ensemble weight and
vetoes nothing. C2 builds registered features on top of it.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.config import (
    TIMEFRAME_CONTRACT_VERSION,
    TIMEFRAME_MIN_BARS,
    TIMEFRAME_ORDER,
    TIMEFRAME_PIPELINE_VERSION,
    TIMEFRAME_SPECS,
    TIMEFRAME_TREND_DOWN,
    TIMEFRAME_TREND_FLAT,
    TIMEFRAME_TREND_FLAT_BAND,
    TIMEFRAME_TREND_UP,
)
from core.schemas import MultiTimeframeSnapshot, TimeframeState

LOGGER = logging.getLogger("core.timeframes")

TIMEFRAME_SOURCE_ID = "yahoo_finance_chart"

STATUS_OK = "OK"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_INVALID = "INVALID"

# Worst-wins ordering for the snapshot verdict.
_STATUS_SEVERITY = {STATUS_OK: 0, STATUS_INCOMPLETE: 1, STATUS_UNAVAILABLE: 2, STATUS_INVALID: 3}

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


class TimeframeError(ValueError):
    """Raised when a timeframe request is structurally invalid."""


def _as_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None) if stamp.tz is not None else stamp.tz_localize(None)
    return stamp


def eligible_bars(
    frame: pd.DataFrame,
    as_of,
    close_after: str,
) -> tuple[pd.DataFrame, int, int]:
    """Bars that had CLOSED at as_of, plus what was excluded and why.

    Returns `(eligible, unclosed_count, future_count)`.

    - `future_count`: bars opening strictly after as_of. Plainly not knowable.
    - `unclosed_count`: bars that opened at or before as_of but whose period
      had not yet elapsed. These are the dangerous ones — they carry a past
      timestamp and a future Close.
    """
    if frame is None or frame.empty:
        return (frame if frame is not None else pd.DataFrame()), 0, 0

    target = _as_timestamp(as_of)
    period = pd.Timedelta(close_after)
    index = pd.to_datetime(frame.index)
    opens_after = index > target
    closes_after = (index + period) > target

    future_count = int(opens_after.sum())
    # Unclosed = has not finished, but did not open in the future either.
    unclosed_count = int((closes_after & ~opens_after).sum())

    eligible = frame.loc[~closes_after].copy()
    return eligible, unclosed_count, future_count


def _resample_yearly(frame: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, str | None]:
    """Aggregate already-eligible bars onto a coarser clock.

    Only closed bars reach this function, so the aggregate cannot contain
    information that did not exist at as_of.

    Returns `(aggregated, last_source_bar)`. The second value exists because
    resampling LABELS each bucket at period end: a bucket fed by Jan-Jun 2025
    is stamped 2025-12-31, so `last_bar_time` would otherwise report a date six
    months after the newest real bar, and a year-over-year comparison would put
    half a year against full years. Reporting the true last source bar is what
    lets `partial_final_bucket` be honest about it.
    """
    if frame.empty:
        return frame, None
    aggregated = frame.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return aggregated.dropna(subset=["Close"]), str(frame.index[-1])


def classify_trend(first_close: float, last_close: float) -> str:
    """Label a window's direction against a flat band.

    The band exists so a rounding-level drift is not reported as a trend.
    """
    if first_close <= 0:
        return TIMEFRAME_TREND_FLAT
    change = (last_close - first_close) / first_close
    if change > TIMEFRAME_TREND_FLAT_BAND:
        return TIMEFRAME_TREND_UP
    if change < -TIMEFRAME_TREND_FLAT_BAND:
        return TIMEFRAME_TREND_DOWN
    return TIMEFRAME_TREND_FLAT


def build_timeframe_state(
    timeframe: str,
    frame: pd.DataFrame | None,
    as_of,
    fetch_failed: bool = False,
) -> TimeframeState:
    """Reduce one raw series to its PIT state at as_of."""
    if timeframe not in TIMEFRAME_SPECS:
        raise TimeframeError(f"unknown timeframe {timeframe!r}")
    spec = TIMEFRAME_SPECS[timeframe]
    interval = str(spec["interval"])

    def _empty(status: str, reason: str, unclosed: int = 0, future: int = 0) -> TimeframeState:
        return TimeframeState(
            timeframe=timeframe,
            interval=interval,
            status=status,
            bar_count=0,
            first_bar_time=None,
            last_bar_time=None,
            last_close=None,
            change_pct=None,
            trend=None,
            excluded_unclosed_bars=unclosed,
            excluded_future_bars=future,
            reason=reason,
        )

    if fetch_failed:
        return _empty(STATUS_UNAVAILABLE, f"{timeframe}: provider fetch failed")
    if frame is None or frame.empty:
        return _empty(STATUS_UNAVAILABLE, f"{timeframe}: provider returned no bars")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return _empty(STATUS_INVALID, f"{timeframe}: payload missing {', '.join(missing)}")
    # An out-of-order frame is a malformed payload, not a degraded one: every
    # window reads backwards from the last row, so reversed bars would produce
    # a confident, wrong trend rather than an absent one.
    if not pd.to_datetime(frame.index).is_monotonic_increasing:
        return _empty(
            STATUS_INVALID,
            f"{timeframe}: bars are not in ascending time order",
        )

    eligible, unclosed, future = eligible_bars(frame, as_of, str(spec["close_after"]))
    last_source_bar: str | None = None
    if spec.get("resample"):
        eligible, last_source_bar = _resample_yearly(eligible, str(spec["resample"]))
    eligible = eligible.dropna(subset=["Close"])

    if eligible.empty:
        return _empty(
            STATUS_INCOMPLETE,
            f"{timeframe}: no bar had closed at as_of",
            unclosed,
            future,
        )

    minimum = TIMEFRAME_MIN_BARS[timeframe]
    first_close = float(eligible["Close"].iloc[0])
    last_close = float(eligible["Close"].iloc[-1])
    change_pct = round((last_close - first_close) / first_close, 6) if first_close > 0 else None

    if len(eligible) < minimum:
        state = _empty(
            STATUS_INCOMPLETE,
            f"{timeframe}: {len(eligible)} eligible bars, {minimum} required for a trend",
            unclosed,
            future,
        )
        # The bars that DO exist are still reported; only the trend is withheld.
        state.bar_count = int(len(eligible))
        state.first_bar_time = str(eligible.index[0])
        state.last_bar_time = str(eligible.index[-1])
        state.last_close = round(last_close, 6)
        state.change_pct = change_pct
        return state

    # A resampled bucket is labelled at period END, so comparing the label to
    # the newest source bar would flag EVERY year: a monthly bar is stamped at
    # month START, so a complete December reads 2025-12-01 against a 2025-12-31
    # label. The real question is whether the source reaches the bucket's final
    # PERIOD, so the comparison is made one source period back from the label.
    partial_final = False
    if last_source_bar is not None:
        label = pd.Timestamp(eligible.index[-1])
        source_period = pd.Timedelta(str(spec["close_after"])) / 12  # monthly feed
        partial_final = pd.Timestamp(last_source_bar) < label - source_period

    return TimeframeState(
        timeframe=timeframe,
        interval=interval,
        status=STATUS_OK,
        bar_count=int(len(eligible)),
        first_bar_time=str(eligible.index[0]),
        last_bar_time=str(eligible.index[-1]),
        last_close=round(last_close, 6),
        change_pct=change_pct,
        trend=classify_trend(first_close, last_close),
        excluded_unclosed_bars=unclosed,
        excluded_future_bars=future,
        last_source_bar=last_source_bar,
        partial_final_bucket=partial_final,
        reason=(
            f"the final {timeframe} bucket is labelled {eligible.index[-1]} but its "
            f"newest source bar is {last_source_bar} — it is a partial period"
            if partial_final else ""
        ),
    )


def build_alignment(states: dict[str, TimeframeState]) -> dict:
    """Summarise agreement across timeframes.

    Agreement is DESCRIPTIVE, not a signal: it says the clocks tell the same
    story, never that the story is right. Only OK timeframes vote, because an
    INCOMPLETE one has no trend to cast.
    """
    voting = {name: state for name, state in states.items() if state.status == STATUS_OK and state.trend}
    trends = {name: state.trend for name, state in voting.items()}
    distinct = set(trends.values())
    return {
        "trends": trends,
        "voting_timeframes": sorted(voting),
        "silent_timeframes": sorted(name for name in states if name not in voting),
        "aligned": len(distinct) == 1 and len(trends) > 1,
        "conflicted": len(distinct) > 1,
        "dominant_trend": next(iter(distinct)) if len(distinct) == 1 and trends else None,
    }


def worst_status(states: dict[str, TimeframeState]) -> str:
    """The snapshot is only as trustworthy as its weakest clock."""
    if not states:
        return STATUS_UNAVAILABLE
    return max((state.status for state in states.values()), key=lambda s: _STATUS_SEVERITY.get(s, 0))


def build_timeframe_snapshot(
    ticker: str,
    as_of,
    fetcher=None,
) -> dict:
    """Assemble every timeframe's state for one ticker at one as_of.

    `fetcher(ticker, period, interval) -> DataFrame` is injectable so the
    contract can be tested without a network round trip; it defaults to the
    project's single price-history entry point.
    """
    if not str(ticker or "").strip():
        raise TimeframeError("a ticker is required")
    target = _as_timestamp(as_of)

    if fetcher is None:
        from fetch_data import fetch_price_history

        def fetcher(symbol, period, interval):  # noqa: ANN001 - thin adapter
            return fetch_price_history(symbol, period=period, interval=interval)

    states: dict[str, TimeframeState] = {}
    for timeframe in TIMEFRAME_ORDER:
        spec = TIMEFRAME_SPECS[timeframe]
        frame = None
        failed = False
        try:
            frame = fetcher(ticker, str(spec["period"]), str(spec["interval"]))
        except Exception as exc:  # provider failure is a status, never a crash
            LOGGER.warning("timeframe fetch failed for %s/%s: %s", ticker, timeframe, exc)
            failed = True
        states[timeframe] = build_timeframe_state(timeframe, frame, target, fetch_failed=failed)

    status = worst_status(states)
    degraded = sorted(name for name, state in states.items() if state.status != STATUS_OK)
    snapshot = MultiTimeframeSnapshot(
        ticker=str(ticker).upper(),
        as_of=target.strftime("%Y-%m-%d %H:%M:%S"),
        status=status,
        source_id=TIMEFRAME_SOURCE_ID,
        calculation_version=TIMEFRAME_CONTRACT_VERSION,
        pipeline_version=TIMEFRAME_PIPELINE_VERSION,
        timeframes={name: states[name].to_dict() for name in TIMEFRAME_ORDER},
        alignment=build_alignment(states),
        reason="" if status == STATUS_OK else f"degraded timeframes: {', '.join(degraded)}",
    )
    return snapshot.to_dict()


def snapshot_problems(snapshot: dict) -> list[str]:
    """Validate a built snapshot against the C1 contract."""
    problems: list[str] = []
    if not isinstance(snapshot, dict):
        return ["snapshot must be a dict"]
    for field_name in ("ticker", "as_of", "status", "calculation_version", "timeframes"):
        if not snapshot.get(field_name):
            problems.append(f"snapshot field {field_name!r} missing/empty")
    timeframes = snapshot.get("timeframes") or {}
    for name in TIMEFRAME_ORDER:
        if name not in timeframes:
            problems.append(f"timeframe {name!r} absent — C1 requires every clock to answer")
            continue
        state = timeframes[name]
        if state.get("status") == STATUS_OK and not state.get("trend"):
            problems.append(f"timeframe {name!r} is OK but carries no trend")
        if state.get("status") != STATUS_OK and state.get("trend"):
            problems.append(f"timeframe {name!r} is {state.get('status')} but still reports a trend")
    return problems
