"""A1 forecast change alert — what changed, and whether it was the forecast.

"Alert when the forecast materially changes." Two obvious implementations both
fire on things that are not changes, and both were measured rather than argued.

**THE FIRST TRAP — DIGEST DIFFING.** A snapshot is digest-addressable, so the
tempting test is "did the digest change?". MEASURED on four consecutive days of
an identical, entirely-unavailable forecast: **four distinct digests**. The
digest covers `as_of`, so it changes every day by construction. A digest-diff
alert fires on 100% of days while the forecast never changes at all — and on
this system, which produces no forecast values, every one of those alerts is
about nothing.

**THE SECOND TRAP — COALESCING NULLS.** MEASURED: comparing values through
`float(x or 0)` turns a forecast APPEARING (REFUSED → 0.56) into a +0.56 move,
and a forecast DISAPPEARING (0.56 → REFUSED) into a −0.56 crash. Nothing fell.
The availability changed, which is a different event and one a reader acts on
differently — a vanished forecast means "the evidence went away", not "the odds
collapsed".

**So availability is compared first, and magnitude only between two values that
both exist.** APPEARED and DISAPPEARED are first-class change kinds rather than
magnitudes, because that fold is exactly what produces the phantom crash.

**The threshold is derived, not chosen.** At F4's point-estimate floor of 40
observations the 95% band on a base rate near 0.5 is ±0.155; at 100
observations it is ±0.098. A threshold of 0.10 is the point where a move starts
to mean something for a realistically-sized cell. Below that the alert fires on
resampling rather than on news.

**An alert reports; it does not trade and does not override governance.** A7
adds suppression and the risk gate on top of this.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    ALERT_CHANGE_APPEARED,
    ALERT_CHANGE_DISAPPEARED,
    ALERT_CHANGE_MOVED,
    ALERT_CHANGE_NONE,
    ALERT_CHANGE_NOT_EVALUATED,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    FORECAST_ALERT_AVAILABILITY_IS_MATERIAL,
    FORECAST_ALERT_BLOCKS_TRADES,
    FORECAST_ALERT_MIN_PROBABILITY_MOVE,
    FORECAST_ALERT_USES_DIGEST,
    FORECAST_ALERT_VERSION,
    FORECAST_CHANGE_KINDS,
    SNAPSHOT_STATUS_PRESENT,
)


class ForecastAlertError(ValueError):
    """Raised when a forecast change request is structurally invalid."""


def _value_of(snapshot: Mapping[str, Any] | None, field: str) -> float | None:
    """A field's value IFF it was actually measured.

    None here means "no measurement", and it is never coerced to 0.0 — that
    coercion is the second trap this module exists to avoid.
    """
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise ForecastAlertError("a snapshot must be a mapping")
    cell = (snapshot.get("fields") or {}).get(field)
    if not isinstance(cell, Mapping):
        return None
    if cell.get("status") != SNAPSHOT_STATUS_PRESENT:
        return None
    value = cell.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ForecastAlertError(
            f"{field} is PRESENT with a non-numeric value {value!r}"
        ) from None


def classify_change(
    previous: float | None,
    current: float | None,
    *,
    threshold: float = FORECAST_ALERT_MIN_PROBABILITY_MOVE,
) -> tuple[str, float | None]:
    """Which kind of change this is, and the magnitude IFF one applies.

    Availability is decided before magnitude. A delta exists only when both
    sides were measured; APPEARED and DISAPPEARED carry no delta at all,
    because there is no number to subtract from.
    """
    if not 0.0 < threshold < 1.0:
        raise ForecastAlertError("the threshold must lie inside (0, 1)")

    if previous is None and current is None:
        return ALERT_CHANGE_NOT_EVALUATED, None
    if previous is None:
        return ALERT_CHANGE_APPEARED, None
    if current is None:
        return ALERT_CHANGE_DISAPPEARED, None

    delta = round(float(current) - float(previous), 8)
    if abs(delta) >= threshold:
        return ALERT_CHANGE_MOVED, delta
    return ALERT_CHANGE_NONE, delta


def forecast_change_alert(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    field: str = "probability_up",
    threshold: float = FORECAST_ALERT_MIN_PROBABILITY_MOVE,
) -> dict:
    """Did the forecast materially change between two snapshots?

    Compares the measured VALUE, never the digest: MEASURED, four consecutive
    days of an identical unavailable forecast produced four distinct digests.
    """
    if current is None:
        raise ForecastAlertError("a current snapshot is required")
    if not isinstance(current, Mapping):
        raise ForecastAlertError("the current snapshot must be a mapping")

    before = _value_of(previous, field)
    after = _value_of(current, field)
    kind, delta = classify_change(before, after, threshold=threshold)

    # NO PRIOR IS NOT NO CHANGE. The first observation of a ticker cannot be a
    # change, and reporting NONE would claim a comparison that never happened.
    if previous is None:
        kind = ALERT_CHANGE_NOT_EVALUATED
        delta = None
        reason = (
            "no previous snapshot was supplied, so nothing can be compared; "
            "a first observation is not a change"
        )
        fired = False
        severity = None
    else:
        fired, severity, reason = _verdict(kind, delta, before, after, threshold)

    return {
        "version": FORECAST_ALERT_VERSION,
        "alert": "forecast_change",
        "ticker": (current.get("fields") or {}).get("ticker", {}).get("value"),
        "as_of": (current.get("fields") or {}).get("as_of", {}).get("value"),
        "horizon": (current.get("fields") or {}).get("horizon", {}).get("value"),
        "field": field,
        "kind": kind,
        "fired": fired,
        "severity": severity,
        "previous": before,
        "current": after,
        "delta": delta,
        "threshold": threshold,
        "reason": reason,
        "compares_digest": FORECAST_ALERT_USES_DIGEST,
        "blocks_trades": FORECAST_ALERT_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _verdict(kind, delta, before, after, threshold):
    """Whether this change fires, at what severity, and why."""
    if kind == ALERT_CHANGE_APPEARED:
        return (
            FORECAST_ALERT_AVAILABILITY_IS_MATERIAL,
            ALERT_SEVERITY_INFO,
            (
                f"the forecast became available at {after:.4f}; it was not "
                f"measured before. This is an APPEARANCE, not a move of "
                f"{after:.4f} from zero"
            ),
        )
    if kind == ALERT_CHANGE_DISAPPEARED:
        return (
            FORECAST_ALERT_AVAILABILITY_IS_MATERIAL,
            ALERT_SEVERITY_WARN,
            (
                f"the forecast was {before:.4f} and is no longer measured. "
                f"The evidence went away; nothing fell to zero"
            ),
        )
    if kind == ALERT_CHANGE_MOVED:
        return (
            True,
            ALERT_SEVERITY_WARN,
            (
                f"{before:.4f} -> {after:.4f} ({delta:+.4f}), at or beyond the "
                f"{threshold} threshold. DERIVED, the 95% sampling band on a "
                f"base rate near 0.5 is +/-0.098 at 100 observations, so a "
                f"move this size is not resampling"
            ),
        )
    if kind == ALERT_CHANGE_NOT_EVALUATED:
        return (
            False,
            None,
            "the forecast was unmeasured before and after; there is nothing "
            "to compare",
        )
    return (
        False,
        None,
        (
            f"{before:.4f} -> {after:.4f} ({delta:+.4f}), inside the "
            f"{threshold} threshold and within sampling noise"
        ),
    )


_NOTE = (
    "A1 compares measured VALUES, never digests. MEASURED, four consecutive "
    "days of an identical unavailable forecast produced four distinct "
    "digests, so a digest-diff alert fires every day on nothing. Availability "
    "is compared before magnitude, because folding a vanished forecast into a "
    "delta reports a crash that never happened."
)


def alert_problems(alert: Mapping[str, Any]) -> list[str]:
    """Contract check on a forecast change alert. Empty means clean."""
    problems: list[str] = []
    if not isinstance(alert, Mapping):
        return ["alert is not a mapping"]

    if alert.get("blocks_trades"):
        problems.append("an alert claims to block trades — it reports")
    if alert.get("compares_digest"):
        problems.append(
            "the alert compares digests. MEASURED, four consecutive days of "
            "an identical unavailable forecast produced four distinct digests"
        )

    kind = alert.get("kind")
    if kind not in FORECAST_CHANGE_KINDS:
        problems.append(f"unknown change kind {kind!r}")
        return problems
    if not str(alert.get("reason") or "").strip():
        problems.append(f"{kind} with no reason")

    previous = alert.get("previous")
    current = alert.get("current")
    delta = alert.get("delta")

    # THE SHAPE RULE: a delta exists IFF both sides were measured.
    if kind in (ALERT_CHANGE_APPEARED, ALERT_CHANGE_DISAPPEARED):
        if delta is not None:
            problems.append(
                f"{kind} carried a delta of {delta!r}. An availability change "
                f"has no magnitude: folding it into one turns a vanished "
                f"forecast into a crash that never happened"
            )
    elif kind in (ALERT_CHANGE_MOVED, ALERT_CHANGE_NONE):
        if previous is None or current is None:
            problems.append(
                f"{kind} between {previous!r} and {current!r} — a magnitude "
                f"comparison needs two measured values"
            )
        elif delta is None:
            problems.append(f"{kind} with no delta")
        else:
            expected = round(float(current) - float(previous), 8)
            if abs(expected - float(delta)) > 1e-9:
                problems.append(
                    f"delta {delta} does not equal {current} - {previous}"
                )

    if kind == ALERT_CHANGE_APPEARED and previous is not None:
        problems.append("APPEARED with a previous value")
    if kind == ALERT_CHANGE_DISAPPEARED and current is not None:
        problems.append("DISAPPEARED with a current value")

    # A FIRED ALERT MUST CARRY A SEVERITY, and a quiet one must not.
    fired = bool(alert.get("fired"))
    severity = alert.get("severity")
    if fired and severity is None:
        problems.append(f"{kind} fired with no severity")
    if not fired and severity is not None:
        problems.append(f"{kind} did not fire but carried severity {severity!r}")

    # A MOVE BELOW THE THRESHOLD MUST NOT FIRE.
    threshold = alert.get("threshold")
    if kind == ALERT_CHANGE_MOVED and delta is not None and threshold is not None:
        if abs(float(delta)) < float(threshold):
            problems.append(
                f"MOVED on a {delta} delta, inside the {threshold} threshold "
                f"— that is sampling noise, not news"
            )
    if kind == ALERT_CHANGE_NONE and fired:
        problems.append("an unchanged forecast fired an alert")
    if kind == ALERT_CHANGE_NOT_EVALUATED and fired:
        problems.append(
            "an alert fired with nothing to compare against — a first "
            "observation is not a change"
        )
    return problems


def render_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable line, plus the reason."""
    mark = "FIRED" if alert.get("fired") else "quiet"
    return [
        f"  {str(alert.get('ticker') or '?'):8s} {str(alert.get('horizon') or '?'):5s} "
        f"{str(alert.get('kind')):14s} {mark:6s} "
        f"{_shown(alert.get('previous'))} -> {_shown(alert.get('current'))}"
        f"  [{alert.get('severity') or '—'}]",
        f"      {alert.get('reason')}",
    ]


def _shown(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"
