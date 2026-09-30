"""A3 high-impact event alert — measured from outcomes, never from the claim.

"Alert when a high-impact event is detected." The obvious implementation reads
the event's own `magnitude` field and fires above a threshold. That is exactly
what the E-sprint contract forbids, and the reason is structural.

**THE TRAP — ALERTING ON AN UNVALIDATED CLAIM.** `magnitude` is documented in
`core/event_contract.py` as "a bounded, unitless claim size — NOT an expected
return", precisely so an unvalidated number is not treated as a forecast.
MEASURED, it is worse than unvalidated: `magnitude` appears **nowhere** in
`core/event_memory.py` or `core/event_study.py`, and is not among
`EventMemory`'s fields. It is never carried into memory, so it has never been
compared against a single realized outcome. An alert keyed on it would fire on
an assertion nobody has ever checked.

**WHAT IS MEASURABLE: the event type's realized history.** E6 stores the
abnormal return per horizon and E5's attribution verdict. MEASURED on 400
seeded memories across four event types, the median absolute 20d move differs
by more than **12×** between them:

    event_type        n    median |move|   p90 |move|
    earnings_beat    94        0.0497        0.0962
    guidance_cut    114        0.0470        0.1056
    analyst_note    102        0.0085        0.0183
    minor_pr         90        0.0040        0.0088

So "high impact" means *this kind of event has historically moved this name*,
measured from stored outcomes rather than claimed in the payload.

**Impact is not direction.** A large move is high-impact whether up or down,
so the measure is the median *absolute* return. A reader needs to know a name
is about to move before knowing which way, and folding sign in would let a
symmetric history cancel itself to zero and read as harmless.

**UNKNOWN is not LOW.** An event type with too little history is not a quiet
one; it is one nobody can rate. MEASURED, a median |move| estimate at n=3
spans a five-fold range across resamples. Reporting an unrated type as
LOW_IMPACT would silence exactly the events the system has never seen before,
which are the ones most worth a human look.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    EVENT_IMPACT_ALERT_VERSION,
    EVENT_IMPACT_BLOCKS_TRADES,
    EVENT_IMPACT_HIGH,
    EVENT_IMPACT_HORIZON,
    EVENT_IMPACT_LOW,
    EVENT_IMPACT_MAGNITUDE_REASON,
    EVENT_IMPACT_MIN_ANALOGS,
    EVENT_IMPACT_MIN_MEDIAN_MOVE,
    EVENT_IMPACT_NO_EVENT,
    EVENT_IMPACT_UNKNOWN,
    EVENT_IMPACT_UNKNOWN_IS_NOT_LOW,
    EVENT_IMPACT_USES_ABSOLUTE_MOVE,
    EVENT_IMPACT_USES_CLAIMED_MAGNITUDE,
    EVENT_IMPACT_VERDICTS,
)


class EventImpactAlertError(ValueError):
    """Raised when an impact request is structurally invalid."""


def realized_impact(
    memories: Iterable[Any],
    event_type: str,
    *,
    horizon: str = EVENT_IMPACT_HORIZON,
    min_analogs: int = EVENT_IMPACT_MIN_ANALOGS,
) -> dict:
    """How far this KIND of event has historically moved the name.

    Measured from stored abnormal returns, never from a claimed magnitude.
    Returns no statistics at all below the analog floor — an unrated type has
    no median, not a median of zero.
    """
    if not str(event_type or "").strip():
        raise EventImpactAlertError("an event type is required")
    if min_analogs < 1:
        raise EventImpactAlertError("the analog floor must be at least 1")

    moves: list[float] = []
    associated = 0
    for memory in memories or []:
        if getattr(memory, "event_type", None) != event_type:
            continue
        response = memory.response_at(horizon)
        if response is None:
            continue
        moves.append(abs(float(response)) if EVENT_IMPACT_USES_ABSOLUTE_MOVE
                     else float(response))
        if memory.is_event_associated(horizon):
            associated += 1

    if len(moves) < min_analogs:
        return {
            "event_type": event_type,
            "horizon": horizon,
            "analogs": len(moves),
            "required": min_analogs,
            "median_move": None,
            "p90_move": None,
            "event_associated_share": None,
            "measured": False,
            "reason": (
                f"{len(moves)} remembered {event_type} outcome(s) against "
                f"{min_analogs} required. MEASURED, a median |move| estimate "
                f"at n=3 spans a five-fold range across resamples, so this "
                f"kind of event cannot be rated yet"
            ),
        }

    ordered = sorted(moves)
    return {
        "event_type": event_type,
        "horizon": horizon,
        "analogs": len(moves),
        "required": min_analogs,
        "median_move": round(statistics.median(ordered), 6),
        "p90_move": round(ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)], 6),
        "event_associated_share": round(associated / len(moves), 4),
        "measured": True,
        "reason": (
            f"{len(moves)} remembered {event_type} outcome(s) at {horizon}; "
            f"impact is the median ABSOLUTE abnormal return, so a symmetric "
            f"history cannot cancel itself to zero"
        ),
    }


def event_impact_alert(
    event: Mapping[str, Any] | None,
    memories: Iterable[Any] | None = None,
    *,
    horizon: str = EVENT_IMPACT_HORIZON,
    threshold: float = EVENT_IMPACT_MIN_MEDIAN_MOVE,
    min_analogs: int = EVENT_IMPACT_MIN_ANALOGS,
) -> dict:
    """Is this event the kind that has historically moved the name?"""
    if not 0.0 < threshold < 1.0:
        raise EventImpactAlertError("the threshold must lie inside (0, 1)")

    if event is None:
        return _alert(
            EVENT_IMPACT_NO_EVENT,
            fired=False,
            severity=None,
            reason="no event was supplied; there is nothing to rate",
            event_type=None,
            impact=None,
            threshold=threshold,
        )
    if not isinstance(event, Mapping):
        raise EventImpactAlertError("the event must be a mapping")

    event_type = str(event.get("event_type") or "").strip()
    if not event_type:
        return _alert(
            EVENT_IMPACT_NO_EVENT,
            fired=False,
            severity=None,
            reason=(
                "the event carries no type, so no historical population "
                "identifies it"
            ),
            event_type=None,
            impact=None,
            threshold=threshold,
        )

    impact = realized_impact(
        memories or [], event_type, horizon=horizon, min_analogs=min_analogs
    )

    # UNKNOWN IS NOT LOW. An unrated type fires for a human look rather than
    # being silenced as quiet.
    if not impact["measured"]:
        return _alert(
            EVENT_IMPACT_UNKNOWN,
            fired=bool(EVENT_IMPACT_UNKNOWN_IS_NOT_LOW),
            severity=ALERT_SEVERITY_INFO,
            reason=(
                f"{event_type} cannot be rated: {impact['reason']}. An "
                f"unrated event type is not a quiet one, and reporting it as "
                f"LOW_IMPACT would silence exactly the events never seen "
                f"before"
            ),
            event_type=event_type,
            impact=impact,
            threshold=threshold,
            ticker=event.get("ticker"),
            published_time=event.get("published_time"),
        )

    median = float(impact["median_move"])
    if median >= threshold:
        return _alert(
            EVENT_IMPACT_HIGH,
            fired=True,
            severity=ALERT_SEVERITY_WARN,
            reason=(
                f"{event_type} has historically moved this name a median "
                f"{median:.2%} at {horizon} over {impact['analogs']} "
                f"remembered outcome(s), at or beyond the {threshold:.1%} "
                f"threshold. This is realized history, not the event's own "
                f"claimed magnitude"
            ),
            event_type=event_type,
            impact=impact,
            threshold=threshold,
            ticker=event.get("ticker"),
            published_time=event.get("published_time"),
        )

    return _alert(
        EVENT_IMPACT_LOW,
        fired=False,
        severity=None,
        reason=(
            f"{event_type} has historically moved this name a median "
            f"{median:.2%} at {horizon} over {impact['analogs']} remembered "
            f"outcome(s), inside the {threshold:.1%} threshold"
        ),
        event_type=event_type,
        impact=impact,
        threshold=threshold,
        ticker=event.get("ticker"),
        published_time=event.get("published_time"),
    )


def _alert(verdict: str, **detail) -> dict:
    """One A3 answer, always carrying why magnitude was not consulted."""
    if verdict not in EVENT_IMPACT_VERDICTS:
        raise EventImpactAlertError(f"unknown impact verdict {verdict!r}")
    payload = {
        "version": EVENT_IMPACT_ALERT_VERSION,
        "alert": "high_impact_event",
        "verdict": verdict,
        "uses_claimed_magnitude": EVENT_IMPACT_USES_CLAIMED_MAGNITUDE,
        "magnitude_reason": EVENT_IMPACT_MAGNITUDE_REASON,
        "blocks_trades": EVENT_IMPACT_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "Impact is measured from REALIZED history, never from the event's claimed "
    "magnitude. MEASURED, magnitude appears in neither event_memory nor "
    "event_study and is not an EventMemory field, so it has never been "
    "compared against an outcome. Event TYPE is measurable: median absolute "
    "20d moves differ by more than 12x across types."
)


def impact_alert_problems(alert: Mapping[str, Any]) -> list[str]:
    """Contract check on an impact alert. Empty means clean."""
    problems: list[str] = []
    if not isinstance(alert, Mapping):
        return ["alert is not a mapping"]

    if alert.get("blocks_trades"):
        problems.append("an alert claims to block trades — it reports")
    if alert.get("uses_claimed_magnitude"):
        problems.append(
            "the alert reads the event's claimed magnitude. MEASURED, that "
            "field appears in neither event_memory nor event_study and has "
            "never been compared against a realized outcome"
        )
    if not str(alert.get("magnitude_reason") or "").strip():
        problems.append(
            "the refusal to read magnitude carries no reason"
        )
    # THE CLAIMED MAGNITUDE MUST NOT APPEAR ANYWHERE IN THE PAYLOAD. Two
    # fields legitimately name it in order to REFUSE it — the declaration that
    # it is unused, and the reason why — so they are exempt by name rather
    # than by a substring rule that would also exempt a smuggled value.
    _MAGNITUDE_DECLARATIONS = {"uses_claimed_magnitude", "magnitude_reason"}
    for key in alert:
        if "magnitude" in str(key).lower() and key not in _MAGNITUDE_DECLARATIONS:
            problems.append(
                f"field {key!r} carries a claimed magnitude into the alert"
            )

    verdict = alert.get("verdict")
    if verdict not in EVENT_IMPACT_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")
        return problems
    if not str(alert.get("reason") or "").strip():
        problems.append(f"{verdict} with no reason")

    impact = alert.get("impact")
    fired = bool(alert.get("fired"))
    severity = alert.get("severity")

    if verdict == EVENT_IMPACT_NO_EVENT:
        if impact is not None:
            problems.append("NO_EVENT carried an impact measurement")
        if fired:
            problems.append("NO_EVENT fired an alert")
        return problems

    if not isinstance(impact, Mapping):
        problems.append(f"{verdict} carried no impact measurement")
        return problems

    # THE SHAPE RULE: statistics exist IFF they were measured.
    if verdict == EVENT_IMPACT_UNKNOWN:
        for field in ("median_move", "p90_move", "event_associated_share"):
            if impact.get(field) is not None:
                problems.append(
                    f"UNKNOWN_IMPACT carried {field}={impact.get(field)!r}; an "
                    f"unrated event type has no statistics"
                )
        if impact.get("measured"):
            problems.append("UNKNOWN_IMPACT claims it was measured")
        if EVENT_IMPACT_UNKNOWN_IS_NOT_LOW and not fired:
            problems.append(
                "an unrated event type did not fire — that silences exactly "
                "the events the system has never seen before"
            )
    else:
        if impact.get("median_move") is None:
            problems.append(f"{verdict} carried no median move")
        elif int(impact.get("analogs") or 0) < int(
            impact.get("required") or EVENT_IMPACT_MIN_ANALOGS
        ):
            problems.append(
                f"{verdict} was rated from {impact.get('analogs')} analogs, "
                f"below the {impact.get('required')} required"
            )
        else:
            median = float(impact["median_move"])
            threshold = float(alert.get("threshold") or EVENT_IMPACT_MIN_MEDIAN_MOVE)
            if verdict == EVENT_IMPACT_HIGH and median < threshold:
                problems.append(
                    f"HIGH_IMPACT on a {median} median, inside the "
                    f"{threshold} threshold"
                )
            if verdict == EVENT_IMPACT_LOW and median >= threshold:
                problems.append(
                    f"LOW_IMPACT on a {median} median, at or beyond the "
                    f"{threshold} threshold"
                )
            if median < 0:
                problems.append(
                    f"a negative median move {median} — impact is measured on "
                    f"the ABSOLUTE move so a symmetric history cannot cancel "
                    f"to zero"
                )

    if verdict == EVENT_IMPACT_HIGH and not fired:
        problems.append("a HIGH_IMPACT event did not fire")
    if verdict == EVENT_IMPACT_LOW and fired:
        problems.append("a LOW_IMPACT event fired")
    if fired and severity is None:
        problems.append(f"{verdict} fired with no severity")
    if not fired and severity is not None:
        problems.append(f"{verdict} did not fire but carried severity {severity!r}")
    return problems


def render_impact_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable line, plus the reason."""
    impact = alert.get("impact") or {}
    mark = "FIRED" if alert.get("fired") else "quiet"
    return [
        f"  {str(alert.get('ticker') or '?'):8s} "
        f"{str(alert.get('event_type') or '—'):18s} "
        f"{str(alert.get('verdict')):15s} {mark:6s} "
        f"median {_pct(impact.get('median_move'))}  "
        f"p90 {_pct(impact.get('p90_move'))}  "
        f"n={impact.get('analogs', '—')}",
        f"      {alert.get('reason')}",
    ]


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"
