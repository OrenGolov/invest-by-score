"""D3 the event context panel — what happened last time, and how much of it was the event.

The roadmap asks for "recent event, classification, historical analog count,
median historical response, current setup similarity". Every one of those has a
producer in E4/E6, so this is mostly assembly. What it must not do is quote the
median response as though it were a clean number.

**THE DECIDING MEASUREMENT: a third of comparable setups are not comparable for
the reason the panel implies.** E5 classifies each remembered response as
`event_associated` or `confounded`. MEASURED across 40 seeded analog sets,
excluding the confounded ones shifts the median historical response by **+1.36
percentage points** (median shift), up to **5.71pp**, and by more than one
percentage point in **24 of 40 cases**.

So "median historical response: +2.7%" is a composite of the event's own
association and whatever else moved those names. The median is reported WITH
the event-associated share beside it, always, and below
`EVENT_CONTEXT_ASSOCIATION_CAUTION` it carries an explicit caution. The
reference set sits at 0.667 — a third of the evidence attributed elsewhere.

**"Too few analogs" is not "no effect".** E6 refuses to summarise below
`EVENT_MEMORY_MIN_ANALOGS` because a typical response from that few examples is
not typical of anything, and when it refuses it emits no `median_response` key
at all rather than a zero. The panel preserves that: an insufficient set has no
median, not a median of 0.0.

**"No event" is not "too few analogs" either.** Nothing happened and we could
not learn from what happened are different answers, and a reader acts
differently on each.

**Provenance travels per event.** An INFERRED event is real evidence about a
real price move; what is uncertain is *which* event produced it. E6 defaults
provenance to `""` precisely so an unlabelled memory cannot launder itself into
evidence, and the panel surfaces that rather than letting inferred read as
observed.

**Association is not causation**, and the panel says so in its own payload
rather than relying on the reader to remember.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.config import (
    EVENT_CONTEXT_ASSOCIATION_CAUTION,
    EVENT_CONTEXT_BLOCKS_TRADES,
    EVENT_CONTEXT_DISCLAIMER,
    EVENT_CONTEXT_FIELDS,
    EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE,
    EVENT_CONTEXT_SHOW_PROVENANCE,
    EVENT_CONTEXT_STATUS_INSUFFICIENT,
    EVENT_CONTEXT_STATUS_MEASURED,
    EVENT_CONTEXT_STATUS_NO_EVENT,
    EVENT_CONTEXT_STATUSES,
    EVENT_CONTEXT_VERSION,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    MEMORY_PROVENANCE_INFERRED,
)


class EventContextError(ValueError):
    """Raised when an event context request is structurally invalid."""


def _event_block(event: Mapping[str, Any] | None) -> dict:
    """The recent event and its classification, provenance stated."""
    if event is None:
        return {
            "status": EVENT_CONTEXT_STATUS_NO_EVENT,
            "event_id": None,
            "event_type": None,
            "direction": None,
            "published_time": None,
            "provenance": None,
            "inferred": None,
            "reason": "no recent event was supplied for this ticker",
        }
    if not isinstance(event, Mapping):
        raise EventContextError("the event must be a mapping")

    provenance = str(event.get("provenance") or "").strip()
    inferred = provenance == MEMORY_PROVENANCE_INFERRED
    block = {
        "status": EVENT_CONTEXT_STATUS_MEASURED,
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "direction": event.get("direction"),
        "published_time": event.get("published_time"),
        "actor": event.get("actor") or None,
        "reason": "",
    }
    if EVENT_CONTEXT_SHOW_PROVENANCE:
        block["provenance"] = provenance or None
        block["inferred"] = inferred
        if not provenance:
            # AN UNLABELLED EVENT IS NOT AN OBSERVED ONE. E6 defaults
            # provenance to "" so a memory cannot launder itself into
            # evidence; the panel refuses to fill that gap either.
            block["provenance_note"] = (
                "this event does not state whether it was OBSERVED from a "
                "source or INFERRED from price behaviour; it is not treated "
                "as observed"
            )
        elif inferred:
            block["provenance_note"] = (
                "INFERRED: a real price move was dated to this event by "
                "method, not reported by a source. WHICH event produced the "
                "move is uncertain"
            )
    return block


def _analog_block(summary: Mapping[str, Any] | None) -> dict:
    """Analog count, median response and similarity — or the refusal.

    The median is never synthesised. When E6 reports `insufficient_analogs` it
    emits no `median_response` key, and the panel keeps it absent rather than
    supplying a zero that would read as "no historical move".
    """
    if summary is None:
        return {
            "status": EVENT_CONTEXT_STATUS_INSUFFICIENT,
            "analog_count": 0,
            "required": EVENT_MEMORY_MIN_ANALOGS,
            "median_response": None,
            "setup_similarity": None,
            "event_associated_share": None,
            "reason": "no analog summary was supplied to the panel",
        }
    if not isinstance(summary, Mapping):
        raise EventContextError("the analog summary must be a mapping")

    status = summary.get("status")
    if status not in EVENT_CONTEXT_STATUSES:
        raise EventContextError(f"unknown analog status {status!r}")

    if status != EVENT_CONTEXT_STATUS_MEASURED:
        return {
            "status": status,
            "analog_count": int(summary.get("analogs") or 0),
            "required": int(summary.get("required") or EVENT_MEMORY_MIN_ANALOGS),
            "median_response": None,
            "setup_similarity": None,
            "event_associated_share": None,
            "reason": str(summary.get("detail") or "").strip()
            or (
                f"fewer than {EVENT_MEMORY_MIN_ANALOGS} comparable setups; a "
                f"typical response from this few examples is not typical of "
                f"anything"
            ),
        }

    median = summary.get("median_response")
    share = summary.get("event_associated_share")

    # THE MEDIAN NEVER TRAVELS ALONE. MEASURED, excluding confounded analogs
    # moves it by more than a percentage point in 24 of 40 cases.
    if EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE and median is not None and share is None:
        raise EventContextError(
            "a median historical response was supplied without an "
            "event-associated share; MEASURED, excluding E5-confounded "
            "analogs shifts that median by more than a percentage point in "
            "24 of 40 cases, so the median alone overstates what the event "
            "explains"
        )

    block = {
        "status": EVENT_CONTEXT_STATUS_MEASURED,
        "analog_count": int(summary.get("analogs") or 0),
        "required": EVENT_MEMORY_MIN_ANALOGS,
        "median_response": None if median is None else round(float(median), 6),
        "mean_response": (
            None
            if summary.get("mean_response") is None
            else round(float(summary["mean_response"]), 6)
        ),
        "dispersion": (
            None
            if summary.get("dispersion") is None
            else round(float(summary["dispersion"]), 6)
        ),
        "positive_share": summary.get("positive_share"),
        "setup_similarity": summary.get("mean_similarity"),
        "min_similarity": EVENT_MEMORY_MIN_SIMILARITY,
        "event_associated_share": None if share is None else round(float(share), 4),
        "reason": str(summary.get("detail") or "").strip(),
    }

    if share is not None and float(share) < EVENT_CONTEXT_ASSOCIATION_CAUTION:
        block["caution"] = (
            f"only {float(share):.0%} of these analogs were classified "
            f"event_associated by E5; the rest were confounded, so this "
            f"median mixes the event with whatever else moved those names. "
            f"MEASURED, excluding confounded analogs shifts the median by "
            f"more than a percentage point in 24 of 40 cases"
        )
    return block


def build_event_context(
    ticker: str,
    as_of: Any,
    *,
    event: Mapping[str, Any] | None = None,
    analogs: Mapping[str, Any] | None = None,
    horizon: str = "20d",
) -> dict:
    """The event context panel, composed from E4/E5/E6 output."""
    if not str(ticker or "").strip():
        raise EventContextError("a ticker is required")

    event_block = _event_block(event)
    analog_block = _analog_block(analogs)

    return {
        "version": EVENT_CONTEXT_VERSION,
        "ticker": str(ticker).upper(),
        "as_of": str(as_of),
        "horizon": horizon,
        "field_order": list(EVENT_CONTEXT_FIELDS),
        "event": event_block,
        "analogs": analog_block,
        "headline": _headline(event_block, analog_block),
        "disclaimer": EVENT_CONTEXT_DISCLAIMER,
        "blocks_trades": EVENT_CONTEXT_BLOCKS_TRADES,
    }


def _headline(event: Mapping[str, Any], analogs: Mapping[str, Any]) -> str:
    """What a reader must know before reading the numbers."""
    if event.get("status") == EVENT_CONTEXT_STATUS_NO_EVENT:
        return "no recent event; there is nothing here to find comparables for"
    if analogs.get("status") != EVENT_CONTEXT_STATUS_MEASURED:
        return (
            f"{event.get('event_type')} on {event.get('published_time')}, but "
            f"only {analogs.get('analog_count')} comparable setup(s) against "
            f"{analogs.get('required')} required — no typical response can be "
            f"stated"
        )
    share = analogs.get("event_associated_share")
    median = analogs.get("median_response")
    base = (
        f"{event.get('event_type')} on {event.get('published_time')}; "
        f"{analogs.get('analog_count')} comparable setups, median "
        f"{float(median):+.2%}"
        if median is not None
        else f"{event.get('event_type')}; {analogs.get('analog_count')} comparable setups"
    )
    if share is not None and float(share) < EVENT_CONTEXT_ASSOCIATION_CAUTION:
        return (
            f"{base} — but only {float(share):.0%} of those were attributed "
            f"to the event itself"
        )
    return base


def context_problems(panel: Mapping[str, Any]) -> list[str]:
    """Contract check on an event context panel. Empty means clean."""
    problems: list[str] = []
    if not isinstance(panel, Mapping):
        return ["panel is not a mapping"]

    if panel.get("blocks_trades"):
        problems.append("the event context panel claims to block trades")
    disclaimer = str(panel.get("disclaimer") or "")
    if "not a forecast" not in disclaimer:
        problems.append(
            "the panel does not state that this is association, not a "
            "forecast and not causation"
        )

    event = panel.get("event") or {}
    analogs = panel.get("analogs") or {}
    if not isinstance(event, Mapping) or not isinstance(analogs, Mapping):
        return problems + ["event or analogs block is not a mapping"]

    if event.get("status") not in EVENT_CONTEXT_STATUSES:
        problems.append(f"unknown event status {event.get('status')!r}")
    if event.get("status") == EVENT_CONTEXT_STATUS_NO_EVENT:
        if not str(event.get("reason") or "").strip():
            problems.append("no event was reported without a reason")
        for field in ("event_id", "event_type", "direction"):
            if event.get(field) is not None:
                problems.append(
                    f"a no_event block carried {field}={event.get(field)!r}"
                )
    elif EVENT_CONTEXT_SHOW_PROVENANCE and "provenance" not in event:
        problems.append(
            "the event carries no provenance — an INFERRED event read as "
            "OBSERVED is a claim about the source that nobody made"
        )

    status = analogs.get("status")
    if status not in EVENT_CONTEXT_STATUSES:
        problems.append(f"unknown analog status {status!r}")
        return problems

    # THE SHAPE RULE: a median exists IFF it was measured.
    if status != EVENT_CONTEXT_STATUS_MEASURED:
        for field in ("median_response", "setup_similarity", "event_associated_share"):
            if analogs.get(field) is not None:
                problems.append(
                    f"{status} carried {field}={analogs.get(field)!r}; a "
                    f"summary that was refused has no statistics"
                )
        if not str(analogs.get("reason") or "").strip():
            problems.append(f"{status} with no reason")
        count = int(analogs.get("analog_count") or 0)
        required = int(analogs.get("required") or EVENT_MEMORY_MIN_ANALOGS)
        if count >= required and status == EVENT_CONTEXT_STATUS_INSUFFICIENT:
            problems.append(
                f"{count} analogs meet the {required} required but the "
                f"summary was refused as insufficient"
            )
    else:
        median = analogs.get("median_response")
        share = analogs.get("event_associated_share")
        if EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE and median is not None:
            if share is None:
                problems.append(
                    "a median historical response was reported with no "
                    "event-associated share. MEASURED, excluding confounded "
                    "analogs shifts that median by more than a percentage "
                    "point in 24 of 40 cases"
                )
            elif (
                float(share) < EVENT_CONTEXT_ASSOCIATION_CAUTION
                and not str(analogs.get("caution") or "").strip()
            ):
                problems.append(
                    f"only {float(share):.0%} of analogs were event-associated "
                    f"and the panel carries no caution — the median reads as "
                    f"the event's own effect"
                )
        count = int(analogs.get("analog_count") or 0)
        if count < EVENT_MEMORY_MIN_ANALOGS:
            problems.append(
                f"a measured summary was reported from {count} analogs, below "
                f"the {EVENT_MEMORY_MIN_ANALOGS} required"
            )
        similarity = analogs.get("setup_similarity")
        if similarity is not None and float(similarity) < EVENT_MEMORY_MIN_SIMILARITY:
            problems.append(
                f"mean setup similarity {float(similarity):.4f} is below the "
                f"{EVENT_MEMORY_MIN_SIMILARITY} retrieval bar, so these are "
                f"not comparable setups"
            )

    if not str(panel.get("headline") or "").strip():
        problems.append("the panel has no headline")
    return problems


def render_context(panel: Mapping[str, Any]) -> list[str]:
    """Human-readable lines: the headline, the event, then the analogs."""
    lines = [f"  {panel.get('headline')}"]
    event = panel.get("event") or {}
    if event.get("status") != EVENT_CONTEXT_STATUS_NO_EVENT:
        provenance = event.get("provenance") or "UNSTATED"
        lines.append(
            f"      event      {str(event.get('event_type')):22s} "
            f"{str(event.get('direction') or '—'):10s} "
            f"{str(event.get('published_time') or '—'):12s} [{provenance}]"
        )
    analogs = panel.get("analogs") or {}
    lines.append(
        f"      analogs    {analogs.get('analog_count')} "
        f"(min {analogs.get('required')})   "
        f"median {_pct(analogs.get('median_response'))}   "
        f"similarity {_num(analogs.get('setup_similarity'))}   "
        f"associated {_pct(analogs.get('event_associated_share'), share=True)}"
    )
    caution = analogs.get("caution")
    if caution:
        lines.append(f"      !! {caution[:120]}")
    return lines


def _pct(value: Any, share: bool = False) -> str:
    if value is None:
        return "—"
    return f"{float(value):.0%}" if share else f"{float(value):+.2%}"


def _num(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"
