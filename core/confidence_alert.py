"""A2 confidence change alert — the number, the band, and the reason.

"Alert when confidence changes materially." A2 is not A1 with a different
field, and the difference was measured rather than assumed.

**Confidence moves when the forecast does not.** MEASURED: holding P(up) fixed
at 0.56 and dropping feature completeness from 10/10 to 6/10 moved confidence
−0.0146 while the forecast value did not move at all. A1 would never fire on
that, and should not — the forecast is unchanged. What changed is how much it
can be trusted.

**THE DECIDING MEASUREMENT: a magnitude-only alert is blind half the time.**
F7's confidence is bounded by its *weakest* factor and always reports which one
bound it. Over 3,000 sampled assessment pairs, 1,170 moved confidence by less
than 0.10 — below any sane threshold — and in **600 of those 1,170 (51%)** the
binding factor changed completely.

A concrete instance: confidence 0.5559 → 0.4640, a −0.0919 move no threshold
would fire on, while the binding factor went from `event_similarity` to
`regime_similarity`. The forecast became uncertain for an entirely different
reason, and a magnitude-only alert says nothing. "We are unsure because the
sample is small" and "we are unsure because no comparable regime exists" are
different problems with different remedies.

**So A2 watches three things: the magnitude, the BAND, and the BINDING FACTOR.**
A band crossing is material at any magnitude — MEASURED, the same 0.10 delta
crosses a band at 0.45→0.55 but not at 0.85→0.95, and the band is what a reader
acts on.

**UNMEASURABLE is not low**, inherited from F7 and R4. A factor becoming
unmeasurable is not a confidence drop, and one becoming measurable is not a
rise: the set of things being measured changed, which is its own event.

**Availability is first-class**, for A1's reason: a confidence appearing or
disappearing is not a move of its own magnitude.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL,
    CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL,
    CONFIDENCE_ALERT_BLOCKS_TRADES,
    CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT,
    CONFIDENCE_ALERT_MIN_MOVE,
    CONFIDENCE_ALERT_VERSION,
    CONFIDENCE_CHANGE_KINDS,
    CONF_CHANGE_APPEARED,
    CONF_CHANGE_BAND,
    CONF_CHANGE_BINDING,
    CONF_CHANGE_DISAPPEARED,
    CONF_CHANGE_MOVED,
    CONF_CHANGE_NONE,
    CONF_CHANGE_NOT_EVALUATED,
    FCONF_STATUS_MEASURED,
)


class ConfidenceAlertError(ValueError):
    """Raised when a confidence change request is structurally invalid."""


def _read(assessment: Mapping[str, Any] | None) -> dict:
    """The three things A2 compares, or None for each that was not measured."""
    if assessment is None:
        return {"confidence": None, "band": None, "binding": None, "measured": None}
    if not isinstance(assessment, Mapping):
        raise ConfidenceAlertError("a confidence assessment must be a mapping")

    value = assessment.get("confidence")
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ConfidenceAlertError(
                f"confidence {value!r} is not a number"
            ) from None
        if not 0.0 <= value <= 1.0:
            raise ConfidenceAlertError(f"confidence {value!r} lies outside [0, 1]")

    measured = assessment.get("measured")
    return {
        "confidence": value,
        "band": assessment.get("band"),
        "binding": assessment.get("binding_factor"),
        "measured": sorted(measured) if measured is not None else None,
    }


def classify_confidence_change(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    threshold: float = CONFIDENCE_ALERT_MIN_MOVE,
) -> dict:
    """Which kind of confidence change this is, and why.

    Availability is decided first, then the binding factor, then the band, then
    the magnitude. The order matters: the binding factor and the band are
    material at ANY magnitude, so checking magnitude first would swallow them.
    """
    if not 0.0 < threshold < 1.0:
        raise ConfidenceAlertError("the threshold must lie inside (0, 1)")

    before = _read(previous)
    after = _read(current)

    if before["confidence"] is None and after["confidence"] is None:
        return {
            "kind": CONF_CHANGE_NOT_EVALUATED,
            "delta": None,
            "reason": "confidence was unmeasured before and after",
        }
    if before["confidence"] is None:
        return {
            "kind": CONF_CHANGE_APPEARED,
            "delta": None,
            "reason": (
                f"confidence became measurable at {after['confidence']:.4f} "
                f"({after['band']}); it was not assessed before. This is an "
                f"APPEARANCE, not a rise from zero"
            ),
        }
    if after["confidence"] is None:
        return {
            "kind": CONF_CHANGE_DISAPPEARED,
            "delta": None,
            "reason": (
                f"confidence was {before['confidence']:.4f} "
                f"({before['band']}) and is no longer assessed. The "
                f"assessment went away; confidence did not fall to zero"
            ),
        }

    delta = round(after["confidence"] - before["confidence"], 8)

    # THE BINDING FACTOR FIRST. MEASURED, 51% of sub-threshold moves change it,
    # and a magnitude check would swallow every one of them.
    if (
        CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL
        and before["binding"]
        and after["binding"]
        and before["binding"] != after["binding"]
    ):
        return {
            "kind": CONF_CHANGE_BINDING,
            "delta": delta,
            "reason": (
                f"the forecast is now limited by {after['binding']} rather "
                f"than {before['binding']} (confidence {delta:+.4f}). "
                f"MEASURED, 51% of sub-threshold confidence moves change "
                f"which factor binds, and the remedy differs by factor"
            ),
        }

    if (
        CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL
        and before["band"]
        and after["band"]
        and before["band"] != after["band"]
    ):
        return {
            "kind": CONF_CHANGE_BAND,
            "delta": delta,
            "reason": (
                f"{before['band']} -> {after['band']} (confidence "
                f"{delta:+.4f}). The band is what a reader acts on, so a "
                f"crossing is news even when the number barely moved"
            ),
        }

    if abs(delta) >= threshold:
        return {
            "kind": CONF_CHANGE_MOVED,
            "delta": delta,
            "reason": (
                f"{before['confidence']:.4f} -> {after['confidence']:.4f} "
                f"({delta:+.4f}), at or beyond the {threshold} threshold, "
                f"without changing band or binding factor"
            ),
        }

    return {
        "kind": CONF_CHANGE_NONE,
        "delta": delta,
        "reason": (
            f"{before['confidence']:.4f} -> {after['confidence']:.4f} "
            f"({delta:+.4f}); same band ({after['band']}), same binding "
            f"factor ({after['binding']})"
        ),
    }


def confidence_change_alert(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    threshold: float = CONFIDENCE_ALERT_MIN_MOVE,
) -> dict:
    """Did confidence materially change between two F7 assessments?"""
    if current is None:
        raise ConfidenceAlertError("a current assessment is required")

    before = _read(previous)
    after = _read(current)

    if previous is None:
        classified = {
            "kind": CONF_CHANGE_NOT_EVALUATED,
            "delta": None,
            "reason": (
                "no previous assessment was supplied; a first observation is "
                "not a change"
            ),
        }
    else:
        classified = classify_confidence_change(
            previous, current, threshold=threshold
        )

    kind = classified["kind"]
    fired, severity = _verdict(kind, classified["delta"])

    # THE SET OF MEASURABLE FACTORS IS ITS OWN EVENT, reported alongside rather
    # than folded into the magnitude.
    measurability = None
    if (
        CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT
        and before["measured"] is not None
        and after["measured"] is not None
        and before["measured"] != after["measured"]
    ):
        gained = sorted(set(after["measured"]) - set(before["measured"]))
        lost = sorted(set(before["measured"]) - set(after["measured"]))
        measurability = {
            "gained": gained,
            "lost": lost,
            "note": (
                "the SET of measurable factors changed. A factor becoming "
                "UNMEASURABLE is not a confidence drop: F7 excludes it from "
                "the weighting rather than scoring it 0.0"
            ),
        }

    return {
        "version": CONFIDENCE_ALERT_VERSION,
        "alert": "confidence_change",
        "ticker": (current or {}).get("ticker"),
        "as_of": (current or {}).get("as_of"),
        "horizon": (current or {}).get("horizon"),
        "kind": kind,
        "fired": fired,
        "severity": severity,
        "previous": before["confidence"],
        "current": after["confidence"],
        "delta": classified["delta"],
        "previous_band": before["band"],
        "current_band": after["band"],
        "previous_binding": before["binding"],
        "current_binding": after["binding"],
        "measurability": measurability,
        "threshold": threshold,
        "reason": classified["reason"],
        "watches_binding_factor": CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL,
        "blocks_trades": CONFIDENCE_ALERT_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _verdict(kind: str, delta: float | None) -> tuple[bool, str | None]:
    """Whether this fires, and at what severity."""
    if kind in (CONF_CHANGE_NONE, CONF_CHANGE_NOT_EVALUATED):
        return False, None
    if kind == CONF_CHANGE_APPEARED:
        return True, ALERT_SEVERITY_INFO
    if kind == CONF_CHANGE_DISAPPEARED:
        return True, ALERT_SEVERITY_WARN
    if kind == CONF_CHANGE_BINDING:
        return True, ALERT_SEVERITY_INFO if (delta or 0) >= 0 else ALERT_SEVERITY_WARN
    if kind == CONF_CHANGE_BAND:
        return True, ALERT_SEVERITY_INFO if (delta or 0) > 0 else ALERT_SEVERITY_WARN
    return True, ALERT_SEVERITY_INFO if (delta or 0) > 0 else ALERT_SEVERITY_WARN


_NOTE = (
    "A2 watches the magnitude, the BAND and the BINDING FACTOR. MEASURED, of "
    "1,170 assessment pairs moving confidence by less than 0.10, 600 (51%) "
    "changed which factor bound it - a magnitude-only alert is silent on all "
    "of them, and 'unsure because the sample is small' is a different problem "
    "from 'unsure because no comparable regime exists'."
)


def confidence_alert_problems(alert: Mapping[str, Any]) -> list[str]:
    """Contract check on a confidence change alert. Empty means clean."""
    problems: list[str] = []
    if not isinstance(alert, Mapping):
        return ["alert is not a mapping"]

    if alert.get("blocks_trades"):
        problems.append("an alert claims to block trades — it reports")
    if not alert.get("watches_binding_factor"):
        problems.append(
            "the alert does not watch the binding factor. MEASURED, 51% of "
            "sub-threshold confidence moves change it, and a magnitude-only "
            "alert is blind to all of them"
        )

    kind = alert.get("kind")
    if kind not in CONFIDENCE_CHANGE_KINDS:
        problems.append(f"unknown change kind {kind!r}")
        return problems
    if not str(alert.get("reason") or "").strip():
        problems.append(f"{kind} with no reason")

    previous = alert.get("previous")
    current = alert.get("current")
    delta = alert.get("delta")

    # THE SHAPE RULE: a delta exists IFF both sides were measured.
    if kind in (CONF_CHANGE_APPEARED, CONF_CHANGE_DISAPPEARED):
        if delta is not None:
            problems.append(
                f"{kind} carried a delta of {delta!r}. An availability change "
                f"has no magnitude"
            )
        if kind == CONF_CHANGE_APPEARED and previous is not None:
            problems.append("APPEARED with a previous confidence")
        if kind == CONF_CHANGE_DISAPPEARED and current is not None:
            problems.append("DISAPPEARED with a current confidence")
    elif kind != CONF_CHANGE_NOT_EVALUATED:
        if previous is None or current is None:
            problems.append(
                f"{kind} between {previous!r} and {current!r} — a comparison "
                f"needs two measured values"
            )
        elif delta is None:
            problems.append(f"{kind} with no delta")
        else:
            expected = round(float(current) - float(previous), 8)
            if abs(expected - float(delta)) > 1e-9:
                problems.append(
                    f"delta {delta} does not equal {current} - {previous}"
                )

    # A BINDING CHANGE MUST ACTUALLY HAVE CHANGED THE BINDING FACTOR.
    if kind == CONF_CHANGE_BINDING:
        if alert.get("previous_binding") == alert.get("current_binding"):
            problems.append(
                f"BINDING_CHANGED with the same factor "
                f"{alert.get('current_binding')!r} on both sides"
            )
    if kind == CONF_CHANGE_BAND:
        if alert.get("previous_band") == alert.get("current_band"):
            problems.append(
                f"BAND_CHANGED with the same band "
                f"{alert.get('current_band')!r} on both sides"
            )

    # A CHANGED BINDING FACTOR MUST NOT BE REPORTED AS NONE. This is the 51%
    # case, caught at the contract boundary as well as at the source.
    if kind == CONF_CHANGE_NONE:
        if (
            alert.get("previous_binding")
            and alert.get("current_binding")
            and alert.get("previous_binding") != alert.get("current_binding")
        ):
            problems.append(
                f"reported NONE while the binding factor moved from "
                f"{alert.get('previous_binding')!r} to "
                f"{alert.get('current_binding')!r} — that is the 51% case "
                f"this alert exists to catch"
            )
        if (
            alert.get("previous_band")
            and alert.get("current_band")
            and alert.get("previous_band") != alert.get("current_band")
        ):
            problems.append(
                f"reported NONE across a band crossing "
                f"{alert.get('previous_band')!r} -> {alert.get('current_band')!r}"
            )
        if alert.get("fired"):
            problems.append("an unchanged confidence fired an alert")

    fired = bool(alert.get("fired"))
    severity = alert.get("severity")
    if fired and severity is None:
        problems.append(f"{kind} fired with no severity")
    if not fired and severity is not None:
        problems.append(f"{kind} did not fire but carried severity {severity!r}")
    if kind == CONF_CHANGE_NOT_EVALUATED and fired:
        problems.append("an alert fired with nothing to compare against")

    measurability = alert.get("measurability")
    if measurability is not None:
        if not isinstance(measurability, Mapping):
            problems.append("measurability is not a mapping")
        elif not (measurability.get("gained") or measurability.get("lost")):
            problems.append(
                "a measurability change was reported with nothing gained or "
                "lost"
            )
    return problems


def render_confidence_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable line, plus the reason."""
    mark = "FIRED" if alert.get("fired") else "quiet"
    lines = [
        f"  {str(alert.get('ticker') or '?'):8s} {str(alert.get('kind')):16s} "
        f"{mark:6s} {_shown(alert.get('previous'))} -> "
        f"{_shown(alert.get('current'))}  "
        f"[{alert.get('previous_band') or '—'} -> "
        f"{alert.get('current_band') or '—'}]  "
        f"bound by {alert.get('current_binding') or '—'}",
        f"      {alert.get('reason')}",
    ]
    measurability = alert.get("measurability")
    if isinstance(measurability, Mapping):
        lines.append(
            f"      !! measurable factors gained "
            f"{measurability.get('gained')}, lost {measurability.get('lost')}"
        )
    return lines


def _shown(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"
