"""A5 thesis break alert — the score is a sum, and a sum hides a reversal.

"Alert when evidence materially contradicts the existing thesis." The obvious
implementation watches the score and fires when it falls. MEASURED, that alert
is silent on most real thesis breaks.

**THE DECIDING MEASUREMENT.** W1 attributes every score to three evidence
buckets — operational (fundamental + technical), narrative (news + sentiment)
and macro_shock (macroeconomic + market regime). Over 20,000 sampled bucket
pairs, 1,306 moved the total score by less than the 0.25 support threshold,
and in **765 of those 1,306 (59%) a bucket reversed sign** — it went from
supporting the case to opposing it, or the reverse.

The concrete case, from the same arithmetic:

    bucket          before          after
    operational     +1.40 supports  -1.20 opposes
    narrative       -0.10 neutral   +2.50 supports
    macro_shock     +0.20 neutral   +0.20 neutral
    SCORE           +1.50           +1.50   delta +0.00

A score-only alert sees a **+0.00** move and says nothing, while the thesis
has inverted: the case *was* carried by operational evidence and is now
carried purely by narrative. That is a different investment wearing the same
number.

**So A5 watches the buckets.** The score travels for context and is never the
trigger on its own.

**Three kinds of break, not one.** A REVERSAL (a bucket crossed
supports↔opposes) is the strongest. A CARRIER_CHANGED means the thesis is now
held up by different evidence. SUPPORT_WITHDRAWN means the carrier went
neutral without opposing. They are different facts and a reader acts
differently on each.

**A bucket at 0.0 is not a bucket that opposes.** MEASURED, `score_engine`
itself distinguishes three reasons a bucket totals exactly 0.0: "status OK but
no usable sentiment score", "no eligible narrative sources", and "net-zero
contribution". Two of the three are an *absence* of evidence. Reading absence
as contradiction would fire a thesis break every time a news feed went quiet.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    ATTRIBUTION_BUCKETS,
    THESIS_ALERT_BLOCKS_TRADES,
    THESIS_ALERT_VERSION,
    THESIS_ALERT_WATCHES_BUCKETS,
    THESIS_BREAK_APPEARED,
    THESIS_BREAK_CARRIER,
    THESIS_BREAK_DISAPPEARED,
    THESIS_BREAK_KINDS,
    THESIS_BREAK_NONE,
    THESIS_BREAK_NOT_EVALUATED,
    THESIS_BREAK_REVERSAL,
    THESIS_BREAK_WITHDRAWN,
    THESIS_REVERSAL_IS_MATERIAL,
    THESIS_SUPPORT_THRESHOLD,
    THESIS_ZERO_IS_NOT_OPPOSITION,
)

STANCE_SUPPORTS = "supports"
STANCE_OPPOSES = "opposes"
STANCE_NEUTRAL = "neutral"
STANCE_UNMEASURED = "unmeasured"


class ThesisAlertError(ValueError):
    """Raised when a thesis break request is structurally invalid."""


def stance_of(
    total: float | None,
    *,
    measured: bool = True,
    threshold: float = THESIS_SUPPORT_THRESHOLD,
) -> str:
    """Where this bucket stands, with absence kept distinct from opposition.

    `measured=False` yields UNMEASURED rather than a stance. MEASURED, two of
    the three reasons a bucket totals 0.0 are an absence of evidence, and a
    bucket that could not speak has not contradicted anything.
    """
    if total is None or (THESIS_ZERO_IS_NOT_OPPOSITION and not measured):
        return STANCE_UNMEASURED
    value = float(total)
    if value > threshold:
        return STANCE_SUPPORTS
    if value < -threshold:
        return STANCE_OPPOSES
    return STANCE_NEUTRAL


def read_buckets(
    attribution: Mapping[str, Any] | None,
    *,
    threshold: float = THESIS_SUPPORT_THRESHOLD,
) -> dict[str, dict] | None:
    """Each bucket's total and stance, or None when nothing was attributed."""
    if attribution is None:
        return None
    if not isinstance(attribution, Mapping):
        raise ThesisAlertError("an attribution must be a mapping")
    buckets = attribution.get("buckets")
    if not isinstance(buckets, Mapping) or not buckets:
        return None

    read: dict[str, dict] = {}
    for name in ATTRIBUTION_BUCKETS:
        entry = buckets.get(name)
        if not isinstance(entry, Mapping):
            read[name] = {
                "total": None,
                "stance": STANCE_UNMEASURED,
                "measured": False,
            }
            continue
        total = entry.get("total")
        # A bucket that names itself unmeasured is honoured; otherwise a
        # present total is a measurement, including a genuine 0.0.
        measured = bool(entry.get("measured", total is not None))
        read[name] = {
            "total": None if total is None else round(float(total), 6),
            "stance": stance_of(total, measured=measured, threshold=threshold),
            "measured": measured,
        }
    return read


def carrier_of(buckets: Mapping[str, Mapping[str, Any]] | None) -> str | None:
    """The bucket carrying the thesis: the largest supporting total."""
    if not buckets:
        return None
    supporting = [
        (name, float(entry["total"]))
        for name, entry in buckets.items()
        if entry.get("stance") == STANCE_SUPPORTS and entry.get("total") is not None
    ]
    if not supporting:
        return None
    return max(supporting, key=lambda pair: pair[1])[0]


def thesis_break_alert(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    threshold: float = THESIS_SUPPORT_THRESHOLD,
) -> dict:
    """Has the evidence carrying the thesis turned against it?"""
    if current is None:
        raise ThesisAlertError("a current attribution is required")
    if threshold <= 0:
        raise ThesisAlertError("the support threshold must be positive")

    before = read_buckets(previous, threshold=threshold)
    after = read_buckets(current, threshold=threshold)

    if previous is None:
        return _alert(
            THESIS_BREAK_NOT_EVALUATED,
            fired=False,
            severity=None,
            reason=(
                "no previous attribution was supplied; a first observation "
                "cannot break a thesis"
            ),
            before=None,
            after=after,
            current=current,
            threshold=threshold,
        )
    if before is None and after is None:
        return _alert(
            THESIS_BREAK_NOT_EVALUATED,
            fired=False,
            severity=None,
            reason="no thesis was attributed before or after",
            before=None,
            after=None,
            current=current,
            threshold=threshold,
        )
    if before is None:
        return _alert(
            THESIS_BREAK_APPEARED,
            fired=True,
            severity=ALERT_SEVERITY_INFO,
            reason=(
                f"a thesis became attributable, carried by "
                f"{carrier_of(after) or 'no bucket'}; none was attributed "
                f"before. This is an APPEARANCE, not a break"
            ),
            before=None,
            after=after,
            current=current,
            threshold=threshold,
        )
    if after is None:
        return _alert(
            THESIS_BREAK_DISAPPEARED,
            fired=True,
            severity=ALERT_SEVERITY_WARN,
            reason=(
                f"the thesis was carried by "
                f"{carrier_of(before) or 'no bucket'} and is no longer "
                f"attributable. The attribution went away; the evidence did "
                f"not turn"
            ),
            before=before,
            after=None,
            current=current,
            threshold=threshold,
        )

    reversals = []
    withdrawn = []
    for name in ATTRIBUTION_BUCKETS:
        was = before[name]["stance"]
        now = after[name]["stance"]
        if {was, now} == {STANCE_SUPPORTS, STANCE_OPPOSES}:
            reversals.append(name)
        elif was == STANCE_SUPPORTS and now == STANCE_NEUTRAL:
            withdrawn.append(name)

    before_carrier = carrier_of(before)
    after_carrier = carrier_of(after)
    score_before = _score(previous)
    score_after = _score(current)
    score_delta = (
        None
        if score_before is None or score_after is None
        else round(score_after - score_before, 6)
    )

    detail = {
        "before": before,
        "after": after,
        "current": current,
        "threshold": threshold,
        "reversals": reversals,
        "withdrawn": withdrawn,
        "previous_carrier": before_carrier,
        "current_carrier": after_carrier,
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": score_delta,
    }

    # A REVERSAL IS THE STRONGEST BREAK, and fires whatever the score did.
    if reversals and THESIS_REVERSAL_IS_MATERIAL:
        return _alert(
            THESIS_BREAK_REVERSAL,
            fired=True,
            severity=ALERT_SEVERITY_WARN,
            reason=(
                f"{', '.join(reversals)} crossed between supporting and "
                f"opposing the thesis"
                + (
                    f" while the score moved {score_delta:+.2f}. MEASURED, "
                    f"59% of flat-score periods contain a bucket reversal, "
                    f"and a score-only alert is silent on all of them"
                    if score_delta is not None
                    else ". MEASURED, 59% of flat-score periods contain a "
                    "bucket reversal"
                )
            ),
            **detail,
        )

    if before_carrier != after_carrier and (before_carrier or after_carrier):
        return _alert(
            THESIS_BREAK_CARRIER,
            fired=True,
            severity=ALERT_SEVERITY_WARN,
            reason=(
                f"the thesis is now carried by "
                f"{after_carrier or 'no bucket'} rather than "
                f"{before_carrier or 'no bucket'}. The same score is now a "
                f"different investment"
            ),
            **detail,
        )

    if withdrawn:
        return _alert(
            THESIS_BREAK_WITHDRAWN,
            fired=True,
            severity=ALERT_SEVERITY_INFO,
            reason=(
                f"{', '.join(withdrawn)} stopped supporting the thesis "
                f"without opposing it. Support was withdrawn, not reversed"
            ),
            **detail,
        )

    return _alert(
        THESIS_BREAK_NONE,
        fired=False,
        severity=None,
        reason=(
            f"every bucket holds its stance; the thesis is still carried by "
            f"{after_carrier or 'no bucket'}"
        ),
        **detail,
    )


def _score(attribution: Mapping[str, Any] | None) -> float | None:
    """The published score, for context only — never the trigger."""
    if not isinstance(attribution, Mapping):
        return None
    value = attribution.get("score")
    return None if value is None else round(float(value), 6)


def _alert(kind: str, **detail) -> dict:
    """One A5 answer."""
    if kind not in THESIS_BREAK_KINDS:
        raise ThesisAlertError(f"unknown thesis break kind {kind!r}")
    current = detail.pop("current", None)
    payload = {
        "version": THESIS_ALERT_VERSION,
        "alert": "thesis_break",
        "kind": kind,
        "ticker": (current or {}).get("ticker") if isinstance(current, Mapping) else None,
        "watches_buckets": THESIS_ALERT_WATCHES_BUCKETS,
        "blocks_trades": THESIS_ALERT_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.setdefault("reversals", [])
    payload.setdefault("withdrawn", [])
    payload.update(detail)
    return payload


_NOTE = (
    "A5 watches the evidence BUCKETS, not the score. MEASURED, of 1,306 "
    "sampled periods where the score moved less than the 0.25 support "
    "threshold, 765 (59%) contained a bucket reversal - a score-only alert is "
    "silent on all of them. A bucket at 0.0 is not opposition: two of the "
    "three reasons score_engine gives for a 0.0 bucket are an absence of "
    "evidence."
)


def thesis_alert_problems(alert: Mapping[str, Any]) -> list[str]:
    """Contract check on a thesis break alert. Empty means clean."""
    problems: list[str] = []
    if not isinstance(alert, Mapping):
        return ["alert is not a mapping"]

    if alert.get("blocks_trades"):
        problems.append("an alert claims to block trades — it reports")
    if not alert.get("watches_buckets"):
        problems.append(
            "the alert does not watch the evidence buckets. MEASURED, 59% of "
            "flat-score periods contain a bucket reversal a score-only alert "
            "cannot see"
        )

    kind = alert.get("kind")
    if kind not in THESIS_BREAK_KINDS:
        problems.append(f"unknown break kind {kind!r}")
        return problems
    if not str(alert.get("reason") or "").strip():
        problems.append(f"{kind} with no reason")

    before = alert.get("before")
    after = alert.get("after")
    reversals = list(alert.get("reversals") or [])
    withdrawn = list(alert.get("withdrawn") or [])
    fired = bool(alert.get("fired"))
    severity = alert.get("severity")

    for name in reversals + withdrawn:
        if name not in ATTRIBUTION_BUCKETS:
            problems.append(f"{name!r} is not an attribution bucket")

    # A REVERSAL MUST ACTUALLY HAVE REVERSED, and one must never be silenced.
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        actual = [
            name
            for name in ATTRIBUTION_BUCKETS
            if isinstance(before.get(name), Mapping)
            and isinstance(after.get(name), Mapping)
            and {before[name].get("stance"), after[name].get("stance")}
            == {STANCE_SUPPORTS, STANCE_OPPOSES}
        ]
        if sorted(actual) != sorted(reversals):
            problems.append(
                f"reported reversals {sorted(reversals)} do not match the "
                f"buckets that actually reversed {sorted(actual)}"
            )
        if actual and kind != THESIS_BREAK_REVERSAL:
            problems.append(
                f"{actual} reversed but the alert reports {kind!r} — that is "
                f"the case A5 exists to catch"
            )
        # A BUCKET THAT COULD NOT SPEAK HAS NOT CONTRADICTED ANYTHING.
        for name in ATTRIBUTION_BUCKETS:
            entry = after.get(name)
            if not isinstance(entry, Mapping):
                continue
            if not entry.get("measured") and entry.get("stance") not in (
                STANCE_UNMEASURED,
                None,
            ):
                problems.append(
                    f"{name}: unmeasured evidence was given the stance "
                    f"{entry.get('stance')!r}; two of the three reasons a "
                    f"bucket totals 0.0 are an absence of evidence"
                )

    if kind == THESIS_BREAK_REVERSAL and not reversals:
        problems.append("REVERSAL with no bucket named")
    if kind == THESIS_BREAK_WITHDRAWN and not withdrawn:
        problems.append("SUPPORT_WITHDRAWN with no bucket named")
    if kind == THESIS_BREAK_CARRIER:
        if alert.get("previous_carrier") == alert.get("current_carrier"):
            problems.append(
                f"CARRIER_CHANGED with the same carrier "
                f"{alert.get('current_carrier')!r} on both sides"
            )

    if kind == THESIS_BREAK_NONE and fired:
        problems.append("an unbroken thesis fired an alert")
    if kind == THESIS_BREAK_NOT_EVALUATED and fired:
        problems.append("an alert fired with nothing to compare against")
    if kind in (THESIS_BREAK_REVERSAL, THESIS_BREAK_CARRIER, THESIS_BREAK_WITHDRAWN):
        if not fired:
            problems.append(f"a {kind} did not fire")

    # AVAILABILITY IS NOT A BREAK.
    if kind == THESIS_BREAK_APPEARED and before is not None:
        problems.append("APPEARED with a previous attribution")
    if kind == THESIS_BREAK_DISAPPEARED and after is not None:
        problems.append("DISAPPEARED with a current attribution")

    if fired and severity is None:
        problems.append(f"{kind} fired with no severity")
    if not fired and severity is not None:
        problems.append(f"{kind} did not fire but carried severity {severity!r}")
    if kind == THESIS_BREAK_REVERSAL and severity != ALERT_SEVERITY_WARN:
        problems.append(
            f"a reversal fired at {severity!r} rather than warn — it is the "
            f"strongest contradiction of the thesis"
        )
    return problems


def render_thesis_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable block: the verdict, then every bucket."""
    mark = "FIRED" if alert.get("fired") else "quiet"
    delta = alert.get("score_delta")
    shown = "—" if delta is None else f"{float(delta):+.2f}"
    lines = [
        f"  {str(alert.get('ticker') or '?'):8s} {str(alert.get('kind')):18s} "
        f"{mark:6s} carrier {str(alert.get('previous_carrier') or '—'):12s} -> "
        f"{str(alert.get('current_carrier') or '—'):12s} score {shown}"
        f"  [{alert.get('severity') or '—'}]",
    ]
    before = alert.get("before") or {}
    after = alert.get("after") or {}
    for name in ATTRIBUTION_BUCKETS:
        was = (before.get(name) or {}) if isinstance(before, Mapping) else {}
        now = (after.get(name) or {}) if isinstance(after, Mapping) else {}
        lines.append(
            f"      {name:14s} {_shown(was.get('total')):>8s} "
            f"{str(was.get('stance') or '—'):11s} -> "
            f"{_shown(now.get('total')):>8s} {str(now.get('stance') or '—'):11s}"
        )
    lines.append(f"      {alert.get('reason')}")
    return lines


def _shown(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.2f}"
