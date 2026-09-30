"""A6 forecast threshold alert — three conditions, one of which has no producer.

"E.g. 20D expected return + P(up) + confidence crossing a defined threshold,
with no veto active." Three conditions and a gate. MEASURED, one of the three
has no producer, and the two obvious ways to handle that are both wrong.

**THE DECIDING MEASUREMENT: A6 cannot fire today, at any horizon.** Across 1d,
5d, 20d and 60d, **zero of the twelve condition inputs are PRESENT** —
`expected_return` is ABSENT at every horizon because no trained model exists,
and `probability_up` and `confidence` are REFUSED because nothing supplied
them.

**THE TWO WRONG ANSWERS**, both measured rather than argued:

- *Coalesce the missing value to 0.0.* Then `0.0 >= 0.03` is False and the
  alert **never fires** — dead, and silent about being dead.
- *Skip the condition that cannot be evaluated.* Then P(up) and confidence
  alone can fire, reporting a "threshold crossing" on two of three conditions
  and silently weakening the rule it claims to enforce.

So an unevaluable condition produces **NOT_EVALUATED**, which is neither fired
nor quiet. A reader learns the alert is blind rather than inferring calm from
its silence.

**W2's fail-closed rule does not transfer wholesale.** Blocking on missing
evidence is right for a *veto*; an *alert* that fires on absent data is pure
noise. The one place A6 does inherit it is governance: an **unknown veto state
is not "no veto active"**, because claiming governance passed when nobody
asked is how a blocked trade gets recommended.

**Every threshold is derived or reused, none invented.** P(up) 0.60 is the
first tenth clearing a coin flip by more than A1's measured ±0.098 sampling
band. The confidence bar is read from F7's own band table, so it tracks
MODERATE rather than drifting from it.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    ALERT_SEVERITY_WARN,
    FORECAST_THRESHOLD_ALERT_VERSION,
    FTHRESHOLD_BLOCKS_TRADES,
    FTHRESHOLD_COERCES_MISSING,
    FTHRESHOLD_CONDITION_CONFIDENCE,
    FTHRESHOLD_CONDITION_PROBABILITY,
    FTHRESHOLD_CONDITION_RETURN,
    FTHRESHOLD_CONDITIONS,
    FTHRESHOLD_FIRED,
    FTHRESHOLD_HORIZON,
    FTHRESHOLD_MIN_CONFIDENCE,
    FTHRESHOLD_MIN_PROBABILITY,
    FTHRESHOLD_MIN_RETURN,
    FTHRESHOLD_NOT_EVALUATED,
    FTHRESHOLD_NOT_MET,
    FTHRESHOLD_REQUIRES_ALL_CONDITIONS,
    FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES,
    FTHRESHOLD_VERDICTS,
    FTHRESHOLD_VETOED,
    SNAPSHOT_STATUS_PRESENT,
)


class ForecastThresholdError(ValueError):
    """Raised when a threshold request is structurally invalid."""


DEFAULT_THRESHOLDS: dict[str, float] = {
    FTHRESHOLD_CONDITION_RETURN: FTHRESHOLD_MIN_RETURN,
    FTHRESHOLD_CONDITION_PROBABILITY: FTHRESHOLD_MIN_PROBABILITY,
    FTHRESHOLD_CONDITION_CONFIDENCE: FTHRESHOLD_MIN_CONFIDENCE,
}


def _measured(snapshot: Mapping[str, Any], field: str) -> tuple[float | None, str]:
    """A field's value IFF it was measured, plus the status that explains it.

    Never coerces. MEASURED, coalescing to 0.0 makes the alert permanently and
    silently dead, because 0.0 never clears a positive threshold.
    """
    cell = (snapshot.get("fields") or {}).get(field)
    if not isinstance(cell, Mapping):
        return None, "MISSING"
    status = str(cell.get("status") or "MISSING")
    if status != SNAPSHOT_STATUS_PRESENT:
        return None, status
    value = cell.get("value")
    if value is None:
        return None, "PRESENT_WITHOUT_VALUE"
    # The confidence field carries F7's WHOLE assessment, not a scalar — the
    # same shape D1's renderer had to handle. The scalar is read from the
    # field that names the quantity, never guessed from the first number in
    # the dict, so a restructured assessment fails loudly instead of
    # thresholding on some unrelated factor value.
    if isinstance(value, Mapping):
        scalar = value.get(field)
        if scalar is None:
            raise ForecastThresholdError(
                f"{field} is PRESENT as a mapping with no {field!r} key; the "
                f"scalar to threshold on cannot be identified"
            )
        value = scalar
    try:
        return float(value), status
    except (TypeError, ValueError):
        raise ForecastThresholdError(
            f"{field} is PRESENT with a non-numeric value {value!r}"
        ) from None


def evaluate_conditions(
    snapshot: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict]:
    """Each condition's value, bar and verdict — or why it could not be tested."""
    if not isinstance(snapshot, Mapping):
        raise ForecastThresholdError("a forecast snapshot is required")
    bars = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        bars.update(thresholds)

    results: dict[str, dict] = {}
    for condition in FTHRESHOLD_CONDITIONS:
        bar = bars.get(condition)
        if bar is None:
            raise ForecastThresholdError(f"{condition} has no threshold")
        value, status = _measured(snapshot, condition)
        if value is None:
            results[condition] = {
                "condition": condition,
                "value": None,
                "threshold": bar,
                "status": status,
                "evaluable": False,
                "met": None,
                "reason": (
                    f"{condition} is {status}, so this condition cannot be "
                    f"tested. It is not treated as 0.0 - that would make the "
                    f"alert permanently and silently dead"
                ),
            }
            continue
        met = value >= bar
        results[condition] = {
            "condition": condition,
            "value": round(value, 6),
            "threshold": bar,
            "status": status,
            "evaluable": True,
            "met": met,
            "reason": (
                f"{condition} {value:.4f} "
                f"{'clears' if met else 'is below'} the {bar} bar"
            ),
        }
    return results


def _veto_state(risk: Mapping[str, Any] | None) -> tuple[bool | None, str]:
    """Whether governance blocks, and whether that was actually established.

    None means unknown — and unknown is NOT "no veto active".
    """
    if risk is None:
        return None, "no risk policy evaluation was supplied"
    if not isinstance(risk, Mapping):
        raise ForecastThresholdError("the risk evaluation must be a mapping")
    if "veto" not in risk:
        return None, "the risk evaluation carries no veto field"
    return bool(risk.get("veto")), ""


def forecast_threshold_alert(
    snapshot: Mapping[str, Any],
    risk: Mapping[str, Any] | None = None,
    *,
    thresholds: Mapping[str, float] | None = None,
    horizon: str = FTHRESHOLD_HORIZON,
) -> dict:
    """Did the forecast cross every threshold, with governance clear?"""
    conditions = evaluate_conditions(snapshot, thresholds=thresholds)
    unevaluable = [
        name for name, entry in conditions.items() if not entry["evaluable"]
    ]
    failed = [
        name
        for name, entry in conditions.items()
        if entry["evaluable"] and not entry["met"]
    ]
    veto, veto_reason = _veto_state(risk)

    # AN UNEVALUABLE CONDITION HAS NO VERDICT. Reported before the veto,
    # because "we could not test this" outranks "governance also blocks" —
    # a reader fixing the alert needs the blindness named first.
    if unevaluable and FTHRESHOLD_REQUIRES_ALL_CONDITIONS:
        return _alert(
            FTHRESHOLD_NOT_EVALUATED,
            fired=False,
            severity=None,
            reason=(
                f"{len(unevaluable)} of {len(FTHRESHOLD_CONDITIONS)} "
                f"condition(s) could not be tested ({', '.join(unevaluable)}). "
                f"MEASURED, expected_return is ABSENT at every horizon because "
                f"no trained model exists, so this alert cannot fire today - "
                f"and says so rather than looking quiet"
            ),
            conditions=conditions,
            unevaluable=unevaluable,
            failed=failed,
            veto=veto,
            veto_reason=veto_reason,
            horizon=horizon,
            snapshot=snapshot,
        )

    if failed:
        return _alert(
            FTHRESHOLD_NOT_MET,
            fired=False,
            severity=None,
            reason=(
                f"every condition was tested and {len(failed)} did not clear "
                f"its bar ({', '.join(failed)})"
            ),
            conditions=conditions,
            unevaluable=unevaluable,
            failed=failed,
            veto=veto,
            veto_reason=veto_reason,
            horizon=horizon,
            snapshot=snapshot,
        )

    # GOVERNANCE LAST, and an unknown veto state suppresses.
    if veto is None and FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES:
        return _alert(
            FTHRESHOLD_VETOED,
            fired=False,
            severity=None,
            reason=(
                f"every threshold was crossed, but the veto state is unknown: "
                f"{veto_reason}. An unknown veto is not 'no veto active' - "
                f"claiming governance passed when nobody asked is how a "
                f"blocked trade gets recommended"
            ),
            conditions=conditions,
            unevaluable=unevaluable,
            failed=failed,
            veto=None,
            veto_reason=veto_reason,
            horizon=horizon,
            snapshot=snapshot,
        )
    if veto:
        return _alert(
            FTHRESHOLD_VETOED,
            fired=False,
            severity=None,
            reason=(
                f"every threshold was crossed, but W2 vetoes this ticker: "
                f"{list((risk or {}).get('veto_rule_ids') or []) or 'unnamed rule'}"
            ),
            conditions=conditions,
            unevaluable=unevaluable,
            failed=failed,
            veto=True,
            veto_reason=veto_reason,
            horizon=horizon,
            snapshot=snapshot,
        )

    return _alert(
        FTHRESHOLD_FIRED,
        fired=True,
        severity=ALERT_SEVERITY_WARN,
        reason=(
            f"all {len(FTHRESHOLD_CONDITIONS)} conditions cleared their bars "
            f"at {horizon} and no veto is active: "
            + "; ".join(entry["reason"] for entry in conditions.values())
        ),
        conditions=conditions,
        unevaluable=unevaluable,
        failed=failed,
        veto=False,
        veto_reason=veto_reason,
        horizon=horizon,
        snapshot=snapshot,
    )


def _alert(verdict: str, **detail) -> dict:
    """One A6 answer."""
    if verdict not in FTHRESHOLD_VERDICTS:
        raise ForecastThresholdError(f"unknown threshold verdict {verdict!r}")
    snapshot = detail.pop("snapshot", None)
    fields = (snapshot or {}).get("fields") or {}
    payload = {
        "version": FORECAST_THRESHOLD_ALERT_VERSION,
        "alert": "forecast_threshold",
        "verdict": verdict,
        "ticker": (fields.get("ticker") or {}).get("value"),
        "as_of": (fields.get("as_of") or {}).get("value"),
        "requires_all_conditions": FTHRESHOLD_REQUIRES_ALL_CONDITIONS,
        "coerces_missing": FTHRESHOLD_COERCES_MISSING,
        "blocks_trades": FTHRESHOLD_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, zero of the twelve condition inputs are PRESENT across 1d, 5d, "
    "20d and 60d - expected_return is ABSENT because no trained model exists. "
    "An unevaluable condition yields NOT_EVALUATED, never a silent quiet: "
    "coalescing to 0.0 makes the alert permanently dead, and skipping the "
    "condition fires on two of three."
)


def threshold_alert_problems(alert: Mapping[str, Any]) -> list[str]:
    """Contract check on a threshold alert. Empty means clean."""
    problems: list[str] = []
    if not isinstance(alert, Mapping):
        return ["alert is not a mapping"]

    if alert.get("blocks_trades"):
        problems.append("an alert claims to block trades — it reports")
    if alert.get("coerces_missing"):
        problems.append(
            "the alert coerces a missing input. MEASURED, coalescing "
            "expected_return to 0.0 makes it permanently and silently dead"
        )
    if not alert.get("requires_all_conditions"):
        problems.append(
            "the alert does not require every condition; skipping one turns "
            "'all three held' into 'the ones we could check held'"
        )

    verdict = alert.get("verdict")
    if verdict not in FTHRESHOLD_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")
        return problems
    if not str(alert.get("reason") or "").strip():
        problems.append(f"{verdict} with no reason")

    conditions = alert.get("conditions") or {}
    if not isinstance(conditions, Mapping):
        return problems + ["conditions is not a mapping"]
    for name in FTHRESHOLD_CONDITIONS:
        if name not in conditions:
            problems.append(
                f"condition {name!r} was not evaluated — the roadmap names it"
            )
    for name, entry in conditions.items():
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: condition is not a mapping")
            continue
        # THE SHAPE RULE: a value exists IFF it was measured.
        if entry.get("evaluable"):
            if entry.get("value") is None:
                problems.append(f"{name}: evaluable with no value")
            if entry.get("met") is None:
                problems.append(f"{name}: evaluable with no verdict")
            elif entry.get("value") is not None and entry.get("threshold") is not None:
                expected = float(entry["value"]) >= float(entry["threshold"])
                if bool(entry["met"]) != expected:
                    problems.append(
                        f"{name}: met={entry['met']} disagrees with "
                        f"{entry['value']} against {entry['threshold']}"
                    )
        else:
            if entry.get("value") is not None:
                problems.append(
                    f"{name}: unevaluable but carried the value "
                    f"{entry.get('value')!r} — a missing input coerced into a "
                    f"number is what this alert refuses"
                )
            if entry.get("met") is not None:
                problems.append(
                    f"{name}: unevaluable but carried met={entry.get('met')!r}"
                )
        if not str(entry.get("reason") or "").strip():
            problems.append(f"{name}: condition with no reason")

    unevaluable = list(alert.get("unevaluable") or [])
    actual_unevaluable = sorted(
        name
        for name, entry in conditions.items()
        if isinstance(entry, Mapping) and not entry.get("evaluable")
    )
    if sorted(unevaluable) != actual_unevaluable:
        problems.append(
            f"reported unevaluable {sorted(unevaluable)} does not match the "
            f"conditions that could not be tested {actual_unevaluable}"
        )

    fired = bool(alert.get("fired"))
    severity = alert.get("severity")

    # AN UNEVALUABLE CONDITION MUST NEVER PRODUCE A FIRED ALERT.
    if actual_unevaluable and verdict != FTHRESHOLD_NOT_EVALUATED:
        problems.append(
            f"{actual_unevaluable} could not be tested but the verdict is "
            f"{verdict!r} — that reports a threshold crossing on a condition "
            f"nobody checked"
        )
    if actual_unevaluable and fired:
        problems.append(
            f"the alert fired with {actual_unevaluable} untested"
        )

    if verdict == FTHRESHOLD_FIRED:
        if not fired:
            problems.append("FIRED did not fire")
        if alert.get("veto") is not False:
            problems.append(
                f"FIRED with veto={alert.get('veto')!r} — an unknown or active "
                f"veto must suppress"
            )
        if alert.get("failed"):
            problems.append(f"FIRED with failed conditions {alert.get('failed')}")
    else:
        if fired:
            problems.append(f"{verdict} fired")

    if verdict == FTHRESHOLD_VETOED and alert.get("veto") is False:
        problems.append("VETOED with veto=False")
    if verdict == FTHRESHOLD_NOT_MET and not alert.get("failed"):
        problems.append("NOT_MET with no failing condition named")

    if fired and severity is None:
        problems.append(f"{verdict} fired with no severity")
    if not fired and severity is not None:
        problems.append(f"{verdict} did not fire but carried severity {severity!r}")
    return problems


def render_threshold_alert(alert: Mapping[str, Any]) -> list[str]:
    """One human-readable block: the verdict, then every condition."""
    mark = "FIRED" if alert.get("fired") else "quiet"
    veto = alert.get("veto")
    shown_veto = "unknown" if veto is None else ("ACTIVE" if veto else "clear")
    lines = [
        f"  {str(alert.get('ticker') or '?'):8s} {str(alert.get('horizon')):5s} "
        f"{str(alert.get('verdict')):16s} {mark:6s} veto {shown_veto}"
        f"  [{alert.get('severity') or '—'}]",
    ]
    conditions = alert.get("conditions") or {}
    for name in FTHRESHOLD_CONDITIONS:
        entry = conditions.get(name) or {}
        value = entry.get("value")
        shown = "—" if value is None else f"{float(value):.4f}"
        met = entry.get("met")
        flag = "—" if met is None else ("ok" if met else "FAIL")
        lines.append(
            f"      {name:18s} {shown:>9s} vs {entry.get('threshold')}   "
            f"{flag:5s} {str(entry.get('status') or ''):10s}"
        )
    lines.append(f"      {alert.get('reason')}")
    return lines
