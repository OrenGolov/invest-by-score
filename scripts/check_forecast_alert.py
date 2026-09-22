"""Gate: the change alert compares values, not digests, and availability first.

"Alert when the forecast materially changes." Two obvious implementations both
fire on things that are not changes.

THE FIRST TRAP - DIGEST DIFFING, recomputed here: four consecutive days of an
identical, entirely-unavailable forecast produce FOUR DISTINCT DIGESTS, because
the digest covers as_of. A digest-diff alert fires on 100% of days while the
forecast never changes. On this system - which produces no forecast values -
every one of those alerts is about nothing.

THE SECOND TRAP - COALESCING NULLS. Comparing through float(x or 0) turns a
forecast APPEARING (REFUSED -> 0.56) into a +0.56 move and a forecast
DISAPPEARING (0.56 -> REFUSED) into a -0.56 crash. Nothing fell; the
availability changed.

THE THRESHOLD IS DERIVED. At F4's point floor of 40 observations the 95% band
on a base rate near 0.5 is +/-0.155; at 100 observations it is +/-0.098. A
threshold of 0.10 is where a move starts to mean something rather than being
resampling.

Verified to FAIL when any of these is reinjected:
  - the alert comparing snapshot digests
  - an unavailable value coerced to 0.0 before comparison
  - an APPEARED or DISAPPEARED carrying a delta
  - a move inside the threshold firing
  - a first observation reported as a change
  - a fired alert with no severity, or a quiet one carrying severity
  - the alert claiming to block trades
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
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
    FORECAST_CHANGE_KINDS,
)
from core.forecast_alert import (  # noqa: E402
    ForecastAlertError,
    alert_problems,
    classify_change,
    forecast_change_alert,
    render_alert,
)
from core.forecast_snapshot import (  # noqa: E402
    build_forecast_snapshot,
    snapshot_digest,
)

FAILURES: list[str] = []
BASE = {
    "samples": 800,
    "interval": {"low": 0.48, "high": 0.60},
    "event": {"entity": "NVDA"},
}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def snapshot(as_of, value=None, horizon="20d"):
    kwargs = {}
    if value is not None:
        kwargs["event_forecast"] = dict(BASE, value=value, as_of=as_of, horizon=horizon)
    return build_forecast_snapshot("NVDA", as_of, horizon, **kwargs)


def main() -> int:
    # 1. THE DIGEST TRAP, RECOMPUTED.
    days = ("2026-09-18", "2026-09-19", "2026-09-21", "2026-09-22")
    digests = {snapshot_digest(snapshot(d)) for d in days}
    check(
        len(digests) == len(days),
        f"{len(digests)} distinct digests across {len(days)} identical "
        f"unavailable forecasts. A1's refusal to compare digests rests on the "
        f"digest changing with as_of; if that has changed the rule must be "
        f"re-derived rather than kept",
    )
    check(
        not FORECAST_ALERT_USES_DIGEST,
        "the alert is configured to compare digests, which fires every day on "
        "an unchanged forecast",
    )
    quiet = forecast_change_alert(snapshot("2026-09-21"), snapshot("2026-09-22"))
    check(
        not quiet["fired"],
        "two consecutive days of an identical unavailable forecast fired an "
        "alert",
    )
    check(
        quiet["kind"] == ALERT_CHANGE_NOT_EVALUATED,
        f"identical unavailable forecasts produced {quiet['kind']!r}",
    )
    check(not quiet["compares_digest"], "the alert reports comparing digests")

    # 2. THE NULL-COALESCING TRAP. Availability is not magnitude.
    appeared = forecast_change_alert(
        snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
    )
    check(
        appeared["kind"] == ALERT_CHANGE_APPEARED,
        f"a forecast becoming available produced {appeared['kind']!r}",
    )
    check(
        appeared["delta"] is None,
        f"an APPEARED alert carried delta={appeared['delta']!r}; "
        f"float(x or 0) would have made this a +0.56 move",
    )
    check(appeared["previous"] is None, "APPEARED carried a previous value")

    disappeared = forecast_change_alert(
        snapshot("2026-09-21", 0.56), snapshot("2026-09-22")
    )
    check(
        disappeared["kind"] == ALERT_CHANGE_DISAPPEARED,
        f"a forecast going away produced {disappeared['kind']!r}",
    )
    check(
        disappeared["delta"] is None,
        f"a DISAPPEARED alert carried delta={disappeared['delta']!r}, which "
        f"reports a crash that never happened",
    )
    check(disappeared["current"] is None, "DISAPPEARED carried a current value")
    check(
        FORECAST_ALERT_AVAILABILITY_IS_MATERIAL
        and appeared["fired"]
        and disappeared["fired"],
        "an availability change did not fire; a forecast arriving or "
        "vanishing is news about the system",
    )
    check(
        appeared["severity"] == ALERT_SEVERITY_INFO
        and disappeared["severity"] == ALERT_SEVERITY_WARN,
        f"severities {appeared['severity']}/{disappeared['severity']} do not "
        f"distinguish an appearance from a disappearance",
    )

    # A genuine 0.0 is a measurement and must not read as absence.
    kind, delta = classify_change(0.0, 0.0)
    check(
        kind == ALERT_CHANGE_NONE and delta == 0.0,
        f"a measured 0.0 on both sides produced {kind!r}; that is a real "
        f"forecast, not an absence",
    )

    # 3. THE THRESHOLD SITS ABOVE SAMPLING NOISE.
    band = 1.96 * math.sqrt(0.25 / 100)
    check(
        FORECAST_ALERT_MIN_PROBABILITY_MOVE >= band * 0.9,
        f"the {FORECAST_ALERT_MIN_PROBABILITY_MOVE} threshold sits inside the "
        f"+/-{band:.4f} sampling band at 100 observations, so the alert fires "
        f"on resampling rather than on news",
    )
    moved = forecast_change_alert(
        snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)
    )
    check(
        moved["kind"] == ALERT_CHANGE_MOVED and moved["fired"],
        f"a 0.16 move produced {moved['kind']!r} fired={moved['fired']}",
    )
    check(
        abs(moved["delta"] - 0.16) < 1e-9,
        f"the delta {moved['delta']} does not equal the move",
    )
    noise = forecast_change_alert(
        snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)
    )
    check(
        noise["kind"] == ALERT_CHANGE_NONE and not noise["fired"],
        f"a 0.02 move produced {noise['kind']!r} fired={noise['fired']} — the "
        f"alert fires on noise and therefore means nothing",
    )
    check(noise["severity"] is None, "a quiet alert carried a severity")

    # 4. A FIRST OBSERVATION IS NOT A CHANGE.
    first = forecast_change_alert(None, snapshot("2026-09-22", 0.56))
    check(
        first["kind"] == ALERT_CHANGE_NOT_EVALUATED and not first["fired"],
        f"a first observation produced {first['kind']!r} "
        f"fired={first['fired']}",
    )
    check(
        ALERT_CHANGE_NOT_EVALUATED != ALERT_CHANGE_NONE,
        "'the forecast did not change' and 'there was nothing to compare' "
        "share a kind",
    )

    # 5. EVERY KIND IS REACHABLE, and every produced alert is contract-clean.
    produced = {
        quiet["kind"],
        appeared["kind"],
        disappeared["kind"],
        moved["kind"],
        noise["kind"],
    }
    for required in FORECAST_CHANGE_KINDS:
        if required == ALERT_CHANGE_NOT_EVALUATED:
            continue
        check(
            required in produced or required == ALERT_CHANGE_NOT_EVALUATED,
            f"{required!r} is declared but the gate never produces it",
        )
    for alert in (quiet, appeared, disappeared, moved, noise, first):
        check(
            alert_problems(alert) == [],
            f"{alert['kind']}: a well-formed alert reported problems: "
            f"{alert_problems(alert)}",
        )

    # 6. MALFORMED INPUT IS REFUSED.
    try:
        forecast_change_alert(snapshot("2026-09-21"), None)
        FAILURES.append("an alert with no current snapshot was accepted")
    except ForecastAlertError:
        pass
    try:
        forecast_change_alert("yesterday", snapshot("2026-09-22"))
        FAILURES.append("a non-mapping previous snapshot was accepted")
    except ForecastAlertError:
        pass
    for bad in (0.0, 1.0, -0.1):
        try:
            classify_change(0.4, 0.6, threshold=bad)
            FAILURES.append(f"a threshold of {bad} was accepted")
        except ForecastAlertError:
            pass
    broken_value = snapshot("2026-09-22", 0.56)
    broken_value["fields"]["probability_up"]["value"] = "high"
    try:
        forecast_change_alert(snapshot("2026-09-21", 0.4), broken_value)
        FAILURES.append("a non-numeric PRESENT value was accepted")
    except ForecastAlertError:
        pass

    # 7. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not FORECAST_ALERT_BLOCKS_TRADES and not moved["blocks_trades"],
        "the alert claims to block trades",
    )

    # 8. THE RENDER CARRIES THE KIND AND KEEPS ABSENCES ABSENT.
    text = "\n".join(render_alert(moved))
    check(ALERT_CHANGE_MOVED in text, "the change kind does not reach the render")
    check("FIRED" in text, "a fired alert is not marked in the render")
    gone = render_alert(disappeared)[0]
    check(
        "—" in gone,
        f"a vanished forecast rendered as {gone!r} rather than as absent",
    )
    for line in render_alert(moved) + render_alert(disappeared):
        check(len(line) < 200, f"a rendered line is {len(line)} characters long")

    # 9. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (lambda a: a.update({"compares_digest": True}), "an alert comparing digests"),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"kind": "PROBABLY"}), "an unknown kind"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"delta": 0.99}), "an inconsistent delta"),
        (lambda a: a.update({"severity": None}), "a fired alert with no severity"),
    ]
    for mutation, label in mutations:
        broken = forecast_change_alert(
            snapshot("2026-09-21", 0.40), snapshot("2026-09-22", 0.56)
        )
        mutation(broken)
        check(alert_problems(broken) != [], f"{label} passed the contract check")

    broken = forecast_change_alert(
        snapshot("2026-09-21"), snapshot("2026-09-22", 0.56)
    )
    broken["delta"] = 0.56
    check(
        alert_problems(broken) != [],
        "an APPEARED alert carrying a delta passed the contract check",
    )
    broken = forecast_change_alert(
        snapshot("2026-09-21", 0.54), snapshot("2026-09-22", 0.56)
    )
    broken["kind"] = ALERT_CHANGE_MOVED
    broken["fired"] = True
    broken["severity"] = ALERT_SEVERITY_WARN
    check(
        alert_problems(broken) != [],
        "a below-threshold move marked MOVED passed the contract check",
    )
    broken = forecast_change_alert(None, snapshot("2026-09-22", 0.56))
    broken["fired"] = True
    broken["severity"] = ALERT_SEVERITY_WARN
    check(
        alert_problems(broken) != [],
        "a first observation that fired passed the contract check",
    )

    if FAILURES:
        print("FORECAST CHANGE ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("FORECAST CHANGE ALERT GATE: PASS")
    print(f"  {len(digests)} distinct digests over {len(days)} identical "
          f"unavailable forecasts (recomputed) — so digests are not compared")
    print("  APPEARED/DISAPPEARED carry no delta: a vanished forecast is not a crash")
    print(f"  threshold {FORECAST_ALERT_MIN_PROBABILITY_MOVE} sits above the "
          f"+/-{band:.3f} sampling band at n=100")
    print("  a first observation is NOT_EVALUATED, never a change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
