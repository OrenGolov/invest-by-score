"""Gate: a condition that cannot be tested is not a condition that failed.

"E.g. 20D expected return + P(up) + confidence crossing a defined threshold,
with no veto active." Three conditions and a gate. MEASURED, one of the three
has no producer, and the two obvious ways to handle that are both wrong.

THE DECIDING MEASUREMENT, recomputed here: across 1d, 5d, 20d and 60d, ZERO of
the twelve condition inputs are PRESENT. expected_return is ABSENT at every
horizon because no trained model exists.

THE TWO WRONG ANSWERS:
  Coalesce the missing value to 0.0 - then 0.0 never clears the return bar and
  the alert is permanently, silently dead.
  Skip the condition that cannot be evaluated - then P(up) and confidence
  alone fire, reporting a threshold crossing on two of three conditions.

So an unevaluable condition yields NOT_EVALUATED, which is neither fired nor
quiet. W2's fail-closed rule does not transfer wholesale: blocking on missing
evidence is right for a VETO, but an ALERT that fires on absent data is noise.
The one place it DOES transfer is governance - an unknown veto state is not
"no veto active".

Verified to FAIL when any of these is reinjected:
  - a missing input coerced to 0.0
  - an unevaluable condition skipped so the rest can fire
  - a fired verdict with a condition untested
  - an unknown veto treated as no veto
  - an active veto ignored
  - NOT_EVALUATED collapsed into NOT_MET
  - the alert claiming to block trades
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ALERT_SEVERITY_WARN,
    FCONF_BAND_MODERATE,
    FORECAST_CONFIDENCE_BANDS,
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
    FTHRESHOLD_VETOED,
    RESEARCH_VIEW_HORIZONS,
)
from core.forecast_confidence import assess_confidence  # noqa: E402
from core.forecast_snapshot import build_forecast_snapshot  # noqa: E402
from core.forecast_threshold_alert import (  # noqa: E402
    ForecastThresholdError,
    evaluate_conditions,
    forecast_threshold_alert,
    render_threshold_alert,
    threshold_alert_problems,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def live_snapshot(horizon=FTHRESHOLD_HORIZON):
    return build_forecast_snapshot("NVDA", "2026-09-23", horizon)


def strong_snapshot(probability=0.72, horizon=FTHRESHOLD_HORIZON):
    confidence = assess_confidence(
        samples=800,
        interval={"low": 0.55, "high": 0.80},
        observed_share=0.95,
        regime_agreement=0.95,
        similarities=[0.95, 0.96, 0.97],
        features_present=10,
        features_expected=10,
        probability=probability,
    )
    return build_forecast_snapshot(
        "NVDA",
        "2026-09-23",
        horizon,
        event_forecast={
            "value": probability,
            "samples": 800,
            "interval": {"low": 0.55, "high": 0.80},
            "as_of": "2026-09-23",
            "horizon": horizon,
            "event": {"entity": "NVDA"},
        },
        confidence=confidence,
        regime="bullish",
    )


def with_return(snapshot, value=0.045):
    full = copy.deepcopy(snapshot)
    full["fields"]["expected_return"] = {
        "status": "PRESENT",
        "value": value,
        "reason": "supplied by the gate",
    }
    return full


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED.
    present = 0
    total = 0
    for horizon in RESEARCH_VIEW_HORIZONS:
        snapshot = live_snapshot(horizon)
        for condition in FTHRESHOLD_CONDITIONS:
            total += 1
            if snapshot["fields"][condition]["status"] == "PRESENT":
                present += 1
    check(
        present == 0,
        f"{present} of {total} condition inputs are now PRESENT; A6's "
        f"NOT_EVALUATED design rests on them being unavailable and must be "
        f"re-derived rather than kept if that has changed",
    )

    # 2. THE LIVE ALERT IS BLIND, AND SAYS SO.
    live = forecast_threshold_alert(live_snapshot(), {"veto": False})
    check(
        live["verdict"] == FTHRESHOLD_NOT_EVALUATED and not live["fired"],
        f"the live alert produced {live['verdict']!r} fired={live['fired']}",
    )
    check(
        live["verdict"] != FTHRESHOLD_NOT_MET,
        "a blind alert reported as tested-and-failed; a reader would infer "
        "the thresholds were checked",
    )
    check(
        FTHRESHOLD_CONDITION_RETURN in live["unevaluable"],
        f"expected_return is not named among the untestable conditions "
        f"{live['unevaluable']}",
    )

    # 3. TWO OF THREE PASSING MUST NOT FIRE. The case that decides the module.
    partial = forecast_threshold_alert(strong_snapshot(), {"veto": False})
    conditions = partial["conditions"]
    check(
        conditions[FTHRESHOLD_CONDITION_PROBABILITY]["met"] is True
        and conditions[FTHRESHOLD_CONDITION_CONFIDENCE]["met"] is True,
        "the gate's strong snapshot no longer clears P(up) and confidence; "
        "it needs two conditions passing for the test to mean anything",
    )
    check(
        conditions[FTHRESHOLD_CONDITION_RETURN]["evaluable"] is False,
        "expected_return became evaluable in the gate's scenario",
    )
    check(
        partial["verdict"] == FTHRESHOLD_NOT_EVALUATED and not partial["fired"],
        f"two of three conditions passing produced {partial['verdict']!r} "
        f"fired={partial['fired']}; that silently weakens the rule to 'the "
        f"ones we could check held'",
    )
    check(
        FTHRESHOLD_REQUIRES_ALL_CONDITIONS,
        "the alert does not require every condition",
    )

    # 4. A MISSING INPUT IS NEVER COERCED.
    check(not FTHRESHOLD_COERCES_MISSING, "the alert coerces a missing input")
    entry = live["conditions"][FTHRESHOLD_CONDITION_RETURN]
    check(
        entry["value"] is None and entry["met"] is None,
        f"an unevaluable condition carried value={entry['value']!r} "
        f"met={entry['met']!r}; 0.0 never clears the return bar, which makes "
        f"the alert permanently and silently dead",
    )

    # 5. A STRUCTURED CONFIDENCE IS READ BY NAME, NOT GUESSED.
    structured = partial["conditions"][FTHRESHOLD_CONDITION_CONFIDENCE]
    check(
        structured["evaluable"] and isinstance(structured["value"], float),
        f"the confidence condition read {structured['value']!r}; F7 stores "
        f"its whole assessment, and the scalar must be read from the named "
        f"field",
    )
    # The probe must carry a DECOY NUMBER. MEASURED: with a mapping holding no
    # numbers at all, a sabotage that guessed "the first numeric value" still
    # passed, because there was nothing to guess. F7's real assessment is full
    # of unrelated floats - binding_value, weighted_sum, every factor score -
    # so the decoy is what makes the check mean anything.
    blind = live_snapshot()
    blind["fields"]["confidence"] = {
        "status": "PRESENT",
        "value": {"band": "HIGH", "binding_value": 0.87, "weighted_sum": 0.96},
        "reason": "no confidence scalar, but plenty of unrelated numbers",
    }
    try:
        evaluate_conditions(blind)
        FAILURES.append(
            "a PRESENT mapping with no 'confidence' key was accepted; the "
            "threshold would compare against binding_value or weighted_sum, "
            "which are factor scores rather than the confidence itself"
        )
    except ForecastThresholdError:
        pass

    # 6. ALL THREE CLEARING, NO VETO -> FIRED.
    full = with_return(strong_snapshot())
    fired = forecast_threshold_alert(full, {"veto": False})
    check(
        fired["verdict"] == FTHRESHOLD_FIRED and fired["fired"],
        f"all three conditions clearing produced {fired['verdict']!r}",
    )
    check(
        fired["severity"] == ALERT_SEVERITY_WARN,
        f"a fired threshold alert carries severity {fired['severity']!r}",
    )

    # 7. A FAILING CONDITION IS NOT_MET, distinct from NOT_EVALUATED.
    weak = forecast_threshold_alert(
        with_return(strong_snapshot(probability=0.52)), {"veto": False}
    )
    check(
        weak["verdict"] == FTHRESHOLD_NOT_MET and not weak["fired"],
        f"a failing condition produced {weak['verdict']!r}",
    )
    check(
        FTHRESHOLD_CONDITION_PROBABILITY in weak["failed"],
        f"the failing condition is not named in {weak['failed']}",
    )
    check(
        FTHRESHOLD_NOT_EVALUATED != FTHRESHOLD_NOT_MET,
        "'could not be tested' and 'tested and failed' share a verdict",
    )

    # 8. GOVERNANCE: AN UNKNOWN VETO IS NOT NO VETO.
    check(
        FTHRESHOLD_UNKNOWN_VETO_SUPPRESSES,
        "an unknown veto state is being treated as no veto",
    )
    vetoed = forecast_threshold_alert(full, {"veto": True, "veto_rule_ids": ["x"]})
    check(
        vetoed["verdict"] == FTHRESHOLD_VETOED and not vetoed["fired"],
        f"an active veto produced {vetoed['verdict']!r}",
    )
    unknown = forecast_threshold_alert(full, None)
    check(
        unknown["verdict"] == FTHRESHOLD_VETOED and not unknown["fired"],
        f"an unknown veto produced {unknown['verdict']!r}; claiming "
        f"governance passed when nobody asked is how a blocked trade gets "
        f"recommended",
    )
    check(unknown["veto"] is None, "an unknown veto was reported as a boolean")
    fieldless = forecast_threshold_alert(full, {"rules": []})
    check(
        fieldless["verdict"] == FTHRESHOLD_VETOED,
        f"a risk report with no veto field produced {fieldless['verdict']!r}",
    )

    # 9. THE THRESHOLDS ARE DERIVED OR REUSED.
    band = 1.96 * math.sqrt(0.25 / 100)
    check(
        FTHRESHOLD_MIN_PROBABILITY > 0.5 + band * 0.9,
        f"the {FTHRESHOLD_MIN_PROBABILITY} probability bar does not clear a "
        f"coin flip by more than the +/-{band:.4f} sampling band",
    )
    check(
        FTHRESHOLD_MIN_CONFIDENCE
        == dict(FORECAST_CONFIDENCE_BANDS)[FCONF_BAND_MODERATE],
        f"the confidence bar {FTHRESHOLD_MIN_CONFIDENCE} is not F7's MODERATE "
        f"floor; a hardcoded value would drift if F7 retuned its bands",
    )
    check(FTHRESHOLD_MIN_RETURN > 0.0, "the return bar is not positive")

    # 10. EVERY ALERT PRODUCED IS CONTRACT-CLEAN.
    for alert in (live, partial, fired, weak, vetoed, unknown, fieldless):
        check(
            threshold_alert_problems(alert) == [],
            f"{alert['verdict']}: a well-formed alert reported problems: "
            f"{threshold_alert_problems(alert)}",
        )

    # 11. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not FTHRESHOLD_BLOCKS_TRADES and not fired["blocks_trades"],
        "the alert claims to block trades",
    )

    # 12. THE RENDER KEEPS ABSENCES ABSENT.
    text = "\n".join(render_threshold_alert(live))
    for condition in FTHRESHOLD_CONDITIONS:
        check(condition in text, f"{condition} does not reach the render")
    return_line = [
        line for line in render_threshold_alert(live)
        if FTHRESHOLD_CONDITION_RETURN in line
    ][0]
    check(
        "—" in return_line and "0.0000" not in return_line,
        f"an unevaluable condition rendered as {return_line!r}",
    )
    check(
        "unknown" in render_threshold_alert(unknown)[0],
        "an unknown veto does not render as unknown",
    )

    # 13. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (lambda a: a.update({"coerces_missing": True}), "an alert that coerces"),
        (
            lambda a: a.update({"requires_all_conditions": False}),
            "an alert not requiring every condition",
        ),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"verdict": "PROBABLY"}), "an unknown verdict"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"unevaluable": []}), "a mismatched unevaluable list"),
        (
            lambda a: a["conditions"][FTHRESHOLD_CONDITION_RETURN].update(
                {"value": 0.0}
            ),
            "a coerced value on an unevaluable condition",
        ),
    ]
    for mutation, label in mutations:
        broken = forecast_threshold_alert(live_snapshot(), {"veto": False})
        mutation(broken)
        check(
            threshold_alert_problems(broken) != [],
            f"{label} passed the contract check",
        )

    broken = forecast_threshold_alert(strong_snapshot(), {"veto": False})
    broken.update({"verdict": FTHRESHOLD_FIRED, "fired": True,
                   "severity": ALERT_SEVERITY_WARN, "veto": False})
    check(
        threshold_alert_problems(broken) != [],
        "an alert that fired with a condition untested passed the contract "
        "check",
    )
    broken = forecast_threshold_alert(full, None)
    broken.update({"verdict": FTHRESHOLD_FIRED, "fired": True,
                   "severity": ALERT_SEVERITY_WARN})
    check(
        threshold_alert_problems(broken) != [],
        "an alert that fired on an unknown veto passed the contract check",
    )
    broken = forecast_threshold_alert(
        with_return(strong_snapshot()), {"veto": False}
    )
    broken["conditions"][FTHRESHOLD_CONDITION_PROBABILITY]["met"] = False
    check(
        threshold_alert_problems(broken) != [],
        "a met flag disagreeing with its own value passed the contract check",
    )

    if FAILURES:
        print("FORECAST THRESHOLD ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("FORECAST THRESHOLD ALERT GATE: PASS")
    print(f"  {present} of {total} condition inputs are PRESENT across "
          f"{len(RESEARCH_VIEW_HORIZONS)} horizons (recomputed)")
    print("  two of three conditions passing yields NOT_EVALUATED, never a fire")
    print("  a missing input is never coerced to 0.0")
    print("  an unknown veto suppresses; it is not 'no veto active'")
    print(f"  bars: P(up) {FTHRESHOLD_MIN_PROBABILITY} (derived), confidence "
          f"{FTHRESHOLD_MIN_CONFIDENCE} (F7 MODERATE floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
