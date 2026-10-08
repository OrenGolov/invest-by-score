"""Gate: the confidence alert watches the reason, not only the number.

"Alert when confidence changes materially." A2 is not A1 with a different
field, and the difference was measured.

CONFIDENCE MOVES WHEN THE FORECAST DOES NOT. Holding P(up) fixed at 0.56 and
dropping feature completeness from 10/10 to 6/10 moves confidence while the
forecast value does not move at all.

THE DECIDING MEASUREMENT, recomputed here: F7's confidence is bounded by its
WEAKEST factor and reports which one bound it. Over sampled assessment pairs,
those moving confidence by less than the threshold - below any sane alert -
change the BINDING FACTOR about half the time. A concrete instance: confidence
0.5559 -> 0.4640, a -0.0919 move no threshold fires on, while the binding
factor goes from event_similarity to regime_similarity. The forecast became
uncertain for an entirely different reason.

So A2 watches the magnitude, the BAND and the BINDING FACTOR, and the binding
factor is checked FIRST - a magnitude check would swallow every one of those
cases.

Verified to FAIL when any of these is reinjected:
  - the binding factor ignored, leaving a magnitude-only alert
  - a changed binding factor reported as NONE
  - a band crossing ignored
  - an availability change folded into a magnitude
  - a first observation reported as a change
  - a fired alert with no severity, or a quiet one carrying severity
  - the alert claiming to block trades
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL,
    CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL,
    CONFIDENCE_ALERT_BLOCKS_TRADES,
    CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT,
    CONFIDENCE_ALERT_MIN_MOVE,
    CONFIDENCE_CHANGE_KINDS,
    CONF_CHANGE_APPEARED,
    CONF_CHANGE_BAND,
    CONF_CHANGE_BINDING,
    CONF_CHANGE_DISAPPEARED,
    CONF_CHANGE_NONE,
    CONF_CHANGE_NOT_EVALUATED,
)
from core.confidence_alert import (  # noqa: E402
    ConfidenceAlertError,
    classify_confidence_change,
    confidence_alert_problems,
    confidence_change_alert,
    render_confidence_alert,
)
from core.forecast_confidence import assess_confidence  # noqa: E402

FAILURES: list[str] = []

BASE = dict(
    samples=800,
    interval={"low": 0.48, "high": 0.60},
    observed_share=0.95,
    regime_agreement=0.9,
    similarities=[0.9, 0.85],
    features_present=10,
    features_expected=10,
    probability=0.56,
)

UNMEASURED = {
    "assessed_object": "forecast",
    "confidence": None,
    "band": None,
    "binding_factor": None,
}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def assessment(**overrides):
    return assess_confidence(**{**BASE, **overrides})


def main() -> int:
    # 1. CONFIDENCE MOVES WHEN THE FORECAST DOES NOT.
    before = assessment()
    fewer_features = assessment(features_present=6)
    check(
        before["probability"] == fewer_features["probability"],
        "the gate's scenario changed the forecast value; A2 is about "
        "confidence moving while the forecast does not",
    )
    check(
        before["confidence"] != fewer_features["confidence"],
        "confidence did not move when a factor weakened; A2 would have "
        "nothing to watch",
    )

    # 2. THE DECIDING MEASUREMENT, RECOMPUTED.
    rng = random.Random(17)

    def sampled():
        return assess_confidence(
            samples=rng.choice([40, 60, 100, 200, 400, 800]),
            interval={
                "low": 0.45,
                "high": 0.45 + rng.choice([0.05, 0.10, 0.20, 0.40]),
            },
            observed_share=rng.choice([0.4, 0.6, 0.8, 0.95]),
            regime_agreement=rng.choice([0.3, 0.5, 0.7, 0.9]),
            similarities=[
                rng.uniform(0.7, 1.0) for _ in range(rng.choice([2, 3, 5]))
            ],
            features_present=rng.choice([5, 7, 9, 10]),
            features_expected=10,
            probability=0.55,
        )

    flat = 0
    changed = 0
    for _ in range(2000):
        a, b = sampled(), sampled()
        if abs(b["confidence"] - a["confidence"]) < CONFIDENCE_ALERT_MIN_MOVE:
            flat += 1
            if a["binding_factor"] != b["binding_factor"]:
                changed += 1
    check(flat > 200, f"only {flat} sub-threshold pairs sampled; too few to measure")
    share = changed / flat if flat else 0.0
    check(
        share > 0.25,
        f"only {changed}/{flat} ({share:.0%}) sub-threshold moves changed the "
        f"binding factor. A2's design rests on that being common; if it has "
        f"vanished the design must be re-derived rather than kept",
    )

    # 3. THE CONCRETE CASE FIRES.
    after = assessment(samples=120, regime_agreement=0.5)
    delta = after["confidence"] - before["confidence"]
    check(
        abs(delta) < CONFIDENCE_ALERT_MIN_MOVE,
        f"the reference pair moves confidence {delta:+.4f}, at or beyond the "
        f"{CONFIDENCE_ALERT_MIN_MOVE} threshold; the gate needs a move no "
        f"magnitude alert would fire on",
    )
    check(
        before["binding_factor"] != after["binding_factor"],
        "the reference pair no longer changes the binding factor",
    )
    binding_alert = confidence_change_alert(before, after)
    check(
        binding_alert["kind"] == CONF_CHANGE_BINDING,
        f"a sub-threshold move with a changed binding factor produced "
        f"{binding_alert['kind']!r} — a magnitude-only alert is silent here",
    )
    check(binding_alert["fired"], "the binding-factor change did not fire")
    check(
        before["binding_factor"] in binding_alert["reason"]
        and after["binding_factor"] in binding_alert["reason"],
        "the alert does not name both binding factors",
    )
    check(
        CONFIDENCE_ALERT_BINDING_CHANGE_IS_MATERIAL
        and binding_alert["watches_binding_factor"],
        "the alert does not declare that it watches the binding factor",
    )

    # 4. THE BAND IS WATCHED TOO, and it must be exercised in isolation.
    #    MEASURED: without a band-cross-only pair here, disabling the band
    #    check entirely still passed this gate — every other scenario either
    #    changes the binding factor or clears the magnitude threshold, so the
    #    band branch was never the one that decided.
    check(CONFIDENCE_ALERT_BAND_CROSS_IS_MATERIAL, "band crossings are not material")
    band_only = None
    for candidate_samples in (400, 600, 800, 1000):
        for candidate_similarities in (
            [0.99, 0.99, 0.99],
            [0.95, 0.96],
            [0.75, 0.78],
            [0.72, 0.71],
        ):
            candidate = assessment(
                samples=candidate_samples, similarities=candidate_similarities
            )
            if (
                candidate["binding_factor"] == before["binding_factor"]
                and candidate["band"] != before["band"]
            ):
                band_only = candidate
                break
        if band_only:
            break
    check(
        band_only is not None,
        "the gate could not construct a band crossing that leaves the binding "
        "factor unchanged, so the band branch is never exercised alone",
    )
    if band_only is not None:
        band_alert = confidence_change_alert(before, band_only)
        check(
            band_alert["kind"] == CONF_CHANGE_BAND,
            f"a band crossing with an unchanged binding factor produced "
            f"{band_alert['kind']!r}; the band is what a reader acts on, so "
            f"a crossing must be reported as one",
        )
        check(band_alert["fired"], "a band crossing did not fire")
        check(
            band_alert["previous_band"] != band_alert["current_band"],
            "the band alert reports the same band on both sides",
        )
        check(
            confidence_alert_problems(band_alert) == [],
            f"a well-formed band alert reported problems: "
            f"{confidence_alert_problems(band_alert)}",
        )

    # 5. AVAILABILITY IS NOT A MAGNITUDE.
    appeared = confidence_change_alert(UNMEASURED, assessment())
    check(
        appeared["kind"] == CONF_CHANGE_APPEARED and appeared["delta"] is None,
        f"a confidence becoming measurable produced {appeared['kind']!r} with "
        f"delta {appeared['delta']!r}",
    )
    check(appeared["previous"] is None, "APPEARED carried a previous confidence")
    disappeared = confidence_change_alert(assessment(), UNMEASURED)
    check(
        disappeared["kind"] == CONF_CHANGE_DISAPPEARED
        and disappeared["delta"] is None,
        f"a confidence going away produced {disappeared['kind']!r} with delta "
        f"{disappeared['delta']!r}",
    )
    check(disappeared["current"] is None, "DISAPPEARED carried a current confidence")
    check(
        appeared["severity"] == ALERT_SEVERITY_INFO
        and disappeared["severity"] == ALERT_SEVERITY_WARN,
        "an appearance and a disappearance carry the same severity",
    )

    # 6. UNMEASURABLE IS NOT LOW. A changed measurable SET is its own event.
    check(
        CONFIDENCE_ALERT_MEASURABILITY_IS_ITS_OWN_EVENT,
        "a factor becoming UNMEASURABLE is being treated as a confidence drop",
    )
    shrunk = dict(before)
    shrunk["measured"] = [m for m in before["measured"] if m != "sample_size"]
    measurability_alert = confidence_change_alert(before, shrunk)
    check(
        measurability_alert["measurability"] is not None
        and "sample_size" in measurability_alert["measurability"]["lost"],
        "a change in the set of measurable factors was not reported",
    )

    # 7. A FIRST OBSERVATION IS NOT A CHANGE, and an unchanged pair is quiet.
    first = confidence_change_alert(None, assessment())
    check(
        first["kind"] == CONF_CHANGE_NOT_EVALUATED and not first["fired"],
        f"a first observation produced {first['kind']!r} fired={first['fired']}",
    )
    same = confidence_change_alert(assessment(), assessment())
    check(
        same["kind"] == CONF_CHANGE_NONE and not same["fired"],
        f"an identical pair produced {same['kind']!r} fired={same['fired']} — "
        f"the alert fires on nothing and therefore means nothing",
    )
    check(same["severity"] is None, "a quiet alert carried a severity")
    check(
        CONF_CHANGE_NOT_EVALUATED != CONF_CHANGE_NONE,
        "'confidence did not change' and 'there was nothing to compare' share "
        "a kind",
    )

    # 8. EVERY ALERT PRODUCED IS CONTRACT-CLEAN.
    for alert in (binding_alert, appeared, disappeared, first, same, measurability_alert):
        check(
            confidence_alert_problems(alert) == [],
            f"{alert['kind']}: a well-formed alert reported problems: "
            f"{confidence_alert_problems(alert)}",
        )

    # 9. MALFORMED INPUT IS REFUSED.
    for bad_call, label in (
        (lambda: confidence_change_alert(assessment(), None), "no current assessment"),
        (lambda: confidence_change_alert("yesterday", assessment()), "a non-mapping prior"),
        (
            lambda: confidence_change_alert({"confidence": 1.5}, assessment()),
            "a confidence above 1",
        ),
        (
            lambda: confidence_change_alert({"confidence": "high"}, assessment()),
            "a non-numeric confidence",
        ),
    ):
        try:
            bad_call()
            FAILURES.append(f"{label} was accepted")
        except ConfidenceAlertError:
            pass
    for bad in (0.0, 1.0, -0.1):
        try:
            classify_confidence_change(assessment(), assessment(), threshold=bad)
            FAILURES.append(f"a threshold of {bad} was accepted")
        except ConfidenceAlertError:
            pass

    # 10. THE ALERT REPORTS; IT DOES NOT TRADE.
    check(
        not CONFIDENCE_ALERT_BLOCKS_TRADES and not binding_alert["blocks_trades"],
        "the alert claims to block trades",
    )

    # 11. THE RENDER CARRIES THE REASON.
    text = "\n".join(render_confidence_alert(binding_alert))
    check(
        binding_alert["current_binding"] in text,
        "the binding factor does not reach the render, so a reader sees the "
        "number without the reason",
    )
    check("FIRED" in text, "a fired alert is not marked in the render")
    gone = render_confidence_alert(disappeared)[0]
    check("—" in gone, f"a vanished confidence rendered as {gone!r}")
    for line in render_confidence_alert(binding_alert):
        check(len(line) < 250, f"a rendered line is {len(line)} characters long")

    # 12. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (
            lambda a: a.update({"watches_binding_factor": False}),
            "an alert not watching the binding factor",
        ),
        (lambda a: a.update({"blocks_trades": True}), "an alert that blocks trades"),
        (lambda a: a.update({"kind": "PROBABLY"}), "an unknown kind"),
        (lambda a: a.update({"reason": "  "}), "a reasonless alert"),
        (lambda a: a.update({"delta": 0.99}), "an inconsistent delta"),
        (lambda a: a.update({"severity": None}), "a fired alert with no severity"),
    ]
    for mutation, label in mutations:
        broken = confidence_change_alert(before, after)
        mutation(broken)
        check(
            confidence_alert_problems(broken) != [],
            f"{label} passed the contract check",
        )

    broken = confidence_change_alert(before, after)
    broken["kind"] = CONF_CHANGE_NONE
    broken["fired"] = False
    broken["severity"] = None
    check(
        confidence_alert_problems(broken) != [],
        "a changed binding factor reported as NONE passed the contract check "
        "— that is the case this alert exists to catch",
    )
    broken = confidence_change_alert(assessment(), assessment())
    broken["previous_band"] = "LOW"
    broken["current_band"] = "HIGH"
    check(
        confidence_alert_problems(broken) != [],
        "a band crossing reported as NONE passed the contract check",
    )
    broken = confidence_change_alert(UNMEASURED, assessment())
    broken["delta"] = 0.56
    check(
        confidence_alert_problems(broken) != [],
        "an APPEARED alert carrying a delta passed the contract check",
    )
    broken = confidence_change_alert(None, assessment())
    broken["fired"] = True
    broken["severity"] = ALERT_SEVERITY_WARN
    check(
        confidence_alert_problems(broken) != [],
        "a first observation that fired passed the contract check",
    )

    if FAILURES:
        print("CONFIDENCE CHANGE ALERT GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("CONFIDENCE CHANGE ALERT GATE: PASS")
    print(f"  {changed}/{flat} ({share:.0%}) sub-threshold moves changed the "
          f"binding factor (recomputed)")
    print(f"  the reference pair moves {delta:+.4f} — below threshold — and "
          f"still fires on the reason")
    print("  band and binding factor are checked BEFORE magnitude")
    print("  APPEARED/DISAPPEARED carry no delta; UNMEASURABLE is not low")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
