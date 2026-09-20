"""CI drift gate for L1 outcome closure.

Two properties carry this sprint, and both are the kind a tidy-up removes:

  - a forecast is scored ONCE, by the measure its CLAIM TIER earned;
  - an error figure never travels without the coverage it came from.

1.  each claim tier is scored by its own measure. Brier on an interval is
    undefined, and a refusal made no claim to be wrong about;
2.  A REFUSAL IS NOT_SCORED. Zero error would reward silence, maximum error
    would punish honesty — neither measures anything;
3.  THE SCOREBOARD IS GAMEABLE BY REFUSING, MEASURED: over 200 forecasts from
    a skilled forecaster, refusing the hardest cases improves Brier from
    0.2206 to 0.0198. The report must expose that, or "refuse everything
    hard" is the winning strategy;
4.  closure is IDEMPOTENT. A CLOSED forecast is terminal: re-closing
    double-counts one observation and silently re-weights every metric;
5.  maturity is COMPOSED from V1/F2, not re-derived, so one notion of "has
    this expired?" governs the forecast layer and the closure layer (W5);
6.  EXPIRED_NO_DATA stays distinct from OPEN: "we could not score it" and
    "it has not happened yet" have different fixes;
7.  the ledger deduplicates by forecast_id — a re-run must not inflate the
    denominator and quietly change every metric;
8.  the shape rule holds: a refused forecast carries no value, or it would
    later be scored as a confident answer nobody gave;
9.  calibration composes M6 and refuses below its sample floor.

Synthetic apart from one live label read.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    CLOSURE_CLOSEABLE_STATES,
    CLOSURE_CLOSED,
    CLOSURE_EXPIRED_NO_DATA,
    CLOSURE_MATURED,
    CLOSURE_MIN_FOR_CALIBRATION,
    CLOSURE_MIN_SCORED_SHARE,
    CLOSURE_OPEN,
    CLOSURE_SCORE_METHODS,
    CLOSURE_SCORE_REFUSALS,
    CLOSURE_STATES,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_COVERAGE,
    SCORE_METHOD_DIRECTION,
    SCORE_METHOD_NOT_SCORED,
)
from core.outcome_closure import (  # noqa: E402
    OutcomeClosureError,
    build_ledger_row,
    close_forecast,
    closure_problems,
    closure_report,
    evaluate_calibration,
    ledger_row_problems,
    load_ledger,
    record_forecast,
    score_forecast,
)

_UP = {"forward_return": 0.04, "direction_up": True}
_DOWN = {"forward_return": -0.04, "direction_up": False}


def _row(claim, **extra):
    payload = dict(
        ticker="NVDA", as_of="2024-06-15", horizon="20d",
        target="probability_up", claim=claim, samples=45,
    )
    payload.update(extra)
    return build_ledger_row(**payload)


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    point = score_forecast(_row("POINT", value=0.9), _DOWN)
    if point.get("method") != SCORE_METHOD_BRIER or not point.get("scored"):
        failures.append("a POINT claim was not scored by Brier")
    elif abs(point["brier"] - 0.81) > 1e-9:
        failures.append(f"Brier for 0.9 against a fall was {point['brier']}, not 0.81")

    interval = score_forecast(
        _row("INTERVAL", interval={"lower": 0.3, "upper": 0.8}), _UP
    )
    if interval.get("method") != SCORE_METHOD_COVERAGE:
        failures.append(
            "an INTERVAL claim was not scored by COVERAGE — Brier on a range "
            "is undefined, and coverage is the quantity the interval claimed"
        )
    if "brier" in interval:
        failures.append("an INTERVAL claim produced a Brier score")

    directional = score_forecast(_row("DIRECTIONAL", direction="HIGHER"), _UP)
    if directional.get("method") != SCORE_METHOD_DIRECTION:
        failures.append("a DIRECTIONAL claim was not scored by direction")
    elif not directional.get("hit"):
        failures.append("a correct HIGHER call against a rise was scored a miss")
    if not score_forecast(_row("DIRECTIONAL", direction="HIGHER"), _DOWN).get(
        "scored"
    ):
        failures.append("a wrong direction call was not scored at all")

    for tier, method in CLOSURE_SCORE_METHODS.items():
        if tier == "INTERVAL" and method != SCORE_METHOD_COVERAGE:
            failures.append("the INTERVAL tier no longer scores by coverage")
        if tier == "INSUFFICIENT" and method != SCORE_METHOD_NOT_SCORED:
            failures.append("the INSUFFICIENT tier is no longer unscored")

    # ---------------------------------------------------------------- 2
    refusal = score_forecast(_row("INSUFFICIENT"), _UP)
    if refusal.get("scored"):
        failures.append(
            "a REFUSAL was scored — zero error rewards silence and maximum "
            "error punishes honesty; neither measures anything"
        )
    if refusal.get("method") != SCORE_METHOD_NOT_SCORED:
        failures.append("a refusal did not report NOT_SCORED")
    if CLOSURE_SCORE_REFUSALS:
        failures.append("CLOSURE_SCORE_REFUSALS was enabled")

    # ---------------------------------------------------------------- 3
    # THE GAMING MEASUREMENT. A report must expose refusal-driven "improvement".
    rng = np.random.default_rng(11)
    predictions = np.clip(rng.normal(0.58, 0.12, 200), 0.02, 0.98)
    actuals = (rng.random(200) < predictions).astype(float)

    def _closed(predicted, actual):
        return {
            "claim": "POINT", "state": CLOSURE_CLOSED,
            "score": {
                "method": SCORE_METHOD_BRIER, "scored": True,
                "predicted": float(predicted), "actual": float(actual),
                "brier": float((predicted - actual) ** 2),
            },
        }

    def _refused():
        return {
            "claim": "INSUFFICIENT", "state": CLOSURE_CLOSED,
            "score": {"method": SCORE_METHOD_NOT_SCORED, "scored": False,
                      "reason": "no claim"},
        }

    order = np.argsort(np.abs(predictions - 0.5))[::-1]
    honest = closure_report(
        [_closed(predictions[i], actuals[i]) for i in range(200)]
    )
    keep = set(order[:2].tolist())
    gamed = closure_report(
        [
            _closed(predictions[i], actuals[i]) if i in keep else _refused()
            for i in range(200)
        ]
    )

    if gamed["mean_brier"] >= honest["mean_brier"]:
        failures.append(
            "the gaming fixture no longer reproduces — refusing the hardest "
            "cases should LOWER the apparent Brier, which is why coverage "
            "must be published"
        )
    if gamed["reliable"]:
        failures.append(
            f"a report scoring only {gamed['scored']} of {gamed['total']} "
            f"forecasts was marked RELIABLE — MEASURED, that Brier "
            f"({gamed['mean_brier']:.4f}) is 11x 'better' than the honest "
            f"{honest['mean_brier']:.4f} purely by refusing"
        )
    if not honest["reliable"]:
        failures.append("a fully-scored report was marked unreliable")
    if not gamed.get("reliability_reason"):
        failures.append("an unreliable report does not say why")
    for name in ("scored", "refused", "pending", "scored_share"):
        if gamed.get(name) is None:
            failures.append(f"the report omits {name!r} beside its error")
    for problem in closure_problems(gamed) + closure_problems(honest):
        failures.append(f"report problem: {problem}")

    # An error published without coverage must be reported as a defect.
    stripped = dict(honest)
    stripped.pop("scored", None)
    if not closure_problems(stripped):
        failures.append(
            "an error figure with no scored count raised no problem — that is "
            "precisely the unreadable number this gate exists to prevent"
        )

    # ---------------------------------------------------------------- 4 + 6
    if CLOSURE_CLOSED in CLOSURE_CLOSEABLE_STATES:
        failures.append(
            "a CLOSED forecast is closeable again — re-closing double-counts "
            "one observation and re-weights every metric"
        )
    already = close_forecast({**_row("POINT", value=0.6), "state": CLOSURE_CLOSED}, None)
    if already.get("score") is not None:
        failures.append("re-closing a CLOSED forecast produced a fresh score")
    if already.get("state") != CLOSURE_CLOSED:
        failures.append("re-closing a CLOSED forecast changed its state")
    for state in (CLOSURE_OPEN, CLOSURE_MATURED, CLOSURE_CLOSED,
                  CLOSURE_EXPIRED_NO_DATA):
        if state not in CLOSURE_STATES:
            failures.append(f"closure state {state!r} is no longer declared")

    # ---------------------------------------------------------------- 5
    # Verified by OBSERVING the call, not by grepping the file: a local
    # re-implementation named horizon_readiness would satisfy a text search
    # while being exactly the second notion of maturity W5 forbids.
    import core.forecast_horizons as horizons_module

    observed = {"n": 0}
    original = horizons_module.horizon_readiness

    def _watched(*args, **kwargs):
        observed["n"] += 1
        return original(*args, **kwargs)

    horizons_module.horizon_readiness = _watched
    try:
        close_forecast(
            _row("POINT", value=0.6),
            {"horizons": {"20d": {"forward_return": 0.02}},
             "matured_horizons": ["20d"], "pending_horizons": []},
        )
    finally:
        horizons_module.horizon_readiness = original
    if observed["n"] != 1:
        failures.append(
            f"core.forecast_horizons.horizon_readiness was called "
            f"{observed['n']} time(s) during closure — maturity must be "
            f"COMPOSED from the forecast layer, not re-derived, or the two "
            f"notions of 'has this expired?' will drift apart"
        )

    # ---------------------------------------------------------------- 7 + 8
    with tempfile.TemporaryDirectory() as folder:
        ledger = Path(folder) / "ledger.jsonl"
        row = _row("POINT", value=0.62)
        record_forecast(row, ledger)
        record_forecast(row, ledger)
        record_forecast(row, ledger)
        if len(load_ledger(ledger)) != 1:
            failures.append(
                f"re-recording produced {len(load_ledger(ledger))} rows — a "
                f"re-run would inflate the denominator and quietly change "
                f"every metric"
            )
        # Attacked at BOTH layers. The builder must refuse the contradictory
        # request, AND validation must catch a row assembled by hand — a
        # single guard behind another is untestable once the first one holds.
        try:
            build_ledger_row(
                ticker="NVDA", as_of="2024-06-15", horizon="20d",
                target="probability_up", claim="INSUFFICIENT", value=0.5,
            )
            failures.append(
                "build_ledger_row accepted a value on an INSUFFICIENT claim — "
                "it would later be scored as a confident answer nobody gave"
            )
        except OutcomeClosureError:
            pass
        # A row assembled by hand bypasses the builder entirely — a stored
        # ledger read back from disk, or a row built by an older version.
        # `ledger_row_problems` is the guard for that path, so it is attacked
        # on its own rather than through the builder that already sanitises.
        hand_made = {
            "forecast_id": "manual", "ticker": "NVDA", "as_of": "2024-06-15",
            "horizon": "20d", "target": "probability_up",
            "claim": "INSUFFICIENT", "score_method": SCORE_METHOD_NOT_SCORED,
            "state": CLOSURE_OPEN, "value": 0.5,
        }
        if not any(
            "confident answer nobody gave" in problem
            for problem in ledger_row_problems(hand_made)
        ):
            failures.append(
                "ledger_row_problems accepted a hand-assembled refusal "
                "carrying a value — that row would later be scored as a "
                "confident answer nobody gave"
            )
        try:
            record_forecast(hand_made, ledger)
            failures.append("a hand-assembled refusal carrying a value was recorded")
        except OutcomeClosureError:
            pass

    try:
        build_ledger_row(
            ticker="NVDA", as_of="x", horizon="20d", target="t", claim="NOPE"
        )
        failures.append("an unknown claim tier was accepted into the ledger")
    except OutcomeClosureError:
        pass

    # ---------------------------------------------------------------- 9
    thin = evaluate_calibration(
        [_closed(predictions[i], actuals[i]) for i in range(5)]
    )
    if thin.get("status") != "INSUFFICIENT":
        failures.append(
            f"calibration was attempted on 5 observations (floor "
            f"{CLOSURE_MIN_FOR_CALIBRATION})"
        )
    full = evaluate_calibration(
        [_closed(predictions[i], actuals[i]) for i in range(200)]
    )
    if full.get("status") != "MEASURED":
        failures.append(f"calibration on 200 observations reported {full.get('status')}")
    elif full.get("brier") is None:
        failures.append("a measured calibration carries no Brier")
    if not 0.0 < CLOSURE_MIN_SCORED_SHARE <= 1.0:
        failures.append("the minimum scored share left (0, 1]")

    # A live read, when labels are available.
    try:
        from core.labels import build_outcome_labels

        labels = build_outcome_labels("NVDA", "2024-06-15")
        if labels.get("status") in ("OK", "PARTIAL"):
            live = close_forecast(_row("POINT", value=0.62), labels)
            if live["state"] != CLOSURE_CLOSED:
                failures.append(
                    f"a 2024 forecast did not close (state {live['state']}): "
                    f"{live.get('closure_reason')}"
                )
            elif not (live.get("score") or {}).get("scored"):
                failures.append("a matured live forecast produced no score")
    except Exception as exc:  # a provider outage is not a contract failure
        print(f"  (live closure check skipped: {type(exc).__name__})")

    if failures:
        print("L1 outcome-closure gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("L1 outcome-closure gate OK:")
    print(
        "  each claim tier scored by its own measure: POINT->brier, "
        "INTERVAL->coverage, DIRECTIONAL->hit, INSUFFICIENT->not scored."
    )
    print(
        f"  gaming is visible: refusing the hardest cases takes Brier "
        f"{honest['mean_brier']:.4f} -> {gamed['mean_brier']:.4f}, and the "
        f"report marks that UNRELIABLE."
    )
    print("  closure is idempotent; a CLOSED forecast is terminal.")
    print("  the ledger deduplicates; a refusal may not carry a value.")
    print("  maturity composes V1/F2; calibration composes M6 and refuses when thin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
