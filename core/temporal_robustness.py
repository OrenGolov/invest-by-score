"""X6 temporal robustness — an edge at one horizon is not an edge.

"Evaluate across 1D/5D/20D/60D/120D." MEASURED, **four of the five horizons have
no trained model at all** — and this blocker differs in kind from X3's and X4's,
because *nothing needs building*.

============================================  ===========================
horizons in ``data/training_runs.jsonl``      ``{'20d': 8}``
horizons X6 requires                          1d, 5d, 20d, 60d, 120d
horizons with a model                         **1 of 5**
============================================  ===========================

X3's blocker is a dropped join key; X4's is data that was never ingested. X6's is
neither: ``LABEL_HORIZON_SESSIONS`` already maps all five horizons,
``TrainingRun`` already carries ``target_horizon``, and ``train_baseline``
already accepts one. The runs were simply never produced. The next action is to
train the four missing horizons, not to build machinery.

**Why a single horizon cannot establish robustness.** Simulating 1,500 books per
arm at n=120, where the "no edge" arm is pure noise:

=======  ==============  =========
require  false-positive  detection
=======  ==============  =========
1 of 5          24.80%     100.00%
2 of 5           2.53%     100.00%
3 of 5           0.13%      99.93%
4 of 5           0.00%      98.73%
5 of 5           0.00%      84.93%
=======  ==============  =========

**A model evaluated at one horizon clears the band 24.8% of the time on pure
noise.** That is the cost of the current state: nearly one in four noise models
would look temporally validated, because "it worked at 20d" *is* the
single-horizon claim the first row prices.

Three of five is where the false-positive rate collapses (2.53% → 0.13%) while
detection is essentially untouched (100.00% → 99.93%). Across 1,500 null trials
the most horizons a noise model ever cleared was 3, so the bar sits at the top of
the null.

**A claim I tested and dropped.** It is intuitive that three *adjacent* horizons
agreeing says less than three spread across the range, since their labels
overlap. Measured over 2,000 null trials per arm, adjacent pairs cleared the band
together 0.50% of the time and spread pairs 0.45% — no difference. Requiring
spread would be an unmeasured threshold dressed as a safeguard, so X6 counts
agreeing horizons without weighting where they sit. The short/long split is
reported so a reader can see the shape, but it does not gate.

**A horizon with no model is MISSING, never "no edge there."**
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    LABEL_HORIZON_SESSIONS,
    TEMPORAL_ABSENT_MEANS_NO_EDGE,
    TEMPORAL_ALPHA,
    TEMPORAL_BLOCKS_TRADES,
    TEMPORAL_FRAGILE,
    TEMPORAL_HORIZONS,
    TEMPORAL_LONG_HORIZONS,
    TEMPORAL_MIN_AGREEING,
    TEMPORAL_MIN_OBSERVATIONS,
    TEMPORAL_NOT_EVALUATED,
    TEMPORAL_REQUIRES_SPREAD,
    TEMPORAL_ROBUST,
    TEMPORAL_ROBUSTNESS_VERSION,
    TEMPORAL_SHORT_HORIZONS,
    TEMPORAL_VERDICTS,
)
from core.oos_validation import sampling_band


class TemporalRobustnessError(ValueError):
    """Raised when temporal robustness cannot be judged without guessing."""


# Per-horizon outcomes. MISSING and NO_EDGE are deliberately distinct.
TR_HORIZON_EDGE = "EDGE"
TR_HORIZON_NO_EDGE = "NO_EDGE"
TR_HORIZON_THIN = "TOO_FEW_OBSERVATIONS"
TR_HORIZON_MISSING = "MISSING"

# Why the sweep could not decide.
TR_REASON_MISSING_HORIZONS = "HORIZONS_MISSING"
TR_REASON_TOO_FEW_AGREE = "TOO_FEW_HORIZONS_AGREE"


def horizon_edge(
    accuracy: float | None,
    observations: int | None,
) -> tuple[str, float | None]:
    """Whether one horizon shows a directional edge, using X1's test.

    Returns ``(outcome, excess)`` where ``excess`` is the distance from a coin
    flip, or None when there was nothing to measure.
    """
    if accuracy is None or observations is None:
        return TR_HORIZON_MISSING, None
    observations = int(observations)
    if observations < TEMPORAL_MIN_OBSERVATIONS:
        return TR_HORIZON_THIN, None
    excess = abs(float(accuracy) - 0.5)
    band = sampling_band(observations, TEMPORAL_ALPHA)
    if excess > band:
        return TR_HORIZON_EDGE, excess
    return TR_HORIZON_NO_EDGE, excess


def evaluate_temporal_robustness(
    by_horizon: Mapping[str, Mapping[str, Any]] | None,
    *,
    estimator: str | None = None,
) -> dict:
    """Judge whether an edge holds across the required horizons.

    ``by_horizon`` maps a horizon to ``{"directional_accuracy": x,
    "observations": n}``. A horizon absent from the mapping is MISSING, which is
    not the same as showing no edge.
    """
    detail: dict[str, Any] = {
        "estimator": estimator,
        "required_horizons": list(TEMPORAL_HORIZONS),
        "min_agreeing": TEMPORAL_MIN_AGREEING,
        "min_observations": TEMPORAL_MIN_OBSERVATIONS,
        "alpha": TEMPORAL_ALPHA,
        "requires_spread": TEMPORAL_REQUIRES_SPREAD,
    }

    supplied = dict(by_horizon or {})
    unknown = sorted(set(supplied) - set(TEMPORAL_HORIZONS))
    if unknown:
        raise TemporalRobustnessError(
            f"results supplied for horizons X6 does not evaluate: {unknown}; "
            f"the required set is {list(TEMPORAL_HORIZONS)}"
        )

    horizons: dict[str, dict[str, Any]] = {}
    for horizon in TEMPORAL_HORIZONS:
        entry = supplied.get(horizon) or {}
        accuracy = entry.get("directional_accuracy")
        observations = entry.get("observations")
        outcome, excess = horizon_edge(accuracy, observations)
        horizons[horizon] = {
            "outcome": outcome,
            "directional_accuracy": accuracy,
            "observations": observations,
            "excess_over_coin_flip": excess,
            "band": (
                sampling_band(int(observations), TEMPORAL_ALPHA)
                if observations is not None and int(observations) > 0
                else None
            ),
            "sessions": LABEL_HORIZON_SESSIONS.get(horizon),
        }

    detail["horizons"] = horizons
    agreeing = [h for h, r in horizons.items() if r["outcome"] == TR_HORIZON_EDGE]
    missing = [h for h, r in horizons.items() if r["outcome"] == TR_HORIZON_MISSING]
    thin = [h for h, r in horizons.items() if r["outcome"] == TR_HORIZON_THIN]
    no_edge = [h for h, r in horizons.items() if r["outcome"] == TR_HORIZON_NO_EDGE]

    detail["agreeing_horizons"] = agreeing
    detail["missing_horizons"] = missing
    detail["thin_horizons"] = thin
    detail["no_edge_horizons"] = no_edge
    detail["evaluated_horizons"] = sorted(agreeing + no_edge)

    # Reported so a reader sees the SHAPE of the agreement. It does not gate:
    # MEASURED, adjacent and spread agreement are indistinguishable under the
    # null at this sample size.
    detail["agreeing_short"] = [h for h in agreeing if h in TEMPORAL_SHORT_HORIZONS]
    detail["agreeing_long"] = [h for h in agreeing if h in TEMPORAL_LONG_HORIZONS]

    # A MISSING HORIZON IS NOT A NEGATIVE RESULT. With horizons unevaluated the
    # sweep did not happen, whatever the tested ones showed.
    if missing or thin:
        blocked = sorted(missing + thin)
        return _report(
            TEMPORAL_NOT_EVALUATED,
            reason_code=TR_REASON_MISSING_HORIZONS,
            reason=(
                f"{len(blocked)} of {len(TEMPORAL_HORIZONS)} horizon(s) could "
                f"not be evaluated ({', '.join(blocked)}); a horizon with no "
                f"model is MISSING, never 'no edge there'. MEASURED, every "
                f"trained run is 20d, and a model judged at one horizon clears "
                f"the band 24.8% of the time on pure noise"
            ),
            **detail,
        )

    if len(agreeing) < TEMPORAL_MIN_AGREEING:
        return _report(
            TEMPORAL_FRAGILE,
            reason_code=TR_REASON_TOO_FEW_AGREE,
            reason=(
                f"only {len(agreeing)} of {len(TEMPORAL_HORIZONS)} horizons "
                f"show an edge ({', '.join(agreeing) or 'none'}), below the "
                f"{TEMPORAL_MIN_AGREEING} required; MEASURED, a noise model "
                f"reaches two horizons 2.53% of the time"
            ),
            **detail,
        )

    return _report(
        TEMPORAL_ROBUST,
        reason_code=None,
        reason=(
            f"{len(agreeing)} of {len(TEMPORAL_HORIZONS)} horizons show an "
            f"edge ({', '.join(agreeing)}), at or above the "
            f"{TEMPORAL_MIN_AGREEING} required; MEASURED, a noise model "
            f"reaches that many 0.13% of the time"
        ),
        **detail,
    )


def _report(verdict: str, **detail) -> dict:
    """One X6 answer."""
    if verdict not in TEMPORAL_VERDICTS:
        raise TemporalRobustnessError(f"unknown temporal verdict {verdict!r}")
    payload = {
        "version": TEMPORAL_ROBUSTNESS_VERSION,
        "gate": "temporal_robustness",
        "verdict": verdict,
        "absent_means_no_edge": TEMPORAL_ABSENT_MEANS_NO_EDGE,
        "blocks_trades": TEMPORAL_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, every trained run is 20d, so four of the five required horizons "
    "have no model - a DATA gap, not a structural one: the labels, the "
    "target_horizon field and the trainer argument all already exist. A model "
    "judged at ONE horizon clears the band 24.80% of the time on pure noise, "
    "against 0.13% at three of five, which is where the false-positive rate "
    "collapses while detection stays at 99.93%."
)


def temporal_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a temporal robustness report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != TEMPORAL_ROBUSTNESS_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{TEMPORAL_ROBUSTNESS_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in TEMPORAL_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X6 reports; the registry promotes")

    if report.get("absent_means_no_edge"):
        problems.append(
            "a horizon with no model must never be read as showing no edge; "
            "that reports an untested horizon as a tested one"
        )

    horizons = report.get("horizons")
    if horizons is not None:
        if not isinstance(horizons, Mapping):
            problems.append("horizons must be a mapping")
        else:
            absent = [h for h in TEMPORAL_HORIZONS if h not in horizons]
            if absent:
                problems.append(
                    f"the roadmap names {absent} and they are not reported; a "
                    f"horizon nobody tests is a horizon whose edge is unproven"
                )
            for name, entry in horizons.items():
                if not isinstance(entry, Mapping):
                    continue
                if (
                    entry.get("outcome") == TR_HORIZON_MISSING
                    and entry.get("excess_over_coin_flip") is not None
                ):
                    problems.append(
                        f"{name} is MISSING but carries a measured excess; a "
                        f"horizon with no model has nothing to measure"
                    )

    # THE CENTRAL INVARIANT: robustness cannot be claimed while horizons are
    # unevaluated.
    if verdict == TEMPORAL_ROBUST:
        missing = report.get("missing_horizons") or []
        thin = report.get("thin_horizons") or []
        if missing or thin:
            problems.append(
                f"ROBUST while {sorted(missing) + sorted(thin)} were never "
                f"evaluated; the sweep did not happen"
            )
        agreeing = report.get("agreeing_horizons") or []
        if len(agreeing) < TEMPORAL_MIN_AGREEING:
            problems.append(
                f"ROBUST on {len(agreeing)} agreeing horizon(s), below the "
                f"{TEMPORAL_MIN_AGREEING} required"
            )

    if verdict == TEMPORAL_FRAGILE and (report.get("missing_horizons") or []):
        problems.append(
            "FRAGILE while horizons were missing; 'tested and too few agreed' "
            "and 'most horizons have no model' are different answers"
        )

    if verdict == TEMPORAL_NOT_EVALUATED and not report.get("reason_code"):
        problems.append(
            "a NOT_EVALUATED verdict must name WHY: missing horizons and thin "
            "samples imply different fixes"
        )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_temporal(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one temporal robustness report."""
    lines = [
        f"Temporal robustness: {_shown(report.get('estimator'))}"
        f" -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"  require {report.get('min_agreeing')} of "
        f"{len(report.get('required_horizons') or [])} horizons"
        f"  (alpha {report.get('alpha')})",
    ]
    for horizon, entry in (report.get("horizons") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            f"    {horizon:<5} {str(entry.get('outcome')):<21}"
            f" acc {_num(entry.get('directional_accuracy'))}"
            f"  n={_shown(entry.get('observations'))}"
            f"  band +/-{_num(entry.get('band'))}"
        )
    missing = report.get("missing_horizons") or []
    if missing:
        lines.append(
            f"  NO MODEL: {', '.join(missing)}"
            f"  - missing is not 'no edge there'"
        )
    lines.append(
        f"  agreeing: {', '.join(report.get('agreeing_horizons') or []) or 'none'}"
        f"  (short: {len(report.get('agreeing_short') or [])},"
        f" long: {len(report.get('agreeing_long') or [])})"
    )
    lines.append(f"  reason: {report.get('reason')}")
    return lines


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.4f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
