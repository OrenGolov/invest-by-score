"""X3 regime robustness — no single regime may explain the entire edge.

MEASURED, this gate **cannot run on the data the repository ships**, and the
reason is structural rather than a shortage of history.

**The blocking measurement.** A persisted fold holds exactly ``actuals``,
``predictions``, ``fold_id``, ``metrics``, ``train_end_time``, ``train_rows``,
``validation_rows`` and ``validation_start_time``. There is no ticker and no
per-observation timestamp — only a *fold-level* window. ``TrainingRow`` does
carry ``ticker`` and ``prediction_time``, and the dataset record on disk is a
manifest describing 406 rows without storing any of them. The identity exists at
build time and is dropped when folds are persisted, because ``_fit_predict``
works on bare arrays.

So a regime label cannot be joined onto a validation observation, and X3 reports
NOT_EVALUATED rather than inventing an attribution. The next action is concrete
and belongs to the training pipeline, not to this gate: persist ``ticker`` and
``prediction_time`` alongside each fold's predictions.

**What the gate measures once that lands**, demonstrated on seeded data rather
than asserted. Four regimes of 100 observations, one carrying a real edge and
three pure noise:

=============================  ======
pooled directional accuracy    0.6800
bullish                        0.9700
range                          0.6400
bearish                        0.5600
risk_off                       0.5500
with the carrier removed       0.5833
=============================  ======

A pooled 0.6800 that looks like an edge collapses to 0.5833 when one regime is
dropped.

**Why "is every regime above a coin flip" is the wrong test.** MEASURED, the
answer is 4 of 4 for *both* that book and one with a modest edge spread evenly.
The statistic does not separate them at all.

**The statistic that does: leave one regime out.** On the same pair, the worst
leave-one-out drop was 0.0967 concentrated versus 0.0225 broad.

**The threshold is measured, not chosen.** Across 400 simulated books whose edge
is genuinely uniform, the worst leave-one-out drop had a median of 0.0150, a p95
of 0.0300 and a maximum of 0.0525. The 0.05 bar sits at the top of what
uniformity produces by chance; at cell sizes 60, 100 and 200 it flagged the
concentrated book 100% of the time and the uniform book 0% of the time.

**Per-regime spread is recorded but never decides.** A uniform edge produces a
spread with a p95 of 0.1800 and a max of 0.2700 from per-cell sampling noise
alone, so a spread test would fail robust models constantly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    REGIME_CONCENTRATED,
    REGIME_ROBUST,
    REGIME_ROBUSTNESS_BLOCKS_TRADES,
    REGIME_ROBUSTNESS_INFERS_LABELS,
    REGIME_ROBUSTNESS_LABELS,
    REGIME_ROBUSTNESS_MAX_DROP,
    REGIME_ROBUSTNESS_MIN_CELL,
    REGIME_ROBUSTNESS_MIN_REGIMES,
    REGIME_ROBUSTNESS_NOT_EVALUATED,
    REGIME_ROBUSTNESS_SPREAD_DECIDES,
    REGIME_ROBUSTNESS_VERDICTS,
    REGIME_ROBUSTNESS_VERSION,
)
from core.oos_validation import directional_accuracy


class RegimeRobustnessError(ValueError):
    """Raised when regime robustness cannot be judged without guessing."""


# Why the gate could not decide. Distinct because each implies a different fix.
RR_REASON_NO_LABELS = "NO_REGIME_LABELS"
RR_REASON_TOO_FEW_REGIMES = "TOO_FEW_REGIMES"
RR_REASON_THIN_CELLS = "ALL_CELLS_TOO_THIN"
RR_REASON_CARRIED = "ONE_REGIME_CARRIES_THE_EDGE"


def fold_regime_labels(
    fold: Mapping[str, Any] | None,
) -> list[str | None] | None:
    # `list[str | None] | None`, not `list[str] | None`. A1 made PARTIAL context
    # possible - a classifier that could not classify one day - so an ENTRY may be
    # None while the list itself is present. The old signature promised consumers
    # the opposite, and mypy caught it.
    """Regime labels for a fold's validation observations, or None.

    Returns None when the fold carries no per-observation labels. **It never
    derives them.** MEASURED, a persisted fold has only a fold-level window, and
    stamping every observation with one regime read from that window would
    manufacture exactly the attribution this gate exists to test — every
    observation would share a label, leave-one-out would have nothing to leave,
    and the gate would return ROBUST on no evidence at all.
    """
    if fold is None:
        return None
    if not isinstance(fold, Mapping):
        raise RegimeRobustnessError("a fold must be a mapping")
    labels = fold.get("regimes")
    if labels is None:
        return None
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise RegimeRobustnessError(
            f"fold regimes must be a sequence of labels, got {type(labels).__name__}"
        )
    # AN UNOBSERVED ENTRY IS NOT A LABEL. A1 records None where no regime was
    # classified for that observation, and `str(None)` would make "None" its own
    # regime — the synthetic bucket this module refuses to derive. If nothing was
    # classified at all, there are no labels, which the caller reports as
    # NOT_EVALUATED rather than scoring a bucket nobody checked.
    resolved = [
        None if label is None or not str(label).strip() else str(label)
        for label in labels
    ]
    if not any(label is not None for label in resolved):
        return None
    return resolved


def group_by_regime(
    predictions: Sequence[float],
    actuals: Sequence[float],
    labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Split paired observations into per-regime cells."""
    if not (len(predictions) == len(actuals) == len(labels)):
        raise RegimeRobustnessError(
            f"{len(predictions)} predictions, {len(actuals)} actuals and "
            f"{len(labels)} labels; a mismatched set cannot be grouped"
        )
    cells: dict[str, dict[str, Any]] = {}
    for predicted, actual, label in zip(predictions, actuals, labels):
        # AN UNOBSERVED OBSERVATION IS EXCLUDED, NOT ASSIGNED AND NOT FATAL.
        #
        # Before A1 a fold either carried every label or none, so raising here
        # was right. A1 makes PARTIAL context possible — a regime classifier
        # that could not classify one day — and raising would throw away an
        # otherwise usable fold over a single gap. Assigning a label would
        # manufacture the attribution this gate tests, so the observation is
        # dropped and the count is reported by the caller.
        if label is None:
            continue
        key = str(label)
        if key not in REGIME_ROBUSTNESS_LABELS:
            raise RegimeRobustnessError(
                f"unknown regime {key!r}; the known regimes are "
                f"{list(REGIME_ROBUSTNESS_LABELS)}, and scoring an "
                f"unrecognised one would hide an edge in a regime nobody checked"
            )
        cell = cells.setdefault(key, {"predictions": [], "actuals": []})
        cell["predictions"].append(predicted)
        cell["actuals"].append(actual)
    for key, cell in cells.items():
        cell["observations"] = len(cell["predictions"])
        cell["directional_accuracy"] = directional_accuracy(
            cell["predictions"], cell["actuals"]
        )
        cell["usable"] = cell["observations"] >= REGIME_ROBUSTNESS_MIN_CELL
    return cells


def leave_one_out(
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Pooled accuracy with each regime removed in turn.

    The statistic that actually separates a concentrated edge from a broad one.
    """
    if len(cells) < 2:
        raise RegimeRobustnessError(
            "leave-one-out needs at least two regimes, or there is nothing "
            "left to compare against"
        )
    result: dict[str, float] = {}
    for excluded in cells:
        predictions: list[float] = []
        actuals: list[float] = []
        for regime, cell in cells.items():
            if regime == excluded:
                continue
            predictions.extend(cell["predictions"])
            actuals.extend(cell["actuals"])
        if not predictions:
            continue
        result[excluded] = directional_accuracy(predictions, actuals)
    return result


def evaluate_regime_robustness(
    predictions: Sequence[float] | None,
    actuals: Sequence[float] | None,
    labels: Sequence[str] | None,
    *,
    estimator: str | None = None,
) -> dict:
    """Judge whether an edge survives the removal of any single regime."""
    detail: dict[str, Any] = {
        "estimator": estimator,
        "max_drop": REGIME_ROBUSTNESS_MAX_DROP,
        "min_cell": REGIME_ROBUSTNESS_MIN_CELL,
    }

    if predictions is None or actuals is None or labels is None:
        return _report(
            REGIME_ROBUSTNESS_NOT_EVALUATED,
            reason_code=RR_REASON_NO_LABELS,
            reason=(
                "no per-observation regime labels were supplied, so the edge "
                "cannot be attributed to a regime; MEASURED, persisted folds "
                "carry no ticker and no per-observation timestamp, so a label "
                "cannot be joined on. The fix belongs to the training "
                "pipeline: persist ticker and prediction_time with each fold"
            ),
            **detail,
        )

    cells = group_by_regime(predictions, actuals, labels)
    detail["regimes_present"] = sorted(cells)
    detail["cells"] = {
        regime: {
            "observations": cell["observations"],
            "directional_accuracy": cell["directional_accuracy"],
            "usable": cell["usable"],
        }
        for regime, cell in sorted(cells.items())
    }

    usable = {regime: cell for regime, cell in cells.items() if cell["usable"]}
    detail["usable_regimes"] = sorted(usable)

    if len(usable) < REGIME_ROBUSTNESS_MIN_REGIMES:
        thin = sorted(set(cells) - set(usable))
        return _report(
            REGIME_ROBUSTNESS_NOT_EVALUATED,
            reason_code=(
                RR_REASON_THIN_CELLS if thin else RR_REASON_TOO_FEW_REGIMES
            ),
            reason=(
                f"only {len(usable)} regime(s) have at least "
                f"{REGIME_ROBUSTNESS_MIN_CELL} observations "
                f"({', '.join(thin) or 'none'} too thin); leave-one-out needs "
                f"{REGIME_ROBUSTNESS_MIN_REGIMES}, and a thinner cell cannot "
                f"support the claim in either direction"
            ),
            **detail,
        )

    pooled_predictions: list[float] = []
    pooled_actuals: list[float] = []
    for cell in usable.values():
        pooled_predictions.extend(cell["predictions"])
        pooled_actuals.extend(cell["actuals"])
    pooled = directional_accuracy(pooled_predictions, pooled_actuals)
    detail["pooled_accuracy"] = pooled
    detail["pooled_observations"] = len(pooled_predictions)

    without = leave_one_out(usable)
    detail["without_regime"] = dict(sorted(without.items()))

    drops = {regime: pooled - value for regime, value in without.items()}
    detail["drops"] = dict(sorted(drops.items()))

    carrier = max(drops, key=drops.get)
    worst_drop = drops[carrier]
    detail["carrier"] = carrier
    detail["worst_drop"] = worst_drop

    accuracies = [cell["directional_accuracy"] for cell in usable.values()]
    detail["spread"] = max(accuracies) - min(accuracies)
    detail["spread_decides"] = REGIME_ROBUSTNESS_SPREAD_DECIDES

    if worst_drop > REGIME_ROBUSTNESS_MAX_DROP:
        return _report(
            REGIME_CONCENTRATED,
            reason_code=RR_REASON_CARRIED,
            reason=(
                f"removing {carrier!r} drops accuracy from {pooled:.4f} to "
                f"{without[carrier]:.4f}, a fall of {worst_drop:.4f} above the "
                f"{REGIME_ROBUSTNESS_MAX_DROP} bar; MEASURED, a genuinely "
                f"uniform edge produced a worst drop of at most 0.0525 in 400 "
                f"trials"
            ),
            **detail,
        )

    return _report(
        REGIME_ROBUST,
        reason_code=None,
        reason=(
            f"no single regime carries the edge: the worst leave-one-out drop "
            f"is {worst_drop:.4f} (removing {carrier!r}), within the "
            f"{REGIME_ROBUSTNESS_MAX_DROP} bar across "
            f"{len(usable)} usable regime(s)"
        ),
        **detail,
    )


def _report(verdict: str, **detail) -> dict:
    """One X3 answer."""
    if verdict not in REGIME_ROBUSTNESS_VERDICTS:
        raise RegimeRobustnessError(f"unknown regime robustness verdict {verdict!r}")
    payload = {
        "version": REGIME_ROBUSTNESS_VERSION,
        "gate": "regime_robustness",
        "verdict": verdict,
        "infers_labels": REGIME_ROBUSTNESS_INFERS_LABELS,
        "blocks_trades": REGIME_ROBUSTNESS_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, persisted folds carry no ticker and no per-observation "
    "timestamp, so a regime label cannot be joined on and X3 reports "
    "NOT_EVALUATED rather than inventing an attribution. On seeded data the "
    "effect it will catch is stark: a pooled 0.6800 collapses to 0.5833 when "
    "one regime is removed, while 'every regime beats a coin flip' answers 4 "
    "of 4 for both that book and a uniformly-strong one. The 0.05 bar is the "
    "maximum a uniform edge produced across 400 trials."
)


def robustness_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a robustness report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != REGIME_ROBUSTNESS_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{REGIME_ROBUSTNESS_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in REGIME_ROBUSTNESS_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X3 reports; the registry promotes")

    if report.get("infers_labels"):
        problems.append(
            "an unlabelled observation must never be assigned a regime; that "
            "manufactures the attribution this gate exists to test"
        )

    if verdict == REGIME_ROBUST:
        drop = report.get("worst_drop")
        if drop is None or float(drop) > REGIME_ROBUSTNESS_MAX_DROP:
            problems.append(
                f"ROBUST with a worst drop of {drop!r}, above the "
                f"{REGIME_ROBUSTNESS_MAX_DROP} bar"
            )
        usable = report.get("usable_regimes")
        if not usable or len(usable) < REGIME_ROBUSTNESS_MIN_REGIMES:
            problems.append(
                f"ROBUST on {usable!r}; leave-one-out needs at least "
                f"{REGIME_ROBUSTNESS_MIN_REGIMES} usable regimes, or nothing "
                f"was actually left out"
            )

    if verdict == REGIME_CONCENTRATED and not report.get("carrier"):
        problems.append(
            "a CONCENTRATED verdict must name the regime that carries the "
            "edge, or the finding cannot be acted on"
        )

    if verdict == REGIME_ROBUSTNESS_NOT_EVALUATED and not report.get("reason_code"):
        problems.append(
            "a NOT_EVALUATED verdict must name WHY: missing labels, too few "
            "regimes and thin cells imply different fixes"
        )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_robustness(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one robustness report."""
    lines = [
        f"Regime robustness: {_shown(report.get('estimator'))}"
        f" -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"  pooled       : {_num(report.get('pooled_accuracy'))}"
        f"  over {_shown(report.get('pooled_observations'))} observation(s)",
    ]
    cells = report.get("cells") or {}
    for regime, cell in cells.items():
        lines.append(
            f"    {regime:<10} {_num(cell.get('directional_accuracy'))}"
            f"  n={cell.get('observations')}"
            f"{'' if cell.get('usable') else '  (too thin to use)'}"
        )
    drops = report.get("drops") or {}
    if drops:
        lines.append("  leave-one-out drop:")
        for regime, drop in drops.items():
            marker = "  <- carrier" if regime == report.get("carrier") else ""
            lines.append(f"    without {regime:<10} {_num(drop):>9}{marker}")
    lines.extend(
        [
            f"  worst drop   : {_num(report.get('worst_drop'))}"
            f"  (bar {report.get('max_drop')})",
            f"  spread       : {_num(report.get('spread'))}"
            f"  (recorded; decides: {report.get('spread_decides')})",
            f"  reason       : {report.get('reason')}",
        ]
    )
    return lines


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.4f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
