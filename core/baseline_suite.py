"""Baseline model suite and the incumbent rule (Sprint M4).

The M4 rule, stated plainly:

    **No advanced model is promoted without beating the incumbent
    out-of-sample.**

Establishing baselines is only half the task. The half that matters is making
the rule enforceable, so this module answers one question precisely:

    given these runs, which model — if any — has EARNED promotion?

and refuses to launder a loss into a win.

Design decisions worth stating:

- **The incumbent is the best SIMPLE baseline, not the best model.** A
  trained model must beat `historical_mean` / `momentum` / `mean_reversion`
  before it is interesting at all. Comparing a boosted tree only against
  ridge would let a whole family of complexity in through the side door.
- **Comparison is like-for-like.** Two runs are only comparable when they
  share a dataset hash, a target horizon and a feature set. Comparing runs
  over different data is the most common way a "win" turns out to be an
  artefact, so `comparison_problems` refuses it rather than trusting the
  caller.
- **A margin is required, not just a higher number.** Beating the incumbent
  by 0.0001 on one fold is noise. A candidate must win by at least
  `BASELINE_PROMOTION_MARGIN` and must not lose on a majority of folds.
- **"Nothing earned promotion" is a first-class result**, returned as a
  populated verdict with reasons — never an exception and never a silently
  chosen best-of.

The output feeds the M2 promotion gate directly: `promotion_comparison`
emits exactly the `oos_comparison` shape `ModelRegistry.promote` demands, so
a promotion can only ever be backed by a real measured comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from core.config import (
    BASELINE_MIN_WINNING_FOLD_RATIO,
    BASELINE_PROMOTION_MARGIN,
    BASELINE_SUITE_VERSION,
)

# The simple baselines a trained model must beat to be interesting at all.
SIMPLE_BASELINES: tuple[str, ...] = ("historical_mean", "momentum", "mean_reversion")

# Metrics where a LOWER value is better.
_LOWER_IS_BETTER = frozenset({"mae", "rmse", "brier_score", "log_loss"})


class BaselineSuiteError(ValueError):
    """Raised when a comparison is invalid or impossible to make."""


def higher_is_better(metric: str) -> bool:
    """Whether a larger value of `metric` is an improvement."""
    return metric not in _LOWER_IS_BETTER


@dataclass
class Verdict:
    """The outcome of comparing a candidate against its incumbent."""

    candidate: str
    incumbent: str
    metric: str
    candidate_value: float
    incumbent_value: float
    margin: float
    winning_folds: int
    total_folds: int
    promoted: bool
    reasons: list[str] = field(default_factory=list)
    suite_version: str = BASELINE_SUITE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def comparison_problems(candidate, incumbent, metric: str) -> list[str]:
    """Refuse a comparison that is not like-for-like.

    Two runs are comparable only when they scored the same data, the same
    horizon and the same feature set. Anything else makes the "win"
    uninterpretable.
    """
    problems: list[str] = []
    if candidate is None or incumbent is None:
        return ["both a candidate and an incumbent run are required"]
    if candidate.estimator == incumbent.estimator:
        problems.append(
            f"{candidate.estimator!r} cannot be compared against itself"
        )
    if candidate.dataset_hash != incumbent.dataset_hash:
        problems.append(
            f"dataset hashes differ ({candidate.dataset_hash[:12]}... vs "
            f"{incumbent.dataset_hash[:12]}...) — runs over different data are "
            f"not comparable"
        )
    if candidate.target_horizon != incumbent.target_horizon:
        problems.append(
            f"target horizons differ ({candidate.target_horizon} vs "
            f"{incumbent.target_horizon})"
        )
    if candidate.feature_set_hash != incumbent.feature_set_hash:
        problems.append("feature sets differ — the comparison is not like-for-like")
    for run, label in ((candidate, "candidate"), (incumbent, "incumbent")):
        if metric not in (run.metrics or {}):
            problems.append(f"{label} run does not report {metric!r}")
        if not run.folds:
            problems.append(f"{label} run has no folds — nothing was validated")
    return problems


def _fold_wins(candidate, incumbent, metric: str) -> tuple[int, int]:
    """How many folds the candidate won, pairing folds by fold_id."""
    incumbent_by_id = {fold.fold_id: fold for fold in incumbent.folds}
    better = higher_is_better(metric)
    wins = 0
    compared = 0
    for fold in candidate.folds:
        other = incumbent_by_id.get(fold.fold_id)
        if other is None:
            continue
        mine = fold.metrics.get(metric)
        theirs = other.metrics.get(metric)
        if mine is None or theirs is None:
            continue
        compared += 1
        if (mine > theirs) if better else (mine < theirs):
            wins += 1
    return wins, compared


def compare_runs(candidate, incumbent, metric: str = "directional_accuracy") -> Verdict:
    """Decide whether `candidate` has earned promotion over `incumbent`.

    A candidate must clear three bars: beat the incumbent on the metric, beat
    it by at least the promotion margin, and win a majority of folds. Failing
    any of them is a recorded verdict with reasons, not an exception.
    """
    problems = comparison_problems(candidate, incumbent, metric)
    if problems:
        raise BaselineSuiteError(
            f"cannot compare {getattr(candidate, 'estimator', '?')!r} against "
            f"{getattr(incumbent, 'estimator', '?')!r}: " + "; ".join(problems)
        )

    candidate_value = float(candidate.metrics[metric])
    incumbent_value = float(incumbent.metrics[metric])
    better = higher_is_better(metric)
    margin = (
        candidate_value - incumbent_value if better else incumbent_value - candidate_value
    )
    wins, compared = _fold_wins(candidate, incumbent, metric)
    fold_ratio = (wins / compared) if compared else 0.0

    reasons: list[str] = []
    if margin <= 0:
        direction = "higher" if better else "lower"
        reasons.append(
            f"candidate {candidate_value:.6f} does not beat incumbent "
            f"{incumbent_value:.6f} on {metric!r} ({direction} is better)"
        )
    elif margin < BASELINE_PROMOTION_MARGIN:
        reasons.append(
            f"margin {margin:.6f} is below the required "
            f"{BASELINE_PROMOTION_MARGIN} — a win this small is noise"
        )
    if compared == 0:
        reasons.append("no folds could be paired between the two runs")
    elif fold_ratio < BASELINE_MIN_WINNING_FOLD_RATIO:
        reasons.append(
            f"candidate won {wins}/{compared} folds "
            f"({fold_ratio:.2f}), below the required "
            f"{BASELINE_MIN_WINNING_FOLD_RATIO} — the edge is not consistent"
        )

    return Verdict(
        candidate=candidate.estimator,
        incumbent=incumbent.estimator,
        metric=metric,
        candidate_value=round(candidate_value, 8),
        incumbent_value=round(incumbent_value, 8),
        margin=round(margin, 8),
        winning_folds=wins,
        total_folds=compared,
        promoted=not reasons,
        reasons=reasons,
    )


def best_simple_baseline(runs: dict, metric: str = "directional_accuracy"):
    """The strongest simple baseline — the bar a trained model must clear."""
    available = [runs[name] for name in SIMPLE_BASELINES if name in runs]
    if not available:
        raise BaselineSuiteError(
            f"no simple baseline present (expected one of {list(SIMPLE_BASELINES)}) "
            f"— a trained model cannot be judged without one"
        )
    better = higher_is_better(metric)
    return sorted(
        available,
        key=lambda run: float(run.metrics.get(metric, 0.0)),
        reverse=better,
    )[0]


def evaluate_suite(runs: dict, metric: str = "directional_accuracy") -> dict[str, Any]:
    """Judge every trained model against the best simple baseline.

    Returns the incumbent, a verdict per candidate, and the promotion
    decision. When nothing clears the bar that is reported plainly — it is a
    valid result, and the most likely one early on.
    """
    if not runs:
        raise BaselineSuiteError("no runs to evaluate")
    incumbent = best_simple_baseline(runs, metric)
    verdicts = {
        name: compare_runs(run, incumbent, metric)
        for name, run in sorted(runs.items())
        if name != incumbent.estimator
    }
    promoted = sorted(name for name, verdict in verdicts.items() if verdict.promoted)
    return {
        "suite_version": BASELINE_SUITE_VERSION,
        "metric": metric,
        "higher_is_better": higher_is_better(metric),
        "incumbent": incumbent.estimator,
        "incumbent_value": round(float(incumbent.metrics[metric]), 8),
        "dataset_hash": incumbent.dataset_hash,
        "verdicts": {name: verdict.to_dict() for name, verdict in verdicts.items()},
        "promotable": promoted,
        "any_promotable": bool(promoted),
        "summary": (
            f"{len(promoted)} of {len(verdicts)} candidates beat the "
            f"{incumbent.estimator} baseline on {metric}"
            if promoted
            else (
                f"no candidate beat the {incumbent.estimator} baseline on "
                f"{metric} — no model has earned promotion"
            )
        ),
    }


def promotion_comparison(
    verdict: Verdict,
    candidate_version: str,
    incumbent_version: str,
) -> dict[str, Any]:
    """Turn a winning verdict into the M2 promotion gate's `oos_comparison`.

    Refuses to emit a comparison for a verdict that did not win: the point of
    the gate is that a promotion must be backed by a real measured result, so
    the two must not be able to disagree.
    """
    if not verdict.promoted:
        raise BaselineSuiteError(
            f"{verdict.candidate!r} did not earn promotion "
            f"({'; '.join(verdict.reasons)}) — refusing to emit a promotion "
            f"comparison for a losing candidate"
        )
    return {
        "primary_metric": verdict.metric,
        "candidate_value": verdict.candidate_value,
        "incumbent_value": verdict.incumbent_value,
        "sample": verdict.total_folds,
        "candidate_version": candidate_version,
        "incumbent_version": incumbent_version,
        "higher_is_better": higher_is_better(verdict.metric),
        "margin": verdict.margin,
        "winning_folds": f"{verdict.winning_folds}/{verdict.total_folds}",
        "suite_version": BASELINE_SUITE_VERSION,
    }
