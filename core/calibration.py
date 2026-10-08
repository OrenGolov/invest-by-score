"""Calibration and uncertainty (Sprint M6 / board M4).

The binding rule:

    **Never expose arbitrary probability numbers as if they were
    calibrated.**

A raw model score is not a probability. `0.7` out of a gradient booster does
not mean "70% of these go up" — it means nothing at all until it has been
mapped through a calibration fitted on held-out data and measured against
what actually happened. This module is that mapping, its measurement, and
the refusal that stops an uncalibrated number reaching a caller.

    fold predictions (out-of-sample)
        |
        v
    fit_calibration()      isotonic, or Platt on thin folds
        |
        v
    CalibrationMap ------> calibrated_probability()
        |                        |
        v                        v
    reliability_curve()     REFUSES when uncalibrated
    brier / log_loss / ECE

What M6 requires, and where it lives:

- probability calibration -> `fit_calibration`, `CalibrationMap.apply`
- reliability curves      -> `reliability_curve`
- Brier score             -> `brier_score`
- log loss                -> `log_loss`
- calibration error       -> `expected_calibration_error`, `max_calibration_error`
- prediction intervals    -> `prediction_interval`
- uncertainty             -> `CalibrationReport.uncertainty`
- fold dispersion         -> `fold_dispersion`

Design decisions worth stating:

- **Calibration is fitted on validation folds only.** The M3 trainer retains
  each fold's out-of-sample predictions precisely so this can be true. A map
  fitted on training predictions would be calibrated to data the model had
  already seen, which is the error calibration exists to prevent.
- **Isotonic preferred, Platt for thin folds.** Isotonic regression is
  non-parametric and fits any monotone shape, but it overfits badly on small
  samples. Below `CALIBRATION_MIN_ISOTONIC_SAMPLES` the fitter falls back to
  Platt and RECORDS that it did, so a reader never has to guess which ran.
- **Too little data is a refusal, not a wider error bar.** Below
  `CALIBRATION_MIN_SAMPLES` there is no honest probability to expose, so
  `fit_calibration` raises rather than returning a map nobody should trust.
- **The mapping is monotone by construction**, so a higher raw score never
  produces a lower probability. Both methods guarantee it and a test pins it.

Pure and deterministic: no wall-clock, no randomness, no provider reads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from core.config import (
    CALIBRATION_METHOD_ISOTONIC,
    CALIBRATION_METHOD_PLATT,
    CALIBRATION_METHODS,
    CALIBRATION_MIN_ISOTONIC_SAMPLES,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_RELIABILITY_BINS,
    CALIBRATION_VERSION,
    PREDICTION_INTERVAL_LEVEL,
)

# Probabilities are clamped away from exactly 0 and 1 before log loss, which
# is otherwise infinite for a confident miss. This is a numerical guard, not
# a modelling choice.
_EPSILON = 1e-15


class CalibrationError(ValueError):
    """Raised when calibration is impossible or misused."""


class UncalibratedProbabilityError(CalibrationError):
    """Raised when an uncalibrated score is requested as a probability."""


@dataclass
class CalibrationMap:
    """A fitted, versioned mapping from raw score to calibrated probability.

    Travels with the model artifact: `calibration_version` is what the M5
    registry entry records, so a prediction can always be traced to the map
    that produced it.
    """

    method: str
    # Isotonic: the breakpoints and their fitted probabilities.
    # Platt: [a, b] for sigmoid(a * score + b).
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    sample_size: int = 0
    fitted_on: str = "validation_folds"
    calibration_version: str = CALIBRATION_VERSION
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_hash(self) -> str:
        """Deterministic identity, so a changed map is always visible."""
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def apply(self, scores: Sequence[float] | float) -> np.ndarray | float:
        """Map raw score(s) to calibrated probabilities in [0, 1]."""
        single = np.isscalar(scores)
        values = np.atleast_1d(np.asarray(scores, dtype=float))

        if self.method == CALIBRATION_METHOD_PLATT:
            a, b = self.coefficients
            # Clip the logit before exp: a huge magnitude score overflows
            # float64 and warns, even though the limit is the correct 0 or 1.
            # +/-700 is past the point where the sigmoid is 0 or 1 in float64.
            logit = np.clip(a * values + b, -700.0, 700.0)
            probabilities = 1.0 / (1.0 + np.exp(-logit))
        elif self.method == CALIBRATION_METHOD_ISOTONIC:
            # Outside the fitted range, clamp to the end knots rather than
            # extrapolating: the data says nothing about scores it never saw.
            probabilities = np.interp(
                values, self.knots_x, self.knots_y,
                left=self.knots_y[0], right=self.knots_y[-1],
            )
        else:  # pragma: no cover - constructor validates
            raise CalibrationError(f"unknown calibration method {self.method!r}")

        probabilities = np.clip(probabilities, 0.0, 1.0)
        return float(probabilities[0]) if single else probabilities


def _binary_outcomes(actuals: Sequence[float]) -> np.ndarray:
    """Direction as a 0/1 outcome. A flat move counts as not-up."""
    return (np.asarray(actuals, dtype=float) > 0.0).astype(float)


def fit_calibration(
    predictions: Sequence[float],
    actuals: Sequence[float],
    method: str | None = None,
) -> CalibrationMap:
    """Fit a calibration map on OUT-OF-SAMPLE predictions.

    `predictions` are raw model scores and `actuals` the realised forward
    returns; the target is P(return > 0). Isotonic is preferred, with Platt
    substituted on thin samples and the substitution recorded.

    Raises when there is too little data to calibrate honestly — a map
    nobody should trust is worse than an explicit refusal.
    """
    scores = np.asarray(predictions, dtype=float)
    outcomes = _binary_outcomes(actuals)
    if scores.shape != outcomes.shape:
        raise CalibrationError(
            f"predictions ({scores.shape[0]}) and actuals ({outcomes.shape[0]}) "
            f"must be the same length"
        )
    if scores.size < CALIBRATION_MIN_SAMPLES:
        raise CalibrationError(
            f"{scores.size} observations is below the "
            f"{CALIBRATION_MIN_SAMPLES}-observation minimum — there is no "
            f"honest probability to expose, so calibration is refused"
        )
    if not np.all(np.isfinite(scores)):
        count = int((~np.isfinite(scores)).sum())
        raise CalibrationError(
            f"{count} of {scores.size} predictions are NaN or infinite — a "
            f"calibration fitted on them would be meaningless, and the raw "
            f"library error would not say so"
        )
    if not np.all(np.isfinite(np.asarray(actuals, dtype=float))):
        raise CalibrationError("actuals contain NaN or infinite values")
    if np.unique(outcomes).size < 2:
        raise CalibrationError(
            "all outcomes are identical — a calibration fitted on a single "
            "class would map every score to the same probability"
        )

    fallback_reason = ""
    chosen = method
    if chosen is None:
        if scores.size < CALIBRATION_MIN_ISOTONIC_SAMPLES:
            chosen = CALIBRATION_METHOD_PLATT
            fallback_reason = (
                f"{scores.size} observations is below the "
                f"{CALIBRATION_MIN_ISOTONIC_SAMPLES}-observation isotonic "
                f"threshold; isotonic overfits thin samples, so Platt was used"
            )
        else:
            chosen = CALIBRATION_METHOD_ISOTONIC
    if chosen not in CALIBRATION_METHODS:
        raise CalibrationError(
            f"unknown calibration method {chosen!r} (known: {sorted(CALIBRATION_METHODS)})"
        )

    if chosen == CALIBRATION_METHOD_ISOTONIC:
        from sklearn.isotonic import IsotonicRegression

        fitted = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fitted.fit(scores, outcomes)
        order = np.argsort(fitted.X_thresholds_)
        return CalibrationMap(
            method=chosen,
            knots_x=[round(float(v), 10) for v in np.asarray(fitted.X_thresholds_)[order]],
            knots_y=[round(float(v), 10) for v in np.asarray(fitted.y_thresholds_)[order]],
            sample_size=int(scores.size),
            fallback_reason=fallback_reason,
        )

    from sklearn.linear_model import LogisticRegression

    fitted = LogisticRegression(max_iter=2000)
    fitted.fit(scores.reshape(-1, 1), outcomes)
    return CalibrationMap(
        method=CALIBRATION_METHOD_PLATT,
        coefficients=[
            round(float(np.ravel(fitted.coef_)[0]), 10),
            round(float(np.ravel(fitted.intercept_)[0]), 10),
        ],
        sample_size=int(scores.size),
        fallback_reason=fallback_reason,
    )


def calibrated_probability(
    score: float,
    calibration_map: CalibrationMap | None,
) -> float:
    """The only sanctioned way to turn a score into a probability.

    A missing map is a refusal, never a passthrough. This is the function
    that makes "never expose arbitrary probability numbers as if they were
    calibrated" true in code rather than in a comment.
    """
    if calibration_map is None:
        raise UncalibratedProbabilityError(
            "no calibration map: a raw model score is not a probability and "
            "must not be presented as one. Fit a calibration on validation "
            "folds first."
        )
    return float(calibration_map.apply(float(score)))


def brier_score(probabilities: Sequence[float], actuals: Sequence[float]) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better."""
    p = np.asarray(probabilities, dtype=float)
    y = _binary_outcomes(actuals)
    if p.size == 0:
        raise CalibrationError("brier score needs at least one observation")
    return round(float(np.mean((p - y) ** 2)), 8)


def log_loss(probabilities: Sequence[float], actuals: Sequence[float]) -> float:
    """Negative log likelihood. Punishes confident mistakes hard."""
    p = np.clip(np.asarray(probabilities, dtype=float), _EPSILON, 1.0 - _EPSILON)
    y = _binary_outcomes(actuals)
    if p.size == 0:
        raise CalibrationError("log loss needs at least one observation")
    return round(float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))), 8)


def reliability_curve(
    probabilities: Sequence[float],
    actuals: Sequence[float],
    bins: int = CALIBRATION_RELIABILITY_BINS,
) -> list[dict[str, Any]]:
    """Predicted vs observed frequency, per probability bin.

    The honest picture of calibration: in the bin where the model said ~70%,
    how often did it actually happen? Empty bins are omitted rather than
    reported as zero — no observations is not the same as never happened.
    """
    p = np.asarray(probabilities, dtype=float)
    y = _binary_outcomes(actuals)
    if p.size == 0:
        raise CalibrationError("a reliability curve needs at least one observation")
    if bins < 2:
        raise CalibrationError(f"bins must be at least 2, got {bins}")

    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[dict[str, Any]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        # Include the right edge in the final bin so p == 1.0 is counted.
        mask = (p >= low) & (p < high) if index < bins - 1 else (p >= low) & (p <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append({
            "bin_lower": round(float(low), 4),
            "bin_upper": round(float(high), 4),
            "count": count,
            "mean_predicted": round(float(np.mean(p[mask])), 6),
            "observed_frequency": round(float(np.mean(y[mask])), 6),
            "gap": round(float(np.mean(p[mask]) - np.mean(y[mask])), 6),
        })
    return curve


def expected_calibration_error(
    probabilities: Sequence[float],
    actuals: Sequence[float],
    bins: int = CALIBRATION_RELIABILITY_BINS,
) -> float:
    """Count-weighted mean gap between predicted and observed frequency."""
    curve = reliability_curve(probabilities, actuals, bins)
    total = sum(entry["count"] for entry in curve)
    if total == 0:
        raise CalibrationError("no populated bins — calibration error is undefined")
    weighted = sum(abs(entry["gap"]) * entry["count"] for entry in curve)
    return round(float(weighted / total), 8)


def max_calibration_error(
    probabilities: Sequence[float],
    actuals: Sequence[float],
    bins: int = CALIBRATION_RELIABILITY_BINS,
) -> float:
    """Worst single-bin gap. ECE can hide a badly wrong region; this cannot."""
    curve = reliability_curve(probabilities, actuals, bins)
    if not curve:
        raise CalibrationError("no populated bins — calibration error is undefined")
    return round(float(max(abs(entry["gap"]) for entry in curve)), 8)


def fold_dispersion(fold_values: Sequence[float]) -> dict[str, float]:
    """Spread of a metric across folds — the honest uncertainty signal.

    A model that scores 0.62 on every fold and one that averages 0.62 by
    swinging 0.30-0.90 are not equally trustworthy, and a single pooled
    number cannot tell them apart.
    """
    values = np.asarray(list(fold_values), dtype=float)
    if values.size == 0:
        raise CalibrationError("fold dispersion needs at least one fold")
    return {
        "folds": int(values.size),
        "mean": round(float(np.mean(values)), 8),
        "std": round(float(np.std(values, ddof=0)), 8),
        "min": round(float(np.min(values)), 8),
        "max": round(float(np.max(values)), 8),
        "range": round(float(np.max(values) - np.min(values)), 8),
    }


def prediction_interval(
    fold_values: Sequence[float],
    level: float = PREDICTION_INTERVAL_LEVEL,
) -> dict[str, float]:
    """An empirical interval from fold outcomes.

    Derived from the observed spread across folds rather than a normality
    assumption: with a handful of folds, a parametric interval would imply
    precision the data does not support.
    """
    values = np.asarray(list(fold_values), dtype=float)
    if values.size == 0:
        raise CalibrationError("a prediction interval needs at least one fold")
    if not 0.0 < level < 1.0:
        raise CalibrationError(f"level must be strictly between 0 and 1, got {level}")
    tail = (1.0 - level) / 2.0
    return {
        "level": round(float(level), 4),
        "lower": round(float(np.quantile(values, tail)), 8),
        "upper": round(float(np.quantile(values, 1.0 - tail)), 8),
        "median": round(float(np.median(values)), 8),
        "folds": int(values.size),
    }


@dataclass
class CalibrationReport:
    """Everything M6 requires about one model's calibration."""

    calibration_map: CalibrationMap
    brier: float
    logloss: float
    expected_calibration_error: float
    max_calibration_error: float
    reliability: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: dict[str, float] = field(default_factory=dict)
    interval: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    # "out_of_fold" (honest) or "in_sample" (too few folds to hold one out).
    # Named explicitly because an in-sample ECE flatters the model badly.
    measurement_basis: str = "out_of_fold"
    calibration_version: str = CALIBRATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration_map"] = self.calibration_map.to_dict()
        return payload


def _out_of_fold_probabilities(run, method: str | None):
    """Leave-one-fold-out probabilities, or None when there are too few folds.

    For each fold: fit a calibration on the OTHER folds, then apply it to
    this one. No observation is ever scored by a map that saw it, so the
    resulting ECE/Brier/log-loss are honest.
    """
    folds = [f for f in run.folds if f.predictions and f.actuals]
    if len(folds) < 3:
        return None

    scored: list[float] = []
    held_actuals: list[float] = []
    for index, fold in enumerate(folds):
        others = [f for position, f in enumerate(folds) if position != index]
        train_predictions = [v for f in others for v in f.predictions]
        train_actuals = [v for f in others for v in f.actuals]
        try:
            fitted = fit_calibration(train_predictions, train_actuals, method)
        except CalibrationError:
            # A fold-out split that cannot be calibrated (too few samples or
            # a single outcome class) is skipped, not silently imputed.
            continue
        scored.extend(np.atleast_1d(fitted.apply(fold.predictions)).tolist())
        held_actuals.extend(fold.actuals)

    if not scored:
        return None
    return np.asarray(scored, dtype=float), held_actuals


def calibrate_training_run(run, method: str | None = None) -> CalibrationReport:
    """Fit and measure calibration for a completed M3 training run.

    Uses the pooled out-of-sample fold predictions the trainer retained, and
    measures the fitted map on those same out-of-sample observations. Fold
    dispersion and the prediction interval come from per-fold directional
    accuracy, so the uncertainty reflects fold-to-fold variation rather than
    a single pooled average.
    """
    if not getattr(run, "folds", None):
        raise CalibrationError("training run has no folds — nothing to calibrate")

    predictions: list[float] = []
    actuals: list[float] = []
    for fold in run.folds:
        predictions.extend(fold.predictions or [])
        actuals.extend(fold.actuals or [])
    if not predictions:
        raise CalibrationError(
            "training run retained no fold predictions — calibration must be "
            "fitted on out-of-sample values, which were not recorded"
        )

    calibration_map = fit_calibration(predictions, actuals, method)

    # Measuring the map on the data it was fitted to reports in-sample
    # calibration, which is near-perfect by construction — isotonic in
    # particular can drive ECE to 0.0 while being badly calibrated on unseen
    # data. So the REPORTED metrics come from leave-one-fold-out: for each
    # fold, fit on the other folds and score that fold. The shipped map is
    # still the all-data fit; only the measurement is held out.
    probabilities = _out_of_fold_probabilities(run, method)
    if probabilities is None:
        # Too few folds to hold one out. Fall back to the in-sample fit and
        # say so, rather than silently reporting flattering numbers.
        probabilities = np.atleast_1d(calibration_map.apply(predictions))
        measurement_basis = "in_sample"
    else:
        probabilities, actuals = probabilities
        measurement_basis = "out_of_fold"

    fold_accuracies = [
        float(fold.metrics["directional_accuracy"])
        for fold in run.folds
        if "directional_accuracy" in (fold.metrics or {})
    ]
    return CalibrationReport(
        calibration_map=calibration_map,
        brier=brier_score(probabilities, actuals),
        logloss=log_loss(probabilities, actuals),
        expected_calibration_error=expected_calibration_error(probabilities, actuals),
        max_calibration_error=max_calibration_error(probabilities, actuals),
        reliability=reliability_curve(probabilities, actuals),
        uncertainty=fold_dispersion(fold_accuracies) if fold_accuracies else {},
        interval=prediction_interval(fold_accuracies) if fold_accuracies else {},
        sample_size=len(probabilities),
        measurement_basis=measurement_basis,
    )
