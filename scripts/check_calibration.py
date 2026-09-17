"""CI drift gate for M6 calibration and uncertainty.

Proves, on every push, that the binding rule still holds:

    "Never expose arbitrary probability numbers as if they were calibrated."

1. an uncalibrated score CANNOT be obtained as a probability;
2. too little data, or a single outcome class, is refused outright;
3. both methods are monotone and bounded to [0, 1];
4. isotonic is preferred, Platt substitutes on thin folds, and the
   substitution is recorded;
5. calibration is MEASURED out-of-fold — and the out-of-fold error is
   genuinely larger than the in-sample error, which is the whole reason the
   path exists;
6. the metrics behave: perfect forecasts score 0 Brier / 0 ECE, an
   overconfident model scores badly, log loss stays finite.

Runs on synthetic data in well under a second.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration import (  # noqa: E402
    CalibrationError,
    UncalibratedProbabilityError,
    brier_score,
    calibrate_training_run,
    calibrated_probability,
    expected_calibration_error,
    fit_calibration,
    fold_dispersion,
    log_loss,
    prediction_interval,
    reliability_curve,
)
from core.config import (  # noqa: E402
    CALIBRATION_METHOD_ISOTONIC,
    CALIBRATION_METHOD_PLATT,
    CALIBRATION_MIN_ISOTONIC_SAMPLES,
    CALIBRATION_MIN_SAMPLES,
)


def _signal(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, n)
    actuals = ((scores + rng.normal(0.0, 1.0, n)) > 0).astype(float) * 0.02 - 0.01
    return scores, actuals


def _run(folds: int = 5, per_fold: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    objects = []
    for index in range(folds):
        scores = rng.normal(0.0, 1.0, per_fold)
        actuals = ((scores + rng.normal(0.0, 1.5, per_fold)) > 0).astype(float) * 0.02 - 0.01
        objects.append(SimpleNamespace(
            fold_id=index,
            predictions=[float(v) for v in scores],
            actuals=[float(v) for v in actuals],
            metrics={"directional_accuracy": 0.5 + index * 0.02},
        ))
    return SimpleNamespace(folds=objects)


def main() -> int:
    failures: list[str] = []

    def refuses(label: str, call, expected=CalibrationError) -> None:
        try:
            call()
        except expected:
            return
        failures.append(f"guard did NOT fire: {label}")

    # 1. The binding rule.
    refuses(
        "an uncalibrated score was returned as a probability",
        lambda: calibrated_probability(0.7, None),
        UncalibratedProbabilityError,
    )

    # 2. Refusals on insufficient data.
    refuses(
        "calibration on fewer than the minimum samples",
        lambda: fit_calibration(*_signal(CALIBRATION_MIN_SAMPLES - 1)),
    )
    refuses(
        "calibration on a single outcome class",
        lambda: fit_calibration(list(np.linspace(0, 1, 60)), [0.01] * 60),
    )
    refuses(
        "calibration on mismatched lengths",
        lambda: fit_calibration([0.1] * 50, [0.01] * 40),
    )
    refuses(
        "an unknown calibration method",
        lambda: fit_calibration(*_signal(200), "astrology"),
    )

    # 3. Monotone and bounded.
    scores, actuals = _signal(300)
    for method in (CALIBRATION_METHOD_ISOTONIC, CALIBRATION_METHOD_PLATT):
        fitted = fit_calibration(scores, actuals, method)
        probabilities = np.atleast_1d(fitted.apply(np.sort(scores)))
        if not np.all(np.diff(probabilities) >= -1e-12):
            failures.append(f"{method} calibration is not monotone")
        extreme = np.atleast_1d(fitted.apply([-1e6, 0.0, 1e6]))
        if not (np.all(extreme >= 0.0) and np.all(extreme <= 1.0)):
            failures.append(f"{method} produced a probability outside [0, 1]")

    # 4. Method selection and its disclosure.
    ample = fit_calibration(*_signal(CALIBRATION_MIN_ISOTONIC_SAMPLES + 50))
    if ample.method != CALIBRATION_METHOD_ISOTONIC:
        failures.append(f"ample data chose {ample.method}, expected isotonic")
    if ample.fallback_reason:
        failures.append("a non-fallback fit reported a fallback reason")
    thin = fit_calibration(*_signal(CALIBRATION_MIN_ISOTONIC_SAMPLES - 10))
    if thin.method != CALIBRATION_METHOD_PLATT:
        failures.append(f"thin data chose {thin.method}, expected the Platt fallback")
    if not thin.fallback_reason:
        failures.append("the Platt fallback did not record why it happened")

    # 5. Out-of-fold measurement, and that it is stricter than in-sample.
    run = _run()
    report = calibrate_training_run(run)
    if report.measurement_basis != "out_of_fold":
        failures.append(
            f"measurement basis is {report.measurement_basis!r}; with 5 folds it "
            f"must be out_of_fold"
        )
    pooled_predictions = [v for fold in run.folds for v in fold.predictions]
    pooled_actuals = [v for fold in run.folds for v in fold.actuals]
    in_sample = expected_calibration_error(
        fit_calibration(pooled_predictions, pooled_actuals).apply(pooled_predictions),
        pooled_actuals,
    )
    if report.expected_calibration_error <= in_sample:
        failures.append(
            f"out-of-fold ECE ({report.expected_calibration_error}) is not greater "
            f"than in-sample ECE ({in_sample}) — the measurement may be in-sample, "
            f"which flatters the model"
        )
    thin_run = _run(folds=2, per_fold=200)
    if calibrate_training_run(thin_run).measurement_basis != "in_sample":
        failures.append("a two-fold run did not label its measurement in_sample")

    # 6. Metric behaviour.
    if brier_score([1.0, 0.0], [0.05, -0.05]) != 0.0:
        failures.append("perfect forecasts did not score 0 Brier")
    if brier_score([0.0, 1.0], [0.05, -0.05]) != 1.0:
        failures.append("worst-possible forecasts did not score 1 Brier")
    if not math.isfinite(log_loss([1.0], [-0.05])):
        failures.append("log loss is infinite for a confident miss — clamping failed")
    if log_loss([0.99], [-0.05]) <= log_loss([0.6], [-0.05]):
        failures.append("log loss does not punish confident mistakes harder")

    perfect_probabilities = [0.25] * 100 + [0.75] * 100
    perfect_actuals = [0.01] * 25 + [-0.01] * 75 + [0.01] * 75 + [-0.01] * 25
    if expected_calibration_error(perfect_probabilities, perfect_actuals) > 1e-6:
        failures.append("a perfectly calibrated set did not score ~0 ECE")
    if expected_calibration_error([0.95] * 100, [0.01] * 50 + [-0.01] * 50) < 0.4:
        failures.append("an overconfident model did not score a large ECE")

    curve = reliability_curve([0.95] * 50, [0.01] * 25 + [-0.01] * 25)
    if len(curve) != 1:
        failures.append("empty reliability bins were reported instead of omitted")

    stable = fold_dispersion([0.62, 0.61, 0.63, 0.62])
    erratic = fold_dispersion([0.30, 0.90, 0.35, 0.93])
    if stable["std"] >= erratic["std"]:
        failures.append("fold dispersion does not separate stable from erratic folds")
    interval = prediction_interval([0.4, 0.5, 0.6, 0.7, 0.8])
    if not interval["lower"] <= interval["median"] <= interval["upper"]:
        failures.append("the prediction interval does not bracket its median")

    if failures:
        print("M6 calibration gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M6 calibration gate OK:")
    print("  an uncalibrated score cannot be obtained as a probability.")
    print(
        f"  isotonic preferred above {CALIBRATION_MIN_ISOTONIC_SAMPLES} samples, "
        f"Platt below, fallback disclosed; both monotone and bounded."
    )
    print(
        f"  measurement is out-of-fold and stricter than in-sample "
        f"(ECE {report.expected_calibration_error} vs {in_sample})."
    )
    print("  Brier / log loss / ECE / reliability / dispersion / interval all behave.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
