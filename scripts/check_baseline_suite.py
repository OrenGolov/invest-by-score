"""CI drift gate for the M4 baseline model suite.

Proves, on every push, that the incumbent rule still holds:

1. every M4 baseline family exists (historical mean, momentum, mean
   reversion, linear/ridge, logistic, tree, boosting);
2. the incumbent is the best SIMPLE baseline, not the best model;
3. the rule can PASS — a genuine consistent winner is promoted;
4. the rule REFUSES a loss, a thin margin, and a one-lucky-fold win;
5. comparisons must be like-for-like;
6. a losing candidate cannot emit a promotion comparison;
7. a real comparison is accepted by the M2 promotion gate end to end.

Runs entirely on synthetic runs, so it is fast — the expensive real-data
training is covered by the M3 gate and the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.baseline_suite import (  # noqa: E402
    SIMPLE_BASELINES,
    BaselineSuiteError,
    best_simple_baseline,
    compare_runs,
    comparison_problems,
    evaluate_suite,
    promotion_comparison,
)
from core.config import BASELINE_PROMOTION_MARGIN  # noqa: E402
from core.training import BASELINE_ESTIMATORS, _FAMILY_BY_ESTIMATOR  # noqa: E402

_METRIC = "directional_accuracy"

# Every family the M4 task names must have an estimator behind it.
_REQUIRED_FAMILIES = (
    "baseline_mean", "momentum", "mean_reversion", "linear", "logistic",
    "tree", "boosting",
)


def _run(estimator, value, folds=None, **overrides):
    values = folds if folds is not None else [value] * 5
    payload = {
        "dataset_hash": "d" * 64,
        "target_horizon": "20d",
        "feature_set_hash": "f" * 64,
    }
    payload.update(overrides)
    return SimpleNamespace(
        estimator=estimator,
        metrics={_METRIC: value},
        folds=[
            SimpleNamespace(fold_id=i, metrics={_METRIC: v}) for i, v in enumerate(values)
        ],
        **payload,
    )


def main() -> int:
    failures: list[str] = []

    # 1. Every M4 family is present.
    families = {_FAMILY_BY_ESTIMATOR[name] for name in BASELINE_ESTIMATORS}
    for required in _REQUIRED_FAMILIES:
        if required not in families:
            failures.append(
                f"M4 requires a {required!r} baseline but no estimator declares it"
            )
    for simple in SIMPLE_BASELINES:
        if simple not in BASELINE_ESTIMATORS:
            failures.append(f"simple baseline {simple!r} is not a registered estimator")

    # 2. Incumbent selection.
    suite = {
        "historical_mean": _run("historical_mean", 0.55),
        "momentum": _run("momentum", 0.51),
        "mean_reversion": _run("mean_reversion", 0.49),
        "gradient_boosting": _run("gradient_boosting", 0.80),
    }
    incumbent = best_simple_baseline(suite)
    if incumbent.estimator != "historical_mean":
        failures.append(
            f"incumbent is {incumbent.estimator!r}; it must be the best SIMPLE "
            f"baseline, never the best model"
        )

    # 3. The rule must be able to PASS.
    winner = compare_runs(
        _run("gradient_boosting", 0.65, [0.66, 0.64, 0.67, 0.63, 0.65]),
        _run("historical_mean", 0.55, [0.54, 0.56, 0.55, 0.53, 0.57]),
    )
    if not winner.promoted:
        failures.append(
            f"a genuine consistent winner was NOT promoted ({winner.reasons}) — "
            f"a rule that can never pass proves nothing"
        )

    # 4. ...and must refuse every weak win.
    losing = compare_runs(_run("ridge", 0.45), _run("historical_mean", 0.60))
    if losing.promoted:
        failures.append("a losing candidate was promoted")

    thin = compare_runs(
        _run("ridge", 0.55 + BASELINE_PROMOTION_MARGIN / 2),
        _run("historical_mean", 0.55),
    )
    if thin.promoted:
        failures.append("a candidate below the promotion margin was promoted")

    lucky = compare_runs(
        _run("ridge", 0.60, [0.95, 0.50, 0.50, 0.50, 0.55]),
        _run("historical_mean", 0.55, [0.55] * 5),
    )
    if lucky.promoted:
        failures.append("a one-lucky-fold candidate was promoted")

    # 5. Like-for-like enforcement.
    for label, other in (
        ("dataset", _run("historical_mean", 0.5, dataset_hash="e" * 64)),
        ("horizon", _run("historical_mean", 0.5, target_horizon="5d")),
        ("feature set", _run("historical_mean", 0.5, feature_set_hash="g" * 64)),
    ):
        if not comparison_problems(_run("ridge", 0.9), other, _METRIC):
            failures.append(f"a comparison across a different {label} was allowed")

    # 6. A loser cannot emit a promotion comparison.
    try:
        promotion_comparison(losing, "ridge-v1", "hm-v1")
        failures.append("a losing verdict produced a promotion comparison")
    except BaselineSuiteError:
        pass

    # 7. End to end into the M2 gate.
    try:
        from core.model_registry import ModelEntry, build_default_model_registry

        registry = build_default_model_registry()
        registry.register(
            ModelEntry(model_version="gate-gb-v1", family="boosting", feature_set_version="fs-1")
        )
        current = registry.incumbent("technical_analysis")
        comparison = promotion_comparison(winner, "gate-gb-v1", current.model_version)
        entry = registry.promote(
            "gate-gb-v1", "oren", "2026-09-17T00:00:00+00:00", comparison
        )
        if entry.status != "approved":
            failures.append(f"M2 gate did not approve a real comparison ({entry.status})")
    except Exception as exc:  # noqa: BLE001 - the gate reports, never crashes
        failures.append(f"M4 -> M2 promotion path failed: {type(exc).__name__}: {exc}")

    # 8. "Nothing promotable" is a result, not an error.
    report = evaluate_suite(
        {
            "historical_mean": _run("historical_mean", 0.60),
            "momentum": _run("momentum", 0.50),
            "mean_reversion": _run("mean_reversion", 0.49),
            "ridge": _run("ridge", 0.45),
        }
    )
    if report["any_promotable"]:
        failures.append("a suite with no real winner reported something promotable")
    if "no candidate beat" not in report["summary"]:
        failures.append(f"empty-promotion summary is unclear: {report['summary']!r}")

    if failures:
        print("M4 baseline-suite gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M4 baseline-suite gate OK:")
    print(
        f"  {len(BASELINE_ESTIMATORS)} baselines covering {len(families)} families; "
        f"incumbent is the best simple baseline"
    )
    print(
        f"  promotion rule passes a genuine winner and refuses loss / thin margin "
        f"(<{BASELINE_PROMOTION_MARGIN}) / one-lucky-fold"
    )
    print("  like-for-like enforced; a loser cannot emit a comparison; M2 gate accepts a real one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
