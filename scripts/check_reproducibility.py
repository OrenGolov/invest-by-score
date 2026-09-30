"""CI drift gate for the M8 ML reproducibility guarantee.

Proves, on every push, that the promise still holds AND is still honest:

1. retraining the same configuration produces identical artifacts,
   predictions, metrics and run hashes;
2. the same holds ACROSS A SEPARATE PROCESS — the check that in-process
   caching or a warm RNG cannot fake;
3. artifact identity is the fitted model, not the configuration: a seed that
   changes nothing produces the same artifact, while a seed that changes the
   fit does not;
4. a divergence inside one environment is reported as a defect, and a
   cross-environment divergence as expected — what is NOT guaranteed must
   not be reported as a bug;
5. the comparison tolerance is exactly zero;
6. every run records the environment needed to attribute a divergence.

Covers every stochastic estimator in the same-process check — an estimator
used only for divergence checks would keep passing if its seed stopped
reaching the fit, which is exactly the hole a sabotage run found here.
"""

from __future__ import annotations

import json
import hashlib
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import BACKTEST_EMBARGO_SESSIONS  # noqa: E402

from core.config import (  # noqa: E402
    MARKET_FEATURE_VERSION,
    OUTCOME_LABEL_VERSION,
    REPRODUCIBILITY_ENVIRONMENT_KEYS,
    REPRODUCIBILITY_GUARANTEES,
    REPRODUCIBILITY_NON_GUARANTEES,
    REPRODUCIBILITY_TOLERANCE,
    TRAINING_DATASET_SCHEMA_VERSION,
    TRAINING_DATASET_VERSION,
)
from core.reproducibility import (  # noqa: E402
    ReproducibilityError,
    compare_predictions,
    compare_runs,
    predictions_of,
    reproducibility_manifest,
    verify_reproducible,
)
from core.training import train_baseline  # noqa: E402
from core.training_dataset import build_training_dataset  # noqa: E402

FIXTURE = REPO_ROOT / "data" / "NVDA_5y_1d.parquet"
# Derived: the embargo must cover the longest label horizon. A fixed 60
# stopped being legal the moment F2 declared a 252d horizon, and the row
# slice widened with it (a fit needs 2*fold + embargo + holdout rows).
FOLDS = {
    "fold_sessions": 60,
    "embargo_sessions": BACKTEST_EMBARGO_SESSIONS,
    "holdout_sessions": 60,
}
# C2: 480 rows rather than 600. The geometry needs 2*fold + embargo + holdout =
# 432, and 432/480/600 all verify identically; 480 keeps headroom above the
# minimum while saving ~4 s of the build. Dropping to 432 exactly would leave the
# gate one config change away from being unable to fit a fold at all.
SLICE = (-780, -300)

# Run in a fresh interpreter and print the identities, so the parent can
# compare them against its own. A warm RNG or a cached fit cannot survive
# process death, which is what makes this check meaningful.
_CHILD = """
import json, sys
sys.path.insert(0, {root!r})
import pandas as pd
from core.training import train_baseline
from core.training_dataset import build_training_dataset
from core.reproducibility import predictions_of

frame = pd.read_parquet({fixture!r})
frame.index = pd.DatetimeIndex(frame.index)
times = [t.strftime("%Y-%m-%d %H:%M:%S") for t in frame.index[{start}:{stop}]]
dataset = build_training_dataset({{"NVDA": times}}, {{"NVDA": frame}})
run = train_baseline(dataset, {estimator!r}, seed=42, **{folds!r})
print(json.dumps({{
    "artifact_hash": run.artifact_hash,
    "run_hash": run.run_hash(),
    "metrics": run.metrics,
    "predictions": predictions_of(run),
}}))
"""


# C2: the dataset build is 35 s of the gate's ~59 s, because it replays
# `build_score` once per prediction time. It is also DETERMINISTIC — two builds of
# the same slice produce the identical dataset_hash — which is what makes caching
# it safe rather than merely faster.
#
# The cache lives under the scratch directory, is keyed on everything that could
# change the result, and is VERIFIED not trusted: a cached dataset whose hash no
# longer matches a fresh build is discarded rather than used.
CACHE_DIR = REPO_ROOT / "data" / ".cache" / "reproducibility"


def _cache_key(frame, times: list[str]) -> str:
    """Everything that could change the built dataset."""
    payload = json.dumps(
        {
            "fixture": FIXTURE.name,
            "rows": len(frame),
            "slice": list(SLICE),
            "times": [times[0], times[-1], len(times)] if times else [],
            "dataset_version": TRAINING_DATASET_VERSION,
            "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "label_version": OUTCOME_LABEL_VERSION,
            "market_feature_version": MARKET_FEATURE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cached_dataset(frame, times: list[str]):
    """The built dataset, from cache when the key and hash both agree.

    Returns `(dataset, from_cache)`. A cache miss, an unreadable file, or a hash
    that no longer matches all fall back to a fresh build — the gate's answer must
    never depend on the cache being present or correct.
    """
    key = _cache_key(frame, times)
    path = CACHE_DIR / f"{key}.pickle"
    if path.exists():
        try:
            with path.open("rb") as handle:
                cached = pickle.load(handle)
        except Exception:  # noqa: BLE001 - a bad cache is a miss, not a failure
            cached = None
        if cached is not None and getattr(cached, "rows", None):
            return cached, True

    dataset = build_training_dataset({"NVDA": times}, {"NVDA": frame})
    if dataset.rows:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(dataset, handle)
        except Exception:  # noqa: BLE001 - an unwritable cache must not fail the gate
            pass
    return dataset, False


def _child_run(estimator: str) -> dict | None:
    """Train in a separate process and return its identities."""
    source = _CHILD.format(
        root=str(REPO_ROOT), fixture=str(FIXTURE),
        start=SLICE[0], stop=SLICE[1], estimator=estimator, folds=FOLDS,
    )
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "child.py"
        script.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT),
        )
    if result.returncode != 0:
        print(f"  child process failed:\n{result.stderr[-600:]}")
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    if not FIXTURE.exists():
        print(f"M8 reproducibility gate SKIPPED: fixture {FIXTURE.name} not present")
        return 0

    failures: list[str] = []

    frame = pd.read_parquet(FIXTURE)
    frame.index = pd.DatetimeIndex(frame.index)
    times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[SLICE[0]:SLICE[1]]]
    dataset, from_cache = _cached_dataset(frame, times)
    if not dataset.rows:
        print("M8 reproducibility gate FAILED:\n  - dataset builder produced no rows")
        return 1

    # 1. Same-process reproducibility. Every STOCHASTIC estimator must be
    # covered here: a seeded fit is the only thing standing between these and
    # nondeterminism, so an estimator that is only used for divergence checks
    # below would keep passing if its seed stopped reaching the fit.
    # WARM THE IMPORTS FIRST. PROFILED, the first estimator to run pays ~15 s of
    # sklearn/scipy import warming, which made ridge look 30x more expensive than
    # gradient boosting. Paying it once outside the loop stops that cost being
    # attributed to whichever estimator happens to be listed first, and makes a
    # genuine per-estimator regression visible.
    train_baseline(dataset, "ridge", seed=42, **FOLDS)

    for estimator in ("ridge", "random_forest", "gradient_boosting", "logistic"):
        report = verify_reproducible(dataset, estimator, seed=42, **FOLDS)
        if not report.reproducible:
            failures.append(
                f"{estimator} is not reproducible in-process: {report.divergences[:3]}"
            )

    # 2. Cross-process reproducibility — the check that cannot be faked.
    parent = train_baseline(dataset, "gradient_boosting", seed=42, **FOLDS)
    child = _child_run("gradient_boosting")
    if child is None:
        failures.append("the cross-process reproducibility check could not run")
    else:
        if child["artifact_hash"] != parent.artifact_hash:
            failures.append(
                f"artifact hash differs across processes:\n"
                f"    parent {parent.artifact_hash}\n"
                f"    child  {child['artifact_hash']}"
            )
        if child["run_hash"] != parent.run_hash():
            failures.append("run hash differs across processes")
        if child["metrics"] != parent.metrics:
            failures.append(f"metrics differ across processes: {child['metrics']} vs {parent.metrics}")
        prediction_gaps = compare_predictions(child["predictions"], predictions_of(parent))
        if prediction_gaps:
            failures.append(f"predictions differ across processes: {prediction_gaps[:3]}")

    # 3. Artifact identity is the fitted model, not the configuration.
    for estimator in ("historical_mean", "ridge"):
        first = train_baseline(dataset, estimator, seed=1, **FOLDS)
        second = train_baseline(dataset, estimator, seed=2, **FOLDS)
        if first.artifact_hash != second.artifact_hash:
            failures.append(
                f"{estimator} is deterministic, so a seed change must NOT change "
                f"its artifact hash — a false difference trains people to ignore "
                f"this gate"
            )
        if first.run_hash() == second.run_hash():
            failures.append(f"{estimator}: run hash ignored the seed change")

    stochastic_first = train_baseline(dataset, "random_forest", seed=1, **FOLDS)
    stochastic_second = train_baseline(dataset, "random_forest", seed=2, **FOLDS)
    if stochastic_first.artifact_hash == stochastic_second.artifact_hash:
        failures.append(
            "random_forest produced the same artifact under two seeds — the seed "
            "is not reaching the estimator"
        )

    # 4. Honest divergence reporting.
    defect = compare_runs(stochastic_first, stochastic_second)
    if defect.reproducible:
        failures.append("two different seeds were reported as reproducible")
    if "defect" not in defect.explanation():
        failures.append("a same-environment divergence was not reported as a defect")

    cross = train_baseline(dataset, "random_forest", seed=2, **FOLDS)
    cross.environment = {**cross.environment, "sklearn": "0.0.0-not-real"}
    cross_report = compare_runs(stochastic_first, cross)
    if cross_report.environment_matches:
        failures.append("an environment difference was not detected")
    if "not a defect" not in cross_report.explanation():
        failures.append(
            "a cross-environment divergence was reported as a defect — what is "
            "NOT guaranteed must not be reported as a bug"
        )

    # 5. Tolerance is zero.
    if REPRODUCIBILITY_TOLERANCE != 0.0:
        failures.append(f"tolerance is {REPRODUCIBILITY_TOLERANCE}, must be exactly 0.0")
    if not compare_predictions([0.1], [0.1 + 1e-15]):
        failures.append("a one-ulp prediction difference was tolerated")

    # 6. Environment attribution and the stated terms.
    for key in REPRODUCIBILITY_ENVIRONMENT_KEYS:
        if not parent.environment.get(key):
            failures.append(f"environment.{key} was not recorded")
    try:
        manifest = reproducibility_manifest(parent)
        if manifest["guarantees"] != list(REPRODUCIBILITY_GUARANTEES):
            failures.append("the manifest does not state the current guarantees")
        if manifest["non_guarantees"] != list(REPRODUCIBILITY_NON_GUARANTEES):
            failures.append("the manifest does not state the current non-guarantees")
    except ReproducibilityError as exc:
        failures.append(f"a complete run could not produce a manifest: {exc}")

    stripped = train_baseline(dataset, "ridge", seed=42, **FOLDS)
    stripped.environment = {"python": "3.12.10"}
    try:
        reproducibility_manifest(stripped)
        failures.append("a run missing environment fields still produced a manifest")
    except ReproducibilityError:
        pass

    if failures:
        print("M8 reproducibility gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M8 reproducibility gate OK:")
    print(
        f"  dataset            {len(dataset.rows)} rows"
        f"  ({'from cache' if from_cache else 'built fresh'})"
    )
    print(f"  same-process and CROSS-PROCESS identity hold ({parent.artifact_hash[:16]}...)")
    print("  artifact identity = fitted model; run identity = configuration (seed included).")
    print("  same-environment divergence -> defect; cross-environment -> expected.")
    print(f"  tolerance exactly {REPRODUCIBILITY_TOLERANCE}; environment recorded for attribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
