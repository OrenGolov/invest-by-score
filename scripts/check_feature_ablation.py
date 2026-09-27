"""X5 governance gate — which sources actually add incremental value.

Exits 1 when any of these fails.

THE DECIDING MEASUREMENT, RECOMPUTED HERE from the tracked training runs: every
feature the pipeline produces is a price or volume derivative, so four of the
five groups the roadmap names have ZERO features and cannot be ablated at all.
The one that can be is HARMFUL — removing all 16 technical features leaves
`historical_mean`, which is better by 0.01028 rmse.

`data/training_runs.jsonl` is TRACKED, so this recomputes identically on a fresh
clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    ABLATION_ABSENT,
    ABLATION_ADDS_VALUE,
    ABLATION_GROUPS,
    ABLATION_GROUP_FUNDAMENTAL,
    ABLATION_GROUP_MACRO,
    ABLATION_GROUP_NEWS,
    ABLATION_GROUP_SENTIMENT,
    ABLATION_GROUP_TECHNICAL,
    ABLATION_HARMFUL,
    ABLATION_MIN_IMPROVEMENT,
    ABLATION_NOT_EVALUATED,
    ABLATION_NO_VALUE,
    REGIME_SPECIALIZATION_MARGIN,
)
from core.feature_ablation import (  # noqa: E402
    FeatureAblationError,
    ablate_group,
    ablation_problems,
    evaluate_ablation,
    group_features,
    group_of,
    incremental_value,
    render_ablation,
)
from core.training import load_training_runs  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


EMPTY_GROUPS = (
    ABLATION_GROUP_NEWS,
    ABLATION_GROUP_MACRO,
    ABLATION_GROUP_SENTIMENT,
    ABLATION_GROUP_FUNDAMENTAL,
)


# --- 1. THE MEASUREMENT: four of five groups have nothing to remove --------------

runs = {r["estimator"]: r for r in load_training_runs()}
check(bool(runs), "no training runs are available; X5 cannot measure anything")

if runs:
    features = runs[next(iter(runs))]["feature_names"]
    grouped = group_features(features)

    check(
        len(grouped[ABLATION_GROUP_TECHNICAL]) == len(features),
        f"only {len(grouped[ABLATION_GROUP_TECHNICAL])} of {len(features)} "
        f"shipped features are technical; the other groups are supposed to be "
        f"empty and this gate's reasoning must be revisited",
    )

    now_present = [g for g in EMPTY_GROUPS if grouped[g]]
    check(
        not now_present,
        f"{now_present} now contribute features. That is the fix X5 asks for, "
        f"and this gate's 'nothing to ablate' claim is stale - the real "
        f"ablation must now run on them",
    )


# --- 2. AN ABSENT GROUP IS NOT A TESTED ONE --------------------------------------
# The rule the whole task turns on.

for group in EMPTY_GROUPS:
    report = ablate_group(group, [])
    check(
        report["verdict"] == ABLATION_ABSENT,
        f"{group} with no features was {report['verdict']}, expected ABSENT",
    )
    check(
        report["verdict"] != ABLATION_NO_VALUE,
        f"{group} was reported as adding no value although it was NEVER "
        f"TESTED; that presents an untested source as a tested one",
    )
    check(
        report.get("value") is None,
        f"{group} is ABSENT but carries a value of {report.get('value')!r}; a "
        f"group that was never tested cannot have a measured difference",
    )
    check(
        "NEVER TESTED" in report["reason"],
        f"{group}'s reason does not say it was never tested",
    )


# --- 3. THE ONE TESTABLE GROUP IS HARMFUL, recomputed ----------------------------

if runs and "historical_mean" in runs:
    without = runs["historical_mean"]["metrics"]["rmse"]
    with_group = min(
        r["metrics"]["rmse"] for name, r in runs.items() if name != "historical_mean"
    )
    value = incremental_value(with_group, without)

    check(
        value is not None and value < -ABLATION_MIN_IMPROVEMENT,
        f"removing all technical features no longer improves rmse "
        f"(value {value!r}); the shipped features may now earn their place, "
        f"and X5's HARMFUL finding is stale",
    )

    report = ablate_group(
        ABLATION_GROUP_TECHNICAL,
        runs["historical_mean"]["feature_names"],
        with_group=with_group,
        without_group=without,
    )
    check(
        report["verdict"] == ABLATION_HARMFUL,
        f"the technical group was {report['verdict']}, expected HARMFUL; "
        f"MEASURED, removing all 16 features improves rmse by "
        f"{-(value or 0):.5f}",
    )
    check(
        report["verdict"] != ABLATION_NO_VALUE,
        "a group whose removal IMPROVES the metric was reported as adding no "
        "value; that understates an actively harmful feature set",
    )


# --- 4. The suite cannot claim to know which sources matter ----------------------

if runs:
    suite = evaluate_ablation(
        features,
        {ABLATION_GROUP_TECHNICAL: {"with": with_group, "without": without}}
        if "historical_mean" in runs
        else None,
    )
    check(
        suite["verdict"] == ABLATION_NOT_EVALUATED,
        f"the shipped suite reported {suite['verdict']}; with four groups "
        f"never tested, the release cannot claim to know which sources add "
        f"value",
    )
    check(
        len(suite["absent_groups"]) == len(EMPTY_GROUPS),
        f"{len(suite['absent_groups'])} groups were reported absent, expected "
        f"{len(EMPTY_GROUPS)}",
    )
    check(
        sorted(suite["groups"]) == sorted(ABLATION_GROUPS),
        "not every roadmap group appears in the report; a group nobody "
        "ablates is a source whose value is never proved",
    )


# --- 5. The gate CAN reach a real verdict, or refusing proves nothing ------------

complete_features = list(features if runs else []) + [
    "news_polarity_5d",
    "macro_cpi_yoy",
    "sentiment_score",
    "fundamental_pe",
]
complete_scores = {
    ABLATION_GROUP_TECHNICAL: {"with": 0.10, "without": 0.20},
    ABLATION_GROUP_NEWS: {"with": 0.10, "without": 0.12},
    ABLATION_GROUP_MACRO: {"with": 0.10, "without": 0.11},
    ABLATION_GROUP_SENTIMENT: {"with": 0.10, "without": 0.1001},
    ABLATION_GROUP_FUNDAMENTAL: {"with": 0.10, "without": 0.13},
}
complete = evaluate_ablation(complete_features, complete_scores)
check(
    complete["verdict"] == ABLATION_ADDS_VALUE,
    f"a complete feature set with real differences was {complete['verdict']}; "
    f"a gate that can only refuse is a constant, not a test",
)
check(
    complete["absent_groups"] == [],
    f"a complete feature set still reported {complete['absent_groups']} absent",
)
check(
    ABLATION_GROUP_SENTIMENT in complete["groups"]
    and complete["groups"][ABLATION_GROUP_SENTIMENT]["verdict"] == ABLATION_NO_VALUE,
    "a group moved by 0.0001 - inside the margin - was not reported as "
    "NO_VALUE; the margin is not being applied",
)


# --- 6. Grouping is by PRODUCER, not by loose substring match --------------------

# Every shipped feature must land in exactly ONE group, and the prefix sets
# must not overlap. MEASURED, `macro_` does not collide with `ma_` because of
# the underscore - but a future prefix could, and a feature claimed by two
# groups would be removed by both ablations.
_PREFIX_OWNERS: dict[str, list[str]] = {}
for _probe in (
    "price_vs_ma_50",
    "ma_50",
    "macro_cpi_yoy",
    "news_polarity_5d",
    "sentiment_score",
    "fundamental_pe",
    "valuation_pb",
    "volatility",
    "volume_ratio_20d",
):
    _owner = group_of(_probe)
    check(
        _owner is not None,
        f"{_probe!r} could not be placed in any group",
    )
    _PREFIX_OWNERS.setdefault(_owner or "unplaced", []).append(_probe)

check(
    group_of("price_vs_ma_50") == ABLATION_GROUP_TECHNICAL
    and group_of("ma_50") == ABLATION_GROUP_TECHNICAL,
    "a moving-average feature was not placed in technicals",
)
check(
    group_of("macro_cpi_yoy") == ABLATION_GROUP_MACRO,
    "'macro_cpi_yoy' was claimed by a non-macro group; the 'ma_' technical "
    "prefix must not swallow macro features",
)

# The decisive property: no feature may be claimed by two groups, or an
# ablation of either would remove it.
_seen: dict[str, str] = {}
for _group, _probes in _PREFIX_OWNERS.items():
    for _probe in _probes:
        if _probe in _seen and _seen[_probe] != _group:
            failures.append(
                f"{_probe!r} is claimed by both {_seen[_probe]} and {_group}; "
                f"a feature in two groups is removed by both ablations"
            )
        _seen[_probe] = _group
check(
    group_of("mystery_signal") is None,
    "an unrecognised feature was swept into a group; that would make that "
    "group's ablation remove something it does not own",
)
try:
    group_features(["mystery_signal"])
except FeatureAblationError:
    pass
else:
    failures.append("an unplaceable feature was silently grouped")

try:
    ablate_group("vibes", ["x"])
except FeatureAblationError:
    pass
else:
    failures.append("an unknown ablation group was accepted")


# --- 7. An unmeasured side is not a zero difference ------------------------------

check(
    incremental_value(0.10, None) is None and incremental_value(None, 0.10) is None,
    "a missing metric produced a number; an unmeasured side is not a zero "
    "difference",
)
present_unmeasured = ablate_group(ABLATION_GROUP_TECHNICAL, ["rsi"])
check(
    present_unmeasured["verdict"] == ABLATION_NOT_EVALUATED,
    f"a PRESENT group with no scores was {present_unmeasured['verdict']}, "
    f"expected NOT_EVALUATED",
)
check(
    present_unmeasured["verdict"] != ABLATION_ABSENT,
    "'present but unmeasured' was collapsed into 'absent'; they imply "
    "different fixes - run the experiment versus build the producer",
)


# --- 8. The margin is REUSED, not invented ---------------------------------------

check(
    ABLATION_MIN_IMPROVEMENT == REGIME_SPECIALIZATION_MARGIN,
    f"the improvement margin {ABLATION_MIN_IMPROVEMENT} is not L7's measured "
    f"{REGIME_SPECIALIZATION_MARGIN}; 'this change is real' would mean "
    f"different things in different gates",
)


# --- 9. A conclusion cannot absorb untested groups -------------------------------

if runs:
    forged = dict(suite)
    forged["verdict"] = ABLATION_ADDS_VALUE
    check(
        any("never tested" in p for p in ablation_problems(forged)),
        "the contract check accepted an ADDS_VALUE verdict while groups were "
        "never tested; that is how an untested source gets absorbed into a "
        "conclusion",
    )

    forged_absent = dict(suite)
    groups = dict(forged_absent["groups"])
    entry = dict(groups[ABLATION_GROUP_NEWS])
    entry["value"] = 0.0
    groups[ABLATION_GROUP_NEWS] = entry
    forged_absent["groups"] = groups
    check(
        any("never tested cannot have" in p for p in ablation_problems(forged_absent)),
        "the contract check accepted an ABSENT group carrying a value",
    )


# --- 10. Every report renders and passes its own contract ------------------------

reports = [("complete", complete)]
if runs:
    reports.append(("shipped", suite))
for label, report in reports:
    problems = ablation_problems(report)
    if problems:
        failures.append(f"the {label} report failed its own contract: {problems}")
        break
    lines = render_ablation(report)
    if not lines or not all(isinstance(line, str) for line in lines):
        failures.append(f"render_ablation returned no lines for {label}")
        break


if failures:
    print("X5 FEATURE ABLATION GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("X5 feature ablation gate: OK")
if runs:
    print(f"  shipped features            {len(features)}, all technical")
    print(f"  groups with nothing to remove {len(EMPTY_GROUPS)} of {len(ABLATION_GROUPS)}"
          f"  ({', '.join(EMPTY_GROUPS)})")
    if "historical_mean" in runs:
        print(f"  technicals                  HARMFUL"
              f"  (removing all 16 improves rmse by {-(value or 0):.5f})")
    print(f"  suite verdict               {suite['verdict']}"
          f"  - cannot claim to know which sources matter")
print("  the gate CAN conclude       verified on a complete feature set")
sys.exit(0)
