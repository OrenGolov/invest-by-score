"""X5 feature ablation — which sources actually add incremental value.

"Test removing news, technicals, macro, sentiment, fundamentals to prove which
sources add incremental value." MEASURED, **four of the five named groups have
nothing to remove** — and unlike X3 and X4, the fifth is genuinely testable, so
X5 is *partially* evaluable rather than blocked outright.

**The deciding measurement.** Every feature the training pipeline can produce is
a price or volume derivative:

============  ========  ===========
group         features  ablatable?
============  ========  ===========
technicals          16  yes
news                 0  nothing to remove
macro                0  nothing to remove
sentiment            0  nothing to remove
fundamentals         0  nothing to remove
============  ========  ===========

**Why, structurally.** The dataset builder defaults to
``CURRENT_SCORE_FEATURES | LONG_TERM_SCORE_FEATURES``, both hardcoded tuples in
``score_engine.py`` holding only price and volume derivatives. The feature
registry declares eight domains, but ``data/feature_registry.jsonl`` does not
exist, so zero features are registered and no non-technical feature has a
producer a model could consume.

An ablation over a group that was never present measures nothing. Reporting
"removing news did not change performance" would present an untested source as a
tested one — the same confusion A6 drew between NOT_EVALUATED and NOT_MET. So an
absent group is **ABSENT**, never NO_VALUE.

**The one ablation the shipped data supports, and it is a real result.** Removing
all technical features leaves a model with no features at all — which is exactly
the ``historical_mean`` baseline:

=========================================  ============  ==========
                                           rmse          directional
=========================================  ============  ==========
all 16 technical features (best learned)        0.13192      0.4583
no features at all (historical_mean)            0.12164      0.5500
=========================================  ============  ==========

Incremental value of all 16 technical features: **+0.01028 rmse — worse.** The
one group that *can* be ablated has negative incremental value; the features
actively hurt. That is X1's finding re-expressed as an ablation, and it is the
answer X5 was asked for, for the only group it can ask about.

**Harmful is not the same as worthless.** A group whose removal *improves* the
metric is reported as HARMFUL rather than NO_VALUE, because calling +0.01028
"no value" understates it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    ABLATION_ABSENT,
    ABLATION_ABSENT_MEANS_NO_VALUE,
    ABLATION_ADDS_VALUE,
    ABLATION_BLOCKS_TRADES,
    ABLATION_GROUPS,
    ABLATION_GROUP_FUNDAMENTAL,
    ABLATION_GROUP_MACRO,
    ABLATION_GROUP_NEWS,
    ABLATION_GROUP_SENTIMENT,
    ABLATION_GROUP_TECHNICAL,
    ABLATION_HARMFUL,
    ABLATION_HIGHER_IS_BETTER,
    ABLATION_METRIC,
    ABLATION_MIN_GROUP_FEATURES,
    ABLATION_MIN_IMPROVEMENT,
    ABLATION_NOT_EVALUATED,
    ABLATION_NO_VALUE,
    ABLATION_REPORTS_HARMFUL,
    ABLATION_VERDICTS,
    FEATURE_ABLATION_VERSION,
)


class FeatureAblationError(ValueError):
    """Raised when an ablation cannot be judged without guessing."""


# Which group a feature belongs to, by the producer that makes it. Declared
# EXPLICITLY rather than inferred from the name: a feature called
# "news_sentiment_5d" would otherwise be sorted by whichever substring matched
# first, and the group a feature belongs to is a fact about its producer.
_TECHNICAL_PREFIXES: tuple[str, ...] = (
    "change_",
    "ma_",
    "price_vs_ma_",
    "rsi",
    "trend_vs_",
    "volatility",
    "volume_",
)
_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    ABLATION_GROUP_TECHNICAL: _TECHNICAL_PREFIXES,
    ABLATION_GROUP_NEWS: ("news_",),
    ABLATION_GROUP_MACRO: ("macro_",),
    ABLATION_GROUP_SENTIMENT: ("sentiment_",),
    ABLATION_GROUP_FUNDAMENTAL: ("fundamental_", "valuation_"),
}


def group_of(feature: str) -> str | None:
    """The ablation group a feature belongs to, or None when unrecognised.

    None is a real answer: a feature nobody can place must not be silently
    swept into a group, because that would make an ablation of that group
    remove something it does not own.
    """
    if not isinstance(feature, str) or not feature:
        raise FeatureAblationError("a feature name must be a non-empty string")
    # Most specific first: "price_vs_ma_50" must not be claimed by "ma_".
    for group in (
        ABLATION_GROUP_NEWS,
        ABLATION_GROUP_MACRO,
        ABLATION_GROUP_SENTIMENT,
        ABLATION_GROUP_FUNDAMENTAL,
        ABLATION_GROUP_TECHNICAL,
    ):
        for prefix in _GROUP_PREFIXES[group]:
            if feature.startswith(prefix):
                return group
    return None


def group_features(features: Sequence[str]) -> dict[str, list[str]]:
    """Split a feature set into the roadmap's groups.

    Every declared group appears in the result, including the empty ones —
    an absent group has to be VISIBLE to be reported as ABSENT.
    """
    grouped: dict[str, list[str]] = {group: [] for group in ABLATION_GROUPS}
    unplaced: list[str] = []
    for feature in features:
        group = group_of(feature)
        if group is None:
            unplaced.append(feature)
            continue
        grouped[group].append(feature)
    if unplaced:
        raise FeatureAblationError(
            f"cannot place {sorted(unplaced)} in any ablation group; sweeping "
            f"an unrecognised feature into a group would make that group's "
            f"ablation remove something it does not own"
        )
    return {group: sorted(names) for group, names in grouped.items()}


def incremental_value(
    with_group: float | None, without_group: float | None
) -> float | None:
    """How much the metric WORSENS when a group is removed.

    Positive means the group earned its place (removing it hurt). Negative means
    removing it helped. None when either side is missing — an unmeasured side is
    not a zero difference.
    """
    if with_group is None or without_group is None:
        return None
    if ABLATION_HIGHER_IS_BETTER:
        return float(with_group) - float(without_group)
    return float(without_group) - float(with_group)


def ablate_group(
    group: str,
    features: Sequence[str],
    *,
    with_group: float | None = None,
    without_group: float | None = None,
) -> dict:
    """Judge one group's incremental value."""
    if group not in ABLATION_GROUPS:
        raise FeatureAblationError(
            f"unknown ablation group {group!r}; the roadmap names "
            f"{list(ABLATION_GROUPS)}"
        )

    detail: dict[str, Any] = {
        "group": group,
        "features": sorted(features),
        "feature_count": len(features),
        "metric": ABLATION_METRIC,
        "with_group": with_group,
        "without_group": without_group,
        "margin": ABLATION_MIN_IMPROVEMENT,
    }

    # AN ABSENT GROUP IS NOT A TESTED ONE. The rule the whole task turns on.
    if len(features) < ABLATION_MIN_GROUP_FEATURES:
        return _report(
            ABLATION_ABSENT,
            reason=(
                f"{group} contributes no features, so there is nothing to "
                f"remove; this group was NEVER TESTED, which is not the same "
                f"as tested and found worthless"
            ),
            value=None,
            **detail,
        )

    value = incremental_value(with_group, without_group)
    detail["value"] = value

    if value is None:
        return _report(
            ABLATION_NOT_EVALUATED,
            reason=(
                f"{group} contributes {len(features)} feature(s) but the "
                f"with/without comparison is incomplete "
                f"(with={with_group!r}, without={without_group!r}); an "
                f"unmeasured side is not a zero difference"
            ),
            **detail,
        )

    if value > ABLATION_MIN_IMPROVEMENT:
        return _report(
            ABLATION_ADDS_VALUE,
            reason=(
                f"removing {group} worsens {ABLATION_METRIC} by {value:.5f}, "
                f"beyond the {ABLATION_MIN_IMPROVEMENT} margin, so the group "
                f"earns its place"
            ),
            **detail,
        )

    if ABLATION_REPORTS_HARMFUL and value < -ABLATION_MIN_IMPROVEMENT:
        return _report(
            ABLATION_HARMFUL,
            reason=(
                f"removing {group} IMPROVES {ABLATION_METRIC} by "
                f"{-value:.5f}; the group actively hurts, which is a stronger "
                f"finding than adding nothing"
            ),
            **detail,
        )

    return _report(
        ABLATION_NO_VALUE,
        reason=(
            f"removing {group} moves {ABLATION_METRIC} by {value:+.5f}, within "
            f"the {ABLATION_MIN_IMPROVEMENT} margin, so it was tested and adds "
            f"nothing measurable"
        ),
        **detail,
    )


def evaluate_ablation(
    features: Sequence[str] | None,
    scores: Mapping[str, Mapping[str, float | None]] | None = None,
    *,
    estimator: str | None = None,
) -> dict:
    """Ablate every named group and report which sources earn their place.

    ``scores`` maps a group to ``{"with": metric, "without": metric}``. A group
    missing from it is simply not measured; a group with no features is ABSENT
    regardless.
    """
    if features is None:
        return {
            "version": FEATURE_ABLATION_VERSION,
            "gate": "feature_ablation",
            "estimator": estimator,
            "verdict": ABLATION_NOT_EVALUATED,
            "reason": "no feature set was supplied, so nothing could be ablated",
            "groups": {},
            "absent_groups": list(ABLATION_GROUPS),
            "tested_groups": [],
            "absent_means_no_value": ABLATION_ABSENT_MEANS_NO_VALUE,
            "blocks_trades": ABLATION_BLOCKS_TRADES,
            "note": _NOTE,
        }

    grouped = group_features(features)
    scores = scores or {}

    reports: dict[str, dict] = {}
    for group in ABLATION_GROUPS:
        pair = scores.get(group) or {}
        reports[group] = ablate_group(
            group,
            grouped[group],
            with_group=pair.get("with"),
            without_group=pair.get("without"),
        )

    absent = [g for g, r in reports.items() if r["verdict"] == ABLATION_ABSENT]
    tested = [
        g
        for g, r in reports.items()
        if r["verdict"] in (ABLATION_ADDS_VALUE, ABLATION_NO_VALUE, ABLATION_HARMFUL)
    ]
    valuable = [g for g, r in reports.items() if r["verdict"] == ABLATION_ADDS_VALUE]
    harmful = [g for g, r in reports.items() if r["verdict"] == ABLATION_HARMFUL]

    if not tested:
        verdict = ABLATION_NOT_EVALUATED
        reason = (
            f"no group could be ablated: {len(absent)} of "
            f"{len(ABLATION_GROUPS)} contribute no features at all "
            f"({', '.join(absent)}), so no source's value has been proved"
        )
    else:
        verdict = ABLATION_NOT_EVALUATED if absent else ABLATION_ADDS_VALUE
        reason = (
            f"{len(tested)} of {len(ABLATION_GROUPS)} group(s) were ablated "
            f"({', '.join(tested)}); "
            f"{len(valuable)} add value, {len(harmful)} are harmful. "
            + (
                f"{len(absent)} group(s) have no features and were NEVER "
                f"TESTED ({', '.join(absent)}), so the release cannot claim "
                f"to know which sources matter"
                if absent
                else "every named group was tested"
            )
        )
        if absent:
            verdict = ABLATION_NOT_EVALUATED
        elif harmful:
            verdict = ABLATION_HARMFUL
        elif valuable:
            verdict = ABLATION_ADDS_VALUE
        else:
            verdict = ABLATION_NO_VALUE

    return {
        "version": FEATURE_ABLATION_VERSION,
        "gate": "feature_ablation",
        "estimator": estimator,
        "verdict": verdict,
        "reason": reason,
        "groups": reports,
        "absent_groups": absent,
        "tested_groups": tested,
        "valuable_groups": valuable,
        "harmful_groups": harmful,
        "absent_means_no_value": ABLATION_ABSENT_MEANS_NO_VALUE,
        "blocks_trades": ABLATION_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _report(verdict: str, **detail) -> dict:
    """One per-group X5 answer."""
    if verdict not in ABLATION_VERDICTS:
        raise FeatureAblationError(f"unknown ablation verdict {verdict!r}")
    payload = {
        "version": FEATURE_ABLATION_VERSION,
        "verdict": verdict,
        "absent_means_no_value": ABLATION_ABSENT_MEANS_NO_VALUE,
    }
    payload.setdefault("value", None)
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, four of the five named groups have ZERO features: every feature "
    "the pipeline produces is a price or volume derivative, because the dataset "
    "defaults to two hardcoded tuples in score_engine.py and "
    "data/feature_registry.jsonl does not exist. An absent group is ABSENT, "
    "never NO_VALUE. The one group that CAN be ablated is harmful: removing all "
    "16 technical features leaves historical_mean, which is BETTER by 0.01028 "
    "rmse."
)


def ablation_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an ablation report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != FEATURE_ABLATION_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{FEATURE_ABLATION_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in ABLATION_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X5 reports; the registry promotes")

    if report.get("absent_means_no_value"):
        problems.append(
            "an absent group must never be reported as adding no value; that "
            "presents an untested source as a tested one"
        )

    groups = report.get("groups")
    if groups is not None:
        if not isinstance(groups, Mapping):
            problems.append("groups must be a mapping")
        else:
            missing = [g for g in ABLATION_GROUPS if g not in groups]
            if missing:
                problems.append(
                    f"the roadmap names {missing} and they are not reported; a "
                    f"group nobody ablates is a source whose value is never proved"
                )
            for name, entry in groups.items():
                if not isinstance(entry, Mapping):
                    continue
                if entry.get("verdict") == ABLATION_ABSENT and entry.get("value") is not None:
                    problems.append(
                        f"{name} is ABSENT but carries a value; a group that "
                        f"was never tested cannot have a measured difference"
                    )
                if (
                    entry.get("verdict") == ABLATION_ABSENT
                    and entry.get("feature_count")
                ):
                    problems.append(
                        f"{name} is ABSENT but contributes "
                        f"{entry['feature_count']} feature(s)"
                    )

        # A release cannot claim to know which sources matter while some were
        # never tested.
        absent = report.get("absent_groups")
        if absent and verdict in (ABLATION_ADDS_VALUE, ABLATION_NO_VALUE):
            problems.append(
                f"the suite verdict is {verdict} while {absent} were never "
                f"tested; an untested group cannot be absorbed into a "
                f"conclusion about which sources add value"
            )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_ablation(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one ablation report."""
    lines = [
        f"Feature ablation: {_shown(report.get('estimator'))}"
        f" -> {report.get('verdict')}",
        f"  metric: {ABLATION_METRIC} (lower is better)"
        f"  margin: {ABLATION_MIN_IMPROVEMENT}",
    ]
    for group, entry in (report.get("groups") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            f"    {group:<13} {entry.get('feature_count', 0):>2} feature(s)"
            f"  {str(entry.get('verdict')):<14}"
            f" value {_num(entry.get('value'))}"
        )
    absent = report.get("absent_groups") or []
    if absent:
        lines.append(
            f"  NEVER TESTED: {', '.join(absent)}"
            f"  - absent is not 'adds nothing'"
        )
    lines.append(f"  reason: {report.get('reason')}")
    return lines


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):+.5f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
