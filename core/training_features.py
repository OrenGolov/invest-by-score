"""Which features reach a training row — the X5 blocker, located and fixed.

X5 reported "four of the five named feature groups have nothing to remove",
and the natural reading was that news, macro, sentiment and fundamentals
had never been built. MEASURED 2026-10-04, that reading was wrong: the
default registry already declares all 40 features across all five groups.

    market 31 | fundamental 5 | macro 1 | news 1 | sentiment 1 | regime 1

**The actual defect is routing.** `build_training_dataset` assembles a row
from `_exposed_feature_surface` — the chart contracts a score publishes
under `feature_metadata` — and never calls `contextual_feature_surface`,
which is where news, macro, regime and sentiment live:

    exposed surface      16 features
    contextual surface    1 feature   (regime_probability_proxy, status OK)
    OVERLAP               0

Two disjoint surfaces, one of them offered to training. So the features
existed, were registered, were computed, and were thrown away one call
short of the dataset.

**Why fundamentals are routed but DEFAULT OFF.** The five fundamental
features compute without a provider key, and that is the trap. MEASURED
with no ALPHAVANTAGE_API_KEY, every underlying valuation metric is null and
the producer substitutes neutral defaults:

    ticker   revenue margin  fcf  balance  valuation
    AAPL       5.0    5.0    3.0   10.0      7.0
    MSFT       5.0    5.0    3.0   10.0      7.0
    NVDA       5.0    5.0    3.0   10.0      7.0
    KO         5.0    5.0    3.0   10.0      7.0
    LLY        5.0    5.0    3.0   10.0      7.0

One distinct vector across five tickers. `balance_sheet_quality = 10.0`
reads as a perfect balance sheet for every company in the universe.

Training on constants would not break a fit — it would dilute it quietly,
inflate the apparent feature count, and let the system report fundamental
coverage it does not have. X5's ablation would then "remove" a group that
never contributed, which is precisely the confusion X5 exists to prevent.
So `fundamental_feature_state` measures whether the values are real, and
the dataset reports zero-variance columns rather than trusting a flag.

**Contextual features are optional per row.** A provider key may be unset
or an agent may report UNAVAILABLE, so requiring every declared feature on
every row would drop ALL rows the moment one domain went dark. Chart
features stay required; this is the one deliberate exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config import (
    TRAINING_CONTEXTUAL_FEATURES_OPTIONAL,
    TRAINING_FEATURE_ROUTING_VERSION,
    TRAINING_INCLUDE_CONTEXTUAL_FEATURES,
    TRAINING_INCLUDE_FUNDAMENTAL_FEATURES,
    TRAINING_REPORT_ZERO_VARIANCE_FEATURES,
)

# The fundamental features the score engine publishes under
# `fundamental_features`, named exactly as the registry declares them.
FUNDAMENTAL_FEATURE_NAMES = (
    "balance_sheet_quality",
    "free_cash_flow_quality",
    "margin_quality",
    "revenue_growth",
    "valuation_quality",
)

# The neutral vector measured with no provider key. Used to recognise the
# defaults rather than to produce them — if a real fetch happens to land on
# these values the variance check still decides, because one matching row is
# not evidence and a whole column of them is.
FUNDAMENTAL_NEUTRAL_VECTOR = {
    "revenue_growth": 5.0,
    "margin_quality": 5.0,
    "free_cash_flow_quality": 3.0,
    "balance_sheet_quality": 10.0,
    "valuation_quality": 7.0,
}

STATE_LIVE = "live"
STATE_NEUTRAL_DEFAULTS = "neutral_defaults"
STATE_ABSENT = "absent"


@dataclass(frozen=True)
class FeatureRoutingReport:
    """What was routed into a dataset, and what was refused."""

    chart_features: tuple[str, ...] = field(default_factory=tuple)
    contextual_features: tuple[str, ...] = field(default_factory=tuple)
    fundamental_features: tuple[str, ...] = field(default_factory=tuple)
    skipped: dict[str, str] = field(default_factory=dict)
    version: str = TRAINING_FEATURE_ROUTING_VERSION

    @property
    def all_features(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.chart_features)
                | set(self.contextual_features)
                | set(self.fundamental_features)
            )
        )


def fundamental_feature_state(values: dict[str, Any] | None) -> str:
    """Are these fundamental values real, neutral defaults, or absent?

    Deliberately three states, not a boolean. "No fundamentals at all" and
    "fundamentals that are secretly constants" need different handling: the
    first is honest absence, the second is absence wearing a number.
    """
    if not isinstance(values, dict) or not values:
        return STATE_ABSENT

    present = {
        name: values.get(name)
        for name in FUNDAMENTAL_FEATURE_NAMES
        if isinstance(values.get(name), (int, float))
    }
    if not present:
        return STATE_ABSENT

    matches = sum(
        1
        for name, value in present.items()
        if name in FUNDAMENTAL_NEUTRAL_VECTOR
        and abs(float(value) - FUNDAMENTAL_NEUTRAL_VECTOR[name]) < 1e-9
    )
    if matches == len(FUNDAMENTAL_NEUTRAL_VECTOR):
        return STATE_NEUTRAL_DEFAULTS
    return STATE_LIVE


def zero_variance_features(rows: list[dict[str, Any]]) -> list[str]:
    """Feature names whose value never changes across the dataset.

    A constant column cannot carry information. It does not break a fit, so
    nothing else would report it — which is why this is measured rather
    than assumed from a config flag.
    """
    if not TRAINING_REPORT_ZERO_VARIANCE_FEATURES or len(rows) < 2:
        return []

    seen: dict[str, set[float]] = {}
    for row in rows:
        for name, value in (row or {}).items():
            if not isinstance(value, (int, float)):
                continue
            seen.setdefault(name, set()).add(round(float(value), 10))
    return sorted(name for name, values in seen.items() if len(values) <= 1)


def contextual_surface_for(score_result: Any) -> dict[str, dict]:
    """The news/macro/regime/sentiment contracts for one score result.

    Returns `{}` when the flag is off or no contextual agent is OK — an
    empty surface is the normal offline posture, not a failure.
    """
    if not TRAINING_INCLUDE_CONTEXTUAL_FEATURES:
        return {}
    try:
        from core.contract_verification import contextual_feature_surface

        surface = contextual_feature_surface(score_result)
    except Exception:  # noqa: BLE001 - a dark agent must not break a build
        return {}
    return surface if isinstance(surface, dict) else {}


def fundamental_surface_for(score_result: Any) -> tuple[dict[str, dict], str]:
    """The fundamental contracts for one score result, plus their state.

    The state is returned alongside rather than folded into the surface, so
    a caller can record WHY fundamentals were or were not routed instead of
    inferring it from an empty dict.
    """
    values = getattr(score_result, "fundamental_features", None)
    state = fundamental_feature_state(values)

    if not TRAINING_INCLUDE_FUNDAMENTAL_FEATURES or state != STATE_LIVE:
        return {}, state

    as_of = str(getattr(score_result, "as_of", "") or "")
    surface: dict[str, dict] = {}
    for name in FUNDAMENTAL_FEATURE_NAMES:
        value = (values or {}).get(name)
        if not isinstance(value, (int, float)):
            continue
        surface[name] = {
            "name": name,
            "value": float(value),
            "as_of": as_of,
            "published_time": as_of,
            "source_id": "alpha_vantage_overview",
        }
    return surface, state


def merged_feature_surface(
    chart_surface: dict[str, dict], score_result: Any
) -> tuple[dict[str, dict], FeatureRoutingReport]:
    """Everything a training row may draw on, and a report of what was routed.

    The chart surface wins a name collision: it is the surface the score
    actually consumed, and silently replacing a consumed contract with a
    contextual one of the same name would make the dataset disagree with
    the decision that produced it.
    """
    chart_surface = chart_surface if isinstance(chart_surface, dict) else {}
    merged: dict[str, dict] = dict(chart_surface)
    skipped: dict[str, str] = {}

    contextual = contextual_surface_for(score_result)
    added_contextual: list[str] = []
    for name, contract in contextual.items():
        if name in merged:
            skipped[name] = "name already provided by the chart surface"
            continue
        merged[name] = contract
        added_contextual.append(name)

    fundamental, state = fundamental_surface_for(score_result)
    added_fundamental: list[str] = []
    for name, contract in fundamental.items():
        if name in merged:
            skipped[name] = "name already provided by the chart surface"
            continue
        merged[name] = contract
        added_fundamental.append(name)

    if state != STATE_LIVE:
        skipped["fundamental_features"] = (
            f"not routed: fundamental state is {state!r}"
            + (
                " — every valuation metric is null and the producer "
                "substitutes neutral defaults, which are constants rather "
                "than features"
                if state == STATE_NEUTRAL_DEFAULTS
                else ""
            )
        )
    elif not TRAINING_INCLUDE_FUNDAMENTAL_FEATURES:
        skipped["fundamental_features"] = (
            "not routed: TRAINING_INCLUDE_FUNDAMENTAL_FEATURES is off"
        )

    report = FeatureRoutingReport(
        chart_features=tuple(sorted(chart_surface)),
        contextual_features=tuple(sorted(added_contextual)),
        fundamental_features=tuple(sorted(added_fundamental)),
        skipped=skipped,
    )
    return merged, report


def optional_feature_names(surface: dict[str, dict]) -> set[str]:
    """Features a row may omit without being dropped.

    Chart features stay required: a missing one means the chart itself could
    not be built, and a row without its chart state is not a row. Contextual
    and fundamental features are legitimately absent whenever a provider key
    is unset or an agent reports UNAVAILABLE.
    """
    if not TRAINING_CONTEXTUAL_FEATURES_OPTIONAL:
        return set()
    optional = set(FUNDAMENTAL_FEATURE_NAMES)
    for name, contract in (surface or {}).items():
        source = str((contract or {}).get("source_id", ""))
        if source and source != "chart_features":
            optional.add(name)
    return optional


def render_routing(report: FeatureRoutingReport) -> list[str]:
    """Operator-readable summary of what reached the dataset."""
    lines = [
        f"feature routing [{report.version}]",
        f"  chart       {len(report.chart_features):3}",
        f"  contextual  {len(report.contextual_features):3}  "
        f"{', '.join(report.contextual_features) or '(none)'}",
        f"  fundamental {len(report.fundamental_features):3}  "
        f"{', '.join(report.fundamental_features) or '(none)'}",
    ]
    for name, reason in sorted(report.skipped.items()):
        lines.append(f"  SKIPPED {name}: {reason}")
    return lines


# The feature names Priority 2 routes into a dataset beyond the chart
# surface. Declared here so the dataset builder can widen its `declared`
# set without re-deriving the list — and so adding a domain is one edit.
#
# Every name is already in the default registry, so the M1 admission gate
# still governs them: this widens what may be ASKED for, never what may
# bypass the registry.
CONTEXTUAL_FEATURE_NAMES = (
    "macro_regime_score",
    "news_sentiment_score",
    "regime_probability_proxy",
    "sentiment_score",
)

ROUTABLE_FEATURE_NAMES = tuple(
    sorted({*CONTEXTUAL_FEATURE_NAMES, *FUNDAMENTAL_FEATURE_NAMES})
)
