"""X4 event robustness — no single viral event or source may explain the edge.

MEASURED, this gate cannot run today, and its blocker is **different from X3's**.
X3's is a dropped join key: the regime data exists, but folds discard the ticker
and timestamp needed to attach it. X4's is that the event data does not exist at
all.

====================================  ==========================================
event memories on disk                0
raw store directories                 alpha_vantage_overview, yahoo_finance_chart
news store                            none
``COLLECT_NEWS_CURSOR_PATH``          does not exist
====================================  ==========================================

The raw store holds price and fundamentals only. No news has ever been ingested,
so there is no event to attribute an edge to. X4 reports NOT_EVALUATED and names
*two* fixes — ingest news, **and** carry the event id and source onto each
validation observation — because doing only the first still leaves nothing
joinable.

**The naive test is wrong.** Removing a viral event that carries the edge drops
accuracy 0.7850 → 0.4750 (a fall of 0.3100) — but removing an *ordinary* event
from a small book still drops it 0.1100. A fixed bar cannot tell a carried edge
from a book with few events, because the null drop scales with the removed
item's **share** of the book:

======  ===========  =============  =========
events  each share   p95 null drop  p95/share
======  ===========  =============  =========
3             33.3%         0.0667      0.200
5             20.0%         0.0367      0.183
10            10.0%         0.0204      0.204
20             5.0%         0.0114      0.228
======  ===========  =============  =========

The ratio is stable across a 6.7× range of event counts, so the test is
**scale-free**: compare the drop to the item's share, never to a fixed number.

**The bar sits in a real gap — but only once three-item books are excluded.**
An early 480-book sweep put the uniform maximum at 0.308; widening it to 1,800
books found 0.350, which *touches* a 0.35 bar. The whole tail came from
three-item books, where removing one item deletes a third of the evidence:

=========  =============  =====  =====
min items  uniform books  p99    max
=========  =============  =====  =====
3 or more          1,800  0.267  0.350
4 or more          1,500  0.263  0.322
=========  =============  =====  =====

So the floor is **four** items. With it the uniform maximum is 0.322 against a
carrier minimum of 0.356, and the 0.35 bar lies between them. The p99 barely
moves (0.267 → 0.263), which is the tell that this was a degenerate-case tail
rather than a shift in the statistic.

**Both axes are tested**, because a source can carry an edge that no single event
does. On a book where one source supplies every third event, the worst *event*
ratio is 0.340 (under the bar) while the worst *source* ratio is 0.425. Testing
events alone would miss that carrier entirely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    EVENT_CARRIED,
    EVENT_ROBUST,
    EVENT_ROBUSTNESS_AXES,
    EVENT_ROBUSTNESS_AXIS_EVENT,
    EVENT_ROBUSTNESS_AXIS_SOURCE,
    EVENT_ROBUSTNESS_BLOCKS_TRADES,
    EVENT_ROBUSTNESS_INFERS_ATTRIBUTION,
    EVENT_ROBUSTNESS_MAX_RATIO,
    EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS,
    EVENT_ROBUSTNESS_MIN_ITEMS,
    EVENT_ROBUSTNESS_NOT_EVALUATED,
    EVENT_ROBUSTNESS_USES_RAW_DROP,
    EVENT_ROBUSTNESS_VERDICTS,
    EVENT_ROBUSTNESS_VERSION,
)
from core.oos_validation import directional_accuracy


class EventRobustnessError(ValueError):
    """Raised when event robustness cannot be judged without guessing."""


# Why the gate could not decide. Distinct because each implies a different fix.
ER_REASON_NO_ATTRIBUTION = "NO_EVENT_ATTRIBUTION"
ER_REASON_TOO_FEW_ITEMS = "TOO_FEW_ITEMS"
ER_REASON_CARRIED = "ONE_ITEM_CARRIES_THE_EDGE"


def attribution_of(
    fold: Mapping[str, Any] | None, axis: str
) -> list[str | None] | None:
    # `list[str | None] | None`: A1 made partial attribution possible, so an entry
    # may be None while the list is present. Caught by mypy in D2.
    """Per-observation event ids or sources for a fold, or None.

    Returns None when the fold carries no attribution. **It never derives one.**
    Bucketing every observation under a single synthetic id would leave
    leave-one-out nothing to leave, and the gate would return ROBUST having
    tested nothing at all.
    """
    if axis not in EVENT_ROBUSTNESS_AXES:
        raise EventRobustnessError(
            f"unknown axis {axis!r}; the tested axes are "
            f"{list(EVENT_ROBUSTNESS_AXES)}"
        )
    if fold is None:
        return None
    if not isinstance(fold, Mapping):
        raise EventRobustnessError("a fold must be a mapping")
    key = "events" if axis == EVENT_ROBUSTNESS_AXIS_EVENT else "sources"
    values = fold.get(key)
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EventRobustnessError(
            f"fold {key} must be a sequence, got {type(values).__name__}"
        )
    # AN UNOBSERVED ENTRY IS NOT AN ID. A1 records None where the provider gave
    # nothing for that observation, and `str(None)` would make "None" a single
    # event covering the whole fold — leave-one-out over one group leaves
    # nothing to leave, and the gate would report a verdict having tested
    # nothing. That is the exact failure the docstring above refuses to commit
    # by deriving an id.
    resolved = [
        None if value is None or not str(value).strip() else str(value)
        for value in values
    ]
    if not any(value is not None for value in resolved):
        return None
    return resolved


def group_by_item(
    predictions: Sequence[float],
    actuals: Sequence[float],
    items: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Split paired observations into per-item cells."""
    if not (len(predictions) == len(actuals) == len(items)):
        raise EventRobustnessError(
            f"{len(predictions)} predictions, {len(actuals)} actuals and "
            f"{len(items)} attributions; a mismatched set cannot be grouped"
        )
    if len(predictions) == 0:
        raise EventRobustnessError("an empty book has no edge to attribute")
    cells: dict[str, dict[str, Any]] = {}
    for predicted, actual, item in zip(predictions, actuals, items):
        if item is None or str(item) == "":
            raise EventRobustnessError(
                "an observation carries no event or source; assigning one "
                "would leave nothing to leave out, and the gate would return "
                "ROBUST having tested nothing"
            )
        cell = cells.setdefault(str(item), {"predictions": [], "actuals": []})
        cell["predictions"].append(predicted)
        cell["actuals"].append(actual)
    total = len(predictions)
    for cell in cells.values():
        cell["observations"] = len(cell["predictions"])
        cell["share"] = cell["observations"] / total
        cell["directional_accuracy"] = directional_accuracy(
            cell["predictions"], cell["actuals"]
        )
        cell["usable"] = cell["observations"] >= EVENT_ROBUSTNESS_MIN_ITEM_OBSERVATIONS
    return cells


def carrier_ratio(
    predictions: Sequence[float],
    actuals: Sequence[float],
    items: Sequence[str],
) -> dict[str, Any]:
    """The worst drop-to-share ratio over leaving each item out.

    The SCALE-FREE statistic. A raw drop cannot be compared across books of
    different composition, because MEASURED, the null drop scales with the
    removed item's share.
    """
    cells = group_by_item(predictions, actuals, items)
    if len(cells) < EVENT_ROBUSTNESS_MIN_ITEMS:
        raise EventRobustnessError(
            f"{len(cells)} item(s) is below the {EVENT_ROBUSTNESS_MIN_ITEMS} "
            f"needed: with one there is nothing to leave out, and with two "
            f"every removal halves the book"
        )
    pooled = directional_accuracy(predictions, actuals)
    ratios: dict[str, dict[str, float]] = {}
    for item, cell in cells.items():
        kept_predictions: list[float] = []
        kept_actuals: list[float] = []
        for other, other_cell in cells.items():
            if other == item:
                continue
            kept_predictions.extend(other_cell["predictions"])
            kept_actuals.extend(other_cell["actuals"])
        if not kept_predictions:
            continue
        without = directional_accuracy(kept_predictions, kept_actuals)
        drop = pooled - without
        ratios[item] = {
            "without": without,
            "drop": drop,
            "share": cell["share"],
            "ratio": drop / cell["share"],
        }
    worst = max(ratios, key=lambda item: ratios[item]["ratio"])
    return {
        "pooled": pooled,
        "observations": len(predictions),
        "cells": cells,
        "ratios": ratios,
        "carrier": worst,
        "worst_ratio": ratios[worst]["ratio"],
        "worst_drop": ratios[worst]["drop"],
        "carrier_share": ratios[worst]["share"],
    }


def evaluate_event_robustness(
    predictions: Sequence[float] | None,
    actuals: Sequence[float] | None,
    events: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    *,
    estimator: str | None = None,
) -> dict:
    """Judge whether an edge survives removing any single event or source."""
    detail: dict[str, Any] = {
        "estimator": estimator,
        "max_ratio": EVENT_ROBUSTNESS_MAX_RATIO,
        "min_items": EVENT_ROBUSTNESS_MIN_ITEMS,
        "axes": list(EVENT_ROBUSTNESS_AXES),
    }

    supplied = {
        EVENT_ROBUSTNESS_AXIS_EVENT: events,
        EVENT_ROBUSTNESS_AXIS_SOURCE: sources,
    }
    available = [axis for axis, values in supplied.items() if values is not None]
    detail["axes_available"] = available

    if predictions is None or actuals is None or not available:
        return _report(
            EVENT_ROBUSTNESS_NOT_EVALUATED,
            reason_code=ER_REASON_NO_ATTRIBUTION,
            reason=(
                "no per-observation event or source attribution was supplied, "
                "so the edge cannot be traced to either; MEASURED, there are 0 "
                "event memories on disk and the raw store holds only price and "
                "fundamentals, so no news has ever been ingested. TWO fixes are "
                "needed, not one: ingest news, AND carry the event id and "
                "source onto each validation observation"
            ),
            **detail,
        )

    axis_results: dict[str, Any] = {}
    unusable: list[str] = []
    for axis in available:
        # `available` already filtered out the None axes at the top, but the dict
        # lookup loses that narrowing. Binding it explicitly keeps the guarantee
        # visible to a reader and to the type checker, rather than asserting it in
        # a comment.
        attribution = supplied[axis]
        if attribution is None:
            continue
        try:
            axis_results[axis] = carrier_ratio(predictions, actuals, attribution)
        except EventRobustnessError as exc:
            unusable.append(f"{axis}: {exc}")

    if not axis_results:
        return _report(
            EVENT_ROBUSTNESS_NOT_EVALUATED,
            reason_code=ER_REASON_TOO_FEW_ITEMS,
            reason="; ".join(unusable) or "no axis could be evaluated",
            **detail,
        )

    detail["axes_tested"] = sorted(axis_results)
    detail["unusable_axes"] = unusable
    detail["by_axis"] = {
        axis: {
            "carrier": result["carrier"],
            "worst_ratio": result["worst_ratio"],
            "worst_drop": result["worst_drop"],
            "carrier_share": result["carrier_share"],
            "items": len(result["cells"]),
            "pooled": result["pooled"],
        }
        for axis, result in sorted(axis_results.items())
    }

    worst_axis = max(
        axis_results, key=lambda axis: axis_results[axis]["worst_ratio"]
    )
    worst = axis_results[worst_axis]
    detail["worst_axis"] = worst_axis
    detail["carrier"] = worst["carrier"]
    detail["worst_ratio"] = worst["worst_ratio"]
    detail["worst_drop"] = worst["worst_drop"]
    detail["carrier_share"] = worst["carrier_share"]
    detail["pooled_accuracy"] = worst["pooled"]
    detail["uses_raw_drop"] = EVENT_ROBUSTNESS_USES_RAW_DROP

    if worst["worst_ratio"] > EVENT_ROBUSTNESS_MAX_RATIO:
        return _report(
            EVENT_CARRIED,
            reason_code=ER_REASON_CARRIED,
            reason=(
                f"removing {worst['carrier']!r} ({worst_axis}, "
                f"{worst['carrier_share']:.1%} of observations) drops accuracy "
                f"by {worst['worst_drop']:.4f} - a drop-to-share ratio of "
                f"{worst['worst_ratio']:.3f} above the "
                f"{EVENT_ROBUSTNESS_MAX_RATIO} bar; MEASURED, a uniform edge "
                f"reached at most 0.308 over 480 books"
            ),
            **detail,
        )

    return _report(
        EVENT_ROBUST,
        reason_code=None,
        reason=(
            f"no single event or source carries the edge: the worst "
            f"drop-to-share ratio is {worst['worst_ratio']:.3f} (removing "
            f"{worst['carrier']!r} on the {worst_axis} axis), within the "
            f"{EVENT_ROBUSTNESS_MAX_RATIO} bar across "
            f"{len(axis_results)} axis/axes"
        ),
        **detail,
    )


def _report(verdict: str, **detail) -> dict:
    """One X4 answer."""
    if verdict not in EVENT_ROBUSTNESS_VERDICTS:
        raise EventRobustnessError(f"unknown event robustness verdict {verdict!r}")
    payload = {
        "version": EVENT_ROBUSTNESS_VERSION,
        "gate": "event_robustness",
        "verdict": verdict,
        "infers_attribution": EVENT_ROBUSTNESS_INFERS_ATTRIBUTION,
        "blocks_trades": EVENT_ROBUSTNESS_BLOCKS_TRADES,
        "note": _NOTE,
    }
    payload.setdefault("uses_raw_drop", EVENT_ROBUSTNESS_USES_RAW_DROP)
    payload.update(detail)
    return payload


_NOTE = (
    "MEASURED, there are 0 event memories and the raw store holds only price "
    "and fundamentals, so no news has ever been ingested - X4's blocker is "
    "MISSING DATA, unlike X3's dropped join key. The statistic is SCALE-FREE: "
    "a raw drop false-flags because the null drop scales with the removed "
    "item's share (p95/share is ~0.20 from 3 to 20 events). The 0.35 bar sits "
    "between a uniform maximum of 0.322 and a carrier minimum of 0.356, once "
    "three-item books are excluded - with them the null reaches 0.350. Both "
    "axes are tested because a source supplying every third event scores 0.425 "
    "on source and only 0.340 on event."
)


def event_robustness_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an event robustness report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if report.get("version") != EVENT_ROBUSTNESS_VERSION:
        problems.append(
            f"version is {report.get('version')!r}, expected "
            f"{EVENT_ROBUSTNESS_VERSION!r}"
        )

    verdict = report.get("verdict")
    if verdict not in EVENT_ROBUSTNESS_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")

    if report.get("blocks_trades"):
        problems.append("X4 reports; the registry promotes")

    if report.get("infers_attribution"):
        problems.append(
            "an unattributed observation must never be assigned an event or "
            "source; that leaves nothing to leave out"
        )

    if report.get("uses_raw_drop"):
        problems.append(
            "a raw drop is not the test: it scales with the removed item's "
            "share, so it false-flags books with few events"
        )

    if verdict == EVENT_ROBUST:
        ratio = report.get("worst_ratio")
        if ratio is None or float(ratio) > EVENT_ROBUSTNESS_MAX_RATIO:
            problems.append(
                f"ROBUST with a worst ratio of {ratio!r}, above the "
                f"{EVENT_ROBUSTNESS_MAX_RATIO} bar"
            )
        tested = report.get("axes_tested")
        if not tested:
            problems.append("ROBUST without testing any axis")

    if verdict == EVENT_CARRIED:
        if not report.get("carrier"):
            problems.append(
                "a CARRIED verdict must name the event or source that carries "
                "the edge, or the finding cannot be acted on"
            )
        if not report.get("worst_axis"):
            problems.append(
                "a CARRIED verdict must name WHICH AXIS found the carrier; "
                "'an event' and 'a source' imply different responses"
            )

    if verdict == EVENT_ROBUSTNESS_NOT_EVALUATED and not report.get("reason_code"):
        problems.append(
            "a NOT_EVALUATED verdict must name WHY: missing attribution and "
            "too few items imply different fixes"
        )

    reason = report.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("every verdict must carry a reason")

    return problems


def render_event_robustness(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for one event robustness report."""
    lines = [
        f"Event robustness: {_shown(report.get('estimator'))}"
        f" -> {report.get('verdict')}"
        f"{'' if not report.get('reason_code') else ' (' + report['reason_code'] + ')'}",
        f"  pooled     : {_num(report.get('pooled_accuracy'))}"
        f"  axes tested: {', '.join(report.get('axes_tested') or []) or 'NONE'}",
    ]
    for axis, result in (report.get("by_axis") or {}).items():
        marker = "  <- worst" if axis == report.get("worst_axis") else ""
        lines.append(
            f"    {axis:<7} {result['items']:>3} item(s)"
            f"  carrier {str(result['carrier'])[:18]:<18}"
            f" share {result['carrier_share']:.1%}"
            f" drop {result['drop'] if 'drop' in result else result['worst_drop']:+.4f}"
            f" ratio {result['worst_ratio']:.3f}{marker}"
        )
    lines.extend(
        [
            f"  worst ratio: {_num3(report.get('worst_ratio'))}"
            f"  (bar {report.get('max_ratio')})",
            f"  raw drop   : {_num(report.get('worst_drop'))}"
            f"  (recorded; decides: {report.get('uses_raw_drop')})",
            f"  reason     : {report.get('reason')}",
        ]
    )
    return lines


def _num(value: Any) -> str:
    """A number at fixed precision, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.4f}"


def _num3(value: Any) -> str:
    """A ratio at three decimals, or why there is none."""
    return "ABSENT" if value is None else f"{float(value):.3f}"


def _shown(value: Any) -> str:
    """A value, or why there is none."""
    return "ABSENT" if value is None else str(value)
