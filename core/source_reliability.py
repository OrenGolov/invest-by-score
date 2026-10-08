"""L4 source reliability — learned per source, conditioned only where earned.

"Source quality may depend on source × event type × sector × horizon — one
global source score is not assumed sufficient."

**The task states a hypothesis, so this module tests it rather than assuming
it.** Two measurements shaped everything here.

**Today's source quality is not learned at all.** `SOURCE_REGISTRY` carries one
asserted `base_confidence` per DOMAIN (news 0.75, fundamentals 0.90) that no
outcome ever updates, and the `source_quality` field on a news article is
populated in 0 of 2650 stored records.

**Conditioning is not free.** MEASURED against a source whose true accuracy
really does vary by cell, per-cell estimation is five times worse than a single
global rate at 5 observations per cell, and only starts to pay from ~25. So the
estimator is SHRINKAGE: a cell is pulled toward the global rate by k
pseudo-counts, which *is* the global score when a cell is empty and becomes the
cell's own score once the evidence exists. That is how this module honours both
halves of the task — it never assumes one global score suffices, and it never
pretends to a conditional score it has not earned.

**Every score says how it was backed.** NO_EVIDENCE, REGISTRY_PRIOR, GLOBAL and
CONDITIONAL are four different epistemic states, and a reader who cannot tell
them apart cannot tell a learned source property from an asserted constant.

**Absence is never zero.** An unmeasured outlet and an outlet measured to be
useless are different facts; collapsing them would discard every new source the
moment it appeared.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from core.config import (
    CONDITIONING_NOISE_INTERCEPT,
    CONDITIONING_NOISE_LOG_SLOPE,
    SOURCE_BACKING_CELL,
    SOURCE_BACKING_GLOBAL,
    SOURCE_BACKING_NONE,
    SOURCE_BACKING_PRIOR,
    SOURCE_RELIABILITY_ABSENT_IS_ZERO,
    SOURCE_RELIABILITY_BACKINGS,
    SOURCE_RELIABILITY_DIMENSIONS,
    SOURCE_RELIABILITY_JOIN_GAP,
    SOURCE_RELIABILITY_MIN_CELL,
    SOURCE_RELIABILITY_SHRINKAGE_K,
    SOURCE_RELIABILITY_VERSION,
)


class SourceReliabilityError(ValueError):
    """Raised when a reliability request is structurally invalid."""


# The dimensions other than the source itself, in fallback order. A cell backs
# off along this sequence: the more specific the cell, the thinner it is.
CONDITIONING_DIMENSIONS: tuple[str, ...] = tuple(
    name for name in SOURCE_RELIABILITY_DIMENSIONS if name != "source"
)


def _rate(hits: float, total: float) -> float | None:
    if total <= 0:
        return None
    return hits / total


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval for a hit rate.

    Reported so a reader can see how little a small cell actually pins down.
    MEASURED at p=0.6: ±0.326 at n=5 and ±0.043 at n=500, which is why
    separating a 0.65 source from a 0.55 one needs ~380 observations.
    """
    if total <= 0:
        return None
    proportion = hits / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def observation_records(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Normalise scored source observations.

    An observation is one article/event from a named OUTLET whose claim was
    later scored against an outcome. A row without an outlet or without a
    scored result teaches nothing about a source and is dropped rather than
    counted as a miss — counting it would penalise a source for the system's
    own missing join.
    """
    records: list[dict] = []
    for row in rows or []:
        source = str((row or {}).get("source") or "").strip()
        if not source:
            continue
        if "hit" not in (row or {}):
            continue
        hit = (row or {}).get("hit")
        if hit is None:
            continue
        records.append(
            {
                "source": source,
                "hit": 1 if bool(hit) else 0,
                "event_type": str((row or {}).get("event_type") or "") or None,
                "sector": str((row or {}).get("sector") or "") or None,
                "horizon": str((row or {}).get("horizon") or "") or None,
            }
        )
    return records


def _cell_key(record: Mapping[str, Any], dimensions: Iterable[str]) -> tuple:
    return tuple(record.get(name) for name in dimensions)


def tally(
    records: Iterable[Mapping[str, Any]], dimensions: Iterable[str] = ()
) -> dict[tuple, dict]:
    """Hits and totals per (source, *dimensions) cell."""
    dimensions = tuple(dimensions)
    counts: dict[tuple, dict] = defaultdict(lambda: {"hits": 0, "total": 0})
    for record in records or []:
        key = (record.get("source"),) + _cell_key(record, dimensions)
        counts[key]["hits"] += int(record.get("hit") or 0)
        counts[key]["total"] += 1
    return dict(counts)


def shrunk_rate(
    hits: float, total: float, prior: float, k: float = SOURCE_RELIABILITY_SHRINKAGE_K
) -> float:
    """A cell rate pulled toward `prior` by `k` pseudo-counts.

    With no evidence this IS the prior; with abundant evidence the prior washes
    out. That is the whole point: the estimator degrades gracefully to the
    global score rather than inventing a conditional one.
    """
    if k < 0:
        raise SourceReliabilityError("shrinkage k cannot be negative")
    denominator = total + k
    if denominator <= 0:
        return float(prior)
    return (hits + k * prior) / denominator


def score_source(
    records: Iterable[Mapping[str, Any]],
    source: str,
    *,
    event_type: str | None = None,
    sector: str | None = None,
    horizon: str | None = None,
    registry_prior: float | None = None,
    k: float = SOURCE_RELIABILITY_SHRINKAGE_K,
    min_cell: int = SOURCE_RELIABILITY_MIN_CELL,
) -> dict:
    """Score one source in the most specific cell its evidence supports.

    Backs off along CONDITIONING_DIMENSIONS: the most specific cell is tried
    first, and a cell below `min_cell` is still USED (shrinkage makes it safe)
    but reported as GLOBAL-backed, because a shrinkage-dominated number is not
    a learned source property and must not be read as one.
    """
    source = str(source or "").strip()
    if not source:
        raise SourceReliabilityError("a source name is required")

    records = [r for r in observation_records(records) if r["source"] == source]

    requested = {"event_type": event_type, "sector": sector, "horizon": horizon}
    asked = tuple(name for name in CONDITIONING_DIMENSIONS if requested.get(name))

    global_hits = sum(r["hit"] for r in records)
    global_total = len(records)
    global_rate = _rate(global_hits, global_total)

    # The fallback prior for shrinkage: the source's own overall rate when it
    # has one, else the registry prior, else neutral. Never 0.0 — see below.
    if global_rate is not None:
        prior = global_rate
    elif registry_prior is not None:
        prior = float(registry_prior)
    else:
        prior = 0.5

    result: dict[str, Any] = {
        "source": source,
        "version": SOURCE_RELIABILITY_VERSION,
        "observations": global_total,
        "shrinkage_k": k,
    }

    # No evidence anywhere for this source.
    if global_total == 0:
        if SOURCE_RELIABILITY_ABSENT_IS_ZERO:  # pragma: no cover - guarded in config
            raise SourceReliabilityError("absence must not be scored as zero")
        backing = (
            SOURCE_BACKING_PRIOR if registry_prior is not None else SOURCE_BACKING_NONE
        )
        result.update(
            {
                "score": float(registry_prior) if registry_prior is not None else None,
                "backing": backing,
                "cell": {},
                "reason": (
                    "no scored observation exists for this source. "
                    + SOURCE_RELIABILITY_JOIN_GAP
                ),
            }
        )
        return result

    # Most specific cell first, then back off.
    for depth in range(len(asked), -1, -1):
        dimensions = asked[:depth]
        matched = [
            r
            for r in records
            if all(r.get(name) == requested.get(name) for name in dimensions)
        ]
        if not matched and dimensions:
            continue
        hits = sum(r["hit"] for r in matched)
        total = len(matched)
        if total == 0:
            continue

        conditional = bool(dimensions) and total >= min_cell
        score = shrunk_rate(hits, total, prior, k)
        interval = wilson_interval(hits, total)
        result.update(
            {
                "score": round(float(score), 6),
                "backing": SOURCE_BACKING_CELL if conditional else SOURCE_BACKING_GLOBAL,
                "cell": {name: requested.get(name) for name in dimensions},
                "cell_observations": total,
                "cell_raw_rate": round(float(hits / total), 6),
                "global_rate": round(float(global_rate), 6),
                "interval": (
                    [round(interval[0], 6), round(interval[1], 6)] if interval else None
                ),
                "reason": (
                    f"{total} observation(s) in this cell"
                    if conditional
                    else (
                        f"{total} observation(s) is below the {min_cell} needed to "
                        f"report a cell-specific rate; the shrunk estimate is "
                        f"dominated by the source's overall rate"
                    )
                ),
            }
        )
        return result

    # Reachable only when every requested cell is empty: fall back to global.
    score = shrunk_rate(global_hits, global_total, prior, k)
    result.update(
        {
            "score": round(float(score), 6),
            "backing": SOURCE_BACKING_GLOBAL,
            "cell": {},
            "cell_observations": global_total,
            "global_rate": round(float(global_rate), 6),
            "reason": "no observation in any requested cell; using the overall rate",
        }
    )
    return result


def conditioning_gain(
    records: Iterable[Mapping[str, Any]], dimension: str
) -> dict:
    """Does conditioning on `dimension` explain anything, or just split noise?

    Reports the spread of per-cell rates against what pure noise would produce
    at the same cell sizes. A dimension whose observed spread sits inside the
    noise band is NOT evidence that source quality depends on it — which is the
    claim the task explicitly declines to assume.
    """
    if dimension not in CONDITIONING_DIMENSIONS:
        raise SourceReliabilityError(
            f"{dimension!r} is not a conditioning dimension; "
            f"expected one of {CONDITIONING_DIMENSIONS}"
        )
    records = observation_records(records)
    if not records:
        return {
            "dimension": dimension,
            "cells": 0,
            "observed_spread": None,
            "noise_spread": None,
            "explains": False,
            "reason": "no observations",
        }

    overall = sum(r["hit"] for r in records) / len(records)
    cells: dict[Any, list[int]] = defaultdict(list)
    for record in records:
        cells[record.get(dimension)].append(record["hit"])

    usable = {key: values for key, values in cells.items() if len(values) > 0}
    if len(usable) < 2:
        return {
            "dimension": dimension,
            "cells": len(usable),
            "observed_spread": None,
            "noise_spread": None,
            "explains": False,
            "reason": "fewer than two cells — nothing to compare",
        }

    rates = [sum(values) / len(values) for values in usable.values()]
    observed = max(rates) - min(rates)

    # What spread would appear from noise alone at these cell sizes?
    #
    # THE BAND MUST TRACK THE TAIL OF THE NULL, NOT ITS MEAN. A first version
    # used a flat 2.0 multiplier, which sits near the MEAN of the null range —
    # so roughly half of clean runs near the boundary exceeded it, and MEASURED
    # it claimed conditioning on 7 of 30 runs with no effect present at all.
    #
    # MEASURED, the multiplier that puts the band at the 95th percentile of the
    # null range grows with the NUMBER OF CELLS, because more cells means more
    # chances for a wide spread by luck — the same multiple-comparison effect
    # L3 measured:
    #
    #     cells    multiplier needed for p95
    #       2                          1.90
    #       3                          2.35
    #       5                          2.70
    #       9                          3.11
    #
    # Fitted: c = 1.4 + 0.8*ln(cells).
    errors = [
        math.sqrt(max(overall * (1.0 - overall), 1e-9) / len(values))
        for values in usable.values()
    ]
    multiplier = (
        CONDITIONING_NOISE_INTERCEPT
        + CONDITIONING_NOISE_LOG_SLOPE * math.log(max(len(usable), 2))
    )
    noise = multiplier * (sum(errors) / len(errors)) * math.sqrt(2.0)

    return {
        "dimension": dimension,
        "cells": len(usable),
        "observed_spread": round(float(observed), 6),
        "noise_spread": round(float(noise), 6),
        "explains": bool(observed > noise),
        "reason": (
            f"observed spread {observed:.3f} exceeds the {noise:.3f} expected "
            f"from noise at these cell sizes"
            if observed > noise
            else (
                f"observed spread {observed:.3f} sits inside the {noise:.3f} "
                f"expected from noise alone — this is not evidence that source "
                f"quality depends on {dimension}"
            )
        ),
    }


def reliability_report(
    records: Iterable[Mapping[str, Any]],
    registry_priors: Mapping[str, float] | None = None,
) -> dict:
    """The full L4 view: per-source scores plus whether conditioning is earned."""
    records = observation_records(records)
    sources = sorted({r["source"] for r in records})
    priors = dict(registry_priors or {})

    scores = {
        source: score_source(
            records, source, registry_prior=priors.get(source)
        )
        for source in sources
    }
    gains = {
        dimension: conditioning_gain(records, dimension)
        for dimension in CONDITIONING_DIMENSIONS
    }

    earned = sorted(name for name, gain in gains.items() if gain.get("explains"))
    return {
        "version": SOURCE_RELIABILITY_VERSION,
        "observations": len(records),
        "sources": len(sources),
        "scores": scores,
        "conditioning": gains,
        "conditioning_earned": earned,
        "join_gap": SOURCE_RELIABILITY_JOIN_GAP if not records else "",
        "absence_note": (
            "A source with no observations is scored from its registry prior, "
            "never 0.0: an unmeasured outlet and a useless one are different "
            "facts."
        ),
    }


def reliability_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    for source, score in (report.get("scores") or {}).items():
        backing = score.get("backing")
        if backing not in SOURCE_RELIABILITY_BACKINGS:
            problems.append(f"{source}: unknown backing {backing!r}")
        # THE SHAPE RULE (inherited F3→L3): a `score` key exists IFF it was
        # measured, because Number(x ?? 0) renders a missing score as 0.0 —
        # the same number as an outlet measured to be worthless.
        if backing == SOURCE_BACKING_NONE and score.get("score") is not None:
            problems.append(
                f"{source}: backed by NO_EVIDENCE but carries a score"
            )
        if backing == SOURCE_BACKING_CELL and not score.get("cell"):
            problems.append(
                f"{source}: reported CONDITIONAL with no cell — a conditional "
                f"score must say what it is conditioned on"
            )
        if backing == SOURCE_BACKING_CELL:
            if int(score.get("cell_observations") or 0) < SOURCE_RELIABILITY_MIN_CELL:
                problems.append(
                    f"{source}: reported CONDITIONAL on "
                    f"{score.get('cell_observations')} observations, below the "
                    f"{SOURCE_RELIABILITY_MIN_CELL} floor"
                )
    return problems


def render_scores(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines. An empty list is a real answer, not a missing one."""
    lines: list[str] = []
    for source, score in sorted((report.get("scores") or {}).items()):
        value = score.get("score")
        shown = "—" if value is None else f"{value:.3f}"
        cell = score.get("cell") or {}
        suffix = f" [{', '.join(f'{k}={v}' for k, v in cell.items())}]" if cell else ""
        lines.append(
            f"{source:28s} {shown:>6s}  {score.get('backing')}{suffix} "
            f"(n={score.get('observations', 0)})"
        )
    return lines
