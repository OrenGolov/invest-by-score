"""B4 cluster exposure — a basket of correlated holdings is one bet.

R1 flags exposure per POSITION, on both weight and risk. A CLUSTER of correlated
holdings is a different question, and R1 cannot see it.

**MEASURED on 3,771 sessions of real returns**, a book of 10% each in
NVDA/AMD/AVGO/SOXX plus 20% each in MSFT/GOOGL/CAT:

    no holding exceeds 17% of risk
      AMD 16.3%  GOOGL 16.1%  MSFT 15.8%  CAT 15.5%  NVDA 14.1%  AVGO 11.1%  SOXX 11.1%

    the four semiconductors TOGETHER hold 52.6% of portfolio variance,
    at a mean pairwise correlation of 0.62

That book raises **zero R1 flags**. R3 addressed sector concentration, but a
cluster need not follow sector labels — SOXX is an ETF, and the correlation that
binds these four is not a label.

**WHY AVERAGE LINKAGE, MEASURED RATHER THAN PREFERRED.** Open item 5 parked this
because "defining a cluster needs its own evidence — by sector, by correlation
clustering, or by factor loading — and each choice is a measurement." Sweeping
both linkage rules across 13 thresholds:

    thr    single   average
    0.450   100.0%    84.5%
    0.500   100.0%    52.6%     <- average finds exactly the four semis
    0.550   100.0%    36.3%
    0.600    52.6%    36.3%     <- single finds them only here
    0.650    36.3%    36.3%

A FIRST READING OF THIS WAS WRONG: I took average linkage to be more stable, but
the two spreads are nearly identical (63.7% vs 62.3%). The real discriminator is
**degeneracy**. Single linkage reports the WHOLE BOOK as one cluster at 5 of 13
thresholds, because SOXX correlates 0.55–0.76 with everything and MSFT–GOOGL is
0.59, so it chains the semis to the non-semis. A cluster containing every holding
cannot distinguish a concentrated book from a diversified one. Average linkage
degenerates at 0 of 13.

**This module reports; it does not block.** Consistent with R1.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

from core.config import (
    CLUSTER_EXPOSURE_LINKAGE,
    CLUSTER_EXPOSURE_MIN_CORRELATION,
    CLUSTER_EXPOSURE_MIN_MEMBERS,
    CLUSTER_EXPOSURE_RISK_REVIEW,
    CLUSTER_EXPOSURE_VERSION,
    CLUSTER_NOT_EVALUATED,
    CLUSTER_VERDICTS,
    CLUSTER_CONCENTRATED,
    CLUSTER_DIFFUSE,
    CLUSTER_EXPOSURE_BLOCKS_TRADES,
)
from core.position_exposure import PositionExposureError, risk_contributions


class ClusterExposureError(ValueError):
    """Raised when a cluster request is structurally invalid."""


LINKAGE_AVERAGE = "average"
LINKAGE_SINGLE = "single"
LINKAGES: tuple[str, ...] = (LINKAGE_AVERAGE, LINKAGE_SINGLE)


def correlation_matrix(
    tickers: Sequence[str], cov: Sequence[Sequence[float]]
) -> dict[tuple[str, str], float]:
    """Pairwise correlations from a covariance matrix.

    A zero-variance asset has no correlation with anything — not a correlation of
    zero, which would read as "independent" when the truth is "unmeasurable".
    """
    index = {ticker: position for position, ticker in enumerate(tickers)}
    correlations: dict[tuple[str, str], float] = {}
    for left, right in itertools.combinations(sorted(index), 2):
        i, j = index[left], index[right]
        denominator = cov[i][i] * cov[j][j]
        if denominator <= 0:
            continue
        correlations[(left, right)] = cov[i][j] / math.sqrt(denominator)
    return correlations


def _pair(correlations: Mapping[tuple[str, str], float], a: str, b: str) -> float | None:
    if a == b:
        return 1.0
    key = (a, b) if (a, b) in correlations else (b, a)
    return correlations.get(key)


def build_clusters(
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
    *,
    threshold: float = CLUSTER_EXPOSURE_MIN_CORRELATION,
    linkage: str = CLUSTER_EXPOSURE_LINKAGE,
) -> list[list[str]]:
    """Group holdings whose mutual correlation clears `threshold`.

    AVERAGE LINKAGE by default, and the reason is measured rather than assumed:
    single linkage merges on ONE qualifying pair, so a broadly-correlated ETF
    chains unrelated holdings together and the whole book becomes one cluster at
    5 of 13 thresholds swept. Average linkage requires the mean cross-correlation
    to clear the bar and degenerated at none of them.
    """
    if linkage not in LINKAGES:
        raise ClusterExposureError(
            f"{linkage!r} is not a known linkage (known: {list(LINKAGES)})"
        )
    if not -1.0 < threshold < 1.0:
        raise ClusterExposureError("the correlation threshold must lie inside (-1, 1)")

    correlations = correlation_matrix(tickers, cov)
    clusters: list[list[str]] = [[ticker] for ticker in sorted(set(tickers))]

    while True:
        best: tuple[float, int, int] | None = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            pairs = [
                value
                for a in clusters[i]
                for b in clusters[j]
                if (value := _pair(correlations, a, b)) is not None
            ]
            if not pairs:
                continue
            score = max(pairs) if linkage == LINKAGE_SINGLE else sum(pairs) / len(pairs)
            if score >= threshold and (best is None or score > best[0]):
                best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        clusters[i] = sorted(clusters[i] + clusters[j])
        clusters.pop(j)

    return sorted(clusters, key=lambda group: (-len(group), group))


def mean_pairwise_correlation(
    members: Sequence[str],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> float | None:
    """The mean correlation inside a group, or None for a single holding.

    None, not 1.0: a lone holding has no internal correlation to report, and 1.0
    would read as a perfectly correlated pair.
    """
    if len(members) < 2:
        return None
    correlations = correlation_matrix(tickers, cov)
    values = [
        value
        for a, b in itertools.combinations(sorted(members), 2)
        if (value := _pair(correlations, a, b)) is not None
    ]
    return sum(values) / len(values) if values else None


def evaluate_cluster_exposure(
    weights: Mapping[str, float] | None,
    tickers: Sequence[str] | None,
    cov: Sequence[Sequence[float]] | None,
    *,
    threshold: float = CLUSTER_EXPOSURE_MIN_CORRELATION,
    linkage: str = CLUSTER_EXPOSURE_LINKAGE,
    risk_review: float = CLUSTER_EXPOSURE_RISK_REVIEW,
) -> dict:
    """How much risk sits in the largest correlated cluster.

    NOT_EVALUATED when the risk contributions cannot be computed — a portfolio
    whose covariance is unavailable has an UNKNOWN cluster exposure, and
    reporting 0% would make the least-understood book look the safest.
    """
    if not weights or not tickers or not cov:
        return _result(
            CLUSTER_NOT_EVALUATED,
            "no portfolio, tickers or covariance was supplied, so no cluster "
            "can be located",
        )
    try:
        contributions = risk_contributions(weights, tickers, cov)
    except (
        PositionExposureError,
        ZeroDivisionError,
        ValueError,
        # A RAGGED covariance raises IndexError, and a first version omitted it —
        # so a malformed matrix CRASHED instead of reporting NOT_EVALUATED. Found
        # by a sabotage that swapped this verdict and passed both the gate and the
        # tests, because no probe reached the branch at all. An unreachable
        # fail-closed path is worse than an untested one: it looks like a
        # guarantee and is not.
        IndexError,
        TypeError,
    ) as error:
        return _result(
            CLUSTER_NOT_EVALUATED,
            f"risk contributions could not be computed: {error}",
        )
    if not contributions:
        return _result(
            CLUSTER_NOT_EVALUATED,
            "no risk contribution could be measured, so a cluster's share of "
            "risk is unknown rather than zero",
        )

    held = {ticker for ticker, weight in weights.items() if weight}
    clusters = [
        [member for member in group if member in held]
        for group in build_clusters(tickers, cov, threshold=threshold, linkage=linkage)
    ]
    clusters = [group for group in clusters if group]

    described = []
    for group in clusters:
        share = sum(contributions.get(member, 0.0) for member in group)
        described.append(
            {
                "members": sorted(group),
                "size": len(group),
                "risk_share": round(share, 6),
                "weight_share": round(
                    sum(float(weights.get(member, 0.0)) for member in group), 6
                ),
                "mean_correlation": (
                    round(value, 4)
                    if (value := mean_pairwise_correlation(group, tickers, cov))
                    is not None
                    else None
                ),
            }
        )
    described.sort(key=lambda entry: -entry["risk_share"])

    # A CLUSTER OF ONE IS A POSITION, and R1 already flags those. Reporting it
    # here would double-count the same concentration under a second name.
    multi = [entry for entry in described if entry["size"] >= CLUSTER_EXPOSURE_MIN_MEMBERS]
    worst = multi[0] if multi else None

    if worst is None:
        return _result(
            CLUSTER_DIFFUSE,
            (
                f"no {CLUSTER_EXPOSURE_MIN_MEMBERS}-holding group correlates above "
                f"{threshold:.2f}; every position stands alone and R1's per-position "
                f"flags are the whole picture"
            ),
            clusters=described,
            largest_cluster=None,
            threshold=threshold,
            linkage=linkage,
        )

    concentrated = worst["risk_share"] > risk_review
    return _result(
        CLUSTER_CONCENTRATED if concentrated else CLUSTER_DIFFUSE,
        (
            f"{worst['members']} hold {worst['risk_share']:.1%} of portfolio "
            f"variance at a mean correlation of {worst['mean_correlation']:.2f}, "
            f"{'above' if concentrated else 'within'} the {risk_review:.0%} review "
            f"level — on {worst['weight_share']:.1%} of the weight"
        ),
        clusters=described,
        largest_cluster=worst,
        threshold=threshold,
        linkage=linkage,
        risk_review=risk_review,
    )


def _result(verdict: str, reason: str, **detail) -> dict:
    if verdict not in CLUSTER_VERDICTS:
        raise ClusterExposureError(f"{verdict!r} is not a cluster verdict")
    record = {
        "version": CLUSTER_EXPOSURE_VERSION,
        "verdict": verdict,
        "reason": reason,
        "blocks_trades": CLUSTER_EXPOSURE_BLOCKS_TRADES,
    }
    record.update(detail)
    return record


def cluster_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a cluster report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    if report.get("verdict") not in CLUSTER_VERDICTS:
        problems.append(f"unknown cluster verdict {report.get('verdict')!r}")
    if not str(report.get("reason") or "").strip():
        problems.append("no reason given")
    if report.get("blocks_trades"):
        problems.append("B4 reports; the risk gate blocks")

    # THE SHAPE RULE: a cluster and its risk share exist IFF one was located.
    # A 0.0 share would read as "no risk here", which is a different claim from
    # "no cluster could be found".
    if report.get("verdict") == CLUSTER_NOT_EVALUATED:
        if report.get("largest_cluster") is not None:
            problems.append(
                "NOT_EVALUATED but a cluster is reported; an unmeasurable "
                "portfolio has an UNKNOWN cluster exposure, not a located one"
            )
    if report.get("verdict") == CLUSTER_CONCENTRATED:
        largest = report.get("largest_cluster")
        if not largest:
            problems.append("CONCENTRATED without naming the cluster")
        elif largest.get("risk_share") is None:
            problems.append("CONCENTRATED without a risk share")
        elif largest.get("size", 0) < CLUSTER_EXPOSURE_MIN_MEMBERS:
            problems.append(
                f"CONCENTRATED on a cluster of {largest.get('size')}; a cluster "
                f"of one is a position and R1 already flags it"
            )
    for entry in report.get("clusters") or []:
        if entry.get("size", 0) >= 2 and entry.get("mean_correlation") is None:
            problems.append(
                f"{entry.get('members')}: a multi-member cluster reports no mean "
                f"correlation, so why it is a cluster cannot be checked"
            )
    return problems


def render_cluster_exposure(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines. Absent facts render as ABSENT, never as zero."""
    lines = [f"  verdict            {report.get('verdict')}"]
    largest = report.get("largest_cluster")
    if largest is None:
        lines.append("  largest cluster    ABSENT")
    else:
        lines.append(
            f"  largest cluster    {', '.join(largest['members'])}"
        )
        lines.append(f"    risk share       {largest['risk_share']:.1%}")
        lines.append(f"    weight share     {largest['weight_share']:.1%}")
        lines.append(
            f"    mean correlation {largest['mean_correlation']:.2f}"
            if largest.get("mean_correlation") is not None
            else "    mean correlation ABSENT"
        )
    lines.append(
        f"  linkage            {report.get('linkage', 'ABSENT')} "
        f"@ {report.get('threshold', 'ABSENT')}"
    )
    for entry in report.get("clusters") or []:
        if entry["size"] >= 2:
            lines.append(
                f"    {', '.join(entry['members']):40s} {entry['risk_share']:6.1%}"
            )
    return lines
