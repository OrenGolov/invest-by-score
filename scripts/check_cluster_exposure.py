"""B4 governance gate — a basket of correlated holdings is one bet.

Exits 1 when any of these fails.

THE DECIDING MEASUREMENT, RECOMPUTED HERE from the TRACKED price frames rather
than quoted: a book of 10% each in NVDA/AMD/AVGO/SOXX plus 20% each in
MSFT/GOOGL/CAT raises ZERO R1 flags while the four semiconductors hold ~52.6% of
portfolio variance at a mean pairwise correlation of ~0.62.

The frames are tracked parquet, so this recomputes identically on a fresh clone —
and it SKIPS rather than fails when a frame is absent, because a missing price
file is an environment fact, not a governance breach.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.cluster_exposure import (  # noqa: E402
    LINKAGE_AVERAGE,
    LINKAGE_SINGLE,
    build_clusters,
    cluster_problems,
    evaluate_cluster_exposure,
    mean_pairwise_correlation,
    render_cluster_exposure,
)
from core.config import (  # noqa: E402
    CLUSTER_CONCENTRATED,
    CLUSTER_DIFFUSE,
    CLUSTER_EXPOSURE_BLOCKS_TRADES,
    CLUSTER_EXPOSURE_LINKAGE,
    CLUSTER_EXPOSURE_MIN_CORRELATION,
    CLUSTER_EXPOSURE_MIN_MEMBERS,
    CLUSTER_NOT_EVALUATED,
    CLUSTER_VERDICTS,
    EXPOSURE_RISK_REVIEW,
)
from core.position_exposure import (  # noqa: E402
    aligned_returns,
    covariance,
    risk_contributions,
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


BOOK = {
    "NVDA": 0.10,
    "AMD": 0.10,
    "AVGO": 0.10,
    "SOXX": 0.10,
    "MSFT": 0.20,
    "GOOGL": 0.20,
    "CAT": 0.20,
}
SEMIS = {"NVDA", "AMD", "AVGO", "SOXX"}


def load(tickers):
    import pandas as pd

    series = {}
    for ticker in tickers:
        path = REPO_ROOT / "data" / f"{ticker}_15y_1d.parquet"
        if not path.exists():
            return None
        closes = pd.read_parquet(path)["Close"].pct_change().dropna()
        series[ticker] = {
            str(stamp.date()): float(value) for stamp, value in closes.items()
        }
    return series


# --- 1. THE DECIDING MEASUREMENT, on real returns --------------------------------

series = load(BOOK)
if series is None:
    print("B4 cluster exposure gate: SKIPPED")
    print("  a 15y price frame is missing; a missing file is an environment fact")
    sys.exit(0)

tickers, matrix = aligned_returns(series)
cov = covariance(matrix)

contributions = risk_contributions(BOOK, tickers, cov)
over = {t: s for t, s in contributions.items() if s > EXPOSURE_RISK_REVIEW}
check(
    not over,
    f"{over} now exceed R1's per-position risk review level, so this book no "
    f"longer demonstrates the blind spot B4 exists to close and the measurement "
    f"must be revisited",
)

semi_share = sum(contributions.get(t, 0.0) for t in SEMIS)
check(
    semi_share > EXPOSURE_RISK_REVIEW,
    f"the four semiconductors hold {semi_share:.1%} of variance, at or below "
    f"R1's {EXPOSURE_RISK_REVIEW:.0%} level; the cluster is no longer a "
    f"concentration and B4's premise is stale",
)

correlation = mean_pairwise_correlation(sorted(SEMIS), tickers, cov)
check(
    correlation is not None and correlation > CLUSTER_EXPOSURE_MIN_CORRELATION,
    f"the semiconductors' mean pairwise correlation is {correlation}, at or "
    f"below the {CLUSTER_EXPOSURE_MIN_CORRELATION} threshold; they would not "
    f"cluster and the measurement is stale",
)

report = evaluate_cluster_exposure(BOOK, tickers, cov)
check(
    report["verdict"] == CLUSTER_CONCENTRATED,
    f"the book reported {report['verdict']}, not CONCENTRATED; B4's whole "
    f"purpose is to catch this one",
)
largest = report.get("largest_cluster") or {}
check(
    set(largest.get("members") or []) == SEMIS,
    f"the largest cluster is {largest.get('members')}, expected the four "
    f"semiconductors",
)
check(
    largest.get("risk_share", 0) > largest.get("weight_share", 1),
    f"the cluster holds {largest.get('risk_share')} of risk on "
    f"{largest.get('weight_share')} of weight; B4 exists because risk exceeds "
    f"weight in a correlated basket",
)
check(
    cluster_problems(report) == [],
    f"the report is not contract-clean: {cluster_problems(report)}",
)


# --- 2. SINGLE LINKAGE DEGENERATES, which is why average is required -------------
#
# Not because it is less STABLE - a first reading claimed that and the swept
# spreads are nearly identical (63.7% vs 62.3%). Because it merges on ONE
# qualifying pair, so a broadly-correlated ETF chains unrelated holdings together
# and the WHOLE BOOK becomes one cluster. A cluster containing every holding
# cannot distinguish a concentrated book from a diversified one.

degenerate = 0
swept = 0
for step in range(13):
    threshold = round(0.45 + 0.025 * step, 3)
    swept += 1
    groups = build_clusters(tickers, cov, threshold=threshold, linkage=LINKAGE_SINGLE)
    if len(groups) == 1 and len(groups[0]) == len(tickers):
        degenerate += 1
check(
    degenerate > 0,
    f"single linkage degenerated at {degenerate} of {swept} thresholds; the "
    f"measured reason for preferring average linkage no longer holds and the "
    f"choice must be re-derived",
)

average_degenerate = sum(
    1
    for step in range(13)
    if (
        lambda groups: len(groups) == 1 and len(groups[0]) == len(tickers)
    )(
        build_clusters(
            tickers,
            cov,
            threshold=round(0.45 + 0.025 * step, 3),
            linkage=LINKAGE_AVERAGE,
        )
    )
)
check(
    average_degenerate == 0,
    f"average linkage degenerated at {average_degenerate} of {swept} thresholds; "
    f"it was chosen because it degenerated at none",
)
check(
    CLUSTER_EXPOSURE_LINKAGE == LINKAGE_AVERAGE,
    "the contract no longer requires average linkage",
)


# --- 3. AND IT CAN REPORT DIFFUSE ------------------------------------------------
# A gate that only ever flags is a constant, not a test.

diffuse_book = {"CAT": 0.34, "GOOGL": 0.33, "NVDA": 0.33}
if load(diffuse_book) is not None:
    diffuse = evaluate_cluster_exposure(diffuse_book, tickers, cov)
    check(
        diffuse["verdict"] == CLUSTER_DIFFUSE,
        f"an uncorrelated three-name book reported {diffuse['verdict']}, not "
        f"DIFFUSE; a gate that cannot pass is a constant",
    )


# --- 4. UNMEASURABLE IS NOT SAFE --------------------------------------------------

for weights, label in ((None, "no portfolio"), ({}, "an empty portfolio")):
    unevaluated = evaluate_cluster_exposure(weights, tickers, cov)
    check(
        unevaluated["verdict"] == CLUSTER_NOT_EVALUATED,
        f"{label} reported {unevaluated['verdict']}; reporting 0% would make the "
        f"least-understood book look the safest",
    )
    check(
        unevaluated.get("largest_cluster") is None,
        f"{label} named a cluster it could not locate",
    )

# A ZERO-VARIANCE ASSET HAS NO CORRELATION, not a correlation of zero — which
# would read as "independent" when the truth is "unmeasurable", and would then let
# a flat holding sit outside every cluster by construction.
from core.cluster_exposure import correlation_matrix  # noqa: E402

flat = correlation_matrix(["A", "B"], [[0.0, 0.0], [0.0, 1.0]])
check(
    flat == {},
    f"a zero-variance asset produced a correlation {flat}; zero reads as "
    f"independent when the truth is unmeasurable",
)

# A RAGGED COVARIANCE MUST REPORT, NOT CRASH. This branch was UNREACHABLE in the
# first version of this gate, which is how a sabotage swapping its verdict passed.
try:
    ragged = evaluate_cluster_exposure({"CAT": 1.0}, tickers, [[1.0]])
    check(
        ragged["verdict"] == CLUSTER_NOT_EVALUATED,
        f"a ragged covariance reported {ragged['verdict']}; an unmeasurable "
        f"portfolio has an UNKNOWN cluster exposure, not a diffuse one",
    )
except Exception as error:  # noqa: BLE001 - the point is that it must not raise
    failures.append(
        f"a ragged covariance raised {type(error).__name__} instead of reporting "
        f"NOT_EVALUATED; a fail-closed path that crashes is not fail-closed"
    )

no_cov = evaluate_cluster_exposure(BOOK, tickers, None)
check(
    no_cov["verdict"] == CLUSTER_NOT_EVALUATED,
    "a portfolio with no covariance reported a verdict other than NOT_EVALUATED",
)


# --- 5. A CLUSTER OF ONE IS A POSITION -------------------------------------------

check(
    CLUSTER_EXPOSURE_MIN_MEMBERS >= 2,
    "a cluster of one is a POSITION and R1 already flags it; reporting it here "
    "would double-count the same concentration under a second name",
)
solo = evaluate_cluster_exposure({"CAT": 1.0}, tickers, cov)
check(
    (solo.get("largest_cluster") or {}).get("size", 2) >= 2,
    "a single-holding book named a one-member cluster",
)


# --- 6. CONTRACT ------------------------------------------------------------------

check(
    CLUSTER_VERDICTS[0] == CLUSTER_NOT_EVALUATED,
    "NOT_EVALUATED is no longer the weakest verdict",
)
check(not CLUSTER_EXPOSURE_BLOCKS_TRADES, "B4 reports; the risk gate blocks")
check(
    bool(cluster_problems({"verdict": CLUSTER_CONCENTRATED, "reason": "trust me"})),
    "a CONCENTRATED report naming no cluster was accepted",
)
check(
    bool(
        cluster_problems(
            {
                "verdict": CLUSTER_NOT_EVALUATED,
                "reason": "r",
                "largest_cluster": {"size": 2, "risk_share": 0.5},
            }
        )
    ),
    "an unevaluated report carrying a located cluster was accepted",
)
lines = render_cluster_exposure(report)
check(
    bool(lines) and all(isinstance(line, str) for line in lines),
    "render_cluster_exposure returned no lines",
)


if failures:
    print("B4 CLUSTER EXPOSURE GATE: FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("B4 cluster exposure gate: OK")
print(f"  sessions aligned                  {len(matrix[0])}")
print(f"  worst single position             "
      f"{max(contributions.values()):.1%}  (R1 level {EXPOSURE_RISK_REVIEW:.0%}) -> no flag")
print(f"  the four semis TOGETHER           {semi_share:.1%} of variance"
      f"  on 40.0% of weight")
print(f"  their mean pairwise correlation   {correlation:.2f}")
print(f"  verdict                           {report['verdict']}")
print(f"  single linkage degenerated at     {degenerate} of {swept} thresholds")
print(f"  average linkage degenerated at    {average_degenerate} of {swept}")
sys.exit(0)
