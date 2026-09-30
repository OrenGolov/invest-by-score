"""Gate: source reliability is LEARNED, and conditioned only where earned.

"Source quality may depend on source x event type x sector x horizon - one
global source score is not assumed sufficient."

The task states a hypothesis. This gate holds the system to testing it rather
than implementing it on faith.

THE DECIDING MEASUREMENT: CONDITIONING IS NOT FREE. Against a source whose
TRUE accuracy really does vary by cell, per-cell estimation is FIVE TIMES
WORSE than one global rate at 5 observations per cell (MSE 0.05935 vs
0.01224), and only starts to pay from ~25. So the shipped estimator is
shrinkage, which IS the global score when a cell is empty and becomes the
cell's own score as evidence arrives.

Verified to FAIL when any of these is reinjected:
  - shrinkage removed (k=0), restoring raw per-cell estimation
  - k raised until every cell is pinned to the global rate
  - the cell floor dropped, so a shrinkage-dominated number is reported as a
    learned source property
  - an unmeasured source scored 0.0 instead of reported as unmeasured
  - the four backing states collapsed
  - the conditioning noise band flattened, manufacturing "quality depends on
    event_type" from noise
  - a CONDITIONAL score that cannot say what it is conditioned on
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    SOURCE_BACKING_CELL,
    SOURCE_BACKING_GLOBAL,
    SOURCE_BACKING_NONE,
    SOURCE_BACKING_PRIOR,
    SOURCE_RELIABILITY_ABSENT_IS_ZERO,
    SOURCE_RELIABILITY_BACKINGS,
    SOURCE_RELIABILITY_MIN_CELL,
    SOURCE_RELIABILITY_SHRINKAGE_K,
)
from core.source_reliability import (  # noqa: E402
    conditioning_gain,
    reliability_problems,
    reliability_report,
    score_source,
    shrunk_rate,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def observations(source, rate, count, seed, **context):
    rng = random.Random(seed)
    return [
        {"source": source, "hit": 1 if rng.random() < rate else 0, **context}
        for _ in range(count)
    ]


def shrinkage_beats_per_cell() -> tuple[float, float, float]:
    """MSE of global / per-cell / shrunk when quality REALLY varies, thin data."""
    cells = 20
    rng = random.Random(7)
    global_errors, cell_errors, shrunk_errors = [], [], []
    for _ in range(200):
        truth = {
            index: min(0.95, max(0.05, 0.60 + rng.gauss(0, 0.10)))
            for index in range(cells)
        }
        seen = defaultdict(list)
        for _draw in range(100):  # 5 per cell
            index = rng.randrange(cells)
            seen[index].append(1 if rng.random() < truth[index] else 0)
        flat = [value for values in seen.values() for value in values]
        overall = sum(flat) / len(flat) if flat else 0.6
        g = c = s = 0.0
        for index in range(cells):
            values = seen.get(index) or []
            g += (overall - truth[index]) ** 2
            raw = sum(values) / len(values) if values else overall
            c += (raw - truth[index]) ** 2
            s += (
                shrunk_rate(sum(values), len(values), overall, 20) - truth[index]
            ) ** 2
        global_errors.append(g / cells)
        cell_errors.append(c / cells)
        shrunk_errors.append(s / cells)
    return (
        statistics.mean(global_errors),
        statistics.mean(cell_errors),
        statistics.mean(shrunk_errors),
    )


def main() -> int:
    # 1. THE DECIDING MEASUREMENT stands: per-cell loses badly when thin.
    global_mse, cell_mse, shrunk_mse = shrinkage_beats_per_cell()
    check(
        cell_mse > global_mse,
        f"per-cell estimation ({cell_mse:.5f}) did not lose to a global rate "
        f"({global_mse:.5f}) at 5 observations per cell — the measurement "
        f"justifying shrinkage no longer holds",
    )
    check(
        shrunk_mse <= global_mse,
        f"shrinkage ({shrunk_mse:.5f}) is worse than a flat global rate "
        f"({global_mse:.5f}) even where quality genuinely varies",
    )
    check(
        shrunk_mse < cell_mse,
        f"shrinkage ({shrunk_mse:.5f}) is worse than raw per-cell estimation "
        f"({cell_mse:.5f})",
    )

    # 2. SHRINKAGE IS REAL. k=0 is raw per-cell estimation.
    check(
        SOURCE_RELIABILITY_SHRINKAGE_K >= 1,
        "shrinkage k below 1 is per-cell estimation, measured five times worse "
        "than a global rate on thin evidence",
    )
    check(
        abs(shrunk_rate(0, 0, 0.75, SOURCE_RELIABILITY_SHRINKAGE_K) - 0.75) < 1e-9,
        "an empty cell did not fall back to the prior — shrinkage is not "
        "degrading to the global score",
    )
    far = shrunk_rate(900, 1000, 0.50, SOURCE_RELIABILITY_SHRINKAGE_K)
    check(
        far > 0.85,
        f"1000 observations at a 0.90 rate shrank to {far:.3f}: k is so large "
        f"that abundant evidence cannot move a cell, which is the single "
        f"global score the task refuses to assume is sufficient",
    )

    # 3. ABSENCE IS NOT ZERO.
    check(
        SOURCE_RELIABILITY_ABSENT_IS_ZERO is False,
        "an unmeasured source must not be scored 0.0 — the same number as one "
        "measured to be worthless",
    )
    unmeasured = score_source([], "Brand New Outlet")
    check(
        unmeasured.get("score") is None,
        f"an unmeasured source carried a score of {unmeasured.get('score')!r} "
        f"— the shape rule: a score key exists IFF it was measured, because "
        f"Number(x ?? 0) renders a missing score as 0.0",
    )
    check(
        unmeasured.get("backing") == SOURCE_BACKING_NONE,
        "an unmeasured source was not reported as NO_EVIDENCE",
    )
    prior_backed = score_source([], "Known Outlet", registry_prior=0.75)
    check(
        prior_backed.get("backing") == SOURCE_BACKING_PRIOR
        and abs(float(prior_backed.get("score") or 0) - 0.75) < 1e-9,
        "a source with only a registry prior was not reported as "
        "REGISTRY_PRIOR carrying that prior",
    )

    # 4. THE FOUR BACKING STATES ARE DISTINCT.
    check(
        len(set(SOURCE_RELIABILITY_BACKINGS)) == 4,
        "the backing states collapsed — an asserted prior, a measured global "
        "rate and a conditional cell are different facts",
    )

    # 5. A CONDITIONAL SCORE IS EARNED, AND SAYS WHAT IT CONDITIONS ON.
    thin = observations("Thin", 0.7, 10, 1, event_type="earnings")
    thin_score = score_source(thin, "Thin", event_type="earnings")
    check(
        thin_score.get("backing") == SOURCE_BACKING_GLOBAL,
        f"{len(thin)} observations were reported as {thin_score.get('backing')} "
        f"— below the {SOURCE_RELIABILITY_MIN_CELL} floor a shrinkage-dominated "
        f"number must not be read as a learned source property",
    )
    thick = observations("Thick", 0.7, 60, 2, event_type="earnings")
    thick_score = score_source(thick, "Thick", event_type="earnings")
    check(
        thick_score.get("backing") == SOURCE_BACKING_CELL,
        "60 observations in one cell were not reported as CONDITIONAL, so no "
        "amount of evidence can ever earn a conditional score",
    )
    check(
        bool(thick_score.get("cell")),
        "a CONDITIONAL score named no cell — it cannot say what it is "
        "conditioned on",
    )

    # 6. CONDITIONING IS NOT CLAIMED FROM NOISE.
    #    MEASURED with a flat noise band: 7 of 30 clean runs claimed
    #    "quality depends on event_type" with no effect present at all.
    false_claims = 0
    for seed in range(40):
        rng = random.Random(900 + seed)
        clean: list[dict] = []
        for event_type in ("earnings", "regulation", "product_launch"):
            clean.extend(
                {
                    "source": "X",
                    "hit": 1 if rng.random() < 0.6 else 0,
                    "event_type": event_type,
                }
                for _ in range(50)
            )
        if conditioning_gain(clean, "event_type")["explains"]:
            false_claims += 1
    check(
        false_claims <= 6,
        f"{false_claims}/40 clean runs claimed source quality depends on "
        f"event_type with NO effect present — the noise band is tracking the "
        f"mean of the null range instead of its tail",
    )

    # And a REAL dependence is still found.
    rng = random.Random(4242)
    real: list[dict] = []
    for event_type, rate in (
        ("earnings", 0.75),
        ("regulation", 0.45),
        ("product_launch", 0.60),
    ):
        real.extend(
            {
                "source": "X",
                "hit": 1 if rng.random() < rate else 0,
                "event_type": event_type,
            }
            for _ in range(300)
        )
    check(
        conditioning_gain(real, "event_type")["explains"],
        "a real 0.30 spread across event types was not detected — the band is "
        "a gag rather than a filter",
    )

    # 7. THE REPORT IS CONTRACT-CLEAN AND NAMES THE JOIN GAP WHEN EMPTY.
    empty = reliability_report([])
    check(
        reliability_problems(empty) == [],
        f"an empty report is not contract-clean: {reliability_problems(empty)}",
    )
    check(
        "MEASURED" in (empty.get("join_gap") or ""),
        "an empty report did not carry the measured join gap — a reader cannot "
        "tell a missing capability from a finding",
    )
    populated = reliability_report(
        observations("A", 0.7, 60, 11, event_type="earnings")
        + observations("B", 0.5, 60, 12, event_type="earnings")
    )
    check(
        reliability_problems(populated) == [],
        f"a populated report is not contract-clean: "
        f"{reliability_problems(populated)}",
    )

    if FAILURES:
        print("L4 source-reliability gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L4 source-reliability gate OK:")
    print(
        f"  conditioning is NOT free: per-cell MSE {cell_mse:.5f} vs global "
        f"{global_mse:.5f} at 5/cell."
    )
    print(f"  shrinkage (k={SOURCE_RELIABILITY_SHRINKAGE_K}) beats both: "
          f"{shrunk_mse:.5f}.")
    print(f"  a cell earns CONDITIONAL only at {SOURCE_RELIABILITY_MIN_CELL}+ "
          f"observations, and must name its cell.")
    print(f"  {false_claims}/40 clean runs claimed conditioning from noise; a "
          f"real 0.30 spread is still found.")
    print("  an unmeasured source scores None, never 0.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
