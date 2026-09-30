"""CI drift gate for the F4 conditional forecast.

Proves, on every push, that the guarantees F5+ and the dashboard build on hold.

The load-bearing claim of F4 is that method selection is PER CELL. A future
edit that "simplifies" it into one global threshold would look tidier and
would silently destroy either reliability or accuracy, so this gate asserts
the selection actually DISCRIMINATES — it is not enough that it runs.

1.  THE SHAPE RULE — a cell carries a `value` key IF AND ONLY IF its claim is
    POINT. `value: None` would coalesce to 0.0 under the dashboard's existing
    `Number(x ?? 0)` idiom and render a withheld probability as CERTAIN DOWN;
2.  a refused cell supplies NO uncertainty and NO direction either;
3.  the selection DISCRIMINATES: the five MEASURED regime slices must not all
    land in one tier. A global rule passes every other check in this file;
4.  tier floors are ordered POINT > DIRECTIONAL > INTERVAL. The middle term is
    counter-intuitive (a weaker claim demanding more evidence) and is exactly
    what a well-meaning refactor would "fix";
5.  width CANNOT stand alone. MEASURED: a unanimous 5/5 gives Wilson width
    0.434, NARROWER than a 9-observation cell at 0.525, so a width-only policy
    admits the thinnest samples it exists to exclude;
6.  Wilson, not M6. `prediction_interval` returns width 0.00 at every N on
    identical observations — it measures fold spread, not sample-size
    uncertainty. Reusing it here would publish perfect certainty from N=2;
7.  Wilson keeps its coverage promise at small N, which is what licenses the
    INTERVAL tier to run down to 8;
8.  an EMPTY slice is distinguishable from a THIN one — different facts, with
    different fixes (longer window vs wait for data);
9.  cell precedence is total, ordered, and OK is LAST;
10. DIRECTIONAL claims carry their multiplicity exposure. 180 cells from one
    history yields ~9 spurious findings at a 5% false-signal rate;
11. with no observations the grid is honest: every cell UNAVAILABLE, nothing
    emitted, every refusal explained.

Synthetic and deterministic — no network, no wall clock.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.calibration import prediction_interval  # noqa: E402
from core.config import (  # noqa: E402
    COND_STATUS_EMPTY_SLICE,
    COND_STATUS_INSUFFICIENT,
    COND_STATUS_OK,
    COND_STATUS_UNAVAILABLE,
    CONDITIONAL_CELL_PRECEDENCE,
    CONDITIONAL_CLAIM_DIRECTIONAL,
    CONDITIONAL_CLAIM_INSUFFICIENT,
    CONDITIONAL_CLAIM_POINT,
    CONDITIONAL_CLAIM_PRECEDENCE,
    CONDITIONAL_MAX_INTERVAL_WIDTH,
    CONDITIONAL_MIN_SAMPLES_DIRECTIONAL,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    CONDITIONAL_MIN_SAMPLES_POINT,
    REGIME_LABELS,
)
from core.forecast_conditional import (  # noqa: E402
    build_cell,
    build_conditional_forecast,
    conditional_problems,
    excludes,
    interval_width,
    select_claim,
    wilson_interval,
)

# The MEASURED regime slices — 5y SPY, 191 PIT-correct sessions, P(+5% in 20D).
# These are observations, not fixtures: they are what the real data holds.
MEASURED_SLICES = (
    ("bullish", 140, 12),
    ("risk_off", 32, 7),
    ("range", 9, 3),
    ("bearish", 5, 1),
    ("stress", 2, 2),
)
MEASURED_BASE_RATE = 12 / 140


def _observations(counts: dict[str, tuple[int, int]]) -> list[dict]:
    """Build an observation set with a given (trials, successes) per regime."""
    observations: list[dict] = []
    for regime, (trials, successes) in counts.items():
        for index in range(trials):
            observations.append(
                {
                    "as_of": f"2024-01-{(index % 28) + 1:02d}",
                    "conditions": {"market_regime": regime},
                    "outcomes": {
                        "probability_up": 1.0 if index < successes else 0.0,
                    },
                }
            )
    return observations


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 3
    # The selection must DISCRIMINATE. This is the check a global rule fails
    # and every other check in this file would still pass.
    tiers = {}
    for name, trials, successes in MEASURED_SLICES:
        claim, _ = select_claim(
            trials, wilson_interval(successes, trials), MEASURED_BASE_RATE
        )
        tiers[name] = claim

    distinct = set(tiers.values())
    if len(distinct) < 2:
        failures.append(
            f"every measured regime landed in the SAME tier ({distinct.pop()}) — "
            f"the per-cell selection has collapsed into a global rule, which "
            f"destroys either reliability or accuracy depending on which way"
        )
    if tiers.get("bullish") != CONDITIONAL_CLAIM_POINT:
        failures.append(
            f"the 140-observation bullish slice no longer earns a POINT claim "
            f"(got {tiers.get('bullish')}) — the grid has gone silent on its "
            f"best-evidenced cell, which is a loss of ACCURACY"
        )
    if tiers.get("stress") != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append(
            f"the 2-observation stress slice earned a {tiers.get('stress')} claim — "
            f"it reads P(+5%)=1.00 and would publish 'rose 100% of the time' "
            f"from two observations, which is a loss of RELIABILITY"
        )
    if tiers.get("bearish") != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append(
            f"the 5-observation bearish slice earned a {tiers.get('bearish')} claim, "
            f"below the weakest floor of {CONDITIONAL_MIN_SAMPLES_INTERVAL}"
        )

    # ---------------------------------------------------------------- 4
    if not (
        CONDITIONAL_MIN_SAMPLES_POINT
        > CONDITIONAL_MIN_SAMPLES_DIRECTIONAL
        > CONDITIONAL_MIN_SAMPLES_INTERVAL
    ):
        failures.append(
            "the tier floors are no longer POINT > DIRECTIONAL > INTERVAL — a "
            "DIRECTIONAL claim says LESS than an interval but needs MORE "
            "evidence, because a directional claim that failed to detect an "
            "effect looks identical to one that found none"
        )

    # A directional claim must be refused below its floor even when the
    # interval cleanly excludes the base rate. This is the specific case a
    # "wording strength" re-sort would break.
    tight_small = wilson_interval(10, 10)
    claim, _ = select_claim(10, tight_small, 0.05)
    if claim == CONDITIONAL_CLAIM_DIRECTIONAL:
        failures.append(
            "a DIRECTIONAL claim was made from 10 observations — a large 15pt "
            "effect is detected only 24% of the time at N=10, so this is a coin "
            "flip presented as a finding"
        )

    # ---------------------------------------------------------------- 5
    # Width cannot stand alone: the unanimous-5 inversion, MEASURED.
    unanimous_five = interval_width(wilson_interval(5, 5))
    well_sampled_nine = interval_width(wilson_interval(3, 9))
    if unanimous_five >= well_sampled_nine:
        failures.append(
            "the unanimous-5/5 fixture no longer produces a NARROWER interval "
            "than the 9-observation cell — the measurement that forbids a "
            "width-only policy no longer reproduces"
        )
    claim, _ = select_claim(5, wilson_interval(5, 5), 0.20)
    if claim != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append(
            f"a unanimous 5/5 earned a {claim} claim on the strength of its "
            f"narrow ({unanimous_five:.3f}) interval — width rewards unanimity, "
            f"and unanimity is what tiny samples manufacture"
        )

    # ---------------------------------------------------------------- 6
    degenerate = prediction_interval([0.55, 0.55, 0.55])
    if degenerate["upper"] - degenerate["lower"] != 0.0:
        failures.append(
            "the M6 degenerate fixture no longer returns zero width — the "
            "measurement that rules it out of F4 no longer reproduces"
        )
    tiny = wilson_interval(2, 2)
    if interval_width(tiny) <= CONDITIONAL_MAX_INTERVAL_WIDTH:
        failures.append(
            f"the N=2 Wilson interval is only {interval_width(tiny):.3f} wide — it "
            f"must exceed the {CONDITIONAL_MAX_INTERVAL_WIDTH} cap, or the width "
            f"guard stops excluding the thinnest slice in the grid"
        )

    # ---------------------------------------------------------------- 7
    # Coverage is what licenses the INTERVAL tier to run down to 8.
    rng = random.Random(20260920)
    for true_rate in (0.2, 0.5):
        covered = 0
        draws = 2000
        for _ in range(draws):
            trials = CONDITIONAL_MIN_SAMPLES_INTERVAL
            successes = sum(1 for _ in range(trials) if rng.random() < true_rate)
            band = wilson_interval(successes, trials)
            if band["lower"] <= true_rate <= band["upper"]:
                covered += 1
        coverage = covered / draws
        if coverage < 0.88:
            failures.append(
                f"Wilson coverage at the INTERVAL floor (N="
                f"{CONDITIONAL_MIN_SAMPLES_INTERVAL}, p={true_rate}) fell to "
                f"{coverage:.3f} — the tier claims 95% and must not drift far "
                f"below it, or the interval is not honest"
            )

    if wilson_interval(0, 0) is not None:
        failures.append(
            "an interval was returned over ZERO observations — [0,1] is the "
            "absence of an answer, and returning it lets an empty slice "
            "masquerade as a very uncertain finding"
        )

    # ---------------------------------------------------------------- 8
    # EMPTY is not THIN.
    empty_cell = build_cell(
        "probability_up", "20d", "stress",
        observations=_observations({"bullish": (50, 20)}),
        base_rate=0.4,
    )
    if empty_cell["status"] != COND_STATUS_EMPTY_SLICE:
        failures.append(
            f"a regime that never occurred reported {empty_cell['status']} rather "
            f"than EMPTY_SLICE — 'never happened' and 'happened 3 times' have "
            f"different fixes and must stay distinguishable"
        )
    thin_cell = build_cell(
        "probability_up", "20d", "stress",
        observations=_observations({"bullish": (50, 20), "stress": (2, 2)}),
        base_rate=0.4,
    )
    if thin_cell["status"] != COND_STATUS_INSUFFICIENT:
        failures.append(
            f"a 2-observation slice reported {thin_cell['status']} rather than "
            f"INSUFFICIENT"
        )
    if "value" in thin_cell:
        failures.append(
            "the 2-observation stress cell carries a value key — it would "
            "coalesce to 0 in the dashboard and render as a real reading"
        )
    if thin_cell.get("samples") != 2:
        failures.append("a refused cell does not report how thin it was")

    # ---------------------------------------------------------------- 9
    if CONDITIONAL_CELL_PRECEDENCE[-1] != COND_STATUS_OK:
        failures.append("OK is not the LAST cell-precedence entry")
    if len(set(CONDITIONAL_CELL_PRECEDENCE)) != len(CONDITIONAL_CELL_PRECEDENCE):
        failures.append("the cell precedence contains a duplicate")
    if CONDITIONAL_CLAIM_PRECEDENCE[-1] != CONDITIONAL_CLAIM_POINT:
        failures.append("POINT is not the LAST claim-precedence entry")
    if CONDITIONAL_CLAIM_PRECEDENCE[0] != CONDITIONAL_CLAIM_INSUFFICIENT:
        failures.append("INSUFFICIENT is not the FIRST claim-precedence entry")

    # ---------------------------------------------------------------- 1, 2, 10
    # A full grid over the measured distribution.
    counts = {name: (trials, successes) for name, trials, successes in MEASURED_SLICES}
    forecast = build_conditional_forecast(
        "SPY",
        "2026-06-15",
        observations=_observations(counts),
        targets=("probability_up",),
        horizons=("20d",),
        base_rates={"probability_up": MEASURED_BASE_RATE},
    )

    for problem in conditional_problems(forecast):
        failures.append(f"contract problem: {problem}")

    row = forecast["rows"]["20d"]["cells"]["probability_up"]
    for regime, cell in row.items():
        claim = cell["claim"]
        if claim == CONDITIONAL_CLAIM_POINT:
            if "value" not in cell:
                failures.append(f"{regime}: a POINT claim carries no value")
        elif "value" in cell:
            failures.append(
                f"{regime}: a {claim} claim carries a value key — it would "
                f"coalesce to 0 in a consumer and render as a real reading"
            )
        if cell["status"] != COND_STATUS_OK:
            if cell.get("interval") is not None:
                failures.append(f"{regime}: a refused cell supplies uncertainty")
            if cell.get("direction") is not None:
                failures.append(f"{regime}: a refused cell states a direction")
            if not cell.get("reason"):
                failures.append(f"{regime}: a refused cell does not explain itself")
        if claim == CONDITIONAL_CLAIM_DIRECTIONAL and not cell.get("comparisons"):
            failures.append(
                f"{regime}: a DIRECTIONAL claim carries no multiplicity exposure, "
                f"so a reader cannot weigh it against the cells it came from"
            )

    # Every regime must appear in the counts, including the ones with zero.
    for label in REGIME_LABELS:
        if label not in forecast["condition_counts"]:
            failures.append(
                f"regime {label!r} is absent from condition_counts — a missing row "
                f"reads as an oversight where a zero reads as a fact"
            )

    # ---------------------------------------------------------------- 11
    empty = build_conditional_forecast(
        "SPY", "2026-06-15", observations=None,
        targets=("probability_up",), horizons=("20d",),
    )
    if empty["emitted_points"] or empty["emitted_intervals"] or empty["emitted_directions"]:
        failures.append("a grid with no observations emitted something")
    for regime, cell in empty["rows"]["20d"]["cells"]["probability_up"].items():
        if cell["status"] != COND_STATUS_UNAVAILABLE:
            failures.append(
                f"{regime}: with no observations the cell reads {cell['status']} "
                f"rather than UNAVAILABLE"
            )
        if "value" in cell:
            failures.append(f"{regime}: an observation-less cell carries a value")
    for problem in conditional_problems(empty):
        failures.append(f"empty-grid contract problem: {problem}")

    # The directional test itself: excludes() must be strict, not numeric-diff.
    if excludes(wilson_interval(7, 32), 0.219):
        failures.append(
            "an interval was judged to EXCLUDE a base rate that lies inside it — "
            "every finite sample differs numerically, and the test must ask "
            "whether the difference is distinguishable"
        )
    if not excludes(wilson_interval(7, 32), MEASURED_BASE_RATE):
        failures.append(
            "the measured risk_off slice no longer separates from the "
            "unconditional rate — the DIRECTIONAL tier can no longer fire at all"
        )

    if failures:
        print("F4 conditional-forecast gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F4 conditional-forecast gate OK:")
    print(f"  selection DISCRIMINATES across the measured slices: {tiers}.")
    print(
        f"  floors POINT>={CONDITIONAL_MIN_SAMPLES_POINT} > "
        f"DIRECTIONAL>={CONDITIONAL_MIN_SAMPLES_DIRECTIONAL} > "
        f"INTERVAL>={CONDITIONAL_MIN_SAMPLES_INTERVAL} (the inversion is intentional)."
    )
    print(
        f"  a value key exists IFF the claim is POINT; refused cells supply no "
        f"interval and no direction."
    )
    print(
        f"  width cannot stand alone: unanimous 5/5 is {unanimous_five:.3f} wide vs "
        f"{well_sampled_nine:.3f} for a 9-observation cell."
    )
    print("  Wilson keeps its coverage promise at the INTERVAL floor; M6 stays out.")
    print("  EMPTY_SLICE stays distinct from INSUFFICIENT; DIRECTIONAL carries its N.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
