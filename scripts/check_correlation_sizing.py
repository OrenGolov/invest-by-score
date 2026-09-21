"""Gate: sizing is correlation-aware, bounded twice, and cannot fix selection.

R1 established that weight is not exposure. R2 asks how big a position should
be, given what is already held.

EQUAL-WEIGHT SIZING IGNORES CORRELATION: MEASURED, a 40% budget in four
correlated semiconductors carries 50.5% more risk than the same 40% in four
diverse names.

THE NEGATIVE RESULT THAT SHAPES THE MODULE: reweighting INSIDE a correlated
basket barely helps. Equal-weight, inverse-volatility and equal-risk all land
within 1% of each other, while replacing a basket member with an uncorrelated
name buys 10-34%. SIZING CANNOT FIX SELECTION.

SO SIZING IS DONE AGAINST THE WHOLE PORTFOLIO: MEASURED, a fixed 10% moved
portfolio volatility +3.07% for AVGO and -9.24% for XOM.

Verified to FAIL when any of these is reinjected:
  - the risk budget removed or widened past a quarter of portfolio vol
  - the weight cap removed, so a diversifier is sized into concentration
  - the correlation ignored, so every ticker gets the same size
  - the minimum-size check dropped, proposing a trade that cannot fit
  - the binding constraint no longer reported
  - a REFUSED or NOT_EVALUATED proposal carrying a size
  - sizing declared non-advisory
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    SIZING_CAPPED,
    SIZING_IS_ADVISORY,
    SIZING_MAX_WEIGHT,
    SIZING_MIN_SESSIONS,
    SIZING_MIN_WEIGHT,
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
    SIZING_REPORT_BINDING_CONSTRAINT,
    SIZING_RISK_BUDGET,
    SIZING_SIZED,
    SIZING_VERDICTS,
)
from core.correlation_sizing import (  # noqa: E402
    BINDING_RISK_BUDGET,
    BINDING_WEIGHT_CAP,
    CorrelationSizingError,
    blended,
    compare_candidates,
    size_position,
    sizing_problems,
    sizing_report,
    volatility,
)
from core.position_exposure import aligned_returns, covariance  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def synthetic(sessions=400, seed=11):
    """Two tightly correlated names and two independent ones, by construction."""
    rng = random.Random(seed)
    dates = [f"d{i:04d}" for i in range(sessions)]
    market = [rng.gauss(0, 0.008) for _ in range(sessions)]
    series = {
        "SEMI_A": {}, "SEMI_B": {}, "INDY_A": {}, "INDY_B": {},
        "LEVERED": {}, "AAA_SEMI": {}, "ZZZ_DIVERSIFIER": {},
    }
    for index, date in enumerate(dates):
        shock = rng.gauss(0, 0.018)
        series["SEMI_A"][date] = market[index] + shock + rng.gauss(0, 0.003)
        series["SEMI_B"][date] = market[index] + shock + rng.gauss(0, 0.003)
        series["INDY_A"][date] = rng.gauss(0, 0.009)
        series["INDY_B"][date] = rng.gauss(0, 0.009)
        # A 3x levered version of SEMI_A: near-perfectly correlated with it AND
        # three times as volatile. Adding it to a SEMI_A portfolio can only
        # raise risk, which is what makes a genuine refusal constructible.
        series["LEVERED"][date] = 3.0 * (market[index] + shock)
        # Aliases whose NAMES disagree with their behaviour, so an
        # alphabetical sort cannot masquerade as a risk-ordered one.
        series["AAA_SEMI"][date] = series["SEMI_A"][date]
        series["ZZZ_DIVERSIFIER"][date] = series["INDY_B"][date]
    return series


def main() -> int:
    series = synthetic()
    tickers, matrix = aligned_returns(series)
    cov = covariance(matrix)
    held = {"SEMI_A": 0.60, "INDY_A": 0.40}

    # 1. CORRELATION CHANGES THE SIZE. The same risk budget must buy less of a
    #    correlated name than of a diversifier.
    correlated = size_position(held, "SEMI_B", tickers, cov)
    diversifier = size_position(held, "INDY_B", tickers, cov)
    check(
        correlated["verdict"] in (SIZING_SIZED, SIZING_CAPPED, SIZING_REFUSED),
        f"a correlated candidate produced {correlated['verdict']}",
    )
    check(
        correlated.get("size", 0) < diversifier.get("size", 0),
        f"a correlated name was sized at {correlated.get('size')} and a "
        f"diversifier at {diversifier.get('size')} — the sizer is ignoring "
        f"correlation and giving every ticker the same answer",
    )
    check(
        diversifier.get("volatility_increase", 1) < correlated.get("volatility_increase", -1),
        "adding a diversifier did not raise portfolio volatility less than "
        "adding a correlated name",
    )

    # 2. THE RISK BUDGET IS RESPECTED.
    check(
        0.0 < SIZING_RISK_BUDGET <= 0.25,
        f"a risk budget of {SIZING_RISK_BUDGET} is not a budget",
    )
    for proposal in (correlated, diversifier):
        increase = proposal.get("volatility_increase")
        if increase is not None:
            check(
                increase <= SIZING_RISK_BUDGET + 1e-6,
                f"{proposal['ticker']} was sized to a {increase:+.4f} volatility "
                f"increase, beyond the {SIZING_RISK_BUDGET} budget",
            )

    # 3. THE WEIGHT CAP IS A SECOND, INDEPENDENT BOUND. MEASURED, an
    #    uncorrelated name reached 50%+ on the risk budget alone because it
    #    REDUCES volatility — without the cap the sizer proposes concentration
    #    in the name of diversification.
    check(
        diversifier.get("size", 1.0) <= SIZING_MAX_WEIGHT + 1e-9,
        f"a diversifier was sized at {diversifier.get('size')}, above the "
        f"{SIZING_MAX_WEIGHT} cap — the risk budget alone does not bound "
        f"concentration",
    )
    check(
        diversifier.get("binding") == BINDING_WEIGHT_CAP,
        f"a volatility-reducing candidate was bound by "
        f"{diversifier.get('binding')!r} rather than the weight cap",
    )
    check(
        correlated.get("binding") == BINDING_RISK_BUDGET,
        f"a correlated candidate was bound by {correlated.get('binding')!r} "
        f"rather than the risk budget",
    )

    # 4. THE BINDING CONSTRAINT IS ALWAYS REPORTED.
    check(
        SIZING_REPORT_BINDING_CONSTRAINT,
        "the binding constraint is not reported — 'the portfolio cannot "
        "absorb more of this' and 'policy stops here' are different answers",
    )

    # 5. A POSITION THAT CANNOT FIT IS REFUSED, not shrunk silently.
    # THE SCENARIO MUST BE CONSTRUCTIBLE. A first version added a near-
    # identical twin to a 100%-single-name portfolio and expected a refusal;
    # MEASURED, that REDUCES volatility (-0.10% at 5%) because idiosyncratic
    # noise averages out, so no budget could ever be breached. A genuine
    # refusal needs a candidate that is correlated AND more volatile than what
    # is held: here a low-volatility portfolio meeting a high-volatility name.
    # A first version added a near-identical twin to a 100%-single-name
    # portfolio and expected a refusal; MEASURED, that REDUCES volatility
    # (-0.10% at 5%) because idiosyncratic noise averages out. A second tried a
    # more volatile but UNCORRELATED name, which diversifies (-5.3%). A genuine
    # refusal needs a candidate correlated with what is held AND more volatile:
    # a levered version of the holding itself.
    hostile = {"SEMI_A": 1.0}
    tight = size_position(
        hostile, "LEVERED", tickers, cov, risk_budget=0.001, min_weight=0.05
    )
    check(
        tight["verdict"] == SIZING_REFUSED,
        f"a candidate whose MINIMUM size breaches the budget produced "
        f"{tight['verdict']} instead of REFUSED",
    )
    check(
        tight.get("size") is None,
        "a REFUSED proposal carried a size — the shape rule: a size exists "
        "IFF one was found, and 0.0 reads as 'propose nothing'",
    )
    check(
        SIZING_REFUSED != SIZING_NOT_EVALUATED,
        "'no size fits' and 'no size could be computed' collapsed into one "
        "verdict",
    )

    # 6. SIZING CANNOT FIX SELECTION. Reweighting inside a correlated basket
    #    must not be presented as a remedy: MEASURED it buys ~1% while
    #    changing composition buys 10-34%.
    equal = {"SEMI_A": 0.20, "SEMI_B": 0.20, "INDY_A": 0.60}
    tilted = {"SEMI_A": 0.28, "SEMI_B": 0.12, "INDY_A": 0.60}
    swapped = {"SEMI_A": 0.20, "INDY_B": 0.20, "INDY_A": 0.60}
    reweight_gain = 1.0 - volatility(tilted, tickers, cov) / volatility(equal, tickers, cov)
    swap_gain = 1.0 - volatility(swapped, tickers, cov) / volatility(equal, tickers, cov)
    check(
        swap_gain > reweight_gain * 3,
        f"changing composition bought {swap_gain:+.3%} and reweighting bought "
        f"{reweight_gain:+.3%} — the measurement showing sizing cannot fix "
        f"selection no longer reproduces",
    )

    # 7. FUNDING IS PRO RATA, so the weights still sum to one.
    mixed = blended(held, "INDY_B", 0.10)
    total = sum(mixed.values())
    check(
        abs(total - 1.0) < 1e-9,
        f"a blended portfolio sums to {total}, not 1 — the trade is being "
        f"funded from nowhere",
    )

    # 8. AN UNMEASURABLE CANDIDATE IS REFUSED, not guessed.
    try:
        size_position(held, "NEVER_SEEN", tickers, cov)
        FAILURES.append(
            "a ticker with no return history was given a size — its "
            "correlation with the portfolio is unmeasured"
        )
    except CorrelationSizingError:
        pass
    thin = sizing_report(held, ["INDY_B"], {t: dict(list(s.items())[:20]) for t, s in series.items()})
    check(
        thin["verdict"] == SIZING_NOT_EVALUATED,
        f"a 20-session history produced {thin['verdict']} instead of refusing",
    )
    check(
        SIZING_MIN_SESSIONS >= 120,
        "the session floor is below what a covariance needs",
    )

    # 9. CANDIDATES ARE ORDERED BY PORTFOLIO EFFECT, not by name.
    # THE NAMES MUST NOT AGREE WITH THE ANSWER. A first version compared
    # INDY_B against SEMI_B, where the diversifier also sorts first
    # ALPHABETICALLY — so an alphabetical sort passed the check and the
    # ordering guarantee was untested. ZZZ_DIVERSIFIER is the same
    # uncorrelated series under a name that sorts LAST.
    ordered = compare_candidates(
        held, ["AAA_SEMI", "ZZZ_DIVERSIFIER"], tickers, cov
    )
    check(
        ordered[0]["ticker"] == "ZZZ_DIVERSIFIER",
        f"candidates ordered {[o['ticker'] for o in ordered]} — a diversifier "
        f"must rank above a correlated name even when its name sorts last, "
        f"which a per-ticker forecast ranking cannot express",
    )

    # 10. ADVISORY, AND CONTRACT-CLEAN.
    check(SIZING_IS_ADVISORY, "sizing is not declared advisory")
    report = sizing_report(held, ["SEMI_B", "INDY_B"], series)
    check(
        report.get("advisory") is True,
        "a sizing report did not declare itself advisory",
    )
    check(
        sizing_problems(report) == [],
        f"a sizing report is not contract-clean: {sizing_problems(report)}",
    )
    check(
        sizing_problems(thin) == [],
        f"an unevaluated report is not contract-clean: {sizing_problems(thin)}",
    )

    if FAILURES:
        print("R2 correlation-sizing gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("R2 correlation-sizing gate OK:")
    print(
        f"  correlation changes the size: correlated {correlated['size']:.1%} "
        f"vs diversifier {diversifier['size']:.1%} at the same budget."
    )
    print(f"  two independent bounds: risk budget {SIZING_RISK_BUDGET:.0%} and "
          f"weight cap {SIZING_MAX_WEIGHT:.0%}, and the report says which bound.")
    print("  a position whose minimum cannot fit is REFUSED, never shrunk.")
    print(f"  sizing cannot fix selection: composition buys {swap_gain:+.1%}, "
          f"reweighting {reweight_gain:+.1%}.")
    print("  candidates ordered by portfolio effect; sizes are advisory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
