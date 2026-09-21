"""Gate: position exposure is measured on BOTH scales, and weight is not exposure.

Sprint R asks "does acting on this forecast improve the CURRENT portfolio?"
R1 is the foundation: what is held, and what holding it exposes you to.

THERE WAS NO PORTFOLIO STATE. PORTFOLIO_TICKERS is a 77-name WATCHLIST with no
share counts, no cost basis and no weights, so the Sprint R question could not
previously be asked.

THE DECIDING MEASUREMENT: POSITION WEIGHT IS NOT EXPOSURE. On real returns, a
portfolio of 10% each in NVDA/AMD/AVGO/SOXX looks four times more diversified
than one holding 40% NVDA outright and is slightly MORE volatile (1.760% vs
1.741%), carrying 56.9% of its variance in the same bet against 58.6%.

A WEIGHT CAP IS GAMEABLE IN THE WRONG DIRECTION: satisfying a 10% cap by
splitting 40% NVDA across four correlated semiconductors RAISED volatility.

AND 100% VOO is the LEAST volatile portfolio tested (1.006%) while scoring
worst on every weight-based concentration measure.

Verified to FAIL when any of these is reinjected:
  - risk contribution dropped, leaving a weight-only report
  - returns aligned by position instead of by date
  - the session floor lowered below what supports a covariance
  - a watchlist entry (no quantity) accepted as a holding
  - a missing return history scored as zero risk
  - exposure wired to block trades
  - risk contributions that do not sum to the whole
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    EXPOSURE_ALIGN_BY_DATE,
    EXPOSURE_BASES,
    EXPOSURE_BASIS_RISK,
    EXPOSURE_BASIS_WEIGHT,
    EXPOSURE_BLOCKS_TRADES,
    EXPOSURE_MIN_SESSIONS,
    EXPOSURE_RISK_REVIEW,
    POSITION_REQUIRED_FIELDS,
)
from core.position_exposure import (  # noqa: E402
    PositionExposureError,
    aligned_returns,
    covariance,
    exposure_problems,
    exposure_report,
    load_positions,
    marginal_exposure,
    portfolio_variance,
    risk_contributions,
    weights,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def synthetic(sessions=400):
    """Two correlated 'semis' and two independent names, by construction."""
    import random

    rng = random.Random(7)
    dates = [f"2025-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}-{i}" for i in range(sessions)]
    market = [rng.gauss(0, 0.010) for _ in range(sessions)]
    series = {
        "SEMI_A": {},
        "SEMI_B": {},
        "INDY_A": {},
        "INDY_B": {},
    }
    for index, date in enumerate(dates):
        shock = rng.gauss(0, 0.015)
        series["SEMI_A"][date] = market[index] + shock + rng.gauss(0, 0.004)
        series["SEMI_B"][date] = market[index] + shock + rng.gauss(0, 0.004)
        series["INDY_A"][date] = market[index] + rng.gauss(0, 0.008)
        series["INDY_B"][date] = market[index] + rng.gauss(0, 0.008)
    return series


def positions(weight_map, prices):
    return [
        {"ticker": t, "quantity": (w * 1_000_000) / prices[t], "as_of": "2026-09-21"}
        for t, w in weight_map.items()
    ]


def main() -> int:
    series = synthetic()
    prices = {t: 100.0 for t in series}

    # 1. A HOLDING IS NOT A WATCHLIST ENTRY.
    check(
        "quantity" in POSITION_REQUIRED_FIELDS and "as_of" in POSITION_REQUIRED_FIELDS,
        "a position does not require a quantity and an as_of — that is a "
        "watchlist entry, which is exactly what PORTFOLIO_TICKERS already was",
    )
    for bad, label in (
        ({"ticker": "AAPL", "as_of": "2026-09-21"}, "no quantity"),
        ({"ticker": "AAPL", "quantity": 10}, "no as_of"),
        ({"quantity": 10, "as_of": "2026-09-21"}, "no ticker"),
    ):
        try:
            load_positions([bad])
            FAILURES.append(f"a position with {label} was accepted as a holding")
        except PositionExposureError:
            pass

    # 2. THE DECIDING MEASUREMENT: weight is not exposure.
    concentrated = {"SEMI_A": 0.40, "INDY_A": 0.30, "INDY_B": 0.30}
    spread = {"SEMI_A": 0.20, "SEMI_B": 0.20, "INDY_A": 0.30, "INDY_B": 0.30}

    report_a = exposure_report(positions(concentrated, prices), series, prices)
    report_b = exposure_report(positions(spread, prices), series, prices)
    check(
        report_a["volatility"] is not None and report_b["volatility"] is not None,
        "a portfolio with full return history produced no volatility",
    )
    # Halving the largest weight must NOT be assumed to halve the risk.
    semi_risk_a = report_a["risk_contributions"]["SEMI_A"]
    semi_risk_b = (
        report_b["risk_contributions"]["SEMI_A"]
        + report_b["risk_contributions"]["SEMI_B"]
    )
    check(
        semi_risk_b > semi_risk_a * 0.75,
        f"splitting a 40% holding across two correlated names cut its variance "
        f"share from {semi_risk_a:.1%} to {semi_risk_b:.1%} — the measurement "
        f"showing weight is not exposure no longer reproduces",
    )
    check(
        max(report_b["weights"].values()) < max(report_a["weights"].values()),
        "the spread portfolio does not have a lower maximum weight, so the "
        "scenario under test is not set up",
    )

    # 3. BOTH SCALES ARE REPORTED, ALWAYS.
    check(
        EXPOSURE_BASIS_RISK in EXPOSURE_BASES and EXPOSURE_BASIS_WEIGHT in EXPOSURE_BASES,
        "exposure is not reported on both weight and risk",
    )
    check(
        report_a["risk_contributions"] and report_a["weights"],
        "a report carried only one scale",
    )
    total = sum(report_a["risk_contributions"].values())
    check(
        abs(total - 1.0) < 0.05,
        f"risk contributions sum to {total:.4f}, not 1 — they are shares of "
        f"variance and must account for all of it",
    )
    check(
        exposure_problems(report_a) == [],
        f"a report is not contract-clean: {exposure_problems(report_a)}",
    )

    # 4. RISK IS FLAGGED ON ITS OWN SCALE, not inferred from weight.
    flagged = [f for f in report_a["flags"] if f["basis"] == EXPOSURE_BASIS_RISK]
    check(
        any(f["ticker"] == "SEMI_A" for f in flagged),
        f"a holding carrying {semi_risk_a:.1%} of portfolio variance was not "
        f"flagged on the risk scale (threshold {EXPOSURE_RISK_REVIEW:.0%})",
    )
    for flag in report_a["flags"]:
        check(
            bool(str(flag.get("reason") or "").strip()),
            f"{flag.get('ticker')}: a flag carried no reason",
        )

    # 5. ALIGNMENT BY DATE. MEASURED, a position-offset join reported MSFT's
    #    correlation with everything as ~0.00 — including 0.00 against VOO,
    #    which is really 0.53 — because its series ended three days earlier.
    check(EXPOSURE_ALIGN_BY_DATE, "returns are not aligned by date")
    offset = {
        "SEMI_A": series["SEMI_A"],
        "SEMI_B": {k: v for k, v in list(series["SEMI_B"].items())[:-3]},
    }
    tickers, matrix = aligned_returns(offset)
    cov = covariance(matrix)
    correlation = cov[0][1] / math.sqrt(cov[0][0] * cov[1][1])
    check(
        correlation > 0.5,
        f"two series built from the SAME shock correlated at {correlation:.2f} "
        f"after alignment — the join is matching by position, which silently "
        f"reports independence rather than failing",
    )

    # 6. THE SESSION FLOOR.
    check(
        EXPOSURE_MIN_SESSIONS >= 120,
        f"{EXPOSURE_MIN_SESSIONS} sessions cannot support a covariance",
    )
    thin = {t: dict(list(s.items())[:20]) for t, s in series.items()}
    try:
        aligned_returns(thin)
        FAILURES.append(
            "a 20-session history produced a covariance — the result would be "
            "a confident-looking number for a relationship never observed"
        )
    except PositionExposureError:
        pass

    # 7. A MISSING HISTORY IS NOT ZERO RISK.
    partial = exposure_report(
        positions({"SEMI_A": 0.5, "UNKNOWN": 0.5}, {"SEMI_A": 100.0, "UNKNOWN": 100.0}),
        series,
        {"SEMI_A": 100.0, "UNKNOWN": 100.0},
    )
    check(
        partial["risk_basis"] == "NOT_EVALUATED",
        "a holding with no return history was given a risk figure anyway — "
        "scored as zero it would render the least-understood position as the "
        "safest",
    )
    check(
        not partial["risk_contributions"],
        "risk contributions were reported for a portfolio that could not be "
        "evaluated",
    )
    check(
        bool(str(partial.get("risk_reason") or "").strip()),
        "an unevaluated risk basis carried no reason",
    )
    check(
        exposure_problems(partial) == [],
        f"a partial report is not contract-clean: {exposure_problems(partial)}",
    )

    # 8. MARGINAL EXPOSURE — the question Sprint R asks.
    tickers, matrix = aligned_returns(series)
    cov = covariance(matrix)
    base = {"SEMI_A": 0.5, "INDY_A": 0.25, "INDY_B": 0.25}
    adding_correlated = marginal_exposure(base, tickers, cov, "SEMI_B", 0.05)
    adding_diversifier = marginal_exposure(base, tickers, cov, "INDY_B", 0.05)
    check(
        adding_correlated["change"] > adding_diversifier["change"],
        f"adding a correlated name ({adding_correlated['change']:+.6f}) did "
        f"not raise risk more than adding a diversifier "
        f"({adding_diversifier['change']:+.6f}) — the same trade must help or "
        f"hurt depending on what is already held",
    )
    check(
        adding_correlated["direction"] in ("ADDS_RISK", "REDUCES_RISK", "NEUTRAL"),
        "a marginal exposure carried no direction",
    )
    try:
        marginal_exposure(base, tickers, cov, "NOT_HELD_XYZ", 0.05)
        FAILURES.append(
            "a ticker with no return history was given a marginal exposure"
        )
    except PositionExposureError:
        pass

    # 9. R1 DESCRIBES, IT DOES NOT ENFORCE.
    check(
        EXPOSURE_BLOCKS_TRADES is False,
        "exposure is wired to block trades — a weight cap enforced alone is "
        "gameable in the wrong direction",
    )
    check(
        report_a.get("blocks_trades") is False,
        "a report claimed to block trades",
    )

    if FAILURES:
        print("R1 position-exposure gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("R1 position-exposure gate OK:")
    print("  a holding needs a quantity and an as_of; a watchlist entry is refused.")
    print(f"  weight is not exposure: splitting one 40% holding across two "
          f"correlated names moved its variance share {semi_risk_a:.1%} -> "
          f"{semi_risk_b:.1%}.")
    print("  both scales always reported; risk contributions sum to the whole.")
    print(f"  returns aligned by DATE (offset series still correlate "
          f"{correlation:.2f}); floor {EXPOSURE_MIN_SESSIONS} sessions.")
    print("  a missing history is NOT_EVALUATED, never zero risk.")
    print("  marginal exposure: the same 5% helps or hurts by what is held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
