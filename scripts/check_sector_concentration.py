"""Gate: sector concentration is visible where per-name caps are blind.

R1 established that weight is not exposure, and that a per-name cap is
gameable: satisfying a 10% cap by splitting 40% NVDA across four correlated
semiconductors RAISED portfolio volatility. R3 is that finding one level up,
and on the tracked portfolio it is not hypothetical.

THE DECIDING MEASUREMENT: MEASURED, 38 of 73 sector-mapped holdings are
Information Technology — 52.1% of the book — while every one of those names
sits near 1.3% at equal weight and clears R1's own 25% per-name threshold. The
name-level report is silent about a bet that is half the portfolio.

TEN SECTORS HELD IS NOT TEN SECTORS OF DIVERSIFICATION: MEASURED, the tracked
portfolio spans 10 GICS sectors and scores a sector HHI of 0.3113 — an
effective count of 3.21.

A HOLDING WITH NO SECTOR GETS NO SECTOR: MEASURED, 4 of 77 tracked holdings
are funds (VOO, SOXX, CIBR, NASA). Filing SOXX under Information Technology
would deepen an already-52% reading using an instrument that is itself
diversified.

Verified to FAIL when any of these is reinjected:
  - the sector map duplicated instead of reused, so the two can drift
  - risk dropped, leaving a weight-only report
  - a single-sector book left unflagged
  - splitting a position across its own sector reducing the sector share
  - UNCLASSIFIED counted as a sector, in the HHI or in the flags
  - an HHI of 0.0 reported when nothing could be classified
  - unmeasured per-name risk summed into a sector total
  - effective sectors exceeding the sectors actually held
  - the HHI threshold set so wide that an even book still flags
  - concentration declared to block trades
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CONTEXT_SECTOR_ETFS,
    SECTOR_BASES,
    SECTOR_BASIS_RISK,
    SECTOR_BASIS_WEIGHT,
    SECTOR_BLOCKS_TRADES,
    SECTOR_HHI_REVIEW,
    SECTOR_RISK_REVIEW,
    SECTOR_UNCLASSIFIED,
    SECTOR_UNCLASSIFIED_REVIEW,
    SECTOR_WEIGHT_REVIEW,
)
from core.market_context import sector_for  # noqa: E402
from core.sector_concentration import (  # noqa: E402
    classify,
    concentration_problems,
    concentration_report,
    effective_sectors,
    group_shares,
    herfindahl,
    render_concentration,
)

FAILURES: list[str] = []

# Real tickers whose sector the shared map knows. Using real names is the
# point: a fixture sector map would pass this gate while production drifts.
IT = ("NVDA", "AMD", "AVGO", "QCOM", "MSFT", "AAPL", "ORCL", "CSCO")
OTHER = ("LLY", "VRTX", "BKR", "CAT")
FUNDS = ("VOO", "SOXX")


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def exposure(weights, risk=None, reason=""):
    """A minimal R1-shaped exposure report."""
    report = {
        "version": "position-exposure-v1",
        "positions": len(weights),
        "weights": dict(weights),
        "bases": [SECTOR_BASIS_WEIGHT, SECTOR_BASIS_RISK],
        "blocks_trades": False,
    }
    if risk is None:
        report.update(
            {
                "risk_contributions": {},
                "risk_basis": "NOT_EVALUATED",
                "risk_reason": reason or "no return history",
            }
        )
    else:
        report.update(
            {
                "risk_contributions": dict(risk),
                "risk_basis": SECTOR_BASIS_RISK,
                "risk_reason": "",
            }
        )
    return report


def equal(tickers):
    return {t: 1.0 / len(tickers) for t in tickers}


def main() -> int:
    # 1. THE SECTOR MAP IS REUSED, NOT DUPLICATED. A second copy would
    #    disagree with market_context the first time either was updated.
    for ticker in IT + OTHER:
        check(
            classify([ticker])[ticker] == sector_for(ticker),
            f"{ticker} classified differently from core.market_context — R3 "
            f"is carrying its own sector table, which will drift",
        )
    check(
        len(CONTEXT_SECTOR_ETFS) >= 10,
        f"the shared sector map holds {len(CONTEXT_SECTOR_ETFS)} sectors; R3 "
        f"depends on it covering the GICS set",
    )

    # 2. THE DECIDING MEASUREMENT. Every per-name cap passes; the sector does
    #    not. This is the blind spot the module exists to close.
    single_sector = equal(IT)
    for ticker, weight in single_sector.items():
        check(
            weight < SECTOR_WEIGHT_REVIEW,
            f"{ticker} at {weight:.1%} breaches the per-name threshold; this "
            f"gate needs a portfolio that clears every name-level check",
        )
    report = concentration_report(exposure(single_sector))
    check(
        concentration_problems(report) == [],
        f"a well-formed report reported problems: {concentration_problems(report)}",
    )
    flagged = {f["sector"] for f in report["flags"]}
    check(
        sector_for("NVDA") in flagged,
        "a 100% single-sector book cleared every per-name cap AND went "
        "unflagged at the sector level — exactly the blind spot R3 closes",
    )

    # 3. SPLITTING ACROSS THE SECTOR DOES NOT HELP. The R1 gaming move must
    #    not be available one level up.
    concentrated = concentration_report(exposure({"NVDA": 1.0}))
    sector = sector_for("NVDA")
    check(
        abs(
            concentrated["weight_by_sector"][sector]
            - report["weight_by_sector"][sector]
        )
        < 1e-6,
        "splitting one name across eight in the same sector changed the "
        "sector share — the cap is gameable again",
    )

    # 4. BOTH SCALES ARE CARRIED. Weight says how much capital sits in a
    #    sector; risk says how much variance it drives, and they disagree.
    check(
        SECTOR_BASIS_WEIGHT in SECTOR_BASES and SECTOR_BASIS_RISK in SECTOR_BASES,
        "concentration is not reported on both weight and risk",
    )
    check(
        SECTOR_BASIS_RISK in report["bases"],
        "the report omits sector risk — weight alone cannot see that "
        "correlated names inside a sector contribute jointly",
    )
    check(
        SECTOR_RISK_REVIEW > SECTOR_WEIGHT_REVIEW,
        f"the risk threshold {SECTOR_RISK_REVIEW} does not sit above the "
        f"weight threshold {SECTOR_WEIGHT_REVIEW}",
    )

    # 5. A SECTOR ABOVE THE VARIANCE THRESHOLD IS FLAGGED ON RISK — the scale
    #    a weight cap cannot see.
    mixed = equal(IT + OTHER)
    risk = dict(mixed)
    for ticker in IT:
        risk[ticker] = SECTOR_RISK_REVIEW / len(IT) + 0.02
    total = sum(risk.values())
    risk = {t: v / total for t, v in risk.items()}
    risk_report = concentration_report(exposure(mixed, risk=risk))
    check(
        concentration_problems(risk_report) == [],
        f"a measured-risk report reported problems: "
        f"{concentration_problems(risk_report)}",
    )
    check(
        any(f["basis"] == SECTOR_BASIS_RISK for f in risk_report["flags"]),
        "a sector above the variance threshold went unflagged on risk",
    )
    check(
        abs(sum(risk_report["risk_by_sector"].values()) - 1.0) < 1e-6,
        "sector risk shares do not account for all of the variance",
    )

    # 6. UNCLASSIFIED IS A BUCKET, NOT A SECTOR. A fund has no single sector,
    #    and a bucket of unrelated funds is not a concentrated position.
    with_funds = equal(IT + FUNDS)
    fund_report = concentration_report(exposure(with_funds))
    named = [s for s in fund_report["weight_by_sector"] if s != SECTOR_UNCLASSIFIED]
    check(
        fund_report["sectors_present"] == len(named),
        f"sectors_present {fund_report['sectors_present']} counts "
        f"UNCLASSIFIED among the {len(named)} sectors actually held",
    )
    check(
        SECTOR_UNCLASSIFIED not in fund_report["benchmarks"],
        "the unclassified bucket was given a sector benchmark ETF",
    )
    check(
        herfindahl({"S0": 0.5, "S1": 0.25, SECTOR_UNCLASSIFIED: 0.25})
        is not None
        and abs(
            herfindahl({"S0": 0.5, "S1": 0.25, SECTOR_UNCLASSIFIED: 0.25})
            - herfindahl({"S0": 2 / 3, "S1": 1 / 3})
        )
        < 1e-6,
        "UNCLASSIFIED was folded into the HHI as though it were a sector, "
        "reporting concentration in a bucket of unrelated funds",
    )
    check(
        herfindahl({SECTOR_UNCLASSIFIED: 1.0}) is None,
        "an HHI of 0.0 was reported when nothing could be classified — that "
        "reads as perfect diversification, the opposite of unknown",
    )

    # 7. A MOSTLY-UNCLASSIFIED BOOK SAYS SO. Otherwise 'top sector 30%' is
    #    really '30% of the part we could classify'.
    opaque = concentration_report(exposure({"NVDA": 0.3, "VOO": 0.7}))
    check(
        any(f["sector"] == SECTOR_UNCLASSIFIED for f in opaque["flags"]),
        f"{SECTOR_UNCLASSIFIED_REVIEW:.0%}+ of the book had no sector and the "
        f"report presented its sector shares without qualification",
    )

    # 8. THE HHI SCALE HOLDS, AND THE THRESHOLD CAN STAY QUIET. A gate that
    #    flags every portfolio measures nothing.
    for n in (2, 4, 10):
        even = {f"S{i}": 1.0 / n for i in range(n)}
        check(
            abs(herfindahl(even) - 1.0 / n) < 1e-6,
            f"{n} equal sectors did not score the 1/n HHI floor",
        )
        check(
            abs(effective_sectors(herfindahl(even)) - n) < 1e-4,
            f"{n} equal sectors did not imply {n} effective sectors",
        )
    spread = {f"S{i}": 1 / 12 for i in range(12)}
    check(
        herfindahl(spread) < SECTOR_HHI_REVIEW,
        f"an evenly spread 12-sector book scored above the "
        f"{SECTOR_HHI_REVIEW} HHI threshold — the threshold flags everything "
        f"and therefore means nothing",
    )
    check(
        herfindahl(equal_sector_shares := group_shares(single_sector, classify(single_sector)))
        >= SECTOR_HHI_REVIEW,
        f"a single-sector book scored HHI "
        f"{herfindahl(equal_sector_shares)} below the review threshold",
    )
    for candidate in (report, risk_report, fund_report, opaque):
        hhi = candidate.get("hhi")
        if hhi is None:
            continue
        present = candidate["sectors_present"]
        check(
            hhi >= (1.0 / present) - 1e-6,
            f"HHI {hhi} sits below the 1/n floor for {present} sectors",
        )
        check(
            candidate["effective_sectors"] <= present + 1e-6,
            f"effective sectors {candidate['effective_sectors']} exceeds the "
            f"{present} sectors actually held",
        )

    # 9. THE SHAPE RULE (inherited R1): a sector risk figure exists IFF it was
    #    measured. Summing absent shares renders the least-understood sector
    #    as the safest.
    unmeasured = concentration_report(
        exposure(single_sector, risk=None, reason="no return history for AVGO")
    )
    check(
        unmeasured["risk_basis"] == "NOT_EVALUATED",
        "a report with no per-name risk claimed a measured risk basis",
    )
    check(
        unmeasured["risk_by_sector"] == {},
        "absent per-name risk was summed into a sector total, rendering the "
        "least-understood sector as the safest",
    )
    check(
        "AVGO" in unmeasured["risk_reason"],
        "the underlying reason risk was unavailable was replaced rather than "
        "carried through",
    )
    check(
        concentration_problems(unmeasured) == [],
        f"an honestly-unevaluated report reported problems: "
        f"{concentration_problems(unmeasured)}",
    )
    for line in render_concentration(unmeasured):
        if "risk" in line:
            check(
                "0.0%" not in line.split("risk", 1)[1],
                "unmeasured sector risk rendered as 0.0%, which reads as "
                "'carries no risk'",
            )

    # 10. SHARES ACCOUNT FOR THE WHOLE BOOK. A dropped holding renormalises
    #     the rest upward and overstates concentration.
    for candidate in (report, risk_report, fund_report, opaque):
        total_weight = sum(candidate["weight_by_sector"].values())
        check(
            abs(total_weight - 1.0) < 1e-6,
            f"sector weights sum to {total_weight:.6f}, not 1",
        )

    # 11. CONCENTRATION IS DESCRIBED, NOT ENFORCED. The portfolio-level
    #     refusal is R7's decision, with its own evidence.
    check(
        not SECTOR_BLOCKS_TRADES,
        "R3 claims to block trades — a measurement wearing a policy's clothes",
    )
    check(
        not report["blocks_trades"],
        "a concentration report claims to block trades",
    )

    # 12. THE CONTRACT CHECK CAN FAIL. A problems() that never fires is not a
    #     check.
    for mutation, label in (
        (lambda r: r.update({"bases": [SECTOR_BASIS_WEIGHT]}), "a weight-only report"),
        (lambda r: r.update({"blocks_trades": True}), "a report that blocks trades"),
        (lambda r: r.update({"hhi": 0.001}), "an HHI below the 1/n floor"),
        (
            lambda r: r.update({"weight_by_sector": {"Information Technology": 0.5}}),
            "sector weights that do not sum to 1",
        ),
        (
            lambda r: r.update({"risk_by_sector": {"Information Technology": 1.0}}),
            "NOT_EVALUATED risk alongside reported shares",
        ),
        (
            lambda r: r.update(
                {"flags": [{"sector": "X", "basis": "vibes", "reason": "because"}]}
            ),
            "a flag on an unknown basis",
        ),
    ):
        broken = concentration_report(exposure(single_sector))
        mutation(broken)
        check(
            concentration_problems(broken) != [],
            f"{label} passed the contract check",
        )

    if FAILURES:
        print("SECTOR CONCENTRATION GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("SECTOR CONCENTRATION GATE: PASS")
    print(f"  sector map reused from market_context ({len(CONTEXT_SECTOR_ETFS)} sectors)")
    print(f"  single-sector book flagged despite every per-name cap passing")
    print(f"  weight >= {SECTOR_WEIGHT_REVIEW:.0%}, risk >= {SECTOR_RISK_REVIEW:.0%}, "
          f"HHI >= {SECTOR_HHI_REVIEW}")
    print(f"  UNCLASSIFIED excluded from sectors, HHI and flags")
    print(f"  concentration described, not enforced (R7 owns the refusal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
