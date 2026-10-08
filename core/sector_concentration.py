"""R3 sector concentration — the per-name cap, one level up.

R1 measured that position weight is not exposure: satisfying a 10% per-name
cap by splitting 40% NVDA across four correlated semiconductors *raised*
portfolio volatility. R3 is that same finding at the sector level, and on the
tracked portfolio it is not hypothetical.

**THE DECIDING MEASUREMENT: every per-name cap can pass while the book is one
bet.** MEASURED on the tracked portfolio, 38 of 73 sector-mapped holdings are
Information Technology — 52.1% of the book. At equal weight each of those 38
names is ~1.3%, so the concentration clears a 10% per-name cap with room to
spare, and clears R1's own 25% review threshold on every single holding. The
name-level report is silent. The sector-level report is not.

**Ten sectors held is not ten sectors of diversification.** MEASURED, the
portfolio spans 10 GICS sectors and scores a sector HHI of 0.3113 — an
effective count of 3.21 sectors. Counting sectors present answers a different
question from measuring how the money is spread across them, and only the
second one is about risk.

**Sector shares are reported on BOTH scales, never weight alone.** The reason
is R1's: weight says how much capital sits in a sector, risk says how much of
the portfolio's variance it drives, and they disagree. A sector's variance
share concentrates faster than its weight share, because correlated names
inside one sector contribute jointly rather than independently.

**A holding with no sector gets no sector.** MEASURED, 4 of 77 tracked
holdings are funds (VOO, SOXX, CIBR, NASA). A fund is not *in* a sector, and
assigning it one either invents concentration that is not there or hides
concentration that is — SOXX filed under Information Technology would deepen
an already-52% reading with an instrument that is itself diversified. They go
to an explicit UNCLASSIFIED bucket, which is reported and never treated as a
sector for flagging, ranking or HHI.

**Sector membership is not re-derived here.** `core.market_context.sector_for`
and `CONTEXT_SECTOR_ETFS` already carry the GICS map. A second sector table
would be a split brain, and the two copies would disagree the first time one
of them was updated.

**Concentration is described, not enforced.** Whether a portfolio-level
refusal follows is R7's decision, with its own evidence.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    SECTOR_BASES,
    SECTOR_BASIS_RISK,
    SECTOR_BASIS_WEIGHT,
    SECTOR_BLOCKS_TRADES,
    SECTOR_CONCENTRATION_VERSION,
    SECTOR_HHI_REVIEW,
    SECTOR_RISK_REVIEW,
    SECTOR_UNCLASSIFIED,
    SECTOR_UNCLASSIFIED_REVIEW,
    SECTOR_WEIGHT_REVIEW,
)
from core.market_context import sector_etf_for, sector_for


class SectorConcentrationError(ValueError):
    """Raised when a sector concentration request is structurally invalid."""


def classify(tickers) -> dict[str, str]:
    """Each ticker's sector, with the unknowns named rather than dropped.

    Every ticker appears in the result. A ticker whose sector cannot be
    established maps to SECTOR_UNCLASSIFIED — dropping it would quietly
    renormalise the remaining shares upward and overstate concentration.
    """
    mapping: dict[str, str] = {}
    for raw in tickers:
        ticker = str(raw or "").strip().upper()
        if not ticker:
            raise SectorConcentrationError("a holding has no ticker")
        mapping[ticker] = sector_for(ticker) or SECTOR_UNCLASSIFIED
    return mapping


def group_shares(
    shares: Mapping[str, float], sectors: Mapping[str, str]
) -> dict[str, float]:
    """Sum per-ticker shares into per-sector shares.

    Holdings with no sector accumulate under SECTOR_UNCLASSIFIED rather than
    being discarded, so the sector shares still sum to the whole book.
    """
    grouped: dict[str, float] = {}
    for ticker, share in shares.items():
        key = str(ticker or "").strip().upper()
        sector = sectors.get(key) or SECTOR_UNCLASSIFIED
        grouped[sector] = grouped.get(sector, 0.0) + float(share)
    return {s: round(v, 8) for s, v in sorted(grouped.items())}


def herfindahl(shares: Mapping[str, float]) -> float | None:
    """HHI over sector shares, renormalised across CLASSIFIED sectors only.

    UNCLASSIFIED is excluded because it is not a sector: a bucket holding four
    unrelated funds is not a concentrated position in anything, and counting
    it as one sector would report false concentration. The remaining shares
    are renormalised so the index stays on its [1/n, 1] scale.

    None when nothing is classified — an HHI over an empty set is not 0.0,
    which would read as perfect diversification.
    """
    classified = {
        s: float(v) for s, v in shares.items() if s != SECTOR_UNCLASSIFIED and v > 0
    }
    total = sum(classified.values())
    if not classified or total <= 0:
        return None
    return round(sum((v / total) ** 2 for v in classified.values()), 8)


def effective_sectors(hhi: float | None) -> float | None:
    """1/HHI — how many equally-weighted sectors this book behaves like.

    MEASURED, the tracked portfolio holds 10 sectors and behaves like 3.21.
    """
    if hhi is None or hhi <= 0:
        return None
    return round(1.0 / hhi, 6)


def _benchmarks(sectors: Mapping[str, str]) -> dict[str, str]:
    """The sector ETF standing in for each sector actually held.

    Derived per HOLDING and then keyed by sector, so a sector whose ETF is
    unknown is simply absent rather than paired with a wrong benchmark. The
    map itself lives in CONTEXT_SECTOR_ETFS; this does not copy it.
    """
    found: dict[str, str] = {}
    for ticker, sector in sectors.items():
        if sector == SECTOR_UNCLASSIFIED or sector in found:
            continue
        etf = sector_etf_for(ticker)
        if etf:
            found[sector] = etf
    return dict(sorted(found.items()))


def _flags(
    weight_by_sector: Mapping[str, float],
    risk_by_sector: Mapping[str, float],
    hhi: float | None,
    effective: float | None,
    sectors_present: int,
) -> list[dict]:
    """Sectors worth a look, on either scale, each naming which."""
    flags: list[dict] = []
    for sector, weight in sorted(weight_by_sector.items()):
        if sector == SECTOR_UNCLASSIFIED:
            continue
        if weight >= SECTOR_WEIGHT_REVIEW:
            flags.append(
                {
                    "sector": sector,
                    "basis": SECTOR_BASIS_WEIGHT,
                    "value": round(float(weight), 6),
                    "threshold": SECTOR_WEIGHT_REVIEW,
                    "reason": (
                        f"{weight:.1%} of portfolio VALUE sits in {sector}. "
                        f"MEASURED, a 52.1% sector share was carried by 38 "
                        f"names at ~1.3% each — no per-name cap can see it"
                    ),
                }
            )
    for sector, share in sorted(risk_by_sector.items()):
        if sector == SECTOR_UNCLASSIFIED:
            continue
        if share >= SECTOR_RISK_REVIEW:
            flags.append(
                {
                    "sector": sector,
                    "basis": SECTOR_BASIS_RISK,
                    "value": round(float(share), 6),
                    "threshold": SECTOR_RISK_REVIEW,
                    "reason": (
                        f"{share:.1%} of portfolio VARIANCE comes from "
                        f"{sector} — correlated names inside one sector "
                        f"contribute jointly, not independently"
                    ),
                }
            )
    if hhi is not None and hhi >= SECTOR_HHI_REVIEW:
        flags.append(
            {
                "sector": "*",
                "basis": SECTOR_BASIS_WEIGHT,
                "value": round(float(hhi), 6),
                "threshold": SECTOR_HHI_REVIEW,
                "reason": (
                    f"sector HHI {hhi:.4f} across {sectors_present} sectors "
                    f"held — an effective diversification of "
                    f"{effective:.2f} sectors. Holding ten is not the same as "
                    f"being spread across ten"
                ),
            }
        )
    unclassified = float(weight_by_sector.get(SECTOR_UNCLASSIFIED, 0.0))
    if unclassified >= SECTOR_UNCLASSIFIED_REVIEW:
        flags.append(
            {
                "sector": SECTOR_UNCLASSIFIED,
                "basis": SECTOR_BASIS_WEIGHT,
                "value": round(unclassified, 6),
                "threshold": SECTOR_UNCLASSIFIED_REVIEW,
                "reason": (
                    f"{unclassified:.1%} of the book has no sector, so every "
                    f"sector share below describes only the part that could "
                    f"be classified — a weaker claim than it appears"
                ),
            }
        )
    return flags


def concentration_report(exposure: Mapping[str, Any]) -> dict:
    """Sector concentration, derived from an R1 exposure report.

    It consumes R1's output rather than re-deriving weights, so the two
    reports cannot disagree about what is held.
    """
    if not isinstance(exposure, Mapping):
        raise SectorConcentrationError("exposure report is not a mapping")

    weights = exposure.get("weights") or {}
    if not isinstance(weights, Mapping):
        raise SectorConcentrationError("exposure report has no weights mapping")

    sectors = classify(weights.keys())
    weight_by_sector = group_shares(weights, sectors)
    present = sorted(s for s in weight_by_sector if s != SECTOR_UNCLASSIFIED)
    hhi = herfindahl(weight_by_sector)
    effective = effective_sectors(hhi)

    report: dict[str, Any] = {
        "version": SECTOR_CONCENTRATION_VERSION,
        "holdings": len(weights),
        "sectors": {
            ticker: sectors[ticker] for ticker in sorted(sectors)
        },
        "benchmarks": _benchmarks(sectors),
        "sectors_present": len(present),
        "weight_by_sector": weight_by_sector,
        "unclassified_weight": round(
            float(weight_by_sector.get(SECTOR_UNCLASSIFIED, 0.0)), 6
        ),
        "hhi": hhi,
        "effective_sectors": effective,
        "bases": list(SECTOR_BASES),
        "blocks_trades": SECTOR_BLOCKS_TRADES,
        "note": (
            "Concentration is described, not enforced. MEASURED, 38 IT names "
            "at ~1.3% each form a 52.1% single-sector bet that clears every "
            "per-name cap. The portfolio-level refusal is R7's decision."
        ),
    }

    # THE SHAPE RULE, inherited from R1: a sector risk figure exists IFF it
    # was measured. R1 reports risk_basis NOT_EVALUATED when a holding has no
    # return history; summing absent shares into a sector would render the
    # least-understood sector as the safest.
    shares = exposure.get("risk_contributions") or {}
    if exposure.get("risk_basis") == "NOT_EVALUATED" or not shares:
        report.update(
            {
                "risk_by_sector": {},
                "risk_basis": "NOT_EVALUATED",
                "risk_reason": (
                    str(exposure.get("risk_reason") or "").strip()
                    or "the exposure report carries no risk contributions"
                ),
            }
        )
    else:
        risk_by_sector = group_shares(shares, sectors)
        report.update(
            {
                "risk_by_sector": risk_by_sector,
                "risk_basis": SECTOR_BASIS_RISK,
                "risk_reason": "",
            }
        )

    report["flags"] = _flags(
        weight_by_sector,
        report.get("risk_by_sector") or {},
        hhi,
        effective,
        len(present),
    )
    return report


def concentration_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a concentration report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    bases = list(report.get("bases") or [])
    if SECTOR_BASIS_RISK not in bases:
        problems.append(
            "the report does not carry sector risk — sector weight alone "
            "cannot see that correlated names contribute jointly"
        )
    if SECTOR_BASIS_WEIGHT not in bases:
        problems.append("the report does not carry sector weight")
    if report.get("blocks_trades"):
        problems.append(
            "the report claims to block trades — R3 describes concentration, "
            "and the portfolio-level refusal is R7's decision"
        )

    weight_by_sector = report.get("weight_by_sector") or {}
    if weight_by_sector:
        total = sum(float(v) for v in weight_by_sector.values())
        if not 0.95 <= total <= 1.05:
            problems.append(
                f"sector weights sum to {total:.4f}, not 1 — a dropped "
                f"holding renormalises the rest upward and overstates "
                f"concentration"
            )

    # UNCLASSIFIED must never be counted as a sector.
    if SECTOR_UNCLASSIFIED in (report.get("weight_by_sector") or {}):
        present = int(report.get("sectors_present") or 0)
        named = len([s for s in weight_by_sector if s != SECTOR_UNCLASSIFIED])
        if present != named:
            problems.append(
                f"sectors_present is {present} but {named} sectors are named "
                f"— UNCLASSIFIED is a bucket, not a sector"
            )

    hhi = report.get("hhi")
    if hhi is not None:
        if not 0.0 < float(hhi) <= 1.0:
            problems.append(f"HHI {hhi} is outside (0, 1]")
        present = int(report.get("sectors_present") or 0)
        if present > 0 and float(hhi) < (1.0 / present) - 1e-6:
            problems.append(
                f"HHI {float(hhi):.6f} is below the 1/n floor "
                f"{1.0 / present:.6f} for {present} sectors"
            )
        effective = report.get("effective_sectors")
        if effective is None:
            problems.append("HHI was measured but effective_sectors is absent")
        elif present and float(effective) > present + 1e-6:
            problems.append(
                f"effective sectors {float(effective):.4f} exceeds the "
                f"{present} sectors actually held"
            )
    elif report.get("effective_sectors") is not None:
        problems.append(
            "effective_sectors was reported without an HHI to derive it from"
        )

    # THE SHAPE RULE: a risk figure exists IFF it was measured.
    if report.get("risk_basis") == "NOT_EVALUATED":
        if report.get("risk_by_sector"):
            problems.append(
                "sector risk was NOT_EVALUATED but risk shares were reported"
            )
        if not str(report.get("risk_reason") or "").strip():
            problems.append("sector risk was NOT_EVALUATED with no reason given")

    for flag in report.get("flags") or []:
        if flag.get("basis") not in SECTOR_BASES:
            problems.append(f"{flag.get('sector')}: flag on an unknown basis")
        if not str(flag.get("reason") or "").strip():
            problems.append(f"{flag.get('sector')}: flag with no reason")
    return problems


def render_concentration(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per sector, heaviest first."""
    lines: list[str] = []
    risk = report.get("risk_by_sector") or {}
    weight_by_sector = report.get("weight_by_sector") or {}
    for sector, weight in sorted(
        weight_by_sector.items(), key=lambda kv: (-float(kv[1]), kv[0])
    ):
        share = risk.get(sector)
        shown = "—" if share is None else f"{float(share):6.1%}"
        lines.append(f"  {sector:24s} weight {float(weight):6.1%}   risk {shown}")
    hhi = report.get("hhi")
    if hhi is not None:
        effective = report.get("effective_sectors")
        lines.append(
            f"  {'HHI':24s} {float(hhi):6.4f}          "
            f"effective {float(effective):.2f} of "
            f"{int(report.get('sectors_present') or 0)} sectors held"
        )
    return lines
