"""R1 position exposure — what is held, and what holding it exposes you to.

Sprint R moves from "is this ticker attractive?" to "does acting on this
forecast improve the current portfolio without violating risk constraints?"
R1 is the foundation.

**There was no portfolio state at all.** `fetch_data.PORTFOLIO_TICKERS` is a
77-name *watchlist* — symbols with no share counts, no cost basis, no weights.
The Sprint R question could not previously be asked, because there was nothing
to improve.

**THE DECIDING MEASUREMENT: position weight is not exposure.** On real returns,
a portfolio of 10% each in NVDA/AMD/AVGO/SOXX looks four times more
diversified than one holding 40% NVDA outright, and is very slightly *more*
volatile (1.760% vs 1.741%) — the same bet, spread thinner.

**A weight cap is gameable in the wrong direction.** Satisfying a 10% cap by
splitting 40% NVDA across four correlated semiconductors *raised* volatility.
A rule that can be satisfied by making risk worse is not merely incomplete.

**And 100% VOO is the least volatile portfolio tested** (1.006%) while scoring
worst on every weight-based concentration measure. A single diversified fund
is not a concentrated position. So exposure is reported on both scales —
weight *and* risk contribution — and never weight alone.

**Returns are aligned by DATE, never by position.** MEASURED, a position-offset
join reported MSFT's correlation with everything as ~0.00 — including against
VOO, which is really 0.53 — because MSFT's series ended three days earlier. It
does not fail loudly; it silently reports independence.

**Exposure is described, not enforced.** Whether a trade is permitted is a
later task with its own evidence.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from core.config import (
    EXPOSURE_ALIGN_BY_DATE,
    EXPOSURE_BASES,
    EXPOSURE_BASIS_RISK,
    EXPOSURE_BASIS_WEIGHT,
    EXPOSURE_BLOCKS_TRADES,
    EXPOSURE_MIN_SESSIONS,
    EXPOSURE_RISK_REVIEW,
    EXPOSURE_WEIGHT_REVIEW,
    POSITION_EXPOSURE_VERSION,
    POSITION_REQUIRED_FIELDS,
)


class PositionExposureError(ValueError):
    """Raised when a portfolio or exposure request is structurally invalid."""


def position_problems(position: Mapping[str, Any]) -> list[str]:
    """What stops this being a holding? Empty means it is one."""
    problems: list[str] = []
    if not isinstance(position, Mapping):
        return ["position is not a mapping"]
    for field in POSITION_REQUIRED_FIELDS:
        value = position.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"{field} is required")
    quantity = position.get("quantity")
    if quantity is not None:
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            problems.append("quantity must be numeric")
        elif math.isnan(float(quantity)) or math.isinf(float(quantity)):
            problems.append("quantity must be finite")
    price = position.get("price")
    if price is not None and (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or float(price) < 0
    ):
        problems.append("price must be a non-negative number")
    return problems


def load_positions(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Normalise holdings, refusing anything that is not a position.

    A watchlist entry is not a holding. Accepting one would produce an
    exposure report with a silent hole in it.
    """
    positions: list[dict] = []
    for row in rows or []:
        problems = position_problems(row)
        if problems:
            raise PositionExposureError(
                f"{(row or {}).get('ticker', '?')!r} is not a position: "
                + "; ".join(problems)
            )
        positions.append(
            {
                "ticker": str(row["ticker"]).upper(),
                "quantity": float(row["quantity"]),
                "as_of": str(row["as_of"]),
                "price": None if row.get("price") is None else float(row["price"]),
                "cost_basis": (
                    None if row.get("cost_basis") is None else float(row["cost_basis"])
                ),
            }
        )
    return positions


def market_values(
    positions: Sequence[Mapping[str, Any]], prices: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Value per ticker, from an explicit price or the position's own."""
    prices = {str(k).upper(): float(v) for k, v in (prices or {}).items()}
    values: dict[str, float] = {}
    for position in positions or []:
        ticker = str(position["ticker"]).upper()
        price = prices.get(ticker, position.get("price"))
        if price is None:
            raise PositionExposureError(
                f"no price for {ticker} — an exposure computed without a price "
                f"would silently weight the position at zero"
            )
        values[ticker] = values.get(ticker, 0.0) + float(position["quantity"]) * float(price)
    return values


def weights(values: Mapping[str, float]) -> dict[str, float]:
    """Share of portfolio value per ticker."""
    total = sum(abs(float(v)) for v in (values or {}).values())
    if total <= 0:
        return {}
    return {ticker: float(value) / total for ticker, value in values.items()}


def aligned_returns(
    series: Mapping[str, Mapping[str, float]],
    *,
    min_sessions: int = EXPOSURE_MIN_SESSIONS,
) -> tuple[list[str], list[list[float]]]:
    """Return series aligned on the dates every ticker shares.

    ALIGNED BY DATE, NOT BY POSITION. MEASURED, taking the last N rows of each
    series instead put MSFT three sessions out of step and reported its
    correlation with every other holding as ~0.00, including 0.00 against VOO
    where the truth is 0.53. A position-offset join reports independence
    rather than failing.
    """
    if not EXPOSURE_ALIGN_BY_DATE:  # pragma: no cover - config-guarded
        raise PositionExposureError("date alignment is mandatory")
    if not series:
        return [], []

    tickers = sorted(series)
    common: set[str] | None = None
    for ticker in tickers:
        dates = {str(d) for d in (series[ticker] or {})}
        common = dates if common is None else (common & dates)
    shared = sorted(common or set())
    if len(shared) < min_sessions:
        raise PositionExposureError(
            f"only {len(shared)} session(s) are common to all holdings, below "
            f"the {min_sessions} required — a covariance from fewer is a "
            f"confident-looking number for a relationship never observed"
        )
    matrix = [[float(series[t][d]) for d in shared] for t in tickers]
    return tickers, matrix


def covariance(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Sample covariance of aligned return rows."""
    if not matrix:
        return []
    count = len(matrix[0])
    if count < 2:
        raise PositionExposureError("a covariance needs at least two observations")
    means = [sum(row) / count for row in matrix]
    size = len(matrix)
    result = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            total = sum(
                (matrix[i][k] - means[i]) * (matrix[j][k] - means[j])
                for k in range(count)
            )
            value = total / (count - 1)
            result[i][j] = value
            result[j][i] = value
    return result


def portfolio_variance(
    weight_map: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> float:
    vector = [float(weight_map.get(t, 0.0)) for t in tickers]
    return sum(
        vector[i] * cov[i][j] * vector[j]
        for i in range(len(tickers))
        for j in range(len(tickers))
    )


def risk_contributions(
    weight_map: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Each holding's share of portfolio VARIANCE.

    This is what weight cannot see. MEASURED, a 4x10% semiconductor basket
    puts 56.9% of its variance in the same bet as a single 40% NVDA position.
    """
    total = portfolio_variance(weight_map, tickers, cov)
    if total <= 0:
        return {}
    vector = [float(weight_map.get(t, 0.0)) for t in tickers]
    shares: dict[str, float] = {}
    for i, ticker in enumerate(tickers):
        marginal = sum(cov[i][j] * vector[j] for j in range(len(tickers)))
        shares[ticker] = vector[i] * marginal / total
    return shares


def marginal_exposure(
    weight_map: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
    ticker: str,
    size: float,
) -> dict:
    """What would adding `size` of `ticker` do to portfolio volatility?

    THE QUESTION SPRINT R ASKS. MEASURED against a held portfolio, the same 5%
    purchase moved volatility +0.016% for SOXX and −0.057% for VOO: the same
    trade helps or hurts depending entirely on what is already held, and no
    per-ticker forecast can answer that.
    """
    ticker = str(ticker).upper()
    if ticker not in tickers:
        raise PositionExposureError(
            f"{ticker!r} has no return history in this portfolio, so its "
            f"marginal effect cannot be measured"
        )
    if not 0.0 < size < 1.0:
        raise PositionExposureError("size must be a fraction inside (0, 1)")

    before = math.sqrt(max(portfolio_variance(weight_map, tickers, cov), 0.0))
    scaled = {t: float(w) * (1.0 - size) for t, w in weight_map.items()}
    scaled[ticker] = scaled.get(ticker, 0.0) + size
    after = math.sqrt(max(portfolio_variance(scaled, tickers, cov), 0.0))
    change = after - before
    return {
        "ticker": ticker,
        "size": size,
        "volatility_before": round(before, 8),
        "volatility_after": round(after, 8),
        "change": round(change, 8),
        "direction": "ADDS_RISK" if change > 0 else "REDUCES_RISK" if change < 0 else "NEUTRAL",
        "relative_change": (round(change / before, 6) if before > 0 else None),
    }


def exposure_report(
    positions: Iterable[Mapping[str, Any]],
    series: Mapping[str, Mapping[str, float]] | None = None,
    prices: Mapping[str, float] | None = None,
    *,
    min_sessions: int = EXPOSURE_MIN_SESSIONS,
) -> dict:
    """What is held, on both scales.

    A weight-only report is not produced: MEASURED, 100% VOO is the least
    volatile portfolio tested while scoring worst on every weight-based
    concentration measure.
    """
    held = load_positions(positions)
    values = market_values(held, prices)
    weight_map = weights(values)

    report: dict[str, Any] = {
        "version": POSITION_EXPOSURE_VERSION,
        "positions": len(held),
        "total_value": round(sum(values.values()), 6),
        "weights": {t: round(w, 6) for t, w in sorted(weight_map.items())},
        "bases": list(EXPOSURE_BASES),
        "blocks_trades": EXPOSURE_BLOCKS_TRADES,
        "note": (
            "Exposure is described, not enforced. Weight alone is gameable: "
            "MEASURED, satisfying a 10% cap by splitting 40% NVDA across four "
            "correlated semiconductors raised portfolio volatility."
        ),
    }

    if not weight_map:
        report.update(
            {
                "risk_contributions": {},
                "volatility": None,
                "risk_basis": "NOT_EVALUATED",
                "risk_reason": "the portfolio holds no value",
                "flags": [],
            }
        )
        return report

    held_tickers = sorted(weight_map)
    available = {t: (series or {}).get(t) for t in held_tickers}
    missing = sorted(t for t, s in available.items() if not s)
    if missing:
        # RISK CONTRIBUTION IS NOT GUESSED. A holding with no return history
        # cannot be given a variance share, and filling it with zero would
        # render the least-understood position as the safest.
        report.update(
            {
                "risk_contributions": {},
                "volatility": None,
                "risk_basis": "NOT_EVALUATED",
                "risk_reason": (
                    f"no return history for {missing}; a missing history "
                    f"scored as zero risk would render the least-understood "
                    f"holding as the safest"
                ),
                "flags": _flags(weight_map, {}),
            }
        )
        return report

    tickers, matrix = aligned_returns(
        {t: available[t] for t in held_tickers}, min_sessions=min_sessions
    )
    cov = covariance(matrix)
    shares = risk_contributions(weight_map, tickers, cov)
    variance = portfolio_variance(weight_map, tickers, cov)

    report.update(
        {
            "tickers": tickers,
            "sessions": len(matrix[0]) if matrix else 0,
            "risk_contributions": {t: round(v, 6) for t, v in sorted(shares.items())},
            "volatility": round(math.sqrt(max(variance, 0.0)), 8),
            "risk_basis": EXPOSURE_BASIS_RISK,
            "risk_reason": "",
            "flags": _flags(weight_map, shares),
        }
    )
    return report


def _flags(
    weight_map: Mapping[str, float], shares: Mapping[str, float]
) -> list[dict]:
    """Positions worth a look, on either scale, each naming which."""
    flags: list[dict] = []
    for ticker, weight in sorted(weight_map.items()):
        if weight >= EXPOSURE_WEIGHT_REVIEW:
            flags.append(
                {
                    "ticker": ticker,
                    "basis": EXPOSURE_BASIS_WEIGHT,
                    "value": round(float(weight), 6),
                    "threshold": EXPOSURE_WEIGHT_REVIEW,
                    "reason": (
                        f"{weight:.1%} of portfolio VALUE. This is a review "
                        f"flag, not a limit: MEASURED, 100% VOO breaches it "
                        f"while being the least volatile portfolio tested"
                    ),
                }
            )
    for ticker, share in sorted(shares.items()):
        if share >= EXPOSURE_RISK_REVIEW:
            flags.append(
                {
                    "ticker": ticker,
                    "basis": EXPOSURE_BASIS_RISK,
                    "value": round(float(share), 6),
                    "threshold": EXPOSURE_RISK_REVIEW,
                    "reason": (
                        f"{share:.1%} of portfolio VARIANCE — the scale a "
                        f"weight cap cannot see"
                    ),
                }
            )
    return flags


def exposure_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on an exposure report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    bases = list(report.get("bases") or [])
    if EXPOSURE_BASIS_RISK not in bases:
        problems.append(
            "the report does not carry risk contribution — weight alone "
            "cannot distinguish a diversified fund from a concentrated bet"
        )
    if EXPOSURE_BASIS_WEIGHT not in bases:
        problems.append("the report does not carry weight")
    if report.get("blocks_trades"):
        problems.append(
            "the report claims to block trades — R1 describes exposure, and a "
            "weight cap enforced alone is gameable in the wrong direction"
        )
    # THE SHAPE RULE (inherited F3->L3): a risk figure exists IFF it was
    # measured, because Number(x ?? 0) renders a missing share as 0.0 — the
    # same number as a holding that genuinely carries no risk.
    if report.get("risk_basis") == "NOT_EVALUATED":
        if report.get("risk_contributions"):
            problems.append(
                "risk was NOT_EVALUATED but risk contributions were reported"
            )
        if not str(report.get("risk_reason") or "").strip():
            problems.append("risk was NOT_EVALUATED with no reason given")
    else:
        shares = report.get("risk_contributions") or {}
        if shares:
            total = sum(float(v) for v in shares.values())
            if not 0.95 <= total <= 1.05:
                problems.append(
                    f"risk contributions sum to {total:.4f}, not 1 — they are "
                    f"shares of variance and must account for all of it"
                )
    for flag in report.get("flags") or []:
        if flag.get("basis") not in EXPOSURE_BASES:
            problems.append(f"{flag.get('ticker')}: flag on an unknown basis")
        if not str(flag.get("reason") or "").strip():
            problems.append(f"{flag.get('ticker')}: flag with no reason")
    return problems


def render_exposure(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per holding."""
    lines: list[str] = []
    shares = report.get("risk_contributions") or {}
    for ticker, weight in sorted((report.get("weights") or {}).items()):
        share = shares.get(ticker)
        shown = "—" if share is None else f"{share:6.1%}"
        lines.append(f"  {ticker:8s} weight {weight:6.1%}   risk {shown}")
    return lines
