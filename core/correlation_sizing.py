"""R2 correlation-aware sizing — how big, given what is already held.

R1 established that weight is not exposure. R2 asks the next question.

**Equal-weight sizing ignores correlation, and the cost is large.** MEASURED, a
40% budget in four correlated semiconductors carries **50.5% more risk** than
the same 40% in four diverse names.

**A NEGATIVE RESULT THAT SHAPES THE WHOLE MODULE.** Reweighting *inside* a
correlated basket barely helps — equal-weight, inverse-volatility and
equal-risk-contribution all land within 1% of each other. The excess came from
the basket's *composition*:

    4 semis, equal weight          1.587%     +0.0%
    4 semis, inverse-vol           1.568%     -1.2%
    3 semis + 1 diversifier        1.429%    -10.0%
    1 semi  + 3 diversifiers       1.054%    -33.6%

Reweighting buys ~1%; changing what is held buys 10–34%. **Sizing cannot fix
selection**, and a module implying otherwise would sell a false remedy.

**So sizing is done against the whole portfolio.** MEASURED against a held
portfolio, a fixed 10% moved volatility +3.07% for AVGO and −9.24% for XOM.
The same nominal size means something different for every ticker; sizing to a
risk budget makes "position size" a comparable unit.

**A size is a proposal, never an order**, and it always says what bound it —
"the portfolio cannot absorb more of this" and "policy stops here" are
different answers.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.config import (
    SIZING_CAPPED,
    SIZING_IS_ADVISORY,
    SIZING_MAX_WEIGHT,
    SIZING_MIN_SESSIONS,
    SIZING_MIN_WEIGHT,
    SIZING_NOT_EVALUATED,
    SIZING_REFUSED,
    SIZING_REPORT_BINDING_CONSTRAINT,
    SIZING_RISK_BUDGET,
    SIZING_SEARCH_TOLERANCE,
    SIZING_SIZED,
    SIZING_VERDICTS,
    CORRELATION_SIZING_VERSION,
)
from core.position_exposure import (
    PositionExposureError,
    aligned_returns,
    covariance,
    portfolio_variance,
)


class CorrelationSizingError(ValueError):
    """Raised when a sizing request is structurally invalid."""


# What bound the answer. Reported always, because the two mean different things.
BINDING_RISK_BUDGET = "risk_budget"
BINDING_WEIGHT_CAP = "weight_cap"
BINDING_NONE = "none"


def _result(verdict: str, reason: str, **detail) -> dict:
    if verdict not in SIZING_VERDICTS:
        raise CorrelationSizingError(f"{verdict!r} is not a sizing verdict")
    record = {"verdict": verdict, "reason": reason, "advisory": SIZING_IS_ADVISORY}
    record.update(detail)
    return record


def blended(
    held: Mapping[str, float], ticker: str, size: float
) -> dict[str, float]:
    """The portfolio after adding `size` of `ticker`, funded pro rata.

    Funding PRO RATA rather than from cash, because the question is "what does
    this trade do to the mix", and a cash-funded version would conflate
    changing the mix with reducing total exposure.
    """
    scaled = {t: float(w) * (1.0 - size) for t, w in (held or {}).items()}
    scaled[ticker] = scaled.get(ticker, 0.0) + size
    return scaled


def volatility(
    weights: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> float:
    return math.sqrt(max(portfolio_variance(weights, tickers, cov), 0.0))


def size_position(
    held: Mapping[str, float],
    ticker: str,
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
    *,
    risk_budget: float = SIZING_RISK_BUDGET,
    max_weight: float = SIZING_MAX_WEIGHT,
    min_weight: float = SIZING_MIN_WEIGHT,
) -> dict:
    """The largest position that keeps the volatility increase inside the budget.

    Bisects on size. Reports WHAT BOUND IT: the risk budget or the weight cap.
    MEASURED, an uncorrelated name reached 50%+ on the risk budget alone while
    a correlated one stopped at 14% — without the cap the sizer would propose
    concentration in the name of diversification.
    """
    ticker = str(ticker).upper()
    if ticker not in tickers:
        raise CorrelationSizingError(
            f"{ticker!r} has no return history here, so its effect on the "
            f"portfolio cannot be measured"
        )
    if not 0.0 < risk_budget < 1.0:
        raise CorrelationSizingError("the risk budget must lie inside (0, 1)")
    if not 0.0 < min_weight < max_weight:
        raise CorrelationSizingError("min_weight must be positive and below the cap")

    base = volatility(held, tickers, cov)
    if base <= 0:
        return _result(
            SIZING_NOT_EVALUATED,
            "the held portfolio has no measurable volatility, so a relative "
            "risk budget has nothing to bound",
            ticker=ticker,
        )

    def increase(size: float) -> float:
        return volatility(blended(held, ticker, size), tickers, cov) / base - 1.0

    # THE MINIMUM MUST FIT FIRST. If even the smallest tradeable position
    # breaches the budget, no size does, and reporting a smaller one would be
    # proposing a trade the portfolio cannot absorb.
    if increase(min_weight) > risk_budget:
        return _result(
            SIZING_REFUSED,
            f"even the minimum {min_weight:.1%} raises portfolio volatility "
            f"{increase(min_weight):+.2%}, beyond the {risk_budget:.0%} budget",
            ticker=ticker,
            minimum=min_weight,
            minimum_increase=round(increase(min_weight), 6),
            risk_budget=risk_budget,
            base_volatility=round(base, 8),
            binding=BINDING_RISK_BUDGET,
        )

    # Bisect for the largest size inside the budget, bounded by the cap.
    low, high = min_weight, max_weight
    if increase(max_weight) <= risk_budget:
        chosen = max_weight
        binding = BINDING_WEIGHT_CAP
    else:
        while high - low > SIZING_SEARCH_TOLERANCE:
            middle = (low + high) / 2.0
            if increase(middle) <= risk_budget:
                low = middle
            else:
                high = middle
        chosen = low
        binding = BINDING_RISK_BUDGET

    final_increase = increase(chosen)
    verdict = SIZING_CAPPED if binding == BINDING_WEIGHT_CAP else SIZING_SIZED
    return _result(
        verdict,
        (
            f"{chosen:.1%} is the weight cap; the risk budget would allow more "
            f"(this size moves volatility {final_increase:+.2%})"
            if binding == BINDING_WEIGHT_CAP
            else f"{chosen:.1%} moves portfolio volatility {final_increase:+.2%}, "
            f"at the {risk_budget:.0%} budget"
        ),
        ticker=ticker,
        size=round(chosen, 6),
        volatility_increase=round(final_increase, 6),
        base_volatility=round(base, 8),
        sized_volatility=round(
            volatility(blended(held, ticker, chosen), tickers, cov), 8
        ),
        risk_budget=risk_budget,
        max_weight=max_weight,
        binding=binding,
        diversifying=bool(final_increase < 0),
    )


def compare_candidates(
    held: Mapping[str, float],
    candidates: Sequence[str],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
    **kwargs,
) -> list[dict]:
    """Size several candidates against the same portfolio, best first.

    Ordered by how much risk each ADDS at its proposed size, so a diversifier
    that lowers portfolio volatility sorts above one that raises it — which a
    per-ticker forecast ranking cannot express.
    """
    sized = []
    for candidate in candidates or []:
        try:
            sized.append(size_position(held, candidate, tickers, cov, **kwargs))
        except CorrelationSizingError as error:
            sized.append(
                _result(
                    SIZING_NOT_EVALUATED,
                    str(error),
                    ticker=str(candidate).upper(),
                )
            )
    return sorted(
        sized,
        key=lambda item: (
            item.get("volatility_increase")
            if item.get("volatility_increase") is not None
            else math.inf
        ),
    )


def sizing_report(
    held: Mapping[str, float],
    candidates: Sequence[str],
    series: Mapping[str, Mapping[str, float]],
    *,
    min_sessions: int = SIZING_MIN_SESSIONS,
    **kwargs,
) -> dict:
    """Size every candidate against the held portfolio."""
    needed = sorted(set(held or {}) | {str(c).upper() for c in candidates or []})
    available = {t: (series or {}).get(t) for t in needed}
    missing = sorted(t for t, s in available.items() if not s)
    if missing:
        return {
            "version": CORRELATION_SIZING_VERSION,
            "verdict": SIZING_NOT_EVALUATED,
            "reason": (
                f"no return history for {missing}; a size proposed without it "
                f"would assume an unmeasured correlation"
            ),
            "proposals": [],
            "advisory": SIZING_IS_ADVISORY,
        }

    try:
        tickers, matrix = aligned_returns(available, min_sessions=min_sessions)
    except PositionExposureError as error:
        return {
            "version": CORRELATION_SIZING_VERSION,
            "verdict": SIZING_NOT_EVALUATED,
            "reason": str(error),
            "proposals": [],
            "advisory": SIZING_IS_ADVISORY,
        }

    cov = covariance(matrix)
    proposals = compare_candidates(held, candidates, tickers, cov, **kwargs)
    return {
        "version": CORRELATION_SIZING_VERSION,
        "verdict": SIZING_SIZED,
        "reason": "",
        "sessions": len(matrix[0]) if matrix else 0,
        "base_volatility": round(volatility(held, tickers, cov), 8),
        "proposals": proposals,
        "advisory": SIZING_IS_ADVISORY,
        "reports_binding_constraint": SIZING_REPORT_BINDING_CONSTRAINT,
        "note": (
            "Sizing cannot fix selection. MEASURED, reweighting inside a "
            "correlated basket buys ~1% of risk while replacing a member with "
            "an uncorrelated name buys 10-34%."
        ),
    }


def sizing_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a sizing report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    if not report.get("advisory", False):
        problems.append(
            "the report does not declare itself advisory — a size is a "
            "proposal, never an order"
        )
    for proposal in report.get("proposals") or []:
        verdict = proposal.get("verdict")
        if verdict not in SIZING_VERDICTS:
            problems.append(f"{proposal.get('ticker')}: unknown verdict {verdict!r}")
        if not str(proposal.get("reason") or "").strip():
            problems.append(f"{proposal.get('ticker')}: no reason given")
        # THE SHAPE RULE: a size exists IFF one was found. A missing size
        # rendered as 0.0 is indistinguishable from "propose nothing", which
        # is a different answer from "this could not be sized".
        if verdict in (SIZING_SIZED, SIZING_CAPPED):
            if proposal.get("size") is None:
                problems.append(f"{proposal.get('ticker')}: sized with no size")
            if SIZING_REPORT_BINDING_CONSTRAINT and not proposal.get("binding"):
                problems.append(
                    f"{proposal.get('ticker')}: does not say what bound it — "
                    f"'the portfolio cannot absorb more' and 'policy stops "
                    f"here' are different answers"
                )
            if proposal.get("size") is not None and proposal["size"] > SIZING_MAX_WEIGHT + 1e-9:
                problems.append(
                    f"{proposal.get('ticker')}: proposed {proposal['size']:.3f}, "
                    f"above the {SIZING_MAX_WEIGHT} cap"
                )
        if verdict in (SIZING_NOT_EVALUATED, SIZING_REFUSED) and proposal.get("size"):
            problems.append(
                f"{proposal.get('ticker')}: {verdict} but carries a size"
            )
    return problems


def render_sizing(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per proposal."""
    lines: list[str] = []
    for proposal in report.get("proposals") or []:
        size = proposal.get("size")
        shown = "—" if size is None else f"{size:6.1%}"
        change = proposal.get("volatility_increase")
        delta = "" if change is None else f"  vol {change:+.2%}"
        lines.append(
            f"  {str(proposal.get('ticker')):8s} {shown} "
            f"{str(proposal.get('verdict')):14s}{delta}"
        )
    return lines
