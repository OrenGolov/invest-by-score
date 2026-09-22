"""R7 portfolio-level NO_TRADE — where every deferral comes due.

R1, R3, R4, R5 and R6 each carry `*_BLOCKS_TRADES = False` and say, in their
own reports, that the refusal belongs here. This is the module that refuses,
and the only one.

**It does not duplicate W2.** `RISK_POLICY_V2` already vetoes on EVIDENCE
quality, and every one of its twelve rules reads a single ticker's inputs —
data quality, source confidence, point-in-time validity, that ticker's score
and regime. None of them can see the portfolio. R7 asks the question W2
structurally cannot: given everything already held, can the book absorb this?

**THE DECIDING MEASUREMENT: a set of individually-correct trades can be
collectively impossible.** On a diversified 4-name book with 6 candidates in
one correlated cluster, R2 sized each candidate against the ORIGINAL portfolio
and approved all six at 20% — every one marked `diversifying=True`, every one
*reducing* volatility in isolation. The approved set sums to **120% of the
book**:

    total executed   portfolio vol   cluster share of RISK
             24%          -0.67%              27%
             48%          +7.88%              60%
             72%         +23.76%              85%
             96%         +44.57%              99%
            120%      INFEASIBLE - exceeds 100% of the book

Every trade was individually right and the set is unexecutable. Nothing below
R7 can see it, because each module sizes ONE candidate against the CURRENT
book and never against the other candidates. That is not a bug in R2 — it is
the boundary of the question R2 asks.

**Fail-closed, inherited from W2.** A missing input TRIGGERS the rule that
reads it rather than passing it. A portfolio check that silently passes when
its evidence is absent is worse than no check: it manufactures a confident
approval out of nothing.

**NO_TRADE and NOT_EVALUATED stay distinct.** "The portfolio refuses this" and
"the portfolio could not be assessed" are different answers. Collapsing them
turns an absence of evidence into a decision — which is the failure this
codebase has removed everywhere else.

**A veto is not overridden by warnings, and warnings never sum into a veto.**
A WARN records something a reader must see; only a veto-severity rule refuses.
Counting warnings until they become a refusal would invent a threshold nobody
measured.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from core.config import (
    PORTFOLIO_BLOCKS_TRADES,
    PORTFOLIO_DECISION_VERSION,
    PORTFOLIO_FAIL_CLOSED,
    PORTFOLIO_MAX_TOTAL_SIZE,
    PORTFOLIO_MAX_VOLATILITY_INCREASE,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    PORTFOLIO_REDUCED,
    PORTFOLIO_RULES,
    PORTFOLIO_SEVERITY_VETO,
    PORTFOLIO_SEVERITY_WARN,
    PORTFOLIO_VERDICTS,
    SECTOR_WEIGHT_REVIEW,
    STRESS_MATERIAL_DEGRADATION,
)
from core.expected_impact import blend
from core.position_exposure import portfolio_variance


class PortfolioDecisionError(ValueError):
    """Raised when a portfolio decision request is structurally invalid."""


def accepted_sizes(proposals: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """The size each candidate would actually trade at, upstream refusals honoured.

    R4's adjusted size wins over R2's raw size, and a candidate carrying no
    size is absent rather than present at 0.0 — the shape rule every module in
    this sprint follows.
    """
    sizes: dict[str, float] = {}
    for ticker, proposal in (proposals or {}).items():
        if not isinstance(proposal, Mapping):
            raise PortfolioDecisionError(f"{ticker}: proposal is not a mapping")
        size = proposal.get("adjusted_size")
        if size is None:
            size = proposal.get("size")
        if size is None:
            continue
        value = float(size)
        if value > 0:
            sizes[str(ticker).upper()] = value
    return sizes


def set_volatility_increase(
    held: Mapping[str, float],
    sizes: Mapping[str, float],
    tickers: Sequence[str],
    cov: Sequence[Sequence[float]],
) -> float | None:
    """Portfolio volatility change if the WHOLE set is executed together.

    This is the number no other module computes. R2 measures one candidate
    against the current book; this measures the book after all of them.
    None when it cannot be measured — never 0.0, which reads as "no change".
    """
    if not sizes:
        return None
    base_variance = portfolio_variance(held, tickers, cov)
    if base_variance <= 0:
        return None
    combined = dict(held)
    for ticker, size in sizes.items():
        combined = blend(combined, ticker, float(size))
    after = portfolio_variance(combined, tickers, cov)
    base = math.sqrt(base_variance)
    if base <= 0:
        return None
    return round(math.sqrt(max(after, 0.0)) / base - 1.0, 8)


def _rule(rule_id: str, triggered: bool, detail: str) -> dict:
    """One evaluated rule, carrying the measurement it enforces."""
    spec = PORTFOLIO_RULES[rule_id]
    return {
        "rule_id": rule_id,
        "severity": spec["severity"],
        "source": spec["source"],
        "triggered": bool(triggered),
        "detail": detail,
        "measurement": spec["detail"],
    }


def evaluate_portfolio(
    held: Mapping[str, float],
    proposals: Mapping[str, Mapping[str, Any]],
    tickers: Sequence[str] | None = None,
    cov: Sequence[Sequence[float]] | None = None,
    *,
    concentration: Mapping[str, Any] | None = None,
    stress: Mapping[str, Any] | None = None,
    max_total: float = PORTFOLIO_MAX_TOTAL_SIZE,
    max_volatility: float = PORTFOLIO_MAX_VOLATILITY_INCREASE,
) -> dict:
    """Can the book absorb this set of trades? Fail-closed throughout.

    `concentration` is an R3 report and `stress` an R6 report. Both are
    optional inputs whose ABSENCE triggers their rules rather than passing
    them, because a portfolio approved without checking concentration is not
    the same as one checked and found clean.
    """
    if not isinstance(held, Mapping):
        raise PortfolioDecisionError("a mapping of held weights is required")
    if not isinstance(proposals, Mapping):
        raise PortfolioDecisionError("a mapping of proposals is required")

    sizes = accepted_sizes(proposals)
    rules: list[dict] = []

    # 1. NO CANDIDATE SURVIVED UPSTREAM. Proceeding would execute nothing
    #    while reporting approval, which is a false statement about the book.
    if not sizes:
        rules.append(
            _rule(
                "no_candidate_survived",
                True,
                f"{len(proposals)} candidate(s) were considered and none "
                f"carries a tradeable size; every one was refused upstream",
            )
        )
    else:
        rules.append(
            _rule(
                "no_candidate_survived",
                False,
                f"{len(sizes)} of {len(proposals)} candidate(s) carry a "
                f"tradeable size",
            )
        )

    # 2. THE SET MUST FIT IN THE BOOK. MEASURED, six individually-approved
    #    trades summed to 120%.
    total = round(sum(sizes.values()), 8)
    if total > max_total:
        rules.append(
            _rule(
                "total_size_exceeds_budget",
                True,
                f"the proposed set totals {total:.1%} of the book, beyond the "
                f"{max_total:.0%} budget. Each candidate was sized against "
                f"the portfolio as it stands, never against the others",
            )
        )
    else:
        rules.append(
            _rule(
                "total_size_exceeds_budget",
                False,
                f"the proposed set totals {total:.1%}, inside the "
                f"{max_total:.0%} budget",
            )
        )

    # 3. THE SET'S JOINT VOLATILITY EFFECT. FAIL-CLOSED: if it cannot be
    #    measured, the rule triggers. MEASURED, a set whose members each
    #    reduced volatility raised it +44.57% together.
    joint = None
    if sizes and tickers is not None and cov is not None:
        try:
            joint = set_volatility_increase(held, sizes, tickers, cov)
        except Exception as error:  # a malformed covariance must not pass
            joint = None
            rules.append(
                _rule(
                    "set_volatility_exceeds_budget",
                    True,
                    f"the set volatility effect could not be measured "
                    f"({error}); an unmeasurable joint effect is not a safe "
                    f"one",
                )
            )
    if not any(r["rule_id"] == "set_volatility_exceeds_budget" for r in rules):
        if joint is None:
            triggered = bool(sizes) and PORTFOLIO_FAIL_CLOSED
            rules.append(
                _rule(
                    "set_volatility_exceeds_budget",
                    triggered,
                    (
                        "no covariance was supplied, so the set's joint "
                        "volatility effect is unknown. Fail-closed: MEASURED, "
                        "a set of individually volatility-REDUCING trades "
                        "raised portfolio volatility +44.57% together"
                        if triggered
                        else "no set to evaluate"
                    ),
                )
            )
        elif joint > max_volatility:
            rules.append(
                _rule(
                    "set_volatility_exceeds_budget",
                    True,
                    f"executing the whole set moves portfolio volatility "
                    f"{joint:+.2%}, beyond the {max_volatility:.0%} budget "
                    f"that bounds any single trade",
                )
            )
        else:
            rules.append(
                _rule(
                    "set_volatility_exceeds_budget",
                    False,
                    f"executing the whole set moves portfolio volatility "
                    f"{joint:+.2%}, inside the {max_volatility:.0%} budget",
                )
            )

    # 4. SECTOR CONCENTRATION (R3). Absence triggers the rule.
    rules.append(_concentration_rule(concentration))

    # 5. STRESS (R6). Absence triggers the rule.
    rules.append(_stress_rule(stress))

    vetoes = [
        r["rule_id"]
        for r in rules
        if r["triggered"] and r["severity"] == PORTFOLIO_SEVERITY_VETO
    ]
    warnings = [
        r["rule_id"]
        for r in rules
        if r["triggered"] and r["severity"] == PORTFOLIO_SEVERITY_WARN
    ]

    # A VETO REFUSES. Warnings never sum into a refusal: counting them until
    # they become one would invent a threshold nobody measured.
    if vetoes:
        verdict = PORTFOLIO_NO_TRADE
        reason = (
            f"the portfolio refuses this set: {', '.join(vetoes)}. "
            f"{'; '.join(r['detail'] for r in rules if r['rule_id'] in vetoes)}"
        )
    elif not sizes:
        verdict = PORTFOLIO_NOT_EVALUATED
        reason = "there is no set to decide on"
    elif warnings:
        verdict = PORTFOLIO_REDUCED
        reason = (
            f"the set may proceed with {len(warnings)} warning(s) a reader "
            f"must see: {', '.join(warnings)}"
        )
    else:
        verdict = PORTFOLIO_PROCEED
        reason = (
            f"{len(sizes)} candidate(s) totalling {total:.1%} fit the book "
            f"on every portfolio-level rule"
        )

    return {
        "version": PORTFOLIO_DECISION_VERSION,
        "verdict": verdict,
        "reason": reason,
        "rules": rules,
        "veto": bool(vetoes),
        "veto_rule_ids": vetoes,
        "warning_rule_ids": warnings,
        "accepted": dict(sorted(sizes.items())),
        "total_size": total,
        "set_volatility_increase": joint,
        "fail_closed": PORTFOLIO_FAIL_CLOSED,
        "blocks_trades": PORTFOLIO_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _concentration_rule(concentration: Mapping[str, Any] | None) -> dict:
    """R3's finding, enforced. Absence triggers the rule."""
    if concentration is None:
        return _rule(
            "sector_concentration_breach",
            PORTFOLIO_FAIL_CLOSED,
            "no sector concentration report was supplied. Fail-closed: a "
            "portfolio approved without checking concentration is not the "
            "same as one checked and found clean",
        )
    if not isinstance(concentration, Mapping):
        raise PortfolioDecisionError("the concentration report must be a mapping")
    flags = [
        f
        for f in (concentration.get("flags") or [])
        if isinstance(f, Mapping) and f.get("sector") != "*"
    ]
    breaching = [
        f
        for f in flags
        if f.get("value") is not None
        and float(f["value"]) >= float(f.get("threshold", SECTOR_WEIGHT_REVIEW))
    ]
    if breaching:
        worst = max(breaching, key=lambda f: float(f["value"]))
        return _rule(
            "sector_concentration_breach",
            True,
            f"{worst.get('sector')} holds {float(worst['value']):.1%} on the "
            f"{worst.get('basis')} basis, at or beyond its "
            f"{float(worst.get('threshold', SECTOR_WEIGHT_REVIEW)):.0%} "
            f"review threshold",
        )
    return _rule(
        "sector_concentration_breach",
        False,
        "no sector breaches its review threshold",
    )


def _stress_rule(stress: Mapping[str, Any] | None) -> dict:
    """R6's finding, enforced. Absence triggers the rule."""
    if stress is None:
        return _rule(
            "stress_diversification_collapse",
            PORTFOLIO_FAIL_CLOSED,
            "no stress report was supplied. Fail-closed: MEASURED on 76 real "
            "tickers, correlation roughly doubles under the worst decile and "
            "diversification falls 38.1%, so an unstressed approval describes "
            "only calm markets",
        )
    if not isinstance(stress, Mapping):
        raise PortfolioDecisionError("the stress report must be a mapping")
    if stress.get("status") == "NOT_EVALUATED":
        return _rule(
            "stress_diversification_collapse",
            PORTFOLIO_FAIL_CLOSED,
            f"stress could not be evaluated: {stress.get('reason')}. "
            f"Fail-closed: not enough stress to measure is not an absence of "
            f"stress",
        )
    material = list(stress.get("material") or [])
    if material:
        scenarios = stress.get("scenarios") or {}
        worst = max(
            (
                (n, float(s.get("diversification_lost") or 0.0))
                for n, s in scenarios.items()
                if isinstance(s, Mapping) and n in material
            ),
            key=lambda pair: pair[1],
            default=(material[0], 0.0),
        )
        return _rule(
            "stress_diversification_collapse",
            True,
            f"{worst[0]} costs {worst[1]:.1%} of the portfolio's "
            f"diversification, at or beyond the "
            f"{STRESS_MATERIAL_DEGRADATION:.0%} threshold",
        )
    return _rule(
        "stress_diversification_collapse",
        False,
        "no stress scenario materially degrades diversification",
    )


_NOTE = (
    "R7 is the only module that blocks trades; R1, R3, R4, R5 and R6 each "
    "defer here. MEASURED, six individually-approved trades - every one "
    "volatility-reducing in isolation - summed to 120% of the book and would "
    "have raised portfolio volatility +44.57% executed together."
)


def decision_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a portfolio decision. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    if not report.get("blocks_trades"):
        problems.append(
            "R7 does not claim to block trades — every other Sprint R module "
            "deferred its refusal here, so if R7 cannot refuse, nothing can"
        )
    if not report.get("fail_closed"):
        problems.append(
            "R7 is not fail-closed — a portfolio check that passes when its "
            "evidence is absent manufactures approval out of nothing"
        )

    verdict = report.get("verdict")
    if verdict not in PORTFOLIO_VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")
    if not str(report.get("reason") or "").strip():
        problems.append("verdict with no reason")

    rules = report.get("rules") or []
    seen = set()
    for rule in rules:
        if not isinstance(rule, Mapping):
            problems.append("a rule is not a mapping")
            continue
        rule_id = rule.get("rule_id")
        seen.add(rule_id)
        if rule_id not in PORTFOLIO_RULES:
            problems.append(f"unknown rule {rule_id!r}")
            continue
        if rule.get("severity") != PORTFOLIO_RULES[rule_id]["severity"]:
            problems.append(
                f"{rule_id}: severity {rule.get('severity')!r} disagrees with "
                f"the declared policy"
            )
        if not str(rule.get("detail") or "").strip():
            problems.append(f"{rule_id}: triggered rule with no detail")
        if not str(rule.get("measurement") or "").strip():
            problems.append(
                f"{rule_id}: rule carries no measurement — a rule that cannot "
                f"say what it enforces is an assertion"
            )
    for rule_id in PORTFOLIO_RULES:
        if rule_id not in seen:
            problems.append(f"rule {rule_id!r} was not evaluated")

    # A VETO MUST REFUSE.
    vetoes = report.get("veto_rule_ids") or []
    if vetoes and verdict != PORTFOLIO_NO_TRADE:
        problems.append(
            f"a veto fired ({vetoes}) but the verdict is {verdict!r} — a veto "
            f"that does not refuse is not a veto"
        )
    if report.get("veto") != bool(vetoes):
        problems.append("the veto flag disagrees with the triggered veto rules")
    if verdict == PORTFOLIO_NO_TRADE and not vetoes:
        problems.append(
            "the set was refused with no veto rule named — a refusal must say "
            "which rule refused"
        )

    # WARNINGS NEVER SUM INTO A REFUSAL.
    warnings = report.get("warning_rule_ids") or []
    if warnings and not vetoes and verdict == PORTFOLIO_NO_TRADE:
        problems.append(
            f"{len(warnings)} warning(s) produced a refusal with no veto — "
            f"counting warnings until they become a veto invents a threshold "
            f"nobody measured"
        )

    # THE SHAPE RULE: an accepted set exists IFF something was accepted.
    accepted = report.get("accepted")
    if verdict in (PORTFOLIO_PROCEED, PORTFOLIO_REDUCED) and not accepted:
        problems.append(f"{verdict} with nothing accepted")
    if verdict == PORTFOLIO_NOT_EVALUATED and accepted:
        problems.append("NOT_EVALUATED carried an accepted set")

    total = report.get("total_size")
    if accepted and total is not None:
        expected = sum(float(v) for v in accepted.values())
        if abs(expected - float(total)) > 1e-6:
            problems.append(
                f"total_size {total} does not equal the accepted sizes "
                f"({expected})"
            )
    return problems


def render_decision(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines: the verdict, then every rule."""
    lines = [
        f"  {report.get('verdict')}: {report.get('reason')}",
    ]
    for rule in report.get("rules") or []:
        if not isinstance(rule, Mapping):
            continue
        mark = "TRIGGERED" if rule.get("triggered") else "ok"
        lines.append(
            f"      [{str(rule.get('severity')):4s}] "
            f"{str(rule.get('rule_id')):34s} {mark:10s} "
            f"({rule.get('source')})"
        )
    return lines
