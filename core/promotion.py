"""Promotion gates (Sprint M5) — the checklist a candidate must pass to serve.

`evaluate_promotion(...)` runs five checks in order and returns a verdict that
is a REFUSAL or an approval, never a score. Sprint L (the learning loop)
states this as its precondition: "promotion requires out-of-sample comparison,
drift checks, reproducibility, and a release gate."

**The gap was narrower than the sprint doc implied, and that was measured.**
`ModelRegistry.promote()` already refuses a candidate without an OOS
comparison and without a named human approver. Three of the five checks were
enforced nowhere:

    [x] OOS beats the incumbent        already in promote()
    [x] human approval recorded        already in promote()
    [ ] no veto-rate regression
    [ ] drift check clean
    [ ] manifest complete

So this module adds the three missing checks and sequences all five. It does
not reimplement the two that work (W5) — it calls them.

**Every check is a refusal, not a score.** A checklist that produces a number
invites "close enough". Each check returns PASS, FAIL or NOT_EVALUATED, and
anything other than PASS on a required check stops the promotion.

**NOT_EVALUATED is not PASS.** A drift check that could not run has not found
the candidate clean — it has found nothing. Collapsing the two would promote
on absent evidence, which is the same failure mode F6's NOT_WIRED and F7's
UNMEASURABLE each guard against in their own surfaces.

**The manifest check runs FIRST.** A candidate that cannot be reproduced
should be rejected before anyone spends time evaluating its metrics.

**A veto rate that FALLS is also a regression.** The obvious check catches a
candidate that vetoes more; a candidate that vetoes far less has not
necessarily improved, it may have stopped checking. Both directions fail.

**History is immutable.** Promotion never rewrites the model version recorded
against a past decision; the append-only audit log guarantees it, and
`history_immutable_problems` verifies the guarantee rather than assuming it.
"""

from __future__ import annotations

import logging

from core.config import (
    PROMO_CHECK_APPROVAL,
    PROMO_CHECK_DRIFT,
    PROMO_CHECK_MANIFEST,
    PROMO_CHECK_OOS,
    PROMO_CHECK_REGRESSION,
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
    PROMOTION_CHECKS,
    PROMOTION_CONTRACT_VERSION,
    PROMOTION_GATE_VERSION,
    PROMOTION_HISTORY_IMMUTABLE,
    PROMOTION_MAX_VETO_RATE_DECREASE,
    PROMOTION_MAX_VETO_RATE_INCREASE,
    PROMOTION_MIN_DECISIONS_FOR_REGRESSION,
    PROMOTION_PSI_FAIL,
    PROMOTION_PSI_WARN,
    PROMOTION_REQUIRED_CHECKS,
    PROMOTION_REQUIRED_MANIFEST_FIELDS,
)
from core.model_registry import approver_problems, oos_comparison_problems
from core.monitoring import score_drift_psi, veto_rate_by_rule

LOGGER = logging.getLogger("core.promotion")


class PromotionError(ValueError):
    """Raised when a promotion request violates the M5 contract."""


def _check(name: str, outcome: str, reason: str, **detail) -> dict:
    """One checklist row. A non-PASS row always says why."""
    if outcome != PROMO_PASS and not reason:
        raise PromotionError(f"{name}: a {outcome} check must state a reason")
    return {"check": name, "outcome": outcome, "reason": reason, **detail}


# ---------------------------------------------------------------------------
# The five checks
# ---------------------------------------------------------------------------


def manifest_check(candidate) -> dict:
    """Can this promotion be reproduced and audited afterwards?

    Runs first: a candidate missing its dataset hash or code commit should be
    rejected before anyone evaluates its metrics.
    """
    if candidate is None:
        return _check(
            PROMO_CHECK_MANIFEST, PROMO_FAIL,
            "no candidate entry was supplied",
        )

    def read(field):
        if isinstance(candidate, dict):
            return candidate.get(field)
        return getattr(candidate, field, None)

    missing = [
        field for field in PROMOTION_REQUIRED_MANIFEST_FIELDS
        if read(field) in (None, "", {}, [])
    ]
    if missing:
        return _check(
            PROMO_CHECK_MANIFEST, PROMO_FAIL,
            (
                f"manifest incomplete: {', '.join(missing)} — without these a "
                f"promoted model cannot be reproduced from its own record"
            ),
            missing=missing,
        )
    return _check(
        PROMO_CHECK_MANIFEST, PROMO_PASS,
        "",
        fields=len(PROMOTION_REQUIRED_MANIFEST_FIELDS),
    )


def oos_check(
    comparison: dict | None,
    candidate_version: str,
    incumbent_version: str | None,
) -> dict:
    """Does the candidate beat the incumbent out of sample?

    Composed from `model_registry.oos_comparison_problems` — the same function
    `promote()` enforces, so the script and the registry cannot disagree.
    """
    problems = oos_comparison_problems(
        comparison, candidate_version, incumbent_version
    )
    if problems:
        return _check(
            PROMO_CHECK_OOS, PROMO_FAIL, "; ".join(problems),
            problems=problems,
        )
    return _check(
        PROMO_CHECK_OOS, PROMO_PASS, "",
        metric=(comparison or {}).get("metric"),
        incumbent=incumbent_version,
    )


def regression_check(
    incumbent_decisions: list[dict] | None,
    candidate_decisions: list[dict] | None,
) -> dict:
    """Has decision quality regressed, in EITHER direction?

    A candidate that vetoes far MORE is not safer, it is less useful. One that
    vetoes far LESS may have stopped checking — the W2 rules exist to refuse
    bad decisions, so a large drop is also a regression.
    """
    incumbent_decisions = incumbent_decisions or []
    candidate_decisions = candidate_decisions or []

    if (
        len(incumbent_decisions) < PROMOTION_MIN_DECISIONS_FOR_REGRESSION
        or len(candidate_decisions) < PROMOTION_MIN_DECISIONS_FOR_REGRESSION
    ):
        return _check(
            PROMO_CHECK_REGRESSION, PROMO_NOT_EVALUATED,
            (
                f"fewer than {PROMOTION_MIN_DECISIONS_FOR_REGRESSION} decisions "
                f"on one side ({len(incumbent_decisions)} incumbent, "
                f"{len(candidate_decisions)} candidate); a verdict here would "
                f"be a reading of noise"
            ),
            incumbent_decisions=len(incumbent_decisions),
            candidate_decisions=len(candidate_decisions),
        )

    before = veto_rate_by_rule(incumbent_decisions)
    after = veto_rate_by_rule(candidate_decisions)
    if before.get("verdict") != "computed" or after.get("verdict") != "computed":
        return _check(
            PROMO_CHECK_REGRESSION, PROMO_NOT_EVALUATED,
            (
                "veto metadata is missing from the decision records, so no "
                "regression could be measured"
            ),
            incumbent=before.get("reason"),
            candidate=after.get("reason"),
        )

    rules = sorted(set(before["rates"]) | set(after["rates"]))
    regressions: list[str] = []
    deltas: dict[str, float] = {}
    for rule in rules:
        old = float(before["rates"].get(rule, 0.0))
        new = float(after["rates"].get(rule, 0.0))
        delta = round(new - old, 6)
        deltas[rule] = delta
        if delta > PROMOTION_MAX_VETO_RATE_INCREASE:
            regressions.append(
                f"{rule}: veto rate rose {old:.3f} -> {new:.3f} "
                f"(+{delta:.3f} > {PROMOTION_MAX_VETO_RATE_INCREASE})"
            )
        elif -delta > PROMOTION_MAX_VETO_RATE_DECREASE:
            regressions.append(
                f"{rule}: veto rate FELL {old:.3f} -> {new:.3f} "
                f"({delta:.3f}); a candidate that stops refusing bad decisions "
                f"has not improved, it has stopped checking"
            )

    if regressions:
        return _check(
            PROMO_CHECK_REGRESSION, PROMO_FAIL, "; ".join(regressions),
            deltas=deltas,
        )
    return _check(
        PROMO_CHECK_REGRESSION, PROMO_PASS, "", deltas=deltas, rules=len(rules),
    )


def drift_check(
    reference_scores: list[float] | None,
    candidate_scores: list[float] | None,
) -> dict:
    """Has the score distribution shifted beyond tolerance?

    MEASURED over 400 synthetic scores: an unshifted distribution scores PSI
    0.013, a +1.7-point shift scores 2.854. The thresholds discriminate.
    """
    reference_scores = list(reference_scores or [])
    candidate_scores = list(candidate_scores or [])
    if not reference_scores or not candidate_scores:
        return _check(
            PROMO_CHECK_DRIFT, PROMO_NOT_EVALUATED,
            "no score distributions were supplied, so no drift was measured",
        )

    try:
        result = score_drift_psi(reference_scores, candidate_scores)
    except Exception as exc:
        # ABSORBS: a PSI computation that cannot be made — too few scores,
        # degenerate bins, mismatched shapes. Reports NOT_EVALUATED with the
        # exception type, never "no drift": an uncomputable drift check is
        # ignorance, and treating it as a pass is the exact failure X10 exists
        # to prevent.
        return _check(
            PROMO_CHECK_DRIFT, PROMO_NOT_EVALUATED,
            f"drift could not be computed: {type(exc).__name__}: {exc}",
        )

    psi = result.get("psi")
    if psi is None:
        return _check(
            PROMO_CHECK_DRIFT, PROMO_NOT_EVALUATED,
            result.get("reason") or "the drift check produced no PSI",
        )
    psi = float(psi)
    if psi >= PROMOTION_PSI_FAIL:
        return _check(
            PROMO_CHECK_DRIFT, PROMO_FAIL,
            (
                f"PSI {psi:.4f} at or above the {PROMOTION_PSI_FAIL} fail "
                f"threshold — the candidate scores a materially different "
                f"distribution from the reference"
            ),
            psi=round(psi, 6),
        )
    return _check(
        PROMO_CHECK_DRIFT, PROMO_PASS,
        "",
        psi=round(psi, 6),
        warn=psi >= PROMOTION_PSI_WARN,
    )


def approval_check(approved_by: str | None, approved_at: str | None) -> dict:
    """Is a named human accountable for this promotion?

    Composed from `model_registry.approver_problems` — the same function
    `promote()` enforces.
    """
    problems = approver_problems(approved_by)
    if not str(approved_at or "").strip():
        problems.append("approved_at is required")
    if problems:
        return _check(
            PROMO_CHECK_APPROVAL, PROMO_FAIL, "; ".join(problems),
            problems=problems,
        )
    return _check(
        PROMO_CHECK_APPROVAL, PROMO_PASS, "",
        approved_by=str(approved_by), approved_at=str(approved_at),
    )


# ---------------------------------------------------------------------------
# History immutability
# ---------------------------------------------------------------------------


def history_immutable_problems(
    before: list[dict] | None, after: list[dict] | None
) -> list[str]:
    """Verify a promotion did not rewrite any past decision's model version.

    Checked rather than assumed: the append-only audit log is the guarantee,
    and this is what proves the guarantee held across a promotion.
    """
    problems: list[str] = []
    if not PROMOTION_HISTORY_IMMUTABLE:
        return ["history immutability has been disabled in config"]

    indexed = {
        str(row.get("decision_id")): row for row in (before or [])
        if row.get("decision_id")
    }
    for row in after or []:
        key = str(row.get("decision_id") or "")
        original = indexed.get(key)
        if original is None:
            continue
        for field in ("model_version", "score", "as_of"):
            if field in original and original.get(field) != row.get(field):
                problems.append(
                    f"decision {key}: {field} changed "
                    f"{original.get(field)!r} -> {row.get(field)!r}; a "
                    f"promotion must never rewrite a past decision"
                )
    missing = set(indexed) - {
        str(row.get("decision_id") or "") for row in (after or [])
    }
    if missing:
        problems.append(
            f"{len(missing)} historical decision(s) disappeared after "
            f"promotion: {', '.join(sorted(missing)[:5])}"
        )
    return problems


# ---------------------------------------------------------------------------
# The checklist
# ---------------------------------------------------------------------------


def evaluate_promotion(
    candidate,
    *,
    incumbent_version: str | None = None,
    oos_comparison: dict | None = None,
    approved_by: str | None = None,
    approved_at: str | None = None,
    incumbent_decisions: list[dict] | None = None,
    candidate_decisions: list[dict] | None = None,
    reference_scores: list[float] | None = None,
    candidate_scores: list[float] | None = None,
) -> dict:
    """Run the five checks in order and return a verdict.

    The verdict is a refusal or an approval, never a score: any check that is
    not PASS blocks promotion, including NOT_EVALUATED.
    """
    candidate_version = (
        candidate.get("model_version") if isinstance(candidate, dict)
        else getattr(candidate, "model_version", None)
    )

    checks: dict[str, dict] = {
        PROMO_CHECK_MANIFEST: manifest_check(candidate),
        PROMO_CHECK_OOS: oos_check(
            oos_comparison, str(candidate_version or ""), incumbent_version
        ),
        PROMO_CHECK_REGRESSION: regression_check(
            incumbent_decisions, candidate_decisions
        ),
        PROMO_CHECK_DRIFT: drift_check(reference_scores, candidate_scores),
        PROMO_CHECK_APPROVAL: approval_check(approved_by, approved_at),
    }

    blocking = [
        name for name in PROMOTION_REQUIRED_CHECKS
        if checks[name]["outcome"] != PROMO_PASS
    ]

    return {
        "candidate_version": candidate_version,
        "incumbent_version": incumbent_version,
        "approved": not blocking,
        "blocking": blocking,
        "check_order": list(PROMOTION_CHECKS),
        "checks": {name: checks[name] for name in PROMOTION_CHECKS},
        "passed": [n for n in PROMOTION_CHECKS if checks[n]["outcome"] == PROMO_PASS],
        "failed": [n for n in PROMOTION_CHECKS if checks[n]["outcome"] == PROMO_FAIL],
        "not_evaluated": [
            n for n in PROMOTION_CHECKS
            if checks[n]["outcome"] == PROMO_NOT_EVALUATED
        ],
        "reason": (
            "every required check passed"
            if not blocking
            else f"blocked by: {', '.join(blocking)}"
        ),
        "gate_version": PROMOTION_GATE_VERSION,
        "contract_version": PROMOTION_CONTRACT_VERSION,
    }


def promotion_problems(verdict: dict) -> list[str]:
    """Validate a verdict against the M5 contract."""
    problems: list[str] = []
    if not isinstance(verdict, dict):
        return ["verdict must be a dict"]

    for field in ("gate_version", "contract_version", "reason"):
        if not verdict.get(field):
            problems.append(f"verdict field {field!r} missing/empty")

    checks = verdict.get("checks") or {}
    if list(checks) != list(PROMOTION_CHECKS):
        problems.append(
            "the verdict does not report every check in declared order — a "
            "missing row reads as an oversight where NOT_EVALUATED reads as a "
            "fact"
        )

    for name, row in checks.items():
        outcome = row.get("outcome")
        if outcome not in (PROMO_PASS, PROMO_FAIL, PROMO_NOT_EVALUATED):
            problems.append(f"{name}: unknown outcome {outcome!r}")
        if outcome != PROMO_PASS and not row.get("reason"):
            problems.append(f"{name}: a {outcome} check does not say why")

    # The load-bearing property: anything but PASS blocks.
    non_pass = [
        name for name, row in checks.items()
        if name in PROMOTION_REQUIRED_CHECKS and row.get("outcome") != PROMO_PASS
    ]
    if non_pass and verdict.get("approved"):
        problems.append(
            f"approved although {', '.join(sorted(non_pass))} did not PASS — "
            f"a NOT_EVALUATED check has not found the candidate clean, it has "
            f"found nothing"
        )
    if not non_pass and not verdict.get("approved"):
        problems.append("every check passed but the verdict is not approved")
    if sorted(verdict.get("blocking") or []) != sorted(non_pass):
        problems.append("the blocking list does not match the non-PASS checks")
    return problems


def render_checklist(verdict: dict) -> list[dict]:
    """One reading row per check, in declared order. Never blank."""
    rows: list[dict] = []
    for name in verdict.get("check_order") or ():
        row = (verdict.get("checks") or {}).get(name) or {}
        rows.append(
            {
                "check": name,
                "outcome": row.get("outcome"),
                "blocking": name in (verdict.get("blocking") or []),
                "reason": row.get("reason") or "",
            }
        )
    return rows
