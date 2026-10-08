"""L5 controlled incremental learning — the chain, sequenced and bounded.

"New data → candidate update → shadow evaluation → drift testing → OOS
validation → promotion gate → human approval → new champion. Never
auto-replace the production champion daily."

**Most of this chain already exists, and this module does not rebuild it.**
Shadow roles are M7's, drift/OOS/approval are M5's, training is the M-sprint's,
collection is the L-sprint's. Rebuilding any of them would be the split-brain
W5 forbids. What was missing was two things.

**The last sentence had no implementation.** MEASURED, the words cooldown,
last_promoted, min_days, interval, elapsed and cadence appear ZERO times across
`core/promotion.py` and `core/model_registry.py`. `crown_champion` checks role,
status and approval but never looks at how long the outgoing champion has
served — DEMONSTRATED by crowning four champions inside fifteen minutes, every
one accepted.

**Why a bound is needed at all, measured.** Two models with identical true
skill: the challenger looks better ~48% of the time at *any* sample size. So
"promote whatever is better on the evidence so far" churns the champion roughly
every other evaluation forever, on pure noise, while a real 3-point edge is
detected only 64% of the time at n=100. Evidence separates them; no threshold
does.

**Nothing ran the chain end to end.** Each stage was individually correct and
collectively unsequenced, so no single call could answer "may this candidate
become champion today, and if not, which stage stopped it".

**A stage that could not run blocks exactly as a failure does** — inherited
from M5's NOT_EVALUATED. Absence of evidence is not evidence of safety.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.config import (
    CHAMPION_MIN_TENURE_DAYS,
    CHAMPION_TENURE_ALLOWS_ROLLBACK,
    CHAMPION_TENURE_MEASURED_FROM,
    CHAMPION_TENURE_VERSION,
    L5_AUTO_PROMOTE,
    L5_OWNED_STAGES,
    L5_PIPELINE_VERSION,
    L5_STAGE_APPROVAL,
    L5_STAGE_CANDIDATE,
    L5_STAGE_CHAMPION,
    L5_STAGE_DATA,
    L5_STAGE_DRIFT,
    L5_STAGE_GATE,
    L5_STAGE_OOS,
    L5_STAGE_SHADOW,
    L5_STAGE_TENURE,
    L5_STAGES,
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
    SHADOW_MIN_OBSERVATIONS,
)


class ControlledLearningError(ValueError):
    """Raised when a pipeline request is structurally invalid."""


def _stage(name: str, outcome: str, reason: str, **detail) -> dict:
    if name not in L5_STAGES:
        raise ControlledLearningError(f"{name!r} is not an L5 stage")
    if outcome not in (PROMO_PASS, PROMO_FAIL, PROMO_NOT_EVALUATED):
        raise ControlledLearningError(f"{outcome!r} is not a valid outcome")
    record = {"stage": name, "outcome": outcome, "reason": reason}
    record.update(detail)
    return record


def _parse(moment: str | None) -> datetime | None:
    """Parse an ISO timestamp, tolerating a trailing Z and naive forms."""
    text = str(moment or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def tenure_days(crowned_at: str | None, now: str | None) -> float | None:
    """Days the incumbent has served, or None when it cannot be determined."""
    start, end = _parse(crowned_at), _parse(now)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 86400.0


def tenure_check(
    incumbent_crowned_at: str | None,
    now: str | None,
    *,
    is_rollback: bool = False,
    min_days: float = CHAMPION_MIN_TENURE_DAYS,
) -> dict:
    """May the incumbent be replaced yet?

    THE STAGE L5 OWNS. Measured from the OUTGOING champion's crowning, because
    measuring from candidate readiness would let a queue of candidates replace
    the champion in sequence, each "ready" on a different day.

    A rollback is exempt: withdrawing a failing champion is not the churn this
    guard exists to prevent, and a guard that trapped a broken model in
    production would be worse than no guard.
    """
    if is_rollback:
        if not CHAMPION_TENURE_ALLOWS_ROLLBACK:  # pragma: no cover - config-guarded
            raise ControlledLearningError("rollback exemption is disabled")
        return _stage(
            L5_STAGE_TENURE,
            PROMO_PASS,
            "safety withdrawal — the tenure bound never traps a failing "
            "champion in production",
            rollback=True,
            min_days=min_days,
        )

    # NO INCUMBENT IS NOT A TENURE VIOLATION. The first champion of a contract
    # has nothing to displace, so the bound has nothing to measure. Refusing
    # here would make the system unable to ever start.
    if not str(incumbent_crowned_at or "").strip():
        return _stage(
            L5_STAGE_TENURE,
            PROMO_PASS,
            "no incumbent champion to displace",
            min_days=min_days,
        )

    served = tenure_days(incumbent_crowned_at, now)
    if served is None:
        # UNPARSEABLE IS NOT PASSING. A timestamp that cannot be read has not
        # shown the bound is satisfied; it has shown nothing.
        return _stage(
            L5_STAGE_TENURE,
            PROMO_NOT_EVALUATED,
            f"could not read the incumbent's crowning time "
            f"({incumbent_crowned_at!r}) or the current time ({now!r}), so "
            f"tenure cannot be established",
            min_days=min_days,
        )

    if served < 0:
        return _stage(
            L5_STAGE_TENURE,
            PROMO_FAIL,
            f"the incumbent was crowned {abs(served):.2f} days in the FUTURE "
            f"relative to {now!r} — a clock or an argument is wrong, and "
            f"passing would let a back-dated crowning bypass the bound",
            served_days=round(served, 4),
            min_days=min_days,
        )

    if served < min_days:
        return _stage(
            L5_STAGE_TENURE,
            PROMO_FAIL,
            f"the incumbent has served {served:.2f} of the {min_days} days "
            f"required. MEASURED, two models of identical skill trade places "
            f"~48% of evaluations at any sample size, so replacing on the "
            f"evidence so far churns the champion on noise",
            served_days=round(served, 4),
            min_days=min_days,
            remaining_days=round(min_days - served, 4),
        )

    return _stage(
        L5_STAGE_TENURE,
        PROMO_PASS,
        f"the incumbent has served {served:.2f} days, at or beyond the "
        f"{min_days}-day bound",
        served_days=round(served, 4),
        min_days=min_days,
    )


def data_check(observations: int | None, *, as_of: str | None = None) -> dict:
    """Is there new data to learn from at all?"""
    if observations is None:
        return _stage(
            L5_STAGE_DATA,
            PROMO_NOT_EVALUATED,
            "no observation count supplied, so it is unknown whether any new "
            "data arrived",
        )
    if int(observations) <= 0:
        return _stage(
            L5_STAGE_DATA,
            PROMO_FAIL,
            "no new observations — a retrain on unchanged data produces a "
            "candidate that differs only by noise",
            observations=int(observations),
        )
    return _stage(
        L5_STAGE_DATA,
        PROMO_PASS,
        f"{int(observations)} new observation(s) since the incumbent's cutoff",
        observations=int(observations),
        as_of=as_of,
    )


def candidate_check(candidate: Any) -> dict:
    """Does a candidate exist and identify itself?"""
    version = str(getattr(candidate, "model_version", "") or "").strip()
    if not candidate or not version:
        return _stage(
            L5_STAGE_CANDIDATE,
            PROMO_FAIL,
            "no candidate model was supplied",
        )
    return _stage(
        L5_STAGE_CANDIDATE,
        PROMO_PASS,
        f"candidate {version} present",
        model_version=version,
    )


def shadow_check(
    observations: int | None, *, min_observations: int = SHADOW_MIN_OBSERVATIONS
) -> dict:
    """Has the candidate earned its way out of shadow?

    DELEGATES the threshold to M7's SHADOW_MIN_OBSERVATIONS rather than
    inventing a second one.
    """
    if observations is None:
        return _stage(
            L5_STAGE_SHADOW,
            PROMO_NOT_EVALUATED,
            "shadow observations were not reported, so the candidate has not "
            "been shown to have run in shadow at all",
            min_observations=min_observations,
        )
    count = int(observations)
    if count < min_observations:
        return _stage(
            L5_STAGE_SHADOW,
            PROMO_FAIL,
            f"{count} shadow observation(s), below the {min_observations} "
            f"required — an unmeasured model has earned nothing",
            observations=count,
            min_observations=min_observations,
        )
    return _stage(
        L5_STAGE_SHADOW,
        PROMO_PASS,
        f"{count} shadow observation(s)",
        observations=count,
        min_observations=min_observations,
    )


_DELEGATED = {
    L5_STAGE_DRIFT: "drift_clean",
    L5_STAGE_OOS: "oos_beats_incumbent",
    L5_STAGE_APPROVAL: "human_approval",
}


def _from_checklist(verdict: Mapping[str, Any], stage: str) -> dict:
    """Lift one M5 check into an L5 stage, preserving its outcome verbatim.

    The outcome is NEVER recomputed here. L5 sequences the chain; M5 decides
    what passes.
    """
    wanted = _DELEGATED[stage]
    # M5 keys `checks` BY CHECK NAME. An earlier version of this function
    # assumed a list and silently reported NOT_EVALUATED for every real
    # verdict — it read the shape it expected rather than the shape M5 emits.
    checks = (verdict or {}).get("checks") or {}
    check = checks.get(wanted) if isinstance(checks, Mapping) else None
    if check is None and not isinstance(checks, Mapping):
        for entry in checks:
            if isinstance(entry, Mapping) and entry.get("check") == wanted:
                check = entry
                break
    if isinstance(check, Mapping):
        reason = str(check.get("reason") or "")
        outcome = str(check.get("outcome") or PROMO_NOT_EVALUATED)
        return _stage(
            stage,
            outcome,
            reason or f"{wanted} reported {outcome} with no reason recorded",
            delegated_to=wanted,
        )
    return _stage(
        stage,
        PROMO_NOT_EVALUATED,
        f"the promotion checklist reported no {wanted!r} check, so this stage "
        f"has found nothing rather than found the candidate clean",
        delegated_to=wanted,
    )


def evaluate_chain(
    *,
    candidate: Any = None,
    new_observations: int | None = None,
    shadow_observations: int | None = None,
    promotion_verdict: Mapping[str, Any] | None = None,
    incumbent_crowned_at: str | None = None,
    now: str | None = None,
    is_rollback: bool = False,
    as_of: str | None = None,
) -> dict:
    """Run the whole L5 chain and report which stage, if any, stopped it.

    Returns every stage in order with its outcome. `may_promote` is true only
    when every stage reached PASS — and even then promotion is a human act,
    never something this function performs.
    """
    stages: list[dict] = [
        data_check(new_observations, as_of=as_of),
        candidate_check(candidate),
        shadow_check(shadow_observations),
    ]

    if promotion_verdict is None:
        for stage_name in (L5_STAGE_DRIFT, L5_STAGE_OOS, L5_STAGE_APPROVAL):
            stages.append(
                _stage(
                    stage_name,
                    PROMO_NOT_EVALUATED,
                    "no promotion checklist was supplied, so this stage could "
                    "not run — which blocks exactly as a failure does",
                    delegated_to=_DELEGATED[stage_name],
                )
            )
        gate_outcome, gate_reason = (
            PROMO_NOT_EVALUATED,
            "no promotion checklist was supplied",
        )
    else:
        stages.append(_from_checklist(promotion_verdict, L5_STAGE_DRIFT))
        stages.append(_from_checklist(promotion_verdict, L5_STAGE_OOS))
        # M5's flag is `approved`, not `promoted`.
        promoted = bool((promotion_verdict or {}).get("approved"))
        gate_outcome = PROMO_PASS if promoted else PROMO_FAIL
        gate_reason = (
            "the M5 promotion gate passed every required check"
            if promoted
            else "the M5 promotion gate refused: "
            + ", ".join(
                str(name)
                for name in (promotion_verdict or {}).get("blocking") or []
            )
        )

    # The gate stage sits between OOS and approval in L5_STAGES order.
    gate_stage = _stage(L5_STAGE_GATE, gate_outcome, gate_reason, delegated_to="m5")
    stages.append(gate_stage)
    if promotion_verdict is not None:
        stages.append(_from_checklist(promotion_verdict, L5_STAGE_APPROVAL))

    stages.append(
        tenure_check(incumbent_crowned_at, now, is_rollback=is_rollback)
    )

    ordered = sorted(stages, key=lambda item: L5_STAGES.index(item["stage"]))
    blocking = [item for item in ordered if item["outcome"] != PROMO_PASS]
    may_promote = not blocking

    ordered.append(
        _stage(
            L5_STAGE_CHAMPION,
            PROMO_PASS if may_promote else PROMO_FAIL,
            (
                "every stage passed — a human may now crown this candidate"
                if may_promote
                else f"blocked at {blocking[0]['stage']}: {blocking[0]['reason']}"
            ),
        )
    )

    return {
        "version": L5_PIPELINE_VERSION,
        "tenure_version": CHAMPION_TENURE_VERSION,
        "candidate": str(getattr(candidate, "model_version", "") or "") or None,
        "stages": ordered,
        "may_promote": may_promote,
        "blocked_at": blocking[0]["stage"] if blocking else None,
        "blocking_reasons": [
            f"{item['stage']}: {item['reason']}" for item in blocking
        ],
        # PROMOTION IS NEVER AUTOMATIC. may_promote says the evidence allows
        # it; a human still performs it. These are different statements and
        # the pipeline must not collapse them.
        "auto_promote": L5_AUTO_PROMOTE,
        "note": (
            "may_promote means the chain is clear, not that promotion has "
            "happened. Crowning is a human act."
        ),
    }


def chain_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a chain report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]

    stages = list(report.get("stages") or [])
    seen = [item.get("stage") for item in stages]

    missing = [name for name in L5_STAGES if name not in seen]
    if missing:
        problems.append(
            f"the chain skipped {missing} — every stage must reach a verdict, "
            f"because a stage that did not run has found nothing"
        )

    positions = [L5_STAGES.index(name) for name in seen if name in L5_STAGES]
    if positions != sorted(positions):
        problems.append("stages are out of order")

    for item in stages:
        if item.get("outcome") not in (PROMO_PASS, PROMO_FAIL, PROMO_NOT_EVALUATED):
            problems.append(f"{item.get('stage')}: unknown outcome")
        if not str(item.get("reason") or "").strip():
            problems.append(
                f"{item.get('stage')}: no reason given — an operator cannot "
                f"act on a bare verdict"
            )

    blocking = [
        item
        for item in stages
        if item.get("stage") != L5_STAGE_CHAMPION
        and item.get("outcome") != PROMO_PASS
    ]
    if blocking and report.get("may_promote"):
        problems.append(
            f"may_promote is true while {len(blocking)} stage(s) did not pass"
        )
    if report.get("auto_promote"):
        problems.append(
            "auto_promote is true — promotion must never be automatic"
        )
    return problems


def render_chain(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines, one per stage."""
    lines: list[str] = []
    for item in report.get("stages") or []:
        mark = {
            PROMO_PASS: "PASS",
            PROMO_FAIL: "FAIL",
            PROMO_NOT_EVALUATED: "NOT_EVALUATED",
        }.get(str(item.get("outcome")), "?")
        lines.append(f"  {str(item.get('stage')):20s} {mark:14s} {item.get('reason')}")
    return lines


def owned_stages() -> tuple[str, ...]:
    """The stages L5 implements rather than delegates."""
    return tuple(L5_OWNED_STAGES)


def delegated_stages() -> dict[str, str]:
    """Stage → the existing owner it is delegated to (W5)."""
    mapping = dict(_DELEGATED)
    mapping[L5_STAGE_GATE] = "core.promotion.evaluate_promotion"
    mapping[L5_STAGE_SHADOW] = "core.config.SHADOW_MIN_OBSERVATIONS (M7)"
    return mapping
