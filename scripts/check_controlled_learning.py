"""Gate: controlled incremental learning — the chain runs, and daily churn cannot.

"New data -> candidate update -> shadow evaluation -> drift testing -> OOS
validation -> promotion gate -> human approval -> new champion. Never
auto-replace the production champion daily."

MOST OF THIS CHAIN ALREADY EXISTED. L5 must SEQUENCE it, not rebuild it (W5):
drift, OOS and approval are M5's checks, shadow is M7's role model, training is
the M-sprint's. This gate verifies the delegation is real.

THE LAST SENTENCE HAD NO IMPLEMENTATION. MEASURED, cooldown/last_promoted/
min_days/interval/elapsed/cadence appear ZERO times across core/promotion.py
and core/model_registry.py, and four champions were crowned inside fifteen
minutes with every crowning accepted.

WHY A BOUND IS NEEDED, MEASURED: two models of IDENTICAL skill trade places
~48% of evaluations at any sample size, so "promote whatever looks better"
churns the champion forever on noise - 433 replacements per three years.

Verified to FAIL when any of these is reinjected:
  - the tenure bound removed or lowered to permit daily replacement
  - the bound raised until a real winner is delayed past usefulness
  - tenure measured from candidate readiness instead of the incumbent
  - a stage silently skipped
  - NOT_EVALUATED treated as passing
  - may_promote true while a stage did not pass
  - auto-promotion enabled
  - the promotion gate reimplemented instead of delegated
  - a rollback blocked by the tenure bound
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CHAMPION_MIN_TENURE_DAYS,
    CHAMPION_TENURE_ALLOWS_ROLLBACK,
    CHAMPION_TENURE_MEASURED_FROM,
    L5_AUTO_PROMOTE,
    L5_OWNED_STAGES,
    L5_STAGE_GATE,
    L5_STAGE_TENURE,
    L5_STAGES,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
)
from core.controlled_learning import (  # noqa: E402
    chain_problems,
    delegated_stages,
    evaluate_chain,
    owned_stages,
    tenure_check,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


class Candidate:
    model_version = "cand-1"


def churn_from_noise(min_days: int, days: int = 1080, reps: int = 120) -> float:
    """Champion replacements when every candidate is a coin flip."""
    rng = random.Random(11)
    counts = []
    for _ in range(reps):
        last = -(10**9)
        changes = 0
        for day in range(days):
            if rng.random() < 0.48 and (day - last) >= min_days:
                last = day
                changes += 1
        counts.append(changes)
    return statistics.mean(counts)


def main() -> int:
    # 1. THE MEASUREMENT JUSTIFYING THE BOUND still holds: an unbounded rule
    #    churns the champion on pure noise.
    unbounded = churn_from_noise(0)
    bounded = churn_from_noise(int(CHAMPION_MIN_TENURE_DAYS))
    check(
        unbounded > 300,
        f"an unbounded policy produced only {unbounded:.0f} replacements per "
        f"three years — the churn measurement no longer reproduces",
    )
    check(
        bounded < unbounded / 5,
        f"the {CHAMPION_MIN_TENURE_DAYS}-day bound cut churn only from "
        f"{unbounded:.0f} to {bounded:.0f}; it is not doing its job",
    )

    # 2. THE BOUND IS A REAL BOUND.
    check(
        CHAMPION_MIN_TENURE_DAYS >= 2,
        "a bound under 2 days permits daily replacement, which the sprint "
        "forbids in as many words",
    )
    check(
        CHAMPION_TENURE_MEASURED_FROM == "incumbent_crowned_at",
        "tenure must be measured from the OUTGOING champion's crowning, or a "
        "queue of candidates each 'ready' on a different day walks past it",
    )

    # 3. THE EXPLOIT THAT STARTED L5 IS REFUSED.
    same_day = tenure_check("2026-09-21T09:00:00Z", "2026-09-21T09:15:00Z")
    check(
        same_day["outcome"] != PROMO_PASS,
        "a champion crowned fifteen minutes ago could be replaced — this is "
        "the exact behaviour DEMONSTRATED before L5 and the reason it exists",
    )
    day_old = tenure_check("2026-09-20T00:00:00Z", "2026-09-21T00:00:00Z")
    check(
        day_old["outcome"] != PROMO_PASS,
        "a one-day-old champion could be replaced: 'never auto-replace the "
        "production champion daily' is unenforced",
    )
    seasoned = tenure_check("2026-01-01T00:00:00Z", "2026-09-21T00:00:00Z")
    check(
        seasoned["outcome"] == PROMO_PASS,
        "a champion of 263 days was still blocked — the bound is a wall, not "
        "a bound, and no model could ever be replaced",
    )

    # 4. A ROLLBACK IS NEVER BLOCKED. Trapping a failing champion in
    #    production would be worse than no guard at all.
    check(
        CHAMPION_TENURE_ALLOWS_ROLLBACK,
        "the tenure bound must never block a safety withdrawal",
    )
    rollback = tenure_check(
        "2026-09-21T09:00:00Z", "2026-09-21T09:15:00Z", is_rollback=True
    )
    check(
        rollback["outcome"] == PROMO_PASS,
        "a safety withdrawal was blocked by the tenure bound, trapping a "
        "failing champion in production",
    )

    # 5. A BACK-DATED CROWNING CANNOT BUY TENURE.
    future = tenure_check("2026-12-01T00:00:00Z", "2026-09-21T00:00:00Z")
    check(
        future["outcome"] != PROMO_PASS,
        "an incumbent crowned in the FUTURE satisfied the bound — a wrong "
        "clock or a forged timestamp walks straight through",
    )

    # 6. UNREADABLE IS NOT PASSING.
    unreadable = tenure_check("not-a-timestamp", "2026-09-21T00:00:00Z")
    check(
        unreadable["outcome"] == PROMO_NOT_EVALUATED,
        "an unparseable crowning time was treated as satisfying the bound; it "
        "has shown nothing, not that the bound is met",
    )

    # 7. EVERY STAGE REACHES A VERDICT, IN ORDER.
    report = evaluate_chain(
        candidate=Candidate(),
        new_observations=500,
        shadow_observations=150,
        incumbent_crowned_at="2026-01-01T00:00:00Z",
        now="2026-09-21T00:00:00Z",
    )
    seen = [item["stage"] for item in report["stages"]]
    check(
        seen == list(L5_STAGES),
        f"the chain did not run every stage in order: {seen}",
    )
    check(
        chain_problems(report) == [],
        f"a chain report is not contract-clean: {chain_problems(report)}",
    )

    # 8. NOT_EVALUATED BLOCKS. Absence of evidence is not evidence of safety.
    check(
        report["may_promote"] is False,
        "a chain with no promotion checklist reported may_promote=True — "
        "stages that could not run were treated as passing",
    )

    # 9. THE DECIDING BEHAVIOUR: identical evidence, different tenure.
    verdict = {
        "approved": True,
        "blocking": [],
        "checks": {
            "drift_clean": {"check": "drift_clean", "outcome": PROMO_PASS, "reason": "clean"},
            "oos_beats_incumbent": {
                "check": "oos_beats_incumbent", "outcome": PROMO_PASS, "reason": "wins"
            },
            "human_approval": {
                "check": "human_approval", "outcome": PROMO_PASS, "reason": "approved"
            },
        },
    }
    seasoned_run = evaluate_chain(
        candidate=Candidate(), new_observations=500, shadow_observations=150,
        promotion_verdict=verdict,
        incumbent_crowned_at="2026-01-01T00:00:00Z", now="2026-09-21T00:00:00Z",
    )
    fresh_run = evaluate_chain(
        candidate=Candidate(), new_observations=500, shadow_observations=150,
        promotion_verdict=verdict,
        incumbent_crowned_at="2026-09-20T00:00:00Z", now="2026-09-21T00:00:00Z",
    )
    check(
        seasoned_run["may_promote"] is True,
        f"a fully clean chain with a seasoned incumbent was still blocked at "
        f"{seasoned_run.get('blocked_at')} — nothing could ever be promoted",
    )
    check(
        fresh_run["may_promote"] is False
        and fresh_run["blocked_at"] == L5_STAGE_TENURE,
        f"identical evidence with a ONE-DAY-OLD incumbent was allowed "
        f"(blocked_at={fresh_run.get('blocked_at')!r}) — the tenure bound is "
        f"not the thing deciding",
    )

    # 10. PROMOTION IS NEVER AUTOMATIC.
    check(
        L5_AUTO_PROMOTE is False,
        "auto-promotion is enabled — a pipeline that supplies its own "
        "approver has removed the only stage a machine cannot satisfy",
    )
    check(
        seasoned_run.get("auto_promote") is False,
        "a chain report claimed auto_promote",
    )
    check(
        "human act" in str(seasoned_run.get("note") or ""),
        "the report does not say that may_promote is permission, not an act",
    )

    # 11. W5: L5 DELEGATES, it does not own a second promotion gate.
    check(
        L5_STAGE_GATE not in owned_stages(),
        "L5 claims to own the promotion gate — that is M5's, and a second "
        "copy is the split-brain W5 forbids",
    )
    check(
        tuple(owned_stages()) == tuple(L5_OWNED_STAGES)
        and L5_STAGE_TENURE in owned_stages(),
        "the owned-stage set no longer matches the contract",
    )
    delegated = delegated_stages()
    check(
        "core.promotion" in str(delegated.get(L5_STAGE_GATE, "")),
        "the promotion gate is not recorded as delegated to core.promotion",
    )
    source = (REPO_ROOT / "core" / "controlled_learning.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("def evaluate_promotion", "def drift_check", "def oos_check"):
        check(
            forbidden not in source,
            f"core/controlled_learning.py defines {forbidden!r} — L5 is "
            f"reimplementing an M5 check instead of delegating to it",
        )

    # 12. THE DELEGATION READS M5'S REAL SHAPE. An earlier version assumed
    #     `checks` was a list and would have reported NOT_EVALUATED for every
    #     real verdict while looking perfectly healthy.
    drift_stage = next(
        item for item in seasoned_run["stages"] if item["stage"] == "drift_testing"
    )
    check(
        drift_stage["outcome"] == PROMO_PASS,
        f"a PASSING M5 drift check surfaced as {drift_stage['outcome']} — the "
        f"delegation is reading a shape M5 does not emit",
    )

    if FAILURES:
        print("L5 controlled-learning gate FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("L5 controlled-learning gate OK:")
    print(
        f"  churn from noise: {unbounded:.0f} replacements/3yr unbounded vs "
        f"{bounded:.0f} at the {CHAMPION_MIN_TENURE_DAYS}-day bound."
    )
    print("  a champion crowned 15 minutes ago cannot be replaced; one of 263 days can.")
    print("  identical evidence + a 1-day-old incumbent blocks at champion_tenure.")
    print("  a safety rollback is never blocked; a back-dated crowning buys nothing.")
    print("  all 9 stages reach a verdict in order; NOT_EVALUATED blocks.")
    print("  L5 owns only champion_tenure and delegates the rest (W5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
