"""CI drift gate for E3 influential person intelligence.

E3 exists before E4 measures any market reaction, so the number the system is
most tempted to invent is an actor's "historical market impact". This gate
proves it refuses to:

1. every E3 attribute is tracked (identity, role, organization, topics,
   credibility, frequency, novelty, impact);
2. credibility is labelled a PRIOR, never a measurement;
3. insider standing is company-specific — a CEO is not a rival's insider;
4. impact is `unmeasured` below the threshold, and reports NO numbers there;
5. statements without a measured reaction do not count toward a track record;
6. the actor registry derives from the E2 entity registry rather than being
   maintained twice.

Synthetic only; runs in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.actor_intelligence import (  # noqa: E402
    ActorIntelligenceError,
    ActorObservation,
    ActorProfile,
    ActorRecord,
    actor_problems,
    build_actor_registry,
    observations_from_events,
    record_observation,
    registry_report,
)
from core.config import (  # noqa: E402
    ACTOR_MIN_OBSERVATIONS,
    ACTOR_STANDING_EXTERNAL,
    ACTOR_STANDING_INSIDER,
)
from core.event_contract import Event  # noqa: E402


def _profile(**overrides) -> ActorProfile:
    payload = dict(
        name="Jensen Huang", role="executive", organization="NVIDIA Corporation",
        standing=ACTOR_STANDING_INSIDER, speaks_for=("NVDA",),
        topics=("earnings", "guidance"),
    )
    payload.update(overrides)
    return ActorProfile(record=ActorRecord(**payload))


def _observations(count: int, abnormal_return=0.02) -> list[ActorObservation]:
    return [
        ActorObservation(
            actor="Jensen Huang", event_type="earnings", ticker="NVDA",
            published_time="2026-01-05 00:00:00", novelty=0.5,
            abnormal_return=abnormal_return,
        )
        for _ in range(count)
    ]


def main() -> int:
    failures: list[str] = []

    # 1. Every E3 attribute is tracked.
    payload = _profile().to_dict()
    for required in (
        "name", "role", "organization", "standing", "speaks_for", "topics",
        "credibility", "statement_frequency", "mean_novelty", "impact",
    ):
        if required not in payload:
            failures.append(f"E3 attribute {required!r} is not tracked")

    if actor_problems(_profile().record):
        failures.append(f"a valid actor was rejected: {actor_problems(_profile().record)[:2]}")
    if not actor_problems(_profile(role="").record):
        failures.append("an actor without a role was accepted")
    if not actor_problems(_profile(speaks_for=()).record):
        failures.append("an insider naming no company was accepted")
    if not actor_problems(_profile(standing="oracle").record):
        failures.append("an unknown standing was accepted")

    # 2. Credibility is a prior, and says so.
    credibility = _profile().credibility("NVDA")
    if credibility["basis"] != "prior":
        failures.append(f"credibility basis is {credibility['basis']!r}, expected 'prior'")
    if "not a measurement" not in credibility["detail"]:
        failures.append(
            "credibility does not disclaim being a measurement — a guessed "
            "number presented as measured is exactly the M6 failure"
        )

    # 3. Insider standing is company-specific.
    profile = _profile()
    if profile.credibility("NVDA")["value"] <= profile.credibility("AMD")["value"]:
        failures.append("an insider is not more credible about their own company")
    if profile.credibility("AMD")["standing"] != ACTOR_STANDING_EXTERNAL:
        failures.append(
            "a CEO commenting on a rival was treated as that rival's insider"
        )
    if profile.record.is_insider_for("AMD"):
        failures.append("is_insider_for is not company-specific")

    # 4/5. Impact refuses to be invented.
    empty = _profile().impact_summary()
    if empty["status"] != "unmeasured":
        failures.append(
            f"impact with no observations is {empty['status']!r}, expected "
            f"'unmeasured' — an invented track record is E3's worst failure"
        )
    for forbidden in ("mean_abnormal_return", "median_abnormal_return", "positive_share"):
        if forbidden in empty:
            failures.append(
                f"an unmeasured impact reported {forbidden} — a consumer could "
                f"treat it as a finding"
            )

    below = _profile()
    below.observations = _observations(ACTOR_MIN_OBSERVATIONS - 1)
    if below.impact_summary()["status"] != "unmeasured":
        failures.append(
            f"{ACTOR_MIN_OBSERVATIONS - 1} observations was treated as a track record"
        )

    at_threshold = _profile()
    at_threshold.observations = _observations(ACTOR_MIN_OBSERVATIONS)
    measured = at_threshold.impact_summary()
    if measured["status"] != "measured":
        failures.append("the impact threshold is unreachable — the gate could never pass")
    if "association only" not in measured.get("detail", ""):
        failures.append(
            "measured impact does not disclaim causality (master context section 37)"
        )

    unreacted = _profile()
    unreacted.observations = _observations(ACTOR_MIN_OBSERVATIONS * 2, abnormal_return=None)
    if unreacted.impact_summary()["status"] != "unmeasured":
        failures.append(
            "statements with no measured reaction counted toward a track record"
        )

    # 6. Derived from E2, not maintained twice.
    profiles = build_actor_registry()
    if "Jensen Huang" not in profiles:
        failures.append("the actor registry did not derive executives from E2")
    for name, derived in profiles.items():
        problems = actor_problems(derived.record)
        if problems:
            failures.append(f"derived actor {name!r} is invalid: {problems[:1]}")
            break
    organizations = {p.record.organization for p in profiles.values()}
    if "Vanguard S&P 500 ETF" in organizations:
        failures.append("a fund contributed an actor — no officer speaks for a fund")

    topics = profiles["Jensen Huang"].record.topics
    if "macro_shock" in topics:
        failures.append("a company officer was declared an authority on macro")

    report = registry_report(profiles)
    if report["with_measured_impact"] != 0:
        failures.append(
            "an actor claims measured impact before E4 has supplied any reactions"
        )
    if report["actors"] < 50:
        failures.append(f"only {report['actors']} actors derived, expected the full roster")

    # Observation plumbing.
    try:
        record_observation({}, _observations(1)[0])
        failures.append("an unknown actor accrued a track record")
    except ActorIntelligenceError:
        pass

    anonymous = [Event(
        entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
        source="src", evidence=[{"source_record_id": "r1"}],
    )]
    if observations_from_events(anonymous):
        failures.append("an anonymous event produced an actor observation")

    named = [Event(
        entity="NVDA", published_time="2026-01-05 00:00:00", event_type="earnings",
        source="src", actor="Jensen Huang", evidence=[{"source_record_id": "r1"}],
    )]
    if len(observations_from_events(named)) != 1:
        failures.append("a named-actor event did not produce an observation")

    if failures:
        print("E3 actor-intelligence gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E3 actor-intelligence gate OK:")
    print(f"  {report['actors']} actors derived from the E2 entity registry; funds contribute none.")
    print("  credibility is labelled a PRIOR; insider standing is company-specific.")
    print(
        f"  impact stays 'unmeasured' below {ACTOR_MIN_OBSERVATIONS} reactions and "
        f"reports no numbers there."
    )
    print("  measured impact disclaims causality; unreacted statements do not count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
