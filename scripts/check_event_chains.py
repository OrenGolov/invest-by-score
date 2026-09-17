"""CI drift gate for E7 event revision / contradiction learning.

Proves, on every push, that a story's evolution stays usable:

1. links classify correctly — flip = reversal, new source = confirmation,
   same source = duplicate, contradictory = correction;
2. a story counts ONCE. Four reports of one claim must not become four
   events in every downstream base rate;
3. a REVERSED claim weighs less than an unconfirmed one — a reversal is
   evidence the source was wrong, not an absence of evidence;
4. `contested` (confirmed AND reversed) stays a distinct state and is never
   averaged toward the middle;
5. the chain window separates distinct stories;
6. the reversal rate is reported — the training information E7 exists for.

Synthetic only; runs in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    CHAIN_LINK_CONFIRMATION,
    CHAIN_LINK_CORRECTION,
    CHAIN_LINK_DUPLICATE,
    CHAIN_LINK_INITIAL,
    CHAIN_LINK_REVERSAL,
    CHAIN_RELIABILITY_WEIGHT,
    CHAIN_STATUS_CONFIRMED,
    CHAIN_STATUS_CONTESTED,
    CHAIN_STATUS_CORRECTED,
    CHAIN_STATUS_OPEN,
    CHAIN_STATUS_REVERSED,
    EVENT_CHAIN_WINDOW_DAYS,
    EVENT_DIRECTION_CONTRADICTORY,
    EVENT_DIRECTION_NEGATIVE,
    EVENT_DIRECTION_POSITIVE,
)
from core.event_chains import (  # noqa: E402
    EventChainError,
    build_chains,
    chain_problems,
    chain_report,
    classify_link,
    deduplicate_events,
)
from core.event_contract import Event  # noqa: E402


def _event(index, day, direction=EVENT_DIRECTION_NEGATIVE, source="reuters",
           month=1, entity="NVDA", event_type="litigation") -> Event:
    return Event(
        entity=entity, published_time=f"2026-{month:02d}-{day:02d} 10:00:00",
        event_type=event_type, source=source, direction=direction,
        evidence=[{"source_record_id": f"r{index}"}],
    )


def main() -> int:
    failures: list[str] = []

    # 1. Link classification.
    for earlier, later, expected, label in (
        (_event(1, 5, EVENT_DIRECTION_NEGATIVE), _event(2, 7, EVENT_DIRECTION_POSITIVE),
         CHAIN_LINK_REVERSAL, "directional flip"),
        (_event(1, 5, source="reuters"), _event(2, 6, source="bloomberg"),
         CHAIN_LINK_CONFIRMATION, "independent corroboration"),
        (_event(1, 5, source="reuters"), _event(2, 6, source="reuters"),
         CHAIN_LINK_DUPLICATE, "same-source repetition"),
        (_event(1, 5), _event(2, 6, EVENT_DIRECTION_CONTRADICTORY),
         CHAIN_LINK_CORRECTION, "contradictory follow-up"),
    ):
        link, _ = classify_link(earlier, later)
        if link != expected:
            failures.append(f"{label} classified as {link!r}, expected {expected!r}")

    # CONTRADICTORY must not be treated as the opposite of a direction.
    link, _ = classify_link(
        _event(1, 5, EVENT_DIRECTION_CONTRADICTORY),
        _event(2, 6, EVENT_DIRECTION_POSITIVE, source="bloomberg"),
    )
    if link == CHAIN_LINK_REVERSAL:
        failures.append(
            "a contradictory claim was treated as reversible — it is already "
            "unresolved, so pairing it invents a flip"
        )

    # 2. One story counts once.
    story = [
        _event(1, 5), _event(2, 6, source="bloomberg"),
        _event(3, 7, source="ft"), _event(4, 8, EVENT_DIRECTION_POSITIVE),
    ]
    chains = build_chains(story)
    if len(chains) != 1:
        failures.append(f"one story produced {len(chains)} chains")
    surviving = deduplicate_events(story, chains)
    if len(surviving) != 1:
        failures.append(
            f"{len(story)} reports of one claim deduplicated to {len(surviving)} "
            f"events — a single story would be multiplied through every base rate"
        )
    if surviving and surviving[0] != story[0].event_id:
        failures.append("the surviving id is not the initial claim")

    if chains and chains[0].links[0].link_type != CHAIN_LINK_INITIAL:
        failures.append("the first link of a chain is not the initial claim")
    for chain in chains:
        problems = chain_problems(chain)
        if problems:
            failures.append(f"a built chain is invalid: {problems[:2]}")

    # 3. Reliability ordering.
    if CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_REVERSED] >= CHAIN_RELIABILITY_WEIGHT[
        CHAIN_STATUS_OPEN
    ]:
        failures.append(
            "a reversed claim does not weigh less than an unconfirmed one — a "
            "reversal is evidence the source was wrong, not an absence"
        )
    for weaker in (CHAIN_STATUS_OPEN, CHAIN_STATUS_CORRECTED,
                   CHAIN_STATUS_CONTESTED, CHAIN_STATUS_REVERSED):
        if CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_CONFIRMED] <= CHAIN_RELIABILITY_WEIGHT[weaker]:
            failures.append(f"a confirmed claim does not outweigh a {weaker} one")
    if CHAIN_RELIABILITY_WEIGHT[CHAIN_STATUS_CONTESTED] <= CHAIN_RELIABILITY_WEIGHT[
        CHAIN_STATUS_REVERSED
    ]:
        failures.append("a contested claim does not outweigh a reversed one")

    # 4. Settled statuses, including contested.
    for events, expected, label in (
        ([_event(1, 5)], CHAIN_STATUS_OPEN, "lone claim"),
        ([_event(1, 5), _event(2, 6, source="bloomberg")],
         CHAIN_STATUS_CONFIRMED, "corroborated claim"),
        ([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)],
         CHAIN_STATUS_REVERSED, "flipped claim"),
        ([_event(1, 5), _event(2, 6, EVENT_DIRECTION_CONTRADICTORY)],
         CHAIN_STATUS_CORRECTED, "contradicted claim"),
        ([_event(1, 5), _event(2, 6, source="bloomberg"),
          _event(3, 8, EVENT_DIRECTION_POSITIVE)],
         CHAIN_STATUS_CONTESTED, "confirmed then reversed"),
    ):
        status = build_chains(events)[0].status
        if status != expected:
            failures.append(f"{label} settled as {status!r}, expected {expected!r}")

    contested = build_chains([
        _event(1, 5), _event(2, 6, source="bloomberg"),
        _event(3, 8, EVENT_DIRECTION_POSITIVE),
    ])[0]
    if contested.status in (CHAIN_STATUS_CONFIRMED, CHAIN_STATUS_REVERSED):
        failures.append(
            "a confirmed-then-reversed chain collapsed to one side — contested "
            "is unresolved, not an average"
        )

    # 5. Windowing separates distinct stories.
    if len(build_chains([_event(1, 5), _event(2, 5, month=3)])) != 2:
        failures.append("events months apart were merged into one chain")
    if len(build_chains([
        _event(1, 5), _event(2, 5 + EVENT_CHAIN_WINDOW_DAYS - 1, source="bloomberg")
    ])) != 1:
        failures.append("events inside the window were not chained")
    if len(build_chains([_event(1, 5, entity="NVDA"), _event(2, 6, entity="AMD")])) != 2:
        failures.append("different entities shared a chain")
    if len(build_chains([
        _event(1, 5, event_type="litigation"), _event(2, 6, event_type="earnings")
    ])) != 2:
        failures.append("different event types shared a chain")

    # Order independence and bad input.
    later, earlier = _event(2, 8, source="bloomberg"), _event(1, 5)
    if build_chains([later, earlier])[0].chain_id != earlier.event_id:
        failures.append("chain building depends on input order")
    broken = _event(1, 5)
    broken.published_time = "not a date"
    if build_chains([broken]):
        failures.append(
            "an event with an unparseable timestamp was placed in a chain — its "
            "position would be invented"
        )
    try:
        build_chains([_event(1, 5)], window_days=0)
        failures.append("a zero-day chain window was accepted")
    except EventChainError:
        pass

    # 6. The reversal rate.
    report = chain_report(build_chains([_event(1, 5), _event(2, 7, EVENT_DIRECTION_POSITIVE)]))
    if report["reversal_rate"] != 1.0:
        failures.append(f"reversal rate is {report['reversal_rate']}, expected 1.0")
    if report["chains"] != 1 or report["events"] != 2:
        failures.append("the report does not count chains and events separately")
    if chain_report([])["reversal_rate"] != 0.0:
        failures.append("an empty batch did not report a zero reversal rate")

    if failures:
        print("E7 event-chain gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E7 event-chain gate OK:")
    print("  flip=reversal, new source=confirmation, same source=duplicate, contradictory=correction.")
    print("  a story counts once; the surviving id is the initial claim.")
    print("  reversed weighs less than unconfirmed; confirmed outweighs all.")
    print(
        f"  contested stays distinct; the {EVENT_CHAIN_WINDOW_DAYS}-day window "
        f"separates stories; reversal rate reported."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
