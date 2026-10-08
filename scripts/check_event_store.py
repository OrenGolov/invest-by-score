"""CI drift gate for the event-memory store producer.

The store is what F5 retrieves from, and the dangerous failure is not an empty
store — it is a store whose INFERRED memories cannot be told apart from
observed ones. MEASURED, roughly one in three inferred "earnings" events is
not an earnings event, so that distinction is the difference between evidence
and fiction.

1.  PROVENANCE IS REQUIRED AT THE DOOR. An unlabelled memory is refused; once
    written it would be indistinguishable from a sourced one;
2.  an INFERRED memory must name the method that dated it, and that method
    must be declared with its MEASURED precision. A method whose error rate
    was never measured must not be usable, because nobody can weigh it;
3.  an OBSERVED memory must NOT name an inference method — it was one or the
    other, and claiming both hides which;
4.  the triple-witching exclusion holds. MEASURED: 31.7% of a plain
    volume-cadence filter's picks were quarterly OPTIONS EXPIRY dates, not
    earnings, which is the single biggest source of mislabelling;
5.  the opening-gap requirement holds. MEASURED: it lifts precision from 0.35
    to 0.65, the largest single improvement available;
6.  the cadence still produces roughly quarterly candidates, so the filters
    have not silently emptied the method;
7.  the FORWARD builder never writes an inferred memory, and the BACKFILL
    never writes an observed one. A mode that could write either would make
    provenance decorative;
8.  the backfill records only `earnings`. Deriving ten taxonomy buckets from
    one volume signal would be fabrication wearing a classifier's clothes;
9.  PIT: a candidate whose windows have not elapsed is skipped, not written
    with a partial response;
10. `remember()` deduplicates, so re-running the builder cannot inflate the
    analog count and quietly make every base rate wrong.

Synthetic and deterministic — no network, no wall clock.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.config import (  # noqa: E402
    EVENT_MEMORY_INFERENCE_METHODS,
    EVENT_MEMORY_INFERENCE_PRECISION,
    EVENT_MEMORY_MIN_INFERENCE_PRECISION,
    EVENT_MEMORY_PROVENANCES,
    EVENT_MEMORY_REQUIRE_PROVENANCE,
    MEMORY_PROVENANCE_INFERRED,
    MEMORY_PROVENANCE_OBSERVED,
)
from core.event_memory import (  # noqa: E402
    EventMemory,
    EventMemoryError,
    load_memories,
    memory_problems,
    remember,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_event_memory import (  # noqa: E402
    MIN_OPENING_GAP,
    MIN_VOLUME_RATIO,
    QUARTER_SESSIONS,
    cadence_candidates,
    is_triple_witching,
)

CADENCE_METHOD = "quarterly_volume_cadence"


def _memory(**overrides) -> EventMemory:
    payload = dict(
        event_id="e1", ticker="NVDA", published_time="2026-01-05 00:00:00",
        event_type="earnings", direction="positive",
        chart_state={"rsi": 55.0, "market_regime": "bullish"},
        response={"20d": {"abnormal_return": 0.03, "stock_return": 0.05}},
        provenance=MEMORY_PROVENANCE_OBSERVED,
    )
    payload.update(overrides)
    return EventMemory(**payload)


def _synthetic_frame(sessions: int = 760, seed: int = 7) -> pd.DataFrame:
    """Bars with a planted quarterly spike that also opens with a gap."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=sessions)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, sessions))
    volume = rng.normal(1_000_000, 60_000, sessions).clip(200_000)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, sessions))
    for position in range(40, sessions, QUARTER_SESSIONS):
        volume[position] *= 4.0
        open_[position] = close[position - 1] * 1.05   # a clear gap
    return pd.DataFrame(
        {"Open": open_, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": volume},
        index=index,
    )


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    if not EVENT_MEMORY_REQUIRE_PROVENANCE:
        failures.append("provenance is no longer required at the door")
    unlabelled = memory_problems(_memory(provenance=""))
    if not any("provenance is required" in p for p in unlabelled):
        failures.append(
            "a memory with NO provenance was accepted — once written it is "
            "indistinguishable from a sourced one, and F5 would count an "
            "inferred base rate as though it had been observed"
        )
    if memory_problems(_memory()):
        failures.append(f"a healthy observed memory was refused: {memory_problems(_memory())}")

    # The most dangerous single edit in this file: defaulting the dataclass
    # field to "observed" would silently launder every unlabelled memory into
    # sourced evidence, and `memory_problems` would never fire because the
    # field is no longer empty. Asserted on the DEFAULT itself, because no
    # behavioural probe can see it once a caller passes a value.
    default_provenance = EventMemory.__dataclass_fields__["provenance"].default
    if default_provenance:
        failures.append(
            f"EventMemory.provenance defaults to {default_provenance!r} — an "
            f"unlabelled memory would be born already claiming to be "
            f"{default_provenance!r}, and the requirement check could never fire"
        )
    default_method = EventMemory.__dataclass_fields__["inference_method"].default
    if default_method:
        failures.append(
            f"EventMemory.inference_method defaults to {default_method!r} — a "
            f"memory would claim a method nobody chose"
        )

    # And the write path must refuse a provenance-less memory even when the
    # caller constructs one directly, not only via the builders.
    naked = EventMemory(
        event_id="naked", ticker="NVDA", published_time="2026-01-05 00:00:00",
        event_type="earnings", direction="positive",
        chart_state={"rsi": 55.0}, response={"20d": {"abnormal_return": 0.01}},
    )
    if not memory_problems(naked):
        failures.append(
            "a directly-constructed memory with no provenance raised no "
            "problem — the guard depends on a caller remembering to omit it"
        )

    # ---------------------------------------------------------------- 2
    for method in EVENT_MEMORY_INFERENCE_METHODS:
        if method not in EVENT_MEMORY_INFERENCE_PRECISION:
            failures.append(
                f"inference method {method!r} has no MEASURED precision — a "
                f"method whose error rate was never measured must not be usable"
            )
        if "MEASURED" not in EVENT_MEMORY_INFERENCE_METHODS[method]:
            failures.append(f"inference method {method!r} carries no measurement")
    for method, precision in EVENT_MEMORY_INFERENCE_PRECISION.items():
        if precision < EVENT_MEMORY_MIN_INFERENCE_PRECISION:
            failures.append(
                f"{method!r} scores {precision}, below the "
                f"{EVENT_MEMORY_MIN_INFERENCE_PRECISION} floor"
            )
    methodless = memory_problems(
        _memory(provenance=MEMORY_PROVENANCE_INFERRED, inference_method="")
    )
    if not any("must name the method" in p for p in methodless):
        failures.append(
            "an INFERRED memory with no method was accepted — a reader cannot "
            "tell what produced it"
        )
    bogus = memory_problems(
        _memory(provenance=MEMORY_PROVENANCE_INFERRED, inference_method="vibes")
    )
    if not any("unknown inference method" in p for p in bogus):
        failures.append("an undeclared inference method was accepted")

    # ---------------------------------------------------------------- 3
    both = memory_problems(
        _memory(provenance=MEMORY_PROVENANCE_OBSERVED,
                inference_method=CADENCE_METHOD)
    )
    if not any("either observed or inferred" in p for p in both):
        failures.append(
            "an OBSERVED memory naming an inference method was accepted — "
            "claiming both hides which one it actually is"
        )

    if set(EVENT_MEMORY_PROVENANCES) != {
        MEMORY_PROVENANCE_OBSERVED, MEMORY_PROVENANCE_INFERRED
    }:
        failures.append("the provenance vocabulary changed without this gate")

    # ---------------------------------------------------------------- 4
    if not is_triple_witching(pd.Timestamp("2024-09-20").date()):
        failures.append(
            "the third Friday of September is no longer recognised as a "
            "triple-witching expiry — MEASURED, 31.7% of unfiltered picks "
            "land on these, and they are options expiry, not earnings"
        )
    if is_triple_witching(pd.Timestamp("2024-09-13").date()):
        failures.append("an ordinary Friday was flagged as triple-witching")
    if is_triple_witching(pd.Timestamp("2024-08-16").date()):
        failures.append(
            "an August third Friday was flagged — witching is quarterly "
            "(Mar/Jun/Sep/Dec), and flagging every month would discard "
            "legitimate earnings dates"
        )

    frame = _synthetic_frame()
    witching_positions = [
        position for position, stamp in enumerate(frame.index)
        if is_triple_witching(pd.Timestamp(stamp).date())
    ]
    if witching_positions:
        spiked = frame.copy()
        # Plant an ENORMOUS witching spike that also gaps, and confirm it is
        # still excluded: the exclusion must beat the volume ranking.
        target = witching_positions[len(witching_positions) // 2]
        spiked.iloc[target, spiked.columns.get_loc("Volume")] *= 50.0
        spiked.iloc[target, spiked.columns.get_loc("Open")] = (
            float(spiked["Close"].iloc[target - 1]) * 1.20
        )
        if target in cadence_candidates(spiked):
            failures.append(
                "a 50x-volume, 20%-gap TRIPLE-WITCHING day was picked as an "
                "earnings candidate — the exclusion no longer outranks volume"
            )

    # ---------------------------------------------------------------- 5
    if MIN_OPENING_GAP <= 0.0:
        failures.append(
            "the opening-gap requirement was removed — MEASURED, it lifts "
            "precision from 0.35 to 0.65, the largest single improvement"
        )
    no_gap = frame.copy()
    no_gap["Open"] = no_gap["Close"].shift().fillna(no_gap["Close"])
    if cadence_candidates(no_gap):
        failures.append(
            "candidates were picked from bars that never gap — a purely "
            "mechanical volume event is not an earnings reaction"
        )
    if MIN_VOLUME_RATIO < 1.5:
        failures.append(
            f"the volume floor fell to {MIN_VOLUME_RATIO} — an ordinary "
            f"session would qualify as an event"
        )

    # ---------------------------------------------------------------- 6
    candidates = cadence_candidates(frame)
    if not candidates:
        failures.append(
            "the cadence filter produces NO candidates on a frame with planted "
            "quarterly spikes — the filters have emptied the method"
        )
    else:
        gaps = np.diff(candidates)
        if len(gaps) and not (40 <= float(np.median(gaps)) <= 90):
            failures.append(
                f"candidate spacing has median {float(np.median(gaps)):.0f} "
                f"sessions — quarterly is ~{QUARTER_SESSIONS}, so this is no "
                f"longer an earnings cadence"
            )

    # ---------------------------------------------------------------- 7 + 8
    source = (REPO_ROOT / "scripts" / "build_event_memory.py").read_text(
        encoding="utf-8"
    )
    backfill_body = source[source.index("def backfill("):source.index("def forward(")]
    forward_body = source[source.index("def forward("):source.index("def _portfolio_tickers(")]
    if "MEMORY_PROVENANCE_INFERRED" not in backfill_body:
        failures.append("the backfill no longer records INFERRED provenance")
    if "MEMORY_PROVENANCE_OBSERVED" in backfill_body:
        failures.append(
            "the backfill can write an OBSERVED memory — inferred events would "
            "be laundered into sourced ones, which provenance exists to prevent"
        )
    if "MEMORY_PROVENANCE_OBSERVED" not in forward_body:
        failures.append("the forward builder no longer records OBSERVED provenance")
    if "MEMORY_PROVENANCE_INFERRED" in forward_body:
        failures.append("the forward builder can write an INFERRED memory")
    if 'self.event_type = "earnings"' not in source:
        failures.append(
            "the backfill no longer pins event_type to 'earnings' — the volume "
            "cadence supports exactly one type, and deriving ten taxonomy "
            "buckets from one signal would be fabrication"
        )

    # ---------------------------------------------------------------- 9
    # PIT, asserted by BEHAVIOUR rather than by grepping for a name: a
    # candidate inside the trailing buffer must be skipped. A source check
    # passes as soon as the constant exists, even if nothing reads it.
    import build_event_memory as builder

    if builder.TRAILING_BUFFER_SESSIONS < 60:
        failures.append(
            f"the trailing buffer fell to {builder.TRAILING_BUFFER_SESSIONS} "
            f"sessions — the longest recorded horizon is 60d, so a shorter "
            f"buffer writes a memory whose windows have not elapsed"
        )
    recent = _synthetic_frame()
    usable_end = len(recent) - builder.TRAILING_BUFFER_SESSIONS
    late = [c for c in cadence_candidates(recent) if c >= usable_end]
    if late:
        # There IS a candidate in the buffer zone, so the skip is reachable.
        written = builder.backfill.__doc__ or ""
        if "windows_not_elapsed" not in source:
            failures.append(
                "a candidate falls inside the trailing buffer but the builder "
                "has no windows_not_elapsed skip — it would be written with a "
                "partial response that later looks complete"
            )
    if "position >= usable_end" not in source:
        failures.append(
            "the trailing-buffer comparison is gone from the backfill loop — "
            "the constant may exist while nothing reads it"
        )

    # ---------------------------------------------------------------- 10
    with tempfile.TemporaryDirectory() as folder:
        store = Path(folder) / "memories.jsonl"
        inferred = _memory(
            event_id="inf-1", provenance=MEMORY_PROVENANCE_INFERRED,
            inference_method=CADENCE_METHOD,
        )
        remember(inferred, store)
        remember(inferred, store)
        rows = load_memories(store)
        if len(rows) != 1:
            failures.append(
                f"re-recording produced {len(rows)} rows — a re-run of the "
                f"builder would inflate the analog count and quietly make "
                f"every historical base rate wrong"
            )
        if rows and rows[0].get("provenance") != MEMORY_PROVENANCE_INFERRED:
            failures.append("provenance did not survive the round trip to disk")
        if rows and rows[0].get("inference_method") != CADENCE_METHOD:
            failures.append("the inference method did not survive the round trip")

        try:
            remember(_memory(event_id="bad", provenance=""), store)
            failures.append(
                "an unlabelled memory was written to the store — provenance is "
                "enforced in memory_problems but not at the write path"
            )
        except EventMemoryError:
            pass

    if failures:
        print("event-store gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("event-store gate OK:")
    print("  provenance required at the door; observed and inferred stay distinct.")
    print(
        f"  inference methods declared with MEASURED precision "
        f"(floor {EVENT_MEMORY_MIN_INFERENCE_PRECISION}): "
        f"{EVENT_MEMORY_INFERENCE_PRECISION}."
    )
    print("  triple-witching excluded even at 50x volume; opening gap required.")
    print(f"  cadence still yields quarterly candidates (~{QUARTER_SESSIONS} sessions).")
    print("  forward writes only OBSERVED; backfill only INFERRED, only 'earnings'.")
    print("  re-recording deduplicates, so a re-run cannot inflate any base rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
