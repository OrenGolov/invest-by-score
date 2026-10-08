"""CI drift gate for the E4 event study engine.

Proves, on every push, that the measurements stay trustworthy:

1. the full chain is measured — baseline, stock, benchmark, sector, abnormal,
   volatility and volume, across every horizon;
2. the baseline ENDS BEFORE the event with a gap, so pre-announcement drift
   cannot be absorbed into "normal" and shrink the abnormal return;
3. an unmatured or unavailable window is ABSENT, never 0.0;
4. a missing benchmark yields NO abnormal return — never a stock return
   relabelled as abnormal;
5. the model is named on every result;
6. every result disclaims causality;
7. studies feed E3 actor observations with real measured returns, and never
   with fabricated ones.

Synthetic frames only; runs in about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    EVENT_STUDY_BASELINE_GAP_SESSIONS,
    EVENT_STUDY_HORIZONS,
    EVENT_STUDY_MODEL_MARKET_ADJUSTED,
    EVENT_STUDY_MODEL_MEAN_ADJUSTED,
)
from core.event_contract import Event  # noqa: E402
from core.event_study import (  # noqa: E402
    EventStudyError,
    observations_from_studies,
    run_event_study,
    study_event,
    study_report,
)


def _frame(bars: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, bars))
    return pd.DataFrame(
        {
            "Open": closes * 0.999, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, bars).astype(float),
        },
        index=pd.bdate_range("2024-01-01", periods=bars),
    )


def main() -> int:
    failures: list[str] = []

    frame, benchmark, sector = _frame(), _frame(seed=1), _frame(seed=2)
    event_time = frame.index[200].strftime("%Y-%m-%d %H:%M:%S")

    def study(**kwargs):
        payload = dict(
            ticker="NVDA", event_time=event_time, price_frame=frame,
            benchmark_frame=benchmark, sector_frame=sector,
            benchmark="VOO", sector="SOXX",
        )
        payload.update(kwargs)
        return run_event_study(**payload)

    # 1. The full chain.
    base = study()
    if base.status != "OK":
        failures.append(f"a study on ample history returned {base.status!r}")
    missing_horizons = set(EVENT_STUDY_HORIZONS) - set(base.reactions)
    if missing_horizons:
        failures.append(f"horizons not measured: {sorted(missing_horizons)}")
    if base.reactions:
        reaction = base.reactions.get("20d", {})
        for stage in (
            "stock_return", "benchmark_return", "sector_return", "abnormal_return",
            "sector_abnormal_return", "volatility_ratio", "volume_ratio",
        ):
            if reaction.get(stage) is None:
                failures.append(f"chain stage {stage!r} was not measured at 20d")
        if reaction:
            expected = reaction["stock_return"] - reaction["benchmark_return"]
            if abs(reaction["abnormal_return"] - expected) > 1e-6:
                failures.append("abnormal return is not stock minus benchmark")

    # 2. The baseline must exclude the event and its run-up.
    if base.status == "OK":
        if base.baseline["end_bar"] >= base.entry_bar:
            failures.append(
                "the baseline does not end before the event — the reaction is "
                "inside the window used to define normal"
            )
        entry_position = list(frame.index).index(pd.Timestamp(base.entry_bar))
        end_position = list(frame.index).index(pd.Timestamp(base.baseline["end_bar"]))
        if entry_position - end_position < EVENT_STUDY_BASELINE_GAP_SESSIONS:
            failures.append(
                f"the pre-event gap is {entry_position - end_position} sessions, "
                f"below the required {EVENT_STUDY_BASELINE_GAP_SESSIONS} — "
                f"pre-announcement drift would be absorbed into the baseline"
            )

    # 3. Thin baseline and unmatured horizons.
    early = study(event_time=frame.index[5].strftime("%Y-%m-%d %H:%M:%S"))
    if early.status != "INSUFFICIENT_BASELINE":
        failures.append(f"a thin baseline returned {early.status!r}")
    if early.reactions:
        failures.append("a refused study still reported reactions")

    recent = study(event_time=frame.index[-3].strftime("%Y-%m-%d %H:%M:%S"))
    for horizon in ("20d", "60d"):
        if recent.abnormal_return_for(horizon) is not None:
            failures.append(
                f"an unmatured {horizon} window reported a value — an absent "
                f"measurement must not become a measured zero"
            )
    if "intraday" not in recent.reactions:
        failures.append("a recent event measured nothing at all, not even intraday")

    if study(event_time="1990-01-01").status != "UNAVAILABLE":
        failures.append("an event predating the history was not reported unavailable")
    if study(price_frame=pd.DataFrame()).status != "UNAVAILABLE":
        failures.append("an empty price frame was not reported unavailable")

    # 4. A missing benchmark must not yield a relabelled stock return.
    no_benchmark = study(benchmark_frame=None)
    if no_benchmark.reactions.get("5d", {}).get("abnormal_return") is not None:
        failures.append(
            "an abnormal return was reported without a benchmark — a stock "
            "return relabelled as abnormal is the worst possible error here"
        )
    if no_benchmark.reactions.get("5d", {}).get("stock_return") is None:
        failures.append("the stock return went missing along with the benchmark")

    disjoint = _frame(bars=50)
    disjoint.index = pd.bdate_range("2030-01-01", periods=50)
    if study(benchmark_frame=disjoint).reactions.get("5d", {}).get("abnormal_return") is not None:
        failures.append("a non-overlapping benchmark still produced an abnormal return")

    # 5. The model is named, and both models work.
    if base.model != EVENT_STUDY_MODEL_MARKET_ADJUSTED:
        failures.append(f"the default model is {base.model!r}")
    mean_adjusted = study(benchmark_frame=None, model=EVENT_STUDY_MODEL_MEAN_ADJUSTED)
    if mean_adjusted.model != EVENT_STUDY_MODEL_MEAN_ADJUSTED:
        failures.append("the model is not recorded on the result")
    if mean_adjusted.reactions.get("5d", {}).get("abnormal_return") is None:
        failures.append("the mean-adjusted model produced no abnormal return")
    try:
        study(model="telepathy")
        failures.append("an unknown model was accepted")
    except EventStudyError:
        pass

    # 6. Causality disclaimer.
    if "association only" not in base.disclaimer:
        failures.append("a result does not disclaim causality (master context section 37)")
    if "association only" not in study_report([]).get("disclaimer", ""):
        failures.append("the study report does not carry the disclaimer")

    # 7. The E3 join.
    events = [
        Event(
            entity="NVDA", event_type="earnings", source="news", actor="Jensen Huang",
            published_time=frame.index[150 + index].strftime("%Y-%m-%d %H:%M:%S"),
            evidence=[{"source_record_id": f"r{index}"}],
        )
        for index in range(5)
    ]
    studies = [study_event(event, frame, benchmark) for event in events]
    observations = observations_from_studies(events, studies, horizon="20d")
    if len(observations) != len(events):
        failures.append(
            f"{len(observations)} observations from {len(events)} studied events"
        )
    if any(o.abnormal_return is None for o in observations):
        failures.append("an actor observation carries no measured abnormal return")

    anonymous = Event(
        entity="NVDA", event_type="earnings", source="news",
        published_time=event_time, evidence=[{"source_record_id": "anon"}],
    )
    if observations_from_studies([anonymous], [study_event(anonymous, frame, benchmark)]):
        failures.append("an anonymous event produced an actor observation")

    unmeasured_event = Event(
        entity="NVDA", event_type="earnings", source="news", actor="Jensen Huang",
        published_time=frame.index[5].strftime("%Y-%m-%d %H:%M:%S"),
        evidence=[{"source_record_id": "early"}],
    )
    if observations_from_studies(
        [unmeasured_event], [study_event(unmeasured_event, frame, benchmark)]
    ):
        failures.append(
            "an unmeasured study produced an actor observation — that would be a "
            "fabricated reaction in a track record"
        )

    if failures:
        print("E4 event-study gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("E4 event-study gate OK:")
    print(
        f"  full chain measured across {len(EVENT_STUDY_HORIZONS)} horizons "
        f"(baseline -> stock -> benchmark -> sector -> abnormal -> vol -> volume)."
    )
    print(
        f"  baseline ends before the event with a {EVENT_STUDY_BASELINE_GAP_SESSIONS}"
        f"-session gap; thin baselines refused."
    )
    print("  unmatured windows are absent, never zero; no benchmark means no abnormal return.")
    print("  model named on every result; causality disclaimed; E3 observations carry real returns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
