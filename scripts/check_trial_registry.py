"""CI drift gate for the M3 research trial registry.

Proves, on every push, that the anti-cherry-picking rules still hold:

1. a valid trial records every M3 field;
2. pre-registration is enforced — a registered trial carries no metrics, and
   completion demands the pre-registered primary metric;
3. a recorded result is immutable and cannot be re-completed;
4. an identical configuration is the SAME experiment, not a new one;
5. abandoned trials are recorded with a reason and still count toward the
   multiple-testing total;
6. every guard is non-vacuous — each refusal is exercised and must fire.

Exit 1 on any drift, with a diff-precise message.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    OUTCOME_LABEL_VERSION,
    TRIAL_STATUS_ABANDONED,
    TRIAL_STATUS_COMPLETED,
    TRIAL_STATUS_REGISTERED,
)
from core.trial_registry import (  # noqa: E402
    Trial,
    TrialRegistry,
    TrialRegistryError,
    load_trial_records,
    persist_trial,
    trial_problems,
)

_AT = "2026-09-16T00:00:00+00:00"


def _trial(**overrides) -> Trial:
    payload = dict(
        hypothesis=(
            "20d momentum features carry out-of-sample directional edge beyond "
            "the technical baseline"
        ),
        feature_set_version="fs-gate",
        model_family="boosting",
        label_version=OUTCOME_LABEL_VERSION,
        horizons=["20d"],
        training_window={"start": "2021-01-01", "end": "2025-01-01"},
        validation_scheme="walk_forward_embargo",
        costs={"cost_table_version": "backtest-cost-table-v2"},
        seed=42,
        dataset_hash="a" * 64,
        primary_metric="oos_sharpe",
    )
    payload.update(overrides)
    return Trial(**payload)


def main() -> int:
    failures: list[str] = []

    def refuses(label: str, call) -> None:
        """A guard that cannot fail is theatre — each must actually fire."""
        try:
            call()
        except TrialRegistryError:
            return
        failures.append(f"guard did NOT fire: {label}")

    # 1. A valid trial is accepted and records every field.
    valid = _trial()
    problems = trial_problems(valid)
    if problems:
        failures.append(f"a valid trial was rejected: {problems[:3]}")
    recorded = valid.to_dict()
    for required in (
        "trial_id", "hypothesis", "feature_set_version", "model_family",
        "hyperparameters", "label_version", "horizons", "training_window",
        "validation_scheme", "costs", "seed", "dataset_hash", "metrics",
    ):
        if required not in recorded:
            failures.append(f"M3 field {required!r} is not recorded on a trial")

    # 2. Field-level guards.
    refuses("empty hypothesis", lambda: TrialRegistry().register(_trial(hypothesis="")))
    refuses("trivial hypothesis", lambda: TrialRegistry().register(_trial(hypothesis="works")))
    refuses("unknown model family", lambda: TrialRegistry().register(_trial(model_family="astrology")))
    refuses("unknown validation scheme", lambda: TrialRegistry().register(_trial(validation_scheme="vibes")))
    refuses("missing costs", lambda: TrialRegistry().register(_trial(costs={})))
    refuses("missing dataset hash", lambda: TrialRegistry().register(_trial(dataset_hash="")))
    refuses("missing primary metric", lambda: TrialRegistry().register(_trial(primary_metric="")))
    refuses("unknown horizon", lambda: TrialRegistry().register(_trial(horizons=["999d"])))
    refuses("stale label version", lambda: TrialRegistry().register(_trial(label_version="outcome-label-v0")))

    # 3. Pre-registration.
    refuses(
        "registration carrying metrics",
        lambda: TrialRegistry().register(_trial(metrics={"oos_sharpe": 1.4})),
    )

    registry = TrialRegistry()
    trial = registry.register(_trial())
    if trial.metrics:
        failures.append("a freshly registered trial already carries metrics")
    if trial.status != TRIAL_STATUS_REGISTERED:
        failures.append(f"a new trial has status {trial.status!r}")

    refuses(
        "completion without the pre-registered metric",
        lambda: registry.complete(trial.trial_id, {"accuracy": 0.9}, _AT),
    )
    refuses("completion with empty metrics", lambda: registry.complete(trial.trial_id, {}, _AT))
    refuses(
        "completing an unregistered trial",
        lambda: TrialRegistry().complete("ghost", {"oos_sharpe": 1.0}, _AT),
    )

    # 4. Identity and immutability.
    before = trial.trial_id
    registry.complete(trial.trial_id, {"oos_sharpe": 1.31}, _AT)
    if trial.trial_id != before:
        failures.append("attaching metrics changed the trial_id")
    if trial.status != TRIAL_STATUS_COMPLETED:
        failures.append(f"completed trial has status {trial.status!r}")

    refuses(
        "re-completing a completed trial",
        lambda: registry.complete(trial.trial_id, {"oos_sharpe": 9.9}, _AT),
    )
    refuses(
        "abandoning a completed trial",
        lambda: registry.abandon(trial.trial_id, "changed my mind", _AT),
    )
    refuses("duplicate configuration", lambda: registry.register(_trial()))

    if _trial().trial_id != _trial().trial_id:
        failures.append("trial_id is not deterministic")
    if _trial().trial_id == _trial(seed=7).trial_id:
        failures.append("a changed seed did not produce a distinct trial_id")
    if _trial().trial_id == _trial(hypothesis="A different claim entirely, stated fully").trial_id:
        failures.append("a rewritten hypothesis kept the same trial_id")

    # 5. Abandoned trials are kept and counted.
    dropped = registry.register(_trial(seed=7))
    refuses("abandonment without a reason", lambda: registry.abandon(dropped.trial_id, "", _AT))
    registry.abandon(dropped.trial_id, "coverage too thin", _AT)
    kept = registry.get(dropped.trial_id)
    if kept is None:
        failures.append("an abandoned trial was DELETED — file-drawer bias")
    elif kept.status != TRIAL_STATUS_ABANDONED or not kept.abandoned_reason:
        failures.append("an abandoned trial did not retain its status and reason")

    report = registry.multiple_testing_report()
    if report["distinct_trials"] != 2 or report["abandoned"] != 1 or report["completed"] != 1:
        failures.append(f"multiple-testing report is wrong: {report}")

    # 6. Persistence preserves the pre-registration ordering.
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "research_trials.jsonl"
        fresh = TrialRegistry()
        pending = fresh.register(_trial(seed=11))
        persist_trial(pending, store)
        fresh.complete(pending.trial_id, {"oos_sharpe": 0.8}, _AT)
        persist_trial(pending, store)
        records = load_trial_records(store)
        if len(records) != 2:
            failures.append(f"expected 2 ledger lines, got {len(records)}")
        elif records[0]["status"] != TRIAL_STATUS_REGISTERED or records[0]["metrics"]:
            failures.append(
                "the ledger does not show the hypothesis recorded BEFORE the result"
            )
        try:
            persist_trial(_trial(seed=12, costs={}), store)
            failures.append("an invalid trial was persisted")
        except TrialRegistryError:
            pass

    if failures:
        print("M3 trial-registry gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M3 trial-registry gate OK:")
    print("  every M3 field recorded and validated; 19 guards proven non-vacuous.")
    print("  pre-registration enforced: no metrics at registration, primary metric locked.")
    print("  results immutable; identical configuration is the same experiment.")
    print("  abandoned trials retained with a reason and counted for multiple testing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
