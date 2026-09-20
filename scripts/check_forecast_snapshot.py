"""CI drift gate for the F8 ForecastSnapshot contract.

The contract's value is that a consumer can depend on it. Two properties carry
that, and both are the kind a convenience refactor removes:

  - every declared field is ALWAYS present as a key, with absence explicit;
  - an absent field carries NO value, because a placeholder becomes a number
    a consumer renders.

1.  every contract field appears, in declared order, on every snapshot;
2.  ABSENT and REFUSED fields carry no value. `expected_return: 0.0` would
    render as "flat" under the dashboard's `Number(x ?? 0)` idiom — a
    confident claim about a quantity nobody computed;
3.  ...and the guard is exercised directly, not merely observed: a caller
    passing a value into a non-PRESENT field must not be able to smuggle it;
4.  the fields declared unavailable stay unavailable while their producers do
    not exist, and each states WHY;
5.  required fields are genuinely required: a snapshot missing one is not a
    degraded forecast, it is not a forecast;
6.  `model_contributions` carries F6 UNCHANGED and declares itself
    non-additive. MEASURED: regime +0.064 and chart +0.067 sum to +0.131
    against an ACTUAL joint effect of +0.060, so the field name must not be
    read literally;
7.  the digest is a real identity — deterministic, and sensitive to content;
8.  absent fields raise a TOP-LEVEL warning, because a headline-only consumer
    would otherwise never learn of the gap;
9.  `feature_digest` composes M1's function rather than hashing again (W5);
10. the accessor returns None for a non-PRESENT field rather than raising,
    so a consumer cannot crash on an honest gap.

Synthetic and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    SNAPSHOT_CONTRIBUTIONS_NOTE,
    SNAPSHOT_FIELDS,
    SNAPSHOT_REQUIRED_FIELDS,
    SNAPSHOT_STATUS_ABSENT,
    SNAPSHOT_STATUS_PRESENT,
    SNAPSHOT_STATUS_REFUSED,
    SNAPSHOT_UNAVAILABLE_FIELDS,
    SNAPSHOT_WARNING_ABSENT_FIELDS,
    SNAPSHOT_WARNING_NO_MODEL,
)
from core.forecast_snapshot import (  # noqa: E402
    ForecastSnapshotError,
    _field,
    build_forecast_snapshot,
    forecast_versions,
    render_snapshot,
    snapshot_digest,
    snapshot_problems,
    snapshot_value,
)


def _snapshot(**overrides):
    payload = dict(
        ticker="NVDA", as_of="2026-09-20", horizon="20d",
        chart_state={"rsi": 55.0, "volatility": 0.22, "change_20d": 0.05},
        confidence={
            "confidence": 0.42, "binding_factor": "sample_size",
            "assessed_object": "forecast",
        },
        decomposition={"decomposed_object": "forecast", "additive": False},
        regime="bullish",
        evidence={"memory_ids": ["m1", "m2"]},
    )
    payload.update(overrides)
    return build_forecast_snapshot(
        payload.pop("ticker"), payload.pop("as_of"), payload.pop("horizon"),
        **payload,
    )


def main() -> int:
    failures: list[str] = []

    snapshot = _snapshot()

    # ---------------------------------------------------------------- 1
    if list(snapshot["fields"]) != list(SNAPSHOT_FIELDS):
        failures.append(
            "the snapshot does not report every contract field in declared "
            "order — a consumer relies on every name being present, with "
            "absence explicit rather than the key missing"
        )
    if len(SNAPSHOT_FIELDS) != 15:
        failures.append(
            f"the contract now declares {len(SNAPSHOT_FIELDS)} fields — adding "
            f"or dropping one changes what consumers may rely on"
        )

    # ---------------------------------------------------------------- 2
    for name, row in snapshot["fields"].items():
        status = row.get("status")
        if status != SNAPSHOT_STATUS_PRESENT:
            if "value" in row:
                failures.append(
                    f"{name}: a {status} field carries a value — it would "
                    f"coalesce to a number in a consumer and render a quantity "
                    f"nobody computed"
                )
            if not row.get("reason"):
                failures.append(f"{name}: a {status} field does not explain itself")

    # ---------------------------------------------------------------- 3
    # The guard, exercised on purpose. No code path passes a value into a
    # non-PRESENT field, so a gate that only inspects built snapshots cannot
    # see whether the guard exists.
    for blocked in (SNAPSHOT_STATUS_ABSENT, SNAPSHOT_STATUS_REFUSED):
        smuggled = _field(blocked, value=0.0, reason="probe", samples=99)
        if "value" in smuggled:
            failures.append(
                f"a {blocked} field kept a value that was passed to it — this "
                f"is how a field nobody computed starts rendering as 0.0"
            )
        if "samples" in smuggled:
            failures.append(f"a {blocked} field kept extra detail passed to it")

    # ---------------------------------------------------------------- 4
    if not SNAPSHOT_UNAVAILABLE_FIELDS:
        failures.append(
            "no field is declared unavailable — if a model really was trained, "
            "this gate must be updated deliberately, because ABSENT existing "
            "is what stops an unproduced field reading as a computed zero"
        )
    for name, reason in SNAPSHOT_UNAVAILABLE_FIELDS.items():
        row = snapshot["fields"].get(name) or {}
        if row.get("status") != SNAPSHOT_STATUS_ABSENT:
            failures.append(
                f"{name}: declared unavailable but reports {row.get('status')!r}"
            )
        if not reason:
            failures.append(f"{name}: declared unavailable with no reason")
    for name in ("model_versions", "expected_return"):
        if name not in SNAPSHOT_UNAVAILABLE_FIELDS:
            failures.append(
                f"{name} is no longer declared unavailable while "
                f"build_joint_forecast still reports NO_MODEL"
            )

    # ---------------------------------------------------------------- 5
    for name in SNAPSHOT_REQUIRED_FIELDS:
        if (snapshot["fields"].get(name) or {}).get("status") != SNAPSHOT_STATUS_PRESENT:
            failures.append(f"{name}: is required but not PRESENT")
    for bad_ticker, bad_horizon in (("", "20d"), ("NVDA", "")):
        try:
            build_forecast_snapshot(bad_ticker, "2026-09-20", bad_horizon)
            failures.append(
                f"a snapshot was built with ticker={bad_ticker!r} "
                f"horizon={bad_horizon!r} — required fields must be enforced"
            )
        except ForecastSnapshotError:
            pass

    # ---------------------------------------------------------------- 6
    contributions = snapshot["fields"]["model_contributions"]
    if contributions.get("status") == SNAPSHOT_STATUS_PRESENT:
        if contributions.get("additive") is not False:
            failures.append(
                "model_contributions does not declare itself non-additive — "
                "the field name invites exactly the additive reading F6 "
                "MEASURED to be wrong (+0.131 sum vs +0.060 actual)"
            )
        # Checked against the CONSTANT, not against the built row: the row's
        # reason IS that constant, so comparing them compares a value to
        # itself and moves whenever the constant does.
        for phrase in ("NOT contributions", "do NOT sum", "MEASURED"):
            if phrase not in SNAPSHOT_CONTRIBUTIONS_NOTE:
                failures.append(
                    f"SNAPSHOT_CONTRIBUTIONS_NOTE no longer says {phrase!r} — "
                    f"the field name invites the additive reading F6 MEASURED "
                    f"to be wrong, and this note is the only thing denying it"
                )
            if phrase not in (contributions.get("reason") or ""):
                failures.append(
                    f"model_contributions no longer carries {phrase!r}"
                )

    # ---------------------------------------------------------------- 7
    if snapshot_digest(snapshot) != snapshot["digest"]:
        failures.append("the stored digest does not match the snapshot content")
    if _snapshot()["digest"] != snapshot["digest"]:
        failures.append(
            "two identical requests produced different digests — the digest "
            "cannot answer whether a forecast changed"
        )
    if _snapshot(regime="bearish")["digest"] == snapshot["digest"]:
        failures.append(
            "a changed input produced the SAME digest — the digest is not "
            "sensitive to content and cannot detect drift"
        )

    # ---------------------------------------------------------------- 8
    codes = snapshot["warning_codes"]
    if snapshot["absent"] and SNAPSHOT_WARNING_ABSENT_FIELDS not in codes:
        failures.append(
            "fields are absent but no top-level warning says so — a "
            "headline-only consumer would never learn of the gap"
        )
    if SNAPSHOT_WARNING_NO_MODEL not in codes:
        failures.append(
            "no top-level no_trained_model warning, while model_versions and "
            "expected_return are both absent for exactly that reason"
        )
    warnings_row = snapshot["fields"]["warnings"]
    if warnings_row.get("status") != SNAPSHOT_STATUS_PRESENT:
        failures.append("the warnings field is not PRESENT")

    # ---------------------------------------------------------------- 9
    digest_row = snapshot["fields"]["feature_digest"]
    if digest_row.get("status") != SNAPSHOT_STATUS_PRESENT:
        failures.append(
            f"feature_digest is {digest_row.get('status')!r}: "
            f"{digest_row.get('reason')}"
        )
    elif "feature_surface_digest" not in (digest_row.get("reason") or ""):
        failures.append(
            "feature_digest no longer names M1's function — a second hashing "
            "scheme is the split-brain W5 forbids"
        )
    source = (REPO_ROOT / "core" / "forecast_snapshot.py").read_text(encoding="utf-8")
    if "feature_surface_digest" not in source:
        failures.append("the snapshot no longer composes M1's digest function")

    # The digest must be M1's ACTUAL output, not merely a hex-shaped string.
    # A constant of the right length passes every structural check, so the
    # value is recomputed independently and compared.
    from core.feature_registry import (  # noqa: PLC0415
        feature_surface_digest,
        load_feature_registry,
    )

    probe_state = {"rsi": 55.0, "volatility": 0.22, "change_20d": 0.05}
    expected = feature_surface_digest(
        {name: {"value": value} for name, value in sorted(probe_state.items())},
        load_feature_registry(),
    )
    published = snapshot_value(_snapshot(chart_state=probe_state), "feature_digest")
    if published != expected:
        failures.append(
            f"feature_digest {str(published)[:16]}... does not match M1's own "
            f"{expected[:16]}... — the snapshot is publishing a digest it did "
            f"not derive from the feature surface"
        )
    # ...and it must change when the surface changes.
    other = snapshot_value(
        _snapshot(chart_state={**probe_state, "rsi": 70.0}), "feature_digest"
    )
    if other == published:
        failures.append(
            "feature_digest did not change when the feature surface did — a "
            "constant would pass every other check in this gate"
        )

    # ---------------------------------------------------------------- 10
    for name in snapshot["absent"] + snapshot["refused"]:
        if snapshot_value(snapshot, name) is not None:
            failures.append(f"{name}: the accessor returned a value for a gap")
    if snapshot_value(snapshot, "ticker") != "NVDA":
        failures.append("the accessor did not return a PRESENT field's value")

    # A snapshot built with nothing still honours the contract.
    bare = build_forecast_snapshot("NVDA", "2026-09-20", "20d")
    for problem in snapshot_problems(bare):
        failures.append(f"bare snapshot problem: {problem}")
    if list(bare["fields"]) != list(SNAPSHOT_FIELDS):
        failures.append("a bare snapshot omits contract fields")
    for row in render_snapshot(bare):
        if row["status"] != SNAPSHOT_STATUS_PRESENT and not row["reason"]:
            failures.append(f"{row['field']}: renders blank for a reader")

    for problem in snapshot_problems(snapshot):
        failures.append(f"contract problem: {problem}")

    # Versions must all be present, or a replay cannot be pinned.
    versions = forecast_versions()
    for part in ("snapshot", "contract", "joint", "conditional", "event",
                 "decomposition", "confidence"):
        if not versions.get(part):
            failures.append(f"forecast_versions is missing {part!r}")

    if failures:
        print("F8 forecast-snapshot gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("F8 forecast-snapshot gate OK:")
    print(
        f"  {len(SNAPSHOT_FIELDS)} contract fields, always present as keys; "
        f"{len(SNAPSHOT_UNAVAILABLE_FIELDS)} ABSENT with stated reasons."
    )
    print("  an ABSENT or REFUSED field carries NO value, and the guard is attacked.")
    print("  model_contributions carries F6 unchanged and declares itself non-additive.")
    print("  the digest is deterministic and content-sensitive.")
    print("  feature_digest composes M1's feature_surface_digest (W5).")
    print(f"  top-level warnings surfaced: {', '.join(codes)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
