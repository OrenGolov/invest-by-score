"""The versioned ForecastSnapshot (Sprint F8) — one object, replayable.

`build_forecast_snapshot(...)` assembles everything F1–F7 produced into a
single versioned object with a stable field set, so a consumer can depend on
its shape and a stored snapshot can be compared against a fresh one.

**A snapshot is an identity, not just a record.** MEASURED: a forecast is
deterministic — two runs of the same request digest identically
(`6deb5e9815782497` both times). `snapshot_digest` makes that usable, so
"did this forecast change?" is answerable rather than assumed.

**Three requested fields have no honest producer today**, and the contract
says so per field rather than filling them:

    model_versions       NO_MODEL — nothing is registered
    expected_return      needs a trained model; F3 emits zero values
    model_contributions  a naming conflict with a measurement F6 made

**Absent is not zero, and that is the same hazard every forecast sprint has
guarded.** `expected_return: None` would coalesce to 0.0 under the dashboard's
`Number(x ?? 0)` idiom and render as "flat" — a confident claim about a
quantity nobody computed. So an absent field carries a STATUS and a REASON,
never a placeholder value.

**The `model_contributions` conflict, stated plainly.** F6 measured that
contributions cannot be reported: regime alone +0.064, chart alone +0.067, sum
+0.131, against an ACTUAL joint effect of +0.060 — the filters overlap and
double-count by more than 2x. F6's validator bars the word outright. This
field therefore carries F6's decomposition UNCHANGED and declares that its
name is not to be read literally. Renaming F6's output to fit would undo a
measurement; omitting the field would break the requested contract.

**Composition, not restatement (W5).** Every part comes from the sprint that
owns it — F4's interval, F5's event context, F6's decomposition, F7's
confidence, M1's `feature_surface_digest`, N4's regime. This module adds
assembly, versioning and warnings; it recomputes nothing.

**Warnings are surfaced at the top level.** A consumer that renders only the
headline must still see that there is no trained model, that the evidence was
inferred, or that the sample was thin — those facts are useless buried three
objects deep.
"""

from __future__ import annotations

import hashlib
import json
import logging

from core.config import (
    CONDITIONAL_FORECAST_VERSION,
    CONDITIONAL_MIN_SAMPLES_INTERVAL,
    EVENT_FORECAST_CONTRACT_VERSION,
    FORECAST_CONFIDENCE_VERSION,
    FORECAST_DECOMPOSITION_VERSION,
    FORECAST_SNAPSHOT_CONTRACT_VERSION,
    FORECAST_SNAPSHOT_VERSION,
    FORECAST_TARGET_CONTRACTS,
    JOINT_FORECAST_CONTRACT_VERSION,
    SNAPSHOT_CONTRIBUTIONS_NOTE,
    SNAPSHOT_FIELDS,
    SNAPSHOT_LOW_CONFIDENCE_THRESHOLD,
    SNAPSHOT_MIN_OBSERVED_SHARE,
    SNAPSHOT_REQUIRED_FIELDS,
    SNAPSHOT_STATUS_ABSENT,
    SNAPSHOT_STATUS_PRESENT,
    SNAPSHOT_STATUS_REFUSED,
    SNAPSHOT_UNAVAILABLE_FIELDS,
    SNAPSHOT_WARNING_ABSENT_FIELDS,
    SNAPSHOT_WARNING_INFERRED_EVIDENCE,
    SNAPSHOT_WARNING_LOW_CONFIDENCE,
    SNAPSHOT_WARNING_NO_MODEL,
    SNAPSHOT_WARNING_THIN_SAMPLE,
)

LOGGER = logging.getLogger("core.forecast_snapshot")


class ForecastSnapshotError(ValueError):
    """Raised when a snapshot request violates the F8 contract."""


def _field(status: str, value=None, reason: str = "", **detail) -> dict:
    """One contract field. A non-PRESENT field never carries a value.

    The guard is the point: a placeholder on an absent field is exactly how a
    consumer ends up rendering a number nobody computed.
    """
    row = {"status": status, "reason": reason}
    if status == SNAPSHOT_STATUS_PRESENT:
        row["value"] = value
        row.update(detail)
    return row


def _absent(name: str) -> dict:
    return _field(
        SNAPSHOT_STATUS_ABSENT,
        reason=SNAPSHOT_UNAVAILABLE_FIELDS.get(
            name, "no producer exists for this field yet"
        ),
    )


def forecast_versions() -> dict:
    """Every version that shaped this snapshot, so a replay can be pinned."""
    return {
        "snapshot": FORECAST_SNAPSHOT_VERSION,
        "contract": FORECAST_SNAPSHOT_CONTRACT_VERSION,
        "joint": JOINT_FORECAST_CONTRACT_VERSION,
        "conditional": CONDITIONAL_FORECAST_VERSION,
        "event": EVENT_FORECAST_CONTRACT_VERSION,
        "decomposition": FORECAST_DECOMPOSITION_VERSION,
        "confidence": FORECAST_CONFIDENCE_VERSION,
        "targets": sorted(FORECAST_TARGET_CONTRACTS),
    }


def snapshot_digest(snapshot: dict) -> str:
    """A stable identity for a snapshot, excluding its own digest.

    MEASURED: a forecast is deterministic, so two runs of the same request
    digest identically. That makes "did this change?" answerable.
    """
    payload = {k: v for k, v in (snapshot or {}).items() if k != "digest"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _feature_digest(chart_state: dict | None) -> dict:
    """Compose M1's digest; never a second hashing scheme (W5)."""
    if not chart_state:
        return _field(
            SNAPSHOT_STATUS_REFUSED,
            reason="no feature surface was supplied to digest",
        )
    try:
        from core.feature_registry import (
            feature_surface_digest,
            load_feature_registry,
        )

        registry = load_feature_registry()
        surface = {
            name: {"value": value} for name, value in sorted(chart_state.items())
        }
        digest = feature_surface_digest(surface, registry)
    except Exception as exc:  # a registry failure must not fake a digest
        LOGGER.warning("feature digest unavailable: %s", exc)
        return _field(
            SNAPSHOT_STATUS_REFUSED,
            reason=f"the M1 feature registry was unavailable: {type(exc).__name__}",
        )
    return _field(
        SNAPSHOT_STATUS_PRESENT, value=digest,
        reason="sha256 over the feature contracts, via M1 feature_surface_digest",
        fields=len(chart_state),
    )


def _collect_warnings(
    *,
    confidence: dict | None,
    event_forecast: dict | None,
    absent_fields: list[str],
    samples: int,
) -> list[dict]:
    """Facts a consumer must see even if it renders only the headline."""
    warnings: list[dict] = []

    if "model_versions" in absent_fields or "expected_return" in absent_fields:
        warnings.append({
            "code": SNAPSHOT_WARNING_NO_MODEL,
            "detail": (
                "no trained forecasting model is registered; every value here "
                "is an empirical base rate over historical slices, not a "
                "model output"
            ),
        })
    if absent_fields:
        warnings.append({
            "code": SNAPSHOT_WARNING_ABSENT_FIELDS,
            "detail": f"absent: {', '.join(sorted(absent_fields))}",
            "fields": sorted(absent_fields),
        })

    value = (confidence or {}).get("confidence")
    if value is not None and float(value) < SNAPSHOT_LOW_CONFIDENCE_THRESHOLD:
        warnings.append({
            "code": SNAPSHOT_WARNING_LOW_CONFIDENCE,
            "detail": (
                f"confidence {float(value):.3f} is below "
                f"{SNAPSHOT_LOW_CONFIDENCE_THRESHOLD}, bound by "
                f"{(confidence or {}).get('binding_factor')}"
            ),
            "binding_factor": (confidence or {}).get("binding_factor"),
        })

    matches = ((event_forecast or {}).get("stages") or {}).get("matches") or {}
    observed = matches.get("observed_share")
    if observed is not None and float(observed) < SNAPSHOT_MIN_OBSERVED_SHARE:
        warnings.append({
            "code": SNAPSHOT_WARNING_INFERRED_EVIDENCE,
            "detail": (
                f"only {float(observed):.0%} of the evidence was OBSERVED; the "
                f"rest was dated by inference, whose MEASURED precision is 0.65"
            ),
            "observed_share": float(observed),
        })

    if 0 < samples < CONDITIONAL_MIN_SAMPLES_INTERVAL:
        warnings.append({
            "code": SNAPSHOT_WARNING_THIN_SAMPLE,
            "detail": (
                f"{samples} observation(s), below F4's weakest claim floor of "
                f"{CONDITIONAL_MIN_SAMPLES_INTERVAL}"
            ),
            "samples": samples,
        })
    return warnings


def build_forecast_snapshot(
    ticker: str,
    as_of,
    horizon: str,
    *,
    event_forecast: dict | None = None,
    conditional_cell: dict | None = None,
    confidence: dict | None = None,
    decomposition: dict | None = None,
    regime: str | None = None,
    chart_state: dict | None = None,
    evidence: dict | None = None,
) -> dict:
    """Assemble the versioned snapshot from what F1–F7 produced.

    Every part is COMPOSED from the sprint that owns it. This function adds
    assembly, versioning and top-level warnings; it recomputes nothing.
    """
    if not str(ticker or "").strip():
        raise ForecastSnapshotError("a ticker is required")
    if not str(horizon or "").strip():
        raise ForecastSnapshotError("a horizon is required")

    source = event_forecast or {}
    cell = conditional_cell or {}
    samples = int(source.get("samples") or cell.get("samples") or 0)

    fields: dict[str, dict] = {}

    fields["ticker"] = _field(SNAPSHOT_STATUS_PRESENT, value=str(ticker).upper())
    fields["as_of"] = _field(SNAPSHOT_STATUS_PRESENT, value=str(as_of))
    fields["horizon"] = _field(SNAPSHOT_STATUS_PRESENT, value=str(horizon))
    fields["forecast_version"] = _field(
        SNAPSHOT_STATUS_PRESENT, value=forecast_versions(),
        reason="every version that shaped this snapshot, for pinned replay",
    )

    # -- the two fields with no producer -----------------------------------
    fields["model_versions"] = _absent("model_versions")
    fields["expected_return"] = _absent("expected_return")

    # -- probability_up ----------------------------------------------------
    probability = source.get("value") if "value" in source else cell.get("value")
    if probability is not None:
        fields["probability_up"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=round(float(probability), 6),
            reason="empirical base rate over historical slices, not a model output",
            samples=samples,
            claim=source.get("claim") or cell.get("claim"),
        )
    else:
        fields["probability_up"] = _field(
            SNAPSHOT_STATUS_REFUSED,
            reason=(
                source.get("stages", {}).get("forecast", {}).get("reason")
                or cell.get("reason")
                or "the underlying forecast published no point estimate"
            ),
            claim=source.get("claim") or cell.get("claim"),
        )

    # -- prediction_interval ----------------------------------------------
    interval = source.get("interval") or cell.get("interval")
    if interval:
        fields["prediction_interval"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=interval,
            reason="Wilson score interval; uncertainty from sample size",
        )
    else:
        fields["prediction_interval"] = _field(
            SNAPSHOT_STATUS_REFUSED,
            reason=(
                "no interval was published; a refused forecast supplies no "
                "uncertainty either, because an interval beside a withheld "
                "estimate is an estimate by another name"
            ),
        )

    # -- confidence (F7) ---------------------------------------------------
    if confidence:
        fields["confidence"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=confidence,
            reason=(
                "F7 forecast confidence, bounded by its weakest factor. This "
                "is NOT P(up): a probability of 0.50 from 1,000 observations "
                "is more reliable than one of 1.00 from 2"
            ),
        )
    else:
        fields["confidence"] = _field(
            SNAPSHOT_STATUS_REFUSED, reason="no confidence assessment was supplied",
        )

    # -- regime (N4) -------------------------------------------------------
    resolved_regime = regime or (
        (source.get("stages") or {}).get("regime", {}).get("regime")
    )
    if resolved_regime:
        fields["regime"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=resolved_regime,
            reason="the governed five-state regime from N4",
            agreement=(source.get("stages") or {}).get("regime", {}).get("agreement"),
        )
    else:
        fields["regime"] = _field(
            SNAPSHOT_STATUS_REFUSED, reason="no regime was established at as_of",
        )

    # -- event_context (F5) ------------------------------------------------
    if event_forecast:
        fields["event_context"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=event_forecast,
            reason="the F5 event-conditioned pipeline, stage by stage",
        )
    else:
        fields["event_context"] = _field(
            SNAPSHOT_STATUS_REFUSED, reason="no event-conditioned forecast was supplied",
        )

    # -- feature_digest (M1) ----------------------------------------------
    fields["feature_digest"] = _feature_digest(chart_state)

    # -- evidence ----------------------------------------------------------
    if evidence:
        fields["evidence"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=evidence,
            reason="identifiers for the records this forecast rests on",
        )
    else:
        fields["evidence"] = _field(
            SNAPSHOT_STATUS_REFUSED,
            reason="no evidence references were supplied for this forecast",
        )

    # -- model_contributions (F6, under a name that is not literal) --------
    if decomposition:
        fields["model_contributions"] = _field(
            SNAPSHOT_STATUS_PRESENT, value=decomposition,
            reason=SNAPSHOT_CONTRIBUTIONS_NOTE,
            additive=False,
        )
    else:
        fields["model_contributions"] = _field(
            SNAPSHOT_STATUS_REFUSED, reason="no decomposition was supplied",
        )

    absent = [
        name for name, row in fields.items()
        if row["status"] == SNAPSHOT_STATUS_ABSENT
    ]
    warnings = _collect_warnings(
        confidence=confidence, event_forecast=event_forecast,
        absent_fields=absent, samples=samples,
    )
    fields["warnings"] = _field(
        SNAPSHOT_STATUS_PRESENT, value=warnings,
        reason="surfaced at the top level so a headline-only consumer sees them",
    )

    snapshot = {
        "snapshot_version": FORECAST_SNAPSHOT_VERSION,
        "contract_version": FORECAST_SNAPSHOT_CONTRACT_VERSION,
        "field_order": list(SNAPSHOT_FIELDS),
        "fields": {name: fields[name] for name in SNAPSHOT_FIELDS},
        "present": [
            n for n in SNAPSHOT_FIELDS
            if fields[n]["status"] == SNAPSHOT_STATUS_PRESENT
        ],
        "absent": absent,
        "refused": [
            n for n in SNAPSHOT_FIELDS
            if fields[n]["status"] == SNAPSHOT_STATUS_REFUSED
        ],
        "warning_codes": [w["code"] for w in warnings],
    }
    snapshot["digest"] = snapshot_digest(snapshot)
    return snapshot


def snapshot_value(snapshot: dict, field: str):
    """The value of one field, or None when it is not PRESENT.

    The accessor exists so consumers do not reach into `fields[...]["value"]`
    and get a KeyError on an absent field — the shape rule means the key is
    genuinely missing, not None.
    """
    row = (snapshot.get("fields") or {}).get(field) or {}
    return row.get("value") if row.get("status") == SNAPSHOT_STATUS_PRESENT else None


def forecast_snapshot_problems(snapshot: dict) -> list[str]:
    """Validate a snapshot against the F8 contract."""
    problems: list[str] = []
    if not isinstance(snapshot, dict):
        return ["snapshot must be a dict"]

    for field in ("snapshot_version", "contract_version", "digest"):
        if not snapshot.get(field):
            problems.append(f"snapshot field {field!r} missing/empty")

    fields = snapshot.get("fields") or {}
    if list(fields) != list(SNAPSHOT_FIELDS):
        problems.append(
            "the snapshot does not report every contract field in order — a "
            "consumer relies on every name being present, with absence "
            "explicit rather than missing"
        )

    for name, row in fields.items():
        status = row.get("status")
        if status not in (
            SNAPSHOT_STATUS_PRESENT, SNAPSHOT_STATUS_ABSENT, SNAPSHOT_STATUS_REFUSED
        ):
            problems.append(f"{name}: unknown status {status!r}")
        if status != SNAPSHOT_STATUS_PRESENT and "value" in row:
            problems.append(
                f"{name}: a {status} field carries a value — it would coalesce "
                f"to a number in a consumer and render a quantity nobody "
                f"computed"
            )
        if status != SNAPSHOT_STATUS_PRESENT and not row.get("reason"):
            problems.append(f"{name}: a {status} field does not explain itself")

    for name in SNAPSHOT_REQUIRED_FIELDS:
        row = fields.get(name) or {}
        if row.get("status") != SNAPSHOT_STATUS_PRESENT:
            problems.append(
                f"{name}: is required, but reports {row.get('status')!r} — a "
                f"snapshot without it is not a degraded forecast, it is not a "
                f"forecast"
            )

    for name in SNAPSHOT_UNAVAILABLE_FIELDS:
        row = fields.get(name) or {}
        if row.get("status") == SNAPSHOT_STATUS_PRESENT:
            problems.append(
                f"{name}: reports PRESENT although it is declared unavailable — "
                f"if a producer now exists, remove it from "
                f"SNAPSHOT_UNAVAILABLE_FIELDS deliberately"
            )

    contributions = fields.get("model_contributions") or {}
    if contributions.get("status") == SNAPSHOT_STATUS_PRESENT:
        if contributions.get("additive") is not False:
            problems.append(
                "model_contributions does not declare itself non-additive — "
                "the field name invites exactly the reading F6 measured wrong"
            )
        if "NOT contributions" not in (contributions.get("reason") or ""):
            problems.append(
                "model_contributions does not say its name is not literal"
            )

    if snapshot.get("digest") and snapshot_digest(snapshot) != snapshot["digest"]:
        problems.append(
            "the stored digest does not match the snapshot's content — it "
            "cannot be used to answer whether the forecast changed"
        )

    absent = snapshot.get("absent") or []
    codes = snapshot.get("warning_codes") or []
    if absent and SNAPSHOT_WARNING_ABSENT_FIELDS not in codes:
        problems.append(
            "fields are absent but no warning says so at the top level — a "
            "headline-only consumer would not learn of the gap"
        )
    return problems


def render_snapshot(snapshot: dict) -> list[dict]:
    """One reading row per contract field, in declared order. Never blank."""
    rows: list[dict] = []
    for name in snapshot.get("field_order") or ():
        row = (snapshot.get("fields") or {}).get(name) or {}
        rows.append(
            {
                "field": name,
                "status": row.get("status"),
                "has_value": "value" in row,
                "reason": row.get("reason") or "",
            }
        )
    return rows


# C4: `snapshot_problems` was defined under that name in THREE modules — this
# one for F8's per-forecast ForecastSnapshot, plus the other two snapshots. They are genuinely different
# objects with different contracts, so this is not a W5 duplicate; the shared
# NAME was the problem. A reader greping it found three functions with no way to
# tell which a call site meant, and importing two into one file would have
# silently shadowed one.
#
# The alias keeps any external caller working. In-repo callers all use the
# explicit name.
snapshot_problems = forecast_snapshot_problems
