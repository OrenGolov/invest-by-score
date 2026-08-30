"""Audit policy evaluation (W3): the performance auditor as a validator.

The auditor independently verifies that a decision is provable and
replayable. Checks emit structured `{check_id, severity, passed, detail}`
findings; any failed veto-severity check sets `veto=True`, which the
orchestrator surfaces as the `auditor_veto` reason (blocking PAPER).
Missing or None inputs fail the dependent check — fail-closed by
construction. Pure and deterministic.
"""

from __future__ import annotations

import json

from core.config import AUDIT_POLICY_VERSION, CONFIDENCE_CAP, CONFIDENCE_FLOOR

_CALIBRATION_TOLERANCE = 0.005
_WEIGHT_SUM_TOLERANCE = 1e-6


def stable_hash(payload) -> str:
    """Canonical, stable hash used across the pipeline (audit-owned)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deep_copy_json(value):
    return json.loads(json.dumps(value, default=str))


def _normalise_result(result_dict):
    """Strip per-run audit artifacts so two builds compare cleanly."""
    normalised = _deep_copy_json(result_dict)
    replay = normalised.get("replay_metadata")
    if isinstance(replay, dict):
        replay.pop("audit_event_id", None)
    return normalised


def _first_difference(left, right, path="$"):
    """Return a human-readable description of the first difference, or None."""
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: present only on second build"
            if key not in right:
                return f"{path}.{key}: present only on first build"
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found = _first_difference(left_item, right_item, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def evaluate_audit_policy(context: dict) -> dict:
    """Evaluate every audit check against the decision context.

    Expected context keys (a missing key fails the checks that read it):
        agents (list of agent dicts, auditor excluded),
        expected_input_hashes (agent -> hash),
        snapshot, snapshot_hash, replay_hash,
        first_result, second_result (full ScoreResult dicts),
        confidence, confidence_breakdown, ensemble_breakdown,
        current_time_score, long_term_score, governance, evidence_ledger
    """

    findings: list[dict] = []

    def add(check_id: str, severity: str, passed: bool, detail: str) -> None:
        findings.append({
            "check_id": check_id,
            "severity": severity,
            "passed": bool(passed),
            "detail": detail,
        })

    agents = context.get("agents")

    # evidence_sufficiency (veto) ------------------------------------------------
    missing_evidence = []
    if not isinstance(agents, list) or not agents:
        add("evidence_sufficiency", "veto", False, "agent outputs missing (fail-closed)")
    else:
        for agent in agents:
            evidence = agent.get("evidence") or []
            if agent.get("status") == "UNAVAILABLE":
                continue
            if not evidence:
                missing_evidence.append(f"{agent.get('agent')}: no evidence entries")
                continue
            for entry in evidence:
                if not str(entry.get("source_record_id", "")).strip():
                    missing_evidence.append(f"{agent.get('agent')}: evidence entry without source_record_id")
    add(
        "evidence_sufficiency",
        "veto",
        not missing_evidence,
        "all eligible agents cite source records" if not missing_evidence else "; ".join(missing_evidence),
    )

    # agent_input_hash_integrity (veto) ---------------------------------------------
    expected_hashes = context.get("expected_input_hashes") or {}
    hash_failures = []
    if not isinstance(agents, list) or not agents:
        hash_failures.append("agent outputs missing (fail-closed)")
    else:
        for agent in agents:
            agent_name = str(agent.get("agent", ""))
            claimed = str(agent.get("input_hash", "") or "")
            if not claimed:
                hash_failures.append(f"{agent_name}: empty input_hash")
                continue
            expected = expected_hashes.get(agent_name)
            if expected is not None and claimed != str(expected):
                hash_failures.append(f"{agent_name}: input_hash does not match its declared inputs")
    add(
        "agent_input_hash_integrity",
        "veto",
        not hash_failures,
        "every agent input_hash matches its declared inputs" if not hash_failures else "; ".join(hash_failures),
    )

    # snapshot_hash_integrity (veto) ---------------------------------------------------
    snapshot = context.get("snapshot")
    claimed_snapshot_hash = context.get("snapshot_hash")
    if not snapshot or not claimed_snapshot_hash:
        add("snapshot_hash_integrity", "veto", False, "snapshot or claimed hash missing (fail-closed)")
    else:
        recomputed = stable_hash(snapshot)
        hashes_match = recomputed == str(claimed_snapshot_hash)
        add(
            "snapshot_hash_integrity",
            "veto",
            hashes_match,
            "snapshot hash recomputes from the payload" if hashes_match else "snapshot hash does not match the payload (tamper suspected)",
        )

    # determinism_probe (veto) ------------------------------------------------------------
    first_result = context.get("first_result")
    second_result = context.get("second_result")
    if first_result is None or second_result is None:
        add("determinism_probe", "veto", False, "replay builds unavailable (fail-closed)")
    else:
        difference = _first_difference(_normalise_result(first_result), _normalise_result(second_result))
        add(
            "determinism_probe",
            "veto",
            difference is None,
            "two in-process builds are identical modulo audit ids"
            if difference is None
            else f"replayed build diverges: {difference}",
        )

    # calibration_sanity (veto) -----------------------------------------------------------
    confidence = context.get("confidence")
    breakdown = context.get("confidence_breakdown") or {}
    breakdown_value = breakdown.get("value")
    calibration_problems = []
    if confidence is None:
        calibration_problems.append("confidence missing (fail-closed)")
    elif not (CONFIDENCE_FLOOR <= float(confidence) <= CONFIDENCE_CAP):
        calibration_problems.append(f"confidence {float(confidence):.4f} outside [{CONFIDENCE_FLOOR}, {CONFIDENCE_CAP}]")
    if breakdown_value is None:
        calibration_problems.append("confidence breakdown value missing (fail-closed)")
    elif confidence is not None and abs(float(breakdown_value) - float(confidence)) > _CALIBRATION_TOLERANCE:
        calibration_problems.append(f"breakdown value {breakdown_value} disagrees with headline {confidence} beyond {_CALIBRATION_TOLERANCE}")
    if not str(breakdown.get("calculation_version", "")).strip():
        calibration_problems.append("confidence breakdown version missing (fail-closed)")
    add(
        "calibration_sanity",
        "veto",
        not calibration_problems,
        "confidence is in band and agrees with its breakdown" if not calibration_problems else "; ".join(calibration_problems),
    )

    # ensemble_consistency (veto) ------------------------------------------------------------
    ensemble = context.get("ensemble_breakdown") or {}
    ensemble_problems = []
    if not ensemble:
        ensemble_problems.append("ensemble breakdown missing (fail-closed)")
    else:
        if ensemble.get("current_time_score") != context.get("current_time_score"):
            ensemble_problems.append("current_time_score disagrees with ensemble breakdown")
        if ensemble.get("long_term_score") != context.get("long_term_score"):
            ensemble_problems.append("long_term_score disagrees with ensemble breakdown")
        agents_weights = ensemble.get("agents") or {}
        for horizon in ("current", "long"):
            total = sum(float(entry.get(f"effective_weight_{horizon}", 0.0)) for entry in agents_weights.values())
            if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
                ensemble_problems.append(f"effective {horizon} weights sum to {total:.6f}, expected 1.0")
        if ensemble.get("no_eligible_agents"):
            ensemble_problems.append("ensemble reported no eligible agents")
    add(
        "ensemble_consistency",
        "veto",
        not ensemble_problems,
        "ensemble breakdown reconciles with the headline scores" if not ensemble_problems else "; ".join(ensemble_problems),
    )

    # replay_identifiers (veto) -----------------------------------------------------------------
    replay_hash = context.get("replay_hash")
    add("replay_identifiers", "veto", bool(replay_hash), "replay hash present" if replay_hash else "replay hash missing (fail-closed)")

    # evidence_ledger_consistency (warning) ---------------------------------------------------------
    governance = context.get("governance") or {}
    ledger = context.get("evidence_ledger") or {}
    expected_status = "analysis_only" if not governance.get("risk_gate_passed") else "ready"
    actual_status = ledger.get("status")
    add(
        "evidence_ledger_consistency",
        "warning",
        actual_status == expected_status,
        f"evidence ledger status {actual_status!r} matches governance" if actual_status == expected_status else f"evidence ledger status {actual_status!r} != expected {expected_status!r}",
    )

    veto_check_ids = [f["check_id"] for f in findings if not f["passed"] and f["severity"] == "veto"]
    warning_check_ids = [f["check_id"] for f in findings if not f["passed"] and f["severity"] != "veto"]
    return {
        "policy_version": AUDIT_POLICY_VERSION,
        "findings": findings,
        "veto": bool(veto_check_ids),
        "veto_check_ids": veto_check_ids,
        "warning_check_ids": warning_check_ids,
    }

