"""Fail-closed risk policy evaluation (W2).

The policy table lives in `core.config.RISK_POLICY_V2`; this module is the
only evaluator. Every rule produces a structured
`{rule_id, severity, triggered, detail}` record. Missing or None inputs
trigger the dependent rule instead of passing it — fail-closed by
construction. Pure and deterministic: identical contexts produce identical
evaluations, so decisions stay replayable.
"""

from __future__ import annotations

from core.config import RISK_POLICY_VERSION, RISK_POLICY_V2

_SEVERITY_VETO = "veto"


def _factor_value(confidence_breakdown: dict | None, name: str) -> float | None:
    """Read a named factor value from the confidence breakdown, if present."""
    for factor in (confidence_breakdown or {}).get("factors", []):
        if factor.get("name") == name:
            value = factor.get("value")
            return None if value is None else float(value)
    return None


def _as_number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value) -> bool | None:
    if value is None:
        return None
    return bool(value)


def evaluate_risk_policy(context: dict, policy: dict | None = None) -> dict:
    """Evaluate every rule in the policy table against a decision context.

    Expected context keys (a missing key triggers the rules that read it):
        market_data_quality, market_source_confidence, market_timestamp_valid,
        fundamental_point_in_time_valid, fundamental_source_confidence,
        fundamental_source_status, score, action, confidence,
        confidence_breakdown

    Returns `{"policy_version", "rules", "veto", "veto_rule_ids",
    "warning_rule_ids"}` where `veto` is True when any veto-severity rule is
    triggered. Callers MUST treat `veto=True` as blocking PAPER posture.
    """
    policy = dict(policy) if policy is not None else dict(RISK_POLICY_V2)
    rules: list[dict] = []

    def add(rule_id: str, triggered: bool, detail: str) -> None:
        spec = policy[rule_id]
        rules.append({
            "rule_id": rule_id,
            "severity": spec["severity"],
            "triggered": bool(triggered),
            "detail": detail,
        })

    # data_quality_below_threshold -------------------------------------------
    threshold = float(policy["data_quality_below_threshold"]["minimum_market_data_quality"])
    quality = _as_number(context.get("market_data_quality"))
    if quality is None:
        add("data_quality_below_threshold", True, "market data quality missing (fail-closed)")
    elif quality < threshold:
        add("data_quality_below_threshold", True, f"quality {quality:.2f} < {threshold:.2f}")
    else:
        add("data_quality_below_threshold", False, f"quality {quality:.2f} >= {threshold:.2f}")

    # market_source_confidence_below_threshold --------------------------------
    threshold = float(policy["market_source_confidence_below_threshold"]["minimum_market_source_confidence"])
    confidence = _as_number(context.get("market_source_confidence"))
    if confidence is None:
        add("market_source_confidence_below_threshold", True, "market source confidence missing (fail-closed)")
    elif confidence < threshold:
        add("market_source_confidence_below_threshold", True, f"confidence {confidence:.2f} < {threshold:.2f}")
    else:
        add("market_source_confidence_below_threshold", False, f"confidence {confidence:.2f} >= {threshold:.2f}")

    # future_dated_market_data --------------------------------------------------
    timestamp_valid = _as_bool(context.get("market_timestamp_valid"))
    if timestamp_valid is None:
        add("future_dated_market_data", True, "market timestamp validity missing (fail-closed)")
    elif not timestamp_valid:
        add("future_dated_market_data", True, "market payload contains bars newer than as_of")
    else:
        add("future_dated_market_data", False, "market timestamps valid")

    # future_dated_fundamental_payload -------------------------------------------
    pit_valid = _as_bool(context.get("fundamental_point_in_time_valid"))
    if pit_valid is None:
        add("future_dated_fundamental_payload", True, "fundamental point-in-time validity missing (fail-closed)")
    elif not pit_valid:
        add("future_dated_fundamental_payload", True, "fundamental payload is dated after as_of")
    else:
        add("future_dated_fundamental_payload", False, "fundamental payload point-in-time valid")

    # fundamental_source_confidence_below_threshold -------------------------------
    threshold = float(policy["fundamental_source_confidence_below_threshold"]["minimum_fundamental_source_confidence"])
    fundamental_confidence = _as_number(context.get("fundamental_source_confidence"))
    fundamental_status = context.get("fundamental_source_status")
    if fundamental_confidence is None:
        add("fundamental_source_confidence_below_threshold", True, "fundamental source confidence missing (fail-closed)")
    elif fundamental_confidence < threshold and fundamental_status != "live_provider":
        add("fundamental_source_confidence_below_threshold", True, f"confidence {fundamental_confidence:.2f} < {threshold:.2f} and status {fundamental_status!r} is not live_provider")
    else:
        add("fundamental_source_confidence_below_threshold", False, f"confidence {fundamental_confidence:.2f} with status {fundamental_status!r}")

    # score_below_threshold -------------------------------------------------------
    threshold = float(policy["score_below_threshold"]["minimum_score"])
    score = _as_number(context.get("score"))
    if score is None:
        add("score_below_threshold", True, "score missing (fail-closed)")
    elif score < threshold:
        add("score_below_threshold", True, f"score {score:.2f} < {threshold:.2f}")
    else:
        add("score_below_threshold", False, f"score {score:.2f} >= {threshold:.2f}")

    # analysis_only_mode ------------------------------------------------------------
    action = context.get("action")
    if action is None:
        add("analysis_only_mode", True, "action missing (fail-closed)")
    elif str(action) == "ANALYSIS_ONLY":
        add("analysis_only_mode", True, "scoring engine held the decision in analysis-only mode")
    else:
        add("analysis_only_mode", False, f"action {action!r} is not analysis-only")

    # confidence_below_minimum --------------------------------------------------------
    threshold = float(policy["confidence_below_minimum"]["minimum_confidence"])
    confidence_value = _as_number(context.get("confidence"))
    if confidence_value is None:
        add("confidence_below_minimum", True, "confidence missing (fail-closed)")
    elif confidence_value < threshold:
        add("confidence_below_minimum", True, f"confidence {confidence_value:.2f} < {threshold:.2f}")
    else:
        add("confidence_below_minimum", False, f"confidence {confidence_value:.2f} >= {threshold:.2f}")

    # confidence_penalty_budget_exceeded (warning) --------------------------------------
    threshold = float(policy["confidence_penalty_budget_exceeded"]["maximum_total_penalty"])
    total_penalty = _as_number((context.get("confidence_breakdown") or {}).get("total_penalty"))
    if total_penalty is None:
        add("confidence_penalty_budget_exceeded", True, "confidence breakdown unavailable (fail-closed)")
    elif total_penalty > threshold:
        add("confidence_penalty_budget_exceeded", True, f"penalties {total_penalty:.2f} > {threshold:.2f}")
    else:
        add("confidence_penalty_budget_exceeded", False, f"penalties {total_penalty:.2f} <= {threshold:.2f}")

    # freshness_degraded (warning) ---------------------------------------------------------
    threshold = float(policy["freshness_degraded"]["minimum_freshness_factor"])
    freshness = _factor_value(context.get("confidence_breakdown"), "freshness")
    if freshness is None:
        add("freshness_degraded", True, "freshness factor unavailable (fail-closed)")
    elif freshness < threshold:
        add("freshness_degraded", True, f"freshness {freshness:.2f} < {threshold:.2f}")
    else:
        add("freshness_degraded", False, f"freshness {freshness:.2f} >= {threshold:.2f}")

    # volatility_regime_elevated (warning) ---------------------------------------------------
    threshold = float(policy["volatility_regime_elevated"]["minimum_volatility_regime_factor"])
    volatility_factor = _factor_value(context.get("confidence_breakdown"), "volatility_regime")
    if volatility_factor is None:
        add("volatility_regime_elevated", True, "volatility regime factor unavailable (fail-closed)")
    elif volatility_factor < threshold:
        add("volatility_regime_elevated", True, f"volatility regime {volatility_factor:.2f} < {threshold:.2f}")
    else:
        add("volatility_regime_elevated", False, f"volatility regime {volatility_factor:.2f} >= {threshold:.2f}")

    veto_rule_ids = [rule["rule_id"] for rule in rules if rule["triggered"] and rule["severity"] == _SEVERITY_VETO]
    warning_rule_ids = [rule["rule_id"] for rule in rules if rule["triggered"] and rule["severity"] != _SEVERITY_VETO]
    return {
        "policy_version": RISK_POLICY_VERSION,
        "rules": rules,
        "veto": bool(veto_rule_ids),
        "veto_rule_ids": veto_rule_ids,
        "warning_rule_ids": warning_rule_ids,
    }
