MAX_SCORE = 10.0
MIN_SCORE = 0.0
DEFAULT_ACTION = "ANALYSIS_ONLY"
DEFAULT_CONFIDENCE = 0.5

MARKET_FEATURE_VERSION = "market-feature-v1"
CURRENT_SCORE_VERSION = "current-score-v2"
LONG_TERM_SCORE_VERSION = "long-term-score-v2"
NEWS_CONTRACT_VERSION = "news-contract-v1"

# --- Ensemble wiring (W1) ------------------------------------------------------
# The published score is the weighted product of agent contributions, not an
# independent hand-built blend. Both weight sets share an identical key set,
# must each sum to 1.0 (validated at import time), and intentionally differ
# per horizon: business quality matters more to the structural view than to
# the tactical one. Agents without a live implementation hold an explicit 0.0
# weight — presence in the dict is the contract; absence fails at import.
ENSEMBLE_VERSION = "ensemble-v1"

ENSEMBLE_WEIGHTS_CURRENT = {
    "market_data": 0.0,          # informational only: feeds confidence/gates
    "technical_analysis": 0.85,  # current-time technical view
    "fundamental_analysis": 0.15,
    "news_intelligence": 0.0,    # UNAVAILABLE stub; dedicated weight lands with Sprint N1
    "sentiment": 0.0,            # not implemented
    "macroeconomic": 0.0,        # not implemented
    "market_regime": 0.0,        # not implemented; regime gates via risk policy
}

ENSEMBLE_WEIGHTS_LONG = {
    "market_data": 0.0,
    "technical_analysis": 0.75,  # long-term structural technical view
    "fundamental_analysis": 0.25,
    "news_intelligence": 0.0,
    "sentiment": 0.0,
    "macroeconomic": 0.0,
    "market_regime": 0.0,
}


def _validate_ensemble_weights(name: str, weights: dict[str, float]) -> None:
    """Import-time guard: complete key set, non-negative, summing to 1.0."""
    if not weights:
        raise ValueError(f"{name} must not be empty")
    if set(weights) != set(ENSEMBLE_WEIGHTS_CURRENT):
        raise ValueError(
            f"{name} keys must match ENSEMBLE_WEIGHTS_CURRENT exactly: "
            f"missing={sorted(set(ENSEMBLE_WEIGHTS_CURRENT) - set(weights))} "
            f"extra={sorted(set(weights) - set(ENSEMBLE_WEIGHTS_CURRENT))}"
        )
    negative = {agent: weight for agent, weight in weights.items() if float(weight) < 0.0}
    if negative:
        raise ValueError(f"{name} weights must be non-negative, got {negative}")
    total = float(sum(weights.values()))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1.0 (tolerance 1e-9), got {total!r}")


_validate_ensemble_weights("ENSEMBLE_WEIGHTS_CURRENT", ENSEMBLE_WEIGHTS_CURRENT)
_validate_ensemble_weights("ENSEMBLE_WEIGHTS_LONG", ENSEMBLE_WEIGHTS_LONG)

# --- Risk policy (W2) -----------------------------------------------------------
# Single source of truth for every governance threshold. The evaluator lives in
# core/risk_policy.py and is the only consumer; nothing else may hard-code these
# limits. severity "veto" blocks PAPER posture; "warning" is visible but does
# not block. Missing/None inputs evaluate to triggered rules — fail-closed.
RISK_POLICY_VERSION = "risk-policy-v2"

# --- Audit policy (W3) ----------------------------------------------------------
# The auditor independently verifies that a decision is provable: evidence
# sufficiency, hash integrity, determinism, calibration sanity, and ensemble
# consistency. Its evaluator lives in core/audit_policy.py; a failed veto-
# severity check appends the "auditor_veto" reason and blocks PAPER posture.
AUDIT_POLICY_VERSION = "audit-policy-v1"

RISK_POLICY_V2 = {
    "data_quality_below_threshold": {
        "severity": "veto",
        "minimum_market_data_quality": 60.0,
    },
    "market_source_confidence_below_threshold": {
        "severity": "veto",
        "minimum_market_source_confidence": 0.7,
    },
    "future_dated_market_data": {
        "severity": "veto",
    },
    "future_dated_fundamental_payload": {
        "severity": "veto",
    },
    "fundamental_source_confidence_below_threshold": {
        "severity": "veto",
        "minimum_fundamental_source_confidence": 0.7,
    },
    "score_below_threshold": {
        "severity": "veto",
        "minimum_score": 5.5,
    },
    "analysis_only_mode": {
        "severity": "veto",
    },
    "confidence_below_minimum": {
        "severity": "veto",
        "minimum_confidence": 0.35,
    },
    "confidence_penalty_budget_exceeded": {
        "severity": "warning",
        "maximum_total_penalty": 0.15,
    },
    "freshness_degraded": {
        "severity": "warning",
        "minimum_freshness_factor": 0.5,
    },
    "volatility_regime_elevated": {
        "severity": "warning",
        "minimum_volatility_regime_factor": 0.3,
    },
}


def _validate_risk_policy() -> None:
    """Import-time guard: every rule is a non-empty spec with valid severity."""
    if not RISK_POLICY_V2:
        raise ValueError("RISK_POLICY_V2 must contain at least one rule")
    for rule_id, spec in RISK_POLICY_V2.items():
        if not isinstance(spec, dict) or not spec:
            raise ValueError(f"RISK_POLICY_V2 rule {rule_id!r} must be a non-empty spec")
        if spec.get("severity") not in {"veto", "warning"}:
            raise ValueError(
                f"RISK_POLICY_V2 rule {rule_id!r} severity must be 'veto' or 'warning', "
                f"got {spec.get('severity')!r}"
            )


_validate_risk_policy()

# Evidence-based confidence model (see core.score_engine._compute_confidence).
# Each factor produces a value in [0, 1]; the confidence is the weighted sum
# minus explicit risk penalties, clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CAP].
# Weights sum to 1.0 so the baseline stays interpretable as a percentage.
CONFIDENCE_VERSION = "evidence-confidence-v2"

CONFIDENCE_WEIGHT_DATA_QUALITY = 0.25
CONFIDENCE_WEIGHT_SOURCE_RELIABILITY = 0.20
CONFIDENCE_WEIGHT_SIGNAL_AGREEMENT = 0.20
CONFIDENCE_WEIGHT_FRESHNESS = 0.15
CONFIDENCE_WEIGHT_HISTORY_COVERAGE = 0.10
CONFIDENCE_WEIGHT_VOLATILITY_REGIME = 0.10

# Calendar-age gap between the newest bar used and as_of before freshness decays,
# and how many calendar days after that grace window reach zero freshness credit.
CONFIDENCE_FRESHNESS_GRACE_DAYS = 4
CONFIDENCE_FRESHNESS_DECAY_DAYS = 26

# Minimum number of valid daily bars for full long-window (200d MA) coverage.
CONFIDENCE_FULL_COVERAGE_BARS = 230

# Daily-return standard deviation treated as fully calm versus fully chaotic.
CONFIDENCE_VOL_CALM_DAILY_STD = 0.015
CONFIDENCE_VOL_CHAOTIC_DAILY_STD = 0.060

# Dead-zone half-width used when reading factor directions so tiny moves do not
# flip a signal between bullish/bearish arbitrarily.
CONFIDENCE_SIGNAL_DEADZONE_RATIO = 0.002

# Named penalties applied once per condition, replacing the previous flat -0.20
# per category. Scaled to reflect how much each condition undermines the result.
RISK_FLAG_CONFIDENCE_PENALTIES = {
    "Weak momentum": 0.10,
    "Low volume": 0.08,
    "Downtrend": 0.12,
    "RSI extreme": 0.06,
}
FUNDAMENTAL_SOURCE_PENALTY = 0.10
GOVERNANCE_RISK_GATE_PENALTY = 0.05

CONFIDENCE_FLOOR = 0.10
CONFIDENCE_CAP = 0.95
