"""Monitoring foundations (Sprint V7) - pure run-health metrics, no dashboard.

Part 1: version constants, score bins, PSI + confidence drift.
"""

from __future__ import annotations

import math

MONITORING_VERSION = "monitoring-v1"

PSI_NO_CHANGE_MAX = 0.1
PSI_MODERATE_MAX = 0.25

MONITORING_MIN_WINDOW = 30

SCORE_BINS = (0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0001)


def _bin_index(score: float) -> int:
    for index in range(len(SCORE_BINS) - 1):
        if SCORE_BINS[index] <= float(score) < SCORE_BINS[index + 1]:
            return index
    return len(SCORE_BINS) - 2


def _binned_proportions(scores: list[float]) -> list[float]:
    counts = [0] * (len(SCORE_BINS) - 1)
    for score in scores:
        counts[_bin_index(score)] += 1
    total = len(scores)
    return [count / total for count in counts]


def _check_scores(label: str, scores: list) -> None:
    for score in scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"{label} scores must be numeric, got {score!r}")
        if math.isnan(float(score)) or math.isinf(float(score)):
            raise ValueError(f"{label} scores must be finite, got {score!r}")
        if not 0.0 <= float(score) <= 10.0:
            raise ValueError(f"{label} scores must be within [0, 10], got {score!r}")


def score_drift_psi(
    reference_scores: list[float],
    current_scores: list[float],
    min_window: int = MONITORING_MIN_WINDOW,
) -> dict:
    """PSI between a reference and a current score window (pure)."""
    _check_scores("reference", reference_scores)
    _check_scores("current", current_scores)
    if len(reference_scores) < min_window or len(current_scores) < min_window:
        return {
            "psi": None,
            "verdict": "insufficient_data",
            "reason": (
                f"need >= {min_window} observations per window "
                f"(reference={len(reference_scores)}, current={len(current_scores)})"
            ),
            "reference_count": len(reference_scores),
            "current_count": len(current_scores),
        }
    reference = _binned_proportions([float(s) for s in reference_scores])
    current = _binned_proportions([float(s) for s in current_scores])
    epsilon = 1e-4
    psi = 0.0
    terms: list[float] = []
    for ref_share, cur_share in zip(reference, current):
        ref_share = max(ref_share, epsilon)
        cur_share = max(cur_share, epsilon)
        term = (cur_share - ref_share) * math.log(cur_share / ref_share)
        terms.append(round(term, 6))
        psi += term
    psi = round(psi, 6)
    if psi < PSI_NO_CHANGE_MAX:
        verdict = "no_significant_change"
    elif psi <= PSI_MODERATE_MAX:
        verdict = "moderate_shift"
    else:
        verdict = "significant_shift"
    return {
        "psi": psi,
        "verdict": verdict,
        "terms": terms,
        "bins": list(SCORE_BINS),
        "reference_count": len(reference_scores),
        "current_count": len(current_scores),
    }


def confidence_drift(
    reference_confidences: list[float],
    current_confidences: list[float],
    min_window: int = MONITORING_MIN_WINDOW,
) -> dict:
    """Mean-confidence delta (current minus reference), same discipline."""
    for label, values in (("reference", reference_confidences), ("current", current_confidences)):
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"confidence_drift: {label} must be numeric, got {value!r}")
            if math.isnan(float(value)) or math.isinf(float(value)):
                raise ValueError(f"confidence_drift: {label} must be finite, got {value!r}")
    if len(reference_confidences) < min_window or len(current_confidences) < min_window:
        return {
            "delta": None,
            "verdict": "insufficient_data",
            "reason": (
                f"need >= {min_window} observations per window "
                f"(reference={len(reference_confidences)}, current={len(current_confidences)})"
            ),
            "reference_count": len(reference_confidences),
            "current_count": len(current_confidences),
        }
    ref_mean = sum(float(v) for v in reference_confidences) / len(reference_confidences)
    cur_mean = sum(float(v) for v in current_confidences) / len(current_confidences)
    return {
        "delta": round(cur_mean - ref_mean, 6),
        "verdict": "computed",
        "reference_mean": round(ref_mean, 6),
        "current_mean": round(cur_mean, 6),
        "reference_count": len(reference_confidences),
        "current_count": len(current_confidences),
    }


def stale_data_rate(decisions: list[dict]) -> dict:
    """Fraction of decisions with a STALE agent status (pure)."""
    total = len(decisions)
    if not total:
        return {
            "rate": None, "verdict": "insufficient_data",
            "reason": "no decisions observed",
            "stale_count": 0, "decision_count": 0, "unknown_count": 0,
        }
    stale_count = 0
    unknown_count = 0
    for decision in decisions:
        statuses = (decision or {}).get("agent_statuses")
        if not isinstance(statuses, dict) or not statuses:
            unknown_count += 1
            continue
        if any(str(s).upper() == "STALE" for s in statuses.values()):
            stale_count += 1
    known = total - unknown_count
    return {
        "rate": round(stale_count / known, 6) if known else None,
        "verdict": "computed" if known else "insufficient_data",
        "reason": (
            f"{unknown_count} of {total} decisions carry no agent-status metadata"
            if unknown_count else f"{stale_count} of {known} decisions with status metadata are stale"
        ),
        "stale_count": stale_count,
        "decision_count": total,
        "unknown_count": unknown_count,
    }


def veto_rate_by_rule(decisions: list[dict]) -> dict:
    """Per-rule veto fractions over decisions with veto metadata (pure)."""
    total = len(decisions)
    if not total:
        return {
            "rates": {}, "verdict": "insufficient_data",
            "reason": "no decisions observed",
            "decision_count": 0, "known_count": 0, "unknown_count": 0,
        }
    rule_hits: dict[str, int] = {}
    known = 0
    unknown_count = 0
    for decision in decisions:
        veto = (decision or {}).get("veto")
        rule_ids = None
        if isinstance(veto, dict):
            rule_ids = veto.get("rule_ids")
            if rule_ids is None:
                rule_ids = veto.get("veto_rule_ids")
        if not isinstance(rule_ids, list):
            unknown_count += 1
            continue
        known += 1
        for rule_id in rule_ids:
            rule_hits[str(rule_id)] = rule_hits.get(str(rule_id), 0) + 1
    rates = {rid: round(hits / known, 6) for rid, hits in sorted(rule_hits.items())} if known else {}
    return {
        "rates": rates,
        "verdict": "computed" if known else "insufficient_data",
        "reason": (
            f"{unknown_count} of {total} decisions carry no veto metadata"
            if unknown_count else f"veto rates over {known} decisions with veto metadata"
        ),
        "decision_count": total,
        "known_count": known,
        "unknown_count": unknown_count,
    }


def label_coverage(decisions: list[dict]) -> dict:
    """Fraction of decisions with a matured 20d label (pure)."""
    total = len(decisions)
    if not total:
        return {
            "rate": None, "verdict": "insufficient_data",
            "reason": "no decisions observed",
            "covered_count": 0, "decision_count": 0,
        }
    covered = 0
    for decision in decisions:
        horizons = ((decision or {}).get("labels") or {}).get("horizons") or {}
        record = horizons.get("20d")
        if isinstance(record, dict) and record.get("label_up") is not None:
            covered += 1
    return {
        "rate": round(covered / total, 6),
        "verdict": "computed",
        "reason": f"{covered} of {total} decisions carry a matured 20d label",
        "covered_count": covered,
        "decision_count": total,
    }


def monitoring_snapshot(
    decisions: list[dict],
    reference_scores: list[float] | None = None,
    current_scores: list[float] | None = None,
    reference_confidences: list[float] | None = None,
    current_confidences: list[float] | None = None,
) -> dict:
    """Assemble the versioned monitoring block for a run (pure)."""
    scores = [
        float(d["score"]) for d in decisions
        if isinstance(d, dict) and isinstance(d.get("score"), (int, float))
        and not isinstance(d.get("score"), bool)
    ]
    confidences = [
        float(d["confidence"]) for d in decisions
        if isinstance(d, dict) and isinstance(d.get("confidence"), (int, float))
        and not isinstance(d.get("confidence"), bool)
    ]
    if reference_scores is None or current_scores is None:
        midpoint = len(scores) // 2
        if reference_scores is None:
            reference_scores = list(scores[:midpoint])
        if current_scores is None:
            current_scores = list(scores[midpoint:])
    if reference_confidences is None or current_confidences is None:
        midpoint = len(confidences) // 2
        if reference_confidences is None:
            reference_confidences = list(confidences[:midpoint])
        if current_confidences is None:
            current_confidences = list(confidences[midpoint:])
    drift = score_drift_psi(list(reference_scores), list(current_scores))
    conf = confidence_drift(list(reference_confidences), list(current_confidences))
    stale = stale_data_rate(decisions)
    veto = veto_rate_by_rule(decisions)
    coverage = label_coverage(decisions)
    degraded = [
        name for name, block in (
            ("score_drift", drift), ("confidence_drift", conf),
            ("stale_data", stale), ("veto_rates", veto),
            ("label_coverage", coverage),
        )
        if block.get("verdict") == "insufficient_data"
    ]
    return {
        "monitoring_version": MONITORING_VERSION,
        "score_drift": drift,
        "confidence_drift": conf,
        "stale_data": stale,
        "veto_rates": veto,
        "label_coverage": coverage,
        "status": "degraded" if degraded else "ok",
        "degraded_metrics": degraded,
    }



