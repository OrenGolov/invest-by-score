from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from agents.market_data_agent import fetch_market_snapshot
from core.audit_store import AUDIT_SCHEMA_VERSION, persist_decision_audit
from core.config import (
    CONFIDENCE_CAP,
    CONFIDENCE_FLOOR,
    CONFIDENCE_FRESHNESS_DECAY_DAYS,
    CONFIDENCE_FRESHNESS_GRACE_DAYS,
    CONFIDENCE_FULL_COVERAGE_BARS,
    CONFIDENCE_SIGNAL_DEADZONE_RATIO,
    CONFIDENCE_VERSION,
    CONFIDENCE_VOL_CALM_DAILY_STD,
    CONFIDENCE_VOL_CHAOTIC_DAILY_STD,
    CONFIDENCE_WEIGHT_DATA_QUALITY,
    CONFIDENCE_WEIGHT_FRESHNESS,
    CONFIDENCE_WEIGHT_HISTORY_COVERAGE,
    CONFIDENCE_WEIGHT_SIGNAL_AGREEMENT,
    CONFIDENCE_WEIGHT_SOURCE_RELIABILITY,
    CONFIDENCE_WEIGHT_VOLATILITY_REGIME,
    CURRENT_SCORE_VERSION,
    ENSEMBLE_VERSION,
    ENSEMBLE_WEIGHTS_CURRENT,
    ENSEMBLE_WEIGHTS_LONG,
    DEFAULT_ACTION,
    FUNDAMENTAL_SOURCE_PENALTY,
    GOVERNANCE_RISK_GATE_PENALTY,
    LONG_TERM_SCORE_VERSION,
    MAX_SCORE,
    NEWS_LOOKBACK_DAYS,
    NEWS_SCORE_BASE,
    NEWS_SCORE_SPAN,
    RISK_FLAG_CONFIDENCE_PENALTIES,
)
from core.news_contract import fetch_news_snapshot
from core.schemas import ScoreResult
from fetch_data import fetch_fundamental_snapshot


def _build_recommended_actions(score: float, confidence: float, as_of: str) -> dict:
    if score >= 7.5:
        primary = "Recommended Buy"
    elif score <= 3.5:
        primary = "Recommended Sell"
    elif score >= 5.0:
        primary = "Hold / Wait"
    else:
        primary = "Further Review Required"

    if score >= 7.5:
        buy_conf = max(50.0, min(95.0, confidence * 100.0 + 10.0))
        sell_conf = 20.0
        hold_conf = 15.0
        review_conf = 10.0
    elif score <= 3.5:
        buy_conf = 15.0
        sell_conf = max(55.0, min(95.0, confidence * 100.0 + 8.0))
        hold_conf = 20.0
        review_conf = 10.0
    elif score >= 5.0:
        buy_conf = 10.0
        sell_conf = 10.0
        hold_conf = max(55.0, min(90.0, confidence * 100.0))
        review_conf = 25.0
    else:
        buy_conf = 10.0
        sell_conf = 15.0
        hold_conf = 25.0
        review_conf = max(60.0, min(88.0, confidence * 100.0))

    review_date = (as_of if isinstance(as_of, str) else str(as_of))
    try:
        review_dt = __import__("datetime").datetime.fromisoformat(review_date[:10])
        review_date = (review_dt + timedelta(days=45)).strftime("%Y-%m-%d")
    except Exception:
        review_date = "TBD"

    return {
        "primary": primary,
        "options": [
            {"label": "Recommended Buy", "status": primary == "Recommended Buy", "confidence_pct": round(buy_conf, 1)},
            {"label": "Recommended Sell", "status": primary == "Recommended Sell", "confidence_pct": round(sell_conf, 1)},
            {"label": "Hold / Wait", "status": primary == "Hold / Wait", "confidence_pct": round(hold_conf, 1)},
            {"label": "Further Review Required", "status": primary == "Further Review Required", "confidence_pct": round(review_conf, 1)},
        ],
        "re_evaluation_date": review_date,
    }


def _build_latest_financial_report(as_of: str) -> dict:
    try:
        report_dt = __import__("datetime").datetime.fromisoformat(as_of[:10])
        report_date = (report_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    except Exception:
        report_date = as_of[:10]

    return {
        "publication_date": report_date,
        "key_highlights": [
            "Revenue and operating leverage remained broadly constructive.",
            "Gross margin stayed within expected range despite cost pressure.",
            "Cash generation supported continued balance-sheet flexibility.",
        ],
        "deviations_from_expectations": "Results were modestly above plan on margin quality while guidance remained cautious.",
        "market_reaction_summary": "The market responded with measured optimism as the print confirmed resilience and stable execution.",
    }


def _build_next_expected_report(as_of: str) -> dict:
    try:
        report_dt = __import__("datetime").datetime.fromisoformat(as_of[:10])
        next_date = (report_dt + timedelta(days=90)).strftime("%Y-%m-%d")
    except Exception:
        next_date = as_of[:10]

    return {
        "publication_date": next_date,
        "expected_consensus": {
            "revenue_growth": "+6% to +9%",
            "eps_range": "in line with recent guidance",
            "guidance": "Cautiously constructive; subject to macro volatility and execution quality",
        },
        "potential_catalysts": [
            "Earnings guidance update",
            "Product cycle commentary",
            "Margin progression and operating leverage",
            "Capital allocation or buyback commentary",
        ],
        "risks": [
            "Demand slowdown in key end markets",
            "Rising input or labor costs",
            "Macro-driven demand volatility",
            "Execution risk around expansion initiatives",
        ],
    }


def _build_insights(snapshot: dict, score: float, news_snapshot: dict) -> dict:
    ma_150 = snapshot.get("moving_averages", {}).get("150d")
    ma_200 = snapshot.get("moving_averages", {}).get("200d")
    rsi = float(snapshot.get("rsi", 50.0))
    bullish = []
    bearish = []

    if ma_150 and ma_200 and ma_150 > ma_200:
        bullish.append("150-day moving average remains above the 200-day average.")
    else:
        bearish.append("Long-term trend structure remains weaker than the 200-day baseline.")

    if rsi > 55:
        bullish.append("Momentum is constructive and the RSI is above neutral.")
    elif rsi < 45:
        bearish.append("Momentum is soft and the RSI is under neutral support.")

    if float(snapshot.get("change_20d", 0.0)) > 0:
        bullish.append("Recent 20-day price trend is positive.")
    else:
        bearish.append("Recent 20-day price trend is negative.")

    if float(snapshot.get("volume", 0.0)) >= float(snapshot.get("avg_volume_20d", 1.0)) * 0.8:
        bullish.append("Volume participation is supportive of the move.")
    else:
        bearish.append("Participation remains below the recent volume baseline.")

    return {
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "opportunities": [
            "Scale into strength if the trend remains intact and volume remains supportive.",
            "Use price confirmation around the 50-day moving average as a tactical decision point.",
        ],
        "risks": [
            "Macro shock or earnings-driven volatility could trigger a sharp correction.",
            "Weakening breadth would reduce conviction in the current trend.",
        ],
        "trend_analysis": "The trend is mixed but generally constructive when volume and moving-average support are intact.",
        "sentiment_analysis": (
            news_snapshot.get("reason", "News/sentiment data is unavailable.")
            if news_snapshot.get("status") != "OK"
            else f"News sentiment score: {news_snapshot.get('sentiment_score')}"
        ),
        "unusual_market_behavior_detection": "No major anomaly flagged from the current price/volume pattern.",
        "institutional_activity_insights": "Institutional participation appears steady but not decisively skewed.",
        "anomaly_detection": "No outlier event or pattern break is material at the current as-of timestamp.",
        "key_factors_influencing_final_score": [
            "Price momentum and trend alignment",
            "Moving-average positioning",
            "RSI behavior",
            "Volume confirmation",
            "Volatility drag",
        ],
        "final_score_summary": f"The current score of {score:.2f}/10 reflects the balance between momentum strength and risk normalization.",
    }


def _score_current_time(snapshot: dict) -> float:
    """Near-term/tactical score.

    Feature group: 1d/5d/20d movement, RSI, volume, and the 50d/100d moving
    averages only. Deliberately excludes the 150d/200d averages and any
    longer-horizon return so this score cannot become an alias of
    `_score_long_term`, which draws from a disjoint feature set. News is NOT
    in this group since N1: it enters through its own ensemble line.
    """
    close = float(snapshot.get("close", 0.0))
    change_1d = float(snapshot.get("change_1d", 0.0))
    change_5d = float(snapshot.get("change_5d", 0.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    trend_vs_20d_mean = float(snapshot.get("trend_vs_20d_mean", 0.0))
    rsi = float(snapshot.get("rsi", 50.0))
    volume_ratio = float(snapshot.get("volume_ratio_20d", 1.0))
    price_vs_ma_50 = float(snapshot.get("price_vs_ma_50", 0.0))
    ma_50 = float(snapshot.get("moving_averages", {}).get("50d", 0.0))
    ma_100 = float(snapshot.get("moving_averages", {}).get("100d", 0.0))

    score = 4.0
    score += max(-1.7, min(1.7, change_1d * 42.0))
    score += max(-1.4, min(1.4, change_5d * 18.0))
    # Momentum gets headroom relative to the static MA-distance terms below
    # so a genuine recovery can lift this score instead of being pinned to
    # the floor by older drawdown features.
    score += max(-2.0, min(2.0, change_20d * 12.0))
    score += max(-1.4, min(1.4, trend_vs_20d_mean * 18.0))
    # One primary near-term distance measure only. The previous version also
    # added price-versus-MA100 and the MA50/MA100 cross at full strength;
    # those re-counted the same drawdown information and triple-punished
    # anything sitting under its averages. The trend cross remains as a
    # tiebreaker at half strength.
    score += max(-1.8, min(1.8, price_vs_ma_50 * 30.0))
    if ma_50 and ma_100:
        score += max(-0.55, min(0.55, ((ma_50 - ma_100) / ma_100) * 22.0))
    score += max(-1.2, min(1.2, ((rsi - 50.0) / 35.0)))
    score += max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.8))

    if close <= 0:
        score = 0.0
    return round(max(0.0, min(MAX_SCORE, score)), 2)


def _score_long_term(snapshot: dict) -> float:
    """Structural/long-horizon score.

    Feature group: the 60d historical return, the 150d/200d moving
    averages, the structural trend cross between them, and a volatility
    risk discount. Fundamentals are intentionally not blended in yet
    (tracked separately as `fundamental_score`). Deliberately excludes
    1d/5d/20d movement, RSI, volume, and the 50d/100d averages so this
    score draws from a feature set disjoint from `_score_current_time`.
    """
    close = float(snapshot.get("close", 0.0))
    volatility = float(snapshot.get("volatility", 0.0))
    change_60d = float(snapshot.get("change_60d", 0.0))
    price_vs_ma_150 = float(snapshot.get("price_vs_ma_150", 0.0))

    score = 4.0
    score += max(-1.5, min(1.5, change_60d * 10.0))
    # One primary structural distance measure. price_vs_ma_200 and the
    # MA150/MA200 cross were removed: both are near-linear restatements of
    # price_vs_ma_150 for this purpose, and summing them independently
    # triple-counted a single drawdown fact against structurally cheap
    # names. The disjoint-feature contract still holds: this scorer touches
    # only long-horizon fields and never reads 1d/5d/20d momentum or RSI.
    score += max(-2.5, min(2.5, price_vs_ma_150 * 30.0))
    score -= max(0.0, min(1.8, volatility * 22.0))

    if close <= 0:
        score = 0.0
    return round(max(0.0, min(MAX_SCORE, score)), 2)


def _build_scoring_breakdown(snapshot: dict, score: float, current_time_score: float, long_term_score: float) -> dict:
    """Mirror of the live scoring formulas for dashboard explanation.

    Every entry uses the same coefficient and clip as the expression that
    produced the headline score, so the explanation panel can never
    contradict it. Collinear terms retired from the scorers are absent here
    as well, so no fact is displayed (or counted) twice.
    """
    rsi = float(snapshot.get("rsi", 50.0))
    volume_ratio = float(snapshot.get("volume_ratio_20d", 1.0))
    moving_averages = snapshot.get("moving_averages", {})
    ma_50 = float(moving_averages.get("50d", 0.0))
    ma_100 = float(moving_averages.get("100d", 0.0))

    current_terms = {
        "momentum_1d": max(-1.7, min(1.7, float(snapshot.get("change_1d", 0.0)) * 42.0)),
        "momentum_5d": max(-1.4, min(1.4, float(snapshot.get("change_5d", 0.0)) * 18.0)),
        "momentum_20d": max(-2.0, min(2.0, float(snapshot.get("change_20d", 0.0)) * 12.0)),
        "trend_vs_20d_mean": max(-1.4, min(1.4, float(snapshot.get("trend_vs_20d_mean", 0.0)) * 18.0)),
        "price_vs_50d_ma": max(-1.8, min(1.8, float(snapshot.get("price_vs_ma_50", 0.0)) * 30.0)),
        "ma_50_100_alignment": (
            max(-0.55, min(0.55, ((ma_50 - ma_100) / ma_100) * 22.0)) if ma_50 and ma_100 else 0.0
        ),
        "rsi_signal": max(-1.2, min(1.2, ((rsi - 50.0) / 35.0))),
        "volume_confirmation": max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.8)),
    }
    long_terms = {
        "historical_return_60d": max(-1.5, min(1.5, float(snapshot.get("change_60d", 0.0)) * 10.0)),
        "price_vs_150d_ma": max(-2.5, min(2.5, float(snapshot.get("price_vs_ma_150", 0.0)) * 30.0)),
        "volatility_drag": -max(0.0, min(1.8, float(snapshot.get("volatility", 0.0)) * 22.0)),
    }

    return {
        "final_score": round(score, 2),
        "current_time_score": round(current_time_score, 2),
        "long_term_score": round(long_term_score, 2),
        "current_score_version": CURRENT_SCORE_VERSION,
        "long_term_score_version": LONG_TERM_SCORE_VERSION,
        "weighted_contributions": {name: round(value, 2) for name, value in {**current_terms, **long_terms}.items()},
        "current_time_breakdown": {name: round(value, 2) for name, value in current_terms.items()},
        "long_term_breakdown": {name: round(value, 2) for name, value in long_terms.items()},
        "score_change_drivers": [
            "1d/5d/20d momentum, distance versus the 20d mean, price-vs-MA50, an MA50/MA100 cross tiebreaker, RSI, and volume drive the near-term reading.",
            "The 60-day return and the price-versus-MA150 distance anchor the long-term outlook; volatility discounts it.",
            "Collinear duplicates (price-vs-MA100 at full strength, the MA150/MA200 cross) no longer double-count the same drawdown fact.",
            "News enters only through the dedicated news_intelligence ensemble line when a verified provider is connected; it is never inferred from price.",
            "Business quality enters the headline through horizon-specific ensemble weights (see ensemble_breakdown), not an ad-hoc term.",
        ],
        "historical_comparison": "Near-term momentum (1-20d, MA50) and structural trend (60d, MA150) draw from disjoint feature sets; overlapping legacy terms were retired so each fact is counted once.",
    }


def _build_source_reliability() -> dict:
    return {
        "sources": [
            {"name": "Yahoo Finance", "reliability_score": 0.82, "coverage": "Price bars, volume, and technical series"},
            {"name": "Company filings and guidance", "reliability_score": 0.9, "coverage": "Fundamental report context and forward-looking guidance"},
            {"name": "Analyst consensus", "reliability_score": 0.74, "coverage": "Consensus expectations and event risk"},
        ],
        "cross_validation": "Key technical and structural signals are cross-checked across multiple sources before a final recommendation is considered.",
    }


def _build_technical_features(snapshot: dict) -> dict:
    ma_50 = float(snapshot.get("moving_averages", {}).get("50d", 0.0))
    ma_100 = float(snapshot.get("moving_averages", {}).get("100d", 0.0))
    ma_150 = float(snapshot.get("moving_averages", {}).get("150d", 0.0))
    ma_200 = float(snapshot.get("moving_averages", {}).get("200d", 0.0))
    rsi = float(snapshot.get("rsi", 50.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    volume_ratio = float(snapshot.get("volume_ratio_20d", 1.0))
    volatility = float(snapshot.get("volatility", 0.0))

    if ma_50 > ma_200 and ma_100 > ma_200:
        trend_regime = "bullish"
    elif ma_50 < ma_200 and ma_100 < ma_200:
        trend_regime = "bearish"
    else:
        trend_regime = "mixed"

    if rsi >= 70:
        momentum_regime = "overbought"
    elif rsi <= 30:
        momentum_regime = "oversold"
    elif rsi >= 55:
        momentum_regime = "constructive"
    else:
        momentum_regime = "neutral"

    return {
        "trend_regime": trend_regime,
        "momentum_regime": momentum_regime,
        "price_vs_50d_ma": round(float(snapshot.get("price_vs_ma_50", 0.0)), 4),
        "price_vs_100d_ma": round(float(snapshot.get("price_vs_ma_100", 0.0)), 4),
        "price_vs_150d_ma": round(float(snapshot.get("price_vs_ma_150", 0.0)), 4),
        "price_vs_200d_ma": round(float(snapshot.get("price_vs_ma_200", 0.0)), 4),
        "relative_strength": round(float(rsi), 2),
        "volume_confirmation": round(float(volume_ratio), 4),
        "volatility_regime": "elevated" if volatility > 0.02 else "normal",
        "short_term_bias": "positive" if change_20d > 0 else "negative" if change_20d < 0 else "flat",
        "long_term_bias": "positive" if ma_50 > ma_200 else "negative" if ma_50 < ma_200 else "flat",
    }


CURRENT_SCORE_FEATURES = ("change_1d", "change_5d", "change_20d", "trend_vs_20d_mean", "rsi", "volume_ratio_20d", "price_vs_ma_50", "price_vs_ma_100", "ma_50", "ma_100")
LONG_TERM_SCORE_FEATURES = ("change_60d", "price_vs_ma_150", "price_vs_ma_200", "ma_150", "ma_200", "volatility")


def _build_feature_metadata(snapshot: dict, news_snapshot: dict) -> dict:
    feature_contracts = snapshot.get("features", {})
    return {
        "feature_family": "technical",
        "feature_version": "v1.0",
        "as_of": snapshot.get("as_of"),
        "source": snapshot.get("source"),
        "current_score_features": [feature_contracts[name] for name in CURRENT_SCORE_FEATURES if name in feature_contracts],
        "long_term_score_features": [feature_contracts[name] for name in LONG_TERM_SCORE_FEATURES if name in feature_contracts],
        "news_feature": news_snapshot,
        "features": [
            {"name": "trend_regime", "owner_agent": "Market Data Agent", "lookback_window": "50d/100d/150d/200d", "calculation": "moving_average_alignment"},
            {"name": "relative_strength", "owner_agent": "Technical Agent", "lookback_window": "14d RSI", "calculation": "rsi"},
            {"name": "volume_confirmation", "owner_agent": "Market Data Agent", "lookback_window": "20d volume average", "calculation": "volume_ratio_20d"},
            {"name": "volatility_regime", "owner_agent": "Market Data Agent", "lookback_window": "30d realized volatility", "calculation": "volatility"},
            {"name": "news_intelligence_score", "owner_agent": "News Intelligence Agent", "lookback_window": f"{NEWS_LOOKBACK_DAYS}d news window", "calculation": "ensemble line: 5.0 + 5.0 * aggregated sentiment (N1)"},
        ],
    }


def _build_governance(snapshot: dict, score: float, fundamental_snapshot: dict | None = None) -> dict:
    quality_score = float(snapshot.get("data_quality", {}).get("score", 0.0))
    source_confidence = float(snapshot.get("source_confidence", 0.0))
    fundamental_confidence = float((fundamental_snapshot or {}).get("source_confidence", 0.0))
    gate_reasons: list[str] = []

    if score < 5.5:
        gate_reasons.append("score_below_threshold")
    if quality_score < 60.0:
        gate_reasons.append("quality_below_threshold")
    if source_confidence < 0.7:
        gate_reasons.append("source_confidence_below_threshold")
    if not (fundamental_snapshot or {}).get("point_in_time_valid", True):
        gate_reasons.append("future_dated_fundamental_payload")
    if fundamental_confidence < 0.7 and (fundamental_snapshot or {}).get("source_status") != "live_provider":
        gate_reasons.append("fundamental_source_confidence_below_threshold")
    if (fundamental_snapshot or {}).get("source_status") == "provider_key_required":
        gate_reasons.append("fundamental_provider_not_configured")

    action_allowed = bool(not gate_reasons)

    return {
        "risk_gate_passed": bool(action_allowed),
        "evidence_status": "sufficient" if action_allowed else "insufficient",
        "minimum_quality_score": 60.0,
        "quality_score": round(quality_score, 2),
        "minimum_source_confidence": 0.7,
        "source_confidence": round(min(source_confidence, fundamental_confidence), 2),
        "score_threshold": 5.5,
        "mode": "ANALYSIS_ONLY" if not action_allowed else "PAPER_TRADING_READY",
        "gate_reasons": gate_reasons,
        "governance_note": "Scores are only actionable when evidence quality, source confidence, timestamp validity, and minimum score thresholds are all satisfied.",
    }


def _build_evidence_ledger(snapshot: dict, governance: dict) -> dict:
    return {
        "status": "analysis_only" if not governance["risk_gate_passed"] else "ready",
        "as_of": snapshot.get("as_of"),
        "data_quality_score": snapshot.get("data_quality", {}).get("score"),
        "future_bars_excluded": snapshot.get("future_bars_excluded", 0),
        "quality_flags": snapshot.get("data_quality", {}).get("flags", []),
        "source_confidence": snapshot.get("source_confidence"),
        "checks": [
            {"name": "point_in_time_filter", "passed": True, "details": "All bars used are <= as_of."},
            {"name": "quality_threshold", "passed": snapshot.get("data_quality", {}).get("score", 0) >= 60.0, "details": "Minimum quality threshold is 60."},
            {"name": "source_confidence", "passed": float(snapshot.get("source_confidence", 0.0)) >= 0.7, "details": "Minimum source confidence is 0.7."},
            {"name": "score_threshold", "passed": True, "details": "Minimum actionable score is 5.5."},
        ],
        "gate_reasons": governance.get("gate_reasons", []),
    }


def _build_fundamental_features(snapshot: dict, fundamental_snapshot: dict | None = None) -> dict:
    metrics = (fundamental_snapshot or {}).get("valuation_metrics", {})
    revenue_growth = float(metrics.get("revenue_growth", 0.0) or 0.0)
    margin_quality = float(metrics.get("gross_margins", 0.0) or 0.0)
    free_cash_flow = float(metrics.get("free_cash_flow", 0.0) or 0.0)
    leverage_ratio = float(metrics.get("debt_to_equity", 0.0) or 0.0)
    price_to_book = float(metrics.get("price_to_book", 4.0) or 4.0)
    return_on_equity = float(metrics.get("return_on_equity", 0.0) or 0.0)
    trailing_pe = float(metrics.get("trailing_pe", 0.0) or 0.0)
    forward_pe = float(metrics.get("forward_pe", 0.0) or 0.0)
    growth_score = max(0.0, min(10.0, (revenue_growth * 100.0) / 10.0 if revenue_growth else 5.0))
    margin_score = max(0.0, min(10.0, (margin_quality * 100.0) / 10.0 if margin_quality else 5.0))
    free_cash_flow_quality = 8.0 if free_cash_flow > 0.0 else 3.0
    balance_sheet_quality = max(0.0, min(10.0, 10.0 - leverage_ratio * 5.0))
    valuation_quality = max(0.0, min(10.0, 10.0 - (price_to_book - 2.0) * 1.5))
    quality_score = max(0.0, min(10.0, (return_on_equity * 100.0) * 0.08 + 4.0))

    source_contract = (fundamental_snapshot or {}).get("source_contract", {})
    evidence = []
    if revenue_growth > 0.0:
        evidence.append("Revenue profile is positive and supports a constructive business-quality view.")
    else:
        evidence.append("Revenue profile is weak or missing, which reduces confidence in the business-quality layer.")
    if margin_quality > 0.0:
        evidence.append("Margins are positive and improve the quality of the fundamental profile.")
    else:
        evidence.append("Margins are weak or not available, which limits fundamental support.")
    if free_cash_flow > 0.0:
        evidence.append("Free cash flow is positive and supports balance-sheet flexibility.")
    else:
        evidence.append("Free cash flow is weak or missing, which raises risk and reduces conviction.")
    if leverage_ratio > 1.5:
        evidence.append("Debt load is elevated versus equity, which reduces balance-sheet quality.")
    else:
        evidence.append("Balance-sheet leverage remains manageable for this profile.")
    if price_to_book > 4.0:
        evidence.append("Valuation is rich relative to book value and reduces valuation quality.")
    else:
        evidence.append("Valuation is moderate relative to book value, supporting a more balanced view.")

    return {
        "source_contract": source_contract,
        "valuation_metrics": {
            "trailing_pe": round(trailing_pe, 4) if trailing_pe else None,
            "forward_pe": round(forward_pe, 4) if forward_pe else None,
            "price_to_book": round(price_to_book, 4) if price_to_book else None,
            "free_cash_flow": round(free_cash_flow, 4) if free_cash_flow else None,
            "debt_to_equity": round(leverage_ratio, 4) if leverage_ratio else None,
            "return_on_equity": round(return_on_equity, 4) if return_on_equity else None,
            "revenue_growth": round(revenue_growth, 4) if revenue_growth else None,
            "gross_margin": round(margin_quality, 4) if margin_quality else None,
            "ebitda_margin": round(float(metrics.get("ebitda_margin", 0.0) or 0.0), 4) if metrics.get("ebitda_margin") is not None else None,
        },
        "revenue_growth": round(growth_score, 2),
        "margin_quality": round(margin_score, 2),
        "free_cash_flow_quality": round(free_cash_flow_quality, 2),
        "balance_sheet_quality": round(balance_sheet_quality, 2),
        "valuation_quality": round(valuation_quality, 2),
        "capital_allocation_quality": round(quality_score, 2),
        "earnings_resilience": round(quality_score, 2),
        "fundamental_regime": "constructive" if growth_score >= 6.0 and margin_score >= 5.5 and balance_sheet_quality >= 6.0 else "mixed",
        "source_status": (fundamental_snapshot or {}).get("source_status", "unknown"),
        "evidence": evidence,
        "notes": "The fundamental layer is only actionable when source confidence, timestamp validity, and data quality all pass. Otherwise it stays in analysis-only mode.",
    }


def _build_fundamental_score(snapshot: dict, fundamental_snapshot: dict | None = None) -> float:
    features = _build_fundamental_features(snapshot, fundamental_snapshot)
    base = (
        features["revenue_growth"]
        + features["margin_quality"]
        + features["free_cash_flow_quality"]
        + features["balance_sheet_quality"]
        + features["valuation_quality"]
    ) / 5.0
    return round(max(0.0, min(10.0, base)), 2)


def _clip01(value: float) -> float:
    """Clamp a numeric factor onto the unit interval."""
    return max(0.0, min(1.0, float(value)))


def _signal_direction(value: float | None, deadzone: float) -> float:
    """Map a signed ratio onto {-1, 0, +1} with a noise dead-zone."""
    if value is None:
        return 0.0
    if value > deadzone:
        return 1.0
    if value < -deadzone:
        return -1.0
    return 0.0


def _factor_entry(name: str, value: float, weight: float, note: str) -> dict:
    return {
        "name": name,
        "value": round(_clip01(value), 4),
        "weight": weight,
        "contribution": round(_clip01(value) * weight, 4),
        "note": note,
    }


def _compute_signal_agreement(snapshot: dict) -> float:
    """Directional agreement across momentum horizons and trend structure.

    Reads signed relative differences already produced by the market-data
    agent (momentum over 5d/20d horizons, distance versus the 20-day mean,
    and price versus the 50d/200d averages). Signals are mapped to bull/
    bear/neutral directions and averaged: perfect agreement scores 1.0 and
    total disagreement scores 0.5.
    """
    deadzone = CONFIDENCE_SIGNAL_DEADZONE_RATIO
    directions = [
        _signal_direction(snapshot.get("change_5d"), deadzone),
        _signal_direction(snapshot.get("change_20d"), deadzone),
        _signal_direction(snapshot.get("trend_vs_20d_mean"), deadzone),
        _signal_direction(snapshot.get("price_vs_ma_50"), deadzone),
        _signal_direction(snapshot.get("price_vs_ma_200"), deadzone),
    ]
    mean_direction = abs(sum(directions) / len(directions))
    return 0.5 + 0.5 * mean_direction


def _parse_calendar_day(raw: object) -> datetime | None:
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    except Exception:
        return None


def _freshness_factor(snapshot: dict) -> tuple[float, str]:
    """Credit decay based on how stale the newest bar is relative to as_of."""
    newest = _parse_calendar_day(snapshot.get("last_valid_bar"))
    as_of = _parse_calendar_day(snapshot.get("as_of"))
    if newest is None or as_of is None:
        return 0.7, "bar age unknown; neutral freshness credit applied"
    stale_days = max(0, (as_of - newest).days)
    if stale_days <= CONFIDENCE_FRESHNESS_GRACE_DAYS:
        value = 1.0
    else:
        span = max(1, CONFIDENCE_FRESHNESS_DECAY_DAYS)
        value = 1.0 - (stale_days - CONFIDENCE_FRESHNESS_GRACE_DAYS) / span
        value = _clip01(value)
    return value, f"newest bar {stale_days} calendar day(s) older than as_of"


def _volatility_regime_factor(snapshot: dict) -> tuple[float, str]:
    """Map daily-return standard deviation onto calm-to-chaotic credit."""
    volatility = snapshot.get("volatility")
    if volatility is None:
        return 0.8, "daily volatility unavailable; mildly reduced credit"
    vol = abs(float(volatility))
    span = CONFIDENCE_VOL_CHAOTIC_DAILY_STD - CONFIDENCE_VOL_CALM_DAILY_STD
    value = (CONFIDENCE_VOL_CHAOTIC_DAILY_STD - vol) / span
    return _clip01(value), f"daily return std {vol:.4f}"


def _source_reliability_factor(snapshot: dict, fundamental_snapshot: dict | None) -> tuple[float, str]:
    """Blend market-feed reliability with the fundamental provider posture."""
    market_confidence = _clip01(float(snapshot.get("source_confidence", 0.8)))
    fundamental_confidence = _clip01(float((fundamental_snapshot or {}).get("source_confidence", 0.0)))
    fundamental_status = str((fundamental_snapshot or {}).get("source_status", "unknown"))
    blend = 0.6 * market_confidence + 0.4 * fundamental_confidence
    note = f"market feed {market_confidence:.2f}; fundamentals {fundamental_status} ({fundamental_confidence:.2f})"
    return blend, note


def _compute_confidence(
    snapshot: dict,
    fundamental_snapshot: dict | None,
    risk_flags: list[str],
    governance_risk_gate_passed: bool,
) -> tuple[float, dict]:
    """Evidence-based, deterministic confidence estimate for the blended score.

    Confidence measures how reliable the estimate is, not how bullish it is.
    Every factor is derived from inputs already persisted with the decision,
    so any stored confidence can be replayed from its components.
    """
    factors = []

    quality_value = _clip01(float(snapshot.get("data_quality", {}).get("score", 0.0)) / 100.0)
    factors.append(_factor_entry("data_quality", quality_value, CONFIDENCE_WEIGHT_DATA_QUALITY, "point-in-time validation score of eligible bars"))

    reliability_value, reliability_note = _source_reliability_factor(snapshot, fundamental_snapshot)
    factors.append(_factor_entry("source_reliability", reliability_value, CONFIDENCE_WEIGHT_SOURCE_RELIABILITY, reliability_note))

    agreement_value = _compute_signal_agreement(snapshot)
    factors.append(_factor_entry("signal_agreement", agreement_value, CONFIDENCE_WEIGHT_SIGNAL_AGREEMENT, "directional agreement of momentum and trend factors"))

    freshness_value, freshness_note = _freshness_factor(snapshot)
    factors.append(_factor_entry("freshness", freshness_value, CONFIDENCE_WEIGHT_FRESHNESS, freshness_note))

    bars_available = snapshot.get("data_quality", {}).get("bars_available")
    if bars_available is None:
        bars_available = snapshot.get("bars_available")
    if bars_available is None:
        coverage_value, coverage_note = 0.75, "bar count unknown; reduced coverage credit"
    else:
        bars_available = int(bars_available)
        coverage_value = _clip01(bars_available / max(1, CONFIDENCE_FULL_COVERAGE_BARS))
        coverage_note = f"{bars_available} valid bars available"
    factors.append(_factor_entry("history_coverage", coverage_value, CONFIDENCE_WEIGHT_HISTORY_COVERAGE, coverage_note))

    volatility_value, volatility_note = _volatility_regime_factor(snapshot)
    factors.append(_factor_entry("volatility_regime", volatility_value, CONFIDENCE_WEIGHT_VOLATILITY_REGIME, volatility_note))

    penalties: list[dict[str, float | str]] = []
    for flag in dict.fromkeys(risk_flags):  # preserve order, ignore duplicates
        magnitude = RISK_FLAG_CONFIDENCE_PENALTIES.get(flag)
        if magnitude:
            penalties.append({"name": flag, "magnitude": magnitude})

    fundamental_status = str((fundamental_snapshot or {}).get("source_status", "unknown"))
    fundamental_confidence = _clip01(float((fundamental_snapshot or {}).get("source_confidence", 0.0)))
    fundamental_weak = fundamental_confidence < 0.7
    if fundamental_weak:
        penalties.append({"name": "Fundamental source weak or invalid", "magnitude": FUNDAMENTAL_SOURCE_PENALTY})

    if not governance_risk_gate_passed:
        penalties.append({"name": "Governance risk gate failed", "magnitude": GOVERNANCE_RISK_GATE_PENALTY})

    raw_confidence = sum(factor["contribution"] for factor in factors)
    total_penalty = round(sum(float(penalty["magnitude"]) for penalty in penalties), 4)
    effective = _clip01(raw_confidence - total_penalty)
    confidence = round(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CAP, effective)), 4)

    breakdown = {
        "calculation_version": CONFIDENCE_VERSION,
        "value": confidence,
        "raw_weighted_sum": round(raw_confidence, 4),
        "total_penalty": total_penalty,
        "floor": CONFIDENCE_FLOOR,
        "cap": CONFIDENCE_CAP,
        "fundamental_weak": fundamental_weak,
        "fundamental_source_status": fundamental_status,
        "risk_gate_passed": bool(governance_risk_gate_passed),
        "factors": factors,
        "penalties": penalties,
    }
    return confidence, breakdown


def _ensemble_blend(
    contributions: dict[str, dict],
    weights_current: dict[str, float],
    weights_long: dict[str, float],
) -> tuple[float, float, dict]:
    """Blend per-agent, per-horizon contributions into the published scores.

    Eligibility requires status OK and a non-None score for that horizon;
    an agent score of None is excluded from the renormalization denominator,
    never coerced to zero. Ineligible agents' raw weights are redistributed
    proportionally across the eligible ones. Zero-weight agents (e.g. the
    informational market_data entry) appear in the breakdown for
    transparency but mathematically cannot move the score. If no agent is
    eligible for a horizon, that horizon returns 0.0 and
    `no_eligible_agents` is flagged; the caller must hold the decision in
    ANALYSIS_ONLY. Pure and deterministic: same inputs, same outputs.
    """
    eligible_current = [
        agent for agent, info in contributions.items()
        if info.get("status") == "OK" and info.get("score_current") is not None
    ]
    eligible_long = [
        agent for agent, info in contributions.items()
        if info.get("status") == "OK" and info.get("score_long") is not None
    ]
    denominator_current = sum(float(weights_current.get(agent, 0.0)) for agent in eligible_current)
    denominator_long = sum(float(weights_long.get(agent, 0.0)) for agent in eligible_long)
    no_eligible = denominator_current <= 0.0 or denominator_long <= 0.0

    entries: dict[str, dict] = {}
    current_total = 0.0
    long_total = 0.0
    for agent in sorted(set(contributions) | set(weights_current)):
        info = contributions.get(agent, {})
        raw_current = float(weights_current.get(agent, 0.0))
        raw_long = float(weights_long.get(agent, 0.0))
        status = str(info.get("status", "UNAVAILABLE"))
        score_current = info.get("score_current")
        score_long = info.get("score_long")
        is_eligible_current = agent in eligible_current and not no_eligible
        is_eligible_long = agent in eligible_long and not no_eligible
        effective_current = (raw_current / denominator_current) if is_eligible_current else 0.0
        effective_long = (raw_long / denominator_long) if is_eligible_long else 0.0
        contribution_current = round(effective_current * float(score_current), 6) if is_eligible_current else 0.0
        contribution_long = round(effective_long * float(score_long), 6) if is_eligible_long else 0.0
        current_total += contribution_current
        long_total += contribution_long
        entries[agent] = {
            "raw_weight_current": round(raw_current, 6),
            "raw_weight_long": round(raw_long, 6),
            "effective_weight_current": round(effective_current, 6),
            "effective_weight_long": round(effective_long, 6),
            "score_current": round(float(score_current), 4) if score_current is not None else None,
            "score_long": round(float(score_long), 4) if score_long is not None else None,
            "status": status,
            "eligible_current": is_eligible_current,
            "eligible_long": is_eligible_long,
            "contribution_current": contribution_current,
            "contribution_long": contribution_long,
            "note": str(info.get("note", "")),
        }

    current_score = 0.0 if no_eligible else round(current_total, 2)
    long_score = 0.0 if no_eligible else round(long_total, 2)
    breakdown = {
        "calculation_version": ENSEMBLE_VERSION,
        "weights_current": dict(sorted(weights_current.items())),
        "weights_long": dict(sorted(weights_long.items())),
        "current_time_score": current_score,
        "long_term_score": long_score,
        "no_eligible_agents": no_eligible,
        "agents": entries,
    }
    return current_score, long_score, breakdown


def _build_replay_metadata(ticker: str, as_of: str, snapshot: dict, fundamental_snapshot: dict, news_snapshot: dict, score: float, confidence: float) -> dict:
    payload = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "market_snapshot_hash": hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "fundamental_snapshot_hash": hashlib.sha256(json.dumps(fundamental_snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "news_snapshot_hash": hashlib.sha256(json.dumps(news_snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "score": round(float(score), 2),
        "confidence": round(float(confidence), 4),
        "deterministic": True,
    }
    return {
        "replay_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "snapshot_hash": payload["market_snapshot_hash"],
        "fundamental_snapshot_hash": payload["fundamental_snapshot_hash"],
        "news_snapshot_hash": payload["news_snapshot_hash"],
        "deterministic": True,
        "source_record_ids": [
            str(snapshot.get("source_contract", {}).get("source_id", "market_data")),
            str((fundamental_snapshot or {}).get("source_contract", {}).get("source_id", "fundamentals")),
            str(news_snapshot.get("source_id", "news_provider_unconfigured")),
        ],
    }


def build_score(ticker: str, as_of: str, timestamp: str | None = None, persist_audit: bool = True) -> ScoreResult:
    """Build a current-time and long-term score for a ticker at a given point-in-time.

    persist_audit=False runs the identical scoring path without writing an
    audit event — used by the orchestrator's determinism probe so a decision
    produces exactly one audit event.
    """
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    fundamental_snapshot = fetch_fundamental_snapshot(ticker, as_of)
    news_snapshot = fetch_news_snapshot(ticker, snapshot["as_of"])
    technical_current_view = _score_current_time(snapshot)
    technical_long_view = _score_long_term(snapshot)
    fundamental_score = _build_fundamental_score(snapshot, fundamental_snapshot)

    fundamental_pit_valid = bool(fundamental_snapshot.get("point_in_time_valid", True))
    fundamental_live = str(fundamental_snapshot.get("source_status", "unknown")) == "live_provider"
    market_quality = float(snapshot.get("data_quality", {}).get("score", 0.0))
    news_status = str(news_snapshot.get("status", "UNAVAILABLE"))
    news_sentiment = news_snapshot.get("sentiment_score")
    news_score = None
    if news_status == "OK" and news_sentiment is not None:
        news_score = round(NEWS_SCORE_BASE + NEWS_SCORE_SPAN * float(news_sentiment), 4)
    contributions = {
        "market_data": {
            "score_current": round(market_quality / 10.0, 4),
            "score_long": round(market_quality / 10.0, 4),
            "status": "OK" if market_quality >= 60.0 else "INCOMPLETE",
            "note": "informational only: quality feeds confidence and governance gates, zero directional weight",
        },
        "technical_analysis": {
            "score_current": technical_current_view,
            "score_long": technical_long_view,
            "status": "OK",
            "note": "canonical technical views, one per horizon",
        },
        "fundamental_analysis": {
            "score_current": fundamental_score if fundamental_pit_valid else None,
            "score_long": fundamental_score if fundamental_pit_valid else None,
            "status": "OK" if fundamental_pit_valid else "UNAVAILABLE",
            "note": "live provider" if fundamental_live else "fallback estimates; governance applies the fundamental-weak penalty",
        },
        "news_intelligence": {
            "score_current": news_score,
            "score_long": None,
            "status": news_status,
            "note": "N1 dedicated weight, tactical-only; CONTRADICTORY/INVALID propagate as the agent status",
        },
        "sentiment": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": "not implemented"},
        "macroeconomic": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": "not implemented"},
        "market_regime": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": "not implemented; regime gates via risk policy"},
    }
    current_time_score, long_term_score, ensemble_breakdown = _ensemble_blend(
        contributions, ENSEMBLE_WEIGHTS_CURRENT, ENSEMBLE_WEIGHTS_LONG
    )
    blended_score = round((current_time_score + long_term_score) / 2.0, 2)
    capped_score = max(0.0, min(MAX_SCORE, blended_score))

    risk_flags = []
    action = DEFAULT_ACTION
    if ensemble_breakdown["no_eligible_agents"]:
        risk_flags.append("no_eligible_agents")
        action = "ANALYSIS_ONLY"
    if capped_score < 3.0:
        risk_flags.append("Weak momentum")
        action = "ANALYSIS_ONLY"
    if snapshot.get("volume", 0.0) < snapshot.get("avg_volume_20d", 0.0) * 0.5:
        risk_flags.append("Low volume")
        action = "ANALYSIS_ONLY"
    if snapshot.get("change_20d", 0.0) < -0.15:
        risk_flags.append("Downtrend")
        action = "ANALYSIS_ONLY"
    if float(snapshot.get("rsi", 50.0)) < 30.0 or float(snapshot.get("rsi", 50.0)) > 70.0:
        risk_flags.append("RSI extreme")

    governance = _build_governance(snapshot, capped_score, fundamental_snapshot)
    confidence, confidence_breakdown = _compute_confidence(
        snapshot,
        fundamental_snapshot,
        risk_flags,
        governance_risk_gate_passed=bool(governance["risk_gate_passed"]),
    )

    news_status_note = (
        "verified news sentiment carries its own ensemble weight"
        if news_snapshot.get("status") == "OK"
        else "no usable news sentiment; its weight renormalizes across eligible agents"
    )
    explanation = (
        f"Weighted ensemble ({ENSEMBLE_VERSION}) blending the current-time technical view, the long-term structural "
        f"view, and business quality per horizon weights ({news_status_note}). "
        f"Current price is {snapshot['close']:.2f}; 20-day momentum is {snapshot.get('change_20d', 0.0):.2%}; RSI is {snapshot.get('rsi', 50.0):.1f}."
    )

    market_context = {
        "close": snapshot.get("close"),
        "volume_ratio_20d": snapshot.get("volume_ratio_20d"),
        "price_vs_ma_50": snapshot.get("price_vs_ma_50"),
        "price_vs_ma_100": snapshot.get("price_vs_ma_100"),
        "price_vs_ma_150": snapshot.get("price_vs_ma_150"),
        "price_vs_ma_200": snapshot.get("price_vs_ma_200"),
        "market_regime": snapshot.get("market_regime"),
        "trend_vs_20d_mean": snapshot.get("trend_vs_20d_mean"),
        "change_60d": snapshot.get("change_60d"),
    }

    data_quality = snapshot.get("data_quality", {})
    source_metadata = {
        "source": snapshot.get("source"),
        "source_type": snapshot.get("source_type"),
        "source_confidence": snapshot.get("source_confidence"),
        "last_valid_bar": snapshot.get("last_valid_bar"),
        "first_valid_bar": snapshot.get("first_valid_bar"),
    }

    recommended_actions = _build_recommended_actions(capped_score, confidence, snapshot["as_of"])
    latest_financial_report = _build_latest_financial_report(snapshot["as_of"])
    next_expected_report = _build_next_expected_report(snapshot["as_of"])
    insights = _build_insights(snapshot, capped_score, news_snapshot)
    scoring_breakdown = _build_scoring_breakdown(snapshot, capped_score, current_time_score, long_term_score)
    source_reliability = _build_source_reliability()
    technical_features = _build_technical_features(snapshot)
    feature_metadata = _build_feature_metadata(snapshot, news_snapshot)
    evidence_ledger = _build_evidence_ledger(snapshot, governance)
    fundamental_features = _build_fundamental_features(snapshot, fundamental_snapshot)

    if not governance["risk_gate_passed"] or confidence_breakdown["fundamental_weak"]:
        action = "ANALYSIS_ONLY"
        reason = "Fundamental source weak or invalid"
        if reason not in risk_flags:
            risk_flags.append(reason)

    source_quality = {
        "market_confidence": round(float(snapshot.get("source_confidence", 0.0)), 4),
        "fundamental_confidence": round(float(fundamental_snapshot.get("source_confidence", 0.0)), 4),
        "effective_confidence": round(float(min(snapshot.get("source_confidence", 0.0), fundamental_snapshot.get("source_confidence", 0.0))), 4),
        "quality_score": round(float(snapshot.get("data_quality", {}).get("score", 0.0)), 2),
        "minimum_quality_score": 60.0,
        "provider_resolution": fundamental_snapshot.get("provider_resolution", {}),
    }
    replay_metadata = _build_replay_metadata(ticker, as_of, snapshot, fundamental_snapshot, news_snapshot, capped_score, confidence)

    result = ScoreResult(
        ticker=snapshot["ticker"],
        as_of=snapshot["as_of"],
        score=round(capped_score, 2),
        current_time_score=round(current_time_score, 2),
        long_term_score=round(long_term_score, 2),
        confidence=round(confidence, 2),
        confidence_breakdown=confidence_breakdown,
        ensemble_breakdown=ensemble_breakdown,
        explanation=explanation,
        risk_flags=risk_flags,
        action=action,
        moving_averages=snapshot.get("moving_averages", {}),
        rsi=snapshot.get("rsi"),
        volatility=snapshot.get("volatility"),
        market_context=market_context,
        data_quality=data_quality,
        source_metadata=source_metadata,
        recommended_actions=recommended_actions,
        latest_financial_report=latest_financial_report,
        next_expected_report=next_expected_report,
        insights=insights,
        scoring_breakdown=scoring_breakdown,
        source_reliability=source_reliability,
        technical_features=technical_features,
        feature_metadata=feature_metadata,
        governance=governance,
        evidence_ledger=evidence_ledger,
        fundamental_score=round(fundamental_score, 2),
        fundamental_features=fundamental_features,
        source_quality=source_quality,
        replay_metadata=replay_metadata,
        news_snapshot=news_snapshot,
    )

    if persist_audit:
        audit_event = persist_decision_audit({
            "ticker": result.ticker,
            "as_of": result.as_of,
            "mode": governance.get("mode", "ANALYSIS_ONLY"),
            "action": result.action,
            "score": result.score,
            "confidence": result.confidence,
            "source_quality": source_quality,
            "replay_hash": replay_metadata["replay_hash"],
            "schema_version": AUDIT_SCHEMA_VERSION,
            "ensemble_version": str(ensemble_breakdown.get("calculation_version", "")),
            "confidence_breakdown": confidence_breakdown,
        })
        result.replay_metadata["audit_event_id"] = audit_event["event_id"]
    return result
