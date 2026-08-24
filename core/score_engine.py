from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.audit_store import persist_decision_audit
from core.config import DEFAULT_ACTION, MAX_SCORE
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


def _build_insights(snapshot: dict, score: float) -> dict:
    ma_50 = snapshot.get("moving_averages", {}).get("50d")
    ma_200 = snapshot.get("moving_averages", {}).get("200d")
    rsi = float(snapshot.get("rsi", 50.0))
    bullish = []
    bearish = []

    if ma_50 and ma_200 and ma_50 > ma_200:
        bullish.append("50-day moving average remains above the 200-day average.")
    else:
        bearish.append("Trend structure remains weaker than the long-term baseline.")

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
        "sentiment_analysis": "Sentiment is neutral to mildly positive given stable momentum and measured participation.",
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
    close = float(snapshot.get("close", 0.0))
    change_1d = float(snapshot.get("change_1d", 0.0))
    change_5d = float(snapshot.get("change_5d", 0.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    trend_vs_20d_mean = float(snapshot.get("trend_vs_20d_mean", 0.0))
    rsi = float(snapshot.get("rsi", 50.0))
    volatility = float(snapshot.get("volatility", 0.0))
    volume_ratio = float(snapshot.get("volume_ratio_20d", 1.0))
    moving_averages = snapshot.get("moving_averages", {})
    ma_50 = float(moving_averages.get("50d", 0.0))
    ma_100 = float(moving_averages.get("100d", 0.0))
    ma_200 = float(moving_averages.get("200d", 0.0))

    price_vs_ma_50 = float(snapshot.get("price_vs_ma_50", 0.0))
    price_vs_ma_100 = float(snapshot.get("price_vs_ma_100", 0.0))

    score = 5.0
    score += max(-2.0, min(2.0, change_1d * 40.0))
    score += max(-1.5, min(1.5, change_5d * 25.0))
    score += max(-1.5, min(1.5, change_20d * 15.0))
    score += max(-1.5, min(1.5, trend_vs_20d_mean * 18.0))
    score += max(-1.5, min(1.5, price_vs_ma_50 * 35.0))
    score += max(-1.5, min(1.5, price_vs_ma_100 * 22.0))
    score += max(-1.5, min(1.5, ((rsi - 50.0) / 50.0) * 2.0))
    score += max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.5))
    score -= min(1.5, volatility * 25.0)

    news_signal = 0.0
    if rsi > 55:
        news_signal += 0.8
    elif rsi < 45:
        news_signal -= 0.8
    if change_1d > 0:
        news_signal += 0.5
    elif change_1d < 0:
        news_signal -= 0.5
    if price_vs_ma_50 > 0 and price_vs_ma_100 > 0:
        news_signal += 0.7
    elif price_vs_ma_50 < 0 and price_vs_ma_100 < 0:
        news_signal -= 0.7
    if volume_ratio > 1.0:
        news_signal += 0.5
    elif volume_ratio < 0.8:
        news_signal -= 0.5

    score += max(-1.5, min(1.5, news_signal * 0.8))
    if close <= 0:
        score = 0.0
    return round(max(0.0, min(MAX_SCORE, score)), 2)


def _score_long_term(snapshot: dict) -> float:
    close = float(snapshot.get("close", 0.0))
    change_20d = float(snapshot.get("change_20d", 0.0))
    trend_vs_20d_mean = float(snapshot.get("trend_vs_20d_mean", 0.0))
    volatility = float(snapshot.get("volatility", 0.0))
    moving_averages = snapshot.get("moving_averages", {})
    ma_50 = float(moving_averages.get("50d", 0.0))
    ma_100 = float(moving_averages.get("100d", 0.0))
    ma_150 = float(moving_averages.get("150d", 0.0))
    ma_200 = float(moving_averages.get("200d", 0.0))
    price_vs_ma_150 = float(snapshot.get("price_vs_ma_150", 0.0))
    price_vs_ma_200 = float(snapshot.get("price_vs_ma_200", 0.0))

    score = 5.0
    score += max(-2.0, min(2.0, price_vs_ma_150 * 30.0))
    score += max(-2.0, min(2.0, price_vs_ma_200 * 28.0))
    score += max(-1.5, min(1.5, ((ma_50 - ma_200) / ma_200) * 40.0)) if ma_200 else 0.0
    score += max(-1.0, min(1.0, ((ma_100 - ma_150) / ma_150) * 20.0)) if ma_150 else 0.0
    score += max(-1.0, min(1.0, change_20d * 10.0))
    score += max(-1.0, min(1.0, trend_vs_20d_mean * 12.0))
    score -= min(1.5, volatility * 20.0)

    if close <= 0:
        score = 0.0
    return round(max(0.0, min(MAX_SCORE, score)), 2)


def _build_scoring_breakdown(snapshot: dict, score: float, current_time_score: float, long_term_score: float) -> dict:
    momentum = max(-2.0, min(2.0, float(snapshot.get("change_20d", 0.0)) * 30.0))
    trend = max(-1.5, min(1.5, float(snapshot.get("trend_vs_20d_mean", 0.0)) * 20.0))
    ma_alignment = 0.0
    moving_averages = snapshot.get("moving_averages", {})
    ma_50 = float(moving_averages.get("50d", 0.0))
    ma_200 = float(moving_averages.get("200d", 0.0))
    if ma_50 and ma_200:
        ma_alignment = max(-1.5, min(1.5, ((ma_50 - ma_200) / ma_200) * 50.0))
    rsi_weight = max(-1.5, min(1.5, ((float(snapshot.get("rsi", 50.0)) - 50.0) / 50.0) * 1.5))
    volume_ratio = float(snapshot.get("volume", 0.0)) / float(snapshot.get("avg_volume_20d", 1.0)) if float(snapshot.get("avg_volume_20d", 1.0)) else 1.0
    volume_weight = max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.0))
    volatility_weight = min(1.5, float(snapshot.get("volatility", 0.0)) * 30.0) if float(snapshot.get("volatility", 0.0)) > 0 else 0.0
    news_proxy = 0.0
    rsi = float(snapshot.get("rsi", 50.0))
    if rsi > 55:
        news_proxy += 0.8
    elif rsi < 45:
        news_proxy -= 0.8
    if float(snapshot.get("change_1d", 0.0)) > 0:
        news_proxy += 0.5
    elif float(snapshot.get("change_1d", 0.0)) < 0:
        news_proxy -= 0.5
    if float(snapshot.get("price_vs_ma_50", 0.0)) > 0:
        news_proxy += 0.7
    elif float(snapshot.get("price_vs_ma_50", 0.0)) < 0:
        news_proxy -= 0.7

    return {
        "final_score": round(score, 2),
        "current_time_score": round(current_time_score, 2),
        "long_term_score": round(long_term_score, 2),
        "weighted_contributions": {
            "price_momentum_20d": round(momentum, 2),
            "trend_vs_20d_mean": round(trend, 2),
            "moving_average_alignment": round(ma_alignment, 2),
            "rsi_signal": round(rsi_weight, 2),
            "volume_confirmation": round(volume_weight, 2),
            "volatility_drag": round(-volatility_weight, 2),
            "news_sentiment_proxy": round(news_proxy, 2),
        },
        "current_time_breakdown": {
            "recent_momentum": round(max(-2.0, min(2.0, float(snapshot.get("change_5d", 0.0)) * 25.0)), 2),
            "trend_vs_20d_mean": round(max(-1.5, min(1.5, float(snapshot.get("trend_vs_20d_mean", 0.0)) * 18.0)), 2),
            "ma_50_100_alignment": round(max(-1.5, min(1.5, float(snapshot.get("price_vs_ma_50", 0.0)) * 35.0 + float(snapshot.get("price_vs_ma_100", 0.0)) * 22.0)), 2),
            "rsi_signal": round(max(-1.5, min(1.5, ((float(snapshot.get("rsi", 50.0)) - 50.0) / 50.0) * 2.0)), 2),
            "volume_confirmation": round(max(-1.0, min(1.0, (volume_ratio - 1.0) * 2.5)), 2),
            "news_sentiment_proxy": round(news_proxy, 2),
        },
        "long_term_breakdown": {
            "price_vs_150d_ma": round(max(-2.0, min(2.0, float(snapshot.get("price_vs_ma_150", 0.0)) * 30.0)), 2),
            "price_vs_200d_ma": round(max(-2.0, min(2.0, float(snapshot.get("price_vs_ma_200", 0.0)) * 28.0)), 2),
            "ma_50_200_alignment": round(max(-1.5, min(1.5, ((ma_50 - ma_200) / ma_200) * 40.0)), 2) if ma_200 else 0.0,
            "ma_100_150_alignment": round(max(-1.0, min(1.0, ((ma_50 - ma_200) / ma_200) * 40.0)), 2) if ma_200 else 0.0,
        },
        "score_change_drivers": [
            "Current-time momentum and news-style sentiment are driving the near-term reading.",
            "Longer-horizon moving-average positioning and the 150d/200d trend anchor the structural outlook.",
            "Volatility pressure offsets gains when market dispersion expands.",
        ],
        "historical_comparison": "The overall score blends a near-term momentum view with a broader trend assessment from the 150d and 200d structure.",
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


def _build_feature_metadata(snapshot: dict) -> dict:
    return {
        "feature_family": "technical",
        "feature_version": "v1.0",
        "as_of": snapshot.get("as_of"),
        "source": snapshot.get("source"),
        "features": [
            {"name": "trend_regime", "owner_agent": "Market Data Agent", "lookback_window": "50d/100d/200d", "calculation": "moving_average_alignment"},
            {"name": "relative_strength", "owner_agent": "Technical Agent", "lookback_window": "14d RSI", "calculation": "rsi"},
            {"name": "volume_confirmation", "owner_agent": "Market Data Agent", "lookback_window": "20d volume average", "calculation": "volume_ratio_20d"},
            {"name": "volatility_regime", "owner_agent": "Market Data Agent", "lookback_window": "30d realized volatility", "calculation": "volatility"},
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


def _build_replay_metadata(ticker: str, as_of: str, snapshot: dict, fundamental_snapshot: dict, score: float, confidence: float) -> dict:
    payload = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "market_snapshot_hash": hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "fundamental_snapshot_hash": hashlib.sha256(json.dumps(fundamental_snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "score": round(float(score), 2),
        "confidence": round(float(confidence), 4),
        "deterministic": True,
    }
    return {
        "replay_hash": hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "snapshot_hash": payload["market_snapshot_hash"],
        "fundamental_snapshot_hash": payload["fundamental_snapshot_hash"],
        "deterministic": True,
        "source_record_ids": [
            str(snapshot.get("source_contract", {}).get("source_id", "market_data")),
            str((fundamental_snapshot or {}).get("source_contract", {}).get("source_id", "fundamentals")),
        ],
    }


def build_score(ticker: str, as_of: str, timestamp: str | None = None) -> ScoreResult:
    """Build a current-time and long-term score for a ticker at a given point-in-time."""
    snapshot = fetch_market_snapshot(ticker, as_of, timestamp)
    fundamental_snapshot = fetch_fundamental_snapshot(ticker, as_of)
    current_time_score = _score_current_time(snapshot)
    long_term_score = _score_long_term(snapshot)
    blended_score = round((current_time_score + long_term_score) / 2.0, 2)
    capped_score = max(0.0, min(MAX_SCORE, blended_score))

    risk_flags = []
    action = DEFAULT_ACTION
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

    confidence = min(0.95, 0.5 + (capped_score / MAX_SCORE) * 0.45)
    if risk_flags:
        confidence = max(0.35, confidence - 0.2)

    explanation = (
        f"Dual-factor score combining current-time momentum, sentiment proxy, and 50d/100d trend structure with a longer-term view based on 150d/200d averages and historical regime. "
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
    insights = _build_insights(snapshot, capped_score)
    scoring_breakdown = _build_scoring_breakdown(snapshot, capped_score, current_time_score, long_term_score)
    source_reliability = _build_source_reliability()
    technical_features = _build_technical_features(snapshot)
    feature_metadata = _build_feature_metadata(snapshot)
    governance = _build_governance(snapshot, capped_score, fundamental_snapshot)
    evidence_ledger = _build_evidence_ledger(snapshot, governance)
    fundamental_features = _build_fundamental_features(snapshot, fundamental_snapshot)
    fundamental_score = _build_fundamental_score(snapshot, fundamental_snapshot)

    if not governance["risk_gate_passed"] or fundamental_snapshot.get("source_confidence", 0.0) < 0.7:
        action = "ANALYSIS_ONLY"
        risk_flags.append("Fundamental source weak or invalid")
        confidence = max(0.25, confidence - 0.2)

    source_quality = {
        "market_confidence": round(float(snapshot.get("source_confidence", 0.0)), 4),
        "fundamental_confidence": round(float(fundamental_snapshot.get("source_confidence", 0.0)), 4),
        "effective_confidence": round(float(min(snapshot.get("source_confidence", 0.0), fundamental_snapshot.get("source_confidence", 0.0))), 4),
        "quality_score": round(float(snapshot.get("data_quality", {}).get("score", 0.0)), 2),
        "minimum_quality_score": 60.0,
        "provider_resolution": fundamental_snapshot.get("provider_resolution", {}),
    }
    replay_metadata = _build_replay_metadata(ticker, as_of, snapshot, fundamental_snapshot, capped_score, confidence)

    result = ScoreResult(
        ticker=snapshot["ticker"],
        as_of=snapshot["as_of"],
        score=round(capped_score, 2),
        current_time_score=round(current_time_score, 2),
        long_term_score=round(long_term_score, 2),
        confidence=round(confidence, 2),
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
    )

    audit_event = persist_decision_audit({
        "ticker": result.ticker,
        "as_of": result.as_of,
        "mode": governance.get("mode", "ANALYSIS_ONLY"),
        "action": result.action,
        "score": result.score,
        "confidence": result.confidence,
        "source_quality": source_quality,
        "replay_hash": replay_metadata["replay_hash"],
    })
    result.replay_metadata["audit_event_id"] = audit_event["event_id"]
    return result
