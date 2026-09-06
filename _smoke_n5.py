"""Temporary N5 smoke check — deleted after the sprint lands."""
from unittest.mock import patch

from core.score_engine import build_score


def _regime_stub(status="OK", label="bullish", proxy=0.9):
    stub = {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": status,
        "source_id": "yahoo_finance_chart", "source_confidence": 0.8,
        "published_time": "2024-01-02 00:00:00", "calculation_version": "regime-contract-v1",
        "lookback_period": "300 sessions", "regime": label, "probability_proxy": proxy,
        "transition_risk": {"window_sessions": 20, "labeled_sessions": 20, "flips": 0,
                            "flip_rate": 0.0, "labels": [label] * 20},
        "inputs": {}, "rule_trace": {}, "reason": "stub",
        "pipeline": {"pipeline_version": "regime-pipeline-v1",
                     "classifier_version": "regime-classifier-v1", "counts": {}},
    }
    return stub


def _ok_news(sentiment):
    return {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": "OK",
        "source_id": "newsapi_news", "source_confidence": 0.7,
        "published_time": "2024-01-02 09:00:00",
        "calculation_version": "news-contract-v1", "lookback_period": "7d",
        "sentiment_score": sentiment, "articles": [], "reason": "",
        "pipeline": {"pipeline_version": "news-pipeline-v1"},
    }


with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
    baseline = build_score("MSFT", "2024-01-02", persist_audit=False)
attr = baseline.scoring_breakdown["attribution"]
assert attr["attribution_version"] == "score-attribution-v1"
assert attr["buckets"]["narrative"]["total"] == 0.0, "phantom narrative!"
assert "contributes exactly 0.0" in attr["summary"]
assert attr["reconciles"] and attr["attributed_total"] == baseline.score
print("1. no-news run: narrative exactly 0.0, reconciles", attr["attributed_total"], "==", baseline.score)
print("   buckets:", {k: v["total"] for k, v in attr["buckets"].items()})
print("   thesis:", attr["thesis_support"])

with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()), \
        patch("core.score_engine.fetch_news_snapshot", return_value=_ok_news(0.5)):
    with_news = build_score("MSFT", "2024-01-02", persist_audit=False)
attr2 = with_news.scoring_breakdown["attribution"]
assert attr2["buckets"]["narrative"]["total"] > 0.0
assert attr2["thesis_support"] == "operational_and_narrative"
assert f"narrative evidence (news + sentiment) {attr2['buckets']['narrative']['total']:+.2f}" in with_news.explanation
assert "thesis support: operational_and_narrative" in with_news.explanation
print("2. news OK: narrative", attr2["buckets"]["narrative"]["total"],
      "| thesis", attr2["thesis_support"])
print("   buckets:", {k: v["total"] for k, v in attr2["buckets"].items()})

# Line/bucket partition sanity on the real run (4dp display tolerance).
for agent, line in attr2["lines"].items():
    expected = (line["contribution_current"] + line["contribution_long"]) / 2.0
    assert abs(line["total"] - expected) <= 1.5e-4, agent
print("3. line math consistent; market_data informational:",
      attr2["lines"]["market_data"]["bucket"] == "informational",
      "| informational_total:", attr2["informational_total"])

with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
    again = build_score("MSFT", "2024-01-02", persist_audit=False)
assert again.scoring_breakdown["attribution"] == attr
print("4. deterministic: attribution identical across replays")

print("ALL SMOKE CHECKS PASSED")