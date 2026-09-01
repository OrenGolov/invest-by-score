"""Temporary N1 smoke check — deleted after the sprint lands."""
from unittest.mock import patch

from core.news_adapter import build_news_snapshot
from core.news_contract import fetch_news_snapshot

AS_OF = "2024-01-05 12:00:00"
ENV = {"NEWS_PROVIDER_API_KEY": "smoke-key"}


def _rec(rid, published, headline, tone=None, quality=None):
    return {
        "source_record_id": rid, "published_time": published, "headline": headline,
        "summary": "", "url": f"https://x/{rid}", "source_name": "W", "ticker": None,
        "company_name": None, "source_quality": quality, "tone": tone,
    }


def _snapshot(records):
    with patch.dict("os.environ", ENV), \
            patch("core.news_adapter.append_raw_records", return_value=True), \
            patch("core.news_adapter.fetch_provider_articles",
                  return_value={"status": "ok", "records": records, "reason": ""}):
        return build_news_snapshot("TEST", AS_OF)


legacy = {
    "ticker": "MSFT", "as_of": "2024-01-02", "status": "UNAVAILABLE",
    "source_id": "news_provider_unconfigured", "source_confidence": 0.0,
    "published_time": None, "calculation_version": "news-contract-v1",
    "lookback_period": "N/A", "sentiment_score": None, "articles": [],
    "reason": "No verified news provider is connected. Sentiment is not inferred from price or technical indicators.",
}
assert fetch_news_snapshot("MSFT", "2024-01-02") == legacy, "no-key contract drifted"
assert "pipeline" not in fetch_news_snapshot("MSFT", "2024-01-02")
print("1. no-key UNAVAILABLE byte-for-byte: OK")

snap = _snapshot([
    _rec("u2", "2024-01-04 10:00:00", "TEST launches new chip", tone=0.4),
    _rec("u1", "2024-01-05 10:00:00", "TEST beats earnings expectations", tone=0.8),
])
print("2. OK path:", snap["status"], snap["sentiment_score"], round(snap["source_confidence"], 4), snap["published_time"])
assert snap["status"] == "OK" and 0 < snap["sentiment_score"] <= 1.0

snap = _snapshot([
    _rec("c1", "2024-01-05 10:00:00", "TEST beats earnings estimates", tone=0.9),
    _rec("c2", "2024-01-05 11:00:00", "TEST misses earnings estimates", tone=-0.9),
])
print("3. CONTRADICTORY path:", snap["status"], snap["sentiment_score"], snap["source_confidence"])
assert snap["status"] == "CONTRADICTORY" and snap["sentiment_score"] is None
assert snap["source_confidence"] == 0.10

snap = _snapshot([
    _rec("f1", "2024-01-06 10:00:00", "TEST beats earnings estimates"),
    _rec("u1", "2024-01-05 10:00:00", "TEST record profits", tone=1.0),
])
print("4. INVALID path:", snap["status"], [r["reason"] for r in snap["pipeline"]["rejected"]])
assert snap["status"] == "INVALID"

snap = _snapshot([_rec("z1", "2024-01-05 10:00:00", "TEST record profits", tone=1.0, quality=0.0)])
print("5. zero-quality path:", snap["status"], snap["sentiment_score"], snap["source_confidence"])
assert snap["status"] == "INCOMPLETE"

clean = _snapshot([_rec("ok1", "2024-01-05 10:00:00", "TEST record profits", tone=1.0)])
dirty = _snapshot([
    _rec("ok1", "2024-01-05 10:00:00", "TEST record profits", tone=1.0),
    _rec("zero", "2024-01-05 10:30:00", "TEST beats everything", tone=1.0, quality=0.0),
])
print("6. zero-quality cannot raise confidence:", clean["source_confidence"], "==", dirty["source_confidence"])
assert clean["sentiment_score"] == dirty["sentiment_score"]
assert clean["source_confidence"] == dirty["source_confidence"]

print("ALL SMOKE CHECKS PASSED")