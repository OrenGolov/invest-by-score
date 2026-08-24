from __future__ import annotations

from datetime import datetime

from agents.market_data_agent import fetch_market_snapshot
from agents.technical_agent import score_technical
from core.config import DEFAULT_ACTION, DEFAULT_CONFIDENCE


def build_score(ticker: str, as_of: str) -> dict:
    snapshot = fetch_market_snapshot(ticker, as_of)
    technical = score_technical(snapshot)

    final_score = min(10.0, max(0.0, technical["score"]))
    final_confidence = min(0.99, max(0.1, technical["confidence"]))

    result = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "score": round(final_score, 2),
        "confidence": round(final_confidence, 2),
        "explanation": technical["explanation"],
        "risk_flags": [],
        "action": DEFAULT_ACTION,
    }
    return result


if __name__ == "__main__":
    print(build_score("AAPL", "2026-08-21"))
