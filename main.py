from __future__ import annotations

import argparse

from core.score_engine import build_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a ticker using the current technical scoring engine.")
    parser.add_argument("ticker", help="Ticker symbol to score, for example AAPL or MSFT")
    parser.add_argument("date", help="Request date in YYYY-MM-DD format")
    parser.add_argument("--timestamp", help="Optional time in HH:MM:SS format", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_score(args.ticker, args.date, args.timestamp)
    print(f"Ticker: {result.ticker}")
    print(f"As of: {result.as_of}")
    print(f"Score: {result.score:.2f}/10")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Action: {result.action}")
    print(f"Risk flags: {', '.join(result.risk_flags) if result.risk_flags else 'None'}")
    print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    main()
