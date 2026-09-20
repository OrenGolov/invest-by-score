"""Is the news provider key set, and does it actually work?

Run this after setting NEWS_PROVIDER_API_KEY. It answers three questions a
failed collection run cannot separate on its own:

    1. is the variable set in THIS process?
    2. does the provider accept the key?
    3. does a real ticker return real articles?

Deliberately NOT a governance gate (no `check_*` CI wiring beyond being
runnable): a missing key is a setup state, not a code defect, and failing CI
for it would punish every clone.

    python scripts/check_news_key.py
    python scripts/check_news_key.py --ticker NVDA
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    NEWS_LOOKBACK_DAYS,
    NEWS_PROVIDER_API_KEY_ENV,
    NEWS_PROVIDER_URL,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    args = parser.parse_args()

    print(f"news provider check  [{NEWS_PROVIDER_API_KEY_ENV}]")
    print(f"  endpoint : {NEWS_PROVIDER_URL}")
    print(f"  window   : {NEWS_LOOKBACK_DAYS} days")
    print()

    # -- 1. is it set here? --------------------------------------------
    key = os.getenv(NEWS_PROVIDER_API_KEY_ENV)
    if not key:
        print(f"  [X] {NEWS_PROVIDER_API_KEY_ENV} is NOT set in this process.")
        print()
        print("      If you just set it with setx, that only affects NEW")
        print("      terminals — close this one and open a fresh one.")
        print()
        print("      Set it for your account (permanent):")
        print(f"        setx {NEWS_PROVIDER_API_KEY_ENV} \"your-key-here\"")
        print()
        print("      Or for this session only:")
        print(f"        $env:{NEWS_PROVIDER_API_KEY_ENV} = \"your-key-here\"")
        return 1

    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "set"
    print(f"  [OK] key is set ({masked}, {len(key)} chars)")

    # -- 2 + 3. does it work on a real request? ------------------------
    from core.news_adapter import fetch_provider_articles

    as_of = datetime.now(timezone.utc)
    result = fetch_provider_articles(args.ticker, as_of)
    status = result.get("status")

    if status == "provider_key_required":
        print("  [X] the adapter still reports no key — the process did not see it")
        return 1
    if status != "ok":
        print(f"  [X] the provider rejected the request:")
        print(f"      {result.get('reason', 'no reason given')}")
        print()
        print("      Common causes:")
        print("        - the key was pasted with a stray space or quote")
        print("        - the free tier's daily limit is used up (100/day)")
        print("        - the account needs email verification")
        return 1

    records = result.get("records") or []
    print(f"  [OK] provider accepted the key")
    print(f"  [OK] {args.ticker}: {len(records)} article(s) in the window")

    if not records:
        print()
        print("      Zero articles is not an error — it can simply mean this")
        print("      ticker had no coverage. Try another with --ticker.")
        return 0

    first = records[0]
    headline = str(first.get("headline") or first.get("title") or "")[:70]
    print(f"      newest: {headline}")
    print()
    print("  Everything is wired. The scheduled task will now capture news")
    print("  every weekday, and OBSERVED event memories will start accruing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
