"""Are the fundamentals and macro provider keys set, and do they work?

Run this after setting ALPHAVANTAGE_API_KEY or FRED_API_KEY. It answers, per
provider, the three questions a failed collection run cannot separate:

    1. is the variable set in THIS process?
    2. does the provider accept the key?
    3. does a real request return real data?

**Why both live here together.** Neither key is required — the system runs
without them and says so — but they fail in opposite ways, and the difference
matters when deciding whether to chase one:

    ALPHAVANTAGE_API_KEY   absent -> every fundamental metric becomes a NEUTRAL
                           CONSTANT (MEASURED: balance_sheet_quality = 10.0 for
                           every company in the universe). Training refuses to
                           consume them, which is correct, so 5 declared
                           features stay dark. Nothing is lost permanently.

    FRED_API_KEY           absent -> the macro snapshot is UNAVAILABLE. FRED
                           serves historical VINTAGES, so a day missed here is
                           recoverable later. This is the lowest-priority key
                           in the project for exactly that reason.

Neither is perishable, unlike news. That is the whole argument for their
priority: a missing news day is gone in 7 days, a missing macro day is not.

NAMED `verify_*`, NOT `check_*`: CI auto-discovers every `scripts/check_*.py`
as a governance gate, and this script exits non-zero whenever a key is absent.
As a gate it would fail every clone that has not been set up, which is a setup
state rather than a code defect.

Usage:
    python scripts/verify_data_keys.py
    python scripts/verify_data_keys.py --ticker NVDA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import MACRO_PROVIDER_API_KEY_ENV  # noqa: E402

ALPHAVANTAGE_ENV = "ALPHAVANTAGE_API_KEY"
TIMEOUT = 15.0


def _masked(value: str) -> str:
    """Enough of the key to recognise it, never enough to use it."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def _get(url: str) -> tuple[dict | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "invest-by-score/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response), ""
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - every failure is reportable here
        return None, f"{type(exc).__name__}: {exc}"


def check_alphavantage(ticker: str) -> bool:
    print(f"fundamentals provider  [{ALPHAVANTAGE_ENV}]")
    print("  endpoint : https://www.alphavantage.co/query?function=OVERVIEW")
    key = os.getenv(ALPHAVANTAGE_ENV)
    if not key:
        print(f"  [X] {ALPHAVANTAGE_ENV} is not set")
        print("      Get a free key at https://www.alphavantage.co/support/#api-key")
        print(f"      Then:  setx {ALPHAVANTAGE_ENV} your_key_here")
        print("      Until then, all 5 fundamental features are neutral")
        print("      constants and training correctly excludes them.")
        return False

    print(f"  [OK] key is set ({_masked(key)})")
    payload, error = _get(
        "https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={urllib.parse.quote(ticker)}&apikey={urllib.parse.quote(key)}"
    )
    if payload is None:
        print(f"  [X] the request failed: {error}")
        return False
    # Alpha Vantage returns HTTP 200 with an explanatory body for a bad key or
    # an exhausted allowance, so the status alone proves nothing.
    note = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
    if note:
        print("  [X] the provider rejected the request:")
        print(f"      {str(note)[:200]}")
        print("      A free key allows 25 requests/day; a full collection run")
        print("      asks for 77, so this is the expected steady state.")
        return False
    name = payload.get("Name") or payload.get("Symbol")
    if not name:
        print(f"  [X] the provider returned no company data for {ticker}")
        return False
    print(f"  [OK] real data returned: {name}")
    print(f"       sector={payload.get('Sector') or '?'}  PE={payload.get('PERatio') or '?'}")
    return True


def check_fred() -> bool:
    print(f"macro provider  [{MACRO_PROVIDER_API_KEY_ENV}]")
    print("  endpoint : https://api.stlouisfed.org/fred/series/observations")
    key = os.getenv(MACRO_PROVIDER_API_KEY_ENV)
    if not key:
        print(f"  [X] {MACRO_PROVIDER_API_KEY_ENV} is not set")
        print("      Get a free key at https://fredaccount.stlouisfed.org/apikeys")
        print(f"      Then:  setx {MACRO_PROVIDER_API_KEY_ENV} your_key_here")
        print("      Until then the macro snapshot is UNAVAILABLE. Lowest")
        print("      priority: FRED serves vintages, so these days are")
        print("      recoverable later, unlike news.")
        return False

    print(f"  [OK] key is set ({_masked(key)})")
    payload, error = _get(
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=DGS10&api_key={urllib.parse.quote(key)}"
        "&file_type=json&limit=1&sort_order=desc"
    )
    if payload is None:
        # FRED answers a bad key with HTTP 400 and an XML/JSON error body, so a
        # 400 here means the KEY, not the request shape.
        print(f"  [X] the request failed: {error}")
        if "400" in error:
            print("      FRED returns 400 for an invalid key. Check for a")
            print("      stray space or a truncated paste.")
        return False
    observations = payload.get("observations") or []
    if not observations:
        print("  [X] the provider returned no observations")
        return False
    latest = observations[0]
    print(f"  [OK] real data returned: DGS10 {latest.get('date')} = {latest.get('value')}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="IBM",
                        help="symbol to probe the fundamentals provider with")
    args = parser.parse_args(argv)

    results = {}
    results[ALPHAVANTAGE_ENV] = check_alphavantage(args.ticker)
    print()
    results[MACRO_PROVIDER_API_KEY_ENV] = check_fred()
    print()

    working = [name for name, ok in results.items() if ok]
    missing = [name for name, ok in results.items() if not ok]
    if working:
        print(f"  working: {', '.join(working)}")
    if missing:
        print(f"  not working: {', '.join(missing)}")
        print()
        print("  NEITHER KEY IS REQUIRED. The system runs without them and")
        print("  reports what it could not compute rather than inventing it.")
        print("  Neither is perishable either — unlike news, these days can be")
        print("  backfilled once a key exists.")
        return 1
    print("  both providers are live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
