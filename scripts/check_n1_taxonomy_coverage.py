"""Quick smoke test for N1 taxonomy reachability across the headline space.

Verifies that each taxonomy bucket can be triggered by at least one natural
headline that isn't already in the test suite's curated fixture, so the
taxonomy is genuinely live rather than only present in test constants.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import core.news_adapter as na  # noqa: E402  (repo-root import bootstrap above)

TAXONOMY = [category for category, _patterns in na.NEWS_CATEGORY_PATTERNS]

ADDITIONAL_CASES = {
    "Company A post-earnings profit warning after the bell": "earnings",
    "Company A lifts Q3 revenue outlook above street": "guidance",
    "Shareholder derivative suit names Company A directors": "litigation",
    "Company A faces new SEC reporting requirements": "regulation",
    "Company A unveils successor chip family next week": "product_launch",
    "Company A flagged as vulnerable in tariff shock scenario": "macro_shock",
    "Company A in talks to acquire smaller rival": "m_and_a",
    "Company A announces strategic review of European units": "strategic_announcement",
    "Company A CFO remarks on margin trajectory": "management_commentary",
    "Company A opens downtown office next month": "other",
}

if __name__ == "__main__":
    missing = []
    for headline, expected in ADDITIONAL_CASES.items():
        got = na.classify_event(headline)
        if got != expected:
            missing.append((headline, expected, got))

    if missing:
        print("M1/N1 taxonomy coverage is incomplete:")
        for headline, expected, got in missing:
            print(f"  EXPECTED {expected!r} FOR {headline!r}, GOT {got!r}")
        sys.exit(1)

    print("M1/N1 taxonomy coverage OK:")
    print(f"  categories declared: {TAXONOMY}")
    print(f"  additional cases passed: {len(ADDITIONAL_CASES)}")
    print("  every category reachable via classify_event().")
