"""Portfolio ticker list hygiene.

The list is hand-edited, and Python silently concatenates adjacent string
literals — `"QCOM"` followed by `"CRWV"` with no comma becomes the single
symbol `"QCOMCRWV"`, which 404s on every fetch. That exact bug reached the
list and nothing caught it, because a failed ticker is logged and skipped by
design (fetch_portfolio must not let one bad symbol crash a batch).

These checks are offline and structural — they never hit the network.
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path

from fetch_data import PORTFOLIO_TICKERS

# US listings on Yahoo's chart API. A non-US symbol needs a suffix
# (e.g. ".TA"), which the module docstring notes was never exercised here.
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,3})?$")

# Longest real US ticker in common use is 5 characters. Anything longer is
# almost certainly two symbols concatenated by a missing comma.
_MAX_PLAIN_TICKER = 5


class TestPortfolioList(unittest.TestCase):
    def test_no_duplicates(self) -> None:
        """A duplicate doubles batch fetches and raw-store appends."""
        duplicates = {t: c for t, c in Counter(PORTFOLIO_TICKERS).items() if c > 1}
        self.assertEqual(duplicates, {}, f"duplicate tickers: {duplicates}")

    def test_no_concatenated_symbols(self) -> None:
        """Regression: a missing comma silently merged QCOM and CRWV."""
        overlong = [t for t in PORTFOLIO_TICKERS if "." not in t and len(t) > _MAX_PLAIN_TICKER]
        self.assertEqual(
            overlong, [],
            f"tickers longer than {_MAX_PLAIN_TICKER} chars are probably two "
            f"symbols joined by a missing comma: {overlong}",
        )

    def test_every_symbol_is_well_formed(self) -> None:
        for ticker in PORTFOLIO_TICKERS:
            with self.subTest(ticker=ticker):
                self.assertRegex(ticker, _TICKER_PATTERN)

    def test_no_blank_or_whitespace_entries(self) -> None:
        for ticker in PORTFOLIO_TICKERS:
            with self.subTest(ticker=ticker):
                self.assertEqual(ticker, ticker.strip())
                self.assertTrue(ticker)

    def test_source_literals_all_have_separators(self) -> None:
        """Catch the typo at its source, not just in its result.

        The length check only catches merges longer than 5 chars
        ("QCOM" + "CRWV" -> QCOMCRWV). A short merge slips past it:
        "V" + "BE" -> VBE is 3 chars, so it passes both the length and
        format checks while being just as broken — no such ticker exists,
        and fetch_portfolio logs and skips it silently.

        Reading the source text catches a merge at any length.
        """
        source = (Path(__file__).resolve().parent.parent / "fetch_data.py").read_text(
            encoding="utf-8"
        )
        block = re.search(r"PORTFOLIO_TICKERS = \[(.*?)\n\]", source, re.S)
        self.assertIsNotNone(block, "could not locate PORTFOLIO_TICKERS")
        self.assertIsNone(
            re.search(r'"\s*\n\s*"', block.group(1)),
            "two adjacent string literals with no comma — Python will "
            "concatenate them into one bogus symbol",
        )


if __name__ == "__main__":
    unittest.main()
