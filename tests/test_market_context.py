"""Market context tests (Sprint C3).

C3 exists because a stock cannot be read in isolation: "+4% while the S&P did
+4%" and "+4% while the S&P did -2%" are the same stock return and completely
different evidence. C2 registered `relative_strength_60d` with an injected
benchmark and no way to get one; C3 is the producer that supplies it.

Two rules are pinned here.

**VIX and the 10Y are never re-fetched as price bars.** Both are registered
macro series (FRED, vintage-aware) from N3. A second Yahoo-sourced copy would
be a second truth for the same quantity (W5).

**An unmapped ticker gets no sector, not a guessed one.** A wrong sector
benchmark produces confident, wrong sector-relative strength — worse than
admitting the sector is unknown.
"""

from __future__ import annotations

import unittest

import pandas as pd

from core.config import (
    CONTEXT_DEFAULT_BENCHMARK,
    CONTEXT_INDEX_PROXIES,
    CONTEXT_MACRO_REFERENCES,
    CONTEXT_RETURN_WINDOW,
    CONTEXT_SECTOR_ETFS,
)
from core.market_context import (
    STATUS_INCOMPLETE,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    MarketContextError,
    benchmark_frame,
    build_market_context,
    context_problems,
    sector_etf_for,
    sector_frame,
    sector_for,
    window_return,
)

AS_OF = "2026-09-15"


def _frame(closes, start="2026-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000] * len(closes),
        },
        index=index,
    )


def _rising(n=120, base=100.0, step=1.0):
    return _frame([base + step * i for i in range(n)])


def _fetcher(overrides=None, fail=()):
    overrides = overrides or {}

    def fetch(symbol, period, interval):
        if symbol in fail:
            raise RuntimeError(f"provider down for {symbol}")
        return overrides.get(symbol, _rising())
    return fetch


def _ok_macro(vix=17.5, yield_10y=4.2):
    return {"status": "OK", "series_values": {"vix": vix, "10y_yield": yield_10y}}


class SectorResolutionTests(unittest.TestCase):
    def test_a_mapped_holding_resolves_to_its_sector_etf(self):
        self.assertEqual(sector_for("NVDA"), "Information Technology")
        self.assertEqual(sector_etf_for("NVDA"), "XLK")

    def test_an_unmapped_ticker_has_no_sector(self):
        """None is a real answer — a guess would be worse than silence."""
        self.assertIsNone(sector_for("ZZZZ"))
        self.assertIsNone(sector_etf_for("ZZZZ"))

    def test_a_broad_fund_is_deliberately_unmapped(self):
        """A fund has no single sector, and an ETF is not its own benchmark."""
        self.assertIsNone(sector_for("VOO"))

    def test_resolution_is_case_insensitive(self):
        self.assertEqual(sector_for("nvda"), sector_for("NVDA"))

    def test_every_mapped_sector_has_an_etf(self):
        """A sector with no ETF would silently lose its industry benchmark."""
        from core.macro_registry import SYMBOL_TO_SECTOR

        for symbol, sector in SYMBOL_TO_SECTOR.items():
            with self.subTest(symbol=symbol):
                self.assertIn(sector, CONTEXT_SECTOR_ETFS)

    def test_the_portfolio_is_broadly_covered(self):
        """Unmapped holdings lose sector-relative context, so keep it small."""
        from fetch_data import PORTFOLIO_TICKERS
        from core.macro_registry import SYMBOL_TO_SECTOR

        unmapped = [t for t in PORTFOLIO_TICKERS if t not in SYMBOL_TO_SECTOR]
        self.assertLess(len(unmapped), 10, f"too many unmapped holdings: {unmapped}")


class WindowReturnTests(unittest.TestCase):
    def test_a_rising_series_returns_positive(self):
        self.assertGreater(window_return(_rising()), 0)

    def test_a_falling_series_returns_negative(self):
        self.assertLess(window_return(_frame([200.0 - i for i in range(120)])), 0)

    def test_short_history_yields_none_not_a_partial_window(self):
        """A partial window would compare different periods across series."""
        self.assertIsNone(window_return(_rising(n=CONTEXT_RETURN_WINDOW - 5)))

    def test_a_missing_frame_yields_none(self):
        self.assertIsNone(window_return(None))


class MacroReferenceTests(unittest.TestCase):
    """VIX and the 10Y come from the macro snapshot, never from price bars."""

    def test_vix_and_10y_are_read_from_the_macro_snapshot(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertEqual(context["macro_readings"]["vix"], 17.5)
        self.assertEqual(context["macro_readings"]["yield_10y"], 4.2)
        self.assertEqual(context["macro_reference_status"], STATUS_OK)

    def test_no_macro_snapshot_yields_no_readings(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher())
        self.assertIsNone(context["macro_readings"]["vix"])
        self.assertEqual(context["macro_reference_status"], STATUS_UNAVAILABLE)

    def test_a_degraded_macro_snapshot_yields_no_neutral_substitute(self):
        """Fail-closed: no value at all, rather than a plausible-looking one."""
        degraded = {"status": "INCOMPLETE", "series_values": {"vix": 17.5, "10y_yield": 4.2}}
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=degraded)
        self.assertIsNone(context["macro_readings"]["vix"])
        self.assertEqual(context["macro_reference_status"], STATUS_UNAVAILABLE)

    def test_vix_is_never_fetched_as_a_price_series(self):
        """W5: a second source for VIX would fork the truth."""
        requested: list[str] = []

        def recording(symbol, period, interval):
            requested.append(symbol)
            return _rising()

        build_market_context("NVDA", AS_OF, fetcher=recording, macro_snapshot=_ok_macro())
        for symbol in requested:
            self.assertNotIn("VIX", symbol.upper())
            self.assertNotIn("TNX", symbol.upper())

    def test_the_macro_reference_table_covers_vix_and_the_10y(self):
        self.assertIn("vix", CONTEXT_MACRO_REFERENCES)
        self.assertIn("yield_10y", CONTEXT_MACRO_REFERENCES)


class ContextAssemblyTests(unittest.TestCase):
    def test_every_declared_index_proxy_is_fetched(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        for name in CONTEXT_INDEX_PROXIES:
            with self.subTest(proxy=name):
                self.assertIn(name, context["frames"])

    def test_a_mapped_holding_gets_an_industry_benchmark(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertEqual(context["industry_benchmark"], "XLK")
        self.assertIsNotNone(sector_frame(context))

    def test_a_healthy_context_is_ok_and_contract_clean(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertEqual(context["status"], STATUS_OK)
        self.assertEqual(context_problems(context), [])

    def test_an_unmapped_ticker_is_incomplete_with_no_benchmark(self):
        context = build_market_context("ZZZZ", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertEqual(context["status"], STATUS_INCOMPLETE)
        self.assertIsNone(context["industry_benchmark"])
        self.assertIsNone(sector_frame(context))
        self.assertIn("no mapped sector", context["reason"])

    def test_a_failed_proxy_degrades_rather_than_raising(self):
        context = build_market_context(
            "NVDA", AS_OF, fetcher=_fetcher(fail={"IWM"}), macro_snapshot=_ok_macro()
        )
        self.assertEqual(context["status"], STATUS_INCOMPLETE)
        self.assertIn("russell2000", context["unavailable"])

    def test_a_total_provider_failure_is_unavailable(self):
        def failing(symbol, period, interval):
            raise RuntimeError("provider down")

        context = build_market_context("NVDA", AS_OF, fetcher=failing, macro_snapshot=_ok_macro())
        self.assertEqual(context["status"], STATUS_UNAVAILABLE)

    def test_an_unknown_benchmark_is_refused(self):
        with self.assertRaises(MarketContextError):
            build_market_context("NVDA", AS_OF, fetcher=_fetcher(), benchmark="ftse100")

    def test_an_empty_ticker_is_refused(self):
        with self.assertRaises(MarketContextError):
            build_market_context("  ", AS_OF, fetcher=_fetcher())

    def test_the_benchmark_is_selectable(self):
        context = build_market_context(
            "NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro(), benchmark="nasdaq100"
        )
        self.assertEqual(context["benchmark_symbol"], CONTEXT_INDEX_PROXIES["nasdaq100"])

    def test_the_default_benchmark_is_declared(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertEqual(context["benchmark"], CONTEXT_DEFAULT_BENCHMARK)

    def test_the_context_is_versioned(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher())
        self.assertTrue(context["calculation_version"])
        self.assertTrue(context["pipeline_version"])


class C2IntegrationTests(unittest.TestCase):
    """The point of C3: C2's relative strength can finally be computed."""

    def test_the_context_supplies_the_benchmark_c2_needs(self):
        from core.chart_features import compute_chart_features

        stock = _rising(n=120, step=2.0)          # outperforming
        flat_market = _frame([100.0] * 120)
        context = build_market_context(
            "NVDA", AS_OF,
            fetcher=_fetcher(overrides={"SPY": flat_market}),
            macro_snapshot=_ok_macro(),
        )
        result = compute_chart_features(stock, benchmark_frame=benchmark_frame(context))
        self.assertIsNotNone(result["features"]["relative_strength_60d"])
        self.assertGreater(result["features"]["relative_strength_60d"], 0)

    def test_without_a_context_relative_strength_stays_none(self):
        """The gap C3 closes, stated as a test."""
        from core.chart_features import compute_chart_features

        result = compute_chart_features(_rising(n=120))
        self.assertIsNone(result["features"]["relative_strength_60d"])

    def test_an_unmapped_holding_yields_no_sector_relative_strength(self):
        from core.chart_features import relative_strength

        context = build_market_context("ZZZZ", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        self.assertIsNone(relative_strength(_rising(n=120), sector_frame(context)))


class ContractValidationTests(unittest.TestCase):
    def test_an_industry_benchmark_without_a_sector_is_a_problem(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        context["sector"] = None
        self.assertTrue(any("without a sector" in problem for problem in context_problems(context)))

    def test_an_ok_context_without_a_benchmark_frame_is_a_problem(self):
        context = build_market_context("NVDA", AS_OF, fetcher=_fetcher(), macro_snapshot=_ok_macro())
        context["frames"] = {}
        self.assertTrue(any("benchmark frame" in problem for problem in context_problems(context)))

    def test_no_sector_etf_is_claimed_by_two_sectors(self):
        etfs = list(CONTEXT_SECTOR_ETFS.values())
        self.assertEqual(len(etfs), len(set(etfs)))


class ReturnsPitTests(unittest.TestCase):
    """The leak this fix closes.

    `returns` was computed on the RAW provider frame, so at a historical as_of
    it measured from the frame's tail -- the future. C5 carried the value into
    a sequence labelled `as_of_t0`, which would have fed post-as_of market
    returns to any C6 model trained on those artifacts.
    """

    @staticmethod
    def _long_frame(n=300, start="2025-06-01"):
        closes = [100.0 + 1.0 * i for i in range(n)]
        index = pd.date_range(start, periods=n, freq="D")
        return pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes,
             "Close": closes, "Volume": [1] * n},
            index=index,
        )

    def _context(self, as_of, frame=None):
        frame = frame if frame is not None else self._long_frame()
        return build_market_context(
            "NVDA", as_of,
            fetcher=lambda symbol, period, interval: frame,
            macro_snapshot=_ok_macro(),
        )

    def test_returns_do_not_read_past_as_of(self):
        frame = self._long_frame()
        early = self._context("2025-09-01", frame)["returns"]["sp500"]
        late = self._context("2026-03-01", frame)["returns"]["sp500"]
        self.assertNotEqual(
            early, late,
            "returns identical across as_of values means they were read off the "
            "frame tail rather than measured at as_of",
        )

    def test_a_later_as_of_sees_a_later_window(self):
        """On a monotonically rising series the trailing 60d return shrinks as
        the base grows, so the two must differ in a predictable direction."""
        frame = self._long_frame()
        early = self._context("2025-09-01", frame)["returns"]["sp500"]
        late = self._context("2026-03-01", frame)["returns"]["sp500"]
        self.assertGreater(early, late)

    def test_insufficient_history_before_as_of_yields_none(self):
        """A partial window is not a return; it compares different periods."""
        frame = self._long_frame(n=300, start="2026-01-01")
        self.assertIsNone(self._context("2026-02-01", frame)["returns"]["sp500"])

    def test_the_measurement_timestamp_is_disclosed(self):
        context = self._context("2025-12-01")
        self.assertEqual(context["returns_as_of"], "2025-12-01 00:00:00")

    def test_window_return_honours_an_explicit_as_of(self):
        frame = self._long_frame()
        unfiltered = window_return(frame)
        filtered = window_return(frame, as_of="2025-09-01")
        self.assertNotEqual(unfiltered, filtered)

    def test_an_as_of_before_all_bars_yields_none(self):
        self.assertIsNone(window_return(self._long_frame(), as_of="2020-01-01"))
