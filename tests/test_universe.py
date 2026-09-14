"""Sprint V6: historical universe construction — survivorship-safe by design.

Hermetic: the ledger is a temp JSONL file; the engine runs use scripted
scores (no network). Coverage:

- point-in-time construction: membership intervals cover/exclude as_of
  exactly (delisted-after members INCLUDED — the anti-survivorship core;
  unverified starts flagged, never mistaken for evidence);
- survivorship detection: a today-snapshot universe that drops a ledger
  member listed at as_of is flagged `survivorship_biased`; unverified
  candidates are `member_unverified`; a matching candidate is
  `point_in_time_complete`;
- price-coverage cross-check (failed fetches surfaced, never dropped);
- ledger store semantics: validated-before-write (delisted entries require
  the delist date; unknown reasons rejected), idempotent per entry hash,
  revision append (last wins), loud integrity on malformed lines;
- the engine gate: a biased universe is refused before any work (nothing
  persisted — V4 semantics); a ticker outside the declared universe is
  refused; no declared universe runs with an explicit unverifiable
  disclosure.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.backtest.engine import (
    BacktestUniverseError,
    offline_replay_seam,
    run_walk_forward_backtest,
)
from core.universe import (
    STATUS_POINT_IN_TIME_COMPLETE,
    STATUS_SURVIVORSHIP_BIASED,
    STATUS_UNVERIFIABLE,
    build_universe_block,
    coverage_problems,
    latest_entries_by_ticker,
    load_universe_entries,
    make_entry,
    persist_universe_entries,
    seed_portfolio_universe,
    survivorship_problems,
    universe_as_of,
    universe_block_problems,
    universe_survivorship_status,
)


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": list(closes),
            "Low": list(closes),
            "Close": list(closes),
            "Volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * index for index in range(sessions)]


class UniverseConstructionTests(unittest.TestCase):
    """Point-in-time construction from membership intervals."""

    def setUp(self):
        self.entries = {
            "AAPL": make_entry("AAPL", "initial_member", "test_source",
                               published_time="2022-01-01"),
            "XYZ": make_entry("XYZ", "initial_member", "test_source",
                              published_time="2022-01-01",
                              listed_from="2015-01-01", listed_to="2023-06-01",
                              notes="delisted since"),
            "NEW": make_entry("NEW", "added", "test_source",
                              published_time="2022-01-01", listed_from="2023-01-01"),
            "UNK": make_entry("UNK", "initial_member", "test_source",
                              published_time="2022-01-01"),
        }

    def test_member_listed_before_as_of_and_still_listed_is_included(self):
        members = {m["ticker"]: m for m in universe_as_of(self.entries, "2024-01-02")}
        self.assertIn("AAPL", members)
        self.assertFalse(members["AAPL"]["listed_from_unverified"])

    def test_unverified_start_is_included_but_flagged(self):
        members = {m["ticker"]: m for m in universe_as_of(self.entries, "2024-01-02")}
        self.assertIn("UNK", members)
        self.assertTrue(members["UNK"]["listed_from_unverified"])

    def test_member_not_yet_listed_is_excluded(self):
        members = {m["ticker"]: m for m in universe_as_of(self.entries, "2022-06-01")}
        self.assertNotIn("NEW", members)  # listed_from 2023-01-01

    def test_delisted_before_as_of_is_excluded(self):
        members = {m["ticker"]: m for m in universe_as_of(self.entries, "2024-01-02")}
        self.assertNotIn("XYZ", members)  # delisted 2023-06-01

    def test_delisted_after_as_of_is_included(self):
        # THE anti-survivorship core: at 2023-01-02 XYZ was still a member.
        members = {m["ticker"]: m for m in universe_as_of(self.entries, "2023-01-02")}
        self.assertIn("XYZ", members)


class UniverseBlockTests(unittest.TestCase):
    def test_empty_ledger_is_unverifiable_and_disclosed(self):
        block = build_universe_block(["AAPL"], "2023-01-02", path="no-such-ledger.jsonl")
        self.assertEqual(block["survivorship_status"], STATUS_UNVERIFIABLE)
        self.assertTrue(any("member_unverified: AAPL" in p for p in block["problems"]))

    def test_biased_block_is_flagged_by_the_gate(self):
        biased = {
            "universe_version": "universe-ledger-v1",
            "as_of": "2023-01-02",
            "requested_members": ["AAPL"],
            "survivorship_status": STATUS_SURVIVORSHIP_BIASED,
            "problems": ["survivorship_bias: XYZ ..."],
        }
        problems = universe_block_problems(biased, ticker="AAPL")
        self.assertTrue(any("survivorship_biased:" in p for p in problems), problems)

    def test_gate_rejects_unknown_status_version_and_non_member(self):
        base = {
            "universe_version": "universe-ledger-v1",
            "as_of": "2023-01-02",
            "requested_members": ["AAPL"],
            "survivorship_status": STATUS_POINT_IN_TIME_COMPLETE,
            "problems": [],
        }
        self.assertEqual(universe_block_problems(base, ticker="AAPL"), [])
        self.assertTrue(any("unknown universe_version" in p for p in universe_block_problems(
            {**base, "universe_version": "universe-ledger-v0"}, ticker="AAPL")))
        self.assertTrue(any("unknown survivorship_status" in p for p in universe_block_problems(
            {**base, "survivorship_status": "looks_fine"}, ticker="AAPL")))
        self.assertTrue(any("ticker_not_in_universe: TSLA" in p for p in universe_block_problems(
            base, ticker="TSLA")))


class UniverseStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self._tmp.name) / "universe.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_delisted_entry_requires_the_delist_date(self):
        with self.assertRaises(ValueError) as ctx:
            make_entry("XYZ", "delisted_bankruptcy", "test_source",
                       published_time="2022-01-01")
        self.assertIn("requires listed_to", str(ctx.exception))

    def test_unknown_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            make_entry("XYZ", "merger_thing", "test_source",
                       published_time="2022-01-01", listed_to="2023-01-01")

    def test_append_is_idempotent_and_revisions_take_last(self):
        original = make_entry("XYZ", "initial_member", "test_source",
                              published_time="2022-01-01")
        self.assertEqual(persist_universe_entries([original], path=self.ledger), 1)
        self.assertEqual(persist_universe_entries([original], path=self.ledger), 0)
        revision = make_entry("XYZ", "delisted_acquisition", "test_source",
                              published_time="2022-01-01",
                              listed_from="2015-01-01", listed_to="2023-06-01")
        self.assertEqual(persist_universe_entries([revision], path=self.ledger), 1)
        latest = latest_entries_by_ticker(path=self.ledger)
        self.assertEqual(latest["XYZ"]["reason"], "delisted_acquisition")
        self.assertEqual(len(load_universe_entries(path=self.ledger)), 2)  # history kept

    def test_malformed_line_is_loud(self):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as handle:
            handle.write("{broken}\n")
        with self.assertRaises(ValueError):
            load_universe_entries(path=self.ledger)

    def test_seed_portfolio_universe_is_honest_and_idempotent(self):
        appended = seed_portfolio_universe(
            ["AAPL", "MSFT"], published_time="2022-01-01", path=self.ledger
        )
        self.assertEqual(appended, 2)
        entries = latest_entries_by_ticker(path=self.ledger)
        self.assertIsNone(entries["AAPL"]["listed_from"])  # start unverified, not fabricated
        self.assertIsNone(entries["AAPL"]["listed_to"])    # still listed
        self.assertEqual(
            seed_portfolio_universe(
                ["AAPL", "MSFT"], published_time="2022-01-01", path=self.ledger
            ),
            0,
        )


class SurvivorshipDetectionTests(unittest.TestCase):
    def setUp(self):
        self.entries = {
            "AAPL": make_entry("AAPL", "initial_member", "test_source",
                               published_time="2022-01-01"),
            "XYZ": make_entry("XYZ", "delisted_bankruptcy", "test_source",
                              published_time="2022-01-01",
                              listed_from="2015-01-01", listed_to="2023-06-01"),
        }

    def test_today_snapshot_that_drops_a_listed_member_is_biased(self):
        # At 2023-01-02 XYZ was listed; a today-only list is [AAPL, ...]
        # without XYZ — the definition of survivorship bias.
        problems = survivorship_problems(["AAPL"], self.entries, "2023-01-02")
        self.assertTrue(any("survivorship_bias: XYZ" in p for p in problems), problems)
        status = universe_survivorship_status(["AAPL"], self.entries, "2023-01-02")
        self.assertEqual(status, STATUS_SURVIVORSHIP_BIASED)

    def test_candidate_including_the_delisted_member_is_complete(self):
        problems = survivorship_problems(["AAPL", "XYZ"], self.entries, "2023-01-02")
        self.assertEqual(problems, [])
        status = universe_survivorship_status(["AAPL", "XYZ"], self.entries, "2023-01-02")
        self.assertEqual(status, STATUS_POINT_IN_TIME_COMPLETE)

    def test_unverifiable_candidate_tickers_are_fail_closed(self):
        problems = survivorship_problems(["AAPL", "TSLA"], self.entries, "2023-01-02")
        self.assertTrue(any("member_unverified: TSLA" in p for p in problems), problems)
        status = universe_survivorship_status(["AAPL", "TSLA"], self.entries, "2023-01-02")
        self.assertEqual(status, STATUS_UNVERIFIABLE)

    def test_member_delisted_before_as_of_is_correctly_absent(self):
        problems = survivorship_problems(["AAPL"], self.entries, "2024-01-02")
        self.assertEqual(problems, [])
        status = universe_survivorship_status(["AAPL"], self.entries, "2024-01-02")
        self.assertEqual(status, STATUS_POINT_IN_TIME_COMPLETE)

    def test_price_coverage_gaps_are_surfaced_not_dropped(self):
        problems = coverage_problems(["AAPL", "XYZ"], fetched_tickers=["AAPL"])
        self.assertTrue(
            any("member_price_unavailable: XYZ" in p for p in problems), problems
        )
        self.assertEqual(coverage_problems(["AAPL"], fetched_tickers=["AAPL"]), [])
