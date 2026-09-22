"""D1 combined research view tests.

The central claim under test: a mostly-empty system must render as mostly
empty. These tests build their snapshots through the real F-sprint assembly
rather than reading any data file, so they mean the same thing on a clean
clone.
"""

from __future__ import annotations

import unittest

from core.config import (
    RESEARCH_COVERAGE_WARN,
    RESEARCH_PANEL_CONFIDENCE,
    RESEARCH_PANEL_FORECAST,
    RESEARCH_PANEL_QUALITY,
    RESEARCH_PANEL_REGIME,
    RESEARCH_PANEL_RISK,
    RESEARCH_PANELS,
    RESEARCH_RETURN_IS_ABSENT,
    RESEARCH_SHOW_RETURN_COLUMN,
    RESEARCH_VIEW_BLOCKS_TRADES,
    RESEARCH_VIEW_HORIZONS,
    RESEARCH_VIEW_VERSION,
    SNAPSHOT_STATUSES,
    VIEW_CELL_ABSENT,
    VIEW_CELL_PRESENT,
    VIEW_CELL_REFUSED,
    VIEW_CELL_STATES,
)
from core.forecast_confidence import assess_confidence
from core.forecast_snapshot import build_forecast_snapshot
from core.research_view import (
    ResearchViewError,
    build_research_view,
    cell_from_field,
    forecast_row,
    render_view,
    view_problems,
)

TICKER = "NVDA"
AS_OF = "2026-09-22"


def bare_snapshots():
    """What the live system actually produces: metadata only."""
    return {
        h: build_forecast_snapshot(TICKER, AS_OF, h)
        for h in RESEARCH_VIEW_HORIZONS
    }


def populated_snapshots():
    """A snapshot set with the inputs the F sprints would supply."""
    confidence = assess_confidence(
        samples=800,
        interval={"low": 0.48, "high": 0.60},
        observed_share=0.95,
        regime_agreement=0.9,
        similarities=[0.9, 0.85],
        features_present=10,
        features_expected=10,
        probability=0.56,
    )
    base = {
        "value": 0.56,
        "samples": 800,
        "interval": {"low": 0.48, "high": 0.60},
        "as_of": AS_OF,
        "event": {"entity": TICKER},
    }
    return {
        h: build_forecast_snapshot(
            TICKER,
            AS_OF,
            h,
            event_forecast=dict(base, horizon=h),
            confidence=confidence,
            regime="bullish",
        )
        for h in RESEARCH_VIEW_HORIZONS
    }


class TheDecidingMeasurementTest(unittest.TestCase):
    """A mostly-empty system renders as mostly empty."""

    def test_the_live_system_is_mostly_unavailable(self):
        for horizon, snapshot in bare_snapshots().items():
            self.assertEqual(
                len(snapshot["present"]),
                5,
                f"{horizon}: the live snapshot no longer has 5 PRESENT "
                f"fields; D1's headline rests on that measurement",
            )
            self.assertIn("expected_return", snapshot["absent"])

    def test_a_bare_view_leads_with_what_is_missing(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        self.assertTrue(view["coverage"]["below_threshold"])
        self.assertIn("available", view["headline"])
        self.assertIn(
            "Read the gaps",
            view["headline"],
            "a mostly-empty view did not tell the reader to read the gaps "
            "first — rendered as blanks it reads as a working system having "
            "a quiet day",
        )

    def test_a_bare_view_has_almost_nothing_present(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        coverage = view["coverage"]
        self.assertLess(coverage["share_present"], 0.25)
        self.assertGreater(coverage["refused"], coverage["present"])

    def test_a_populated_view_is_not_warned(self):
        """The headline must be capable of staying quiet."""
        view = build_research_view(
            TICKER,
            AS_OF,
            populated_snapshots(),
            quality={"score": 78.5, "action": "BUY"},
            risk={"veto": False, "veto_rule_ids": []},
            portfolio={"verdict": "PROCEED", "reason": "fits the book"},
        )
        self.assertFalse(
            view["coverage"]["below_threshold"],
            "a well-populated view still warned — a headline that always "
            "warns carries no information",
        )
        self.assertEqual(view_problems(view), [])


class ReturnColumnTest(unittest.TestCase):
    """Shown, and always ABSENT."""

    def test_the_column_exists(self):
        self.assertTrue(RESEARCH_SHOW_RETURN_COLUMN)
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        for row in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]:
            self.assertIn(
                "expected_return",
                row,
                "the return column was omitted, hiding that the field was "
                "requested and could not be produced",
            )

    def test_every_return_cell_is_absent(self):
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        for row in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]:
            cell = row["expected_return"]
            self.assertEqual(cell["state"], VIEW_CELL_ABSENT)
            self.assertIsNone(cell["value"])
            self.assertTrue(cell["reason"].strip())

    def test_a_present_return_is_caught(self):
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        view["panels"][RESEARCH_PANEL_FORECAST]["rows"][0]["expected_return"] = {
            "state": VIEW_CELL_PRESENT,
            "value": 0.031,
            "reason": "model says so",
        }
        self.assertTrue(
            view_problems(view),
            "a fabricated expected return passed the contract check",
        )

    def test_the_config_records_why(self):
        self.assertTrue(RESEARCH_RETURN_IS_ABSENT)


class CellStateTest(unittest.TestCase):
    """PRESENT / ABSENT / REFUSED, never collapsed."""

    def test_the_view_reuses_the_snapshot_vocabulary(self):
        self.assertEqual(set(VIEW_CELL_STATES), set(SNAPSHOT_STATUSES))

    def test_absent_and_refused_stay_distinct(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        states = {
            cell["state"]
            for row in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]
            for cell in row.values()
            if isinstance(cell, dict)
        }
        self.assertIn(VIEW_CELL_ABSENT, states)
        self.assertIn(VIEW_CELL_REFUSED, states)

    def test_a_non_present_cell_cannot_carry_a_value(self):
        """The guard sits in _cell, where every cell is built."""
        from core.research_view import _cell

        for state in (VIEW_CELL_ABSENT, VIEW_CELL_REFUSED):
            with self.assertRaises(ResearchViewError):
                _cell(state, value=0.0, reason="looks measured")

    def test_a_non_present_cell_must_carry_a_reason(self):
        from core.research_view import _cell

        for state in (VIEW_CELL_ABSENT, VIEW_CELL_REFUSED):
            with self.assertRaises(ResearchViewError):
                _cell(state, reason="  ")

    def test_a_refused_field_becomes_a_refused_cell(self):
        cell = cell_from_field(
            {"status": VIEW_CELL_REFUSED, "reason": "nothing supplied"},
            "probability_up",
        )
        self.assertEqual(cell["state"], VIEW_CELL_REFUSED)
        self.assertIsNone(cell["value"])
        self.assertTrue(cell["reason"].strip())

    def test_an_absent_field_keeps_its_standing_reason(self):
        cell = cell_from_field(
            {"status": VIEW_CELL_ABSENT, "reason": ""}, "expected_return"
        )
        self.assertEqual(cell["state"], VIEW_CELL_ABSENT)
        self.assertIn("model", cell["reason"].lower())

    def test_a_missing_field_is_refused_not_invented(self):
        cell = cell_from_field(None, "confidence")
        self.assertEqual(cell["state"], VIEW_CELL_REFUSED)

    def test_an_unknown_status_is_refused(self):
        with self.assertRaises(ResearchViewError):
            cell_from_field({"status": "PROBABLY"}, "confidence")

    def test_a_non_mapping_field_is_refused(self):
        with self.assertRaises(ResearchViewError):
            cell_from_field("present", "confidence")


class PanelTest(unittest.TestCase):
    """The sprint requires all five shown TOGETHER."""

    def test_every_panel_is_present(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        for panel in RESEARCH_PANELS:
            self.assertIn(panel, view["panels"])

    def test_every_requested_horizon_appears(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        shown = {
            row["horizon"] for row in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]
        }
        self.assertEqual(shown, set(RESEARCH_VIEW_HORIZONS))

    def test_a_missing_horizon_still_appears_as_refused(self):
        """A dropped horizon must not look like one never asked for."""
        snapshots = populated_snapshots()
        del snapshots["60d"]
        view = build_research_view(TICKER, AS_OF, snapshots)
        rows = {r["horizon"]: r for r in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]}
        self.assertIn("60d", rows)
        self.assertEqual(rows["60d"]["probability_up"]["state"], VIEW_CELL_REFUSED)
        self.assertEqual(view_problems(view), [])

    def test_the_risk_panel_shows_w2_and_r7_separately(self):
        view = build_research_view(
            TICKER,
            AS_OF,
            bare_snapshots(),
            risk={"veto": True, "veto_rule_ids": ["data_quality_below_threshold"]},
            portfolio={"verdict": "NO_TRADE", "reason": "the book refuses"},
        )
        panel = view["panels"][RESEARCH_PANEL_RISK]
        self.assertEqual(panel["veto"]["value"], True)
        self.assertEqual(panel["portfolio_verdict"]["value"], "NO_TRADE")
        self.assertIn("W2", panel["note"])
        self.assertIn("R7", panel["note"])

    def test_missing_risk_inputs_are_refused_not_assumed_safe(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        panel = view["panels"][RESEARCH_PANEL_RISK]
        self.assertEqual(panel["veto"]["state"], VIEW_CELL_REFUSED)
        self.assertEqual(panel["portfolio_verdict"]["state"], VIEW_CELL_REFUSED)

    def test_the_confidence_panel_reads_the_same_rows(self):
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        forecast_rows = {
            r["horizon"]: r["confidence"]
            for r in view["panels"][RESEARCH_PANEL_FORECAST]["rows"]
        }
        panel = view["panels"][RESEARCH_PANEL_CONFIDENCE]["per_horizon"]
        self.assertEqual(panel, forecast_rows)

    def test_the_quality_panel_refuses_when_no_score_is_supplied(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        panel = view["panels"][RESEARCH_PANEL_QUALITY]
        self.assertEqual(panel["score"]["state"], VIEW_CELL_REFUSED)

    def test_the_regime_panel_falls_back_to_the_snapshot(self):
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        cell = view["panels"][RESEARCH_PANEL_REGIME]["regime"]
        self.assertEqual(cell["state"], VIEW_CELL_PRESENT)
        self.assertEqual(cell["value"], "bullish")


class ContractTest(unittest.TestCase):
    def test_a_clean_view_has_no_problems(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        self.assertEqual(view_problems(view), [])
        self.assertEqual(view["version"], RESEARCH_VIEW_VERSION)

    def test_the_view_does_not_block_trades(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        self.assertFalse(view["blocks_trades"])
        self.assertFalse(RESEARCH_VIEW_BLOCKS_TRADES)

    def test_a_view_that_blocks_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["blocks_trades"] = True
        self.assertTrue(view_problems(view))

    def test_a_missing_panel_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        del view["panels"][RESEARCH_PANEL_RISK]
        self.assertTrue(view_problems(view))

    def test_a_dropped_horizon_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["panels"][RESEARCH_PANEL_FORECAST]["rows"] = view["panels"][
            RESEARCH_PANEL_FORECAST
        ]["rows"][:-1]
        self.assertTrue(view_problems(view))

    def test_a_valued_refusal_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["panels"][RESEARCH_PANEL_QUALITY]["score"]["value"] = 0.0
        self.assertTrue(
            view_problems(view),
            "a REFUSED cell carrying 0.0 passed — that is the exact failure "
            "this view exists to prevent",
        )

    def test_a_reasonless_refusal_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["panels"][RESEARCH_PANEL_QUALITY]["score"]["reason"] = "  "
        self.assertTrue(view_problems(view))

    def test_a_present_cell_with_no_value_is_caught(self):
        view = build_research_view(
            TICKER, AS_OF, bare_snapshots(), quality={"score": 70.0, "action": "HOLD"}
        )
        view["panels"][RESEARCH_PANEL_QUALITY]["score"]["value"] = None
        self.assertTrue(view_problems(view))

    def test_a_miscounted_coverage_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["coverage"]["cells"] = 999
        self.assertTrue(view_problems(view))

    def test_a_silenced_headline_is_caught(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        view["headline"] = ""
        self.assertTrue(view_problems(view))

    def test_a_missing_ticker_is_refused(self):
        with self.assertRaises(ResearchViewError):
            build_research_view("", AS_OF, {})

    def test_a_non_mapping_snapshot_set_is_refused(self):
        with self.assertRaises(ResearchViewError):
            build_research_view(TICKER, AS_OF, [])

    def test_a_non_mapping_snapshot_is_refused(self):
        with self.assertRaises(ResearchViewError):
            forecast_row("snapshot", "20d")


class RenderTest(unittest.TestCase):
    def test_the_headline_comes_first(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        lines = render_view(view)
        self.assertIn(TICKER, lines[0])
        self.assertEqual(lines[1].strip(), view["headline"])

    def test_every_horizon_is_rendered(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        text = "\n".join(render_view(view))
        for horizon in RESEARCH_VIEW_HORIZONS:
            self.assertIn(horizon, text)

    def test_a_structured_value_renders_as_a_scalar(self):
        """F7's whole assessment must not land in a table column."""
        view = build_research_view(TICKER, AS_OF, populated_snapshots())
        for line in render_view(view):
            self.assertNotIn(
                "aggregation_evidence",
                line,
                "the confidence cell dumped its whole assessment into the "
                "table instead of the number a reader needs",
            )
            self.assertLess(len(line), 200)

    def test_unavailable_values_render_as_their_state(self):
        view = build_research_view(TICKER, AS_OF, bare_snapshots())
        text = "\n".join(render_view(view))
        self.assertIn(VIEW_CELL_ABSENT, text)
        self.assertIn(VIEW_CELL_REFUSED, text)
        self.assertNotIn(
            "0.0",
            text,
            "an unavailable value rendered as a number",
        )


if __name__ == "__main__":
    unittest.main()
