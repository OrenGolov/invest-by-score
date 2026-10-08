"""D2 WHY breakdown tests.

The central claims under test: the components are NOT additive, and the three
the roadmap names which cannot be measured must not render as zeros. These
tests drive the real F6 decomposition rather than reading any data file, so
they mean the same thing on a clean clone.
"""

from __future__ import annotations

import unittest

from core.config import (
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_WIRED_COMPONENTS,
    DECOMP_FUNDAMENTAL,
    DECOMP_HISTORICAL_ANALOG,
    DECOMP_MACRO,
    DECOMP_SENTIMENT,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    DECOMP_TECHNICAL,
    WHY_BARRED_TERMS,
    WHY_MIN_PRESENT,
    WHY_PANEL_BLOCKS_TRADES,
    WHY_PANEL_VERSION,
    WHY_REPORTS_CONTRIBUTIONS,
    WHY_ROADMAP_NAMES,
    WHY_SHOW_ALL_COMPONENTS,
    WHY_STATUSES,
)
from core.forecast_decomposition import decompose_event_forecast
from core.why_panel import (
    WhyPanelError,
    build_why_panel,
    display_name,
    render_why,
    why_problems,
)

ROADMAP_SIX = (
    DECOMP_TECHNICAL,
    DECOMP_FUNDAMENTAL,
    "news_event",
    DECOMP_MACRO,
    "regime",
    DECOMP_SENTIMENT,
)


def live_decomposition():
    """A decomposition from the real F6 machinery."""
    forecast = {
        "value": 0.56,
        "horizon": "20d",
        "samples": 800,
        "as_of": "2026-09-22",
        "event": {"entity": "NVDA", "event_type": "earnings_beat"},
        "stages": {
            "matches": {"observed_share": 0.95, "count": 40},
            "regime": {"label": "bullish", "agreement": 0.9, "trials": 120},
            "forecast": {"effective_samples": 800},
            "chart": {"fields": 10, "similarity": 0.87},
        },
    }
    return decompose_event_forecast(forecast, pool_size=4000)


class NotContributionsTest(unittest.TestCase):
    """The parts overlap and double-count by more than 2x."""

    def test_the_panel_refuses_to_report_contributions(self):
        self.assertFalse(WHY_REPORTS_CONTRIBUTIONS)
        panel = build_why_panel(live_decomposition())
        self.assertFalse(panel["reports_contributions"])

    def test_the_decomposition_is_not_additive(self):
        self.assertFalse(DECOMPOSITION_ADDITIVE)
        panel = build_why_panel(live_decomposition())
        self.assertFalse(panel["additive"])

    def test_no_row_field_names_a_contribution(self):
        panel = build_why_panel(live_decomposition())
        for row in panel["rows"]:
            for key in row:
                for barred in WHY_BARRED_TERMS:
                    self.assertNotIn(
                        barred,
                        str(key).lower(),
                        f"{row['component']}: field {key!r} names a "
                        f"contribution; the parts cannot be added",
                    )

    def test_the_overlap_evidence_travels_with_the_panel(self):
        panel = build_why_panel(live_decomposition())
        self.assertIn("0.131", panel["overlap_evidence"])
        self.assertIn("0.060", panel["overlap_evidence"])

    def test_a_panel_claiming_additivity_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["additive"] = True
        self.assertTrue(why_problems(panel))

    def test_a_panel_claiming_contributions_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["reports_contributions"] = True
        self.assertTrue(why_problems(panel))

    def test_a_contribution_field_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["rows"][0]["contribution"] = 0.064
        self.assertTrue(
            why_problems(panel),
            "a contribution number passed the contract check",
        )


class NotWiredIsNotZeroTest(unittest.TestCase):
    """Three of the six the roadmap names cannot be measured at all."""

    def test_exactly_the_roadmap_three_are_unwired(self):
        unwired = [
            c for c in DECOMPOSITION_COMPONENTS
            if c not in DECOMPOSITION_WIRED_COMPONENTS
        ]
        self.assertEqual(
            sorted(unwired),
            sorted([DECOMP_FUNDAMENTAL, DECOMP_MACRO, DECOMP_SENTIMENT]),
            "the unwired set changed; D2's headline rests on three of the "
            "roadmap's six being unmeasurable",
        )

    def test_unwired_components_report_not_wired(self):
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        for component in (DECOMP_FUNDAMENTAL, DECOMP_MACRO, DECOMP_SENTIMENT):
            self.assertEqual(rows[component]["status"], DECOMP_STATUS_NOT_WIRED)
            self.assertIsNone(
                rows[component]["effect"],
                f"{component}: an unwired component carried an effect",
            )
            self.assertFalse(rows[component]["wired"])

    def test_not_wired_and_absent_stay_distinct(self):
        panel = build_why_panel(live_decomposition())
        self.assertTrue(panel["not_wired"])
        self.assertTrue(panel["absent"])
        self.assertFalse(set(panel["not_wired"]) & set(panel["absent"]))

    def test_an_unwired_component_reported_absent_is_caught(self):
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        rows[DECOMP_MACRO]["status"] = DECOMP_STATUS_ABSENT
        self.assertTrue(
            why_problems(panel),
            "an unwired component reported ABSENT passed — that says the "
            "macro data was checked and had nothing to say",
        )

    def test_a_wired_component_reported_not_wired_is_caught(self):
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        rows[DECOMP_TECHNICAL]["status"] = DECOMP_STATUS_NOT_WIRED
        self.assertTrue(why_problems(panel))

    def test_no_unwired_component_ever_carries_a_number(self):
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        for component in (DECOMP_FUNDAMENTAL, DECOMP_MACRO, DECOMP_SENTIMENT):
            for key, value in rows[component].items():
                if key in ("component", "label", "status", "reason", "wired"):
                    continue
                self.assertIsNone(
                    value,
                    f"{component}: carried {key}={value!r}; an unwired "
                    f"component rendered as a number claims a measurement "
                    f"nobody made",
                )


class AllComponentsShownTest(unittest.TestCase):
    def test_every_f6_component_has_a_row(self):
        panel = build_why_panel(live_decomposition())
        shown = {r["component"] for r in panel["rows"]}
        self.assertEqual(shown, set(DECOMPOSITION_COMPONENTS))

    def test_the_roadmap_six_all_appear(self):
        panel = build_why_panel(live_decomposition())
        shown = {r["component"] for r in panel["rows"]}
        for component in ROADMAP_SIX:
            self.assertIn(component, shown)

    def test_the_seventh_component_is_shown_too(self):
        """historical_analog produced the value; omitting it hides the work."""
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        self.assertIn(DECOMP_HISTORICAL_ANALOG, rows)
        self.assertEqual(
            rows[DECOMP_HISTORICAL_ANALOG]["status"], DECOMP_STATUS_PRESENT
        )

    def test_a_dropped_component_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["rows"] = panel["rows"][:-1]
        self.assertTrue(why_problems(panel))

    def test_every_component_has_a_display_name(self):
        for component in DECOMPOSITION_COMPONENTS:
            self.assertTrue(display_name(component).strip())

    def test_an_unknown_component_is_refused(self):
        with self.assertRaises(WhyPanelError):
            display_name("astrology")

    def test_a_mismatched_label_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["rows"][0]["label"] = "Vibes"
        self.assertTrue(why_problems(panel))


class HeadlineTest(unittest.TestCase):
    def test_a_mostly_empty_panel_says_so(self):
        panel = build_why_panel(live_decomposition())
        self.assertLess(len(panel["present"]), WHY_MIN_PRESENT)
        self.assertIn("not a weighting", panel["headline"])

    def test_the_headline_counts_every_component(self):
        panel = build_why_panel(live_decomposition())
        total = len(panel["present"]) + len(panel["absent"]) + len(panel["not_wired"])
        self.assertEqual(total, len(DECOMPOSITION_COMPONENTS))
        self.assertIn(str(total), panel["headline"])

    def test_a_populated_panel_still_warns_about_addition(self):
        """Even a full panel must say its rows do not add up."""
        decomposition = live_decomposition()
        for component in DECOMPOSITION_WIRED_COMPONENTS:
            decomposition["components"][component] = {
                "status": DECOMP_STATUS_PRESENT,
                "effect": "NARROWED",
                "reason": "evidence entered",
            }
        panel = build_why_panel(decomposition)
        self.assertGreaterEqual(len(panel["present"]), WHY_MIN_PRESENT)
        self.assertIn("not additive", panel["headline"])

    def test_a_silenced_headline_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["headline"] = ""
        self.assertTrue(why_problems(panel))


class InputTest(unittest.TestCase):
    def test_no_decomposition_still_produces_every_row(self):
        panel = build_why_panel(None)
        self.assertEqual(len(panel["rows"]), len(DECOMPOSITION_COMPONENTS))
        self.assertEqual(why_problems(panel), [])

    def test_no_decomposition_reports_nothing_present(self):
        panel = build_why_panel(None)
        self.assertEqual(panel["present"], [])
        self.assertEqual(len(panel["not_wired"]), 3)

    def test_a_non_mapping_decomposition_is_refused(self):
        with self.assertRaises(WhyPanelError):
            build_why_panel("technical was strong")

    def test_non_mapping_components_are_refused(self):
        with self.assertRaises(WhyPanelError):
            build_why_panel({"components": ["technical"]})

    def test_a_component_with_an_unknown_status_is_refused(self):
        with self.assertRaises(WhyPanelError):
            build_why_panel(
                {"components": {DECOMP_TECHNICAL: {"status": "STRONG", "reason": "x"}}}
            )

    def test_a_component_with_no_reason_is_refused(self):
        with self.assertRaises(WhyPanelError):
            build_why_panel(
                {
                    "components": {
                        DECOMP_TECHNICAL: {"status": DECOMP_STATUS_ABSENT, "reason": ""}
                    }
                }
            )


class ContractTest(unittest.TestCase):
    def test_a_clean_panel_has_no_problems(self):
        panel = build_why_panel(live_decomposition())
        self.assertEqual(why_problems(panel), [])
        self.assertEqual(panel["version"], WHY_PANEL_VERSION)

    def test_the_panel_does_not_block_trades(self):
        panel = build_why_panel(live_decomposition())
        self.assertFalse(panel["blocks_trades"])
        self.assertFalse(WHY_PANEL_BLOCKS_TRADES)

    def test_a_panel_that_blocks_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["blocks_trades"] = True
        self.assertTrue(why_problems(panel))

    def test_a_missing_overlap_evidence_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["overlap_evidence"] = ""
        self.assertTrue(why_problems(panel))

    def test_an_effect_on_a_non_present_row_is_caught(self):
        panel = build_why_panel(live_decomposition())
        rows = {r["component"]: r for r in panel["rows"]}
        rows[DECOMP_TECHNICAL]["effect"] = "NARROWED"
        self.assertTrue(why_problems(panel))

    def test_a_reasonless_row_is_caught(self):
        panel = build_why_panel(live_decomposition())
        panel["rows"][0]["reason"] = "  "
        self.assertTrue(why_problems(panel))

    def test_the_panel_reuses_f6_statuses(self):
        from core.config import DECOMPOSITION_STATUSES

        self.assertEqual(tuple(WHY_STATUSES), tuple(DECOMPOSITION_STATUSES))


class RenderTest(unittest.TestCase):
    def test_the_headline_comes_first(self):
        panel = build_why_panel(live_decomposition())
        self.assertEqual(render_why(panel)[0].strip(), panel["headline"])

    def test_every_component_is_rendered_by_its_roadmap_name(self):
        panel = build_why_panel(live_decomposition())
        text = "\n".join(render_why(panel))
        for component in DECOMPOSITION_COMPONENTS:
            self.assertIn(WHY_ROADMAP_NAMES[component], text)

    def test_unwired_components_never_render_as_zero(self):
        """Only the effect COLUMN is under test.

        F6's own reason text legitimately contains numbers — "sentiment
        carries ensemble weight 0.0" is prose explaining why it is unwired,
        not a rendered measurement.
        """
        panel = build_why_panel(live_decomposition())
        for line in render_why(panel):
            if "NOT_WIRED" not in line:
                continue
            effect_column = line.split("NOT_WIRED", 1)[1][:14]
            self.assertNotIn(
                "0.0",
                effect_column,
                "an unwired component rendered an effect of 0.0, which says "
                "it was measured and found irrelevant",
            )
            self.assertIn("—", effect_column)

    def test_lines_stay_readable(self):
        panel = build_why_panel(live_decomposition())
        for line in render_why(panel):
            self.assertLess(len(line), 200)


if __name__ == "__main__":
    unittest.main()
