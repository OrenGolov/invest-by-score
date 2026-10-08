"""D3 event context panel tests.

The central claim under test: a median historical response quoted without its
event-associated share overstates what the event itself explains. These tests
build their memories in-process rather than reading the gitignored event
memory store, so they mean the same thing on a clean clone.
"""

from __future__ import annotations

import random
import statistics
import unittest

from core.config import (
    EVENT_CONTEXT_ASSOCIATION_CAUTION,
    EVENT_CONTEXT_BLOCKS_TRADES,
    EVENT_CONTEXT_REQUIRE_ASSOCIATION_SHARE,
    EVENT_CONTEXT_SHOW_PROVENANCE,
    EVENT_CONTEXT_STATUS_INSUFFICIENT,
    EVENT_CONTEXT_STATUS_MEASURED,
    EVENT_CONTEXT_STATUS_NO_EVENT,
    EVENT_CONTEXT_VERSION,
    EVENT_MEMORY_MIN_ANALOGS,
    EVENT_MEMORY_MIN_SIMILARITY,
    MEMORY_PROVENANCE_INFERRED,
)
from core.event_context_panel import (
    EventContextError,
    build_event_context,
    context_problems,
    render_context,
)
from core.event_memory import EventMemory, analog_summary, find_analogs

CURRENT = {"trend": 0.62, "momentum": 0.51, "volatility": 0.30}
EVENT = {
    "event_id": "e-now",
    "event_type": "earnings_beat",
    "direction": "positive",
    "published_time": "2026-09-19",
    "provenance": "observed",
}


def memories(seed=7, count=15, associated_every=3):
    """Analogs whose confounded members moved differently from the rest."""
    rng = random.Random(seed)
    built = []
    for index in range(count):
        chart = {
            "trend": 0.6 + rng.gauss(0, 0.03),
            "momentum": 0.5 + rng.gauss(0, 0.03),
            "volatility": 0.3 + rng.gauss(0, 0.02),
        }
        associated = index % associated_every != 0
        response = (
            rng.gauss(0.03, 0.04) if associated else rng.gauss(-0.02, 0.04)
        )
        built.append(
            EventMemory(
                event_id=f"e{index}",
                ticker="NVDA",
                published_time="2025-01-10",
                event_type="earnings_beat",
                direction="positive",
                chart_state=chart,
                response={"20d": {"abnormal_return": response}},
                attribution={
                    "20d": "event_associated" if associated else "confounded"
                },
                provenance="observed",
            )
        )
    return built


def summary(seed=7, count=15):
    analogs = find_analogs(CURRENT, "earnings_beat", memories=memories(seed, count))
    return analog_summary(analogs, "20d")


class TheDecidingMeasurementTest(unittest.TestCase):
    """Confounded analogs move the median a reader would act on."""

    def test_excluding_confounded_analogs_shifts_the_median(self):
        shifts = []
        for seed in range(1, 41):
            pool = memories(seed=seed)
            analogs = find_analogs(CURRENT, "earnings_beat", memories=pool)
            associated = [
                a for a in analogs if a["memory"].is_event_associated("20d")
            ]
            if len(analogs) < EVENT_MEMORY_MIN_ANALOGS:
                continue
            if len(associated) < EVENT_MEMORY_MIN_ANALOGS:
                continue
            everything = analog_summary(analogs, "20d")
            clean = analog_summary(associated, "20d")
            shifts.append(
                clean["median_response"] - everything["median_response"]
            )
        self.assertGreaterEqual(len(shifts), 20, "too few comparable seeds")
        material = sum(1 for s in shifts if abs(s) > 0.01)
        self.assertGreater(
            material,
            len(shifts) * 0.3,
            f"only {material} of {len(shifts)} analog sets shifted by more "
            f"than a percentage point when confounded analogs were excluded. "
            f"D3's caution rests on that shift being material; if it has "
            f"vanished the design must be re-derived rather than kept",
        )

    def test_the_median_carries_its_association_share(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        analogs = panel["analogs"]
        self.assertIsNotNone(analogs["median_response"])
        self.assertIsNotNone(
            analogs["event_associated_share"],
            "a median was reported with no event-associated share",
        )

    def test_a_low_association_share_carries_a_caution(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        share = panel["analogs"]["event_associated_share"]
        self.assertLess(share, EVENT_CONTEXT_ASSOCIATION_CAUTION)
        self.assertTrue(
            panel["analogs"].get("caution", "").strip(),
            "a third of the evidence was attributed elsewhere and the panel "
            "said nothing",
        )

    def test_the_headline_names_the_association_share(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        self.assertIn("attributed to the event itself", panel["headline"])

    def test_a_median_without_a_share_is_refused(self):
        raw = dict(summary())
        raw.pop("event_associated_share")
        with self.assertRaises(EventContextError):
            build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=raw)

    def test_a_missing_caution_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        panel["analogs"].pop("caution")
        self.assertTrue(context_problems(panel))

    def test_a_high_association_share_needs_no_caution(self):
        """The caution must be capable of staying quiet."""
        clean = dict(summary())
        clean["event_associated_share"] = 0.95
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=clean)
        self.assertNotIn("caution", panel["analogs"])
        self.assertEqual(context_problems(panel), [])


class InsufficientIsNotZeroTest(unittest.TestCase):
    """Too few analogs is not a measured absence of movement."""

    def test_an_empty_store_reports_insufficient(self):
        panel = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
        )
        analogs = panel["analogs"]
        self.assertEqual(analogs["status"], EVENT_CONTEXT_STATUS_INSUFFICIENT)
        self.assertEqual(analogs["analog_count"], 0)

    def test_an_insufficient_set_has_no_median(self):
        panel = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
        )
        for field in ("median_response", "setup_similarity", "event_associated_share"):
            self.assertIsNone(
                panel["analogs"][field],
                f"{field} was supplied for a refused summary; 0.0 there reads "
                f"as 'no historical move'",
            )

    def test_an_insufficient_set_carries_its_reason(self):
        panel = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
        )
        self.assertIn("typical", panel["analogs"]["reason"])
        self.assertEqual(context_problems(panel), [])

    def test_a_refused_summary_carrying_statistics_is_caught(self):
        panel = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
        )
        panel["analogs"]["median_response"] = 0.0
        self.assertTrue(
            context_problems(panel),
            "a refused summary carrying a 0.0 median passed the contract check",
        )

    def test_a_measured_summary_below_the_floor_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        panel["analogs"]["analog_count"] = 2
        self.assertTrue(context_problems(panel))


class NoEventTest(unittest.TestCase):
    """Nothing happened is not the same as nothing could be learned."""

    def test_no_event_is_its_own_status(self):
        panel = build_event_context("NVDA", "2026-09-22")
        self.assertEqual(panel["event"]["status"], EVENT_CONTEXT_STATUS_NO_EVENT)
        self.assertNotEqual(
            EVENT_CONTEXT_STATUS_NO_EVENT, EVENT_CONTEXT_STATUS_INSUFFICIENT
        )

    def test_no_event_carries_no_event_fields(self):
        panel = build_event_context("NVDA", "2026-09-22")
        for field in ("event_id", "event_type", "direction"):
            self.assertIsNone(panel["event"][field])
        self.assertEqual(context_problems(panel), [])

    def test_no_event_says_so_in_the_headline(self):
        panel = build_event_context("NVDA", "2026-09-22")
        self.assertIn("no recent event", panel["headline"])

    def test_a_no_event_block_carrying_a_type_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22")
        panel["event"]["event_type"] = "earnings_beat"
        self.assertTrue(context_problems(panel))


class ProvenanceTest(unittest.TestCase):
    """An inferred event must not read as an observed one."""

    def test_provenance_is_shown(self):
        self.assertTrue(EVENT_CONTEXT_SHOW_PROVENANCE)
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        self.assertEqual(panel["event"]["provenance"], "observed")
        self.assertFalse(panel["event"]["inferred"])

    def test_an_inferred_event_is_marked(self):
        event = dict(EVENT, provenance=MEMORY_PROVENANCE_INFERRED)
        panel = build_event_context("NVDA", "2026-09-22", event=event)
        self.assertTrue(panel["event"]["inferred"])
        self.assertIn("INFERRED", panel["event"]["provenance_note"])

    def test_an_unlabelled_event_is_not_treated_as_observed(self):
        event = dict(EVENT)
        event.pop("provenance")
        panel = build_event_context("NVDA", "2026-09-22", event=event)
        self.assertIsNone(panel["event"]["provenance"])
        self.assertIn("not treated as observed", panel["event"]["provenance_note"])

    def test_a_missing_provenance_field_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        panel["event"].pop("provenance")
        self.assertTrue(context_problems(panel))


class ContractTest(unittest.TestCase):
    def test_a_clean_panel_has_no_problems(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        self.assertEqual(context_problems(panel), [])
        self.assertEqual(panel["version"], EVENT_CONTEXT_VERSION)

    def test_the_panel_does_not_block_trades(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        self.assertFalse(panel["blocks_trades"])
        self.assertFalse(EVENT_CONTEXT_BLOCKS_TRADES)

    def test_a_panel_that_blocks_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        panel["blocks_trades"] = True
        self.assertTrue(context_problems(panel))

    def test_the_disclaimer_denies_causation(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        self.assertIn("not a forecast", panel["disclaimer"])
        self.assertIn("caused", panel["disclaimer"])

    def test_a_stripped_disclaimer_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        panel["disclaimer"] = "historical analogs"
        self.assertTrue(context_problems(panel))

    def test_similarity_below_the_retrieval_bar_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        panel["analogs"]["setup_similarity"] = EVENT_MEMORY_MIN_SIMILARITY - 0.2
        self.assertTrue(
            context_problems(panel),
            "analogs below the retrieval bar passed as comparable setups",
        )

    def test_a_silenced_headline_is_caught(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        panel["headline"] = ""
        self.assertTrue(context_problems(panel))

    def test_a_missing_ticker_is_refused(self):
        with self.assertRaises(EventContextError):
            build_event_context("", "2026-09-22")

    def test_a_non_mapping_event_is_refused(self):
        with self.assertRaises(EventContextError):
            build_event_context("NVDA", "2026-09-22", event="earnings")

    def test_a_non_mapping_summary_is_refused(self):
        with self.assertRaises(EventContextError):
            build_event_context("NVDA", "2026-09-22", analogs="lots")

    def test_an_unknown_analog_status_is_refused(self):
        with self.assertRaises(EventContextError):
            build_event_context(
                "NVDA", "2026-09-22", analogs={"status": "probably_fine"}
            )


class RenderTest(unittest.TestCase):
    def test_the_headline_comes_first(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        self.assertEqual(render_context(panel)[0].strip(), panel["headline"])

    def test_provenance_reaches_the_render(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT)
        self.assertIn("observed", "\n".join(render_context(panel)))

    def test_the_caution_reaches_the_render(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        self.assertIn("!!", "\n".join(render_context(panel)))

    def test_an_absent_median_renders_as_absent(self):
        panel = build_event_context(
            "NVDA", "2026-09-22", event=EVENT, analogs=analog_summary([], "20d")
        )
        line = [l for l in render_context(panel) if "analogs" in l][0]
        self.assertIn("median —", line)
        self.assertNotIn("0.00%", line)

    def test_lines_stay_readable(self):
        panel = build_event_context("NVDA", "2026-09-22", event=EVENT, analogs=summary())
        for line in render_context(panel):
            self.assertLess(len(line), 200)


if __name__ == "__main__":
    unittest.main()
