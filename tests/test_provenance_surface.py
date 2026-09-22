"""D4 provenance surface tests.

The central claim under test: a value with no provenance must be shown AS
untraced, never shown with the question unasked. These tests build their view
through the real D1 assembly and their raw records in-process, so they mean the
same thing on a clean clone.
"""

from __future__ import annotations

import unittest

from core.config import (
    PROVENANCE_BLOCKS_TRADES,
    PROVENANCE_COVERAGE_WARN,
    PROVENANCE_REPORT_UNTRACED,
    PROVENANCE_REQUIRED_FIELDS,
    PROVENANCE_SURFACE_VERSION,
    PROVENANCE_TRACE_REFUSALS,
    PROV_FIELD_PAYLOAD,
    PROV_FIELD_SOURCE,
    PROV_ORIGIN_COMPOSED,
    PROV_ORIGIN_DERIVED,
    PROV_ORIGIN_OBSERVED,
    PROV_ORIGIN_UNTRACED,
    RESEARCH_VIEW_HORIZONS,
    VIEW_CELL_PRESENT,
)
from core.forecast_snapshot import build_forecast_snapshot
from core.provenance_surface import (
    ProvenanceError,
    build_provenance_surface,
    from_raw_record,
    provenance_problems,
    render_provenance,
    trace,
    untraced,
)
from core.research_view import build_research_view

RAW = {
    "source_id": "yahoo_finance_chart",
    "request_key": "NVDA_5y_1d",
    "payload_sha256": "4b31729a9bc1703f2a730b880f75f5712023f13a68629c6d062b686804add6b3",
    "ingested_time": "2026-09-22T06:22:00.728733+00:00",
    "schema_version": "raw-store-v1",
}


def view():
    snapshots = {
        h: build_forecast_snapshot("NVDA", "2026-09-22", h)
        for h in RESEARCH_VIEW_HORIZONS
    }
    return build_research_view(
        "NVDA",
        "2026-09-22",
        snapshots,
        quality={"score": 78.5, "action": "BUY"},
        risk={"veto": False, "veto_rule_ids": []},
    )


class TheDecidingMeasurementTest(unittest.TestCase):
    """Displayed values carry no provenance today, and the surface says so."""

    def test_a_live_snapshot_carries_no_per_value_provenance(self):
        snapshot = build_forecast_snapshot("NVDA", "2026-09-22", "20d")
        for name, field in snapshot["fields"].items():
            if field.get("status") != VIEW_CELL_PRESENT:
                continue
            identifying = set(field) - {"status", "value", "reason"}
            self.assertFalse(
                identifying,
                f"{name} now carries {sorted(identifying)}; D4 exists because "
                f"PRESENT fields carried nothing identifying",
            )

    def test_an_untraced_view_reports_zero_coverage(self):
        surface = build_provenance_surface(view())
        self.assertEqual(surface["coverage"]["traced"], 0)
        self.assertEqual(surface["coverage"]["share_traced"], 0.0)
        self.assertTrue(surface["coverage"]["below_threshold"])

    def test_untraced_values_are_reported_not_omitted(self):
        surface = build_provenance_surface(view())
        self.assertTrue(PROVENANCE_REPORT_UNTRACED)
        self.assertGreater(
            len(surface["entries"]),
            0,
            "the surface omitted everything it could not trace, which would "
            "report perfect provenance over an empty set",
        )
        for entry in surface["entries"]:
            self.assertFalse(entry["traced"])
            self.assertEqual(
                entry["provenance"]["origin"], PROV_ORIGIN_UNTRACED
            )

    def test_the_headline_names_the_gap(self):
        surface = build_provenance_surface(view())
        self.assertIn("cannot", surface["headline"])
        self.assertIn("without a citation", surface["headline"])

    def test_every_untraced_entry_explains_itself(self):
        surface = build_provenance_surface(view())
        for entry in surface["entries"]:
            self.assertTrue(
                entry["provenance"]["reason"].strip(),
                f"{entry['path']}: untraced with no reason",
            )


class TracingTest(unittest.TestCase):
    """When provenance exists, it reaches the surface intact."""

    def test_a_raw_record_becomes_an_observed_trace(self):
        record = from_raw_record(RAW)
        self.assertEqual(record["origin"], PROV_ORIGIN_OBSERVED)
        self.assertEqual(record[PROV_FIELD_SOURCE], RAW["source_id"])
        self.assertEqual(record[PROV_FIELD_PAYLOAD], RAW["payload_sha256"])

    def test_a_traced_view_reports_full_coverage(self):
        surface = build_provenance_surface(
            view(),
            {
                "panels.investment_quality.score": from_raw_record(
                    RAW, origin=PROV_ORIGIN_DERIVED
                ),
                "panels.investment_quality.action": trace(
                    PROV_ORIGIN_COMPOSED,
                    inputs=["panels.investment_quality.score"],
                    calculation_version="score-v1",
                ),
                "panels.risk_status.veto": from_raw_record(RAW),
            },
        )
        self.assertEqual(surface["coverage"]["share_traced"], 1.0)
        self.assertFalse(surface["coverage"]["below_threshold"])
        self.assertEqual(provenance_problems(surface), [])

    def test_the_three_origins_stay_distinct(self):
        surface = build_provenance_surface(
            view(),
            {
                "panels.investment_quality.score": from_raw_record(
                    RAW, origin=PROV_ORIGIN_DERIVED
                ),
                "panels.investment_quality.action": trace(
                    PROV_ORIGIN_COMPOSED,
                    inputs=["panels.investment_quality.score"],
                ),
                "panels.risk_status.veto": from_raw_record(RAW),
            },
        )
        by_origin = surface["coverage"]["by_origin"]
        self.assertEqual(by_origin[PROV_ORIGIN_OBSERVED], 1)
        self.assertEqual(by_origin[PROV_ORIGIN_DERIVED], 1)
        self.assertEqual(by_origin[PROV_ORIGIN_COMPOSED], 1)

    def test_an_observed_value_must_be_rederivable(self):
        for missing in PROVENANCE_REQUIRED_FIELDS:
            kwargs = {
                "source_id": RAW["source_id"],
                "payload_sha256": RAW["payload_sha256"],
                "ingested_time": RAW["ingested_time"],
                "calculation_version": RAW["schema_version"],
            }
            kwargs[missing] = None
            with self.assertRaises(
                ProvenanceError, msg=f"{missing} was allowed to be absent"
            ):
                trace(PROV_ORIGIN_OBSERVED, **kwargs)

    def test_a_composed_value_must_name_its_inputs(self):
        with self.assertRaises(ProvenanceError):
            trace(PROV_ORIGIN_COMPOSED, inputs=[])

    def test_an_untraced_value_must_carry_a_reason(self):
        with self.assertRaises(ProvenanceError):
            trace(PROV_ORIGIN_UNTRACED, reason="  ")

    def test_an_untraced_value_carries_no_identifiers(self):
        record = untraced("nothing recorded this")
        for field in PROVENANCE_REQUIRED_FIELDS:
            self.assertIsNone(record[field])

    def test_an_unknown_origin_is_refused(self):
        with self.assertRaises(ProvenanceError):
            trace("PROBABLY_FINE", reason="x")

    def test_a_non_mapping_raw_record_is_refused(self):
        with self.assertRaises(ProvenanceError):
            from_raw_record("yahoo")


class RefusalsAreTracedTest(unittest.TestCase):
    """Why a value is absent is itself evidence."""

    def test_refusals_appear_in_the_surface(self):
        surface = build_provenance_surface(view())
        refused = [e for e in surface["entries"] if e["state"] != VIEW_CELL_PRESENT]
        self.assertTrue(
            refused,
            "no refused cells reached the surface; a reader cannot see why a "
            "value is missing",
        )

    def test_the_config_requires_tracing_refusals(self):
        self.assertTrue(PROVENANCE_TRACE_REFUSALS)
        surface = build_provenance_surface(view())
        self.assertTrue(surface["traces_refusals"])

    def test_coverage_counts_only_displayed_values(self):
        surface = build_provenance_surface(view())
        coverage = surface["coverage"]
        self.assertLess(
            coverage["displayed"],
            coverage["entries"],
            "every entry was counted as displayed; refusals are traced but "
            "are not displayed values",
        )


class ContractTest(unittest.TestCase):
    def test_a_clean_surface_has_no_problems(self):
        surface = build_provenance_surface(view())
        self.assertEqual(provenance_problems(surface), [])
        self.assertEqual(surface["version"], PROVENANCE_SURFACE_VERSION)

    def test_the_surface_does_not_block_trades(self):
        surface = build_provenance_surface(view())
        self.assertFalse(surface["blocks_trades"])
        self.assertFalse(PROVENANCE_BLOCKS_TRADES)

    def test_a_surface_that_blocks_is_caught(self):
        surface = build_provenance_surface(view())
        surface["blocks_trades"] = True
        self.assertTrue(provenance_problems(surface))

    def test_an_untraced_entry_marked_traced_is_caught(self):
        surface = build_provenance_surface(view())
        surface["entries"][0]["traced"] = True
        self.assertTrue(provenance_problems(surface))

    def test_an_untraced_entry_carrying_a_digest_is_caught(self):
        surface = build_provenance_surface(view())
        surface["entries"][0]["provenance"][PROV_FIELD_PAYLOAD] = "deadbeef"
        self.assertTrue(
            provenance_problems(surface),
            "an UNTRACED record carrying a payload digest passed — that "
            "claims a traceability nobody established",
        )

    def test_an_untraced_entry_with_no_reason_is_caught(self):
        surface = build_provenance_surface(view())
        surface["entries"][0]["provenance"]["reason"] = "  "
        self.assertTrue(provenance_problems(surface))

    def test_a_surface_that_skips_refusals_is_caught(self):
        surface = build_provenance_surface(view())
        surface["traces_refusals"] = False
        self.assertTrue(provenance_problems(surface))

    def test_a_miscounted_coverage_is_caught(self):
        surface = build_provenance_surface(view())
        surface["coverage"]["traced"] = 99
        self.assertTrue(provenance_problems(surface))

    def test_a_silenced_headline_is_caught(self):
        surface = build_provenance_surface(view())
        surface["headline"] = ""
        self.assertTrue(provenance_problems(surface))

    def test_a_duplicated_path_is_caught(self):
        surface = build_provenance_surface(view())
        surface["entries"].append(dict(surface["entries"][0]))
        self.assertTrue(provenance_problems(surface))

    def test_a_pathless_entry_is_caught(self):
        surface = build_provenance_surface(view())
        surface["entries"][0]["path"] = ""
        self.assertTrue(provenance_problems(surface))

    def test_a_non_mapping_view_is_refused(self):
        with self.assertRaises(ProvenanceError):
            build_provenance_surface(["a view"])

    def test_non_mapping_records_are_refused(self):
        with self.assertRaises(ProvenanceError):
            build_provenance_surface(view(), ["a record"])

    def test_a_non_mapping_record_entry_is_refused(self):
        with self.assertRaises(ProvenanceError):
            build_provenance_surface(
                view(), {"panels.investment_quality.score": "yahoo"}
            )


class RenderTest(unittest.TestCase):
    def test_the_headline_comes_first(self):
        surface = build_provenance_surface(view())
        self.assertEqual(render_provenance(surface)[0].strip(), surface["headline"])

    def test_untraced_reaches_the_render(self):
        surface = build_provenance_surface(view())
        self.assertIn(PROV_ORIGIN_UNTRACED, "\n".join(render_provenance(surface)))

    def test_a_digest_reaches_the_render(self):
        surface = build_provenance_surface(
            view(), {"panels.investment_quality.score": from_raw_record(RAW)}
        )
        self.assertIn(RAW["payload_sha256"][:12], "\n".join(render_provenance(surface)))

    def test_lines_stay_readable(self):
        surface = build_provenance_surface(view())
        for line in render_provenance(surface):
            self.assertLess(len(line), 200)


if __name__ == "__main__":
    unittest.main()
