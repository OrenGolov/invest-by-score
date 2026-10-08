"""F8 ForecastSnapshot contract tests.

The behaviour under test is dependability: every declared field is always a
key, absence is explicit, and an absent field never carries a placeholder a
consumer would render.
"""

from __future__ import annotations

import unittest

from core.config import (
    SNAPSHOT_CONTRIBUTIONS_NOTE,
    SNAPSHOT_FIELDS,
    SNAPSHOT_LOW_CONFIDENCE_THRESHOLD,
    SNAPSHOT_REQUIRED_FIELDS,
    SNAPSHOT_STATUS_ABSENT,
    SNAPSHOT_STATUS_PRESENT,
    SNAPSHOT_STATUS_REFUSED,
    SNAPSHOT_UNAVAILABLE_FIELDS,
    SNAPSHOT_WARNING_ABSENT_FIELDS,
    SNAPSHOT_WARNING_INFERRED_EVIDENCE,
    SNAPSHOT_WARNING_LOW_CONFIDENCE,
    SNAPSHOT_WARNING_NO_MODEL,
    SNAPSHOT_WARNING_THIN_SAMPLE,
)
from core.forecast_snapshot import (
    ForecastSnapshotError,
    _field,
    build_forecast_snapshot,
    forecast_versions,
    render_snapshot,
    snapshot_digest,
    snapshot_problems,
    snapshot_value,
)


def _snapshot(**overrides):
    payload = dict(
        ticker="NVDA", as_of="2026-09-20", horizon="20d",
        chart_state={"rsi": 55.0, "volatility": 0.22, "change_20d": 0.05},
        confidence={
            "confidence": 0.42, "binding_factor": "sample_size",
            "assessed_object": "forecast",
        },
        decomposition={"decomposed_object": "forecast", "additive": False},
        regime="bullish",
        evidence={"memory_ids": ["m1", "m2"]},
    )
    payload.update(overrides)
    return build_forecast_snapshot(
        payload.pop("ticker"), payload.pop("as_of"), payload.pop("horizon"),
        **payload,
    )


class ContractShapeTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = _snapshot()

    def test_every_field_is_present_as_a_key(self):
        self.assertEqual(list(self.snapshot["fields"]), list(SNAPSHOT_FIELDS))

    def test_a_bare_snapshot_still_carries_every_field(self):
        bare = build_forecast_snapshot("NVDA", "2026-09-20", "20d")
        self.assertEqual(list(bare["fields"]), list(SNAPSHOT_FIELDS))
        self.assertEqual(snapshot_problems(bare), [])

    def test_the_contract_declares_fifteen_fields(self):
        self.assertEqual(len(SNAPSHOT_FIELDS), 15)

    def test_required_fields_are_enforced(self):
        with self.assertRaises(ForecastSnapshotError):
            build_forecast_snapshot("", "2026-09-20", "20d")
        with self.assertRaises(ForecastSnapshotError):
            build_forecast_snapshot("NVDA", "2026-09-20", "")

    def test_required_fields_are_present(self):
        for name in SNAPSHOT_REQUIRED_FIELDS:
            self.assertEqual(
                self.snapshot["fields"][name]["status"],
                SNAPSHOT_STATUS_PRESENT,
                name,
            )

    def test_a_healthy_snapshot_raises_nothing(self):
        self.assertEqual(snapshot_problems(self.snapshot), [])


class AbsentIsNotZeroTests(unittest.TestCase):
    """A placeholder on an absent field becomes a number a consumer renders."""

    def setUp(self):
        self.snapshot = _snapshot()

    def test_the_model_dependent_fields_are_absent(self):
        for name in ("model_versions", "expected_return"):
            self.assertEqual(
                self.snapshot["fields"][name]["status"], SNAPSHOT_STATUS_ABSENT, name
            )

    def test_absent_fields_carry_no_value(self):
        for name in self.snapshot["absent"]:
            self.assertNotIn("value", self.snapshot["fields"][name], name)

    def test_refused_fields_carry_no_value(self):
        for name in self.snapshot["refused"]:
            self.assertNotIn("value", self.snapshot["fields"][name], name)

    def test_every_gap_explains_itself(self):
        for name in self.snapshot["absent"] + self.snapshot["refused"]:
            self.assertTrue(self.snapshot["fields"][name]["reason"], name)

    def test_a_blocked_status_discards_a_smuggled_value(self):
        for blocked in (SNAPSHOT_STATUS_ABSENT, SNAPSHOT_STATUS_REFUSED):
            row = _field(blocked, value=0.0, reason="probe", samples=99)
            self.assertNotIn("value", row, blocked)
            self.assertNotIn("samples", row, blocked)

    def test_a_present_status_keeps_its_value_and_detail(self):
        row = _field(SNAPSHOT_STATUS_PRESENT, value=0.42, reason="r", samples=10)
        self.assertEqual(row["value"], 0.42)
        self.assertEqual(row["samples"], 10)

    def test_every_unavailable_field_states_a_reason(self):
        for name, reason in SNAPSHOT_UNAVAILABLE_FIELDS.items():
            self.assertTrue(reason, name)
            self.assertIn(name, SNAPSHOT_FIELDS)

    def test_the_accessor_returns_none_for_a_gap(self):
        # A consumer must not crash on an honest absence.
        self.assertIsNone(snapshot_value(self.snapshot, "expected_return"))
        self.assertIsNone(snapshot_value(self.snapshot, "model_versions"))

    def test_the_accessor_returns_a_present_value(self):
        self.assertEqual(snapshot_value(self.snapshot, "ticker"), "NVDA")


class ContributionsNamingTests(unittest.TestCase):
    """The field name conflicts with a measurement F6 already made."""

    def test_the_note_denies_the_literal_reading(self):
        for phrase in ("NOT contributions", "do NOT sum", "MEASURED"):
            self.assertIn(phrase, SNAPSHOT_CONTRIBUTIONS_NOTE, phrase)

    def test_the_field_declares_itself_non_additive(self):
        row = _snapshot()["fields"]["model_contributions"]
        self.assertEqual(row["status"], SNAPSHOT_STATUS_PRESENT)
        self.assertIs(row["additive"], False)

    def test_the_field_carries_f6_unchanged(self):
        decomposition = {"decomposed_object": "forecast", "additive": False, "x": 1}
        row = _snapshot(decomposition=decomposition)["fields"]["model_contributions"]
        self.assertEqual(row["value"], decomposition)

    def test_a_missing_decomposition_is_refused_not_faked(self):
        row = _snapshot(decomposition=None)["fields"]["model_contributions"]
        self.assertEqual(row["status"], SNAPSHOT_STATUS_REFUSED)
        self.assertNotIn("value", row)


class DigestTests(unittest.TestCase):
    def test_the_digest_matches_the_content(self):
        snapshot = _snapshot()
        self.assertEqual(snapshot_digest(snapshot), snapshot["digest"])

    def test_identical_requests_digest_identically(self):
        self.assertEqual(_snapshot()["digest"], _snapshot()["digest"])

    def test_a_changed_input_changes_the_digest(self):
        self.assertNotEqual(
            _snapshot(regime="bearish")["digest"], _snapshot()["digest"]
        )

    def test_a_tampered_snapshot_is_reported(self):
        snapshot = _snapshot()
        snapshot["fields"]["ticker"]["value"] = "TSLA"
        self.assertTrue(
            any("does not match" in p for p in snapshot_problems(snapshot))
        )


class FeatureDigestTests(unittest.TestCase):
    def test_it_composes_m1s_digest(self):
        from core.feature_registry import (
            feature_surface_digest,
            load_feature_registry,
        )

        state = {"rsi": 55.0, "volatility": 0.22, "change_20d": 0.05}
        expected = feature_surface_digest(
            {name: {"value": value} for name, value in sorted(state.items())},
            load_feature_registry(),
        )
        self.assertEqual(
            snapshot_value(_snapshot(chart_state=state), "feature_digest"), expected
        )

    def test_it_changes_with_the_surface(self):
        first = snapshot_value(
            _snapshot(chart_state={"rsi": 55.0}), "feature_digest"
        )
        second = snapshot_value(
            _snapshot(chart_state={"rsi": 70.0}), "feature_digest"
        )
        self.assertNotEqual(first, second)

    def test_no_surface_is_refused_not_hashed(self):
        row = _snapshot(chart_state=None)["fields"]["feature_digest"]
        self.assertEqual(row["status"], SNAPSHOT_STATUS_REFUSED)
        self.assertNotIn("value", row)


class WarningTests(unittest.TestCase):
    """A headline-only consumer must still see the caveats."""

    def test_absent_fields_raise_a_top_level_warning(self):
        snapshot = _snapshot()
        self.assertTrue(snapshot["absent"])
        self.assertIn(SNAPSHOT_WARNING_ABSENT_FIELDS, snapshot["warning_codes"])

    def test_the_no_model_warning_is_surfaced(self):
        self.assertIn(SNAPSHOT_WARNING_NO_MODEL, _snapshot()["warning_codes"])

    def test_low_confidence_warns(self):
        low = _snapshot(confidence={"confidence": 0.10, "binding_factor": "x"})
        self.assertIn(SNAPSHOT_WARNING_LOW_CONFIDENCE, low["warning_codes"])

    def test_adequate_confidence_does_not_warn(self):
        high = _snapshot(
            confidence={
                "confidence": SNAPSHOT_LOW_CONFIDENCE_THRESHOLD + 0.2,
                "binding_factor": "x",
            }
        )
        self.assertNotIn(SNAPSHOT_WARNING_LOW_CONFIDENCE, high["warning_codes"])

    def test_inferred_evidence_warns(self):
        snapshot = _snapshot(
            event_forecast={
                "stages": {"matches": {"observed_share": 0.1}}, "samples": 30,
            }
        )
        self.assertIn(SNAPSHOT_WARNING_INFERRED_EVIDENCE, snapshot["warning_codes"])

    def test_a_thin_sample_warns(self):
        snapshot = _snapshot(event_forecast={"samples": 3, "stages": {}})
        self.assertIn(SNAPSHOT_WARNING_THIN_SAMPLE, snapshot["warning_codes"])

    def test_a_healthy_sample_does_not_warn_thin(self):
        snapshot = _snapshot(event_forecast={"samples": 50, "stages": {}})
        self.assertNotIn(SNAPSHOT_WARNING_THIN_SAMPLE, snapshot["warning_codes"])


class VersionTests(unittest.TestCase):
    def test_every_forecast_sprint_version_is_carried(self):
        versions = forecast_versions()
        for part in ("snapshot", "contract", "joint", "conditional", "event",
                     "decomposition", "confidence", "targets"):
            self.assertTrue(versions.get(part), part)

    def test_the_versions_are_published_on_the_snapshot(self):
        self.assertEqual(
            snapshot_value(_snapshot(), "forecast_version"), forecast_versions()
        )


class RenderTests(unittest.TestCase):
    def test_render_covers_every_field_in_order(self):
        rows = render_snapshot(_snapshot())
        self.assertEqual([r["field"] for r in rows], list(SNAPSHOT_FIELDS))

    def test_render_never_leaves_a_gap_blank(self):
        for row in render_snapshot(build_forecast_snapshot("NVDA", "x", "20d")):
            if row["status"] != SNAPSHOT_STATUS_PRESENT:
                self.assertTrue(row["reason"], row["field"])

    def test_render_reports_whether_a_value_exists(self):
        rows = {r["field"]: r for r in render_snapshot(_snapshot())}
        self.assertTrue(rows["ticker"]["has_value"])
        self.assertFalse(rows["expected_return"]["has_value"])


class ContractProblemTests(unittest.TestCase):
    def _mutated(self, mutate):
        snapshot = _snapshot()
        mutate(snapshot)
        return snapshot_problems(snapshot)

    def test_a_value_on_an_absent_field_is_reported(self):
        problems = self._mutated(
            lambda s: s["fields"]["expected_return"].update({"value": 0.0})
        )
        self.assertTrue(any("nobody computed" in p for p in problems))

    def test_a_missing_field_is_reported(self):
        problems = self._mutated(lambda s: s["fields"].pop("regime"))
        self.assertTrue(any("every contract field" in p for p in problems))

    def test_a_present_unavailable_field_is_reported(self):
        def mutate(snapshot):
            snapshot["fields"]["model_versions"] = {
                "status": SNAPSHOT_STATUS_PRESENT, "value": {}, "reason": "",
            }

        self.assertTrue(
            any("declared unavailable" in p for p in self._mutated(mutate))
        )

    def test_a_missing_required_field_is_reported(self):
        def mutate(snapshot):
            snapshot["fields"]["ticker"] = {
                "status": SNAPSHOT_STATUS_REFUSED, "reason": "gone",
            }

        self.assertTrue(any("is required" in p for p in self._mutated(mutate)))

    def test_an_additive_contributions_claim_is_reported(self):
        problems = self._mutated(
            lambda s: s["fields"]["model_contributions"].update({"additive": True})
        )
        self.assertTrue(any("non-additive" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
