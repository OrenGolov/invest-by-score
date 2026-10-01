"""A8 priority/action tests — the five operator bands, and what must not happen.

Two properties carry the most weight here, and both were found by measurement
rather than by reading the code:

1. EVERY BAND IS REACHABLE. A band no detector can produce is a dead legend entry
   and a filter that always returns nothing. Enumerating the six detector
   vocabularies against both severities proves each of the five is reachable.
2. A FIRED STATE WITH NO SEVERITY IS NOT GRADED. The same enumeration caught
   `DISAPPEARED/none` reaching Urgent and `LOW_IMPACT/none` reaching Low, because
   the availability and impact rules did not look at severity. No shipped
   detector emits those pairs, so it was not a live defect — it was fail-OPEN,
   and the next detector to omit severity would silently have been handed the top
   band.
"""

from __future__ import annotations

import unittest

from core.alert_priority import (
    AlertPriorityError,
    directional_action_permitted,
    grade,
    priority_for,
    priority_rank,
    recommended_action,
    state_of,
)
from core.config import (
    ALERT_ACTION_REVIEW,
    ALERT_ACTION_WATCH,
    ALERT_DIRECTIONAL_ACTIONS,
    ALERT_PRIORITIES,
    ALERT_PRIORITY_HIGH,
    ALERT_PRIORITY_LOW,
    ALERT_PRIORITY_MEDIUM,
    ALERT_PRIORITY_URGENT,
    ALERT_PRIORITY_VERY_HIGH,
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    CONFIDENCE_CHANGE_KINDS,
    EVENT_IMPACT_VERDICTS,
    FORECAST_CHANGE_KINDS,
    FTHRESHOLD_VERDICTS,
    REGIME_CHANGE_KINDS,
    THESIS_BREAK_KINDS,
)

VOCABULARIES = {
    "A1_forecast": FORECAST_CHANGE_KINDS,
    "A2_confidence": CONFIDENCE_CHANGE_KINDS,
    "A3_impact": EVENT_IMPACT_VERDICTS,
    "A4_regime": REGIME_CHANGE_KINDS,
    "A5_thesis": THESIS_BREAK_KINDS,
    "A6_threshold": FTHRESHOLD_VERDICTS,
}


def alert(state, severity=None, **extra):
    return {"kind": state, "severity": severity, **extra}


class StateReadingTests(unittest.TestCase):
    def test_kind_and_verdict_are_both_read(self):
        self.assertEqual(state_of({"kind": "MOVED"}), "MOVED")
        self.assertEqual(state_of({"verdict": "HIGH_IMPACT"}), "HIGH_IMPACT")

    def test_an_alert_with_neither_field_raises(self):
        # Grading it anyway would assign a band on no evidence.
        with self.assertRaises(AlertPriorityError):
            state_of({"ticker": "NVDA"})

    def test_a_non_string_state_raises(self):
        with self.assertRaises(AlertPriorityError):
            state_of({"kind": 7})

    def test_a_non_mapping_raises(self):
        with self.assertRaises(AlertPriorityError):
            state_of(["MOVED"])


class EveryBandIsReachableTests(unittest.TestCase):
    """A band nothing can produce is a filter that always returns nothing."""

    @classmethod
    def setUpClass(cls):
        cls.reached: dict[str | None, list[str]] = {p: [] for p in ALERT_PRIORITIES}
        cls.reached[None] = []
        for module, states in VOCABULARIES.items():
            for state in states:
                for severity in (ALERT_SEVERITY_WARN, ALERT_SEVERITY_INFO, None):
                    band, _ = priority_for(alert(state, severity))
                    cls.reached[band].append(f"{module}/{state}/{severity}")

    def test_no_band_is_dead(self):
        for band in ALERT_PRIORITIES:
            with self.subTest(band=band):
                self.assertTrue(
                    self.reached[band],
                    f"{band} is unreachable from every detector vocabulary, so "
                    f"the operator sees a legend entry that never appears",
                )

    def test_some_states_are_deliberately_ungraded(self):
        # Quiet and unevaluated states must NOT be graded; if this list empties,
        # the grader has started inventing bands for non-findings.
        self.assertTrue(self.reached[None])


class AFiredStateWithoutSeverityIsNotGradedTests(unittest.TestCase):
    """The fail-open shape the enumeration caught."""

    def test_disappeared_without_severity_is_not_urgent(self):
        band, reason = priority_for(alert("DISAPPEARED", None))
        self.assertIsNone(
            band,
            "a severity-less DISAPPEARED reached Urgent before this guard; the "
            "next detector to omit severity would have been handed the top band",
        )
        self.assertIn("no severity", reason)

    def test_low_impact_without_severity_is_not_low(self):
        band, _ = priority_for({"verdict": "LOW_IMPACT", "severity": None})
        self.assertIsNone(band)

    def test_the_same_states_DO_grade_once_severity_is_present(self):
        # The guard must not have broken the ordinary path.
        self.assertEqual(
            priority_for(alert("DISAPPEARED", ALERT_SEVERITY_WARN))[0],
            ALERT_PRIORITY_URGENT,
        )
        self.assertEqual(
            priority_for({"verdict": "LOW_IMPACT", "severity": ALERT_SEVERITY_INFO})[0],
            ALERT_PRIORITY_LOW,
        )

    def test_pending_is_the_one_deliberate_exception(self):
        # A4 reports PENDING with no severity BY DESIGN: an unconfirmed flip has
        # no adverseness yet. Its band comes from the confirmation measurement.
        band, reason = priority_for(alert("PENDING", None))
        self.assertEqual(band, ALERT_PRIORITY_MEDIUM)
        self.assertIn("26%", reason)


class PriorityOrderingTests(unittest.TestCase):
    def test_availability_loss_outranks_any_magnitude_move(self):
        # MEASURED by A1: folding availability into magnitude reports a -0.56
        # crash that never happened. A vanished measurement is a pipeline fact.
        gone = priority_rank(priority_for(alert("DISAPPEARED", ALERT_SEVERITY_WARN))[0])
        moved = priority_rank(priority_for(alert("MOVED", ALERT_SEVERITY_WARN))[0])
        self.assertGreater(gone, moved)

    def test_an_unconfirmed_change_ranks_below_a_confirmed_one(self):
        # A4 MEASURED 26% of single-session runs reversing the next day.
        pending = priority_rank(priority_for(alert("PENDING", None))[0])
        confirmed = priority_rank(
            priority_for(alert("CONFIRMED", ALERT_SEVERITY_WARN))[0]
        )
        self.assertGreater(confirmed, pending)

    def test_warn_outranks_info_for_the_same_state(self):
        warn = priority_rank(priority_for(alert("CONFIRMED", ALERT_SEVERITY_WARN))[0])
        info = priority_rank(priority_for(alert("CONFIRMED", ALERT_SEVERITY_INFO))[0])
        self.assertGreater(warn, info)

    def test_high_impact_outranks_low_impact(self):
        high = priority_rank(
            priority_for({"verdict": "HIGH_IMPACT", "severity": ALERT_SEVERITY_WARN})[0]
        )
        low = priority_rank(
            priority_for({"verdict": "LOW_IMPACT", "severity": ALERT_SEVERITY_INFO})[0]
        )
        self.assertGreater(high, low)

    def test_an_ungraded_alert_has_no_rank(self):
        # None is a real answer: inventing 0 would order it below Low rather
        # than as incomparable.
        self.assertIsNone(priority_rank(None))

    def test_an_unknown_priority_cannot_be_ranked(self):
        with self.assertRaises(AlertPriorityError):
            priority_rank("Catastrophic")


class UnevaluatedIsNotLowTests(unittest.TestCase):
    """Absence of evidence is not evidence of safety — the project's third state."""

    def test_not_evaluated_carries_no_band(self):
        band, reason = priority_for(alert("NOT_EVALUATED", None))
        self.assertIsNone(band)
        self.assertIn("not a verdict", reason)

    def test_not_evaluated_is_not_the_lowest_band(self):
        # The distinction that matters: Low means "measured and small";
        # ungraded means "we could not measure it".
        self.assertNotEqual(priority_for(alert("NOT_EVALUATED", None))[0],
                            ALERT_PRIORITY_LOW)

    def test_a_quiet_alert_carries_no_band(self):
        for state in ("NONE", "NOT_MET", "NO_EVENT"):
            with self.subTest(state=state):
                self.assertIsNone(priority_for(alert(state, None))[0])

    def test_an_unknown_severity_raises(self):
        with self.assertRaises(AlertPriorityError):
            priority_for(alert("MOVED", "catastrophic"))


class NoAlertEverRecommendsADirectionTests(unittest.TestCase):
    """The governance constraint, stated as a test rather than a comment."""

    def test_no_state_in_any_vocabulary_produces_buy_or_sell(self):
        for module, states in VOCABULARIES.items():
            for state in states:
                for severity in (ALERT_SEVERITY_WARN, ALERT_SEVERITY_INFO, None):
                    with self.subTest(module=module, state=state, sev=severity):
                        result = grade(alert(state, severity))
                        self.assertNotIn(result["action"], ALERT_DIRECTIONAL_ACTIONS)

    def test_a_graded_market_change_is_watched_not_traded(self):
        result = grade(alert("CONFIRMED", ALERT_SEVERITY_WARN))
        self.assertEqual(result["action"], ALERT_ACTION_WATCH)

    def test_an_ungraded_alert_is_reviewed_by_a_person(self):
        result = grade(alert("NOT_EVALUATED", None))
        self.assertEqual(result["action"], ALERT_ACTION_REVIEW)

    def test_an_availability_change_is_reviewed_not_watched(self):
        # The evidence base changed, which is a pipeline question rather than a
        # position question, however severe it looks.
        result = grade(alert("DISAPPEARED", ALERT_SEVERITY_WARN))
        self.assertEqual(result["priority"], ALERT_PRIORITY_URGENT)
        self.assertEqual(result["action"], ALERT_ACTION_REVIEW)

    def test_an_unknown_approval_state_is_not_an_approval(self):
        permitted, reason = directional_action_permitted(
            governance_clear=True, release_approved=None
        )
        self.assertFalse(permitted)
        self.assertIn("unknown", reason)

    def test_an_unknown_governance_state_is_not_clear(self):
        permitted, reason = directional_action_permitted(
            governance_clear=None, release_approved=True
        )
        self.assertFalse(permitted)
        self.assertIn("unknown", reason)

    def test_a_refused_release_is_named_with_its_measurement(self):
        _, reason = directional_action_permitted(
            governance_clear=True, release_approved=False
        )
        self.assertIn("NOT APPROVED", reason)

    def test_even_a_cleared_release_does_not_make_the_grader_directional(self):
        # The direction must come from a promoted forecast, not from the grader.
        action, _ = recommended_action(
            alert("CONFIRMED", ALERT_SEVERITY_WARN),
            governance_clear=True,
            release_approved=True,
        )
        self.assertNotIn(action, ALERT_DIRECTIONAL_ACTIONS)


class GradeShapeTests(unittest.TestCase):
    def test_grade_reports_the_reasoning_for_both_decisions(self):
        result = grade(alert("CONFIRMED", ALERT_SEVERITY_WARN))
        for field in (
            "priority",
            "priority_rank",
            "priority_reason",
            "action",
            "action_reason",
            "state",
            "severity",
        ):
            self.assertIn(field, result)

    def test_the_reason_carries_the_measurement_not_just_a_verdict(self):
        _, reason = priority_for(alert("PENDING", None))
        self.assertIn("26%", reason)
        _, reason = priority_for({"verdict": "HIGH_IMPACT", "severity": "warn"})
        self.assertIn("12x", reason)

    def test_every_graded_band_is_a_known_band(self):
        for state in ("CONFIRMED", "MOVED", "APPEARED", "DISAPPEARED"):
            for severity in (ALERT_SEVERITY_WARN, ALERT_SEVERITY_INFO):
                band = grade(alert(state, severity))["priority"]
                with self.subTest(state=state, severity=severity):
                    self.assertIn(band, ALERT_PRIORITIES)


class RealDetectorOutputGradesTests(unittest.TestCase):
    """Grading records built by the SHIPPED builders, not by hand.

    A grader tested only on dicts written in the test file proves self-consistency
    and nothing else.
    """

    @staticmethod
    def snapshot(ticker, as_of, horizon, probability):
        fields = {
            "ticker": {"value": ticker, "status": "PRESENT"},
            "as_of": {"value": as_of, "status": "PRESENT"},
            "horizon": {"value": horizon, "status": "PRESENT"},
        }
        if probability is None:
            fields["probability_up"] = {"value": None, "status": "ABSENT"}
        else:
            fields["probability_up"] = {"value": probability, "status": "PRESENT"}
        return {"fields": fields}

    def test_a1_moved_grades_very_high(self):
        from core.forecast_alert import forecast_change_alert

        built = forecast_change_alert(
            self.snapshot("NVDA", "2026-09-30", "20d", 0.40),
            self.snapshot("NVDA", "2026-10-01", "20d", 0.62),
        )
        self.assertEqual(grade(built)["priority"], ALERT_PRIORITY_VERY_HIGH)

    def test_a1_disappeared_grades_urgent(self):
        from core.forecast_alert import forecast_change_alert

        built = forecast_change_alert(
            self.snapshot("NVDA", "2026-09-30", "20d", 0.56),
            self.snapshot("NVDA", "2026-10-01", "20d", None),
        )
        self.assertEqual(grade(built)["priority"], ALERT_PRIORITY_URGENT)

    def test_a1_appeared_grades_high(self):
        from core.forecast_alert import forecast_change_alert

        built = forecast_change_alert(
            self.snapshot("NVDA", "2026-09-30", "20d", None),
            self.snapshot("NVDA", "2026-10-01", "20d", 0.56),
        )
        self.assertEqual(grade(built)["priority"], ALERT_PRIORITY_HIGH)

    def test_a4_confirmed_grades_very_high_and_pending_grades_medium(self):
        from core.regime_alert import regime_change_alert

        def classification(label, recent=None):
            out = {"ticker": "MSFT", "label": label, "computable": True}
            if recent is not None:
                out["recent_labels"] = recent
            return out

        confirmed = regime_change_alert(
            classification("bullish"),
            classification("risk_off", recent=["risk_off"] * 5),
        )
        pending = regime_change_alert(
            classification("bullish"), classification("risk_off")
        )
        self.assertEqual(grade(confirmed)["priority"], ALERT_PRIORITY_VERY_HIGH)
        self.assertEqual(grade(pending)["priority"], ALERT_PRIORITY_MEDIUM)

    def test_a5_reversal_grades_and_is_not_directional(self):
        from core.thesis_alert import thesis_break_alert

        def attribution(operational, narrative, macro):
            return {
                "ticker": "AVGO",
                "buckets": {
                    "operational": {"total": operational},
                    "narrative": {"total": narrative},
                    "macro_shock": {"total": macro},
                },
            }

        built = thesis_break_alert(
            attribution(1.4, -0.1, 0.2), attribution(-1.2, 2.5, 0.2)
        )
        result = grade(built)
        self.assertIn(result["priority"], ALERT_PRIORITIES)
        self.assertNotIn(result["action"], ALERT_DIRECTIONAL_ACTIONS)

    def test_a_first_observation_of_any_detector_is_ungraded(self):
        from core.forecast_alert import forecast_change_alert
        from core.regime_alert import regime_change_alert

        first_forecast = forecast_change_alert(
            None, self.snapshot("NVDA", "2026-10-01", "20d", 0.56)
        )
        first_regime = regime_change_alert(
            None, {"ticker": "MSFT", "label": "bullish", "computable": True}
        )
        for built in (first_forecast, first_regime):
            with self.subTest(alert=built["alert"]):
                self.assertIsNone(grade(built)["priority"])


if __name__ == "__main__":
    unittest.main()
