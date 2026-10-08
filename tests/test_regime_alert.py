"""A4 regime change alert tests.

The central claim under test: a label difference is not a regime change. These
tests build their classifications in-process rather than reading the gitignored
price ledger, so they mean the same thing on a clean clone.

The real-data measurements quoted here came from 74 tickers and 74,600
labelled sessions in the local 5-year ingest: the classifier flips 16.6 times
per 252 sessions and 26% of runs last a single session.
"""

from __future__ import annotations

import random
import statistics
import unittest

from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    REGIME_ALERT_BLOCKS_TRADES,
    REGIME_ALERT_CONFIRM_SESSIONS,
    REGIME_ALERT_ESCALATED_LABELS,
    REGIME_ALERT_PENDING_FIRES,
    REGIME_ALERT_REQUIRES_CONFIRMATION,
    REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME,
    REGIME_ALERT_VERSION,
    REGIME_CHANGE_APPEARED,
    REGIME_CHANGE_CONFIRMED,
    REGIME_CHANGE_DISAPPEARED,
    REGIME_CHANGE_NONE,
    REGIME_CHANGE_NOT_EVALUATED,
    REGIME_CHANGE_PENDING,
    REGIME_LABELS,
    REGIME_RISKOFF_LABEL,
    REGIME_STRESS_LABEL,
)
from core.regime_alert import (
    RegimeAlertError,
    confirmation_run,
    regime_alert_problems,
    regime_change_alert,
    render_regime_alert,
)


def classification(label, recent=None, computable=True, ticker="NVDA"):
    return {
        "computable": computable,
        "label": label,
        "recent_labels": list(recent or []),
        "ticker": ticker,
    }


def flickering_sequence(seed=5, sessions=3000):
    """A label series with the measured flicker structure.

    26% of runs last one session, which is what makes an unconfirmed alert a
    quarter noise by construction.
    """
    rng = random.Random(seed)
    labels = list(REGIME_LABELS)
    sequence = []
    current = labels[0]
    while len(sequence) < sessions:
        # Draw a run length whose distribution puts about a quarter at 1.
        run = rng.choices([1, 2, 3, 5, 8, 13], weights=[26, 12, 10, 20, 18, 14])[0]
        sequence.extend([current] * run)
        current = rng.choice([entry for entry in labels if entry != current])
    return sequence[:sessions]


class TheDecidingMeasurementTest(unittest.TestCase):
    """A label difference is mostly flicker."""

    def test_the_sequence_has_the_measured_flicker_structure(self):
        sequence = flickering_sequence()
        runs = []
        current, length = sequence[0], 1
        for label in sequence[1:]:
            if label == current:
                length += 1
            else:
                runs.append(length)
                current, length = label, 1
        runs.append(length)
        one_day = sum(1 for run in runs if run == 1) / len(runs)
        self.assertGreater(
            one_day,
            0.15,
            f"only {one_day:.0%} of runs last a single session; A4's "
            f"confirmation rule rests on flicker being common",
        )

    def test_confirmation_cuts_the_alert_rate(self):
        sequence = flickering_sequence()

        def alerts(confirm):
            fired = 0
            last = sequence[0]
            pending, run = None, 0
            for label in sequence:
                if label != last:
                    if pending == label:
                        run += 1
                    else:
                        pending, run = label, 1
                    if run >= confirm:
                        fired += 1
                        last, pending, run = label, None, 0
                else:
                    pending, run = None, 0
            return fired

        unconfirmed = alerts(1)
        confirmed = alerts(REGIME_ALERT_CONFIRM_SESSIONS)
        self.assertLess(
            confirmed,
            unconfirmed * 0.75,
            f"{REGIME_ALERT_CONFIRM_SESSIONS} confirming sessions cut alerts "
            f"only from {unconfirmed} to {confirmed}; the rule buys nothing",
        )

    def test_a_single_session_difference_does_not_fire(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_PENDING)
        self.assertFalse(
            alert["fired"],
            "a one-session flicker fired; MEASURED, 26% of regime runs last "
            "exactly one session",
        )

    def test_a_confirmed_change_fires(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_CONFIRMED)
        self.assertTrue(alert["fired"])
        self.assertEqual(alert["run"], REGIME_ALERT_CONFIRM_SESSIONS)

    def test_the_config_requires_confirmation(self):
        self.assertTrue(REGIME_ALERT_REQUIRES_CONFIRMATION)
        self.assertGreaterEqual(REGIME_ALERT_CONFIRM_SESSIONS, 2)

    def test_an_unconfirmed_change_marked_confirmed_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
        )
        alert["kind"] = REGIME_CHANGE_CONFIRMED
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(
            regime_alert_problems(alert),
            "a one-session change marked CONFIRMED passed the contract check",
        )


class ConfirmationRunTest(unittest.TestCase):
    def test_it_counts_backwards_from_the_newest_session(self):
        self.assertEqual(confirmation_run(["a", "b", "b", "b"], "b"), 3)

    def test_an_interrupted_run_does_not_count(self):
        self.assertEqual(
            confirmation_run(["b", "b", "a", "b"], "b"),
            1,
            "an interrupted run was counted as persistence; that "
            "interruption is exactly the flicker being filtered",
        )

    def test_no_history_is_no_confirmation(self):
        self.assertEqual(confirmation_run(None, "bullish"), 0)
        self.assertEqual(confirmation_run([], "bullish"), 0)

    def test_no_label_is_no_confirmation(self):
        self.assertEqual(confirmation_run(["a", "a"], None), 0)

    def test_without_history_a_change_stays_pending(self):
        """An unverifiable claim of persistence is not evidence."""
        alert = regime_change_alert(
            classification("bullish"), classification("risk_off")
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_PENDING)
        self.assertFalse(alert["fired"])


class PendingTest(unittest.TestCase):
    """A transition in progress is shown, not raised."""

    def test_pending_does_not_fire(self):
        self.assertFalse(REGIME_ALERT_PENDING_FIRES)

    def test_pending_is_distinct_from_none(self):
        self.assertNotEqual(REGIME_CHANGE_PENDING, REGIME_CHANGE_NONE)

    def test_pending_reports_its_progress(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 18 + ["risk_off"] * 2),
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_PENDING)
        self.assertEqual(alert["run"], 2)
        self.assertIn("2 of 3", alert["reason"])

    def test_a_pending_that_fired_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
        )
        alert["fired"] = True
        alert["severity"] = ALERT_SEVERITY_WARN
        self.assertTrue(regime_alert_problems(alert))

    def test_a_confirmed_run_reported_pending_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        alert["kind"] = REGIME_CHANGE_PENDING
        alert["fired"] = False
        alert["severity"] = None
        self.assertTrue(regime_alert_problems(alert))


class EscalationTest(unittest.TestCase):
    """Escalation is in severity, not in skipping evidence."""

    def test_risk_off_is_escalated(self):
        self.assertIn(REGIME_RISKOFF_LABEL, REGIME_ALERT_ESCALATED_LABELS)
        self.assertIn(REGIME_STRESS_LABEL, REGIME_ALERT_ESCALATED_LABELS)

    def test_a_confirmed_risk_off_warns(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        self.assertTrue(alert["escalated"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)

    def test_a_confirmed_bullish_informs(self):
        alert = regime_change_alert(
            classification("risk_off"),
            classification("bullish", ["risk_off"] * 17 + ["bullish"] * 3),
        )
        self.assertFalse(alert["escalated"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_INFO)

    def test_escalation_does_not_skip_confirmation(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("stress", ["bullish"] * 19 + ["stress"]),
        )
        self.assertEqual(
            alert["kind"],
            REGIME_CHANGE_PENDING,
            "an escalated label skipped confirmation; the escalation is in "
            "severity, not in the evidence required",
        )

    def test_an_escalated_change_firing_at_info_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        alert["severity"] = ALERT_SEVERITY_INFO
        self.assertTrue(regime_alert_problems(alert))


class AvailabilityTest(unittest.TestCase):
    """An uncomputable regime is not a calm one."""

    def test_the_config_says_so(self):
        self.assertTrue(REGIME_ALERT_UNCOMPUTABLE_IS_NOT_A_REGIME)

    def test_a_regime_becoming_computable_is_an_appearance(self):
        alert = regime_change_alert(
            classification(None, computable=False),
            classification("bullish", ["bullish"] * 3),
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_APPEARED)
        self.assertIsNone(alert["previous"])
        self.assertTrue(alert["fired"])

    def test_a_regime_going_away_is_a_disappearance(self):
        alert = regime_change_alert(
            classification("bullish"), classification(None, computable=False)
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_DISAPPEARED)
        self.assertIsNone(alert["current"])
        self.assertEqual(alert["severity"], ALERT_SEVERITY_WARN)
        self.assertIn("did not become calm", alert["reason"])

    def test_a_stale_label_on_an_uncomputable_regime_is_not_live(self):
        """The real hazard, and the one a naive guard misses.

        Every other uncomputable case already carries label=None, so the
        computable guard never decides. A classifier reporting
        computable=False while leaving its previous label in place would
        otherwise be read as a live regime and fire CONFIRMED.
        """
        alert = regime_change_alert(
            classification("bullish"),
            {
                "computable": False,
                "label": "risk_off",
                "recent_labels": ["risk_off"] * 3,
                "ticker": "NVDA",
            },
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_DISAPPEARED)
        self.assertIsNone(
            alert["current"],
            "a label that survived its own computability was reported as the "
            "current regime",
        )
        self.assertEqual(regime_alert_problems(alert), [])

    def test_both_uncomputable_is_not_evaluated(self):
        alert = regime_change_alert(
            classification(None, computable=False),
            classification(None, computable=False),
        )
        self.assertEqual(alert["kind"], REGIME_CHANGE_NOT_EVALUATED)
        self.assertFalse(alert["fired"])

    def test_an_appearance_with_a_previous_regime_is_caught(self):
        alert = regime_change_alert(
            classification(None, computable=False),
            classification("bullish", ["bullish"] * 3),
        )
        alert["previous"] = "risk_off"
        self.assertTrue(regime_alert_problems(alert))


class ContractTest(unittest.TestCase):
    def test_clean_alerts_have_no_problems(self):
        for previous, current in (
            (classification("bullish"), classification("bullish", ["bullish"] * 20)),
            (
                classification("bullish"),
                classification("risk_off", ["bullish"] * 19 + ["risk_off"]),
            ),
            (
                classification("bullish"),
                classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
            ),
            (classification(None, computable=False), classification("bullish")),
            (classification("bullish"), classification(None, computable=False)),
            (None, classification("bullish")),
        ):
            alert = regime_change_alert(previous, current)
            self.assertEqual(regime_alert_problems(alert), [], alert["kind"])
            self.assertEqual(alert["version"], REGIME_ALERT_VERSION)

    def test_a_first_observation_is_not_a_change(self):
        alert = regime_change_alert(None, classification("bullish"))
        self.assertEqual(alert["kind"], REGIME_CHANGE_NOT_EVALUATED)
        self.assertFalse(alert["fired"])

    def test_the_alert_does_not_block_trades(self):
        alert = regime_change_alert(
            classification("bullish"), classification("bullish", ["bullish"] * 20)
        )
        self.assertFalse(alert["blocks_trades"])
        self.assertFalse(REGIME_ALERT_BLOCKS_TRADES)

    def test_an_alert_that_blocks_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"), classification("bullish", ["bullish"] * 20)
        )
        alert["blocks_trades"] = True
        self.assertTrue(regime_alert_problems(alert))

    def test_a_silenced_transition_is_caught(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        alert["kind"] = REGIME_CHANGE_NONE
        alert["fired"] = False
        alert["severity"] = None
        self.assertTrue(regime_alert_problems(alert))

    def test_an_unknown_label_is_refused(self):
        with self.assertRaises(RegimeAlertError):
            regime_change_alert(
                classification("bullish"), classification("euphoric")
            )

    def test_a_non_mapping_classification_is_refused(self):
        with self.assertRaises(RegimeAlertError):
            regime_change_alert("bullish", classification("bullish"))

    def test_a_missing_current_classification_is_refused(self):
        with self.assertRaises(RegimeAlertError):
            regime_change_alert(classification("bullish"), None)

    def test_an_impossible_confirmation_window_is_refused(self):
        with self.assertRaises(RegimeAlertError):
            regime_change_alert(
                classification("bullish"),
                classification("risk_off"),
                confirm_sessions=0,
            )


class RenderTest(unittest.TestCase):
    def test_the_transition_and_run_are_rendered(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        text = "\n".join(render_regime_alert(alert))
        self.assertIn("bullish", text)
        self.assertIn("risk_off", text)
        self.assertIn("3/3", text)
        self.assertIn("FIRED", text)

    def test_an_escalated_change_is_marked(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        self.assertIn("!!", "\n".join(render_regime_alert(alert)))

    def test_an_absent_regime_renders_as_absent(self):
        alert = regime_change_alert(
            classification("bullish"), classification(None, computable=False)
        )
        self.assertIn("—", render_regime_alert(alert)[0])

    def test_lines_stay_readable(self):
        alert = regime_change_alert(
            classification("bullish"),
            classification("risk_off", ["bullish"] * 17 + ["risk_off"] * 3),
        )
        for line in render_regime_alert(alert):
            self.assertLess(len(line), 250)


if __name__ == "__main__":
    unittest.main()
