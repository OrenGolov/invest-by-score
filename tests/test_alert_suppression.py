"""A7 alert suppression tests.

The measurements these tests defend:

* 93.3% of alert-days are repeats (4,622 sessions, 308 episodes, 15.01 per
  episode) - so suppression must actually suppress.
* 11.4% of same-label repeats crossed a decision band and the largest move
  under an unchanged label was a full 1.00 reversal - so it must not suppress
  on the label alone.
"""

from __future__ import annotations

import unittest

from core.alert_suppression import (
    AlertSuppressionError,
    disposition,
    fired_state,
    governance_state,
    render_suppression,
    severity_rank,
    suppression_key,
    suppression_problems,
)
from core.config import (
    ALERT_SEVERITY_INFO,
    ALERT_SEVERITY_WARN,
    ALERT_SUPPRESSION_VERSION,
    PORTFOLIO_NO_TRADE,
    PORTFOLIO_NOT_EVALUATED,
    PORTFOLIO_PROCEED,
    SUPPRESS_COOLDOWN_SESSIONS,
    SUPPRESS_DELIVER,
    SUPPRESS_GATED,
    SUPPRESS_KEY_FIELDS,
    SUPPRESS_KEY_FORBIDDEN,
    SUPPRESS_NOT_EVALUATED,
    SUPPRESS_SUPPRESSED,
)

CLEAR = {"verdict": PORTFOLIO_PROCEED}
NO_VETO = {"veto": False}


def alert(**over):
    """A fired regime alert, the A4 shape."""
    base = {
        "alert": "regime_change",
        "ticker": "AAPL",
        "horizon": "20d",
        "kind": "CONFIRMED",
        "severity": ALERT_SEVERITY_WARN,
    }
    base.update(over)
    return base


class SeverityRankTests(unittest.TestCase):
    def test_warn_outranks_info(self):
        self.assertGreater(
            severity_rank(ALERT_SEVERITY_WARN), severity_rank(ALERT_SEVERITY_INFO)
        )

    def test_absent_severity_is_none_not_zero(self):
        """None is incomparable, not 'less severe than info'."""
        self.assertIsNone(severity_rank(None))

    def test_unknown_severity_raises(self):
        with self.assertRaises(AlertSuppressionError):
            severity_rank("catastrophic")


class FiredStateTests(unittest.TestCase):
    def test_reads_fired_key_from_a1_and_a2(self):
        fired, state = fired_state({"fired": True, "kind": "MOVED"})
        self.assertTrue(fired)
        self.assertEqual(state, "MOVED")

    def test_reads_verdict_key_from_a3_and_a6(self):
        fired, state = fired_state({"verdict": "FIRED"})
        self.assertTrue(fired)
        self.assertEqual(state, "FIRED")

    def test_reads_kind_key_from_a4_and_a5(self):
        fired, state = fired_state({"kind": "CONFIRMED"})
        self.assertTrue(fired)
        self.assertEqual(state, "CONFIRMED")

    def test_not_met_is_quiet(self):
        fired, _ = fired_state({"verdict": "NOT_MET"})
        self.assertFalse(fired)

    def test_not_evaluated_is_neither_fired_nor_quiet(self):
        """A6's whole point: an untestable condition is not a passed one."""
        fired, state = fired_state({"verdict": "NOT_EVALUATED"})
        self.assertIsNone(fired)
        self.assertEqual(state, "NOT_EVALUATED")

    def test_alert_with_no_readable_key_raises(self):
        """Treating an unrecognised shape as quiet would silently drop it."""
        with self.assertRaises(AlertSuppressionError) as ctx:
            fired_state({"alert": "mystery", "ticker": "AAPL"})
        self.assertIn("silently drop", str(ctx.exception))

    def test_non_bool_fired_raises_rather_than_coercing(self):
        with self.assertRaises(AlertSuppressionError):
            fired_state({"fired": "yes"})

    def test_non_mapping_raises(self):
        with self.assertRaises(AlertSuppressionError):
            fired_state(["fired"])


class SuppressionKeyTests(unittest.TestCase):
    def test_key_covers_the_declared_fields(self):
        self.assertEqual(len(suppression_key(alert())), len(SUPPRESS_KEY_FIELDS))

    def test_as_of_never_enters_the_key(self):
        """MEASURED: a key containing as_of is unique every day."""
        a = alert(as_of="2026-09-22")
        b = alert(as_of="2026-09-23")
        self.assertEqual(suppression_key(a), suppression_key(b))

    def test_forbidden_fields_are_not_in_the_key_fields(self):
        for forbidden in SUPPRESS_KEY_FORBIDDEN:
            self.assertNotIn(forbidden, SUPPRESS_KEY_FIELDS)

    def test_different_state_is_a_different_key(self):
        """The 11.4% measurement: state must separate, not just the label."""
        self.assertNotEqual(
            suppression_key(alert(kind="CONFIRMED")),
            suppression_key(alert(kind="ESCALATED")),
        )

    def test_different_ticker_is_a_different_key(self):
        self.assertNotEqual(
            suppression_key(alert(ticker="AAPL")),
            suppression_key(alert(ticker="MSFT")),
        )


class GovernanceStateTests(unittest.TestCase):
    def test_proceed_with_no_veto_is_clear(self):
        clear, _ = governance_state(CLEAR, NO_VETO)
        self.assertIs(clear, True)

    def test_nothing_supplied_is_unknown_not_clear(self):
        clear, reason = governance_state(None, None)
        self.assertIsNone(clear)
        self.assertIn("unknown", reason)

    def test_risk_report_without_veto_field_is_unknown(self):
        clear, reason = governance_state(CLEAR, {"policy_version": "v1"})
        self.assertIsNone(clear)
        self.assertIn("unknown", reason)

    def test_active_veto_blocks(self):
        clear, reason = governance_state(CLEAR, {"veto": True, "veto_rule_ids": ["dd"]})
        self.assertIs(clear, False)
        self.assertIn("dd", reason)

    def test_portfolio_no_trade_blocks(self):
        clear, _ = governance_state({"verdict": PORTFOLIO_NO_TRADE}, NO_VETO)
        self.assertIs(clear, False)

    def test_portfolio_not_evaluated_is_unknown_not_clear(self):
        clear, _ = governance_state({"verdict": PORTFOLIO_NOT_EVALUATED}, NO_VETO)
        self.assertIsNone(clear)

    def test_missing_verdict_is_unknown(self):
        clear, _ = governance_state({"reason": "x"}, NO_VETO)
        self.assertIsNone(clear)

    def test_unknown_verdict_raises_rather_than_passing(self):
        with self.assertRaises(AlertSuppressionError):
            governance_state({"verdict": "MAYBE"}, NO_VETO)


class DispositionTests(unittest.TestCase):
    def test_new_alert_with_clear_governance_is_delivered(self):
        r = disposition(alert(), portfolio=CLEAR, risk=NO_VETO)
        self.assertEqual(r["disposition"], SUPPRESS_DELIVER)
        self.assertTrue(r["actionable"])
        self.assertEqual(suppression_problems(r), [])

    def test_repeat_inside_cooldown_is_suppressed(self):
        r = disposition(
            alert(), previous=alert(), sessions_since=3, portfolio=CLEAR, risk=NO_VETO
        )
        self.assertEqual(r["disposition"], SUPPRESS_SUPPRESSED)
        self.assertFalse(r["actionable"])

    def test_repeat_past_cooldown_is_delivered(self):
        r = disposition(
            alert(),
            previous=alert(),
            sessions_since=SUPPRESS_COOLDOWN_SESSIONS,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_DELIVER)

    def test_escalation_breaks_suppression_inside_cooldown(self):
        """THE DECIDING CASE: 697 of 4,314 repeats moved by 0.10 or more."""
        r = disposition(
            alert(severity=ALERT_SEVERITY_WARN),
            previous=alert(severity=ALERT_SEVERITY_INFO),
            sessions_since=1,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_DELIVER)
        self.assertTrue(r["escalated"])

    def test_de_escalation_does_not_break_suppression(self):
        r = disposition(
            alert(severity=ALERT_SEVERITY_INFO),
            previous=alert(severity=ALERT_SEVERITY_WARN),
            sessions_since=1,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_SUPPRESSED)
        self.assertFalse(r["escalated"])

    def test_severity_appearing_is_not_escalation(self):
        """The shape rule applied to ordering: absent is incomparable, not 0.

        If an absent severity ranked 0, a severity APPEARING would compare as
        1 > 0 and break the cooldown on an alert that did not get worse.
        """
        r = disposition(
            alert(severity=ALERT_SEVERITY_INFO),
            previous=alert(severity=None),
            sessions_since=1,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertFalse(r["escalated"])
        self.assertEqual(r["disposition"], SUPPRESS_SUPPRESSED)

    def test_severity_vanishing_is_not_escalation(self):
        r = disposition(
            alert(severity=None),
            previous=alert(severity=ALERT_SEVERITY_WARN),
            sessions_since=1,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertFalse(r["escalated"])

    def test_unchanged_repeat_past_cooldown_delivers(self):
        """Without this, the de-escalation test would pass on a module that
        simply suppressed everything old."""
        r = disposition(
            alert(severity=ALERT_SEVERITY_WARN),
            previous=alert(severity=ALERT_SEVERITY_WARN),
            sessions_since=SUPPRESS_COOLDOWN_SESSIONS + 5,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_DELIVER)

    def test_de_escalation_past_cooldown_still_suppresses(self):
        """The branch only DECIDES past the cooldown; inside it the cooldown
        already suppresses and this rule proves nothing."""
        r = disposition(
            alert(severity=ALERT_SEVERITY_INFO),
            previous=alert(severity=ALERT_SEVERITY_WARN),
            sessions_since=SUPPRESS_COOLDOWN_SESSIONS + 5,
            portfolio=CLEAR,
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_SUPPRESSED)

    def test_quiet_alert_is_suppressed(self):
        r = disposition(alert(kind="NONE"), portfolio=CLEAR, risk=NO_VETO)
        self.assertEqual(r["disposition"], SUPPRESS_SUPPRESSED)

    def test_not_evaluated_is_its_own_disposition(self):
        r = disposition({"alert": "x", "ticker": "AAPL", "verdict": "NOT_EVALUATED"})
        self.assertEqual(r["disposition"], SUPPRESS_NOT_EVALUATED)
        self.assertIsNone(r["fired"])

    def test_unknown_age_delivers_rather_than_suppressing_on_a_guess(self):
        r = disposition(
            alert(), previous=alert(), sessions_since=None, portfolio=CLEAR, risk=NO_VETO
        )
        self.assertEqual(r["disposition"], SUPPRESS_DELIVER)

    def test_previous_under_a_different_key_raises(self):
        with self.assertRaises(AlertSuppressionError):
            disposition(
                alert(ticker="AAPL"),
                previous=alert(ticker="MSFT"),
                sessions_since=1,
                portfolio=CLEAR,
                risk=NO_VETO,
            )

    def test_negative_sessions_since_raises(self):
        with self.assertRaises(AlertSuppressionError):
            disposition(
                alert(), previous=alert(), sessions_since=-1, portfolio=CLEAR, risk=NO_VETO
            )

    def test_bool_sessions_since_raises(self):
        """True is an int in Python; accepting it would read as 1 session."""
        with self.assertRaises(AlertSuppressionError):
            disposition(
                alert(), previous=alert(), sessions_since=True, portfolio=CLEAR, risk=NO_VETO
            )


class GovernanceGateTests(unittest.TestCase):
    def test_veto_gates_but_does_not_silence(self):
        """The alert a reader most needs is the one that fires while frozen."""
        r = disposition(
            alert(), portfolio={"verdict": PORTFOLIO_NO_TRADE}, risk=NO_VETO
        )
        self.assertEqual(r["disposition"], SUPPRESS_GATED)
        self.assertFalse(r["actionable"])
        self.assertNotEqual(r["disposition"], SUPPRESS_SUPPRESSED)

    def test_risk_veto_gates(self):
        r = disposition(alert(), portfolio=CLEAR, risk={"veto": True, "veto_rule_ids": ["dd"]})
        self.assertEqual(r["disposition"], SUPPRESS_GATED)

    def test_unknown_governance_gates(self):
        r = disposition(alert())
        self.assertEqual(r["disposition"], SUPPRESS_GATED)
        self.assertIsNone(r["governance_clear"])

    def test_escalated_alert_is_still_gated_not_delivered_under_veto(self):
        """Escalation breaks SUPPRESSION; it does not override GOVERNANCE."""
        r = disposition(
            alert(severity=ALERT_SEVERITY_WARN),
            previous=alert(severity=ALERT_SEVERITY_INFO),
            sessions_since=1,
            portfolio={"verdict": PORTFOLIO_NO_TRADE},
            risk=NO_VETO,
        )
        self.assertEqual(r["disposition"], SUPPRESS_GATED)
        self.assertTrue(r["escalated"])
        self.assertFalse(r["actionable"])


class ContractTests(unittest.TestCase):
    def test_clean_record_has_no_problems(self):
        r = disposition(alert(), portfolio=CLEAR, risk=NO_VETO)
        self.assertEqual(suppression_problems(r), [])

    def test_gated_but_actionable_is_a_problem(self):
        r = dict(disposition(alert(), portfolio={"verdict": PORTFOLIO_NO_TRADE}, risk=NO_VETO))
        r["actionable"] = True
        self.assertTrue(any("must not be actionable" in p for p in suppression_problems(r)))

    def test_delivered_without_clear_governance_is_a_problem(self):
        r = dict(disposition(alert(), portfolio=CLEAR, risk=NO_VETO))
        r["governance_clear"] = None
        self.assertTrue(
            any("affirmatively clear" in p for p in suppression_problems(r))
        )

    def test_escalated_and_suppressed_is_a_problem(self):
        r = dict(disposition(alert(kind="NONE"), portfolio=CLEAR, risk=NO_VETO))
        r["escalated"] = True
        self.assertTrue(any("never be suppressed" in p for p in suppression_problems(r)))

    def test_blocks_trades_is_a_problem(self):
        r = dict(disposition(alert(), portfolio=CLEAR, risk=NO_VETO))
        r["blocks_trades"] = True
        self.assertTrue(any("does not trade" in p for p in suppression_problems(r)))

    def test_veto_silences_is_a_problem(self):
        r = dict(disposition(alert(), portfolio=CLEAR, risk=NO_VETO))
        r["veto_silences"] = True
        self.assertTrue(any("must not silence" in p for p in suppression_problems(r)))

    def test_unrecorded_suppression_is_a_problem(self):
        r = dict(disposition(alert(), portfolio=CLEAR, risk=NO_VETO))
        r["records_suppressed"] = False
        self.assertTrue(any("must be recorded" in p for p in suppression_problems(r)))

    def test_missing_reason_is_a_problem(self):
        r = dict(disposition(alert(), portfolio=CLEAR, risk=NO_VETO))
        r["reason"] = ""
        self.assertTrue(any("reason" in p for p in suppression_problems(r)))

    def test_non_mapping_is_a_problem(self):
        self.assertEqual(suppression_problems(["x"]), ["record is not a mapping"])

    def test_version_is_carried(self):
        r = disposition(alert(), portfolio=CLEAR, risk=NO_VETO)
        self.assertEqual(r["version"], ALERT_SUPPRESSION_VERSION)


class RenderTests(unittest.TestCase):
    def test_render_names_the_disposition(self):
        r = disposition(alert(), portfolio=CLEAR, risk=NO_VETO)
        self.assertTrue(any(SUPPRESS_DELIVER in line for line in render_suppression(r)))

    def test_render_shows_unknown_governance_as_three_states(self):
        r = disposition(alert())
        self.assertTrue(any("UNKNOWN" in line for line in render_suppression(r)))

    def test_render_shows_absent_rather_than_a_number(self):
        """The shape rule: an absent severity is never shown as 0."""
        r = disposition(alert())
        text = "\n".join(render_suppression(r))
        self.assertIn("ABSENT", text)

    def test_render_returns_lines_not_a_blob(self):
        r = disposition(alert(), portfolio=CLEAR, risk=NO_VETO)
        lines = render_suppression(r)
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
