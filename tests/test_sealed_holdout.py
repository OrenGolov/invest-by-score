"""X8 sealed holdout tests.

The load-bearing cases: ABSENT is not UNOPENED, an intact-but-small seal is
INSUFFICIENT and stays sealed, a second opening is refused rather than logged,
and a boundary that cannot be checked is not a pass.
"""

from __future__ import annotations

import unittest

from core.config import (
    HOLDOUT_ABSENT,
    HOLDOUT_INSUFFICIENT,
    HOLDOUT_MAX_OPENINGS,
    HOLDOUT_MIN_OBSERVATIONS,
    HOLDOUT_OPENED,
    HOLDOUT_STATES,
    HOLDOUT_UNOPENED,
    OOS_MIN_OBSERVATIONS,
    SEALED_HOLDOUT_VERSION,
)
from core.sealed_holdout import (
    SealedHoldoutError,
    evaluate_sealed_holdout,
    holdout_state,
    record_opening,
    sampling_band,
    seal_problems,
    sealed_holdout_problems,
    render_sealed_holdout,
    verify_boundary,
)


def _seal(**overrides):
    """The seal X8 measured on the shipped runs, overridable per test."""
    seal = {
        "dataset_hash": "d18d5e5e3cb62dd3365b0d5c363b3e54e9bfd831e7de0b9d9916ee8b92c0a0de",
        "sealed_rows": [346, 405],
        "row_count": 60,
        "sealed_at": "2026-09-18T00:00:00Z",
        "openings": [],
    }
    seal.update(overrides)
    return seal


def _big_seal(**overrides):
    """A seal at or above the floor, so the opening path is reachable."""
    base = {"sealed_rows": [300, 499], "row_count": 200}
    base.update(overrides)
    return _seal(**base)


class SamplingBandTests(unittest.TestCase):

    def test_the_band_matches_the_measured_values(self):
        # The numbers quoted in the X8 docstring and config must be the ones the
        # code computes, or the justification cites a measurement nobody ran.
        self.assertAlmostEqual(sampling_band(60), 0.1265, places=4)
        self.assertAlmostEqual(sampling_band(106), 0.0952, places=4)
        self.assertAlmostEqual(sampling_band(120), 0.0895, places=4)

    def test_a_coin_reaches_sixty_two_percent_at_sixty_rows(self):
        # Finding 3, the reason the seal stays sealed.
        self.assertGreater(0.5 + sampling_band(60), 0.62)

    def test_the_band_narrows_as_observations_grow(self):
        bands = [sampling_band(n) for n in (30, 60, 120, 240)]
        self.assertEqual(bands, sorted(bands, reverse=True))

    def test_no_observations_raises_rather_than_returning_a_wide_band(self):
        for count in (0, -1):
            with self.subTest(count=count):
                with self.assertRaises(SealedHoldoutError):
                    sampling_band(count)


class SealContractTests(unittest.TestCase):

    def test_the_measured_seal_is_usable(self):
        self.assertEqual(seal_problems(_seal()), [])

    def test_a_missing_seal_is_named_not_crashed_on(self):
        self.assertEqual(seal_problems(None), ["no seal record"])

    def test_a_non_mapping_seal_is_refused(self):
        self.assertTrue(seal_problems("sealed"))
        self.assertTrue(seal_problems([346, 405]))

    def test_every_required_field_is_checked(self):
        for field in ("dataset_hash", "sealed_rows", "row_count", "sealed_at"):
            with self.subTest(field=field):
                seal = _seal()
                del seal[field]
                self.assertTrue(
                    any(field in problem for problem in seal_problems(seal)),
                    f"removing {field} was not reported",
                )

    def test_a_row_count_disagreeing_with_the_range_is_refused(self):
        # A seal describing two different windows cannot be verified against
        # the data, so the disagreement must be loud.
        problems = seal_problems(_seal(row_count=59))
        self.assertTrue(any("does not match" in p for p in problems), problems)

    def test_an_inflated_row_count_cannot_make_a_small_seal_look_decisive(self):
        # Regression for an inert gate probe. Without the count/range check,
        # a 60-row seal declaring 500 rows evaluates to UNOPENED with
        # may_open TRUE — an insufficient seal presenting itself as decisive,
        # which is precisely what X8 exists to prevent.
        inflated = _seal(row_count=500)
        self.assertTrue(seal_problems(inflated))
        with self.assertRaises(SealedHoldoutError):
            evaluate_sealed_holdout(inflated, touched_through=299)

    def test_a_deflated_row_count_is_refused_too(self):
        # The same cause in the other direction: understating the evidence.
        self.assertTrue(seal_problems(_seal(row_count=10)))

    def test_a_backwards_range_is_refused(self):
        problems = seal_problems(_seal(sealed_rows=[405, 346], row_count=60))
        self.assertTrue(any("backwards" in p for p in problems), problems)

    def test_a_range_that_is_not_a_pair_is_refused(self):
        for rows in ([346], [346, 400, 405], "346-405"):
            with self.subTest(rows=rows):
                self.assertTrue(seal_problems(_seal(sealed_rows=rows)))

    def test_a_seal_covering_nothing_is_refused(self):
        problems = seal_problems(_seal(sealed_rows=[346, 346], row_count=0))
        self.assertTrue(problems)

    def test_more_openings_than_permitted_is_refused(self):
        openings = [
            {"opened_at": "2026-09-20", "opened_by": "a", "reason": "first"},
            {"opened_at": "2026-09-21", "opened_by": "b", "reason": "second"},
        ]
        problems = seal_problems(_seal(openings=openings))
        self.assertTrue(any("openings recorded" in p for p in problems), problems)

    def test_an_unattributed_opening_is_refused(self):
        for field in ("opened_at", "opened_by", "reason"):
            with self.subTest(field=field):
                opening = {"opened_at": "x", "opened_by": "y", "reason": "z"}
                opening[field] = ""
                problems = seal_problems(_seal(openings=[opening]))
                self.assertTrue(
                    any(field in p for p in problems),
                    f"a blank {field} was not reported",
                )


class HoldoutStateTests(unittest.TestCase):

    def test_no_seal_is_absent_not_unopened(self):
        # The distinction X8 exists to protect: "no evidence is held" and "held
        # evidence is intact" are opposite facts.
        self.assertEqual(holdout_state(None), HOLDOUT_ABSENT)
        self.assertNotEqual(HOLDOUT_ABSENT, HOLDOUT_UNOPENED)

    def test_the_measured_seal_is_insufficient(self):
        self.assertEqual(holdout_state(_seal()), HOLDOUT_INSUFFICIENT)

    def test_a_seal_at_the_floor_is_unopened(self):
        seal = _seal(sealed_rows=[300, 300 + HOLDOUT_MIN_OBSERVATIONS - 1],
                     row_count=HOLDOUT_MIN_OBSERVATIONS)
        self.assertEqual(holdout_state(seal), HOLDOUT_UNOPENED)

    def test_one_below_the_floor_is_insufficient(self):
        count = HOLDOUT_MIN_OBSERVATIONS - 1
        seal = _seal(sealed_rows=[300, 300 + count - 1], row_count=count)
        self.assertEqual(holdout_state(seal), HOLDOUT_INSUFFICIENT)

    def test_an_opened_small_seal_reports_opened_not_insufficient(self):
        # Order matters: reporting a spent seal as INSUFFICIENT would hide that
        # the evidence is already gone.
        seal = _seal(openings=[
            {"opened_at": "2026-09-20", "opened_by": "orengolov02@gmail.com",
             "reason": "final evaluation"},
        ])
        self.assertEqual(holdout_state(seal), HOLDOUT_OPENED)

    def test_an_invalid_seal_raises_rather_than_guessing_a_state(self):
        with self.assertRaises(SealedHoldoutError):
            holdout_state(_seal(row_count=59))

    def test_every_state_is_declared(self):
        for state in (HOLDOUT_ABSENT, HOLDOUT_INSUFFICIENT, HOLDOUT_OPENED, HOLDOUT_UNOPENED):
            with self.subTest(state=state):
                self.assertIn(state, HOLDOUT_STATES)


class BoundaryTests(unittest.TestCase):

    def test_the_measured_boundary_is_intact(self):
        # Validation reached row 299; the seal starts at 346.
        result = verify_boundary(_seal(), touched_through=299)
        self.assertTrue(result["verified"])
        self.assertEqual(result["gap_rows"], 46)

    def test_validation_inside_the_seal_is_refused(self):
        result = verify_boundary(_seal(), touched_through=350)
        self.assertFalse(result["verified"])
        self.assertIn("INSIDE", result["reason"])

    def test_touching_the_first_sealed_row_is_a_breach(self):
        # The seal is inclusive of its start, so reaching it is reaching in.
        result = verify_boundary(_seal(), touched_through=346)
        self.assertFalse(result["verified"])

    def test_stopping_one_row_short_is_intact(self):
        result = verify_boundary(_seal(), touched_through=345)
        self.assertTrue(result["verified"])
        self.assertEqual(result["gap_rows"], 0)

    def test_an_unrecorded_reach_is_not_verified(self):
        # "Nobody recorded what was touched" is not "nothing was touched".
        result = verify_boundary(_seal(), touched_through=None)
        self.assertFalse(result["verified"])
        self.assertIsNone(result["touched_through"])

    def test_an_invalid_seal_cannot_be_verified(self):
        with self.assertRaises(SealedHoldoutError):
            verify_boundary(_seal(sealed_rows=[405, 346]), touched_through=299)


class EvaluateTests(unittest.TestCase):

    def test_the_shipped_state_is_insufficient_and_stays_sealed(self):
        report = evaluate_sealed_holdout(_seal(), touched_through=299)
        self.assertEqual(report["state"], HOLDOUT_INSUFFICIENT)
        self.assertFalse(report["may_open"])
        self.assertEqual(report["row_count"], 60)
        self.assertIn("STAYS SEALED", report["reason"])
        self.assertEqual(sealed_holdout_problems(report), [])

    def test_an_absent_seal_reports_no_row_count(self):
        report = evaluate_sealed_holdout(None)
        self.assertEqual(report["state"], HOLDOUT_ABSENT)
        self.assertIsNone(report["row_count"])
        self.assertFalse(report["may_open"])
        self.assertEqual(sealed_holdout_problems(report), [])

    def test_a_sufficient_verified_seal_may_open_once(self):
        report = evaluate_sealed_holdout(_big_seal(), touched_through=299)
        self.assertEqual(report["state"], HOLDOUT_UNOPENED)
        self.assertTrue(report["may_open"])
        self.assertEqual(sealed_holdout_problems(report), [])

    def test_a_sufficient_seal_with_an_unverified_boundary_may_not_open(self):
        report = evaluate_sealed_holdout(_big_seal(), touched_through=None)
        self.assertFalse(report["may_open"])
        self.assertIn("boundary not verified", report["reason"])

    def test_a_breached_seal_may_not_open(self):
        report = evaluate_sealed_holdout(_big_seal(), touched_through=400)
        self.assertFalse(report["may_open"])
        self.assertEqual(sealed_holdout_problems(report), [])

    def test_an_opened_seal_may_never_open_again(self):
        seal = _big_seal(openings=[
            {"opened_at": "2026-09-20", "opened_by": "orengolov02@gmail.com",
             "reason": "final evaluation"},
        ])
        report = evaluate_sealed_holdout(seal, touched_through=299)
        self.assertEqual(report["state"], HOLDOUT_OPENED)
        self.assertFalse(report["may_open"])
        self.assertIn("exactly once", report["reason"])

    def test_the_report_never_blocks_trades(self):
        report = evaluate_sealed_holdout(_seal(), touched_through=299)
        self.assertFalse(report["blocks_trades"])

    def test_the_reasons_for_refusal_are_distinguishable(self):
        # Three different "no" answers must not collapse into one message.
        small = evaluate_sealed_holdout(_seal(), touched_through=299)["reason"]
        unverified = evaluate_sealed_holdout(_big_seal(), touched_through=None)["reason"]
        opened = evaluate_sealed_holdout(
            _big_seal(openings=[{"opened_at": "a", "opened_by": "b", "reason": "c"}]),
            touched_through=299,
        )["reason"]
        self.assertEqual(len({small, unverified, opened}), 3)


class RecordOpeningTests(unittest.TestCase):

    def test_opening_a_sufficient_seal_appends_one_record(self):
        updated = record_opening(
            _big_seal(), opened_by="orengolov02@gmail.com",
            reason="final frozen evaluation", opened_at="2026-09-27T00:00:00Z",
            touched_through=299,
        )
        self.assertEqual(len(updated["openings"]), 1)
        self.assertEqual(holdout_state(updated), HOLDOUT_OPENED)

    def test_a_second_opening_is_refused_not_logged(self):
        once = record_opening(
            _big_seal(), opened_by="orengolov02@gmail.com", reason="first",
            opened_at="2026-09-27T00:00:00Z", touched_through=299,
        )
        with self.assertRaises(SealedHoldoutError) as caught:
            record_opening(
                once, opened_by="orengolov02@gmail.com", reason="second",
                opened_at="2026-09-28T00:00:00Z", touched_through=299,
            )
        self.assertIn("exactly once", str(caught.exception))

    def test_the_insufficient_seal_cannot_be_opened(self):
        with self.assertRaises(SealedHoldoutError) as caught:
            record_opening(
                _seal(), opened_by="orengolov02@gmail.com", reason="curiosity",
                opened_at="2026-09-27T00:00:00Z", touched_through=299,
            )
        self.assertIn("STAYS SEALED", str(caught.exception))

    def test_an_unattributed_opening_is_refused(self):
        for field in ("opened_by", "reason", "opened_at"):
            with self.subTest(field=field):
                kwargs = {
                    "opened_by": "orengolov02@gmail.com",
                    "reason": "final evaluation",
                    "opened_at": "2026-09-27T00:00:00Z",
                }
                kwargs[field] = "   "
                with self.assertRaises(SealedHoldoutError):
                    record_opening(_big_seal(), touched_through=299, **kwargs)

    def test_opening_does_not_mutate_the_original_seal(self):
        seal = _big_seal()
        record_opening(
            seal, opened_by="orengolov02@gmail.com", reason="final",
            opened_at="2026-09-27T00:00:00Z", touched_through=299,
        )
        self.assertEqual(seal["openings"], [])

    def test_a_breached_seal_cannot_be_opened(self):
        with self.assertRaises(SealedHoldoutError):
            record_opening(
                _big_seal(), opened_by="orengolov02@gmail.com", reason="final",
                opened_at="2026-09-27T00:00:00Z", touched_through=400,
            )


class ReportContractTests(unittest.TestCase):

    def test_a_missing_report_is_named(self):
        self.assertEqual(sealed_holdout_problems(None), ["no sealed-holdout report"])

    def test_a_non_mapping_report_is_refused(self):
        self.assertTrue(sealed_holdout_problems("UNOPENED"))

    def test_a_wrong_version_is_caught(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["version"] = "sealed-holdout-v0"
        self.assertTrue(sealed_holdout_problems(report))

    def test_an_undeclared_state_is_caught(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["state"] = "PROBABLY_FINE"
        self.assertTrue(sealed_holdout_problems(report))

    def test_a_report_claiming_to_block_trades_is_refused(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["blocks_trades"] = True
        self.assertTrue(sealed_holdout_problems(report))

    def test_unopened_below_the_floor_is_caught(self):
        # The contract check must catch a state asserted against its own numbers.
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["state"] = HOLDOUT_UNOPENED
        problems = sealed_holdout_problems(report)
        self.assertTrue(any("below the" in p for p in problems), problems)

    def test_insufficient_above_the_floor_is_caught(self):
        report = dict(evaluate_sealed_holdout(_big_seal(), touched_through=299))
        report["state"] = HOLDOUT_INSUFFICIENT
        problems = sealed_holdout_problems(report)
        self.assertTrue(any("meets the" in p for p in problems), problems)

    def test_opened_without_an_opening_is_caught(self):
        report = dict(evaluate_sealed_holdout(_big_seal(), touched_through=299))
        report["state"] = HOLDOUT_OPENED
        self.assertTrue(sealed_holdout_problems(report))

    def test_unopened_with_an_opening_is_caught(self):
        report = dict(evaluate_sealed_holdout(_big_seal(), touched_through=299))
        report["openings"] = 1
        self.assertTrue(sealed_holdout_problems(report))

    def test_may_open_on_an_insufficient_seal_is_caught(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["may_open"] = True
        problems = sealed_holdout_problems(report)
        self.assertTrue(any("stays sealed" in p for p in problems), problems)

    def test_may_open_on_an_unverified_boundary_is_caught(self):
        report = dict(evaluate_sealed_holdout(_big_seal(), touched_through=None))
        report["may_open"] = True
        self.assertTrue(sealed_holdout_problems(report))

    def test_a_forged_sampling_band_is_caught(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["sampling_band"] = 0.01
        problems = sealed_holdout_problems(report)
        self.assertTrue(any("does not match" in p for p in problems), problems)

    def test_a_boundary_verified_while_breached_is_caught(self):
        report = dict(evaluate_sealed_holdout(_big_seal(), touched_through=400))
        report["boundary"] = dict(report["boundary"], verified=True)
        problems = sealed_holdout_problems(report)
        self.assertTrue(any("inside the seal" in p for p in problems), problems)

    def test_an_absent_report_carrying_a_row_count_is_caught(self):
        report = dict(evaluate_sealed_holdout(None))
        report["row_count"] = 60
        self.assertTrue(sealed_holdout_problems(report))

    def test_an_absent_report_that_may_open_is_caught(self):
        report = dict(evaluate_sealed_holdout(None))
        report["may_open"] = True
        self.assertTrue(sealed_holdout_problems(report))


class RenderTests(unittest.TestCase):

    def test_the_shipped_report_renders_the_measured_numbers(self):
        text = render_sealed_holdout(
            evaluate_sealed_holdout(_seal(), touched_through=299)
        )
        self.assertIn("INSUFFICIENT", text)
        self.assertIn("[346, 405] (60 rows)", text)
        self.assertIn("+/-0.1265", text)
        self.assertIn("STAYS SEALED", text)

    def test_an_absent_seal_renders_without_numbers_it_does_not_have(self):
        text = render_sealed_holdout(evaluate_sealed_holdout(None))
        self.assertIn(HOLDOUT_ABSENT, text)
        self.assertNotIn("sealed rows", text)

    def test_an_opening_is_rendered_with_its_attribution(self):
        seal = _big_seal(openings=[
            {"opened_at": "2026-09-20", "opened_by": "orengolov02@gmail.com",
             "reason": "final evaluation"},
        ])
        text = render_sealed_holdout(evaluate_sealed_holdout(seal, touched_through=299))
        self.assertIn("orengolov02@gmail.com", text)
        self.assertIn("final evaluation", text)

    def test_an_invalid_report_is_not_rendered(self):
        report = dict(evaluate_sealed_holdout(_seal(), touched_through=299))
        report["may_open"] = True
        with self.assertRaises(SealedHoldoutError):
            render_sealed_holdout(report)


class ConfigReuseTests(unittest.TestCase):

    def test_the_floor_is_x1s_observation_floor(self):
        # Reused rather than reinvented, so "enough evidence" means one thing.
        self.assertEqual(HOLDOUT_MIN_OBSERVATIONS, OOS_MIN_OBSERVATIONS)

    def test_exactly_one_opening_is_permitted(self):
        self.assertEqual(HOLDOUT_MAX_OPENINGS, 1)

    def test_the_version_is_declared(self):
        self.assertEqual(SEALED_HOLDOUT_VERSION, "sealed-holdout-v1")


if __name__ == "__main__":
    unittest.main()
