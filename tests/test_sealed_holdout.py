"""X8 sealed-holdout tests.

The behaviour under test is that a seal must be RECORDED to count. The shipped
runs keep their tail untouched by arithmetic accident and record none of the
four facts needed to prove it, so the honest verdict is NOT_EVALUATED.
"""

from __future__ import annotations

import json
import pathlib
import unittest

from core.config import (
    HOLDOUT_BLOCKS_TRADES,
    HOLDOUT_BURNED,
    HOLDOUT_BURNED_IS_TERMINAL,
    HOLDOUT_MAX_OPENINGS,
    HOLDOUT_NOT_EVALUATED,
    HOLDOUT_OPENED,
    HOLDOUT_OPENING_REQUIRES_REASON,
    HOLDOUT_RECORDS_BOUNDS,
    HOLDOUT_RECORDS_GEOMETRY,
    HOLDOUT_REQUIRES_EMBARGO,
    HOLDOUT_SEALED,
    HOLDOUT_VERDICTS,
    LABEL_HORIZON_SESSIONS,
)
from core.sealed_holdout import (
    HOLDOUT_REASON_EMBARGO_SHORT,
    HOLDOUT_REASON_NO_BOUNDS,
    HOLDOUT_REASON_NO_RUNS,
    HOLDOUT_REASON_OVERLAP,
    SEAL_REQUIRED_FACTS,
    SealedHoldoutError,
    describe_seal,
    evaluate_sealed_holdout,
    geometry_problems,
    holdout_problems,
    load_openings,
    max_label_horizon,
    minimum_rows,
    opening_verdict,
    recorded_facts,
    render_sealed_holdout,
)


def run(rows=700, fold=60, embargo=252, holdout=60, validation_end=371, extra=None):
    """A run that records all four seal facts. 700 rows clears the 432 minimum."""
    return {
        "estimator": "ridge",
        "holdout": [rows - holdout, rows - 1],
        "dataset_rows": rows,
        "geometry": {
            "fold_sessions": fold,
            "embargo_sessions": embargo,
            "holdout_sessions": holdout,
        },
        "folds": [
            {
                "fold_id": 0,
                "train": [0, fold - 1],
                "validation": [validation_end - fold + 1, validation_end],
                "predictions": [],
                "actuals": [],
            }
        ]
        + (extra or []),
    }


def shipped_run():
    """The shape the 8 runs in `data/training_runs.jsonl` actually have."""
    return {
        "estimator": "ridge",
        "folds": [
            {
                "fold_id": 0,
                "train_rows": 120,
                "validation_rows": 120,
                "train_end_time": "2025-04-16 13:30:00",
                "validation_start_time": "2025-07-21 13:30:00",
            }
        ],
    }


class ARecordedSealIsRequiredTests(unittest.TestCase):
    """The deciding measurement: an unrecorded seal cannot be audited."""

    def test_the_contract_requires_recorded_bounds(self):
        self.assertTrue(HOLDOUT_RECORDS_BOUNDS)
        self.assertTrue(HOLDOUT_RECORDS_GEOMETRY)

    def test_the_shipped_run_shape_records_no_seal_facts(self):
        facts = recorded_facts(shipped_run())
        self.assertEqual(sorted(facts), sorted(SEAL_REQUIRED_FACTS))
        self.assertEqual([f for f, present in facts.items() if present], [])

    def test_a_run_without_a_seal_is_not_evaluated(self):
        report = evaluate_sealed_holdout([shipped_run()], dataset_rows=406)
        self.assertEqual(report["verdict"], HOLDOUT_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], HOLDOUT_REASON_NO_BOUNDS)

    def test_an_unevaluated_report_carries_no_bounds(self):
        # THE SHAPE RULE: [0, 0] would read as "the seal is empty", a different
        # claim from "the seal could not be found".
        report = evaluate_sealed_holdout([shipped_run()], dataset_rows=406)
        self.assertIsNone(report.get("holdout"))
        self.assertIsNone(report.get("holdout_share"))

    def test_the_missing_facts_are_named(self):
        report = evaluate_sealed_holdout([shipped_run()], dataset_rows=406)
        self.assertEqual(sorted(report["missing_facts"]), sorted(SEAL_REQUIRED_FACTS))

    def test_row_counts_are_not_indices(self):
        # `train_rows` says HOW MANY, never WHERE. Only position proves a fold
        # stopped short of the tail.
        facts = recorded_facts(
            {"folds": [{"train_rows": 120, "validation_rows": 120}]}
        )
        self.assertFalse(facts["fold_indices"])

    def test_no_runs_is_not_evaluated(self):
        for empty in ([], None):
            report = evaluate_sealed_holdout(empty)
            self.assertEqual(report["verdict"], HOLDOUT_NOT_EVALUATED)
            self.assertEqual(report["reason_code"], HOLDOUT_REASON_NO_RUNS)

    def test_a_non_mapping_run_is_refused(self):
        with self.assertRaises(SealedHoldoutError):
            recorded_facts(["not", "a", "mapping"])


class AProperSealTests(unittest.TestCase):
    """SEALED must be reachable, or NOT_EVALUATED proves nothing."""

    def setUp(self):
        self.report = evaluate_sealed_holdout([run()])

    def test_a_recorded_legal_unread_holdout_is_sealed(self):
        self.assertEqual(self.report["verdict"], HOLDOUT_SEALED)

    def test_a_sealed_report_names_its_bounds(self):
        self.assertEqual(self.report["holdout"], [640, 699])
        self.assertEqual(self.report["holdout_rows"], 60)

    def test_a_sealed_report_is_contract_clean(self):
        self.assertEqual(holdout_problems(self.report), [])

    def test_a_sealed_holdout_has_no_openings(self):
        self.assertEqual(self.report["openings"], 0)

    def test_the_share_is_reported(self):
        self.assertAlmostEqual(self.report["holdout_share"], 60 / 700, places=6)


class TheGapIsNotTheEmbargoTests(unittest.TestCase):
    """MEASURED: a legal embargo SETTING does not imply an embargoed seal.

    The setting bounds fold spacing; the gap is where the last fold actually
    stopped. Only the gap can leak a holdout label, so checking the setting
    alone passes a run whose final fold ran up against the tail.
    """

    def test_a_legal_embargo_can_still_leave_a_short_gap(self):
        described = describe_seal(700, 60, 252, 60, last_validation_row=600)
        self.assertTrue(described["legal"])
        self.assertEqual(described["gap_to_holdout"], 39)
        self.assertFalse(described["gap_covers_horizon"])

    def test_a_short_gap_is_not_evaluated(self):
        report = evaluate_sealed_holdout([run(validation_end=600)])
        self.assertEqual(report["verdict"], HOLDOUT_NOT_EVALUATED)
        self.assertEqual(report["reason_code"], HOLDOUT_REASON_EMBARGO_SHORT)

    def test_the_embargo_requirement_is_declared(self):
        self.assertTrue(HOLDOUT_REQUIRES_EMBARGO)

    def test_the_gap_is_measured_against_the_longest_horizon(self):
        self.assertEqual(max_label_horizon(), max(LABEL_HORIZON_SESSIONS.values()))
        self.assertEqual(max_label_horizon(), 252)


class OverlapBurnsTheSealTests(unittest.TestCase):
    def test_a_fold_inside_the_holdout_burns_it(self):
        report = evaluate_sealed_holdout([run(validation_end=660)])
        self.assertEqual(report["verdict"], HOLDOUT_BURNED)
        self.assertEqual(report["reason_code"], HOLDOUT_REASON_OVERLAP)

    def test_overlap_is_measured_across_every_run(self):
        # The seal is a property of the WHOLE search: one estimator reaching
        # into the tail spends it for all of them.
        clean, dirty = run(), run(validation_end=660)
        report = evaluate_sealed_holdout([clean, dirty])
        self.assertEqual(report["verdict"], HOLDOUT_BURNED)

    def test_a_burned_report_is_contract_clean(self):
        report = evaluate_sealed_holdout([run(validation_end=660)])
        self.assertEqual(holdout_problems(report), [])


class OpeningTests(unittest.TestCase):
    """Opening is a one-way event."""

    def test_no_openings_is_not_a_verdict(self):
        # "Never opened" is not "sealed" — the geometry still has to hold.
        self.assertIsNone(opening_verdict([]))
        self.assertIsNone(opening_verdict(None))

    def test_one_named_opening_is_opened(self):
        report = evaluate_sealed_holdout(
            [run()], openings=[{"opened_by": "oren", "reason": "X9 snapshot"}]
        )
        self.assertEqual(report["verdict"], HOLDOUT_OPENED)
        self.assertEqual(holdout_problems(report), [])

    def test_two_openings_burn_it(self):
        report = evaluate_sealed_holdout(
            [run()],
            openings=[
                {"opened_by": "a", "reason": "one"},
                {"opened_by": "b", "reason": "two"},
            ],
        )
        self.assertEqual(report["verdict"], HOLDOUT_BURNED)

    def test_an_anonymous_opening_burns_it(self):
        self.assertTrue(HOLDOUT_OPENING_REQUIRES_REASON)
        for entry in ({"opened_by": "", "reason": "x"}, {"opened_by": "a", "reason": ""}, {}):
            report = evaluate_sealed_holdout([run()], openings=[entry])
            self.assertEqual(report["verdict"], HOLDOUT_BURNED)

    def test_using_it_for_selection_burns_it(self):
        report = evaluate_sealed_holdout(
            [run()],
            openings=[
                {"opened_by": "a", "reason": "tuning", "used_for_selection": True}
            ],
        )
        self.assertEqual(report["verdict"], HOLDOUT_BURNED)

    def test_an_opening_outranks_the_geometry(self):
        # A burned holdout stays burned even where the geometry is unrecorded:
        # the read already happened.
        report = evaluate_sealed_holdout(
            [shipped_run()],
            dataset_rows=406,
            openings=[{"opened_by": "a", "reason": "r"}, {"opened_by": "b", "reason": "r"}],
        )
        self.assertEqual(report["verdict"], HOLDOUT_BURNED)

    def test_the_limit_is_one(self):
        self.assertEqual(HOLDOUT_MAX_OPENINGS, 1)

    def test_burned_is_terminal(self):
        self.assertTrue(HOLDOUT_BURNED_IS_TERMINAL)


class GeometryTests(unittest.TestCase):
    """MEASURED: no legal geometry fits the 406 rows the dataset has."""

    def test_the_shipped_geometry_is_illegal_today(self):
        # embargo 60 against the 252-session horizon F2 introduced.
        problems = geometry_problems(406, 120, 60, 60)
        self.assertTrue(problems)
        self.assertIn("252", " ".join(problems))

    def test_no_legal_geometry_fits_the_shipped_dataset(self):
        for fold in (60, 120, 252):
            for holdout in (60, 126):
                with self.subTest(fold=fold, holdout=holdout):
                    self.assertTrue(geometry_problems(406, fold, 252, holdout))

    def test_the_cheapest_legal_geometry_needs_432_rows(self):
        self.assertEqual(minimum_rows(60, 252, 60), 432)

    def test_the_shortfall_is_named(self):
        problems = geometry_problems(406, 60, 252, 60)
        self.assertIn("short by 26", " ".join(problems))

    def test_a_sufficient_dataset_is_legal(self):
        self.assertEqual(geometry_problems(432, 60, 252, 60), [])

    def test_a_short_embargo_is_refused(self):
        self.assertTrue(geometry_problems(10_000, 60, 251, 60))

    def test_nonpositive_parameters_are_refused(self):
        for args in ((406, 0, 252, 60), (406, 60, 252, 0), (0, 60, 252, 60)):
            with self.subTest(args=args):
                self.assertTrue(geometry_problems(*args))


class ContractTests(unittest.TestCase):
    def test_the_verdicts_run_weakest_to_strongest(self):
        self.assertEqual(HOLDOUT_VERDICTS[0], HOLDOUT_NOT_EVALUATED)
        self.assertEqual(HOLDOUT_VERDICTS[-1], HOLDOUT_SEALED)

    def test_burned_is_not_not_evaluated(self):
        # "The seal was broken" and "the seal could not be established" are
        # different facts with different fixes.
        self.assertNotEqual(HOLDOUT_BURNED, HOLDOUT_NOT_EVALUATED)

    def test_sealed_is_not_opened(self):
        self.assertNotEqual(HOLDOUT_SEALED, HOLDOUT_OPENED)

    def test_x8_reports_and_does_not_block(self):
        self.assertFalse(HOLDOUT_BLOCKS_TRADES)
        self.assertFalse(evaluate_sealed_holdout([run()])["blocks_trades"])

    def test_an_unknown_verdict_is_refused(self):
        with self.assertRaises(SealedHoldoutError):
            from core.sealed_holdout import _result

            _result("MAYBE", "no")

    def test_a_forged_sealed_report_without_bounds_is_caught(self):
        forged = {"verdict": HOLDOUT_SEALED, "reason": "trust me"}
        self.assertTrue(holdout_problems(forged))

    def test_a_forged_sealed_report_with_openings_is_caught(self):
        forged = {
            "verdict": HOLDOUT_SEALED,
            "reason": "r",
            "holdout": [1, 2],
            "geometry": {},
            "dataset_rows": 700,
            "openings": 1,
        }
        self.assertTrue(holdout_problems(forged))

    def test_a_report_without_a_reason_is_caught(self):
        self.assertTrue(holdout_problems({"verdict": HOLDOUT_SEALED, "reason": ""}))

    def test_a_blocking_report_is_caught(self):
        forged = {"verdict": HOLDOUT_SEALED, "reason": "r", "blocks_trades": True}
        self.assertTrue(holdout_problems(forged))

    def test_not_evaluated_without_a_reason_code_is_caught(self):
        forged = {"verdict": HOLDOUT_NOT_EVALUATED, "reason": "r"}
        self.assertTrue(holdout_problems(forged))

    def test_a_non_mapping_report_is_caught(self):
        self.assertTrue(holdout_problems("not a mapping"))


class OpeningLedgerTests(unittest.TestCase):
    def test_an_absent_ledger_means_never_opened(self):
        self.assertEqual(load_openings(None), [])
        self.assertEqual(load_openings("does/not/exist.jsonl"), [])

    def test_a_malformed_ledger_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "openings.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(SealedHoldoutError):
                load_openings(path)

    def test_a_valid_ledger_is_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "openings.jsonl"
            path.write_text(
                json.dumps({"opened_by": "a", "reason": "r"}) + "\n", encoding="utf-8"
            )
            self.assertEqual(len(load_openings(path)), 1)


class RenderTests(unittest.TestCase):
    def test_absent_facts_render_as_absent_not_zero(self):
        text = "\n".join(
            render_sealed_holdout(
                evaluate_sealed_holdout([shipped_run()], dataset_rows=406)
            )
        )
        self.assertIn("ABSENT", text)
        self.assertNotIn("0.0%", text)

    def test_a_sealed_report_renders_its_bounds(self):
        text = "\n".join(render_sealed_holdout(evaluate_sealed_holdout([run()])))
        self.assertIn("640..699", text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_sealed_holdout(evaluate_sealed_holdout([run()]))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))

    def test_the_recorded_fact_count_is_shown(self):
        text = "\n".join(
            render_sealed_holdout(
                evaluate_sealed_holdout([shipped_run()], dataset_rows=406)
            )
        )
        self.assertIn("0 of 4", text)


class ShippedDataTests(unittest.TestCase):
    """Against the tracked ledgers, so this recomputes on a fresh clone."""

    def setUp(self):
        from core.training import load_training_runs

        self.runs = load_training_runs()
        if not self.runs:
            self.skipTest("no training runs on disk")
        datasets = pathlib.Path("data/training_datasets.jsonl")
        if not datasets.exists():
            self.skipTest("no dataset ledger on disk")
        self.rows = json.loads(
            datasets.read_text(encoding="utf-8").splitlines()[0]
        )["row_count"]

    def test_the_dataset_is_406_rows(self):
        self.assertEqual(self.rows, 406)

    def test_the_regenerated_runs_record_a_seal(self):
        """RESTATED after A1. The blocker was that NO run recorded any seal fact.

        A1 added the four facts to `TrainingRun` and the ledger was regenerated,
        so the runs carrying a `dataset_rows` must carry all four — a partial
        record cannot be audited, which was the original complaint.
        """
        sealed = [run for run in self.runs if run.get("dataset_rows")]
        self.assertTrue(
            sealed,
            "no run records a dataset row count; A1's seal facts have been lost",
        )
        for run_record in sealed:
            with self.subTest(estimator=run_record["estimator"]):
                facts = recorded_facts(run_record)
                self.assertEqual([f for f, ok in facts.items() if not ok], [])

    def test_the_shipped_verdict_is_sealed(self):
        """RESTATED after A1/A2: the seal is recorded and the geometry embargoes it.

        NO `dataset_rows` OVERRIDE. The run records its own count — that is what
        A1 added it for — and the dataset ledger's first line describes the OLD
        406-row dataset, which would make a 1,464-row run's folds look like they
        reached into a 406-row holdout.
        """
        sealed = [run for run in self.runs if run.get("dataset_rows")]
        if not sealed:
            self.skipTest("no run records a seal")
        report = evaluate_sealed_holdout(sealed)
        self.assertEqual(report["verdict"], HOLDOUT_SEALED)
        self.assertIsNotNone(report.get("holdout"))
        self.assertEqual(holdout_problems(report), [])

    def test_the_shipped_dataset_admits_no_legal_geometry(self):
        for fold in (60, 120, 252):
            for holdout in (60, 126):
                with self.subTest(fold=fold, holdout=holdout):
                    self.assertTrue(geometry_problems(self.rows, fold, 252, holdout))


if __name__ == "__main__":
    unittest.main()
