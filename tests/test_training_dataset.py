"""Official training dataset builder tests (Sprint M2).

The M2 acceptance criteria, pinned:

- every row is prediction_time -> information available at prediction_time ->
  features -> future outcome;
- the dataset is deterministically hashed;
- an unregistered feature cannot enter a dataset (M1 gate applies here too);
- future information in a row is rejected — leakage is caught, not assumed
  absent;
- an unmatured horizon is an excluded row with a reason, never a zero.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.config import (
    MARKET_FEATURE_VERSION,
    OUTCOME_LABEL_VERSION,
    TRAINING_DATASET_VERSION,
)
from core.feature_registry import build_default_registry
from core.training_dataset import (
    TrainingDatasetError,
    TrainingRow,
    build_training_dataset,
    build_training_row,
    dataset_hash,
    load_training_datasets,
    persist_training_dataset,
    row_problems,
)

_FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "data" / "NVDA_5y_1d.parquet"


def _frame() -> pd.DataFrame:
    frame = pd.read_parquet(_FIXTURE)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _times(frame: pd.DataFrame, start: int, stop: int) -> list[str]:
    return [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[start:stop]]


def _contract(name: str, value: float, as_of: str, published: str | None = None) -> dict:
    """A registry-conformant market feature contract."""
    return {
        "name": name,
        "value": value,
        "as_of": as_of,
        "source_id": "yahoo_finance_chart",
        "published_time": published or as_of,
        "calculation_version": MARKET_FEATURE_VERSION,
        "lookback_period": "1d" if name == "change_1d" else "20d",
    }


def _label_set(record_hash: str = "abc123", status: str = "OK") -> dict:
    return {
        "ticker": "NVDA",
        "label_version": OUTCOME_LABEL_VERSION,
        "horizons": {
            "20d": {
                "status": status,
                "forward_return": 0.042,
                "label_up": True,
                "realized_vol": 0.018,
                "adverse_excursion": -0.031,
                "record_hash": record_hash,
            }
        },
    }


class TestRowShape(unittest.TestCase):
    """prediction_time -> features -> future outcome."""

    def test_row_carries_features_and_future_outcome(self) -> None:
        surface = {"change_1d": _contract("change_1d", 0.01, "2026-01-05 00:00:00")}
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        self.assertEqual(reason, "")
        self.assertEqual(row.prediction_time, "2026-01-05 00:00:00")
        self.assertEqual(row.features["change_1d"], 0.01)
        self.assertEqual(row.forward_return, 0.042)
        self.assertTrue(row.label_up)
        self.assertEqual(row.label_record_hash, "abc123")

    def test_missing_feature_excludes_the_row(self) -> None:
        """No imputation, no neutral fill."""
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", {}, _label_set(), ["change_1d"]
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "feature_contract_missing")

    def test_null_feature_value_excludes_the_row(self) -> None:
        surface = {"change_1d": _contract("change_1d", None, "2026-01-05 00:00:00")}
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "feature_value_missing")

    def test_nan_feature_value_excludes_the_row(self) -> None:
        surface = {"change_1d": _contract("change_1d", float("nan"), "2026-01-05 00:00:00")}
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "feature_value_missing")

    def test_unavailable_label_excludes_the_row(self) -> None:
        surface = {"change_1d": _contract("change_1d", 0.01, "2026-01-05 00:00:00")}
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(status="UNAVAILABLE"), ["change_1d"]
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "label_unavailable")

    def test_missing_horizon_excludes_the_row(self) -> None:
        """An unmatured horizon is an excluded row, never a zero."""
        surface = {"change_1d": _contract("change_1d", 0.01, "2026-01-05 00:00:00")}
        empty = {"label_version": OUTCOME_LABEL_VERSION, "horizons": {}}
        row, reason = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, empty, ["change_1d"]
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "label_horizon_missing")


class TestLeakageRejection(unittest.TestCase):
    def test_future_published_time_is_rejected(self) -> None:
        """A feature published after the prediction is future information."""
        surface = {
            "change_1d": _contract(
                "change_1d", 0.01, "2026-01-05 00:00:00", published="2026-01-09 00:00:00"
            )
        }
        row, _ = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        problems = row_problems(row)
        self.assertTrue(
            any("future information in a training row" in p for p in problems),
            problems,
        )

    def test_same_instant_publication_is_allowed(self) -> None:
        surface = {
            "change_1d": _contract(
                "change_1d", 0.01, "2026-01-05 00:00:00", published="2026-01-05 00:00:00"
            )
        }
        row, _ = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        self.assertEqual(row_problems(row), [])

    def test_unregistered_feature_cannot_enter_a_dataset(self) -> None:
        """The M1 gate applies to datasets, not just models."""
        with self.assertRaises(TrainingDatasetError) as ctx:
            build_training_dataset({}, {}, feature_names=["totally_made_up_feature"])
        self.assertIn("not registered", str(ctx.exception))

    def test_empty_feature_set_is_refused(self) -> None:
        with self.assertRaises(TrainingDatasetError):
            build_training_dataset({}, {}, feature_names=[])

    def test_version_drifted_contract_is_rejected(self) -> None:
        surface = {"change_1d": _contract("change_1d", 0.01, "2026-01-05 00:00:00")}
        surface["change_1d"]["calculation_version"] = "market-feature-v0"
        row, _ = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        self.assertTrue(row_problems(row))

    def test_unknown_horizon_is_refused(self) -> None:
        with self.assertRaises(TrainingDatasetError):
            build_training_dataset({}, {}, target_horizon="999d")

    def test_wrong_label_version_is_rejected(self) -> None:
        surface = {"change_1d": _contract("change_1d", 0.01, "2026-01-05 00:00:00")}
        stale = _label_set()
        stale["label_version"] = "outcome-label-v0"
        row, _ = build_training_row(
            "NVDA", "2026-01-05 00:00:00", surface, stale, ["change_1d"]
        )
        self.assertTrue(any("label_version" in p for p in row_problems(row)))


class TestDeterministicHash(unittest.TestCase):
    def _row(self, value: float = 0.01, ticker: str = "NVDA") -> TrainingRow:
        surface = {"change_1d": _contract("change_1d", value, "2026-01-05 00:00:00")}
        row, _ = build_training_row(
            ticker, "2026-01-05 00:00:00", surface, _label_set(), ["change_1d"]
        )
        return row

    def test_same_rows_hash_identically(self) -> None:
        self.assertEqual(
            dataset_hash([self._row()], "fsh", "20d"),
            dataset_hash([self._row()], "fsh", "20d"),
        )

    def test_row_order_does_not_change_the_hash(self) -> None:
        a, b = self._row(ticker="AAA"), self._row(ticker="BBB")
        self.assertEqual(
            dataset_hash([a, b], "fsh", "20d"), dataset_hash([b, a], "fsh", "20d")
        )

    def test_different_feature_value_changes_the_hash(self) -> None:
        self.assertNotEqual(
            dataset_hash([self._row(0.01)], "fsh", "20d"),
            dataset_hash([self._row(0.02)], "fsh", "20d"),
        )

    def test_different_feature_set_changes_the_hash(self) -> None:
        self.assertNotEqual(
            dataset_hash([self._row()], "fsh_a", "20d"),
            dataset_hash([self._row()], "fsh_b", "20d"),
        )

    def test_different_target_horizon_changes_the_hash(self) -> None:
        self.assertNotEqual(
            dataset_hash([self._row()], "fsh", "20d"),
            dataset_hash([self._row()], "fsh", "5d"),
        )

    def test_contract_provenance_is_part_of_identity(self) -> None:
        """Same number, different source, is different training data."""
        base = self._row()
        moved = self._row()
        moved.feature_contracts["change_1d"]["source_id"] = "some_other_source"
        self.assertNotEqual(
            dataset_hash([base], "fsh", "20d"), dataset_hash([moved], "fsh", "20d")
        )


class TestEndToEndBuild(unittest.TestCase):
    """Against the real cached frame, through the live scoring path."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _FIXTURE.exists():
            raise unittest.SkipTest(f"fixture {_FIXTURE.name} not available")
        cls.frame = _frame()
        cls.times = _times(cls.frame, -400, -385)
        cls.dataset = build_training_dataset(
            {"NVDA": cls.times}, {"NVDA": cls.frame}
        )

    def test_builds_rows_from_the_live_scoring_path(self) -> None:
        self.assertEqual(len(self.dataset), len(self.times))
        self.assertEqual(self.dataset.excluded, [])

    def test_every_row_is_registry_conformant(self) -> None:
        registry = build_default_registry()
        for row in self.dataset.rows:
            with self.subTest(prediction_time=row.prediction_time):
                self.assertEqual(row_problems(row, registry), [])

    def test_rebuild_reproduces_the_hash(self) -> None:
        """Deterministic: same request, same dataset identity."""
        rebuilt = build_training_dataset({"NVDA": self.times}, {"NVDA": self.frame})
        self.assertEqual(rebuilt.dataset_hash, self.dataset.dataset_hash)

    def test_rows_are_sorted_by_prediction_time(self) -> None:
        stamps = [row.prediction_time for row in self.dataset.rows]
        self.assertEqual(stamps, sorted(stamps))

    def test_no_feature_is_published_after_its_prediction_time(self) -> None:
        for row in self.dataset.rows:
            prediction = pd.Timestamp(row.prediction_time)
            for name, contract in row.feature_contracts.items():
                with self.subTest(prediction_time=row.prediction_time, feature=name):
                    self.assertLessEqual(pd.Timestamp(contract["published_time"]), prediction)

    def test_outcome_is_strictly_forward(self) -> None:
        """The label's entry bar is at or before the prediction instant."""
        for row in self.dataset.rows:
            with self.subTest(prediction_time=row.prediction_time):
                self.assertIsNotNone(row.forward_return)
                self.assertNotEqual(row.label_record_hash, "")

    def test_unmatured_horizon_rows_are_excluded_with_a_reason(self) -> None:
        tail = _times(self.frame, -8, None)
        dataset = build_training_dataset({"NVDA": tail}, {"NVDA": self.frame})
        self.assertEqual(len(dataset), 0)
        self.assertEqual(len(dataset.excluded), len(tail))
        self.assertEqual(
            dataset.report()["exclusion_reasons"], {"label_horizon_missing": len(tail)}
        )

    def test_versions_are_stamped(self) -> None:
        versions = self.dataset.versions
        self.assertEqual(versions["dataset"], TRAINING_DATASET_VERSION)
        self.assertEqual(versions["outcome_label"], OUTCOME_LABEL_VERSION)
        self.assertEqual(versions["market_feature"], MARKET_FEATURE_VERSION)


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name) / "training_datasets.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def _dataset(self):
        if not _FIXTURE.exists():
            self.skipTest("fixture not available")
        frame = _frame()
        return build_training_dataset(
            {"NVDA": _times(frame, -400, -396)}, {"NVDA": frame}
        )

    def test_persist_is_idempotent_per_dataset_hash(self) -> None:
        dataset = self._dataset()
        persist_training_dataset(dataset, self.path)
        persist_training_dataset(dataset, self.path)
        self.assertEqual(len(load_training_datasets(self.path)), 1)

    def test_manifest_carries_provenance_not_rows(self) -> None:
        dataset = self._dataset()
        record = persist_training_dataset(dataset, self.path)
        self.assertIn("dataset_hash", record)
        self.assertIn("feature_set_hash", record)
        self.assertIn("row_count", record)
        self.assertNotIn("rows", record)

    def test_dataset_without_hash_is_refused(self) -> None:
        dataset = self._dataset()
        dataset.dataset_hash = ""
        with self.assertRaises(TrainingDatasetError):
            persist_training_dataset(dataset, self.path)

    def test_missing_store_reads_empty(self) -> None:
        self.assertEqual(load_training_datasets(self.path), [])

    def test_malformed_line_raises_loudly(self) -> None:
        self.path.write_text("{not json\n", encoding="utf-8")
        with self.assertRaises(TrainingDatasetError):
            load_training_datasets(self.path)


if __name__ == "__main__":
    unittest.main()
