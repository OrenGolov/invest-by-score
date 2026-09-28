"""A1 tests — the training fold records the context the gates need.

X3 could not compare per-regime performance, X4 had nothing to leave out, and X8
could not locate the seal. None of those were estimator problems: the information
was never recorded. These tests pin the recording, its alignment, and the two
invariants A1 must not break — the dataset hash and the artifact hash.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.event_robustness import (
    EVENT_ROBUSTNESS_AXIS_EVENT,
    EVENT_ROBUSTNESS_AXIS_SOURCE,
    attribution_of,
    evaluate_event_robustness,
)
from core.feature_registry import build_default_registry, feature_set_hash
from core.regime_robustness import evaluate_regime_robustness, fold_regime_labels
from core.sealed_holdout import (
    HOLDOUT_SEALED,
    evaluate_sealed_holdout,
    recorded_facts,
)
from core.training import FoldResult, train_baseline
from core.training_dataset import (
    TrainingDataset,
    TrainingRow,
    context_snapshot_of,
    dataset_hash,
    observed_event,
    observed_regime,
    observed_source,
)

REGIMES = ("bullish", "bearish", "range")


def row(index: int, *, regime=None, event=None, source=None, rng=None):
    rng = rng or np.random.default_rng(index)
    return TrainingRow(
        ticker="AAA",
        prediction_time=f"t{index:05d}",
        features={
            "rsi": float(rng.normal(50, 10)),
            "volatility": float(abs(rng.normal(0.01, 0.003))),
        },
        feature_contracts={
            "rsi": {"calculation_version": "v1"},
            "volatility": {"calculation_version": "v1"},
        },
        target_horizon="20d",
        forward_return=float(rng.normal(0, 0.02)),
        label_up=True,
        realized_vol=0.1,
        adverse_excursion=-0.01,
        label_version="lv1",
        label_record_hash=f"h{index}",
        regime=regime,
        event_id=event,
        source_id=source,
    )


def dataset(count=1500, *, with_context=True):
    rng = np.random.default_rng(3)
    rows = [
        row(
            index,
            regime=REGIMES[index % 3] if with_context else None,
            event=f"ev{index % 7}" if with_context else None,
            source=f"src{index % 4}" if with_context else None,
            rng=rng,
        )
        for index in range(count)
    ]
    registry = build_default_registry()
    digest = feature_set_hash(["rsi", "volatility"], registry)
    return (
        TrainingDataset(
            rows=rows,
            feature_names=["rsi", "volatility"],
            target_horizon="20d",
            dataset_hash=dataset_hash(rows, digest, "20d"),
            feature_set_hash=digest,
        ),
        registry,
    )


def trained(count=1500, *, with_context=True, fold=300, holdout=60):
    data, registry = dataset(count, with_context=with_context)
    return train_baseline(
        data,
        estimator="ridge",
        fold_sessions=fold,
        embargo_sessions=252,
        holdout_sessions=holdout,
        registry=registry,
    )


class ContextIsNotIdentityTests(unittest.TestCase):
    """The invariant A1 must not break: the dataset hash cannot move."""

    def test_two_rows_differing_only_in_context_hash_alike(self):
        # M8 keys every model artifact to a dataset hash. Context describes a
        # row's conditions; it does not define which example the row IS.
        plain = row(1)
        contextual = row(1, regime="bullish", event="ev1", source="reuters")
        self.assertEqual(plain.identity(), contextual.identity())

    def test_the_dataset_hash_is_unchanged_by_context(self):
        digest = feature_set_hash(["rsi", "volatility"], build_default_registry())
        plain = [row(i) for i in range(20)]
        contextual = [
            row(i, regime=REGIMES[i % 3], event=f"ev{i}", source="s") for i in range(20)
        ]
        self.assertEqual(
            dataset_hash(plain, digest, "20d"),
            dataset_hash(contextual, digest, "20d"),
        )

    def test_context_is_still_exposed(self):
        # Excluded from identity, but not hidden: a consumer must be able to ask.
        contextual = row(1, regime="bullish", event="ev1", source="reuters")
        self.assertEqual(
            contextual.context(),
            {"regime": "bullish", "event_id": "ev1", "source_id": "reuters"},
        )

    def test_an_uncontextual_row_reports_none_not_a_default(self):
        self.assertEqual(
            row(1).context(), {"regime": None, "event_id": None, "source_id": None}
        )


class ExtractorsRefuseToInventTests(unittest.TestCase):
    """None means NOT OBSERVED. A synthetic bucket would defeat the gates."""

    def test_a_classified_regime_is_recorded(self):
        snapshot = {"regime": {"regime": "bullish"}}
        self.assertEqual(observed_regime(snapshot), "bullish")

    def test_a_missing_regime_is_none(self):
        for snapshot in ({}, {"regime": {}}, {"regime": {"regime": ""}}, None):
            with self.subTest(snapshot=snapshot):
                self.assertIsNone(observed_regime(snapshot))

    def test_a_failed_regime_snapshot_is_none(self):
        snapshot = {"regime": {"regime": "bullish", "status": "UNAVAILABLE"}}
        self.assertIsNone(observed_regime(snapshot))

    def test_an_unavailable_news_provider_yields_no_event(self):
        # THE DISTINCTION THAT MATTERS: "the provider failed" is ignorance,
        # "there was no news" is evidence. Recording the first as the second
        # would put every outage into a single bucket X4 then treats as a real
        # event group.
        snapshot = {
            "news": {"status": "UNAVAILABLE", "source_id": "newsapi_news", "articles": []}
        }
        self.assertIsNone(observed_event(snapshot))
        self.assertIsNone(observed_source(snapshot))

    def test_an_ok_snapshot_with_no_articles_yields_no_event(self):
        snapshot = {"news": {"status": "OK", "articles": []}}
        self.assertIsNone(observed_event(snapshot))

    def test_an_event_id_is_taken_from_the_article(self):
        snapshot = {
            "news": {
                "status": "OK",
                "articles": [{"event_id": "ev-42", "source_id": "reuters"}],
            }
        }
        self.assertEqual(observed_event(snapshot), "ev-42")
        self.assertEqual(observed_source(snapshot), "reuters")

    def test_a_nested_source_object_is_resolved(self):
        snapshot = {
            "news": {
                "status": "OK",
                "articles": [{"title": "t", "source": {"id": "bbc", "name": "BBC"}}],
            }
        }
        self.assertEqual(observed_source(snapshot), "bbc")

    def test_the_snapshot_is_read_from_a_score_result(self):
        result = {
            "market_regime_snapshot": {"regime": "bearish"},
            "news_snapshot": {"status": "OK", "articles": [{"id": "x"}]},
            "sentiment_snapshot": {},
        }
        snapshot = context_snapshot_of(result)
        self.assertEqual(observed_regime(snapshot), "bearish")
        self.assertEqual(observed_event(snapshot), "x")

    def test_a_non_dict_score_result_does_not_raise(self):
        self.assertEqual(
            context_snapshot_of(None), {"regime": {}, "news": {}, "sentiment": {}}
        )


class FoldAlignmentTests(unittest.TestCase):
    """A misaligned context slice reads as evidence while describing the wrong rows."""

    def test_a_short_context_list_is_caught(self):
        fold = FoldResult(
            fold_id=0,
            train_rows=2,
            validation_rows=2,
            train_end_time="a",
            validation_start_time="b",
            predictions=[0.1, 0.2],
            actuals=[0.1, 0.2],
            regimes=["bullish"],
        )
        self.assertTrue(fold.context_problems())

    def test_an_aligned_context_list_is_clean(self):
        fold = FoldResult(
            fold_id=0,
            train_rows=2,
            validation_rows=2,
            train_end_time="a",
            validation_start_time="b",
            predictions=[0.1, 0.2],
            actuals=[0.1, 0.2],
            regimes=["bullish", "bearish"],
            events=["e1", "e2"],
            sources=["s1", "s2"],
        )
        self.assertEqual(fold.context_problems(), [])

    def test_absent_context_is_clean(self):
        # An empty list means "not recorded", which the gates report as
        # NOT_EVALUATED. That is a legitimate state, not a contract breach.
        fold = FoldResult(
            fold_id=0,
            train_rows=2,
            validation_rows=2,
            train_end_time="a",
            validation_start_time="b",
            predictions=[0.1, 0.2],
            actuals=[0.1, 0.2],
        )
        self.assertEqual(fold.context_problems(), [])

    def test_a_backwards_window_is_caught(self):
        fold = FoldResult(
            fold_id=0,
            train_rows=1,
            validation_rows=1,
            train_end_time="a",
            validation_start_time="b",
            validation=[500, 100],
        )
        self.assertTrue(fold.context_problems())

    def test_a_malformed_window_is_caught(self):
        fold = FoldResult(
            fold_id=0,
            train_rows=1,
            validation_rows=1,
            train_end_time="a",
            validation_start_time="b",
            train=[1, 2, 3],
        )
        self.assertTrue(fold.context_problems())


class TrainedRunCarriesContextTests(unittest.TestCase):
    """End to end: train, and the fold has what the gates read."""

    @classmethod
    def setUpClass(cls):
        cls.training_run = trained()
        cls.fold = cls.training_run.folds[0].to_dict()

    def test_the_fold_records_absolute_indices(self):
        # `train_rows` says HOW MANY; only position proves a fold stopped short
        # of the sealed tail.
        self.assertEqual(len(self.fold["validation"]), 2)
        self.assertEqual(len(self.fold["train"]), 2)
        self.assertLess(self.fold["train"][1], self.fold["validation"][0])

    def test_context_is_aligned_with_predictions(self):
        width = len(self.fold["predictions"])
        for key in ("regimes", "events", "sources"):
            with self.subTest(key=key):
                self.assertEqual(len(self.fold[key]), width)

    def test_every_regime_is_represented(self):
        self.assertEqual(set(self.fold["regimes"]), set(REGIMES))

    def test_the_run_records_the_seal(self):
        self.assertEqual(self.training_run.dataset_rows, 1500)
        self.assertEqual(
            sorted(self.training_run.geometry),
            ["embargo_sessions", "fold_sessions", "holdout_sessions"],
        )
        self.assertEqual(len(self.training_run.holdout), 2)

    def test_all_four_seal_facts_are_recorded(self):
        facts = recorded_facts(self.training_run.to_dict())
        self.assertEqual([name for name, ok in facts.items() if not ok], [])

    def test_the_artifact_hash_does_not_depend_on_context(self):
        # The artifact identifies the FITTED MODEL. Recording where the holdout
        # sat did not change which model was fitted.
        without = trained(with_context=False)
        self.assertEqual(self.training_run.artifact_hash, without.artifact_hash)


class TheGatesCanNowRunTests(unittest.TestCase):
    """A1's whole purpose. Each of these was NOT_EVALUATED before."""

    @classmethod
    def setUpClass(cls):
        cls.training_run = trained()
        cls.fold = cls.training_run.folds[0].to_dict()

    def test_x3_regime_robustness_is_evaluable(self):
        report = evaluate_regime_robustness(
            self.fold["predictions"], self.fold["actuals"], fold_regime_labels(self.fold)
        )
        self.assertNotEqual(report["verdict"], "NOT_EVALUATED")

    def test_x4_event_robustness_is_evaluable(self):
        report = evaluate_event_robustness(
            self.fold["predictions"],
            self.fold["actuals"],
            attribution_of(self.fold, EVENT_ROBUSTNESS_AXIS_EVENT),
            attribution_of(self.fold, EVENT_ROBUSTNESS_AXIS_SOURCE),
        )
        self.assertNotEqual(report["verdict"], "NOT_EVALUATED")

    def test_x8_reaches_sealed(self):
        report = evaluate_sealed_holdout([self.training_run.to_dict()])
        self.assertEqual(report["verdict"], HOLDOUT_SEALED)

    def test_a_run_without_context_still_reports_not_evaluated(self):
        # The gates must not be fooled into ROBUST by empty lists.
        bare = trained(with_context=False).folds[0].to_dict()
        self.assertIsNone(fold_regime_labels(bare))
        self.assertIsNone(attribution_of(bare, EVENT_ROBUSTNESS_AXIS_EVENT))


class TheGapIsBoundedByTheFoldTests(unittest.TestCase):
    """MEASURED while building A1, and it constrains A2.

    The walk-forward loop marches folds forward until one would touch the
    holdout, so the gap left to the tail is a remainder. Its maximum is
    `fold_sessions - 4`, which means a holdout embargoed against the
    252-session horizon REQUIRES `fold_sessions > 252`. A first reading of this
    concluded the gap could never cover the horizon; that was wrong — 327
    combinations reach it — but only above that fold size.
    """

    @staticmethod
    def last_validation_end(rows, fold, embargo, holdout):
        holdout_start = rows - holdout
        fold_id, last = 0, None
        while True:
            train_end = (fold_id + 1) * fold - 1
            start = train_end + embargo + 1
            end = start + fold - 1
            if end >= holdout_start:
                return last
            last = end
            fold_id += 1

    def test_the_gap_cannot_exceed_the_fold_size(self):
        for fold in (60, 120, 252, 300):
            widest = 0
            for holdout in (60, 126, 252):
                for rows in range(600, 3001, 20):
                    end = self.last_validation_end(rows, fold, 252, holdout)
                    if end is not None:
                        widest = max(widest, rows - holdout - end - 1)
            with self.subTest(fold=fold):
                self.assertLess(widest, fold)

    def test_a_fold_below_the_horizon_can_never_embargo_the_holdout(self):
        for fold in (60, 120, 200, 252):
            for holdout in (60, 126):
                for rows in range(600, 2001, 20):
                    end = self.last_validation_end(rows, fold, 252, holdout)
                    if end is None:
                        continue
                    with self.subTest(fold=fold, holdout=holdout, rows=rows):
                        self.assertLess(rows - holdout - end - 1, 252)

    def test_a_fold_above_the_horizon_can(self):
        end = self.last_validation_end(1500, 300, 252, 60)
        self.assertIsNotNone(end)
        self.assertGreaterEqual(1500 - 60 - end - 1, 252)


if __name__ == "__main__":
    unittest.main()
