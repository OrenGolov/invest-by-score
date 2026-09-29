"""B2 tests — a price outcome attributed to an OUTLET, so L4 can learn.

Open item 4 parked this as unbuildable: "0 of 2650 stored articles carry a ticker,
so no article joins to a price outcome." That measurement was correct and the
conclusion was not: every raw envelope carries a `request_key` naming the ticker
the fetch was made FOR, so all 3,753 articles are attributable from data already
tracked.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from core.config import (
    SOURCE_OUTCOME_IS_ASSOCIATION_ONLY,
    SOURCE_OUTCOME_MIN_TONE_ABS,
)
from core.source_outcome_join import (
    ATTRIBUTION_ENTITY_RESOLUTION,
    ATTRIBUTION_REQUEST_KEY,
    DROPPED_NO_OUTLET,
    DROPPED_NO_PRICE,
    DROPPED_NO_TICKER,
    DROPPED_NO_TONE,
    SourceOutcomeJoinError,
    build_observations,
    join_problems,
    load_news_envelopes,
    render_join,
    ticker_from_request_key,
)
from core.source_reliability import observation_records, score_source


def envelope(request_key="MSFT_2026-09-17", records=None):
    return {"request_key": request_key, "records": records or []}


def article(
    *,
    outlet="Reuters",
    tone=0.8,
    headline="Revenue beats expectations and the stock surges",
    published="2026-09-17 12:00:00",
    ticker=None,
):
    return {
        "source_name": outlet,
        "tone": tone,
        "headline": headline,
        "summary": "",
        "published_time": published,
        "ticker": ticker,
    }


def rising(_ticker, _published, _horizon):
    return 0.03


def falling(_ticker, _published, _horizon):
    return -0.03


def unavailable(_ticker, _published, _horizon):
    return None


class TheTickerIsRecoverableTests(unittest.TestCase):
    """The correction that unparked this: the request key carries it."""

    def test_a_request_key_yields_its_ticker(self):
        self.assertEqual(ticker_from_request_key("MSFT_2026-09-17"), "MSFT")

    def test_a_lowercase_key_is_normalised(self):
        self.assertEqual(ticker_from_request_key("msft_2026-09-17"), "MSFT")

    def test_a_key_without_a_ticker_yields_none(self):
        # A wrong attribution is worse than a missing one: it would credit an
        # outlet for a move in another company.
        for key in ("", None, "no-underscore", "_2026-09-17", "123_2026-09-17"):
            with self.subTest(key=key):
                self.assertIsNone(ticker_from_request_key(key))

    def test_an_implausibly_long_prefix_is_refused(self):
        self.assertIsNone(ticker_from_request_key("NOTATICKER_2026-09-17"))

    def test_an_article_level_ticker_wins(self):
        # Resolved from the text, so stronger than inferred from the request.
        report = build_observations(
            [envelope(records=[article(ticker="AAPL")])], rising
        )
        self.assertEqual(report["observations"][0]["ticker"], "AAPL")
        self.assertEqual(
            report["observations"][0]["attribution"], ATTRIBUTION_ENTITY_RESOLUTION
        )

    def test_the_request_key_is_the_fallback_and_is_labelled(self):
        report = build_observations([envelope(records=[article()])], rising)
        self.assertEqual(report["observations"][0]["ticker"], "MSFT")
        self.assertEqual(
            report["observations"][0]["attribution"], ATTRIBUTION_REQUEST_KEY
        )

    def test_an_unattributable_article_is_dropped(self):
        report = build_observations(
            [envelope(request_key="", records=[article()])], rising
        )
        self.assertEqual(report["joined"], 0)
        self.assertEqual(report["dropped"][DROPPED_NO_TICKER], 1)


class AHitIsAgreementTests(unittest.TestCase):
    """Defined here so nothing downstream has to guess."""

    def test_a_positive_claim_before_a_rise_is_a_hit(self):
        report = build_observations([envelope(records=[article(tone=0.8)])], rising)
        self.assertEqual(report["observations"][0]["hit"], 1)

    def test_a_positive_claim_before_a_fall_is_a_miss(self):
        report = build_observations([envelope(records=[article(tone=0.8)])], falling)
        self.assertEqual(report["observations"][0]["hit"], 0)

    def test_a_negative_claim_before_a_fall_is_a_hit(self):
        report = build_observations([envelope(records=[article(tone=-0.8)])], falling)
        self.assertEqual(report["observations"][0]["hit"], 1)

    def test_the_forward_return_is_recorded(self):
        report = build_observations([envelope(records=[article()])], rising)
        self.assertAlmostEqual(report["observations"][0]["forward_return"], 0.03)

    def test_association_only_is_declared(self):
        # The join reports what FOLLOWED publication. E5 owns any methodology
        # that could claim causation.
        self.assertTrue(SOURCE_OUTCOME_IS_ASSOCIATION_ONLY)


class ANeutralArticleMakesNoClaimTests(unittest.TestCase):
    """Scoring it would penalise an outlet for a missing classification."""

    def test_a_tone_inside_the_band_is_dropped(self):
        report = build_observations(
            [envelope(records=[article(tone=SOURCE_OUTCOME_MIN_TONE_ABS / 2)])],
            rising,
        )
        self.assertEqual(report["joined"], 0)
        self.assertEqual(report["dropped"][DROPPED_NO_TONE], 1)

    def test_a_neutral_article_is_not_counted_as_a_miss(self):
        report = build_observations(
            [envelope(records=[article(tone=0.0)])], falling
        )
        self.assertEqual(report["observations"], [])

    def test_an_absent_tone_is_derived_from_the_headline(self):
        # MEASURED: tone is None for all 3,753 stored articles. The missing link
        # was the CLASSIFICATION, not the ticker.
        report = build_observations(
            [
                envelope(
                    records=[
                        article(tone=None, headline="Shares plunge after earnings miss")
                    ]
                )
            ],
            falling,
        )
        self.assertEqual(report["joined"], 1)
        self.assertEqual(report["observations"][0]["tone_direction"], -1)

    def test_the_derivation_is_stamped(self):
        # A lexicon tone and a provider tone are not equally trustworthy, and
        # MEASURED the v1 lexicon agreed with the obvious reading on 5 of 6
        # hand probes.
        report = build_observations(
            [envelope(records=[article(tone=None, headline="Stock tumbles badly")])],
            falling,
        )
        self.assertIn(
            "lexicon", report["observations"][0]["tone_derivation"]
        )

    def test_the_derivation_mix_is_reported(self):
        report = build_observations([envelope(records=[article()])], rising)
        self.assertTrue(report["tone_derivation_mix"])


class LossesAreCountedTests(unittest.TestCase):
    """A join that silently loses rows cannot be told from one with no evidence."""

    def test_a_missing_outlet_is_counted(self):
        report = build_observations(
            [envelope(records=[article(outlet="")])], rising
        )
        self.assertEqual(report["dropped"][DROPPED_NO_OUTLET], 1)

    def test_an_unmatured_outcome_is_counted(self):
        # The legitimate, common case: news published after the last price bar.
        report = build_observations([envelope(records=[article()])], unavailable)
        self.assertEqual(report["dropped"][DROPPED_NO_PRICE], 1)

    def test_the_join_rate_is_reported(self):
        report = build_observations(
            [envelope(records=[article(), article(outlet="")])], rising
        )
        self.assertAlmostEqual(report["join_rate"], 0.5)

    def test_an_empty_input_reports_no_rate(self):
        # ABSENT, not 0.0: a rate of zero claims articles were consumed.
        self.assertIsNone(build_observations([], rising)["join_rate"])


class ContractTests(unittest.TestCase):
    def test_a_clean_report_has_no_problems(self):
        report = build_observations([envelope(records=[article()])], rising)
        self.assertEqual(join_problems(report), [])

    def test_an_unknown_horizon_is_refused(self):
        with self.assertRaises(SourceOutcomeJoinError):
            build_observations([], rising, horizon="7d")

    def test_a_non_callable_outcome_source_is_refused(self):
        with self.assertRaises(SourceOutcomeJoinError):
            build_observations([], None)

    def test_a_forged_observation_without_an_outlet_is_caught(self):
        self.assertTrue(
            join_problems({"observations": [{"hit": 1, "attribution": "request_key"}]})
        )

    def test_a_forged_hit_is_caught(self):
        self.assertTrue(
            join_problems(
                {
                    "observations": [
                        {
                            "source": "X",
                            "hit": 0.5,
                            "attribution": "request_key",
                            "forward_return": 0.01,
                        }
                    ]
                }
            )
        )

    def test_an_undeclared_attribution_is_caught(self):
        self.assertTrue(
            join_problems(
                {
                    "observations": [
                        {
                            "source": "X",
                            "hit": 1,
                            "attribution": "guessed",
                            "forward_return": 0.01,
                        }
                    ]
                }
            )
        )

    def test_a_non_mapping_report_is_caught(self):
        self.assertTrue(join_problems("not a mapping"))

    def test_render_returns_lines(self):
        lines = render_join(build_observations([envelope(records=[article()])], rising))
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


class LoadingTests(unittest.TestCase):
    def test_an_absent_directory_is_empty(self):
        self.assertEqual(load_news_envelopes("does/not/exist"), [])

    def test_a_malformed_line_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "a.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(SourceOutcomeJoinError):
                load_news_envelopes(directory)

    def test_valid_envelopes_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "a.jsonl"
            path.write_text(json.dumps(envelope()) + "\n", encoding="utf-8")
            self.assertEqual(len(load_news_envelopes(directory)), 1)


class L4CanNowLearnTests(unittest.TestCase):
    """The point of B2. Every L4 score was previously a registry prior."""

    def setUp(self):
        records = [
            article(outlet="Reuters", tone=0.8),
            article(outlet="Reuters", tone=0.8),
            article(outlet="Bloomberg", tone=-0.8),
        ]
        self.report = build_observations([envelope(records=records)], rising)

    def test_the_observations_satisfy_l4s_contract(self):
        accepted = observation_records(self.report["observations"])
        self.assertEqual(len(accepted), len(self.report["observations"]))

    def test_l4_produces_a_score_from_them(self):
        scored = score_source(self.report["observations"], "Reuters")
        self.assertIsNotNone(scored.get("score"))

    def test_outlets_are_scored_separately(self):
        # Reuters called it right twice, Bloomberg called it wrong once.
        good = score_source(self.report["observations"], "Reuters")["score"]
        bad = score_source(self.report["observations"], "Bloomberg")["score"]
        self.assertGreater(good, bad)


class ShippedDataTests(unittest.TestCase):
    """Against the tracked raw store, so this recomputes on a fresh clone."""

    def setUp(self):
        self.envelopes = load_news_envelopes("data/raw/newsapi_news")
        if not self.envelopes:
            self.skipTest("no raw news envelopes on disk")

    def test_every_envelope_is_ticker_attributable(self):
        unattributed = [
            envelope
            for envelope in self.envelopes
            if ticker_from_request_key(envelope.get("request_key")) is None
        ]
        self.assertEqual(unattributed, [])

    def test_the_stored_articles_carry_no_tone(self):
        # The measurement that explains why the join needed a classifier. If this
        # ever changes, the derived-tone path is no longer the only one and the
        # reasoning above is stale.
        toned = [
            article
            for envelope in self.envelopes
            for article in envelope.get("records") or []
            if article.get("tone") is not None
        ]
        self.assertEqual(toned, [])

    def test_the_join_produces_observations_from_shipped_data(self):
        report = build_observations(self.envelopes, rising, horizon="1d")
        self.assertGreater(report["joined"], 0)
        self.assertGreater(report["distinct_sources"], 1)
        self.assertEqual(join_problems(report), [])


if __name__ == "__main__":
    unittest.main()
