"""Sprint N5: narrative-vs-fundamental attribution.

Hermetic: the regime fetch is patched (no network), news is either the
no-key UNAVAILABLE contract or a canned OK snapshot, macro runs keyless.
Coverage: the versioned bucket partition (import-time guard), per-line
mapping and math against the real ensemble blend, the N5 acceptance
criterion (with news at zero weight the narrative bucket reads exactly
0.0 — no phantom narrative), stance/thesis classification, exact score
reconciliation, and the explanation string assembling from the block so
the UI can never contradict it.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core import config as core_config
from core.score_engine import _ensemble_blend, build_attribution, build_score


def _contributions(**overrides):
    base = {
        "market_data": {"score_current": 10.0, "score_long": 10.0, "status": "OK", "note": "informational"},
        "technical_analysis": {"score_current": 8.0, "score_long": 6.0, "status": "OK", "note": ""},
        "fundamental_analysis": {"score_current": 5.0, "score_long": 5.0, "status": "OK", "note": ""},
        "news_intelligence": {"score_current": 7.0, "score_long": None, "status": "OK", "note": ""},
        "sentiment": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        "macroeconomic": {"score_current": 5.0, "score_long": 5.0, "status": "OK", "note": ""},
        "market_regime": {"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": "gate"},
    }
    base.update(overrides)
    return base


def _blend(contributions=None):
    return _ensemble_blend(
        contributions if contributions is not None else _contributions(),
        core_config.ENSEMBLE_WEIGHTS_CURRENT,
        core_config.ENSEMBLE_WEIGHTS_LONG,
    )


def _regime_stub(status="OK", label="bullish", proxy=0.9):
    stub = {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": status,
        "source_id": "yahoo_finance_chart", "source_confidence": 0.8,
        "published_time": "2024-01-02 00:00:00", "calculation_version": "regime-contract-v1",
        "lookback_period": "300 sessions", "regime": label, "probability_proxy": proxy,
        "transition_risk": {"window_sessions": 20, "labeled_sessions": 20, "flips": 0,
                            "flip_rate": 0.0, "labels": [label] * 20},
        "inputs": {}, "rule_trace": {}, "reason": "stub",
    }
    if status == "OK":
        stub["pipeline"] = {
            "pipeline_version": "regime-pipeline-v1",
            "classifier_version": "regime-classifier-v1",
            "counts": {"sessions_used": 300, "labeled_sessions": 18, "transition_labels": 20},
        }
    return stub


def _ok_news(sentiment):
    return {
        "ticker": "MSFT", "as_of": "2024-01-02 00:00:00", "status": "OK",
        "source_id": "newsapi_news", "source_confidence": 0.7,
        "published_time": "2024-01-02 09:00:00",
        "calculation_version": "news-contract-v1", "lookback_period": "7d",
        "sentiment_score": sentiment, "articles": [], "reason": "",
        "pipeline": {"pipeline_version": core_config.NEWS_PIPELINE_VERSION},
    }


class AttributionConfigTests(unittest.TestCase):
    """The versioned bucket partition is a governance constant."""

    def test_buckets_partition_the_ensemble_lines_exactly(self):
        assigned = [
            agent
            for members in core_config.ATTRIBUTION_BUCKETS.values()
            for agent in members
        ]
        covered = set(assigned) | set(core_config.ATTRIBUTION_INFORMATIONAL_LINES)
        self.assertEqual(
            covered,
            set(core_config.ENSEMBLE_WEIGHTS_CURRENT),
            "attribution buckets must partition the ensemble lines exactly",
        )
        self.assertEqual(len(assigned), len(set(assigned)), "duplicate bucket member")

    def test_bucket_semantics_match_the_n5_spec(self):
        self.assertEqual(
            core_config.ATTRIBUTION_BUCKETS,
            {
                "operational": ("fundamental_analysis", "technical_analysis"),
                "narrative": ("news_intelligence", "sentiment"),
                "macro_shock": ("macroeconomic", "market_regime"),
            },
        )
        self.assertEqual(core_config.ATTRIBUTION_VERSION, "score-attribution-v1")
        self.assertGreater(core_config.ATTRIBUTION_SUPPORT_THRESHOLD, 0.0)


class BuildAttributionTests(unittest.TestCase):
    """build_attribution over the real ensemble blend math."""

    def test_lines_mirror_the_ensemble_contributions(self):
        _, _, breakdown = _blend()
        blended = (breakdown["current_time_score"] + breakdown["long_term_score"]) / 2.0
        attribution = build_attribution(breakdown, blended)
        for agent, entry in breakdown["agents"].items():
            line = attribution["lines"][agent]
            with self.subTest(agent=agent):
                expected_total = (entry["contribution_current"] + entry["contribution_long"]) / 2.0
                self.assertEqual(line["total"], round(expected_total, 4))
                self.assertEqual(line["contribution_current"], round(entry["contribution_current"], 4))
                self.assertEqual(line["contribution_long"], round(entry["contribution_long"], 4))
                self.assertEqual(line["status"], entry["status"])

    def test_line_bucket_mapping(self):
        _, _, breakdown = _blend()
        attribution = build_attribution(breakdown, 6.0)
        expected = {
            "fundamental_analysis": "operational",
            "technical_analysis": "operational",
            "news_intelligence": "narrative",
            "sentiment": "narrative",
            "macroeconomic": "macro_shock",
            "market_regime": "macro_shock",
            "market_data": "informational",
        }
        for agent, bucket in expected.items():
            self.assertEqual(attribution["lines"][agent]["bucket"], bucket, agent)

    def test_bucket_totals_sum_their_member_lines(self):
        _, _, breakdown = _blend()
        attribution = build_attribution(breakdown, 6.0)
        for bucket_name, bucket in attribution["buckets"].items():
            expected = round(
                sum(attribution["lines"][agent]["total"] for agent in bucket["members"]), 4
            )
            self.assertEqual(bucket["total"], expected, bucket_name)
            self.assertEqual(bucket["members"], list(core_config.ATTRIBUTION_BUCKETS[bucket_name]))

    def test_reconciliation_is_exact_against_the_published_score(self):
        current, long_term, breakdown = _blend()
        blended = round((current + long_term) / 2.0, 2)
        attribution = build_attribution(breakdown, blended)
        self.assertTrue(attribution["reconciles"], attribution)
        self.assertEqual(attribution["attributed_total"], blended)
        self.assertEqual(attribution["score"], blended)

    def test_narrative_is_exactly_zero_when_news_has_zero_weight(self):
        # N5 acceptance: no phantom narrative. News ineligible (weight
        # renormalized away) and sentiment a placeholder -> narrative 0.0.
        contributions = _contributions(
            news_intelligence={"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        )
        _, _, breakdown = _blend(contributions)
        attribution = build_attribution(breakdown, 6.0)
        self.assertEqual(attribution["buckets"]["narrative"]["total"], 0.0)
        self.assertEqual(attribution["buckets"]["narrative"]["stance"], "neutral")
        self.assertIn("contributes exactly 0.0", attribution["summary"])
        self.assertIn("no eligible narrative sources", attribution["summary"])
        self.assertFalse(attribution["lines"]["news_intelligence"]["eligible_current"])

    def test_news_ok_but_scoreless_also_reads_zero_with_the_right_reason(self):
        # News OK in name but with no usable sentiment score -> still not
        # eligible -> exactly 0.0, and the summary names the right reason
        # (distinguishing it from the no-provider case).
        contributions = _contributions(
            news_intelligence={"score_current": None, "score_long": None, "status": "OK", "note": ""},
        )
        _, _, breakdown = _blend(contributions)
        attribution = build_attribution(breakdown, 6.0)
        self.assertEqual(attribution["buckets"]["narrative"]["total"], 0.0)
        self.assertIn("narrative status OK but no usable sentiment score", attribution["summary"])
        self.assertNotIn("no eligible narrative sources", attribution["summary"])

    def test_narrative_supports_when_news_is_eligible_and_directional(self):
        _, _, breakdown = _blend()
        attribution = build_attribution(breakdown, 6.0)
        self.assertGreater(
            attribution["buckets"]["narrative"]["total"], core_config.ATTRIBUTION_SUPPORT_THRESHOLD
        )
        self.assertEqual(attribution["buckets"]["narrative"]["stance"], "supports")
        self.assertEqual(attribution["thesis_support"], "operational_and_narrative")
        self.assertIn("thesis support: operational_and_narrative", attribution["summary"])

    def test_stance_opposes_for_negatively_contributing_buckets(self):
        # With the current 0-10 weighted-average ensemble every contribution
        # is non-negative, so "opposes" is unreachable through the blend; it
        # is reserved for the signed contribution semantics of forecast
        # decomposition (Sprint F6). build_attribution is pure, so the stance
        # contract is pinned here with a synthetic signed breakdown.
        breakdown = {"agents": {
            "technical_analysis": {"contribution_current": -0.6, "contribution_long": -0.4,
                                   "status": "OK", "eligible_current": True, "eligible_long": True},
            "fundamental_analysis": {"contribution_current": 0.0, "contribution_long": 0.0,
                                     "status": "OK", "eligible_current": False, "eligible_long": False},
            "news_intelligence": {"contribution_current": 0.0, "contribution_long": 0.0,
                                  "status": "UNAVAILABLE", "eligible_current": False, "eligible_long": False},
            "sentiment": {"contribution_current": 0.0, "contribution_long": 0.0,
                          "status": "UNAVAILABLE", "eligible_current": False, "eligible_long": False},
            "macroeconomic": {"contribution_current": 0.0, "contribution_long": 0.0,
                              "status": "UNAVAILABLE", "eligible_current": False, "eligible_long": False},
            "market_regime": {"contribution_current": 0.0, "contribution_long": 0.0,
                              "status": "UNAVAILABLE", "eligible_current": False, "eligible_long": False},
            "market_data": {"contribution_current": 0.0, "contribution_long": 0.0,
                            "status": "OK", "eligible_current": False, "eligible_long": False},
        }}
        attribution = build_attribution(breakdown, 4.5)
        self.assertEqual(attribution["buckets"]["operational"]["total"], -0.5)
        self.assertEqual(attribution["buckets"]["operational"]["stance"], "opposes")
        self.assertEqual(attribution["buckets"]["narrative"]["total"], 0.0)

    def test_thesis_support_reads_narrative_only_when_operational_does_not_support(self):
        # Operational evidence flat at zero (scores of 0.0): the narrative
        # alone carries the thesis.
        contributions = _contributions(
            technical_analysis={"score_current": 0.0, "score_long": 0.0, "status": "OK", "note": ""},
            fundamental_analysis={"score_current": 0.0, "score_long": 0.0, "status": "OK", "note": ""},
        )
        _, _, breakdown = _blend(contributions)
        attribution = build_attribution(breakdown, 2.0)
        self.assertEqual(attribution["buckets"]["operational"]["total"], 0.0)
        self.assertEqual(attribution["thesis_support"], "narrative")

    def test_thesis_support_reads_neither_when_nothing_supports(self):
        contributions = _contributions(
            technical_analysis={"score_current": 0.0, "score_long": 0.0, "status": "OK", "note": ""},
            fundamental_analysis={"score_current": 0.0, "score_long": 0.0, "status": "OK", "note": ""},
            news_intelligence={"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
            macroeconomic={"score_current": None, "score_long": None, "status": "UNAVAILABLE", "note": ""},
        )
        _, _, breakdown = _blend(contributions)
        attribution = build_attribution(breakdown, 0.0)
        self.assertEqual(attribution["thesis_support"], "neither")
        for bucket in attribution["buckets"].values():
            self.assertEqual(bucket["stance"], "neutral")

    def test_deterministic_output(self):
        _, _, breakdown = _blend()
        self.assertEqual(build_attribution(breakdown, 6.32), build_attribution(breakdown, 6.32))


class ScoreEngineAttributionIntegrationTests(unittest.TestCase):
    """build_score carries the block and the explanation assembles from it."""

    def test_scoring_breakdown_carries_the_attribution_block(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
            result = build_score("MSFT", "2024-01-02", persist_audit=False)
        attribution = result.scoring_breakdown["attribution"]
        self.assertEqual(attribution["attribution_version"], "score-attribution-v1")
        self.assertTrue(attribution["reconciles"], attribution)
        self.assertEqual(attribution["score"], result.score)
        # Lines mirror the ensemble contributions one-for-one.
        for agent, entry in result.ensemble_breakdown["agents"].items():
            line = attribution["lines"][agent]
            self.assertEqual(
                line["total"],
                round((entry["contribution_current"] + entry["contribution_long"]) / 2.0, 4),
                agent,
            )

    def test_narrative_reads_exactly_zero_on_the_default_run(self):
        # N5 acceptance, end to end: no news key -> news UNAVAILABLE ->
        # sentiment placeholder -> the narrative bucket is exactly 0.0 and
        # the explanation states the thesis is carried by operational
        # evidence — no phantom narrative.
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
            result = build_score("MSFT", "2024-01-02", persist_audit=False)
        attribution = result.scoring_breakdown["attribution"]
        self.assertEqual(attribution["buckets"]["narrative"]["total"], 0.0)
        self.assertEqual(result.news_snapshot["status"], "UNAVAILABLE")
        self.assertIn("contributes exactly 0.0", result.explanation)
        self.assertIn(f"thesis support: {attribution['thesis_support']}", result.explanation)
        self.assertNotEqual(attribution["thesis_support"], "narrative")
        self.assertIn("operational", attribution["thesis_support"])

    def test_narrative_moves_and_explanation_assembles_from_the_block(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()), \
                patch("core.score_engine.fetch_news_snapshot", return_value=_ok_news(0.5)):
            with_news = build_score("MSFT", "2024-01-02", persist_audit=False)
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
            baseline = build_score("MSFT", "2024-01-02", persist_audit=False)
        with_news_attribution = with_news.scoring_breakdown["attribution"]
        baseline_attribution = baseline.scoring_breakdown["attribution"]
        # The narrative bucket moves from exactly 0.0 to a real contribution.
        self.assertEqual(baseline_attribution["buckets"]["narrative"]["total"], 0.0)
        self.assertGreater(with_news_attribution["buckets"]["narrative"]["total"], 0.0)
        self.assertEqual(with_news_attribution["thesis_support"], "operational_and_narrative")
        # The explanation sentence is assembled from the same block — the UI
        # can never contradict the numbers.
        narrative_total = with_news_attribution["buckets"]["narrative"]["total"]
        operational_total = with_news_attribution["buckets"]["operational"]["total"]
        self.assertIn(f"narrative evidence (news + sentiment) {narrative_total:+.2f}", with_news.explanation)
        self.assertIn(f"operational evidence (fundamental + technical) {operational_total:+.2f}", with_news.explanation)
        self.assertIn("thesis support: operational_and_narrative", with_news.explanation)
        # Weight renormalization moves every ABSOLUTE contribution when a
        # line enters or leaves; what must not move is the underlying agent
        # score, and the narrative bucket must equal its news line exactly.
        self.assertEqual(
            with_news.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
            baseline.ensemble_breakdown["agents"]["technical_analysis"]["score_current"],
        )
        self.assertEqual(
            with_news_attribution["buckets"]["narrative"]["total"],
            with_news_attribution["lines"]["news_intelligence"]["total"],
        )

    def test_attribution_is_deterministic_across_replays(self):
        with patch("core.score_engine.fetch_regime_snapshot", return_value=_regime_stub()):
            first = build_score("MSFT", "2024-01-02", persist_audit=False)
            second = build_score("MSFT", "2024-01-02", persist_audit=False)
        self.assertEqual(
            first.scoring_breakdown["attribution"],
            second.scoring_breakdown["attribution"],
        )


if __name__ == "__main__":
    unittest.main()
