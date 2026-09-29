"""B4 tests — a basket of correlated holdings is one bet.

MEASURED on 3,771 sessions of real returns: a book of 10% each in
NVDA/AMD/AVGO/SOXX plus 20% each in MSFT/GOOGL/CAT raises ZERO R1 flags (no
position above 17% of risk) while the four semiconductors hold 52.6% of portfolio
variance at a mean pairwise correlation of 0.62.

Synthetic covariances here, so the tests are deterministic and fast; the real-data
reproduction lives in the gate.
"""

from __future__ import annotations

import random
import unittest

from core.cluster_exposure import (
    LINKAGE_AVERAGE,
    LINKAGE_SINGLE,
    ClusterExposureError,
    build_clusters,
    cluster_problems,
    correlation_matrix,
    evaluate_cluster_exposure,
    mean_pairwise_correlation,
    render_cluster_exposure,
)
from core.config import (
    CLUSTER_CONCENTRATED,
    CLUSTER_DIFFUSE,
    CLUSTER_EXPOSURE_BLOCKS_TRADES,
    CLUSTER_EXPOSURE_LINKAGE,
    CLUSTER_EXPOSURE_MIN_CORRELATION,
    CLUSTER_EXPOSURE_MIN_MEMBERS,
    CLUSTER_NOT_EVALUATED,
    CLUSTER_VERDICTS,
)
from core.position_exposure import aligned_returns, covariance


def synthetic(sessions=800, seed=11):
    """Four 'semis' sharing a factor, plus three independent names.

    Built so the cluster is present BY CONSTRUCTION — a test that had to
    discover whether a cluster existed would be testing the generator.
    """
    rng = random.Random(seed)
    dates = [f"d{index:04d}" for index in range(sessions)]
    series = {
        name: {}
        for name in ("SEMI_A", "SEMI_B", "SEMI_C", "SEMI_ETF", "SOFT", "BANK", "INDY")
    }
    for index, date in enumerate(dates):
        market = rng.gauss(0, 0.006)
        semi = rng.gauss(0, 0.014)
        for name in ("SEMI_A", "SEMI_B", "SEMI_C"):
            series[name][date] = market + semi + rng.gauss(0, 0.004)
        series["SEMI_ETF"][date] = market + semi + rng.gauss(0, 0.002)
        series["SOFT"][date] = market + rng.gauss(0, 0.010)
        series["BANK"][date] = market + rng.gauss(0, 0.012)
        # A BRIDGE, on purpose. SEMI_ETF is to this fixture what SOXX is to the
        # real book: broadly correlated with everything, so SINGLE linkage chains
        # the semis to the non-semis through it. Without a bridge the fixture
        # cannot exhibit degeneracy, and a test that swept it would conclude the
        # two linkage rules are equivalent — which they are, absent a bridge.
        series["SEMI_ETF"][date] = (
            series["SEMI_ETF"][date] + 0.6 * series["SOFT"][date]
        )
        series["INDY"][date] = 0.5 * market + rng.gauss(0, 0.008)
    return series


SERIES = synthetic()
TICKERS, MATRIX = aligned_returns(SERIES)
COV = covariance(MATRIX)

CLUSTERED = {
    "SEMI_A": 0.10,
    "SEMI_B": 0.10,
    "SEMI_C": 0.10,
    "SEMI_ETF": 0.10,
    "SOFT": 0.20,
    "BANK": 0.20,
    "INDY": 0.20,
}
SEMIS = {"SEMI_A", "SEMI_B", "SEMI_C", "SEMI_ETF"}


class TheClusterIsFoundTests(unittest.TestCase):
    """What R1 could not see."""

    def setUp(self):
        self.report = evaluate_cluster_exposure(CLUSTERED, TICKERS, COV)

    def test_the_semis_are_grouped(self):
        largest = self.report["largest_cluster"]
        self.assertEqual(set(largest["members"]), SEMIS)

    def test_the_cluster_carries_more_risk_than_weight(self):
        # The whole point: 40% of the weight holding far more of the variance.
        largest = self.report["largest_cluster"]
        self.assertGreater(largest["risk_share"], largest["weight_share"])

    def test_the_verdict_is_concentrated(self):
        self.assertEqual(self.report["verdict"], CLUSTER_CONCENTRATED)

    def test_the_mean_correlation_is_reported(self):
        # Why it is a cluster must be checkable, not asserted.
        self.assertGreater(self.report["largest_cluster"]["mean_correlation"], 0.5)

    def test_the_report_is_contract_clean(self):
        self.assertEqual(cluster_problems(self.report), [])

    def test_no_single_position_would_have_flagged(self):
        from core.position_exposure import risk_contributions
        from core.config import EXPOSURE_RISK_REVIEW

        contributions = risk_contributions(CLUSTERED, TICKERS, COV)
        self.assertTrue(
            all(share <= EXPOSURE_RISK_REVIEW for share in contributions.values())
        )


class DiffuseIsReachableTests(unittest.TestCase):
    """A gate that only ever flags is a constant."""

    def test_an_uncorrelated_book_is_diffuse(self):
        book = {"INDY": 0.34, "BANK": 0.33, "SOFT": 0.33}
        report = evaluate_cluster_exposure(book, TICKERS, COV)
        self.assertEqual(report["verdict"], CLUSTER_DIFFUSE)

    def test_a_single_holding_forms_no_cluster(self):
        report = evaluate_cluster_exposure({"INDY": 1.0}, TICKERS, COV)
        self.assertEqual(report["verdict"], CLUSTER_DIFFUSE)
        self.assertIsNone(report["largest_cluster"])

    def test_a_small_cluster_below_the_bar_is_diffuse(self):
        # Two correlated names holding little risk is not a concentration.
        book = {"SEMI_A": 0.05, "SEMI_B": 0.05, "INDY": 0.45, "BANK": 0.45}
        report = evaluate_cluster_exposure(book, TICKERS, COV)
        self.assertEqual(report["verdict"], CLUSTER_DIFFUSE)


class AverageLinkageIsRequiredTests(unittest.TestCase):
    """MEASURED: single linkage reports the whole book as one cluster.

    Not because it is less stable — the swept spreads are nearly identical
    (63.7% vs 62.3%) — but because it DEGENERATES. It merges on one qualifying
    pair, so a broadly-correlated ETF chains unrelated holdings together, and a
    cluster containing every holding cannot distinguish a concentrated book from
    a diversified one.
    """

    def test_the_contract_requires_average_linkage(self):
        self.assertEqual(CLUSTER_EXPOSURE_LINKAGE, LINKAGE_AVERAGE)

    def test_single_linkage_chains_more_readily(self):
        # At a threshold low enough to chain, single linkage produces a strictly
        # larger largest-cluster than average linkage.
        single = max(
            build_clusters(TICKERS, COV, threshold=0.35, linkage=LINKAGE_SINGLE),
            key=len,
        )
        average = max(
            build_clusters(TICKERS, COV, threshold=0.35, linkage=LINKAGE_AVERAGE),
            key=len,
        )
        self.assertGreaterEqual(len(single), len(average))

    def test_single_linkage_degenerates_where_average_does_not(self):
        """The measured reason for the choice, pinned.

        A sabotage swapping the linkage rule passed these tests until this
        existed: nothing here compared the two on the property that decided it.
        """
        swept = [round(0.20 + 0.05 * step, 3) for step in range(13)]

        def degenerate(linkage):
            return sum(
                1
                for threshold in swept
                if len(
                    groups := build_clusters(
                        TICKERS, COV, threshold=threshold, linkage=linkage
                    )
                )
                == 1
                and len(groups[0]) == len(TICKERS)
            )

        single = degenerate(LINKAGE_SINGLE)
        average = degenerate(LINKAGE_AVERAGE)
        self.assertGreater(
            single,
            average,
            "single linkage no longer degenerates more readily than average, so "
            "the measured reason for requiring average linkage is stale",
        )

    def test_the_default_linkage_does_not_swallow_the_book(self):
        # The property that matters: a cluster containing every holding cannot
        # distinguish a concentrated book from a diversified one.
        groups = build_clusters(
            TICKERS, COV, threshold=CLUSTER_EXPOSURE_MIN_CORRELATION
        )
        self.assertGreater(len(groups), 1)

    def test_average_linkage_recovers_the_semis(self):
        clusters = build_clusters(
            TICKERS, COV, threshold=CLUSTER_EXPOSURE_MIN_CORRELATION
        )
        self.assertIn(sorted(SEMIS), [sorted(group) for group in clusters])

    def test_an_unknown_linkage_is_refused(self):
        with self.assertRaises(ClusterExposureError):
            build_clusters(TICKERS, COV, linkage="complete")

    def test_an_impossible_threshold_is_refused(self):
        for threshold in (-1.0, 1.0, 1.5):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ClusterExposureError):
                    build_clusters(TICKERS, COV, threshold=threshold)

    def test_a_high_threshold_splits_everything(self):
        clusters = build_clusters(TICKERS, COV, threshold=0.99)
        self.assertEqual(len(clusters), len(TICKERS))


class AClusterOfOneIsAPositionTests(unittest.TestCase):
    """R1 already flags those; reporting them here double-counts."""

    def test_the_minimum_is_two(self):
        self.assertEqual(CLUSTER_EXPOSURE_MIN_MEMBERS, 2)

    def test_a_lone_holding_is_never_the_largest_cluster(self):
        report = evaluate_cluster_exposure({"INDY": 0.6, "BANK": 0.4}, TICKERS, COV)
        largest = report.get("largest_cluster")
        if largest is not None:
            self.assertGreaterEqual(largest["size"], 2)

    def test_a_lone_holding_reports_no_mean_correlation(self):
        # None, not 1.0: a single holding has no internal correlation, and 1.0
        # would read as a perfectly correlated pair.
        self.assertIsNone(mean_pairwise_correlation(["INDY"], TICKERS, COV))


class UnmeasurableIsNotSafeTests(unittest.TestCase):
    """0% would make the least-understood book look the safest."""

    def test_no_portfolio_is_not_evaluated(self):
        for weights in (None, {}):
            with self.subTest(weights=weights):
                report = evaluate_cluster_exposure(weights, TICKERS, COV)
                self.assertEqual(report["verdict"], CLUSTER_NOT_EVALUATED)

    def test_no_covariance_is_not_evaluated(self):
        report = evaluate_cluster_exposure(CLUSTERED, TICKERS, None)
        self.assertEqual(report["verdict"], CLUSTER_NOT_EVALUATED)

    def test_an_unevaluated_report_names_no_cluster(self):
        report = evaluate_cluster_exposure(None, None, None)
        self.assertIsNone(report.get("largest_cluster"))
        self.assertEqual(cluster_problems(report), [])

    def test_a_ragged_covariance_reports_rather_than_crashing(self):
        # FOUND BY SABOTAGE: this branch was unreachable, because a ragged matrix
        # raised IndexError which was not in the except clause. A fail-closed path
        # that crashes is not fail-closed.
        report = evaluate_cluster_exposure({"SEMI_A": 1.0}, TICKERS, [[1.0]])
        self.assertEqual(report["verdict"], CLUSTER_NOT_EVALUATED)
        self.assertEqual(cluster_problems(report), [])

    def test_a_zero_variance_asset_has_no_correlation(self):
        # Not a correlation of zero, which would read as "independent" when the
        # truth is "unmeasurable".
        flat = correlation_matrix(["A", "B"], [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(flat, {})


class ContractTests(unittest.TestCase):
    def test_not_evaluated_is_the_weakest_verdict(self):
        self.assertEqual(CLUSTER_VERDICTS[0], CLUSTER_NOT_EVALUATED)

    def test_b4_reports_and_does_not_block(self):
        self.assertFalse(CLUSTER_EXPOSURE_BLOCKS_TRADES)
        self.assertFalse(
            evaluate_cluster_exposure(CLUSTERED, TICKERS, COV)["blocks_trades"]
        )

    def test_an_unknown_verdict_is_refused(self):
        from core.cluster_exposure import _result

        with self.assertRaises(ClusterExposureError):
            _result("MAYBE", "no")

    def test_a_forged_concentrated_report_without_a_cluster_is_caught(self):
        self.assertTrue(
            cluster_problems({"verdict": CLUSTER_CONCENTRATED, "reason": "trust me"})
        )

    def test_a_forged_concentrated_report_on_one_holding_is_caught(self):
        self.assertTrue(
            cluster_problems(
                {
                    "verdict": CLUSTER_CONCENTRATED,
                    "reason": "r",
                    "largest_cluster": {"size": 1, "risk_share": 0.9, "members": ["X"]},
                }
            )
        )

    def test_a_forged_unevaluated_report_with_a_cluster_is_caught(self):
        self.assertTrue(
            cluster_problems(
                {
                    "verdict": CLUSTER_NOT_EVALUATED,
                    "reason": "r",
                    "largest_cluster": {"size": 2, "risk_share": 0.5},
                }
            )
        )

    def test_a_multi_member_cluster_without_a_correlation_is_caught(self):
        self.assertTrue(
            cluster_problems(
                {
                    "verdict": CLUSTER_DIFFUSE,
                    "reason": "r",
                    "clusters": [
                        {"members": ["A", "B"], "size": 2, "mean_correlation": None}
                    ],
                }
            )
        )

    def test_a_blocking_report_is_caught(self):
        self.assertTrue(
            cluster_problems(
                {"verdict": CLUSTER_DIFFUSE, "reason": "r", "blocks_trades": True}
            )
        )

    def test_a_non_mapping_report_is_caught(self):
        self.assertTrue(cluster_problems("not a mapping"))


class RenderTests(unittest.TestCase):
    def test_a_concentrated_report_names_its_members(self):
        text = "\n".join(
            render_cluster_exposure(evaluate_cluster_exposure(CLUSTERED, TICKERS, COV))
        )
        for member in SEMIS:
            self.assertIn(member, text)

    def test_an_absent_cluster_renders_as_absent(self):
        text = "\n".join(
            render_cluster_exposure(evaluate_cluster_exposure(None, None, None))
        )
        self.assertIn("ABSENT", text)

    def test_render_returns_lines_not_a_blob(self):
        lines = render_cluster_exposure(
            evaluate_cluster_exposure(CLUSTERED, TICKERS, COV)
        )
        self.assertIsInstance(lines, list)
        self.assertTrue(all(isinstance(line, str) for line in lines))


if __name__ == "__main__":
    unittest.main()
