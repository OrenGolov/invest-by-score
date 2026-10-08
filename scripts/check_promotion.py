"""CI drift gate for the M5 promotion checklist.

A promotion gate is only a gate if it refuses. The failure mode is not that it
crashes — it is that it quietly approves, and nobody notices until a bad model
is serving.

1.  a clean candidate is APPROVED, so the gate is not simply a refusal machine;
2.  every single check, failed alone, BLOCKS promotion;
3.  NOT_EVALUATED blocks exactly as FAIL does. A drift check that could not run
    has not found the candidate clean, it has found nothing — this is the
    check most likely to be "simplified" into a pass;
4.  the manifest check runs FIRST: a candidate that cannot be reproduced is
    rejected before anyone evaluates its metrics;
5.  a veto rate that FALLS is a regression too. The obvious implementation
    catches only an increase, and a candidate that stops refusing bad
    decisions has not improved, it has stopped checking;
6.  the drift thresholds discriminate. MEASURED over 400 synthetic scores:
    an unshifted distribution scores PSI 0.013, a +1.9-point shift 2.854;
7.  history is immutable — a rewritten or vanished past decision is caught;
8.  the checklist COMPOSES the registry's own functions, so the script and
    `ModelRegistry.promote` cannot judge a candidate differently (W5);
9.  every check is reported, in order, and a non-PASS check says why;
10. `scripts/promote.py` exists and refuses by default.

Synthetic and deterministic.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from core.config import (  # noqa: E402
    PROMO_CHECK_APPROVAL,
    PROMO_CHECK_DRIFT,
    PROMO_CHECK_MANIFEST,
    PROMO_CHECK_OOS,
    PROMO_CHECK_REGRESSION,
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
    PROMOTION_CHECKS,
    PROMOTION_MAX_VETO_RATE_DECREASE,
    PROMOTION_MAX_VETO_RATE_INCREASE,
    PROMOTION_MIN_DECISIONS_FOR_REGRESSION,
    PROMOTION_PSI_FAIL,
    PROMOTION_PSI_WARN,
    PROMOTION_REQUIRED_CHECKS,
    PROMOTION_REQUIRED_MANIFEST_FIELDS,
)
from core.promotion import (  # noqa: E402
    drift_check,
    evaluate_promotion,
    history_immutable_problems,
    promotion_problems,
    regression_check,
    render_checklist,
)

_RNG = np.random.default_rng(7)
_REFERENCE = list(_RNG.normal(5.5, 1.2, 400).clip(0, 10))
_SAME = list(_RNG.normal(5.5, 1.2, 400).clip(0, 10))
_DRIFTED = list(_RNG.normal(7.4, 1.2, 400).clip(0, 10))

_CANDIDATE = {
    "model_version": "cand-1", "family": "linear", "feature_set_version": "fs-1",
    "training_data_cutoff": "2026-01-01", "dataset_hash": "d" * 16,
    "code_commit": "abc123", "seed": 42, "artifact_hash": "a" * 16,
    "hyperparameters": {"alpha": 1.0}, "forecast_target": "probability_up",
    "horizon": "20d",
}
_OOS = {
    "primary_metric": "directional_accuracy", "candidate_version": "cand-1",
    "candidate_value": 0.61, "incumbent_value": 0.575, "sample": 406,
}


def _decisions(count: int, every: int) -> list[dict]:
    return [
        {
            "decision_id": f"d{index}",
            "veto": {"rule_ids": ["position_cap"] if index % every == 0 else []},
        }
        for index in range(count)
    ]


def _verdict(**overrides):
    payload = dict(
        incumbent_version="inc-1", oos_comparison=_OOS,
        approved_by="reviewer@example.com", approved_at="2026-09-20T12:00:00Z",
        incumbent_decisions=_decisions(100, 5),
        candidate_decisions=_decisions(100, 5),
        reference_scores=_REFERENCE, candidate_scores=_SAME,
    )
    candidate = overrides.pop("candidate", _CANDIDATE)
    payload.update(overrides)
    return evaluate_promotion(candidate, **payload)


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    clean = _verdict()
    if not clean["approved"]:
        failures.append(
            f"a clean candidate was refused, blocked by {clean['blocking']} — "
            f"a gate that never approves is not a gate, it is a wall"
        )
    for problem in promotion_problems(clean):
        failures.append(f"clean verdict problem: {problem}")

    # ---------------------------------------------------------------- 2
    # Every check, failed ALONE, must block.
    alone = {
        PROMO_CHECK_MANIFEST: {"candidate": {**_CANDIDATE, "dataset_hash": None}},
        PROMO_CHECK_OOS: {"oos_comparison": {**_OOS, "candidate_value": 0.40}},
        PROMO_CHECK_REGRESSION: {"candidate_decisions": _decisions(100, 100)},
        PROMO_CHECK_DRIFT: {"candidate_scores": _DRIFTED},
        PROMO_CHECK_APPROVAL: {"approved_by": None},
    }
    for check, override in alone.items():
        verdict = _verdict(**override)
        if verdict["approved"]:
            failures.append(
                f"{check}: a candidate failing ONLY this check was approved — "
                f"every check must be able to block on its own"
            )
        if check not in verdict["blocking"]:
            failures.append(
                f"{check}: failed but did not appear in the blocking list "
                f"(blocking: {verdict['blocking']})"
            )
        for problem in promotion_problems(verdict):
            failures.append(f"{check} verdict problem: {problem}")

    # ---------------------------------------------------------------- 3
    # NOT_EVALUATED blocks exactly as FAIL does.
    unmeasured_drift = _verdict(reference_scores=None, candidate_scores=None)
    if unmeasured_drift["approved"]:
        failures.append(
            "a candidate with NO drift measurement was approved — a check that "
            "could not run has not found the candidate clean, it has found "
            "nothing, and collapsing that into PASS promotes on absent evidence"
        )
    if unmeasured_drift["checks"][PROMO_CHECK_DRIFT]["outcome"] != PROMO_NOT_EVALUATED:
        failures.append("missing drift data did not produce NOT_EVALUATED")

    thin = _verdict(
        incumbent_decisions=_decisions(5, 5), candidate_decisions=_decisions(5, 5)
    )
    if thin["approved"]:
        failures.append(
            f"a candidate with fewer than "
            f"{PROMOTION_MIN_DECISIONS_FOR_REGRESSION} decisions was approved — "
            f"a regression verdict on that few decisions is a reading of noise"
        )
    if thin["checks"][PROMO_CHECK_REGRESSION]["outcome"] != PROMO_NOT_EVALUATED:
        failures.append("a thin decision set did not produce NOT_EVALUATED")

    # ---------------------------------------------------------------- 4
    if PROMOTION_CHECKS[0] != PROMO_CHECK_MANIFEST:
        failures.append(
            "the manifest check no longer runs first — a candidate that cannot "
            "be reproduced should be rejected before its metrics are weighed"
        )
    for field in ("model_version", "dataset_hash", "code_commit", "seed"):
        if field not in PROMOTION_REQUIRED_MANIFEST_FIELDS:
            failures.append(
                f"{field!r} is no longer a required manifest field — without it "
                f"a promoted model cannot be reproduced from its own record"
            )
        stripped = _verdict(candidate={**_CANDIDATE, field: None})
        if stripped["approved"]:
            failures.append(f"a candidate missing {field!r} was approved")

    # ---------------------------------------------------------------- 5
    # A veto rate that FALLS is a regression too.
    collapsed = regression_check(_decisions(100, 5), _decisions(100, 100))
    if collapsed["outcome"] != PROMO_FAIL:
        failures.append(
            f"a candidate whose veto rate COLLAPSED (0.20 -> 0.01) scored "
            f"{collapsed['outcome']} — the W2 rules exist to refuse bad "
            f"decisions, and a candidate that stops refusing them has not "
            f"improved, it has stopped checking"
        )
    elif "FELL" not in collapsed["reason"]:
        failures.append("the collapsed-veto failure does not name the direction")

    spiked = regression_check(_decisions(100, 20), _decisions(100, 2))
    if spiked["outcome"] != PROMO_FAIL:
        failures.append(
            f"a candidate whose veto rate spiked (0.05 -> 0.50) scored "
            f"{spiked['outcome']}"
        )
    steady = regression_check(_decisions(100, 5), _decisions(100, 5))
    if steady["outcome"] != PROMO_PASS:
        failures.append(
            f"an unchanged veto rate scored {steady['outcome']} — the "
            f"regression check must not fire on stability"
        )
    if PROMOTION_MAX_VETO_RATE_DECREASE <= 0:
        failures.append("the veto-rate DECREASE tolerance was removed")
    if PROMOTION_MAX_VETO_RATE_INCREASE <= 0:
        failures.append("the veto-rate increase tolerance was removed")

    # ---------------------------------------------------------------- 6
    quiet = drift_check(_REFERENCE, _SAME)
    loud = drift_check(_REFERENCE, _DRIFTED)
    if quiet["outcome"] != PROMO_PASS:
        failures.append(
            f"an unshifted distribution scored {quiet['outcome']} "
            f"(psi={quiet.get('psi')}) — the drift check fires on noise"
        )
    if loud["outcome"] != PROMO_FAIL:
        failures.append(
            f"a materially shifted distribution scored {loud['outcome']} "
            f"(psi={loud.get('psi')}) — MEASURED, that shift is PSI ~2.85, "
            f"far above the {PROMOTION_PSI_FAIL} fail threshold"
        )
    if not 0.0 < PROMOTION_PSI_WARN < PROMOTION_PSI_FAIL:
        failures.append("the PSI thresholds no longer ascend")

    # ---------------------------------------------------------------- 7
    before = [
        {"decision_id": "d1", "model_version": "inc-1", "score": 7.2},
        {"decision_id": "d2", "model_version": "inc-1", "score": 5.1},
    ]
    if history_immutable_problems(before, list(before)):
        failures.append("an unchanged history was reported as mutated")
    rewritten = [{**before[0], "model_version": "cand-1"}, before[1]]
    if not history_immutable_problems(before, rewritten):
        failures.append(
            "a past decision's model_version was rewritten and not caught — "
            "promotion must never rewrite history"
        )
    if not history_immutable_problems(before, before[:1]):
        failures.append("a vanished historical decision was not caught")

    # ---------------------------------------------------------------- 8
    # The checklist composes the registry's functions; it does not restate them.
    from core import promotion as promotion_module

    source = inspect.getsource(promotion_module)
    for name in ("oos_comparison_problems", "approver_problems"):
        if name not in source:
            failures.append(
                f"core.promotion no longer composes {name} — the checklist and "
                f"ModelRegistry.promote would then judge a candidate by "
                f"different rules"
            )

    # ---------------------------------------------------------------- 9
    if list(clean["checks"]) != list(PROMOTION_CHECKS):
        failures.append("the verdict does not report every check in declared order")
    if set(PROMOTION_REQUIRED_CHECKS) != set(PROMOTION_CHECKS):
        failures.append(
            "not every check is required — an optional promotion gate is not "
            "a gate"
        )
    for name, row in _verdict(approved_by=None)["checks"].items():
        if row["outcome"] != PROMO_PASS and not row.get("reason"):
            failures.append(f"{name}: a {row['outcome']} check does not say why")
    for row in render_checklist(_verdict(approved_by=None)):
        if row["outcome"] != PROMO_PASS and not row["reason"]:
            failures.append(f"{row['check']}: renders blank for a reader")

    # ---------------------------------------------------------------- 10
    script = REPO_ROOT / "scripts" / "promote.py"
    if not script.exists():
        failures.append(
            "scripts/promote.py is missing — the sprint names it as the "
            "automated checklist"
        )
    else:
        text = script.read_text(encoding="utf-8")
        if "evaluate_promotion" not in text:
            failures.append("promote.py does not run the checklist")
        if "dry_run" not in text and "dry-run" not in text:
            failures.append(
                "promote.py has no dry run — a candidate must be inspectable "
                "before anyone is asked to approve it"
            )
        if 'verdict["approved"]' not in text:
            failures.append("promote.py does not gate on the verdict")

    if failures:
        print("M5 promotion gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M5 promotion gate OK:")
    print(
        f"  {len(PROMOTION_CHECKS)} checks, all required, manifest first; a "
        f"clean candidate is approved and each check blocks alone."
    )
    print("  NOT_EVALUATED blocks exactly as FAIL does — absent evidence is not safety.")
    print(
        f"  a veto rate that FALLS is a regression too "
        f"(tolerance {PROMOTION_MAX_VETO_RATE_DECREASE} either way)."
    )
    print(
        f"  drift discriminates: unshifted psi={quiet.get('psi')}, "
        f"shifted psi={loud.get('psi')} against a {PROMOTION_PSI_FAIL} floor."
    )
    print("  history immutability is verified, not assumed.")
    print("  the checklist composes the registry's own functions (W5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
