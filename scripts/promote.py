"""Run the M5 promotion checklist against a candidate model.

The sprint names this script: "Promotion checklist automated in
`scripts/promote.py`". It sequences the five checks and either promotes the
candidate in the registry or refuses, naming what blocked it.

    python scripts/promote.py --candidate cand-1 --approved-by you@example.com
    python scripts/promote.py --candidate cand-1 --dry-run
    python scripts/promote.py --list

**Refusal is the default.** A candidate is promoted only when every required
check reaches PASS. NOT_EVALUATED blocks exactly as FAIL does: a drift check
that could not run has not found the candidate clean, it has found nothing.

**The script does not decide anything the library does not.** It loads the
registry, gathers evidence, calls `core.promotion.evaluate_promotion`, and on
approval calls `ModelRegistry.promote` — which independently re-checks the OOS
comparison and the approver. The two cannot disagree, because the checklist
composes the very functions the registry enforces.

**A dry run is the honest default for inspection.** `--dry-run` evaluates and
prints without touching the registry, so a candidate can be examined before
anyone is asked to approve it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    PROMO_FAIL,
    PROMO_NOT_EVALUATED,
    PROMO_PASS,
)
from core.model_registry import (  # noqa: E402
    ModelRegistryError,
    load_model_registry,
)
from core.promotion import (  # noqa: E402
    evaluate_promotion,
    promotion_problems,
    render_checklist,
)

_SYMBOL = {PROMO_PASS: "[x]", PROMO_FAIL: "[!]", PROMO_NOT_EVALUATED: "[?]"}


def _load_json(path: str | None, label: str):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  could not read {label} from {path}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="", help="candidate model_version")
    parser.add_argument("--approved-by", default="", help="the accountable human")
    parser.add_argument("--approved-at", default="")
    parser.add_argument("--oos", default="", help="path to an OOS comparison JSON")
    parser.add_argument("--incumbent-decisions", default="")
    parser.add_argument("--candidate-decisions", default="")
    parser.add_argument("--reference-scores", default="")
    parser.add_argument("--candidate-scores", default="")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and print; never touch the registry")
    parser.add_argument("--list", action="store_true",
                        help="list registered models and exit")
    args = parser.parse_args()

    registry = load_model_registry()

    if args.list or not args.candidate:
        entries = list(registry.all_models().values())
        print(f"registered models: {len(entries)}")
        for entry in entries:
            print(
                f"  {entry.model_version:<28} {entry.family:<12} "
                f"{entry.status:<10} role={entry.role or '-'}"
            )
        if not args.candidate:
            print()
            print("no --candidate given; nothing was evaluated.")
        return 0

    candidate = registry.get(args.candidate)
    if candidate is None:
        print(f"candidate {args.candidate!r} is not registered.")
        print("A model must be registered before it can be promoted.")
        return 2

    incumbent = registry.incumbent(candidate.family)
    # Resolved ONCE and reused below. Loading it twice, or reading it back off
    # a check row, is how the checklist and the registry end up judging
    # different evidence — which this script then reports as a defect.
    oos_comparison = (
        _load_json(args.oos, "oos comparison")
        or getattr(candidate, "oos_comparison", None)
    )
    verdict = evaluate_promotion(
        candidate,
        incumbent_version=incumbent.model_version if incumbent else None,
        oos_comparison=oos_comparison,
        approved_by=args.approved_by or None,
        approved_at=args.approved_at
        or (datetime.now(timezone.utc).isoformat() if args.approved_by else None),
        incumbent_decisions=_load_json(args.incumbent_decisions, "incumbent decisions"),
        candidate_decisions=_load_json(args.candidate_decisions, "candidate decisions"),
        reference_scores=_load_json(args.reference_scores, "reference scores"),
        candidate_scores=_load_json(args.candidate_scores, "candidate scores"),
    )

    print(f"promotion checklist [{verdict['candidate_version']}]")
    print(f"  family    : {candidate.family}")
    print(f"  incumbent : {verdict['incumbent_version'] or '(none)'}")
    print()
    for row in render_checklist(verdict):
        mark = _SYMBOL.get(row["outcome"], "[ ]")
        print(f"  {mark} {row['check']:<24} {row['outcome']}")
        if row["reason"]:
            print(f"        {row['reason'][:100]}")
    print()

    for problem in promotion_problems(verdict):
        print(f"  CONTRACT PROBLEM: {problem}")

    if not verdict["approved"]:
        print(f"  REFUSED: {verdict['reason']}")
        if verdict["not_evaluated"]:
            print(
                "  note: a NOT_EVALUATED check blocks exactly as a failure does — "
                "it has not found the candidate clean, it has found nothing."
            )
        return 1

    if args.dry_run:
        print("  would promote (dry run; the registry was not modified).")
        return 0

    try:
        entry = registry.promote(
            verdict["candidate_version"],
            approved_by=args.approved_by,
            approved_at=args.approved_at
            or datetime.now(timezone.utc).isoformat(),
            oos_comparison=oos_comparison or {},
        )
    except ModelRegistryError as exc:
        # The registry independently re-checks the OOS comparison and the
        # approver. Disagreement here is a real defect, not a formality.
        print(f"  REGISTRY REFUSED after the checklist passed: {exc}")
        print(
            "  The checklist and the registry disagree — that is a defect in "
            "one of them, not a reason to override."
        )
        return 1

    from core.model_registry import persist_model_registry

    persist_model_registry(registry)
    print(f"  PROMOTED: {entry.model_version} -> {entry.status}")
    print("  historical predictions are unchanged; promotion is append-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
