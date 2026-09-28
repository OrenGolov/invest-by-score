"""X8 sealed holdout — the final period, and whether anyone can prove it is untouched.

"Maintain an untouched final period. Once opened, do not use it for further
model selection."

**The tail IS untouched, and that is an accident of arithmetic rather than a
seal.** MEASURED on the 8 shipped runs, the holdout occupies rows 346–405 of a
406-row dataset and no fold reaches past row 299. But of the four facts needed
to verify that, the run ledger records NONE: not the bounds, not the row count,
not the geometry, not the absolute fold indices. Recovering the seal required a
second file and a guessed geometry.

**And the run cannot be reproduced.** F2 added the 252-session horizon the day
AFTER the ledger was written, and `build_walk_forward_folds` requires
`embargo >= max horizon`. The geometry that produced the ledger — and the
default in `scripts/train.py` — are both rejected today.

**No legal geometry fits the data.** With the embargo at 252, the cheapest
walk-forward layout that still reserves a holdout needs 432 rows against 406
available. So X8's shipped verdict is NOT_EVALUATED with a named shortfall.
Reporting SEALED because the tail happens to be unread would credit the system
for a property it cannot demonstrate.

The module therefore does two separable things: it DESCRIBES a seal (bounds,
geometry, embargo, openings) so a future run can record one, and it REFUSES to
call the present state sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    SEALED_HOLDOUT_VERSION,
)


class SealedHoldoutError(ValueError):
    """Raised when a seal request is structurally invalid."""


# Why a seal could not be established. Distinct codes, because the fixes differ.
HOLDOUT_REASON_NO_BOUNDS = "NO_RECORDED_BOUNDS"
HOLDOUT_REASON_NO_GEOMETRY = "NO_RECORDED_GEOMETRY"
HOLDOUT_REASON_GEOMETRY_ILLEGAL = "GEOMETRY_ILLEGAL"
HOLDOUT_REASON_NO_RUNS = "NO_RUNS"
HOLDOUT_REASON_EMBARGO_SHORT = "EMBARGO_SHORTER_THAN_HORIZON"
HOLDOUT_REASON_OVERLAP = "FOLD_OVERLAPS_HOLDOUT"

# The four facts a run must carry for its seal to be auditable. Named as data so
# the gate and the report agree on what "recorded" means.
SEAL_REQUIRED_FACTS: tuple[str, ...] = (
    "holdout_bounds",
    "dataset_rows",
    "geometry",
    "fold_indices",
)


def max_label_horizon() -> int:
    """The longest label horizon, in sessions. The embargo floor."""
    return max(LABEL_HORIZON_SESSIONS.values())


def recorded_facts(run: Mapping[str, Any] | None) -> dict[str, bool]:
    """Which of the four seal facts a persisted run actually carries.

    A run that carries none cannot be audited — its seal can only be guessed at
    from a second file, which is how the shipped ledger had to be read.
    """
    if run is None:
        return {fact: False for fact in SEAL_REQUIRED_FACTS}
    if not isinstance(run, Mapping):
        raise SealedHoldoutError("a run must be a mapping")
    folds = run.get("folds") or []
    first = folds[0] if folds and isinstance(folds[0], Mapping) else {}
    return {
        "holdout_bounds": run.get("holdout") is not None,
        "dataset_rows": any(
            run.get(key) is not None
            for key in ("dataset_rows", "row_count", "rows")
        ),
        "geometry": any(
            run.get(key) is not None
            for key in ("geometry", "fold_sessions", "holdout_sessions")
        ),
        # ABSOLUTE indices, not row COUNTS. `train_rows` says how many rows a
        # fold used; it does not say WHERE they were, and only position can
        # prove a fold stopped short of the holdout.
        "fold_indices": any(
            first.get(key) is not None
            for key in ("validation", "train", "validation_index")
        ),
    }


def _result(verdict: str, reason: str, **detail) -> dict:
    if verdict not in HOLDOUT_VERDICTS:
        raise SealedHoldoutError(f"{verdict!r} is not a holdout verdict")
    record = {
        "version": SEALED_HOLDOUT_VERSION,
        "verdict": verdict,
        "reason": reason,
        "blocks_trades": HOLDOUT_BLOCKS_TRADES,
    }
    record.update(detail)
    return record


def minimum_rows(
    fold_sessions: int, embargo_sessions: int, holdout_sessions: int
) -> int:
    """Rows needed for one walk-forward fold that still reserves a holdout."""
    return 2 * int(fold_sessions) + int(embargo_sessions) + int(holdout_sessions)


def geometry_problems(
    dataset_rows: int,
    fold_sessions: int,
    embargo_sessions: int,
    holdout_sessions: int,
) -> list[str]:
    """Why this geometry cannot produce a sealed walk-forward fit. Empty = legal.

    Checked SEPARATELY from `build_walk_forward_folds` so X8 can report the
    shortfall in rows rather than only that construction failed.
    """
    problems: list[str] = []
    horizon = max_label_horizon()
    for name, value in (
        ("fold_sessions", fold_sessions),
        ("embargo_sessions", embargo_sessions),
        ("holdout_sessions", holdout_sessions),
        ("dataset_rows", dataset_rows),
    ):
        if int(value) < 1:
            problems.append(f"{name} must be positive, got {value}")
    if problems:
        return problems
    if int(embargo_sessions) < horizon:
        problems.append(
            f"the embargo ({embargo_sessions}) is shorter than the longest "
            f"label horizon ({horizon}), so a label inside the holdout was "
            f"computed from prices the last fold trained on"
        )
    needed = minimum_rows(fold_sessions, embargo_sessions, holdout_sessions)
    if int(dataset_rows) < needed:
        problems.append(
            f"{dataset_rows} rows cannot hold one fold plus a sealed tail at "
            f"this geometry: {needed} are needed, short by {needed - int(dataset_rows)}"
        )
    return problems


def describe_seal(
    dataset_rows: int,
    fold_sessions: int,
    embargo_sessions: int,
    holdout_sessions: int,
    *,
    last_validation_row: int | None = None,
) -> dict:
    """Where the seal falls, and whether it is legal.

    Separated from `evaluate_sealed_holdout` so a FUTURE run can record a seal
    with this, which is the fix X8 asks for, independent of the current verdict.
    """
    problems = geometry_problems(
        dataset_rows, fold_sessions, embargo_sessions, holdout_sessions
    )
    rows = int(dataset_rows)
    holdout_start = rows - int(holdout_sessions)
    described = {
        "dataset_rows": rows,
        "geometry": {
            "fold_sessions": int(fold_sessions),
            "embargo_sessions": int(embargo_sessions),
            "holdout_sessions": int(holdout_sessions),
        },
        "holdout": [holdout_start, rows - 1],
        "holdout_rows": int(holdout_sessions),
        "holdout_share": round(int(holdout_sessions) / rows, 6) if rows else None,
        "max_label_horizon": max_label_horizon(),
        "minimum_rows": minimum_rows(
            fold_sessions, embargo_sessions, holdout_sessions
        ),
        "legal": not problems,
        "problems": problems,
    }
    if last_validation_row is not None:
        gap = holdout_start - int(last_validation_row) - 1
        described["last_validation_row"] = int(last_validation_row)
        described["gap_to_holdout"] = gap
        # THE GAP IS SEPARATE FROM THE EMBARGO SETTING. A run can be configured
        # with a legal embargo and still leave a short gap if its last fold ran
        # close to the tail, and it is the GAP that determines whether a
        # holdout label leaked.
        described["gap_covers_horizon"] = gap >= max_label_horizon()
        described["overlaps_holdout"] = int(last_validation_row) >= holdout_start
    return described


def evaluate_sealed_holdout(
    runs: Sequence[Mapping[str, Any]] | None,
    *,
    dataset_rows: int | None = None,
    openings: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Can the seal be proved from what is recorded?

    Returns NOT_EVALUATED when it cannot — which is the shipped answer. A seal
    that has to be reconstructed by guessing the geometry is not a seal, so this
    never infers one from a second file the way I had to by hand.
    """
    if not runs:
        return _result(
            HOLDOUT_NOT_EVALUATED,
            "no training runs are recorded, so there is no holdout to seal",
            reason_code=HOLDOUT_REASON_NO_RUNS,
            recorded_facts={fact: False for fact in SEAL_REQUIRED_FACTS},
        )

    # An opening is a one-way event, and it outranks everything below: a holdout
    # read twice is burned whatever its geometry says.
    spent = opening_verdict(openings)
    if spent is not None:
        return spent

    facts = recorded_facts(runs[0])
    missing = [fact for fact, present in facts.items() if not present]
    if missing:
        return _result(
            HOLDOUT_NOT_EVALUATED,
            (
                f"the runs record none of what a seal needs (missing: "
                f"{', '.join(missing)}). MEASURED, recovering the shipped seal "
                f"required a second file plus a guessed geometry, and a seal "
                f"that must be guessed at cannot be audited"
            ),
            reason_code=(
                HOLDOUT_REASON_NO_BOUNDS
                if "holdout_bounds" in missing
                else HOLDOUT_REASON_NO_GEOMETRY
            ),
            recorded_facts=facts,
            missing_facts=missing,
            runs=len(runs),
            # NO BOUNDS AND NO SHARE ARE REPORTED. The shape rule: a `holdout`
            # key here would be a guess wearing a measurement's clothes.
        )

    run = runs[0]
    geometry = run.get("geometry") or {
        "fold_sessions": run.get("fold_sessions"),
        "embargo_sessions": run.get("embargo_sessions"),
        "holdout_sessions": run.get("holdout_sessions"),
    }
    rows = dataset_rows if dataset_rows is not None else run.get("dataset_rows")
    if rows is None:
        rows = run.get("row_count") or run.get("rows")
    described = describe_seal(
        int(rows),
        int(geometry["fold_sessions"]),
        int(geometry["embargo_sessions"]),
        int(geometry["holdout_sessions"]),
        last_validation_row=_last_validation_row(runs),
    )
    if described["overlaps_holdout"]:
        return _result(
            HOLDOUT_BURNED,
            (
                f"a validation window reaches row "
                f"{described['last_validation_row']}, inside the holdout that "
                f"starts at {described['holdout'][0]}: the tail has already "
                f"been scored against and is not an unseen estimate"
            ),
            reason_code=HOLDOUT_REASON_OVERLAP,
            recorded_facts=facts,
            **described,
        )
    if not described["legal"]:
        return _result(
            HOLDOUT_NOT_EVALUATED,
            "; ".join(described["problems"]),
            reason_code=(
                HOLDOUT_REASON_EMBARGO_SHORT
                if int(geometry["embargo_sessions"]) < max_label_horizon()
                else HOLDOUT_REASON_GEOMETRY_ILLEGAL
            ),
            recorded_facts=facts,
            **described,
        )
    if HOLDOUT_REQUIRES_EMBARGO and not described["gap_covers_horizon"]:
        return _result(
            HOLDOUT_NOT_EVALUATED,
            (
                f"the gap between the last validation row and the holdout is "
                f"{described['gap_to_holdout']} rows against a "
                f"{described['max_label_horizon']}-session horizon, so a label "
                f"inside the holdout was computed from prices the last fold "
                f"trained on"
            ),
            reason_code=HOLDOUT_REASON_EMBARGO_SHORT,
            recorded_facts=facts,
            **described,
        )
    return _result(
        HOLDOUT_SEALED,
        (
            f"rows {described['holdout'][0]}..{described['holdout'][1]} "
            f"({described['holdout_share']:.1%} of the data) are recorded, "
            f"embargoed by {described['gap_to_holdout']} rows, and never read"
        ),
        reason_code="",
        recorded_facts=facts,
        openings=0,
        **described,
    )


def _last_validation_row(runs: Sequence[Mapping[str, Any]]) -> int | None:
    """The furthest row any fold validated on, across every run.

    ACROSS EVERY RUN, not the first: the seal is a property of the whole search.
    One estimator reaching into the tail spends it for all of them.
    """
    furthest: int | None = None
    for run in runs or []:
        for fold in run.get("folds") or []:
            window = fold.get("validation")
            if isinstance(window, Sequence) and len(window) == 2:
                end = int(window[1])
                furthest = end if furthest is None else max(furthest, end)
    return furthest


def opening_verdict(
    openings: Sequence[Mapping[str, Any]] | None,
) -> dict | None:
    """BURNED / OPENED when the holdout has been read, else None.

    None means "not opened", which is not the same as "sealed" — the geometry
    still has to hold. Returning a verdict here would let an unrecorded seal
    pass on the strength of never having been read.
    """
    entries = list(openings or [])
    if not entries:
        return None
    if HOLDOUT_OPENING_REQUIRES_REASON:
        anonymous = [
            index
            for index, entry in enumerate(entries)
            if not str((entry or {}).get("reason") or "").strip()
            or not str((entry or {}).get("opened_by") or "").strip()
        ]
        if anonymous:
            return _result(
                HOLDOUT_BURNED,
                (
                    f"{len(anonymous)} opening(s) name no reason or no opener. "
                    f"'Do not use it for further model selection' cannot be "
                    f"enforced against an anonymous read, so the safe reading "
                    f"is that it was used"
                ),
                openings=len(entries),
                anonymous_openings=len(anonymous),
            )
    if len(entries) > HOLDOUT_MAX_OPENINGS:
        return _result(
            HOLDOUT_BURNED,
            (
                f"the holdout was opened {len(entries)} times against a limit "
                f"of {HOLDOUT_MAX_OPENINGS}; after the first read it is no "
                f"longer an unseen estimate of anything"
            ),
            openings=len(entries),
        )
    used_for_selection = [
        entry for entry in entries if (entry or {}).get("used_for_selection")
    ]
    if used_for_selection:
        return _result(
            HOLDOUT_BURNED,
            (
                "the holdout was used for model selection after opening, which "
                "is the one thing a sealed holdout forbids"
            ),
            openings=len(entries),
            used_for_selection=len(used_for_selection),
        )
    entry = entries[0]
    return _result(
        HOLDOUT_OPENED,
        (
            f"opened once by {entry.get('opened_by')!r} for "
            f"{entry.get('reason')!r}; the result stands but the period is "
            f"spent and may not inform further selection"
        ),
        openings=len(entries),
    )


def holdout_problems(report: Mapping[str, Any]) -> list[str]:
    """Contract check on a holdout report. Empty means clean."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return ["report is not a mapping"]
    verdict = report.get("verdict")
    if verdict not in HOLDOUT_VERDICTS:
        problems.append(f"unknown holdout verdict {verdict!r}")
    if not str(report.get("reason") or "").strip():
        problems.append("no reason given")
    if report.get("blocks_trades"):
        problems.append("X8 reports; the registry promotes")
    # THE SHAPE RULE: bounds and a share exist IFF a seal was located. Rendering
    # an unlocated holdout as [0, 0] or a 0.0 share would read as "the seal is
    # empty", which is a different claim from "the seal could not be found".
    if verdict == HOLDOUT_NOT_EVALUATED:
        if not str(report.get("reason_code") or "").strip():
            problems.append("NOT_EVALUATED without a reason code")
        if report.get("legal"):
            problems.append(
                "NOT_EVALUATED but the geometry is marked legal; one of the two "
                "is wrong"
            )
    if verdict == HOLDOUT_SEALED:
        for key in ("holdout", "geometry", "dataset_rows"):
            if report.get(key) is None:
                problems.append(
                    f"SEALED without {key}: a seal that is not recorded cannot "
                    f"be audited, which is the defect X8 measured"
                )
        if report.get("openings"):
            problems.append("SEALED but recorded as opened")
        if HOLDOUT_REQUIRES_EMBARGO and report.get("gap_covers_horizon") is False:
            problems.append(
                "SEALED while the gap to the holdout is shorter than the "
                "longest label horizon"
            )
    if verdict == HOLDOUT_BURNED and HOLDOUT_BURNED_IS_TERMINAL:
        if report.get("legal") and not report.get("openings") and not report.get(
            "overlaps_holdout"
        ):
            problems.append(
                "BURNED but nothing explains how: no openings, no overlap, and "
                "a legal geometry"
            )
    return problems


def load_openings(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the opening ledger. Malformed lines raise — integrity is loud.

    An absent ledger means NEVER OPENED, which is the honest default: there is
    no way to open a holdout without writing here, so absence is evidence.
    """
    if path is None:
        return []
    store = Path(path)
    if not store.exists():
        return []
    entries: list[dict[str, Any]] = []
    for number, line in enumerate(
        store.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw = line.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise SealedHoldoutError(
                f"{store.name} line {number} is not valid JSON: {exc}"
            ) from exc
    return entries


def render_sealed_holdout(report: Mapping[str, Any]) -> list[str]:
    """Human-readable lines. Absent facts render as ABSENT, never as zero."""
    lines = [f"  verdict            {report.get('verdict')}"]
    bounds = report.get("holdout")
    lines.append(
        f"  holdout rows       "
        f"{'ABSENT' if bounds is None else f'{bounds[0]}..{bounds[1]}'}"
    )
    share = report.get("holdout_share")
    lines.append(
        f"  share of data      {'ABSENT' if share is None else f'{share:.1%}'}"
    )
    gap = report.get("gap_to_holdout")
    lines.append(
        f"  gap to holdout     {'ABSENT' if gap is None else f'{gap} rows'}"
        f" (horizon {report.get('max_label_horizon')})"
    )
    facts = report.get("recorded_facts") or {}
    if facts:
        recorded = sum(1 for present in facts.values() if present)
        lines.append(f"  seal facts on disk {recorded} of {len(facts)}")
    lines.append(f"  openings           {report.get('openings', 0)}")
    if str(report.get("reason_code") or "").strip():
        lines.append(f"  reason code        {report['reason_code']}")
    return lines
