"""X8 sealed holdout — the seal held, but nothing was holding it.

"Maintain an untouched final period; once opened, never use it for further model
selection." MEASURED, the good news first: **a tail holdout exists and was never
opened.** The eight shipped runs in ``data/training_runs.jsonl`` record no
holdout, seal or final key — their metrics are ``directional_accuracy``,
``folds``, ``mae``, ``observations`` and ``rmse``. The final period is genuinely
untouched.

The geometry that produced those runs, recomputed exactly (406-row dataset
``d18d5e5e``, pre-F2 geometry: fold 120, embargo 60, holdout 60):

=================================  =================
fold 0 train                       rows [0, 119]
fold 0 validation                  rows [180, 299]
declared holdout                   rows [346, 405]
last row validation touched        299
**untouched tail**                 **106 rows**
=================================  =================

Three things are nevertheless wrong, and they get worse in order.

**Finding 1 — "once" is a string, not a mechanism.** ``core/training.py``
iterates ``geometry["folds"]`` and never reads ``geometry["holdout"]``; the
backtest engine labels its result ``{"evaluation": "holdout_once"}``. Nothing
*records* that a holdout was opened, so nothing can *refuse* a second opening.
The seal held because nobody pushed on it.

**Finding 2 — the seal boundary cannot be recomputed.** The embargo floor is
``max(LABEL_HORIZON_SESSIONS)``. F2 added the 252d horizon on 2026-09-19; the
runs were written at Sprint C on 2026-09-18, when the floor was 60. The current
code therefore *refuses* the geometry that produced the shipped runs::

    ValueError: embargo (60) must be >= the max label horizon (252)

and over 406 rows ``fold_sessions=120`` forces ``embargo + holdout <= 166 <
252``, so no current geometry reproduces that boundary at all. A boundary derived
from live config moves when config moves — under a seal that is supposed to be
fixed. So X8 verifies the boundary against a **recorded** seal and never
recomputes it.

**Finding 3 — and the seal is too small to decide anything.** Measured against
the bars this repo already set:

============================  =====  ========
bar                           value  verdict
============================  =====  ========
untouched tail                  106  —
declared holdout window          60  —
``OOS_MIN_OBSERVATIONS``        120  both FAIL
``CALIBRATION_GATE_MIN_HOLDOUT``  200  both FAIL
============================  =====  ========

The sampling band ``1.96*sqrt(0.25/n)`` is ±0.1265 at n=60 and ±0.0952 at n=106.
**At 60 rows a fair coin posts up to 62.6% directional accuracy inside the
band.** Opening this holdout would spend the only unopened evidence the project
has and buy a number that cannot separate an edge from a coin flip. So X8 reports
``INSUFFICIENT`` and keeps it **sealed** — which is both the honest answer and
the useful one, because the seal remains available once the dataset grows.

**A seal that does not exist is ABSENT, never "unopened."** The two are opposite
facts: one means the evidence is intact, the other means there is none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.config import (
    HOLDOUT_ABSENT,
    HOLDOUT_BOUNDARY_IS_RECORDED,
    HOLDOUT_FORBIDS_SELECTION_AFTER_OPENING,
    HOLDOUT_INSUFFICIENT,
    HOLDOUT_INSUFFICIENT_STAYS_SEALED,
    HOLDOUT_MAX_OPENINGS,
    HOLDOUT_MIN_OBSERVATIONS,
    HOLDOUT_OPENED,
    HOLDOUT_STATES,
    HOLDOUT_UNOPENED,
    SEALED_HOLDOUT_BLOCKS_TRADES,
    SEALED_HOLDOUT_VERSION,
)


class SealedHoldoutError(Exception):
    """Raised when a seal is malformed or an opening rule is violated."""


def sampling_band(observations: int) -> float:
    """The half-width a fair coin can reach at ``observations`` draws.

    ``1.96 * sqrt(0.25/n)`` — the same band X1 uses, so "inside the noise" means
    one thing across the release gate. Raises on a non-positive count rather
    than returning an infinite band, because "no observations" is a different
    statement from "a very wide band".
    """
    count = int(observations)
    if count < 1:
        raise SealedHoldoutError(
            f"a sampling band needs at least one observation, got {count}"
        )
    return 1.96 * (0.25 / count) ** 0.5


def seal_problems(seal: Mapping[str, Any] | None) -> list[str]:
    """Contract check for a recorded seal. Empty list means usable.

    A seal must carry its own boundary (finding 2): the row range it covers and
    the row count it holds. It must NOT be reconstructible from live config,
    because the config that produced the shipped seal no longer validates.
    """
    if seal is None:
        return ["no seal record"]
    if not isinstance(seal, Mapping):
        return [f"a seal must be a mapping, got {type(seal).__name__}"]

    problems: list[str] = []
    for field in ("dataset_hash", "sealed_rows", "row_count", "sealed_at"):
        if field not in seal:
            problems.append(f"seal is missing {field!r}")

    rows = seal.get("sealed_rows")
    if rows is not None:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            problems.append("sealed_rows must be a [start, end] pair")
        elif len(rows) != 2:
            problems.append(
                f"sealed_rows must be a [start, end] pair, got {len(rows)} values"
            )
        else:
            start, end = int(rows[0]), int(rows[1])
            if start > end:
                problems.append(
                    f"sealed_rows [{start}, {end}] runs backwards"
                )
            declared = seal.get("row_count")
            if declared is not None and int(declared) != end - start + 1:
                # The recorded count and the recorded range must agree, or the
                # seal describes two different windows and neither can be
                # verified against the data.
                problems.append(
                    f"row_count {int(declared)} does not match sealed_rows "
                    f"[{start}, {end}] ({end - start + 1} rows)"
                )

    count = seal.get("row_count")
    if count is not None and int(count) < 1:
        problems.append(f"a seal covering {int(count)} rows covers nothing")

    openings = seal.get("openings", [])
    if not isinstance(openings, Sequence) or isinstance(openings, (str, bytes)):
        problems.append("openings must be a sequence")
    elif len(openings) > HOLDOUT_MAX_OPENINGS:
        problems.append(
            f"{len(openings)} openings recorded, at most "
            f"{HOLDOUT_MAX_OPENINGS} permitted — the second look is where "
            f"selection bias enters"
        )
    else:
        for index, opening in enumerate(openings):
            if not isinstance(opening, Mapping):
                problems.append(f"opening {index} is not a mapping")
                continue
            for field in ("opened_at", "opened_by", "reason"):
                if not opening.get(field):
                    problems.append(f"opening {index} is missing {field!r}")
    return problems


def holdout_state(seal: Mapping[str, Any] | None) -> str:
    """The factual state of a seal: ABSENT, OPENED, INSUFFICIENT or UNOPENED.

    Order matters and is deliberate. ABSENT first: with no seal there is nothing
    to size. OPENED next: an opened seal is opened whatever its size, and
    reporting a too-small opened seal as INSUFFICIENT would hide that the
    evidence is already spent. INSUFFICIENT only then, for an intact seal below
    the floor.
    """
    if seal is None:
        return HOLDOUT_ABSENT
    problems = seal_problems(seal)
    if problems:
        raise SealedHoldoutError(
            "cannot state the holdout state of an invalid seal: "
            + "; ".join(problems)
        )
    if len(seal.get("openings", [])) > 0:
        return HOLDOUT_OPENED
    if int(seal["row_count"]) < HOLDOUT_MIN_OBSERVATIONS:
        return HOLDOUT_INSUFFICIENT
    return HOLDOUT_UNOPENED


def verify_boundary(seal: Mapping[str, Any], touched_through: int | None) -> dict:
    """Check that no validation window reached into the sealed rows.

    ``touched_through`` is the last row index any training or validation fold
    consumed. The check compares it against the RECORDED seal start, never
    against a recomputed geometry: finding 2 measured that the geometry which
    produced the shipped seal no longer validates under the current embargo
    floor, so a recomputed boundary would move under the seal.
    """
    problems = seal_problems(seal)
    if problems:
        raise SealedHoldoutError(
            "cannot verify the boundary of an invalid seal: " + "; ".join(problems)
        )
    start = int(seal["sealed_rows"][0])
    if touched_through is None:
        # Not "nothing was touched" — nobody recorded what was touched. A
        # boundary that cannot be checked is NOT_VERIFIED, never a pass.
        return {
            "verified": False,
            "reason": (
                "no fold records which row the training run consumed last, so "
                "the boundary cannot be checked"
            ),
            "sealed_from": start,
            "touched_through": None,
            "gap_rows": None,
        }
    touched = int(touched_through)
    intact = touched < start
    return {
        "verified": intact,
        "reason": (
            f"validation reached row {touched} and the seal starts at {start}"
            if intact
            else f"validation reached row {touched}, INSIDE the seal at {start}"
        ),
        "sealed_from": start,
        "touched_through": touched,
        "gap_rows": start - touched - 1,
    }


def evaluate_sealed_holdout(
    seal: Mapping[str, Any] | None,
    touched_through: int | None = None,
) -> dict:
    """The X8 report: is there a seal, is it intact, and can it decide anything?

    Reports rather than repairs. A seal below the floor stays sealed, because
    opening it would consume the evidence and return a number inside the noise
    band (finding 3).
    """
    state = holdout_state(seal)
    report: dict[str, Any] = {
        "version": SEALED_HOLDOUT_VERSION,
        "state": state,
        "min_observations": HOLDOUT_MIN_OBSERVATIONS,
        "max_openings": HOLDOUT_MAX_OPENINGS,
        "boundary_is_recorded": HOLDOUT_BOUNDARY_IS_RECORDED,
        "forbids_selection_after_opening": HOLDOUT_FORBIDS_SELECTION_AFTER_OPENING,
        "insufficient_stays_sealed": HOLDOUT_INSUFFICIENT_STAYS_SEALED,
        "blocks_trades": SEALED_HOLDOUT_BLOCKS_TRADES,
    }

    if state == HOLDOUT_ABSENT:
        report.update({
            "row_count": None,
            "sampling_band": None,
            "openings": 0,
            "may_open": False,
            "boundary": None,
            "reason": (
                "no seal is recorded: no final period is being held, which is "
                "not the same as holding one that has not been opened"
            ),
        })
        return report

    assert seal is not None  # holdout_state returned ABSENT for None
    count = int(seal["row_count"])
    openings = list(seal.get("openings", []))
    band = sampling_band(count)
    boundary = verify_boundary(seal, touched_through)

    # May this seal be opened now? Only an intact seal, at or above the floor,
    # whose boundary has been verified. Each "no" is a different reason and is
    # reported as such rather than collapsed into a single refusal.
    if openings:
        may_open = False
        reason = (
            f"already opened {len(openings)} time(s); a sealed holdout is "
            f"opened exactly once and may never inform model selection again"
        )
    elif count < HOLDOUT_MIN_OBSERVATIONS:
        may_open = False
        reason = (
            f"{count} sealed rows is below the {HOLDOUT_MIN_OBSERVATIONS} floor; "
            f"at n={count} the sampling band is +/-{band:.4f}, so a fair coin "
            f"reaches {0.5 + band:.1%} directional accuracy - opening it would "
            f"spend the evidence for a number inside the noise, so it STAYS SEALED"
        )
    elif not boundary["verified"]:
        may_open = False
        reason = f"boundary not verified: {boundary['reason']}"
    else:
        may_open = True
        reason = (
            f"{count} sealed rows, boundary verified ({boundary['reason']}); "
            f"opening is permitted exactly once"
        )

    report.update({
        "dataset_hash": seal.get("dataset_hash"),
        "sealed_rows": [int(seal["sealed_rows"][0]), int(seal["sealed_rows"][1])],
        "row_count": count,
        "sampling_band": round(band, 6),
        "coin_flip_ceiling": round(0.5 + band, 6),
        "openings": len(openings),
        "opening_records": [dict(opening) for opening in openings],
        "boundary": boundary,
        "may_open": may_open,
        "reason": reason,
    })
    return report


def record_opening(
    seal: Mapping[str, Any],
    opened_by: str,
    reason: str,
    opened_at: str,
    touched_through: int | None = None,
) -> dict:
    """Return the seal with one opening appended, or raise.

    Refuses rather than logs. A second opening is not a second data point: it is
    the point at which the holdout stops being a holdout, so it must fail loudly
    rather than accumulate.
    """
    report = evaluate_sealed_holdout(seal, touched_through)
    if not report["may_open"]:
        raise SealedHoldoutError(f"refusing to open the holdout: {report['reason']}")
    for field, value in (("opened_by", opened_by), ("reason", reason), ("opened_at", opened_at)):
        if not str(value or "").strip():
            raise SealedHoldoutError(
                f"an opening must record {field!r}: an unattributed opening "
                f"cannot be audited"
            )
    updated = dict(seal)
    updated["openings"] = [
        *[dict(item) for item in seal.get("openings", [])],
        {"opened_at": opened_at, "opened_by": opened_by, "reason": reason},
    ]
    return updated


def sealed_holdout_problems(report: Mapping[str, Any] | None) -> list[str]:
    """Contract check for an X8 report. Empty list means the report is honest."""
    if report is None:
        return ["no sealed-holdout report"]
    if not isinstance(report, Mapping):
        return [f"a report must be a mapping, got {type(report).__name__}"]

    problems: list[str] = []
    if report.get("version") != SEALED_HOLDOUT_VERSION:
        problems.append(
            f"report version {report.get('version')!r} is not "
            f"{SEALED_HOLDOUT_VERSION!r}"
        )
    state = report.get("state")
    if state not in HOLDOUT_STATES:
        problems.append(f"{state!r} is not a declared holdout state")
    if report.get("blocks_trades"):
        problems.append("X8 reports; the registry promotes")
    if not report.get("boundary_is_recorded"):
        problems.append(
            "the report must state that the boundary is recorded, not derived"
        )
    if not report.get("forbids_selection_after_opening"):
        problems.append(
            "the report must state that an opened holdout cannot inform "
            "model selection"
        )

    count = report.get("row_count")
    floor = report.get("min_observations")

    if state == HOLDOUT_ABSENT:
        if count is not None:
            problems.append("an absent seal cannot report a row count")
        if report.get("may_open"):
            problems.append("an absent seal cannot be opened")
        return problems

    if not isinstance(count, int) or count < 1:
        problems.append(f"a present seal must report a positive row count, got {count!r}")
        return problems
    if not isinstance(floor, int) or floor < 1:
        problems.append(f"the observation floor must be a positive int, got {floor!r}")
        return problems

    openings = report.get("openings")
    if not isinstance(openings, int) or openings < 0:
        problems.append(f"openings must be a non-negative int, got {openings!r}")
    elif openings > HOLDOUT_MAX_OPENINGS:
        problems.append(
            f"{openings} openings exceeds the {HOLDOUT_MAX_OPENINGS} permitted"
        )

    # The state must follow from the numbers, not be asserted alongside them.
    if state == HOLDOUT_OPENED and not openings:
        problems.append("state is OPENED but no opening is recorded")
    if state == HOLDOUT_UNOPENED and openings:
        problems.append(f"state is UNOPENED but {openings} opening(s) are recorded")
    if state == HOLDOUT_INSUFFICIENT and count >= floor:
        problems.append(
            f"state is INSUFFICIENT but {count} rows meets the {floor} floor"
        )
    if state == HOLDOUT_UNOPENED and count < floor:
        problems.append(
            f"state is UNOPENED but {count} rows is below the {floor} floor - "
            f"an intact seal too small to decide is INSUFFICIENT"
        )

    if report.get("may_open") and openings:
        problems.append("an already-opened holdout may never be opened again")
    if report.get("may_open") and count < floor:
        problems.append(
            f"may_open is set on a {count}-row seal below the {floor} floor: "
            f"an insufficient seal stays sealed"
        )

    band = report.get("sampling_band")
    if band is None:
        problems.append("a present seal must report its sampling band")
    else:
        expected = sampling_band(count)
        if abs(float(band) - expected) > 1e-6:
            problems.append(
                f"sampling band {band} does not match "
                f"1.96*sqrt(0.25/{count}) = {expected:.6f}"
            )

    boundary = report.get("boundary")
    if not isinstance(boundary, Mapping):
        problems.append("a present seal must report a boundary check")
    else:
        if boundary.get("verified") is not True and report.get("may_open"):
            problems.append(
                "may_open is set while the boundary is not verified"
            )
        touched = boundary.get("touched_through")
        start = boundary.get("sealed_from")
        if (
            isinstance(touched, int)
            and isinstance(start, int)
            and touched >= start
            and boundary.get("verified") is True
        ):
            problems.append(
                f"boundary reports verified while validation reached row "
                f"{touched} inside the seal starting at {start}"
            )
    return problems


def render_sealed_holdout(report: Mapping[str, Any]) -> str:
    """Plain-text X8 report. An insufficient seal says so, in rows."""
    problems = sealed_holdout_problems(report)
    if problems:
        raise SealedHoldoutError(
            "refusing to render an invalid X8 report: " + "; ".join(problems)
        )
    lines = [
        "SEALED HOLDOUT (X8)",
        f"  version           : {report['version']}",
        f"  state             : {report['state']}",
    ]
    if report["state"] == HOLDOUT_ABSENT:
        lines.append(f"  reason            : {report['reason']}")
        return "\n".join(lines)

    rows = report["sealed_rows"]
    lines.extend([
        f"  dataset           : {str(report.get('dataset_hash') or 'unknown')[:16]}",
        f"  sealed rows       : [{rows[0]}, {rows[1]}] ({report['row_count']} rows)",
        f"  floor             : {report['min_observations']} observations",
        f"  sampling band     : +/-{report['sampling_band']:.4f} "
        f"(a coin reaches {report['coin_flip_ceiling']:.1%})",
        f"  openings          : {report['openings']} of {report['max_openings']}",
        f"  boundary verified : {report['boundary']['verified']}",
        f"    {report['boundary']['reason']}",
        f"  may open          : {report['may_open']}",
        f"  reason            : {report['reason']}",
    ])
    for opening in report.get("opening_records", []):
        lines.append(
            f"  opened {opening.get('opened_at')} by {opening.get('opened_by')}: "
            f"{opening.get('reason')}"
        )
    return "\n".join(lines)
