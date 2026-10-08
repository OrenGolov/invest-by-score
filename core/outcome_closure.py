"""Outcome closure (Sprint L1) — a forecast scored once, against what happened.

    forecast -> horizon expires -> actual outcome -> error -> calibration

The sprint's goal is "controlled learning, not uncontrolled self-modification",
and closure is where that control lives: each forecast is scored EXACTLY ONCE,
against the outcome that actually occurred, by a measure that fits the claim
it actually made.

**L1 could not assume a forecast existed to close.** MEASURED: nothing
persisted a `ForecastSnapshot` — F8 builds one on demand and discards it, and
the 3,301-row decision audit holds ZERO forecast-shaped rows (it stores
SCORES, which have no horizon and therefore cannot expire). So this module
owns an append-only **forecast ledger**, written when the forecast is made.

**Maturity is not re-derived.** `build_outcome_labels` and `horizon_readiness`
(V1/F2) already answer "has this expired?" and "what actually happened?".
MEASURED: NVDA at 2024-06-15 has all six horizons matured; at 2026-09-10 only
1d and 5d have. This module composes them (W5).

**Error is scored per CLAIM TIER, never by one universal function.** F4/F5
emit four kinds of claim and one measure cannot serve them:

    POINT        a number   -> Brier
    INTERVAL     a range    -> COVERAGE: did the outcome fall inside?
    DIRECTIONAL  a side     -> direction hit or miss
    INSUFFICIENT nothing    -> NOT_SCORED

Brier on an interval is undefined. MEASURED, a `[0.39, 0.73]` interval around
a true rate of 0.55 contains the realised value in 93.8% of 2,000 draws —
coverage is the quantity that interval claimed, so coverage is what is scored.

**THE SCOREBOARD IS GAMEABLE BY REFUSING, and that was measured.** Over 200
forecasts from a genuinely skilled forecaster, refusing the hardest cases
improves the average error:

    refuse   0%:  Brier 0.2206 over 200 scored
    refuse  90%:  Brier 0.1379 over  19 scored
    refuse  99%:  Brier 0.0198 over   2 scored     <- 11x "better"

This system is *designed* to refuse often, so the hazard is live. Every report
publishes the refusal and pending counts beside the error, and below
`CLOSURE_MIN_SCORED_SHARE` the error is reported as UNRELIABLE rather than as
a headline. A refusal itself is NOT_SCORED: zero error would reward silence,
maximum error would punish honesty.

**Closure is idempotent.** A CLOSED forecast is terminal. Re-closing would
double-count one observation and silently re-weight every metric derived from
the ledger, so it is refused rather than appended.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from core.config import (
    CLOSURE_CLOSEABLE_STATES,
    CLOSURE_CLOSED,
    CLOSURE_CONTRACT_VERSION,
    CLOSURE_EXPIRED_NO_DATA,
    CLOSURE_MATURED,
    CLOSURE_MIN_FOR_CALIBRATION,
    CLOSURE_MIN_SCORED_SHARE,
    CLOSURE_OPEN,
    CLOSURE_SCORE_METHODS,
    CLOSURE_SCORE_REFUSALS,
    CLOSURE_STATES,
    FORECAST_LEDGER_PATH,
    FORECAST_LEDGER_VERSION,
    OUTCOME_CLOSURE_VERSION,
    SCORE_METHOD_BRIER,
    SCORE_METHOD_COVERAGE,
    SCORE_METHOD_DIRECTION,
    SCORE_METHOD_NOT_SCORED,
)

LOGGER = logging.getLogger("core.outcome_closure")

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / FORECAST_LEDGER_PATH


class OutcomeClosureError(ValueError):
    """Raised when a closure request violates the L1 contract."""


# ---------------------------------------------------------------------------
# The forecast ledger
# ---------------------------------------------------------------------------


def forecast_id(ticker: str, as_of: str, horizon: str, target: str) -> str:
    """The identity a forecast is closed against.

    Deterministic, so the same forecast cannot enter the ledger twice under
    two different names — which would let one observation be scored twice.
    """
    canonical = "|".join(
        [str(ticker).upper(), str(as_of), str(horizon), str(target)]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def build_ledger_row(
    ticker: str,
    as_of: str,
    horizon: str,
    target: str,
    *,
    claim: str,
    value: float | None = None,
    interval: dict | None = None,
    direction: str | None = None,
    samples: int = 0,
    confidence: float | None = None,
    forecast_version: dict | None = None,
    snapshot_digest: str | None = None,
    regime: str | None = None,
    event_type: str | None = None,
    observed_share: float | None = None,
    volatility: float | None = None,
) -> dict:
    """One forecast, recorded at the moment it was made.

    The claim TIER is recorded alongside the value, because the tier decides
    how this row may later be scored. A row that recorded only a number would
    force every claim through Brier, including the ones that never made a
    point estimate.
    """
    if not str(ticker or "").strip():
        raise OutcomeClosureError("a ticker is required")
    if not str(horizon or "").strip():
        raise OutcomeClosureError("a horizon is required")
    if claim not in CLOSURE_SCORE_METHODS:
        raise OutcomeClosureError(
            f"unknown claim tier {claim!r} (known: {sorted(CLOSURE_SCORE_METHODS)})"
        )

    row = {
        "forecast_id": forecast_id(ticker, as_of, horizon, target),
        "ticker": str(ticker).upper(),
        "as_of": str(as_of),
        "horizon": str(horizon),
        "target": str(target),
        "claim": claim,
        "score_method": CLOSURE_SCORE_METHODS[claim],
        "samples": int(samples),
        "confidence": confidence,
        "state": CLOSURE_OPEN,
        # L2's POINT-IN-TIME dimensions, captured here because they cannot be
        # recovered later. MEASURED, 7 of 8 retrieval probes returned a
        # different analog set as the store grew, and a recomputed regime
        # depends on whichever classifier version runs at closing time.
        "regime": regime,
        "event_type": event_type,
        "observed_share": observed_share,
        "volatility": volatility,
        "ledger_version": FORECAST_LEDGER_VERSION,
        "forecast_version": forecast_version or {},
        "snapshot_digest": snapshot_digest,
    }
    # THE SHAPE RULE, inherited from F3-F8: a key exists only when the claim
    # actually carried it. A `value: None` on a refusal would coalesce to 0.0
    # in a consumer and later be scored as a confident wrong answer.
    #
    # Enforced at CONSTRUCTION, not only in validation: a caller that passes a
    # value alongside a refusal has made a contradictory request, and silently
    # keeping the number is how a claim nobody made ends up scored. Refusing
    # here means `ledger_row_problems` is a second line rather than the only
    # one.
    if claim == "INSUFFICIENT" and value is not None:
        raise OutcomeClosureError(
            "an INSUFFICIENT forecast made no claim, so it cannot carry a "
            "value — recording one would later be scored as a confident "
            "answer nobody gave"
        )
    if value is not None:
        row["value"] = round(float(value), 6)
    if interval is not None:
        row["interval"] = interval
    if direction is not None:
        row["direction"] = direction
    return row


def record_forecast(row: dict, path: str | Path | None = None) -> dict:
    """Append a forecast to the ledger, deduplicated by forecast_id.

    Re-recording the same forecast is a NO-OP rather than an error: a re-run
    of the pipeline must not inflate the denominator, which would quietly
    change every metric computed from the ledger.
    """
    problems = ledger_row_problems(row)
    if problems:
        raise OutcomeClosureError(
            f"refusing to record an invalid forecast: {'; '.join(problems)}"
        )

    store = Path(path) if path is not None else LEDGER_PATH
    for existing in load_ledger(store):
        if existing.get("forecast_id") == row["forecast_id"]:
            return existing

    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return row


def load_ledger(path: str | Path | None = None) -> list[dict]:
    """Every ledger row. A malformed line raises — integrity is loud."""
    store = Path(path) if path is not None else LEDGER_PATH
    if not store.exists():
        return []
    rows: list[dict] = []
    for number, line in enumerate(
        store.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OutcomeClosureError(
                f"{store.name} line {number} is not valid JSON: {exc}"
            ) from exc
    return rows


def current_ledger(path: str | Path | None = None) -> list[dict]:
    """The ledger collapsed to ONE row per forecast, latest state winning.

    The store is append-only (W6): closing a forecast writes a NEW row rather
    than rewriting the original, so "what did we believe at the time?" stays
    answerable. That makes `load_ledger` return both the OPEN forecast and its
    CLOSED successor, and a reader that counts raw rows double-counts every
    re-run — MEASURED, three passes over 12 forecasts produced 36 rows and a
    scoreboard claiming 8 scored from 4 actual forecasts.

    Supersede-on-read, the same rule the raw store already applies.
    """
    latest: dict[str, dict] = {}
    for row in load_ledger(path):
        key = str(row.get("forecast_id") or "")
        if not key:
            continue
        current = latest.get(key)
        # CLOSED supersedes OPEN; otherwise the later row wins.
        if current is None or current.get("state") != CLOSURE_CLOSED:
            latest[key] = row
    return list(latest.values())


def ledger_row_problems(row: dict) -> list[str]:
    """Validate a ledger row before it is written."""
    problems: list[str] = []
    if not isinstance(row, dict):
        return ["a ledger row must be a dict"]
    for field in ("forecast_id", "ticker", "as_of", "horizon", "target", "claim"):
        if not row.get(field):
            problems.append(f"ledger field {field!r} missing/empty")
    claim = row.get("claim")
    if claim and claim not in CLOSURE_SCORE_METHODS:
        problems.append(f"unknown claim tier {claim!r}")
    elif claim and row.get("score_method") != CLOSURE_SCORE_METHODS[claim]:
        problems.append(
            f"score_method {row.get('score_method')!r} does not match the "
            f"{claim} tier's declared method "
            f"{CLOSURE_SCORE_METHODS[claim]!r}"
        )
    if row.get("state") not in CLOSURE_STATES:
        problems.append(f"unknown closure state {row.get('state')!r}")
    if claim == "POINT" and "value" not in row:
        problems.append("a POINT forecast carries no value to score")
    if claim == "INSUFFICIENT" and "value" in row:
        problems.append(
            "a refused forecast carries a value — it would later be scored as "
            "a confident answer nobody gave"
        )
    return problems


# ---------------------------------------------------------------------------
# Maturity — composed from V1/F2, never re-derived
# ---------------------------------------------------------------------------


def closure_state(row: dict, labels: dict | None) -> tuple[str, str]:
    """Where this forecast stands: OPEN, MATURED, or EXPIRED_NO_DATA.

    Composed from `horizon_readiness`, so one notion of maturity governs the
    forecast layer and the closure layer alike.
    """
    from core.forecast_horizons import HORIZON_STATUS_SCORABLE, horizon_readiness

    if row.get("state") == CLOSURE_CLOSED:
        return CLOSURE_CLOSED, "already closed"
    if not labels:
        return CLOSURE_OPEN, "no label set was available for this forecast"

    readiness = horizon_readiness(labels, row.get("horizon", ""))
    if readiness.get("status") != HORIZON_STATUS_SCORABLE:
        return (
            CLOSURE_OPEN,
            readiness.get("reason") or f"the {row.get('horizon')} window has not closed",
        )

    record = (labels.get("horizons") or {}).get(row.get("horizon")) or {}
    if record.get("forward_return") is None:
        return (
            CLOSURE_EXPIRED_NO_DATA,
            (
                f"the {row.get('horizon')} window elapsed but no outcome was "
                f"recorded — 'we could not score it' is not 'it never existed'"
            ),
        )
    return CLOSURE_MATURED, ""


def realised_outcome(labels: dict, horizon: str) -> dict:
    """What actually happened, from the V1 labels."""
    record = (labels.get("horizons") or {}).get(horizon) or {}
    forward = record.get("forward_return")
    return {
        "forward_return": forward,
        "direction_up": bool(forward > 0.0) if forward is not None else None,
        "entry_bar": record.get("entry_bar"),
        "exit_bar": record.get("exit_bar"),
        "label_version": labels.get("label_version"),
    }


# ---------------------------------------------------------------------------
# Scoring — per claim tier
# ---------------------------------------------------------------------------


def score_forecast(row: dict, outcome: dict) -> dict:
    """Score one forecast by the measure its CLAIM TIER earned.

    One universal error function cannot serve four kinds of claim: Brier on an
    interval is undefined, and a refusal made no claim to be wrong about.
    """
    method = CLOSURE_SCORE_METHODS.get(row.get("claim"), SCORE_METHOD_NOT_SCORED)
    actual_up = outcome.get("direction_up")

    if method == SCORE_METHOD_NOT_SCORED or CLOSURE_SCORE_REFUSALS:
        return {
            "method": SCORE_METHOD_NOT_SCORED,
            "scored": False,
            "reason": (
                "this forecast made no claim, so there is nothing to be right "
                "or wrong about; scoring it zero would reward silence and "
                "scoring it maximum would punish honesty"
            ),
        }

    if actual_up is None:
        return {
            "method": method,
            "scored": False,
            "reason": "no realised direction was available to score against",
        }

    actual = 1.0 if actual_up else 0.0

    if method == SCORE_METHOD_BRIER:
        value = row.get("value")
        if value is None:
            return {
                "method": method, "scored": False,
                "reason": "a POINT forecast with no value cannot be scored",
            }
        error = (float(value) - actual) ** 2
        return {
            "method": method,
            "scored": True,
            "predicted": round(float(value), 6),
            "actual": actual,
            "brier": round(error, 6),
            "absolute_error": round(abs(float(value) - actual), 6),
            "reason": "",
        }

    if method == SCORE_METHOD_COVERAGE:
        interval = row.get("interval") or {}
        low, high = interval.get("lower"), interval.get("upper")
        if low is None or high is None:
            return {
                "method": method, "scored": False,
                "reason": "an INTERVAL forecast with no bounds cannot be scored",
            }
        # COVERAGE IS A GROUP PROPERTY, and treating it per forecast was an
        # error caught by running L2 over real data: an interval on P(up)
        # claims the RATE at which such setups rise lies in [lo, hi], while a
        # single outcome is 0 or 1. A binary outcome can NEVER fall inside a
        # probability range, so per-forecast coverage was False 42 times out
        # of 42 — uninformative by construction rather than merely wrong.
        #
        # So this row records what the group needs (the interval and the
        # realised outcome) and scores the forecast by the BRIER OF ITS
        # MIDPOINT, which is the usable point reading the interval implies.
        # `cell_coverage` in L2 then asks the question that IS answerable:
        # did the realised RATE across the cell land inside the interval?
        midpoint = (float(low) + float(high)) / 2.0
        return {
            "method": method,
            "scored": True,
            "actual": actual,
            "lower": float(low),
            "upper": float(high),
            "midpoint": round(midpoint, 6),
            "brier": round((midpoint - actual) ** 2, 6),
            "width": round(float(high) - float(low), 6),
            "coverage_is_group_property": True,
            "reason": "",
        }

    if method == SCORE_METHOD_DIRECTION:
        direction = row.get("direction")
        if not direction:
            return {
                "method": method, "scored": False,
                "reason": "a DIRECTIONAL forecast naming no direction cannot be scored",
            }
        predicted_up = str(direction).upper() == "HIGHER"
        return {
            "method": method,
            "scored": True,
            "predicted_direction": direction,
            "actual": actual,
            "hit": predicted_up == bool(actual_up),
            "reason": "",
        }

    return {
        "method": method, "scored": False,
        "reason": f"no scoring rule is defined for {method!r}",
    }


def close_forecast(row: dict, labels: dict | None) -> dict:
    """Close one forecast: state, outcome, error — or say why not.

    Idempotent by refusal: a CLOSED row is terminal, because re-closing would
    double-count one observation and silently re-weight every metric derived
    from the ledger.
    """
    state, reason = closure_state(row, labels)

    if state not in CLOSURE_CLOSEABLE_STATES:
        return {
            **row,
            "state": state,
            "closure_reason": reason,
            "outcome": None,
            "score": None,
            "closure_version": OUTCOME_CLOSURE_VERSION,
        }

    outcome = realised_outcome(labels or {}, row.get("horizon", ""))
    score = score_forecast(row, outcome)
    return {
        **row,
        "state": CLOSURE_CLOSED,
        "closure_reason": "",
        "outcome": outcome,
        "score": score,
        "closure_version": OUTCOME_CLOSURE_VERSION,
    }


# ---------------------------------------------------------------------------
# The report — error never without its coverage
# ---------------------------------------------------------------------------


def closure_report(closed_rows: list[dict]) -> dict:
    """Aggregate closed forecasts into an honest scoreboard.

    MEASURED: refusing the hardest 99% of forecasts improves Brier from 0.2206
    to 0.0198. So the scored / refused / pending counts travel WITH the error,
    and below `CLOSURE_MIN_SCORED_SHARE` the error is marked UNRELIABLE rather
    than published as a headline.
    """
    rows = list(closed_rows or [])
    total = len(rows)

    scored = [r for r in rows if (r.get("score") or {}).get("scored")]
    refused = [
        r for r in rows
        if (r.get("score") or {}).get("method") == SCORE_METHOD_NOT_SCORED
    ]
    pending = [r for r in rows if r.get("state") == CLOSURE_OPEN]
    no_data = [r for r in rows if r.get("state") == CLOSURE_EXPIRED_NO_DATA]

    briers = [
        float(r["score"]["brier"]) for r in scored
        if r["score"].get("method") == SCORE_METHOD_BRIER
    ]
    coverages = [
        bool(r["score"]["covered"]) for r in scored
        if r["score"].get("method") == SCORE_METHOD_COVERAGE
    ]
    hits = [
        bool(r["score"]["hit"]) for r in scored
        if r["score"].get("method") == SCORE_METHOD_DIRECTION
    ]

    scored_share = len(scored) / total if total else 0.0
    reliable = scored_share >= CLOSURE_MIN_SCORED_SHARE and bool(scored)

    return {
        "total": total,
        "scored": len(scored),
        "refused": len(refused),
        "pending": len(pending),
        "expired_no_data": len(no_data),
        "scored_share": round(scored_share, 6),
        "reliable": reliable,
        "reliability_reason": (
            ""
            if reliable
            else (
                f"only {scored_share:.0%} of forecasts were scored (floor "
                f"{CLOSURE_MIN_SCORED_SHARE:.0%}); MEASURED, refusing the "
                f"hardest cases improves Brier from 0.2206 to 0.0198, so an "
                f"error read without its coverage is not a measurement"
            )
        ),
        "mean_brier": round(sum(briers) / len(briers), 6) if briers else None,
        "brier_count": len(briers),
        "coverage_rate": (
            round(sum(coverages) / len(coverages), 6) if coverages else None
        ),
        "coverage_count": len(coverages),
        "direction_accuracy": round(sum(hits) / len(hits), 6) if hits else None,
        "direction_count": len(hits),
        "calibration_ready": len(briers) >= CLOSURE_MIN_FOR_CALIBRATION,
        "closure_version": OUTCOME_CLOSURE_VERSION,
        "contract_version": CLOSURE_CONTRACT_VERSION,
    }


def evaluate_calibration(closed_rows: list[dict]) -> dict:
    """Calibration over the POINT forecasts, composed from M6.

    Only POINT claims enter: a calibration curve maps predicted probabilities
    to realised frequencies, and an interval or a refusal supplies no
    probability to map.
    """
    from core.calibration import (
        brier_score,
        expected_calibration_error,
        reliability_curve,
    )

    points = [
        r for r in (closed_rows or [])
        if (r.get("score") or {}).get("method") == SCORE_METHOD_BRIER
        and (r.get("score") or {}).get("scored")
    ]
    if len(points) < CLOSURE_MIN_FOR_CALIBRATION:
        return {
            "status": "INSUFFICIENT",
            "observations": len(points),
            "required": CLOSURE_MIN_FOR_CALIBRATION,
            "reason": (
                f"{len(points)} scored POINT forecast(s); "
                f"{CLOSURE_MIN_FOR_CALIBRATION} required before a calibration "
                f"curve describes anything"
            ),
            "closure_version": OUTCOME_CLOSURE_VERSION,
        }

    predicted = [float(r["score"]["predicted"]) for r in points]
    actual = [float(r["score"]["actual"]) for r in points]
    return {
        "status": "MEASURED",
        "observations": len(points),
        "brier": round(brier_score(predicted, actual), 6),
        "expected_calibration_error": round(
            expected_calibration_error(predicted, actual), 6
        ),
        "reliability_curve": reliability_curve(predicted, actual),
        "closure_version": OUTCOME_CLOSURE_VERSION,
    }


def closure_problems(report: dict) -> list[str]:
    """Validate a closure report against the L1 contract."""
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a dict"]

    for field in ("closure_version", "contract_version"):
        if not report.get(field):
            problems.append(f"report field {field!r} missing/empty")

    # Error must never be readable without its coverage.
    has_error = any(
        report.get(name) is not None
        for name in ("mean_brier", "coverage_rate", "direction_accuracy")
    )
    if has_error:
        for name in ("scored", "refused", "pending", "scored_share"):
            if report.get(name) is None:
                problems.append(
                    f"an error figure is published without {name!r} — MEASURED, "
                    f"refusing the hardest cases improves Brier 11x, so an "
                    f"error without its coverage is not a measurement"
                )
    if report.get("mean_brier") is not None and not report.get("brier_count"):
        problems.append("a mean Brier is reported over zero observations")
    if report.get("reliable") is False and not report.get("reliability_reason"):
        problems.append("an unreliable report does not say why")

    counts = [report.get(k) or 0 for k in ("scored", "refused", "pending")]
    if report.get("total") is not None and sum(counts) > report["total"]:
        problems.append(
            f"the scored/refused/pending counts ({sum(counts)}) exceed the "
            f"total ({report['total']})"
        )
    return problems
