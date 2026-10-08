"""Framing contract (Sprint V8) — a backtest is evidence about historical
behavior under explicit assumptions, not proof the future behaves the same way.

Every completed backtest run carries a versioned framing block alongside its
manifest and monitoring snapshot. The block:

- states the canonical interpretation (historical evidence, not future proof);
- lists the explicit assumptions the run actually used (cost table, price
  basis, universe, fold geometry, strategy, data digest — pulled from the
  manifest so the framing can never disagree with the run it frames);
- lists the accepted limitations of the simulation model (no intraday
  timing, no partial fills, price-return basis, etc.);
- lists what the run explicitly does NOT claim — strong historical results
  never upgrade the framing (test-pinned).

Fail-closed: `framing_problems(manifest)` refuses to call a run framable when
a required assumption is missing — evidence without its assumptions stated is
not framed evidence. Everything here is pure and deterministic; the store is
append-only, idempotent per run hash.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import FRAMING_VERSION

FRAMING_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "framing.jsonl"

# Canonical interpretation: the one sentence that never changes with
# performance. Acceptance test-pinned against upgrading wording.
FRAMING_STATEMENT = (
    "This backtest is evidence about historical behavior under the explicit "
    "assumptions listed in this block. It is not proof that the future will "
    "behave the same way."
)

# Accepted limitations of the simulation model (V3 costs + V6 price basis,
# kept as an explicit, versioned list so they are never rediscovered silently).
FRAMING_LIMITATIONS = {
    "no_intraday_timing": "Decisions made at bar t act at bar t+1 open; no intraday timing is modeled.",
    "no_partial_fills": "Orders fill at full notional; partial fills and queue position are not modeled.",
    "no_borrow_or_rebate_costs": "Shorting borrow costs and stock-lending rebates are not modeled.",
    "open_gap_risk": "Gap risk between decision and execution is borne by the strategy.",
    "price_return_basis": "Returns are price returns on split-adjusted closes; dividends are not reinvested.",
    "no_exchange_fees_or_taxes": "Exchange fees, rebates, and taxes are not modeled.",
    "pit_only": "Only point-in-time information is used; no future or revised values may enter the run.",
}

# Explicit non-claims. The verb "does_not_claim" is a contract: these bullets
# are the only framing vocabulary a run may use about its own meaning.
FRAMING_NON_CLAIMS = {
    "not_proof_of_future": (
        "Strong historical results are not evidence that the future will "
        "behave the same way."
    ),
    "not_guarantee": "This is not a guarantee of future returns, nor a trade recommendation.",
    "not_generalization": (
        "The result describes the tested window under the tested assumptions; "
        "it does not claim the edge generalizes to unseen periods, markets, "
        "or asset classes."
    ),
    "not_release_gate_waiver": (
        "Nothing here waives the release gates: out-of-sample validation, "
        "calibration, and regime/event robustness are still required before "
        "any forecasting claim."
    ),
}

# The manifest surface every framable run must expose.
REQUIRED_ASSUMPTION_KEYS = (
    "cost_table",
    "strategy",
    "price_basis",
    "universe",
    "embargo_sessions",
    "fold_sessions",
    "holdout_sessions",
    "enter_score",
    "exit_score",
    "data_digest",
)


def assumption_entries(manifest: dict) -> list[dict]:
    """Explicit assumptions the run actually used, sourced from the manifest."""
    versions = manifest.get("versions") or {}
    config = manifest.get("config") or {}
    provider = manifest.get("provider_overrides") or {}
    return [
        {"key": "cost_table", "value": versions.get("cost_table"),
         "sourced_from": "manifest.versions.cost_table"},
        {"key": "strategy", "value": versions.get("strategy"),
         "sourced_from": "manifest.versions.strategy"},
        {"key": "price_basis", "value": provider.get("price_basis"),
         "sourced_from": "manifest.provider_overrides.price_basis"},
        {"key": "universe", "value": config.get("universe"),
         "sourced_from": "manifest.config.universe"},
        {"key": "embargo_sessions", "value": config.get("embargo_sessions"),
         "sourced_from": "manifest.config.embargo_sessions"},
        {"key": "fold_sessions", "value": config.get("fold_sessions"),
         "sourced_from": "manifest.config.fold_sessions"},
        {"key": "holdout_sessions", "value": config.get("holdout_sessions"),
         "sourced_from": "manifest.config.holdout_sessions"},
        {"key": "enter_score", "value": config.get("enter_score"),
         "sourced_from": "manifest.config.enter_score"},
        {"key": "exit_score", "value": config.get("exit_score"),
         "sourced_from": "manifest.config.exit_score"},
        {"key": "data_digest", "value": manifest.get("data_digest"),
         "sourced_from": "manifest.data_digest"},
    ]


def framing_problems(manifest: dict) -> list[str]:
    """Return the list of framing problems; empty means the run is framable.

    A run whose assumed surface is incomplete cannot claim to be framed
    evidence — the assumption was left unstated, so the evidence it would
    have framed is unsupported. Fail-closed by construction.
    """
    if not isinstance(manifest, dict):
        return ["manifest is not a dict — cannot frame a run without its manifest"]
    problems: list[str] = []
    versions = manifest.get("versions") or {}
    if versions.get("cost_table") is None:
        problems.append("framing: cost_table version is missing — transaction costs are an explicit assumption")
    if versions.get("strategy") is None:
        problems.append("framing: strategy version is missing")
    if versions.get("framing") is None:
        problems.append("framing: framing version is missing from the manifest")
    provider = manifest.get("provider_overrides") or {}
    if not provider.get("price_basis"):
        problems.append("framing: price_basis is missing — the return basis is an explicit assumption")
    config = manifest.get("config") or {}
    for key in ("embargo_sessions", "fold_sessions", "holdout_sessions", "enter_score", "exit_score"):
        if config.get(key) is None:
            problems.append(f"framing: config.{key} is missing")
    if not config.get("universe"):
        problems.append("framing: universe is missing — the membership basis is an explicit assumption")
    if not manifest.get("data_digest"):
        problems.append("framing: data_digest is missing — the data identity is an explicit assumption")
    return problems


def build_framing_block(manifest: dict, aggregate: dict | None = None) -> dict:
    """Assemble the versioned framing block for one run (pure).

    `aggregate` only informs the evidence note (how many decisions the run
    actually exercised); it can never change the framing status — a run with
    superb or terrible results is framed identically: historical evidence
    under the listed assumptions.
    """
    decision_count = int((aggregate or {}).get("decision_count", 0) or 0)
    assumptions = assumption_entries(manifest)
    if decision_count > 0:
        evidence_status = "historical_evidence"
        evidence_note = (
            f"Decision count: {decision_count}. Any trades are evidence about "
            "that historical window only, under the listed assumptions."
        )
    else:
        evidence_status = "historical_evidence_no_decisions"
        evidence_note = (
            "Decision count: 0 — the run exercised no trade evidence. That is "
            "itself a historical fact under the listed assumptions, not a "
            "statement about the future."
        )
    return {
        "framing_version": FRAMING_VERSION,
        "statement": FRAMING_STATEMENT,
        "evidence_status": evidence_status,
        "evidence_note": evidence_note,
        "assumptions": assumptions,
        "limitations": [
            {"key": key, "detail": detail}
            for key, detail in FRAMING_LIMITATIONS.items()
        ],
        "does_not_claim": [
            {"key": key, "detail": detail}
            for key, detail in FRAMING_NON_CLAIMS.items()
        ],
    }


# --- Append-only framing store ------------------------------------------------------
# One JSONL record per run snapshot. Idempotent per (run type, run hash): a
# byte-identical recompute appends nothing; a same-key/different-content
# record is an integrity violation, never a silent revision. Malformed lines
# raise on load (loud, never skipped).


def _read_framing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Framing store {path} line {line_number} is not valid JSON "
                    f"(append-only integrity violated): {exc}"
                ) from exc
    return records


def persist_framing_snapshot(
    run_type: str,
    run_hash: str,
    snapshot: dict,
    path: str | Path | None = None,
) -> bool:
    """Append a framing snapshot; idempotent per (run_type, run_hash).

    Returns True when a record was appended, False when an identical record
    exists (deterministic recompute). The same-key record with different
    content raises — the run hash is the canonical digest of the run's
    inputs, so diverging framing is corruption, not a revision.
    """
    if not isinstance(run_type, str) or not run_type.strip():
        raise ValueError("run_type must be a non-empty string")
    if not isinstance(run_hash, str) or not run_hash.strip():
        raise ValueError("run_hash must be a non-empty string")
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("snapshot must be a non-empty dict")
    store_path = Path(path) if path is not None else FRAMING_STORE_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "framing_version": snapshot.get("framing_version", FRAMING_VERSION),
        "run_type": run_type,
        "run_hash": run_hash,
        "snapshot": dict(sorted(snapshot.items())),
    }
    for existing in _read_framing_records(store_path):
        if existing.get("run_type") == run_type and existing.get("run_hash") == run_hash:
            if existing.get("snapshot") == record["snapshot"]:
                return False  # identical recompute: nothing to append
            raise ValueError(
                f"framing integrity violation: {run_type} run {run_hash} "
                f"already exists with different content"
            )
    with store_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return True


def load_framing_snapshots(path: str | Path | None = None) -> list[dict]:
    """Read every persisted framing snapshot, newest last; strict on I/O."""
    store_path = Path(path) if path is not None else FRAMING_STORE_PATH
    return _read_framing_records(store_path)