"""D4 the provenance surface — every displayed value says where it came from.

"The UI must expose provenance, not hide it." Exposing it requires that it
exist per displayed value, and MEASURED, it does not.

**THE DECIDING MEASUREMENT: of the 5 PRESENT fields in a live forecast
snapshot, ZERO carry any identification of where the value came from.** The
snapshot as a whole is digest-addressable and its versions are pinned, but no
individual value names the source, payload or calculation that produced it. A
reader looking at "P(up) 0.56" has no way to reach the bytes behind it.

The substrate exists: MEASURED, the raw ledger holds **6,641 payloads across
2,758 distinct SHA-256 digests**, each carrying `source_id`, `request_key`,
`ingested_time` and `schema_version`. What was missing is the LINK from a
displayed value to one of them, and that link is what this module adds.

**UNTRACED is reported, never hidden.** This is the whole point of the sprint
bullet. A value with no provenance is shown *as* untraced rather than shown
with the question unasked. MEASURED, that is currently every PRESENT value in
the snapshot — so a surface that quietly omitted untraced values would report
perfect provenance over an empty set, which is the opposite of exposure.

**OBSERVED, DERIVED and COMPOSED are different claims.** A value read from a
source, one computed from other values, and one assembled from other traced
values are not interchangeable, and a reader must be able to tell which without
inspecting the pipeline. A COMPOSED value is traceable only as far as its
parts, and says so.

**A refusal carries provenance too.** Why a value is absent is itself evidence:
a reader who cannot see the reason cannot judge whether to wait for it. The
surface traces refusals alongside values rather than skipping them.

**The surface exposes; it decides nothing and recomputes nothing.**
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.config import (
    PROVENANCE_BLOCKS_TRADES,
    PROVENANCE_COVERAGE_WARN,
    PROVENANCE_ORIGINS,
    PROVENANCE_REPORT_UNTRACED,
    PROVENANCE_REQUIRED_FIELDS,
    PROVENANCE_SURFACE_VERSION,
    PROVENANCE_TRACE_REFUSALS,
    PROV_FIELD_INGESTED,
    PROV_FIELD_PAYLOAD,
    PROV_FIELD_SOURCE,
    PROV_FIELD_VERSION,
    PROV_ORIGIN_COMPOSED,
    PROV_ORIGIN_OBSERVED,
    PROV_ORIGIN_UNTRACED,
    VIEW_CELL_PRESENT,
)


class ProvenanceError(ValueError):
    """Raised when a provenance request is structurally invalid."""


def trace(
    origin: str,
    *,
    source_id: str | None = None,
    payload_sha256: str | None = None,
    ingested_time: str | None = None,
    calculation_version: str | None = None,
    inputs: Iterable[str] = (),
    reason: str = "",
) -> dict:
    """One provenance record.

    An UNTRACED record carries no identifiers and must carry a reason: saying
    "we do not know where this came from" is information, and saying nothing
    is not.
    """
    if origin not in PROVENANCE_ORIGINS:
        raise ProvenanceError(f"unknown origin {origin!r}")

    record: dict[str, Any] = {"origin": origin, "reason": reason}

    if origin == PROV_ORIGIN_UNTRACED:
        if not str(reason or "").strip():
            raise ProvenanceError(
                "an UNTRACED value must say why it cannot be traced; silence "
                "is what this surface exists to remove"
            )
        for field in PROVENANCE_REQUIRED_FIELDS:
            record[field] = None
        record["inputs"] = []
        return record

    if origin == PROV_ORIGIN_COMPOSED:
        parts = sorted(str(i) for i in inputs)
        if not parts:
            raise ProvenanceError(
                "a COMPOSED value must name the values it was assembled from, "
                "or it is not composed of anything"
            )
        record.update(
            {
                PROV_FIELD_SOURCE: source_id,
                PROV_FIELD_PAYLOAD: payload_sha256,
                PROV_FIELD_INGESTED: ingested_time,
                PROV_FIELD_VERSION: calculation_version,
                "inputs": parts,
            }
        )
        return record

    # OBSERVED and DERIVED must both be re-derivable.
    record.update(
        {
            PROV_FIELD_SOURCE: source_id,
            PROV_FIELD_PAYLOAD: payload_sha256,
            PROV_FIELD_INGESTED: ingested_time,
            PROV_FIELD_VERSION: calculation_version,
            "inputs": sorted(str(i) for i in inputs),
        }
    )
    missing = [f for f in PROVENANCE_REQUIRED_FIELDS if not record.get(f)]
    if missing:
        raise ProvenanceError(
            f"a {origin} value is missing {missing}; without them the value "
            f"cannot be traced back to what produced it"
        )
    return record


def untraced(reason: str) -> dict:
    """Shorthand for the case that currently dominates."""
    return trace(PROV_ORIGIN_UNTRACED, reason=reason)


def from_raw_record(record: Mapping[str, Any], *, origin: str = PROV_ORIGIN_OBSERVED) -> dict:
    """Provenance for a value read straight from a raw ledger payload."""
    if not isinstance(record, Mapping):
        raise ProvenanceError("a raw record mapping is required")
    return trace(
        origin,
        source_id=record.get("source_id"),
        payload_sha256=record.get("payload_sha256"),
        ingested_time=record.get("ingested_time"),
        calculation_version=record.get("schema_version"),
    )


def _entry(path: str, cell: Mapping[str, Any], record: Mapping[str, Any] | None) -> dict:
    """One displayed value paired with its provenance."""
    state = cell.get("state") or cell.get("status")
    displayed = cell.get("value")

    if record is None:
        record = untraced(
            f"no provenance was recorded for {path}; the value is displayed "
            f"but cannot be traced to a source"
        )
    if not isinstance(record, Mapping):
        raise ProvenanceError(f"{path}: provenance is not a mapping")
    if record.get("origin") not in PROVENANCE_ORIGINS:
        raise ProvenanceError(f"{path}: unknown origin {record.get('origin')!r}")

    return {
        "path": path,
        "state": state,
        "displayed": displayed,
        "traced": record["origin"] != PROV_ORIGIN_UNTRACED,
        "provenance": dict(record),
    }


def build_provenance_surface(
    view: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Pair every displayed value in a view with its provenance.

    `records` maps a dotted path ("panels.multi_horizon_forecast.rows.20d.
    probability_up") to a provenance record. A path with no record is reported
    UNTRACED rather than omitted.
    """
    if not isinstance(view, Mapping):
        raise ProvenanceError("a view mapping is required")
    records = records or {}
    if not isinstance(records, Mapping):
        raise ProvenanceError("records must be a mapping of path to provenance")

    entries = [
        _entry(path, cell, records.get(path)) for path, cell in _walk(view)
    ]

    if not PROVENANCE_REPORT_UNTRACED:  # pragma: no cover - config forbids it
        entries = [e for e in entries if e["traced"]]

    displayed = [e for e in entries if e["state"] == VIEW_CELL_PRESENT]
    traced = [e for e in displayed if e["traced"]]
    share = round(len(traced) / len(displayed), 6) if displayed else None

    by_origin: dict[str, int] = {origin: 0 for origin in PROVENANCE_ORIGINS}
    for entry in entries:
        by_origin[entry["provenance"]["origin"]] += 1

    coverage = {
        "entries": len(entries),
        "displayed": len(displayed),
        "traced": len(traced),
        "untraced": len(displayed) - len(traced),
        "share_traced": share,
        "below_threshold": bool(share is not None and share < PROVENANCE_COVERAGE_WARN),
        "threshold": PROVENANCE_COVERAGE_WARN,
        "by_origin": by_origin,
    }

    return {
        "version": PROVENANCE_SURFACE_VERSION,
        "ticker": view.get("ticker"),
        "as_of": view.get("as_of"),
        "entries": entries,
        "coverage": coverage,
        "headline": _headline(coverage),
        "traces_refusals": PROVENANCE_TRACE_REFUSALS,
        "blocks_trades": PROVENANCE_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _walk(view: Mapping[str, Any]):
    """Every cell in a view, with the dotted path that locates it."""
    panels = view.get("panels") or {}
    for panel_name, panel in panels.items():
        if not isinstance(panel, Mapping):
            continue
        for key, value in panel.items():
            if isinstance(value, Mapping) and _is_cell(value):
                yield f"panels.{panel_name}.{key}", value
            elif key == "rows" and isinstance(value, list):
                for row in value:
                    if not isinstance(row, Mapping):
                        continue
                    horizon = row.get("horizon", "?")
                    for cell_key, cell in row.items():
                        if isinstance(cell, Mapping) and _is_cell(cell):
                            yield (
                                f"panels.{panel_name}.rows.{horizon}.{cell_key}",
                                cell,
                            )
            elif isinstance(value, Mapping):
                for sub_key, sub in value.items():
                    if isinstance(sub, Mapping) and _is_cell(sub):
                        yield f"panels.{panel_name}.{key}.{sub_key}", sub


def _is_cell(candidate: Mapping[str, Any]) -> bool:
    return "state" in candidate or "status" in candidate


def _headline(coverage: Mapping[str, Any]) -> str:
    """What a reader must know before trusting any value on screen."""
    if not coverage["displayed"]:
        return (
            "no values are displayed, so there is no provenance to expose; "
            "the gaps above are the finding"
        )
    if coverage["below_threshold"]:
        return (
            f"{coverage['traced']} of {coverage['displayed']} displayed "
            f"values can be traced to a source "
            f"({coverage['share_traced']:.0%}); {coverage['untraced']} cannot. "
            f"An untraced number is a claim without a citation"
        )
    return (
        f"{coverage['traced']} of {coverage['displayed']} displayed values "
        f"trace to a source ({coverage['share_traced']:.0%})"
    )


_NOTE = (
    "MEASURED, 0 of 5 PRESENT fields in a live forecast snapshot carry any "
    "identification of where the value came from, while the raw ledger holds "
    "6,641 payloads across 2,758 distinct SHA-256 digests. The substrate "
    "exists; this surface is the link, and it reports UNTRACED rather than "
    "omitting what it cannot trace."
)


def provenance_problems(surface: Mapping[str, Any]) -> list[str]:
    """Contract check on a provenance surface. Empty means clean."""
    problems: list[str] = []
    if not isinstance(surface, Mapping):
        return ["surface is not a mapping"]

    if surface.get("blocks_trades"):
        problems.append("the provenance surface claims to block trades")
    if not surface.get("traces_refusals"):
        problems.append(
            "the surface does not trace refusals — why a value is absent is "
            "itself evidence a reader needs"
        )

    entries = surface.get("entries")
    if entries is None:
        return problems + ["the surface carries no entries"]
    if not isinstance(entries, list):
        return problems + ["entries is not a list"]

    seen_paths = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            problems.append("an entry is not a mapping")
            continue
        path = entry.get("path")
        if not str(path or "").strip():
            problems.append("an entry has no path — it cannot be located")
            continue
        if path in seen_paths:
            problems.append(f"{path}: duplicated entry")
        seen_paths.add(path)

        record = entry.get("provenance")
        if not isinstance(record, Mapping):
            problems.append(f"{path}: provenance is not a mapping")
            continue
        origin = record.get("origin")
        if origin not in PROVENANCE_ORIGINS:
            problems.append(f"{path}: unknown origin {origin!r}")
            continue

        if origin == PROV_ORIGIN_UNTRACED:
            if entry.get("traced"):
                problems.append(
                    f"{path}: marked traced while its origin is UNTRACED"
                )
            if not str(record.get("reason") or "").strip():
                problems.append(
                    f"{path}: UNTRACED with no reason — silence is what this "
                    f"surface exists to remove"
                )
            for field in PROVENANCE_REQUIRED_FIELDS:
                if record.get(field) is not None:
                    problems.append(
                        f"{path}: UNTRACED carried {field}="
                        f"{record.get(field)!r}"
                    )
        else:
            if not entry.get("traced"):
                problems.append(
                    f"{path}: origin {origin} but the entry is marked untraced"
                )
            if origin == PROV_ORIGIN_COMPOSED:
                if not record.get("inputs"):
                    problems.append(
                        f"{path}: COMPOSED names no inputs, so it is not "
                        f"composed of anything"
                    )
            else:
                missing = [
                    f for f in PROVENANCE_REQUIRED_FIELDS if not record.get(f)
                ]
                if missing:
                    problems.append(
                        f"{path}: {origin} is missing {missing}; the value "
                        f"cannot be traced back to what produced it"
                    )

    coverage = surface.get("coverage") or {}
    if coverage:
        displayed = int(coverage.get("displayed", 0))
        traced = int(coverage.get("traced", 0))
        untraced_count = int(coverage.get("untraced", 0))
        if traced + untraced_count != displayed:
            problems.append(
                f"coverage counts {traced} traced + {untraced_count} untraced "
                f"against {displayed} displayed"
            )
        if traced > displayed:
            problems.append("more values are traced than are displayed")
        if coverage.get("below_threshold") and not str(
            surface.get("headline") or ""
        ).strip():
            problems.append(
                "provenance is below the threshold and the surface has no "
                "headline — untraced numbers would read as cited ones"
            )
    return problems


def render_provenance(surface: Mapping[str, Any], limit: int = 20) -> list[str]:
    """Human-readable lines: the headline, then one line per traced value."""
    lines = [f"  {surface.get('headline')}"]
    for entry in (surface.get("entries") or [])[:limit]:
        if not isinstance(entry, Mapping):
            continue
        record = entry.get("provenance") or {}
        origin = str(record.get("origin"))
        digest = record.get(PROV_FIELD_PAYLOAD)
        shown = f"{str(digest)[:12]}…" if digest else "—"
        lines.append(
            f"      {str(entry.get('path'))[:52]:52s} {origin:9s} "
            f"{str(record.get(PROV_FIELD_SOURCE) or '—')[:22]:22s} {shown}"
        )
    remaining = len(surface.get("entries") or []) - limit
    if remaining > 0:
        lines.append(f"      … {remaining} more")
    return lines
