"""Gate: an untraced value is shown AS untraced, never with the question unasked.

"The UI must expose provenance, not hide it." Exposing it requires that it
exist per displayed value, and MEASURED, it does not.

THE DECIDING MEASUREMENT, recomputed here against the real snapshot assembly:
of the PRESENT fields in a live forecast snapshot, ZERO carry any
identification of where the value came from. The snapshot as a whole is
digest-addressable and its versions are pinned, but no individual value names
the source, payload or calculation that produced it.

The substrate exists - the raw ledger records source_id, request_key,
payload_sha256, ingested_time and schema_version per payload - so what was
missing is the LINK from a displayed value to one of them.

UNTRACED IS REPORTED, NEVER OMITTED. A surface that quietly dropped what it
could not trace would report perfect provenance over an empty set, which is
the opposite of exposure.

Verified to FAIL when any of these is reinjected:
  - untraced values omitted from the surface
  - an UNTRACED record carrying a payload digest
  - an UNTRACED record with no reason
  - an OBSERVED or DERIVED value missing its identifiers
  - a COMPOSED value naming no inputs
  - refusals skipped instead of traced
  - a zero-coverage surface rendering without its headline
  - the surface claiming to block trades
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (  # noqa: E402
    PROVENANCE_BLOCKS_TRADES,
    PROVENANCE_COVERAGE_WARN,
    PROVENANCE_REPORT_UNTRACED,
    PROVENANCE_REQUIRED_FIELDS,
    PROVENANCE_TRACE_REFUSALS,
    PROV_FIELD_PAYLOAD,
    PROV_FIELD_SOURCE,
    PROV_ORIGIN_COMPOSED,
    PROV_ORIGIN_DERIVED,
    PROV_ORIGIN_OBSERVED,
    PROV_ORIGIN_UNTRACED,
    RESEARCH_VIEW_HORIZONS,
    VIEW_CELL_PRESENT,
)
from core.forecast_snapshot import build_forecast_snapshot  # noqa: E402
from core.provenance_surface import (  # noqa: E402
    ProvenanceError,
    build_provenance_surface,
    from_raw_record,
    provenance_problems,
    render_provenance,
    trace,
    untraced,
)
from core.research_view import build_research_view  # noqa: E402

FAILURES: list[str] = []

RAW = {
    "source_id": "yahoo_finance_chart",
    "request_key": "NVDA_5y_1d",
    "payload_sha256": "4b31729a9bc1703f2a730b880f75f5712023f13a68629c6d062b686804add6b3",
    "ingested_time": "2026-09-22T06:22:00.728733+00:00",
    "schema_version": "raw-store-v1",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def view():
    snapshots = {
        h: build_forecast_snapshot("NVDA", "2026-09-22", h)
        for h in RESEARCH_VIEW_HORIZONS
    }
    return build_research_view(
        "NVDA",
        "2026-09-22",
        snapshots,
        quality={"score": 78.5, "action": "BUY"},
        risk={"veto": False, "veto_rule_ids": []},
    )


def main() -> int:
    # 1. THE DECIDING MEASUREMENT, RECOMPUTED. No PRESENT field identifies
    #    where its value came from.
    snapshot = build_forecast_snapshot("NVDA", "2026-09-22", "20d")
    carrying = []
    for name, field in snapshot["fields"].items():
        if field.get("status") != VIEW_CELL_PRESENT:
            continue
        identifying = set(field) - {"status", "value", "reason"}
        if identifying:
            carrying.append((name, sorted(identifying)))
    check(
        not carrying,
        f"snapshot fields now carry identifying provenance ({carrying}); D4 "
        f"exists because they carried nothing. If this has changed the "
        f"surface must be re-derived rather than kept",
    )

    # 2. AN UNTRACED VIEW REPORTS ZERO COVERAGE AND SAYS SO.
    bare = build_provenance_surface(view())
    check(
        provenance_problems(bare) == [],
        f"a well-formed untraced surface reported problems: "
        f"{provenance_problems(bare)}",
    )
    check(
        bare["coverage"]["traced"] == 0,
        f"{bare['coverage']['traced']} values traced with no records supplied",
    )
    check(
        bare["coverage"]["below_threshold"],
        f"zero provenance was not flagged below the "
        f"{PROVENANCE_COVERAGE_WARN:.0%} threshold",
    )
    check(
        "without a citation" in bare["headline"],
        "a surface with no traceable values did not say so; untraced numbers "
        "would read as cited ones",
    )

    # 3. UNTRACED IS REPORTED, NOT OMITTED.
    check(PROVENANCE_REPORT_UNTRACED, "untraced values are configured to be hidden")
    check(
        len(bare["entries"]) > 0,
        "the surface omitted everything it could not trace, reporting perfect "
        "provenance over an empty set",
    )
    for entry in bare["entries"]:
        check(
            entry["provenance"]["origin"] == PROV_ORIGIN_UNTRACED,
            f"{entry['path']}: origin {entry['provenance']['origin']} with no "
            f"record supplied",
        )
        check(
            not entry["traced"],
            f"{entry['path']}: marked traced with no record supplied",
        )
        check(
            str(entry["provenance"].get("reason") or "").strip() != "",
            f"{entry['path']}: UNTRACED with no reason — silence is what this "
            f"surface exists to remove",
        )
        for field in PROVENANCE_REQUIRED_FIELDS:
            check(
                entry["provenance"].get(field) is None,
                f"{entry['path']}: UNTRACED carried {field}="
                f"{entry['provenance'].get(field)!r}",
            )

    # 4. WHEN PROVENANCE EXISTS IT REACHES THE SURFACE INTACT, and the
    #    coverage warning can stay quiet.
    traced = build_provenance_surface(
        view(),
        {
            "panels.investment_quality.score": from_raw_record(
                RAW, origin=PROV_ORIGIN_DERIVED
            ),
            "panels.investment_quality.action": trace(
                PROV_ORIGIN_COMPOSED,
                inputs=["panels.investment_quality.score"],
                calculation_version="score-v1",
            ),
            "panels.risk_status.veto": from_raw_record(RAW),
        },
    )
    check(
        provenance_problems(traced) == [],
        f"a well-formed traced surface reported problems: "
        f"{provenance_problems(traced)}",
    )
    check(
        traced["coverage"]["share_traced"] == 1.0,
        f"fully-traced values reported {traced['coverage']['share_traced']} "
        f"coverage",
    )
    check(
        not traced["coverage"]["below_threshold"],
        "a fully-traced surface still warned — the threshold fires on "
        "everything and therefore means nothing",
    )
    by_origin = traced["coverage"]["by_origin"]
    for origin in (PROV_ORIGIN_OBSERVED, PROV_ORIGIN_DERIVED, PROV_ORIGIN_COMPOSED):
        check(
            by_origin[origin] >= 1,
            f"{origin} did not survive to the surface; 'read from a source', "
            f"'computed' and 'assembled' are different claims",
        )

    # 5. A TRACEABLE VALUE MUST BE RE-DERIVABLE.
    for missing in PROVENANCE_REQUIRED_FIELDS:
        kwargs = {
            "source_id": RAW["source_id"],
            "payload_sha256": RAW["payload_sha256"],
            "ingested_time": RAW["ingested_time"],
            "calculation_version": RAW["schema_version"],
        }
        kwargs[missing] = None
        try:
            trace(PROV_ORIGIN_OBSERVED, **kwargs)
            FAILURES.append(
                f"an OBSERVED value was accepted without {missing}; the value "
                f"cannot be traced back to what produced it"
            )
        except ProvenanceError:
            pass
    try:
        trace(PROV_ORIGIN_COMPOSED, inputs=[])
        FAILURES.append("a COMPOSED value naming no inputs was accepted")
    except ProvenanceError:
        pass
    try:
        trace(PROV_ORIGIN_UNTRACED, reason="  ")
        FAILURES.append("an UNTRACED value with no reason was accepted")
    except ProvenanceError:
        pass
    try:
        trace("PROBABLY_FINE", reason="x")
        FAILURES.append("an unknown origin was accepted")
    except ProvenanceError:
        pass

    # 6. REFUSALS ARE TRACED. Why a value is absent is itself evidence.
    check(PROVENANCE_TRACE_REFUSALS, "refusals are configured not to be traced")
    check(bare["traces_refusals"], "the surface does not trace refusals")
    refused = [e for e in bare["entries"] if e["state"] != VIEW_CELL_PRESENT]
    check(
        bool(refused),
        "no refused cells reached the surface; a reader cannot see why a "
        "value is missing",
    )
    check(
        bare["coverage"]["displayed"] < bare["coverage"]["entries"],
        "every entry was counted as a displayed value; refusals are traced "
        "but are not displayed values",
    )

    # 7. THE SURFACE EXPOSES; IT DECIDES NOTHING.
    check(
        not PROVENANCE_BLOCKS_TRADES and not bare["blocks_trades"],
        "the provenance surface claims to block trades",
    )

    # 8. THE RENDER CARRIES THE STATE AND THE DIGEST.
    lines = render_provenance(bare)
    check(lines[0].strip() == bare["headline"], "the headline is not rendered first")
    check(
        PROV_ORIGIN_UNTRACED in "\n".join(lines),
        "UNTRACED does not reach the render",
    )
    traced_lines = "\n".join(render_provenance(traced))
    check(
        RAW["payload_sha256"][:12] in traced_lines,
        "the payload digest does not reach the render; a reader cannot get "
        "from the screen to the bytes",
    )
    for line in lines:
        check(len(line) < 200, f"a rendered line is {len(line)} characters long")

    # 9. MALFORMED INPUT IS REFUSED.
    for bad_args, label in (
        ((["a view"], None), "a non-mapping view"),
        ((view(), ["a record"]), "non-mapping records"),
        (
            (view(), {"panels.investment_quality.score": "yahoo"}),
            "a non-mapping record entry",
        ),
    ):
        try:
            build_provenance_surface(*[a for a in bad_args if a is not None])
            FAILURES.append(f"{label} was accepted")
        except ProvenanceError:
            pass

    # 10. THE CONTRACT CHECK CAN FAIL.
    mutations = [
        (lambda s: s["entries"][0].update({"traced": True}), "an untraced entry marked traced"),
        (
            lambda s: s["entries"][0]["provenance"].update(
                {PROV_FIELD_PAYLOAD: "deadbeef"}
            ),
            "an UNTRACED record carrying a digest",
        ),
        (
            lambda s: s["entries"][0]["provenance"].update({"reason": "  "}),
            "an UNTRACED record with no reason",
        ),
        (lambda s: s.update({"traces_refusals": False}), "a surface skipping refusals"),
        (lambda s: s["coverage"].update({"traced": 99}), "a miscounted coverage"),
        (lambda s: s.update({"headline": ""}), "a silenced headline"),
        (lambda s: s.update({"blocks_trades": True}), "a surface that blocks trades"),
        (lambda s: s["entries"][0].update({"path": ""}), "a pathless entry"),
        (
            lambda s: s["entries"].append(dict(s["entries"][0])),
            "a duplicated path",
        ),
    ]
    for mutation, label in mutations:
        broken = build_provenance_surface(view())
        mutation(broken)
        check(provenance_problems(broken) != [], f"{label} passed the contract check")

    if FAILURES:
        print("PROVENANCE SURFACE GATE: FAIL")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("PROVENANCE SURFACE GATE: PASS")
    print("  0 PRESENT snapshot fields carry per-value provenance (recomputed)")
    print(f"  the untraced view reports {bare['coverage']['traced']}/"
          f"{bare['coverage']['displayed']} traced and leads with the gap")
    print("  UNTRACED is reported, never omitted, and always carries a reason")
    print("  OBSERVED / DERIVED / COMPOSED stay distinct; digests reach the screen")
    print("  refusals are traced too: why a value is absent is evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
