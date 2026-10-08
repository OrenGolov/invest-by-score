"""D2 the WHY breakdown — what evidence entered, not what caused.

The roadmap asks for "Technical / Fundamental / News / Macro / Regime /
Sentiment contributions". Two measured facts stand between that wording and an
honest panel, and this module is built around both.

**FIRST: THEY ARE NOT CONTRIBUTIONS.** F6 MEASURED, over 4,000 observations
with a realistic regime/chart correlation:

    regime alone   +0.064
    chart alone    +0.067
    sum            +0.131
    ACTUAL joint   +0.060

The parts overlap and double-count by more than 2x. A panel showing six
numbers that sum to the forecast would be arithmetically wrong, and would
imply each factor independently *caused* its share. So the panel reports, per
component, **whether it supplied evidence and how much it narrowed the claim**
— F6's vocabulary — and the word "contribution" is barred from its fields.

**SECOND: HALF OF WHAT THE ROADMAP NAMES CANNOT BE MEASURED.** Of F6's seven
components only four are wired, and the three that are NOT_WIRED —
`fundamental`, `macro`, `sentiment` — are exactly three of the six the roadmap
asks for. MEASURED on a live decomposition: **1 PRESENT, 3 ABSENT, 3
NOT_WIRED**. Rendering those three as 0.0 would say they were measured and
found irrelevant. They were never measured, which is a different fact and the
one a "why?" panel must not blur.

**ABSENT and NOT_WIRED are different answers.** ABSENT means this component
could have spoken for this forecast and did not; NOT_WIRED means it cannot
speak at all yet. Collapsing them would tell a reader that the fundamentals
were checked and had nothing to say.

**A seventh component is shown although the roadmap does not name it.**
MEASURED, `historical_analog` is the component that actually produced the
forecast value. A panel answering "why?" without it would omit the only wired
evidence that spoke.

**The panel explains; it decides nothing, and recomputes nothing.** Every
value is F6's, so the panel cannot disagree with the decomposition it shows.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.config import (
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_OVERLAP_EVIDENCE,
    DECOMPOSITION_WIRED_COMPONENTS,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    WHY_BARRED_TERMS,
    WHY_MIN_PRESENT,
    WHY_PANEL_BLOCKS_TRADES,
    WHY_PANEL_VERSION,
    WHY_REPORTS_CONTRIBUTIONS,
    WHY_ROADMAP_NAMES,
    WHY_SHOW_ALL_COMPONENTS,
    WHY_STATUSES,
)


class WhyPanelError(ValueError):
    """Raised when a WHY panel request is structurally invalid."""


def display_name(component: str) -> str:
    """The roadmap's name for an F6 component."""
    name = WHY_ROADMAP_NAMES.get(component)
    if not name:
        raise WhyPanelError(f"unknown component {component!r}")
    return name


def _row(component: str, detail: Mapping[str, Any] | None) -> dict:
    """One component's row. Never a contribution, always a status.

    A row carries an `effect` only when the component is PRESENT, for the
    reason the rest of this codebase carries values only when measured: an
    effect on an unwired component would read as a measurement nobody made.
    """
    label = display_name(component)
    wired = component in DECOMPOSITION_WIRED_COMPONENTS

    if detail is None:
        status = DECOMP_STATUS_NOT_WIRED if not wired else DECOMP_STATUS_ABSENT
        return {
            "component": component,
            "label": label,
            "status": status,
            "wired": wired,
            "effect": None,
            "reason": (
                f"{label} is not wired into the forecast yet"
                if not wired
                else f"no {label.lower()} evidence was supplied to the panel"
            ),
        }
    if not isinstance(detail, Mapping):
        raise WhyPanelError(f"{component}: component detail is not a mapping")

    status = detail.get("status")
    if status not in WHY_STATUSES:
        raise WhyPanelError(f"{component}: unknown status {status!r}")

    effect = detail.get("effect") if status == DECOMP_STATUS_PRESENT else None
    reason = str(detail.get("reason") or "").strip()
    if not reason:
        raise WhyPanelError(f"{component}: no reason given for {status}")

    row = {
        "component": component,
        "label": label,
        "status": status,
        "wired": wired,
        "effect": effect,
        "reason": reason,
    }
    # Carry F6's narrowing detail through untouched where it exists. This is
    # the honest replacement for a contribution number: how much the claim
    # narrowed, not how much this factor "caused".
    for field in ("base_rate", "narrowed_rate", "delta", "trials", "narrowed_trials"):
        if field in detail:
            row[field] = detail[field]
    return row


def build_why_panel(decomposition: Mapping[str, Any] | None) -> dict:
    """The WHY breakdown, composed from an F6 decomposition.

    `None` is a real input: it means no decomposition was supplied, and every
    component is reported as unavailable rather than as silent.
    """
    if decomposition is not None and not isinstance(decomposition, Mapping):
        raise WhyPanelError("the decomposition must be a mapping")

    components = (decomposition or {}).get("components") or {}
    if components and not isinstance(components, Mapping):
        raise WhyPanelError("the decomposition components must be a mapping")

    order = list(DECOMPOSITION_COMPONENTS) if WHY_SHOW_ALL_COMPONENTS else [
        c for c in DECOMPOSITION_COMPONENTS if c in WHY_ROADMAP_NAMES
    ]
    rows = [_row(component, components.get(component)) for component in order]

    present = [r["component"] for r in rows if r["status"] == DECOMP_STATUS_PRESENT]
    absent = [r["component"] for r in rows if r["status"] == DECOMP_STATUS_ABSENT]
    not_wired = [
        r["component"] for r in rows if r["status"] == DECOMP_STATUS_NOT_WIRED
    ]

    return {
        "version": WHY_PANEL_VERSION,
        "decomposed_object": (decomposition or {}).get("decomposed_object"),
        "ticker": (decomposition or {}).get("ticker"),
        "as_of": (decomposition or {}).get("as_of"),
        "horizon": (decomposition or {}).get("horizon"),
        "component_order": order,
        "rows": rows,
        "present": present,
        "absent": absent,
        "not_wired": not_wired,
        "additive": DECOMPOSITION_ADDITIVE,
        "reports_contributions": WHY_REPORTS_CONTRIBUTIONS,
        "overlap_evidence": DECOMPOSITION_OVERLAP_EVIDENCE,
        "headline": _headline(present, absent, not_wired),
        "blocks_trades": WHY_PANEL_BLOCKS_TRADES,
        "note": _NOTE,
    }


def _headline(present, absent, not_wired) -> str:
    """What a reader must know before reading the rows."""
    total = len(present) + len(absent) + len(not_wired)
    if len(present) < WHY_MIN_PRESENT:
        return (
            f"{len(present)} of {total} components supplied evidence; "
            f"{len(absent)} had none for this forecast and {len(not_wired)} "
            f"are not wired into the system at all. This is not a weighting "
            f"of six factors — it is one or two that spoke"
        )
    return (
        f"{len(present)} of {total} components supplied evidence; "
        f"{len(not_wired)} are not wired yet. The rows below are not "
        f"additive: MEASURED, the parts overlap and double-count by over 2x"
    )


_NOTE = (
    "These are not contributions. MEASURED over 4,000 observations: regime "
    "alone +0.064, chart alone +0.067, sum +0.131, ACTUAL joint effect "
    "+0.060 - the parts overlap and double-count by more than 2x. Each row "
    "says whether that component supplied evidence and how much it NARROWED "
    "the claim, never how much it caused."
)


def why_problems(panel: Mapping[str, Any]) -> list[str]:
    """Contract check on a WHY panel. Empty means clean."""
    problems: list[str] = []
    if not isinstance(panel, Mapping):
        return ["panel is not a mapping"]

    if panel.get("blocks_trades"):
        problems.append("the WHY panel claims to block trades — it explains")
    if panel.get("additive"):
        problems.append(
            "the panel claims its components are additive. MEASURED, regime "
            "+0.064 and chart +0.067 sum to +0.131 against an ACTUAL joint "
            "effect of +0.060"
        )
    if panel.get("reports_contributions"):
        problems.append(
            "the panel claims to report contributions — a number per factor "
            "that sums to the forecast asserts additivity and causation that "
            "were both measured to be false"
        )
    if not str(panel.get("overlap_evidence") or "").strip():
        problems.append(
            "the panel carries no overlap evidence — the reason it refuses to "
            "add its own rows must travel with it"
        )

    rows = panel.get("rows") or []
    seen = []
    for row in rows:
        if not isinstance(row, Mapping):
            problems.append("a row is not a mapping")
            continue
        component = row.get("component")
        seen.append(component)
        if component not in WHY_ROADMAP_NAMES:
            problems.append(f"unknown component {component!r}")
            continue
        if row.get("label") != WHY_ROADMAP_NAMES[component]:
            problems.append(
                f"{component}: label {row.get('label')!r} does not match the "
                f"declared name {WHY_ROADMAP_NAMES[component]!r}"
            )
        status = row.get("status")
        if status not in WHY_STATUSES:
            problems.append(f"{component}: unknown status {status!r}")
            continue
        if not str(row.get("reason") or "").strip():
            problems.append(f"{component}: {status} with no reason")

        # NOT_WIRED MUST NOT BECOME ABSENT. They are different facts.
        wired = component in DECOMPOSITION_WIRED_COMPONENTS
        if status == DECOMP_STATUS_NOT_WIRED and wired:
            problems.append(
                f"{component}: reported NOT_WIRED although it is wired"
            )
        if not wired and status != DECOMP_STATUS_NOT_WIRED:
            problems.append(
                f"{component}: an unwired component reported {status} — that "
                f"says it was measured and had nothing to say, which is a "
                f"different and false claim"
            )

        # THE SHAPE RULE: an effect exists IFF the component supplied evidence.
        if status != DECOMP_STATUS_PRESENT and row.get("effect") is not None:
            problems.append(
                f"{component}: {status} carried the effect "
                f"{row.get('effect')!r}"
            )

        # NO CONTRIBUTION NUMBERS, anywhere in the row.
        for key in row:
            lowered = str(key).lower()
            for barred in WHY_BARRED_TERMS:
                if barred in lowered:
                    problems.append(
                        f"{component}: field {key!r} names a contribution; "
                        f"the parts overlap and cannot be added"
                    )

    for component in DECOMPOSITION_COMPONENTS:
        if WHY_SHOW_ALL_COMPONENTS and component not in seen:
            problems.append(
                f"component {component!r} is missing — a 'why?' panel that "
                f"drops a component answers the question incompletely without "
                f"saying so"
            )

    if len(panel.get("present") or []) < WHY_MIN_PRESENT:
        if not str(panel.get("headline") or "").strip():
            problems.append(
                "almost nothing supplied evidence and the panel has no "
                "headline — six rows of ABSENT read as a considered "
                "weighting rather than an empty one"
            )
    return problems


def render_why(panel: Mapping[str, Any]) -> list[str]:
    """Human-readable lines: the headline, then one line per component."""
    lines = [f"  {panel.get('headline')}"]
    for row in panel.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        effect = row.get("effect")
        shown = str(effect) if effect is not None else "—"
        lines.append(
            f"      {str(row.get('label')):20s} {str(row.get('status')):10s} "
            f"{shown:12s} {str(row.get('reason'))[:60]}"
        )
    return lines
