"""Forecast decomposition (Sprint F6) — what evidence entered, not what caused.

`decompose_forecast(...)` breaks a forecast into the seven components the
sprint names — technical, fundamental, news-event, macro, regime, sentiment,
historical-analog — and reports, for each, **whether it supplied evidence and
how much it narrowed the claim**.

**Why not "contributions".** The obvious implementation assigns each component
a number that sums to the forecast. It is arithmetically wrong here, and it
was measured rather than argued. F5's value is a base rate over analogs
retrieved by chart similarity, filtered by event type, within a regime — and
those filters are not independent, because one historical day can satisfy all
three. Over 4,000 observations with a realistic regime/chart correlation:

    all sessions          P(up) 0.546
    bullish regime        P(up) 0.611    "contribution" +0.064
    uptrend chart         P(up) 0.614    "contribution" +0.067
    bullish AND uptrend   P(up) 0.606    SUM +0.131, ACTUAL +0.060

The parts overlap and double-count by more than 2x. Presenting them as
additive contributions would both misstate the arithmetic and imply each
factor independently *caused* its share. `DECOMPOSITION_ADDITIVE` is False and
the import-time validator refuses to let it become True.

**There are no coefficients to attribute.** MEASURED: `build_joint_forecast`
returns NO_MODEL — "no trained forecasting model is registered". So this is
not feature attribution, not SHAP, not a weight table. What produces a
forecast value today is F4 (a regime-conditioned base rate) and F5 (an analog
base rate), both empirical slices of history. A decomposition of a slice is a
statement about **which filters produced it**.

**NOT_WIRED is not zero.** Four of the seven components can supply forecast
evidence today; fundamental, macro and sentiment cannot. Reporting those as
`0.0` would say they were measured and found irrelevant. They were never
measured, which is a different fact and the one a decomposition must not blur.

**Not a second ensemble breakdown (W5).** W1 already decomposes the SCORE
across these same seven agent names, and `score_engine` renders per-term
values for the dashboard. F6 decomposes the FORECAST — a different number from
different machinery — and every output says which object it describes.

**Claim strength is F4's.** A component's marginal slice is just a smaller
sample, so `select_claim` decides what it can support rather than a second
sample-size policy that would drift from the first.
"""

from __future__ import annotations

import logging

from core.config import (
    DECOMP_EFFECT_NARROWED,
    DECOMP_EFFECT_NO_EFFECT,
    DECOMP_EFFECT_UNMEASURED,
    DECOMP_HISTORICAL_ANALOG,
    DECOMP_NEWS_EVENT,
    DECOMP_REGIME,
    DECOMP_STATUS_ABSENT,
    DECOMP_STATUS_NOT_WIRED,
    DECOMP_STATUS_PRESENT,
    DECOMP_TECHNICAL,
    DECOMPOSITION_ADDITIVE,
    DECOMPOSITION_COMPONENTS,
    DECOMPOSITION_CONTRACT_VERSION,
    DECOMPOSITION_DISCLAIMER,
    DECOMPOSITION_OVERLAP_EVIDENCE,
    DECOMPOSITION_WIRED_COMPONENTS,
    FORECAST_DECOMPOSITION_VERSION,
)
from core.forecast_conditional import (
    CONDITIONAL_CLAIM_INSUFFICIENT,
    interval_width,
    select_claim,
    wilson_interval,
)

LOGGER = logging.getLogger("core.forecast_decomposition")

# The object being decomposed. Stated in every output so a reader can never
# mistake this for W1's ensemble breakdown of the SCORE.
DECOMPOSED_OBJECT = "forecast"


class DecompositionError(ValueError):
    """Raised when a decomposition request violates the F6 contract."""


def _component(
    name: str,
    status: str,
    *,
    reason: str = "",
    effect: str | None = None,
    **detail,
) -> dict:
    """One component row. A non-PRESENT row never carries an effect."""
    row = {
        "component": name,
        "status": status,
        "wired": name in DECOMPOSITION_WIRED_COMPONENTS,
        "effect": effect if status == DECOMP_STATUS_PRESENT else None,
        "reason": reason,
    }
    if status == DECOMP_STATUS_PRESENT:
        row.update(detail)
    return row


def _not_wired(name: str) -> dict:
    """A component no forecast path consumes yet.

    Deliberately NOT reported as a zero effect: zero would claim it was
    measured and found irrelevant.
    """
    reasons = {
        "fundamental": (
            "fundamentals are captured daily into the W6 ledger, but no "
            "forecast path consumes them yet — this is unmeasured, not zero"
        ),
        "macro": (
            "the macro snapshot reads UNAVAILABLE without a FRED key, and no "
            "forecast path consumes it yet — this is unmeasured, not zero"
        ),
        "sentiment": (
            "sentiment carries ensemble weight 0.0 and has no verified "
            "provider; nothing is inferred from price — this is unmeasured, "
            "not zero"
        ),
    }
    return _component(
        name,
        DECOMP_STATUS_NOT_WIRED,
        reason=reasons.get(name, "no forecast path consumes this component yet"),
    )


# ---------------------------------------------------------------------------
# Marginal effects — measured, never summed
# ---------------------------------------------------------------------------


def marginal_effect(
    narrowed_successes: int,
    narrowed_trials: int,
    base_successes: int,
    base_trials: int,
) -> dict:
    """How one filter changed the rate, as a MARGINAL slice — never a share.

    Returns the narrowed rate with its own sample size and interval, the base
    rate it is measured against, and the difference. The difference is
    reported as `rate_difference`, never `contribution`: these differences
    overlap across components and do not sum to the forecast.
    """
    if narrowed_trials < 0 or base_trials < 0:
        raise DecompositionError("sample sizes must not be negative")
    if narrowed_trials > base_trials:
        raise DecompositionError(
            f"a narrowed slice ({narrowed_trials}) cannot be larger than the "
            f"base it was drawn from ({base_trials})"
        )

    narrowed = wilson_interval(narrowed_successes, narrowed_trials)
    base = wilson_interval(base_successes, base_trials)
    narrowed_rate = (
        narrowed_successes / narrowed_trials if narrowed_trials else None
    )
    base_rate = base_successes / base_trials if base_trials else None

    difference = (
        round(narrowed_rate - base_rate, 6)
        if narrowed_rate is not None and base_rate is not None
        else None
    )

    claim, claim_reason = select_claim(narrowed_trials, narrowed, base_rate)

    return {
        "samples": narrowed_trials,
        "base_samples": base_trials,
        "excluded": base_trials - narrowed_trials,
        # The rate is only published when the slice can support a claim at
        # all — F4's floors, not a second policy.
        "rate": round(narrowed_rate, 6)
        if narrowed_rate is not None and claim != CONDITIONAL_CLAIM_INSUFFICIENT
        else None,
        "base_rate": round(base_rate, 6) if base_rate is not None else None,
        # NOT "contribution". These differences overlap and do not sum.
        "rate_difference": difference,
        "interval": narrowed if claim != CONDITIONAL_CLAIM_INSUFFICIENT else None,
        "claim": claim,
        "claim_reason": claim_reason,
        "additive": DECOMPOSITION_ADDITIVE,
    }


def _effect_label(narrowed_trials: int, base_trials: int) -> str:
    """Did this filter actually cut anything?"""
    if base_trials <= 0:
        return DECOMP_EFFECT_UNMEASURED
    if narrowed_trials == base_trials:
        return DECOMP_EFFECT_NO_EFFECT
    return DECOMP_EFFECT_NARROWED


# ---------------------------------------------------------------------------
# Decomposing an F5 event-conditioned forecast
# ---------------------------------------------------------------------------


def decompose_event_forecast(forecast: dict, pool_size: int | None = None) -> dict:
    """Decompose an F5 pipeline into its seven components.

    F5 is the richest surface to decompose: its value is an analog base rate
    produced by three successive filters — chart similarity (technical), event
    type (news-event) and the retrieved set itself (historical-analog) — with
    the regime reported alongside.
    """
    if not isinstance(forecast, dict):
        raise DecompositionError("a forecast dict is required")

    stages = forecast.get("stages") or {}
    matches = stages.get("matches") or {}
    regime_stage = stages.get("regime") or {}
    forecast_stage = stages.get("forecast") or {}

    analogs = int(matches.get("analogs") or 0)
    scorable = int(matches.get("scorable_analogs") or 0)
    pool = int(pool_size if pool_size is not None else (matches.get("pool_size") or 0))
    samples = int(forecast.get("samples") or 0)
    successes = int(forecast_stage.get("successes") or 0)

    rows: dict[str, dict] = {}

    # -- technical: the chart state IS the retrieval key --------------------
    if analogs:
        rows[DECOMP_TECHNICAL] = _component(
            DECOMP_TECHNICAL,
            DECOMP_STATUS_PRESENT,
            effect=_effect_label(analogs, pool),
            reason=(
                f"chart similarity selected {analogs} of {pool} memories; the "
                f"chart state is the retrieval key, so this filter and the "
                f"event-type filter act on the SAME pool and their effects "
                f"overlap"
            ),
            selected=analogs,
            pool=pool,
            similarity_floor=matches.get("min_similarity"),
        )
    else:
        rows[DECOMP_TECHNICAL] = _component(
            DECOMP_TECHNICAL,
            DECOMP_STATUS_ABSENT,
            reason=matches.get("reason")
            or "no comparable chart state was retrieved",
        )

    # -- news_event: the event type the analogs had to match ---------------
    event_type = (forecast.get("event") or {}).get("event_type")
    if analogs and event_type:
        rows[DECOMP_NEWS_EVENT] = _component(
            DECOMP_NEWS_EVENT,
            DECOMP_STATUS_PRESENT,
            effect=_effect_label(analogs, pool),
            reason=(
                f"analogs were restricted to {event_type!r} events; an "
                f"earnings surprise is not a comparable for a regulatory "
                f"action, so the type filter is part of what produced this rate"
            ),
            event_type=event_type,
            observed_share=matches.get("observed_share"),
            associated_share=matches.get("associated_share"),
        )
    else:
        rows[DECOMP_NEWS_EVENT] = _component(
            DECOMP_NEWS_EVENT,
            DECOMP_STATUS_ABSENT,
            reason="no event-typed analog set entered this forecast",
        )

    # -- regime: reported, not filtered on ---------------------------------
    regime = regime_stage.get("regime")
    if regime:
        agreement = regime_stage.get("agreement")
        rows[DECOMP_REGIME] = _component(
            DECOMP_REGIME,
            DECOMP_STATUS_PRESENT,
            # The regime did not narrow this set: F5 reports agreement rather
            # than filtering, because market_regime is already inside the
            # similarity and filtering again would double-weight it.
            effect=DECOMP_EFFECT_NO_EFFECT,
            reason=(
                f"regime {regime!r} held at as_of; {agreement if agreement is not None else 'an unmeasured share'} "
                f"of the analogs occurred in the same regime. F5 REPORTS this "
                f"rather than filtering on it, because market_regime is "
                f"already one of the similarity fields"
            ),
            regime=regime,
            agreement=agreement,
            filtered=False,
        )
    else:
        rows[DECOMP_REGIME] = _component(
            DECOMP_REGIME,
            DECOMP_STATUS_ABSENT,
            reason=regime_stage.get("reason") or "no regime was established",
        )

    # -- historical_analog: the set that actually produced the value -------
    if samples:
        effect = marginal_effect(successes, samples, successes, samples)
        rows[DECOMP_HISTORICAL_ANALOG] = _component(
            DECOMP_HISTORICAL_ANALOG,
            DECOMP_STATUS_PRESENT,
            effect=_effect_label(scorable or samples, analogs or samples),
            reason=(
                f"{samples} analog outcome(s) produced the forecast value; "
                f"this is the component the number IS, not one input among "
                f"several"
            ),
            samples=samples,
            distinct_tickers=forecast_stage.get("distinct_tickers"),
            effective_samples=forecast_stage.get("effective_samples"),
            claim=forecast.get("claim"),
            interval=forecast.get("interval"),
            measured=effect,
        )
    else:
        rows[DECOMP_HISTORICAL_ANALOG] = _component(
            DECOMP_HISTORICAL_ANALOG,
            DECOMP_STATUS_ABSENT,
            reason=forecast_stage.get("reason")
            or "no analog outcome was available",
        )

    for name in DECOMPOSITION_COMPONENTS:
        rows.setdefault(name, _not_wired(name))

    return _assemble(
        rows,
        source="forecast_event",
        ticker=(forecast.get("event") or {}).get("entity"),
        as_of=forecast.get("as_of"),
        horizon=forecast.get("horizon"),
        headline_claim=forecast.get("claim"),
        headline_value=forecast.get("value") if "value" in forecast else None,
        has_value="value" in forecast,
    )


# ---------------------------------------------------------------------------
# Decomposing an F4 conditional cell
# ---------------------------------------------------------------------------


def decompose_conditional_cell(cell: dict, total_observations: int = 0) -> dict:
    """Decompose one F4 conditional cell.

    F4 conditions on the regime alone, so its decomposition is deliberately
    sparse: regime and historical-analog are PRESENT, the rest are not.
    Padding it with plausible-looking rows would be the false certainty this
    sprint exists to avoid.
    """
    if not isinstance(cell, dict):
        raise DecompositionError("a cell dict is required")

    rows: dict[str, dict] = {}
    samples = int(cell.get("samples") or 0)
    successes = int(cell.get("successes") or 0)
    base_rate = cell.get("base_rate")

    condition_value = cell.get("condition_value")
    if condition_value:
        rows[DECOMP_REGIME] = _component(
            DECOMP_REGIME,
            DECOMP_STATUS_PRESENT,
            effect=_effect_label(samples, total_observations or samples),
            reason=(
                f"the slice was restricted to regime {condition_value!r}; "
                f"{samples} of {total_observations or samples} observation(s) "
                f"qualified"
            ),
            regime=condition_value,
            filtered=True,
            samples=samples,
        )
    else:
        rows[DECOMP_REGIME] = _component(
            DECOMP_REGIME, DECOMP_STATUS_ABSENT,
            reason="no condition value was supplied",
        )

    if samples:
        base_successes = (
            int(round(float(base_rate) * (total_observations or samples)))
            if base_rate is not None
            else successes
        )
        rows[DECOMP_HISTORICAL_ANALOG] = _component(
            DECOMP_HISTORICAL_ANALOG,
            DECOMP_STATUS_PRESENT,
            effect=_effect_label(samples, total_observations or samples),
            reason=(
                f"{samples} historical observation(s) in this slice produced "
                f"the cell's reading"
            ),
            samples=samples,
            claim=cell.get("claim"),
            interval=cell.get("interval"),
            measured=marginal_effect(
                successes, samples,
                base_successes, total_observations or samples,
            ),
        )
    else:
        rows[DECOMP_HISTORICAL_ANALOG] = _component(
            DECOMP_HISTORICAL_ANALOG,
            DECOMP_STATUS_ABSENT,
            reason=cell.get("reason") or "the slice held no observations",
        )

    # F4 conditions on the regime only. Technical and news-event are WIRED in
    # general but supplied nothing HERE, which is ABSENT rather than NOT_WIRED.
    for name in (DECOMP_TECHNICAL, DECOMP_NEWS_EVENT):
        rows[name] = _component(
            name, DECOMP_STATUS_ABSENT,
            reason=(
                "an F4 conditional cell slices on the market regime alone; "
                "this component is wired elsewhere but supplied nothing here"
            ),
        )

    for name in DECOMPOSITION_COMPONENTS:
        rows.setdefault(name, _not_wired(name))

    return _assemble(
        rows,
        source="forecast_conditional",
        ticker=None,
        as_of=None,
        horizon=cell.get("horizon"),
        headline_claim=cell.get("claim"),
        headline_value=cell.get("value") if "value" in cell else None,
        has_value="value" in cell,
    )


def _assemble(
    rows: dict,
    *,
    source: str,
    ticker,
    as_of,
    horizon,
    headline_claim,
    headline_value,
    has_value: bool,
) -> dict:
    present = [n for n, r in rows.items() if r["status"] == DECOMP_STATUS_PRESENT]
    absent = [n for n, r in rows.items() if r["status"] == DECOMP_STATUS_ABSENT]
    unwired = [n for n, r in rows.items() if r["status"] == DECOMP_STATUS_NOT_WIRED]

    payload = {
        "decomposed_object": DECOMPOSED_OBJECT,
        "source": source,
        "ticker": ticker,
        "as_of": as_of,
        "horizon": horizon,
        "component_order": list(DECOMPOSITION_COMPONENTS),
        "components": {name: rows[name] for name in DECOMPOSITION_COMPONENTS},
        "present": present,
        "absent": absent,
        "not_wired": unwired,
        "headline_claim": headline_claim,
        "additive": DECOMPOSITION_ADDITIVE,
        "overlap_evidence": DECOMPOSITION_OVERLAP_EVIDENCE,
        "disclaimer": DECOMPOSITION_DISCLAIMER,
        "calculation_version": FORECAST_DECOMPOSITION_VERSION,
        "contract_version": DECOMPOSITION_CONTRACT_VERSION,
    }
    # THE SHAPE RULE, inherited from F3/F4/F5: a headline value key exists IFF
    # the underlying forecast published one.
    if has_value and headline_value is not None:
        payload["headline_value"] = headline_value
    return payload


def decomposition_problems(decomposition: dict) -> list[str]:
    """Validate a decomposition against the F6 contract."""
    problems: list[str] = []
    if not isinstance(decomposition, dict):
        return ["decomposition must be a dict"]

    if decomposition.get("decomposed_object") != DECOMPOSED_OBJECT:
        problems.append(
            "the decomposition does not say it describes the FORECAST — a "
            "reader could mistake it for W1's ensemble breakdown of the score"
        )
    if decomposition.get("additive") is not False:
        problems.append(
            "the decomposition claims to be additive; " + DECOMPOSITION_OVERLAP_EVIDENCE
        )
    for field in ("disclaimer", "overlap_evidence", "calculation_version"):
        if not decomposition.get(field):
            problems.append(f"decomposition field {field!r} missing/empty")

    components = decomposition.get("components") or {}
    if list(components) != list(DECOMPOSITION_COMPONENTS):
        problems.append(
            "the decomposition does not report every declared component in "
            "order — a missing row reads as an oversight where an explicit "
            "NOT_WIRED reads as a fact"
        )

    for name, row in components.items():
        status = row.get("status")
        if status not in (
            DECOMP_STATUS_PRESENT, DECOMP_STATUS_ABSENT, DECOMP_STATUS_NOT_WIRED
        ):
            problems.append(f"{name}: unknown status {status!r}")
        if status != DECOMP_STATUS_PRESENT and row.get("effect") is not None:
            problems.append(
                f"{name}: a {status} component reports an effect — only a "
                f"PRESENT component has one to report"
            )
        if status != DECOMP_STATUS_PRESENT and not row.get("reason"):
            problems.append(f"{name}: a {status} component does not explain itself")
        if status == DECOMP_STATUS_NOT_WIRED:
            if name in DECOMPOSITION_WIRED_COMPONENTS:
                problems.append(
                    f"{name}: reported NOT_WIRED although it is declared wired"
                )
            for key in ("rate", "rate_difference", "samples"):
                if key in row:
                    problems.append(
                        f"{name}: an unwired component carries {key!r} — it was "
                        f"never measured, and a number here claims otherwise"
                    )

        measured = row.get("measured") or {}
        if measured:
            if measured.get("additive") is not False:
                problems.append(f"{name}: a measured effect claims to be additive")
            if "contribution" in measured:
                problems.append(
                    f"{name}: a measured effect reports a 'contribution' — the "
                    f"components overlap and do not sum to the forecast"
                )
            if (
                measured.get("claim") == CONDITIONAL_CLAIM_INSUFFICIENT
                and measured.get("rate") is not None
            ):
                problems.append(
                    f"{name}: an INSUFFICIENT slice published a rate"
                )
            width = interval_width(measured.get("interval"))
            if measured.get("interval") is not None and width is None:
                problems.append(f"{name}: an interval carries no bounds")

    return problems


def render_components(decomposition: dict) -> list[dict]:
    """One reading row per component, in declared order.

    Never leaves a blank: a component that supplied nothing renders its
    reason, so the panel always explains itself.
    """
    rows: list[dict] = []
    for name in decomposition.get("component_order") or ():
        row = (decomposition.get("components") or {}).get(name) or {}
        measured = row.get("measured") or {}
        rows.append(
            {
                "component": name,
                "status": row.get("status"),
                "effect": row.get("effect"),
                "samples": measured.get("samples") or row.get("samples"),
                "rate": measured.get("rate"),
                "rate_difference": measured.get("rate_difference"),
                "interval": measured.get("interval") or row.get("interval"),
                "reason": row.get("reason") or "",
            }
        )
    return rows
