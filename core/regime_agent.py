"""Market regime classifier (Sprint N4) — five states, versioned, deterministic.

Pipeline (every stage a pure, deterministic function; `build_regime_snapshot`
wires them in order):

    FETCH -> PIT FILTER -> FEATURE SERIES -> RULE CHAIN (per session)
         -> LABEL + PROBABILITY PROXY + TRANSITION RISK -> EVIDENCE-BACKED OUTPUT

States: bullish, bearish, range, risk_off, stress (REGIME_LABELS).

Classification v1 rules, evaluated per session with strict precedence
stress > risk_off > range > bearish > bullish:

- stress:   30d realized vol strictly above its own trailing 1y 95th
            percentile AND drawdown from the 60-session high strictly above
            REGIME_STRESS_DRAWDOWN_THRESHOLD.
- risk_off: 30d realized vol strictly above its own trailing 1y 80th
            percentile OR (MA50 < MA200 with 20d AND 60d momentum both
            strictly negative).
- range:    |price_vs_ma_50| and |price_vs_ma_200| both strictly inside
            +/- REGIME_RANGE_FLAT_THRESHOLD.
- bearish:  close < MA200 AND MA50 < MA200 (the legacy market-data
            heuristic), without the aligned negative momentum that would
            have selected risk_off first.
- bullish:  everything else once every input is available.

Governance rules enforced here:

- Single source: the five-state governed regime lives ONLY in this module.
  `agents.market_data_agent.fetch_market_snapshot["market_regime"]` stays a
  legacy, ungoverned display heuristic and is never read for decisions.
- Point-in-time: bars with timestamp > as_of are excluded before any feature
  is computed (and counted, never silently dropped). Volatility percentiles
  compare the current session against the PRIOR sessions only, so a session
  is never part of its own reference distribution.
- Fail-closed: a failed or empty fetch is UNAVAILABLE; a payload violating
  the OHLC schema is INVALID; eligible history shorter than the strict
  windows is INCOMPLETE with `regime` explicitly None. A partial rule
  evaluation would make the label depend on data availability rather than
  the contract, so absence of regime evidence is a status, never a neutral
  label.
- Determinism: no wall-clock reads in the classification path, no
  randomness; identical eligible inputs produce identical snapshots.
- STRESS -> NO_TRADE: enforced via the W2 veto rule `market_regime_stress`
  (core/risk_policy.py is the only evaluator; core/config.py the only table).
- RISK_OFF dampening: applied in core/score_engine.py via
  REGIME_RISKOFF_MOMENTUM_DAMPING; this module only reports the label.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.config import (
    REGIME_CALENDAR_COVERAGE_DAYS,
    REGIME_CLASSIFIER_VERSION,
    REGIME_CONTRACT_VERSION,
    REGIME_DRAWDOWN_HIGH_WINDOW_SESSIONS,
    REGIME_LABELS,
    REGIME_MOMENTUM_MARGIN_SCALE,
    REGIME_PIPELINE_VERSION,
    REGIME_RANGE_FLAT_THRESHOLD,
    REGIME_REALIZED_VOL_WINDOW_SESSIONS,
    REGIME_REQUIRED_SESSIONS,
    REGIME_RISKOFF_LABEL,
    REGIME_RISKOFF_VOL_PERCENTILE,
    REGIME_STRESS_DRAWDOWN_THRESHOLD,
    REGIME_STRESS_LABEL,
    REGIME_STRESS_VOL_PERCENTILE,
    REGIME_TREND_MARGIN_SCALE,
    REGIME_TRANSITION_WINDOW_SESSIONS,
    REGIME_VOL_PERCENTILE_LOOKBACK_SESSIONS,
)
from fetch_data import fetch_price_history

LOGGER = logging.getLogger("core.regime_agent")

REGIME_SOURCE_ID = "yahoo_finance_chart"
UNAVAILABLE_SOURCE_ID = "regime_source_unavailable"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _regime_fetch_period(days_back: int) -> str:
    """Deterministic provider period covering as_of plus the strict windows.

    The trailing 1y volatility percentile needs ~283 sessions behind as_of
    (~430 calendar days). When this period string matches the market-data
    agent's for the same as_of, both agents share one cached fetch.
    """
    needed = int(days_back) + REGIME_CALENDAR_COVERAGE_DAYS
    if needed <= 730:
        return "2y"
    if needed <= 1825:
        return "5y"
    return "10y"


def evaluate_rules(
    realized_vol: float,
    vol_q80: float,
    vol_q95: float,
    drawdown_from_60d_high: float,
    price_vs_ma_50: float,
    price_vs_ma_200: float,
    ma_50: float,
    ma_200: float,
    close: float,
    change_20d: float,
    change_60d: float,
) -> dict:
    """Evaluate the v1 rule chain for one session from exact scalar inputs.

    Pure and deterministic; boundary semantics are strict by contract:
    every threshold comparison in the spec is "strictly above/inside". This
    is the boundary-testable core; `classify_regime` supplies it with the
    per-session series values.
    """
    vol_above_q95 = realized_vol > vol_q95
    drawdown_exceeded = drawdown_from_60d_high > REGIME_STRESS_DRAWDOWN_THRESHOLD
    stress = vol_above_q95 and drawdown_exceeded

    vol_above_q80 = realized_vol > vol_q80
    ma_downtrend = ma_50 < ma_200
    aligned_negative_momentum = change_20d < 0.0 and change_60d < 0.0
    risk_off = (not stress) and (
        vol_above_q80 or (ma_downtrend and aligned_negative_momentum)
    )

    flat = (
        abs(price_vs_ma_50) < REGIME_RANGE_FLAT_THRESHOLD
        and abs(price_vs_ma_200) < REGIME_RANGE_FLAT_THRESHOLD
    )
    range_hit = (not stress) and (not risk_off) and flat

    bearish = (
        (not stress) and (not risk_off) and (not range_hit)
        and close < ma_200 and ma_50 < ma_200
    )

    if stress:
        label = REGIME_STRESS_LABEL
    elif risk_off:
        label = REGIME_RISKOFF_LABEL
    elif range_hit:
        label = "range"
    elif bearish:
        label = "bearish"
    else:
        label = "bullish"

    rule_trace = {
        "stress": {
            "triggered": stress,
            "vol_above_q95": vol_above_q95,
            "drawdown_exceeded": drawdown_exceeded,
            "detail": (
                f"vol {realized_vol:.6f} vs q95 {vol_q95:.6f}; "
                f"drawdown {drawdown_from_60d_high:.4f} vs "
                f"{REGIME_STRESS_DRAWDOWN_THRESHOLD:.2f}"
            ),
        },
        "risk_off": {
            "triggered": risk_off,
            "vol_above_q80": vol_above_q80,
            "ma_downtrend": ma_downtrend,
            "aligned_negative_momentum": aligned_negative_momentum,
            "detail": (
                f"vol {realized_vol:.6f} vs q80 {vol_q80:.6f}; "
                f"ma50 {ma_50:.4f} vs ma200 {ma_200:.4f}; "
                f"chg20 {change_20d:.4f}, chg60 {change_60d:.4f}"
            ),
        },
        "range": {
            "triggered": range_hit,
            "detail": (
                f"|pvs50| {abs(price_vs_ma_50):.4f}, |pvs200| {abs(price_vs_ma_200):.4f} "
                f"vs flat band {REGIME_RANGE_FLAT_THRESHOLD:.2f}"
            ),
        },
        "bearish": {
            "triggered": bearish,
            "detail": f"close {close:.4f} vs ma200 {ma_200:.4f}; ma50 {ma_50:.4f}",
        },
        "bullish": {
            "triggered": label == "bullish",
            "detail": "default state once every higher-precedence rule is clean",
        },
        "precedence": "stress > risk_off > range > bearish > bullish",
    }
    return {"label": label, "rule_trace": rule_trace}


def _probability_proxy(label: str, inputs: dict) -> float:
    """Distance from the deciding boundary, scaled to [0, 1] and clipped.

    Per-label semantics (documented, versioned via REGIME_CLASSIFIER_VERSION):
    - stress:   min of the two condition depths — the binding constraint
                leaves the region first.
    - risk_off: max of the triggered branch depths (vol excess over q80, or
                momentum shortfall over the margin scale).
    - range:    depth inside the flat band (1.0 = dead flat).
    - bearish:  binding shortfall across the two bearish conditions.
    - bullish:  distance above the MA200 cross (the bearish boundary).
    """
    vol = inputs["realized_vol_30d"]
    q80 = inputs["vol_q80"]
    q95 = inputs["vol_q95"]
    drawdown = inputs["drawdown_from_60d_high"]
    pvs50 = inputs["price_vs_ma_50"]
    pvs200 = inputs["price_vs_ma_200"]
    ma_50 = inputs["ma_50"]
    ma_200 = inputs["ma_200"]
    chg20 = inputs["change_20d"]
    chg60 = inputs["change_60d"]

    if label == REGIME_STRESS_LABEL:
        vol_depth = _clip01((vol - q95) / q95) if q95 > 0 else 1.0
        dd_depth = _clip01(
            (drawdown - REGIME_STRESS_DRAWDOWN_THRESHOLD) / REGIME_STRESS_DRAWDOWN_THRESHOLD
        )
        return round(min(vol_depth, dd_depth), 4)
    if label == REGIME_RISKOFF_LABEL:
        vol_depth = _clip01((vol - q80) / q80) if q80 > 0 else 1.0
        momentum_depth = _clip01(min(-chg20, -chg60) / REGIME_MOMENTUM_MARGIN_SCALE)
        return round(max(vol_depth, momentum_depth), 4)
    if label == "range":
        depth = 1.0 - max(abs(pvs50), abs(pvs200)) / REGIME_RANGE_FLAT_THRESHOLD
        return round(_clip01(depth), 4)
    if label == "bearish":
        close_shortfall = (ma_200 - inputs["close"]) / ma_200 if ma_200 > 0 else 0.0
        ma_shortfall = (ma_200 - ma_50) / ma_200 if ma_200 > 0 else 0.0
        return round(_clip01(min(close_shortfall, ma_shortfall) / REGIME_TREND_MARGIN_SCALE), 4)
    # bullish
    return round(_clip01(pvs200 / REGIME_TREND_MARGIN_SCALE), 4)


def classify_regime(history: pd.DataFrame) -> dict:
    """Classify every eligible session and return the latest classification.

    `history` must be the point-in-time frame (bars <= as_of already). Pure
    and deterministic. Returns a dict with `computable`, and — when the
    newest session has every input — `label`, `probability_proxy`,
    `transition_risk`, `inputs`, `rule_trace`, and `recent_labels`.
    """
    empty = {
        "computable": False,
        "label": None,
        "probability_proxy": None,
        "transition_risk": None,
        "inputs": None,
        "rule_trace": None,
        "recent_labels": [],
        "sessions_used": int(len(history)) if history is not None else 0,
    }
    if history is None or history.empty:
        return empty
    if "Close" not in history.columns or "High" not in history.columns:
        return empty

    close = history["Close"].astype(float)
    high = history["High"].astype(float)
    sessions = len(close)

    returns = close.pct_change()
    realized_vol = returns.rolling(
        window=REGIME_REALIZED_VOL_WINDOW_SESSIONS, min_periods=REGIME_REALIZED_VOL_WINDOW_SESSIONS
    ).std(ddof=0)
    vol_prior = realized_vol.shift(1)
    quantile_window = REGIME_VOL_PERCENTILE_LOOKBACK_SESSIONS
    vol_q95 = vol_prior.rolling(window=quantile_window, min_periods=quantile_window).quantile(
        REGIME_STRESS_VOL_PERCENTILE
    )
    vol_q80 = vol_prior.rolling(window=quantile_window, min_periods=quantile_window).quantile(
        REGIME_RISKOFF_VOL_PERCENTILE
    )

    high_window = high.rolling(
        window=REGIME_DRAWDOWN_HIGH_WINDOW_SESSIONS, min_periods=REGIME_DRAWDOWN_HIGH_WINDOW_SESSIONS
    ).max()
    drawdown = 1.0 - close / high_window

    ma_50 = close.rolling(window=50, min_periods=50).mean()
    ma_200 = close.rolling(window=200, min_periods=200).mean()
    pvs50 = close / ma_50 - 1.0
    pvs200 = close / ma_200 - 1.0
    change_20d = close.pct_change(20)
    change_60d = close.pct_change(60)

    computable = (
        realized_vol.notna()
        & vol_q95.notna()
        & vol_q80.notna()
        & drawdown.notna()
        & ma_50.notna()
        & ma_200.notna()
        & change_20d.notna()
        & change_60d.notna()
    )

    labels = pd.Series(pd.NA, index=close.index, dtype="object")
    result = {**empty, "sessions_used": sessions}

    if not bool(computable.iloc[-1]):
        result["shortfall"] = {
            "eligible_sessions": sessions,
            "required_sessions": REGIME_REQUIRED_SESSIONS,
            "windows": {
                "realized_vol_30d": int(realized_vol.notna().sum()),
                "vol_percentile_1y": int(vol_q95.notna().sum()),
                "drawdown_60d_high": int(drawdown.notna().sum()),
                "ma_200": int(ma_200.notna().sum()),
            },
        }
        return result

    latest: dict | None = None
    for position in range(len(close)):
        if not bool(computable.iloc[position]):
            continue
        evaluated = evaluate_rules(
            realized_vol=float(realized_vol.iloc[position]),
            vol_q80=float(vol_q80.iloc[position]),
            vol_q95=float(vol_q95.iloc[position]),
            drawdown_from_60d_high=float(drawdown.iloc[position]),
            price_vs_ma_50=float(pvs50.iloc[position]),
            price_vs_ma_200=float(pvs200.iloc[position]),
            ma_50=float(ma_50.iloc[position]),
            ma_200=float(ma_200.iloc[position]),
            close=float(close.iloc[position]),
            change_20d=float(change_20d.iloc[position]),
            change_60d=float(change_60d.iloc[position]),
        )
        labels.iloc[position] = evaluated["label"]
        latest = evaluated

    labeled = labels.dropna()
    window = labeled.iloc[-REGIME_TRANSITION_WINDOW_SESSIONS:]
    window_labels = [str(value) for value in window.tolist()]
    flips = sum(
        1 for previous, current in zip(window_labels, window_labels[1:]) if previous != current
    )
    transition_risk = {
        "window_sessions": REGIME_TRANSITION_WINDOW_SESSIONS,
        "labeled_sessions": len(window_labels),
        "flips": int(flips),
        "flip_rate": round(flips / (len(window_labels) - 1), 4) if len(window_labels) >= 2 else 0.0,
        "labels": window_labels,
    }

    inputs = {
        "close": round(float(close.iloc[-1]), 4),
        "realized_vol_30d": round(float(realized_vol.iloc[-1]), 6),
        "vol_q80": round(float(vol_q80.iloc[-1]), 6),
        "vol_q95": round(float(vol_q95.iloc[-1]), 6),
        "drawdown_from_60d_high": round(float(drawdown.iloc[-1]), 4),
        "price_vs_ma_50": round(float(pvs50.iloc[-1]), 4),
        "price_vs_ma_200": round(float(pvs200.iloc[-1]), 4),
        "ma_50": round(float(ma_50.iloc[-1]), 4),
        "ma_200": round(float(ma_200.iloc[-1]), 4),
        "change_20d": round(float(change_20d.iloc[-1]), 4),
        "change_60d": round(float(change_60d.iloc[-1]), 4),
        "sessions_used": sessions,
        "labeled_sessions": int(labeled.shape[0]),
    }

    result.update({
        "computable": True,
        "label": latest["label"],
        "probability_proxy": _probability_proxy(latest["label"], inputs),
        "transition_risk": transition_risk,
        "inputs": inputs,
        "rule_trace": latest["rule_trace"],
        "recent_labels": window_labels,
    })
    if result["label"] not in REGIME_LABELS:
        # Defensive: the rule chain can only emit labels from the versioned
        # set; anything else is a contract violation, not a regime.
        result.update({
            "computable": False, "label": None, "probability_proxy": None,
            "transition_risk": None, "inputs": None, "rule_trace": None,
        })
    return result


def _snapshot_frame(
    ticker: str,
    as_of: str,
    status: str,
    source_id: str,
    reason: str,
    classification: dict | None = None,
    published_time: str | None = None,
    lookback_period: str = "0 sessions",
) -> dict:
    classification = classification or {}
    snapshot = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "status": status,
        "source_id": source_id,
        "source_confidence": 0.8 if status in {"OK", "INCOMPLETE"} else 0.0,
        "published_time": published_time,
        "calculation_version": REGIME_CONTRACT_VERSION,
        "lookback_period": lookback_period,
        "regime": classification.get("label"),
        "probability_proxy": classification.get("probability_proxy"),
        "transition_risk": classification.get("transition_risk") or {},
        "inputs": classification.get("inputs") or {},
        "rule_trace": classification.get("rule_trace") or {},
        "reason": reason,
    }
    if status == "OK":
        snapshot["pipeline"] = {
            "pipeline_version": REGIME_PIPELINE_VERSION,
            "classifier_version": REGIME_CLASSIFIER_VERSION,
            "counts": {
                "sessions_used": (classification.get("inputs") or {}).get("sessions_used", 0),
                "labeled_sessions": (classification.get("inputs") or {}).get("labeled_sessions", 0),
                "transition_labels": len(classification.get("recent_labels") or []),
            },
        }
    return snapshot


def build_regime_snapshot(ticker: str, as_of: str, timestamp: str | None = None) -> dict:
    """Build the point-in-time five-state regime snapshot for a ticker.

    Failure states are statuses, never neutral labels: UNAVAILABLE (fetch
    failed / no eligible bars), INVALID (schema violation), INCOMPLETE
    (history shorter than the strict windows), OK (full classification).
    """
    as_of_text = pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S")
    target = pd.Timestamp(as_of_text)

    days_back = (pd.Timestamp.now().normalize() - target.normalize()).days
    period = _regime_fetch_period(days_back)

    try:
        history = fetch_price_history(ticker, period=period, interval="1d")
    except Exception as exc:
        LOGGER.warning("Regime fetch failed for %s: %s", ticker, exc)
        return _snapshot_frame(
            ticker, as_of_text, "UNAVAILABLE", UNAVAILABLE_SOURCE_ID,
            f"Price history fetch failed; no regime can be classified "
            f"(a status, not a neutral label). ({exc})",
        )

    if history is None or history.empty:
        return _snapshot_frame(
            ticker, as_of_text, "UNAVAILABLE", UNAVAILABLE_SOURCE_ID,
            "Provider returned no price history; no regime can be classified.",
        )

    history = history.sort_index()
    if "Close" not in history.columns or "High" not in history.columns:
        return _snapshot_frame(
            ticker, as_of_text, "INVALID", REGIME_SOURCE_ID,
            "Provider payload violated the OHLC schema (missing Close/High columns).",
        )

    future_bars_excluded = int((history.index > target).sum())
    history = history[history.index <= target]
    if history.empty:
        return _snapshot_frame(
            ticker, as_of_text, "UNAVAILABLE", UNAVAILABLE_SOURCE_ID,
            f"No bars at or before {as_of_text}; no regime can be classified.",
        )

    published_time = history.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    classification = classify_regime(history)

    if not classification["computable"]:
        shortfall = classification.get("shortfall") or {}
        reason = (
            "Eligible history is shorter than the strict regime windows: "
            f"{shortfall.get('eligible_sessions', 0)} sessions available, "
            f"{shortfall.get('required_sessions', REGIME_REQUIRED_SESSIONS)} required "
            f"(30d realized vol, trailing 1y vol percentile, MA200, 60-session drawdown "
            f"high); {future_bars_excluded} future bar(s) excluded."
        )
        return _snapshot_frame(
            ticker, as_of_text, "INCOMPLETE", REGIME_SOURCE_ID, reason,
            classification=classification, published_time=published_time,
            lookback_period=f"{classification.get('sessions_used', 0)} sessions",
        )

    inputs = classification["inputs"]
    reason = (
        f"Regime {classification['label']} classified from {inputs['sessions_used']} eligible "
        f"sessions (vol q80/q95 vs trailing {REGIME_VOL_PERCENTILE_LOOKBACK_SESSIONS}-session "
        f"window; drawdown vs {REGIME_DRAWDOWN_HIGH_WINDOW_SESSIONS}-session high); "
        f"{future_bars_excluded} future bar(s) excluded."
    )
    return _snapshot_frame(
        ticker, as_of_text, "OK", REGIME_SOURCE_ID, reason,
        classification=classification, published_time=published_time,
        lookback_period=f"{inputs['sessions_used']} sessions",
    )
