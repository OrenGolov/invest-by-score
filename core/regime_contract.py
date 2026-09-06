"""Market regime contract (Sprint N4) — stable entry point.

Thin, documented wrapper over `core.regime_agent.build_regime_snapshot`.
All status semantics are documented on the `RegimeSnapshot` dataclass and
enforced at the orchestrator level: STRESS couples to NO_TRADE through the
W2 veto rule `market_regime_stress` (core/risk_policy.py is the only
evaluator); RISK_OFF dampens momentum coefficients in core/score_engine.py;
INCOMPLETE/UNAVAILABLE regime evidence floors the decision posture and
triggers the fail-closed veto — absence of regime context is never a
neutral label.
"""

from __future__ import annotations

from core.regime_agent import build_regime_snapshot


def fetch_regime_snapshot(ticker: str, as_of: str) -> dict:
    """Return the point-in-time five-state market regime snapshot.

    Thin, stable entry point over `core.regime_agent.build_regime_snapshot`
    (Sprint N4). Callers treat `status` as the source of truth:

    - OK: every strict window was populated from eligible PIT bars and the
      rule chain produced a label from REGIME_LABELS with a probability
      proxy and trailing transition risk.
    - INCOMPLETE: eligible history is shorter than the strict windows; the
      regime is explicitly None (never a fabricated neutral).
    - UNAVAILABLE: the provider fetch failed or returned no eligible bars.
    - INVALID: the payload violated the OHLC schema.
    """
    return build_regime_snapshot(ticker, as_of)
