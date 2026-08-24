# Multi-Agent System Design

## 1. Objective

The system outputs a score for a ticker at an exact timestamp, not a generic rating. The score must be explainable, risk-aware, and tied to point-in-time evidence.

## 2. Agent roster

### 1. Market Data Agent
Responsible for:
- live and historical OHLCV
- volume and liquidity
- volatility metrics
- VWAP, ATR
- order flow proxies
- market breadth

Outputs:
- price context
- volatility context
- volume context
- quality flags

### 2. Technical Analysis Agent
Responsible for:
- trend analysis
- momentum
- support/resistance
- breakouts
- mean reversion
- chart pattern detection

Outputs:
- bullish_score
- bearish_score
- confidence_score

### 3. Fundamental Analysis Agent
Responsible for:
- earnings growth
- revenue growth
- cash flow
- margins
- balance sheet strength
- valuation metrics
- competitive positioning

Outputs:
- fundamental_strength_score
- long_term_quality_score
- timestamp_aligned_score

### 4. News Intelligence Agent
Responsible for:
- earnings release analysis
- guidance revisions
- SEC filings
- regulatory events
- analyst actions
- macro-linked news

Outputs:
- positive_impact_probability
- negative_impact_probability
- narrative_summary

### 5. Sentiment Agent
Responsible for:
- social media sentiment
- retail positioning
- institutional discussion proxies
- market tone signals

Outputs:
- sentiment_score
- sentiment_trend

### 6. Macroeconomic Agent
Responsible for:
- rates
- inflation
- employment
- bond yields
- GDP
- policy changes

Outputs:
- macro_risk_score
- scenario_impact

### 7. Market Regime Agent
Responsible for:
- bull market
- bear market
- sideways market
- high volatility
- crisis mode
- risk-on / risk-off detection

Outputs:
- regime_label
- regime_probability
- model_weight_adjustment

### 8. Risk Management Agent
Responsible for:
- position sizing
- exposure control
- concentration checks
- correlation analysis
- stress testing
- drawdown management
- veto authority

Outputs:
- risk_score
- allowed_position_size
- veto_reason

### 9. Performance Auditor Agent
Responsible for:
- contradiction checking
- overfitting detection
- leakage checking
- bias analysis
- unsupported explanations
- veto authority

Outputs:
- audit_result
- contradictions
- veto_reason

### 10. Execution Agent
Responsible for:
- API management
- order routing
- execution quality
- slippage controls
- portfolio updates

Outputs:
- order_status
- slippage_impact
- execution_quality

## 3. Governance rules

- No single agent creates the final trade decision.
- Risk Management and Performance Auditor have veto authority.
- Execution is not allowed to override governance.
- Abnormal conditions force NO_TRADE or ANALYSIS_ONLY.

## 4. Final score flow

1. Request enters with ticker and as_of timestamp.
2. Feature store retrieves only point-in-time valid data.
3. Agents run in parallel.
4. Ensemble combines valid model outputs.
5. Risk gate validates exposure and abnormal conditions.
6. Auditor validates contradictions and leakage risk.
7. Final score is stored as a score snapshot.

## 5. Decision contract

Each agent should output this structure:

```json
{
  "agent": "technical",
  "ticker": "AAPL",
  "as_of": "2026-08-23T14:30:00Z",
  "status": "OK",
  "score": 7.1,
  "confidence": 0.64,
  "evidence": [{"source_id": "...", "reason": "..."}],
  "warnings": []
}
```

## 6. Final score output contract

```json
{
  "ticker": "AAPL",
  "as_of": "2026-08-23T14:30:00Z",
  "score": 7.4,
  "score_out_of": 10,
  "confidence": 0.78,
  "action": "ANALYSIS_ONLY",
  "components": {
    "technical": 7.9,
    "fundamental": 6.8,
    "news": 6.5,
    "sentiment": 7.2,
    "macro": 5.9,
    "regime": 7.1
  },
  "explain": [
    "Trend remains constructive",
    "Fundamental quality remains strong",
    "News is mixed but not contradictory"
  ],
  "risks": [
    "Macro sensitivity high",
    "Volatility elevated"
  ],
  "veto_status": "PASS"
}
```
