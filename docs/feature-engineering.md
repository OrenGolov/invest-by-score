# Feature Engineering

## 1. Goal

Convert raw financial and market data into point-in-time features that improve predictive quality while preserving causal validity.

## 2. Feature categories

### Market features
- returns over 1d, 5d, 20d, 60d
- rolling volatility
- realized volatility
- beta vs benchmark
- correlation with sector index
- ATR ratio
- VWAP deviation
- volume trend

### Technical features
- RSI
- MACD
- stochastic momentum
- moving average slopes
- breakout state
- support/resistance proximity
- pattern flags

### Fundamental features
- earnings growth
- free cash flow growth
- operating margin
- return on equity
- valuation spread
- debt and liquidity ratios

### News features
- positive vs negative mention ratio
- earnings surprise signal
- guidance change signal
- regulatory risk signal
- analyst action summary

### Sentiment features
- social tone
- mention momentum
- institutional net sentiment
- crowd positioning drift

### Macro features
- policy-rate change expectation
- real interest rate proxies
- recession risk indicators
- yield spread shifts

## 3. Point-in-time rules

A feature can be used only if:
- the source record was published by the timestamp
- the feature has no future leakage
- the value is consistent with the exact observation date

Example:
A quarterly earnings release from 2026-06-20 is valid for an as_of of 2026-06-21, but not for an as_of of 2026-06-19.

## 4. Feature quality validation

Each feature should include:
- missingness rate
- staleness window
- volatility
- correlation with outcome variable
- source reliability score

## 5. Feature store contract

Each stored feature should include:
- symbol
- feature_name
- feature_value
- as_of
- published_time
- source_id
- quality_score
- coverage_score

## 6. Model-ready output

Feature engineering should output a table with rows for each symbol/timestamp pair and columns like:

- return_1d
- vol_20d
- rsi_14
- macd_signal
- pe_ratio
- earnings_growth_qoq
- news_impact_score
- sentiment_score
- regime_flag
- risk_score

## 7. Learning rule

The system must automatically learn which features are historically most predictive and adjust source weighting and model impact accordingly.
