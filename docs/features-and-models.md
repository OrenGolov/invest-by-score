# Features and Models

## Feature Registry

The registry is the contract between ingestion, feature generation, agents, and
backtesting. Each row defines `feature_name`, owner agent, domain, input tables,
formula or transformer version, frequency, lookback, units, null policy,
publication-time rule, quality threshold, and model compatibility. A feature
cannot be used until its registry row is approved and its point-in-time tests
pass.

| Owner | Feature families |
| --- | --- |
| Market Data | returns, OHLCV, VWAP, ATR, volatility, volume surprise, breadth, liquidity |
| Technical Analysis | trend, momentum, relative strength, support/resistance, breakout, mean reversion |
| Fundamental Analysis | growth, margins, cash flow, leverage, valuation, dilution, buybacks |
| News Intelligence | event type, novelty, relevance, recency, cited fact extraction, contradiction |
| Sentiment | source-normalized tone, volume, disagreement, positioning, manipulation flags |
| Macroeconomic | rates, inflation, yields, employment, GDP, policy surprise, sensitivities |
| Market Regime | regime probabilities, transition risk, volatility state, risk-on/off |
| Risk Management | exposure, correlation, concentration, stress, drawdown, liquidity capacity |
| Performance Auditor | leakage indicators, drift, calibration residuals, disagreement, bias checks |

## Feature Groups

- Market: returns, OHLCV, gaps, VWAP distance, ATR, realized volatility, volume
  surprise, liquidity, breadth, and options-implied volatility where licensed.
- Technical: multi-horizon trend and momentum, relative strength, volatility
  regime, support/resistance distance, breakout and mean-reversion signals.
- Fundamental: revenue and earnings growth, margins, cash flow quality,
  leverage, liquidity, valuation, dilution, buybacks, guidance, and moat.
- News: event type, novelty, source reliability, entity relevance, recency,
  sentiment, and contradiction, with extracted facts beside citations.
- Sentiment: source-normalized tone, volume, disagreement, positioning,
  trend, and manipulation indicators.
- Macro: rates, inflation, yields, employment, GDP, policy surprises, and
  ticker/sector sensitivity to macro scenarios.

Every feature has a calculation version, unit, lookback, missingness flag, and
point-in-time eligibility test. Avoid features that require future bars or
revised data unavailable at prediction time.

The feature store writes to the TimescaleDB `features` hypertable defined in
`db/migrations/001_initial_schema.sql`. Features are materialized by
`(instrument_id, as_of, feature_set_version)` and retain all source hashes.
The backtester consumes this same store and cannot receive a separately
constructed research dataset.

## Leakage Enforcement

At feature-build time, require `published_time <= as_of` for every input and
`source_cutoff <= as_of` for the output. At training time, reject rows whose
label horizon overlaps the validation interval, apply a time embargo, and
freeze the feature/model registry version. Tests must include future-dated
filings, revised macro releases, corporate actions, and unavailable publication
times to prove they are excluded.

## Labels and Scores

Define labels before training. Store future returns and adverse excursion for
1d, 5d, 20d, and 60d horizons, plus a risk-adjusted outcome. Calibrate
probabilities on time-ordered validation data. The user-facing 0-to-10 score is
a documented mapping of calibrated expected quality, confidence, and risk; it
is not a probability of profit.

## Learning Loop

Daily: ingest and validate data, create eligible features, persist predictions,
close mature horizons, calculate calibration and false-positive/false-negative
metrics, update source reliability, and generate a learning report. Retraining
creates a candidate version. Promotion requires out-of-sample comparison,
drift checks, reproducibility, and a release gate; historical predictions never
change.