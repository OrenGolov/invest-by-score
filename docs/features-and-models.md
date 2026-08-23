# Features and Models

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