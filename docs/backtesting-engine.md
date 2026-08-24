# Backtesting Engine

## 1. Goal

Validate the quality of scoring and risk rules before paper or live trading.

## 2. Required market regimes to test

- bull market
- bear market
- sideways market
- high volatility periods
- earnings seasons
- interest rate changes
- stress events

## 3. Minimum validation requirements

The platform must backtest across multiple market cycles and produce metrics including:

- CAGR
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- profit factor
- win rate
- max drawdown
- turnover and slippage impact

## 4. Rejection criteria

Reject any strategy that fails risk thresholds, especially if:
- max drawdown is too high
- Sharpe is negative or weak
- false positives are excessive
- regime dependence is unstable
- data leakage is detected

## 5. Walk-forward design

Use rolling windows for model training and validation.

Example:
- train on 3 years of data
- validate on next 3 months
- test on next 1 month
- repeat forward over time

This prevents optimistic overfitting.

## 6. Outcome labeling

Each forecast must be matched to realized outcomes after the relevant horizon.

Record:
- forecast timestamp
- prediction value
- realized return
- direction accuracy
- portfolio impact
- whether quality or risk rules blocked trading

## 7. Simulation standards

- simulate realistic fills and costs
- include slippage assumptions
- separate signal quality from execution quality
- simulate risk limits and vetoes

## 8. Backtesting result storage

Store results in a report table including:
- run_id
- model_version
- training_window
- validation_window
- metrics
- regime_coverage
- failures
- recommendation

## 9. Backtesting policy

No model can move into paper or live mode unless it passes the required criteria with evidence and reproducibility.
