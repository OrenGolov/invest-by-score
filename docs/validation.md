# Validation, Paper Trading, and Monitoring

## Backtest Protocol

Use walk-forward splits with an embargo around evaluation windows, point-in-time
data snapshots, realistic fees, spread, slippage, liquidity, delistings,
corporate actions, and rejected orders. Test bull, bear, sideways,
high-volatility, crisis, earnings, and rate-change periods. Keep an untouched
final test period.

Report CAGR, Sharpe, Sortino, Calmar, profit factor, win rate, maximum drawdown,
turnover, exposure, rejection rate, calibration, and false-positive rate.
Reject leakage, unstable results, unacceptable drawdown, or failed risk
criteria. Do not optimize on the final test period.

## Paper-Trading Gate

Live capital remains disabled until all are true:

- at least 6 months of paper trading and 500 or more simulated trades
- stable profitability across rolling windows and relevant regimes
- positive Sharpe and profit factor, acceptable drawdown, and no unexplained exceptions
- vetoes function under failure injection
- predictions, fills, lineage, and model versions are replayable
- explicit human approval is recorded

## Capital Controls

Initial hard limits are: 1% maximum allocation per position, 20% maximum total
exposure, 1% daily loss, 3% weekly loss, and 8% monthly drawdown. Breaching any
limit immediately halts order generation and returns to `ANALYSIS_ONLY`.

## Monitoring Dashboard

Show ingestion freshness and source health, missingness, drift, agent latency
and availability, score distributions, calibration, false positives, false
negatives, regime transitions, veto counts, exposure, drawdown, paper fills,
slippage, and model version changes. Alert on stale critical data, drift,
unexpected score shifts, limit breaches, abnormal volatility, and failed audits.