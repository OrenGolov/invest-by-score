# Validation, Paper Trading, and Monitoring

## Shared Research Contract

The backtesting engine reads the phase 2 normalized tables and phase 4 feature
store directly. It shares the same feature registry, model registry, source
lineage, timestamp eligibility checks, and score schema as live analysis. An
ad-hoc dataset or feature calculation outside those registries is invalid.

Every run records a `backtest_id`, code commit, data snapshot/hash, feature-set
version, model versions, configuration, transaction-cost assumptions, calendar,
and random seeds. A model version cannot be evaluated without its training data
cutoff and artifact identity.

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

Walk-forward sequence: train on `[t0, t1]`, validate on `[t1 + embargo, t2]`,
advance the window, and test the final frozen model on untouched data. No
future labels, revised values, or later source reliability weights may enter a
prior fold.

## Paper-Trading Gate

Live capital remains disabled until all are true:

- at least 6 months of paper trading and 500 or more simulated trades
- stable profitability across rolling windows and relevant regimes
- positive Sharpe and profit factor, acceptable drawdown, and no unexplained exceptions
- vetoes function under failure injection
- predictions, fills, lineage, and model versions are replayable
- explicit human approval is recorded

## A/B and Shadow Testing

Every candidate model first runs in shadow mode beside the approved model with
identical point-in-time inputs and no order authority. A/B allocation is
simulated only after shadow parity checks pass; assignments are deterministic,
balanced by instrument and regime, and recorded in `audit_events`. Compare
calibration, false positives, veto rate, drawdown, slippage, and latency. A
candidate cannot promote if it worsens false-positive or risk thresholds, even
when its return is higher.

The paper engine logs every decision, including `NO_TRADE`: all nine agent
outputs, evidence IDs, model versions, feature hash, ensemble weights, regime,
risk checks, auditor objections, vetoes, order intent, simulated fill, and
rejection reason. The minimums are a release gate, not a target that can be
waived by performance.

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

## Dashboard API Contract

Execution is out of scope until phases 1 through 7 are approved. The future API
must return a complete JSON snapshot to the dashboard, not an order command:

```json
{
	"ticker": "AAPL",
	"as_of": "2026-08-23T14:30:00Z",
	"score": 7.1,
	"confidence": 0.64,
	"action_state": "ANALYSIS_ONLY",
	"agent_outputs": [],
	"regime": {},
	"risk": {"veto": true, "reasons": []},
	"auditor": {"veto": false, "findings": []},
	"source_reliability": [],
	"fx_exposure": {},
	"data_quality": {},
	"model_versions": [],
	"evidence": [],
	"generated_at": "2026-08-23T14:30:02Z"
}
```

The dashboard must display per-agent health and confidence over time, source
reliability trends, live exposure/drawdown limits, and FX exposure. API output
must include stale-data and missing-agent warnings so a score cannot hide
degraded confidence.