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

The rule is enforced, not aspirational. `core/contract_verification.py`
is the shared-contract toolkit: feature-contract conformance (complete
provenance, current `MARKET_FEATURE_VERSION`), diff-precise snapshot and
score-layer identity between the direct (production) path and the offline
research seam, manifest-version drift detection against the live config
constants, and a structural scan proving the research package calls the
canonical producers (`build_score`, the V1 label builder) and never
defines parallel implementations. `tests/test_shared_contracts.py` pins
all of it on every commit; a violation is a build failure.

## Canonical Feature Registry (M1)

Every feature that enters a production model must be registered in
`core/feature_registry.py` with complete metadata: name, owner, domain,
formula, version, unit, frequency, lookback, minimum history, null policy,
PIT rule, source dependencies, feature family, and model compatibility.
The version (`feature-registry-v1`, owned by
`core/config.py::FEATURE_REGISTRY_VERSION`) is stamped into every manifest.

Binding rules (enforced by the contract verifier and the engine):

- **Unregistered features are rejected.** The verifier's
  `feature_registry_problems(snapshot)` checks every feature contract on a
  snapshot against the registry; an unregistered feature is a hard failure.
  The engine also refuses a run whose registry is structurally broken.
- **Producer must exist.** A feature's owner must be one of the known
  agents (`market_data_agent`, `technical_agent`, etc.) —
  `spec_problems` rejects any unknown owner.
- **Future/revised input is rejected.** A feature contract whose
  `published_time` is after `as_of` violates the PIT rule and is rejected.
- **Deterministic feature hash.** Each `FeatureSpec` carries a
  `canonical_hash()` (SHA-256 over its canonical JSON); `registry_hash()`
  covers the whole registry. Any metadata change changes the hash.

The default registry is pre-populated with the 22 market-data features
produced by `agents/market_data_agent.py` (momentum, volatility, trend,
volume families). Persistence is append-only at
`data/feature_registry.jsonl`, idempotent per registry hash, with the same
integrity model as the framing and monitoring stores.


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

## Historical Universe (Survivorship Discipline)

A universe is a set of point-in-time membership intervals, not a list of
today's tickers. `core/universe.py` (`universe-ledger-v1`, version owned by
`core/config.py::UNIVERSE_VERSION`) keeps the append-only ledger at
`data/universe.jsonl`: each entry is `{ticker, listed_from, listed_to,
reason, source_id, published_time}` with delisted intervals preserved, so a
backtest can query who was actually listed at `as_of`.

- Fail-closed construction: `universe_as_of` includes a member only when its
  interval covers `as_of`; a `null listed_from` means "start unverified" and
  is flagged, never mistaken for evidence.
- Detection: `survivorship_problems` flags a candidate universe that drops a
  ledger member listed at `as_of` as `survivorship_biased` (fatal), and an
  unknown candidate ticker as `member_unverified` (warning).
- Engine gate: `run_walk_forward_backtest(..., universe=build_universe_block(...))`
  refuses a biased universe with `BacktestUniverseError` before any work;
  a run without a declared universe proceeds but is disclosed as
  `unverifiable_no_ledger`. Pass `fetched_tickers` so ledger members with no
  fetched price history (delisted names typically 404) surface as an explicit
  `member_price_unavailable` warning instead of being silently dropped.
- The manifest pins `versions.universe`, checked by
  `core/contract_verification.py` against the live constant — ledger-version
  drift is a failure, not a warning.

Known limitation: no index-constituent provider is connected yet, so the
ledger records what is known (seeded portfolio starts are unverified) and
flags what is not. Connecting a real constituents adapter is the upgrade
path that fills the ledger with verifiable delisted intervals.

## Corporate Actions (Price Basis)

The return basis is **price return on split-adjusted closes, dividends NOT
reinvested**. Verified live against the provider (NVDA 10:1 split ex-date
2024-06-10: pre-split bars arrive at post-split scale, not raw ~1200s), so
splits need no local adjustment math — and get none, by design. The
provider's `adjclose` series is intentionally not substituted: it is a
second, silently-revising price truth that would fork research from
production. Every run manifest carries this basis in
`provider_overrides.price_basis`, so no backtest can silently claim total
return. A dividend-reinvested basis would be a new, versioned price-basis
change — never a silent substitution.

## Transaction Costs and Slippage Assumptions

Binding for every backtest and paper fill. The cost model is a versioned
table (core/backtest/costs.py, acktest-cost-table-v2 is current;
acktest-cost-table-v1 is frozen for replaying historical assumptions),
never inline constants.

Model (per side):

`
total_bps = clamp(
    half_spread(liquidity_bucket)
    + impact_coefficient_bps * vol_factor * sqrt(participation)
    + commission_bps,
    min_total_side_cost_bps, max_total_side_cost_bps)

vol_factor = clamp(daily_vol / impact_vol_baseline_daily,
                   impact_vol_factor_min, impact_vol_factor_max)
`

- liquidity_bucket: micro / small / mid / large from the 20-session
  average dollar volume. Unknown liquidity is fail-closed: the widest
  spread applies.
- participation = order notional / average dollar volume. Unknown
  participation is fail-closed: worst case 1.0 (the order is the whole
  day's volume).
- daily_vol: realized close-to-close volatility of the trailing 20
  sessions. Unknown volatility falls back to the neutral vol factor 1.0
  and the execution record states ol_available: false — the engine
  always computes it for replays.
- The clamp makes degenerate fills impossible: never a zero-cost fill,
  never an absurd-cost fill. Every record states clamped.
- Every execution emits an itemized record (bucket, half-spread, impact,
  vol factor, commission, raw and clamped totals, cost notional) — costs
  are never hidden inside a price. The backtest aggregate reports
  	otal_cost_notional, vg_execution_cost_bps, and cost_drag.

Known limitations (explicitly accepted in v2): no intraday timing inside
the bar (fills at the next open only), no partial fills, no borrow cost or
call-loan modeling for shorts, no exchange fees/rebates or taxes, open-gap
risk between decision and execution is borne by the strategy, and the
close series is the provider's split-adjusted feed (dividends are not
reinvested). Any of these becoming load-bearing requires a new cost-table
version.

## Source Reliability and Hard Gates

The governance layer enforces a fail-closed quality model.

- A source is invalid if its timestamp is after the requested `as_of`.
- A source is invalid if freshness or completeness falls below the domain floor.
- A source cannot be used for a decision if confidence is below the required threshold.
- Missing, stale, or contradictory critical sources force `ANALYSIS_ONLY` or `NO_TRADE`.

The system records source failures with the reason, the source ID, the timestamp violation,
and the applied fallback or rejection outcome. This is not a soft warning: it is a decision gate.

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

## Monitoring Snapshot (V7)

Every completed backtest run stamps a versioned monitoring block
(`core.monitoring`, `monitoring-v1`) computed by pure functions over the
run's own decisions — no dashboard dependency — and persists it
append-only next to the manifest at `data/monitoring.jsonl`
(`monitoring_store_path` override; idempotent per run hash).

- **Score-distribution drift (PSI):** population stability index between a
  trailing reference window and the current window, fixed 0-10 bins;
  `< 0.1` no significant change, `0.1-0.25` moderate, `> 0.25` significant.
- **Confidence drift:** mean-confidence delta with the same discipline.
- **Stale-data rate:** fraction of decisions with a `STALE` agent status.
- **Veto rate by rule:** per-rule veto fractions over decisions carrying
  veto metadata.
- **Label coverage:** fraction of decisions with a matured 20d label — the
  outcome loop's health signal.

Insufficient observations degrade explicitly (`insufficient_data`), and a
decision without metadata is counted as unknown, never as clean — a run
without evidence says so instead of claiming health.

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