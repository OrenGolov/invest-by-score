# Implementation Roadmap

## Phase 1: Foundation

- Define schemas, UTC timestamp policy, instrument master, and append-only lineage.
- Turn `fetch_data.py` into a provider adapter with quality reports and tests.
- Add configuration for modes, limits, horizons, and source credentials.

## Phase 2: Historical Research

- Build normalized market, fundamentals, filings, news, sentiment, and macro
  adapters with licensing and publication timestamps.
- Build point-in-time features and deterministic score snapshots.
- Implement market-data, technical, fundamental, and macro agents first.

## Phase 3: Governance and Scoring

- Add news, sentiment, regime, risk, and auditor agents.
- Add model registry, calibrated ensemble, evidence citations, and veto engine.
- Add a ticker/timestamp API and dashboard for score explanations.

## Phase 4: Validation

- Implement walk-forward backtesting, stress tests, paper portfolio, realistic
  fills, and validation reports.
- Add outcome labeling, source reliability, drift monitoring, and candidate
  retraining with promotion gates.

## Phase 5: Execution Readiness

- Implement execution only for paper mode first, with kill switches and
  idempotent order handling.
- Complete the 6-month/500-trade evidence gate. Live execution remains disabled
  unless every criterion and explicit approval exists.

## Definition of Done for the First Score

Given a ticker and timestamp, the system returns a stored, replayable score;
uses no future information; identifies missing or stale sources; cites evidence;
shows components and uncertainty; records model/data versions; and returns
`NO_TRADE` when safety conditions are not satisfied.