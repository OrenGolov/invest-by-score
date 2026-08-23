# System Architecture

## Goal

For an input such as `AAPL` and `2026-08-23T14:30:00Z`, return a reproducible
score snapshot with ticker, as-of timestamp, score from 0 to 10, confidence,
action state, component scores, evidence, risks, data quality, and model IDs.

The product is an analysis and validation platform first. Trading is a later,
separately gated capability.

## Logical Components

```text
Providers -> Ingestion -> Raw immutable store -> Normalized store
                                      |                 |
                               Quality/lineage    Feature builder
                                                        |
             Macro/news -> Agent runtime -> Ensemble -> Auditor/Risk gates
                                                        |
                         Score API/UI <- Snapshot + explanations
                                                        |
                         Outcome tracker -> Evaluation -> Retraining registry
```

- Provider adapters isolate APIs, credentials, licensing, and rate limits.
- Ingestion validates payloads, normalizes time zones and symbols, stores the
  original payload hash, and never overwrites raw records.
- Feature generation uses only records whose effective time is at or before
  `as_of`.
- Agents use typed inputs and outputs; unavailable data is explicit.
- The ensemble records calibrated, versioned model weights.
- Risk Management and Performance Auditor vetoes are fail-closed and visible.

## Point-in-Time Contract

Every record must include `event_time`, `published_time`, `observed_time`,
`ingested_time`, `source_id`, and `source_record_id`. For a prediction at
`as_of`, only records with `published_time <= as_of` are eligible. If published
time is unknown, the record is excluded from predictive features and marked
`unusable_for_point_in_time`.

Prices use UTC internally and retain exchange-local session metadata. Corporate
actions, ticker changes, delistings, and restatements are versioned rather than
mutated in place.

## Modes and Failure Policy

- `ANALYSIS_ONLY`: scoring and explanation; no orders.
- `PAPER`: simulated orders and fills only.
- `LIVE_DISABLED`: default until all gates pass.
- `LIVE_APPROVED`: explicit approval and audit trail; out of scope initially.

Stale, conflicting, incomplete, or low-quality critical data causes degraded
confidence or `NO_TRADE`. The system must never fill missing financial data
with a guessed value. All failures attach to the score snapshot.