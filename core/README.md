# Core

Core system components for orchestration, config, and scoring contracts.

## Contents

- config for mode limits and source settings
- timestamp, symbol, and evidence contracts
- scoring result schema
- risk gate definitions
- feature store contracts
- news adapter (N1): PIT-filtered ingestion, event classification, source
  quality weighting, contradiction detection, and evidence-backed aggregation
  behind `core.news_contract.fetch_news_snapshot`

## Principles

- Point-in-time correctness is mandatory.
- Use UTC internally and retain publication timestamps.
- Do not infer values for missing source data.
- Every output must include source traceability.
