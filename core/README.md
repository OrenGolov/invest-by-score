# Core

Core system components for orchestration, config, and scoring contracts.

## Contents

- config for mode limits and source settings
- timestamp, symbol, and evidence contracts
- scoring result schema
- risk gate definitions
- feature store contracts

## Principles

- Point-in-time correctness is mandatory.
- Use UTC internally and retain publication timestamps.
- Do not infer values for missing source data.
- Every output must include source traceability.
