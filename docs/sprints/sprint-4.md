# Sprint 4: Fundamental Data and Valuation Layer

## Goal

Turn the scoring platform from a technical-only engine into an evidence-led system that can include real valuation and company-quality signals without violating point-in-time rules.

## Scope

- Add a real provider adapter for valuation and fundamentals
- Require an explicit timestamp check before any value is used in a score
- Expose valuation metrics, cash-flow quality, and balance-sheet quality in structured output
- Keep the layer in analysis-only mode until quality, confidence, and source checks pass
- Prepare the system for the next news/sentiment/regime merge

## Required outcomes

- Live or env-configured provider fetch for company overview and valuation fields
- Timestamp guard that rejects future-dated provider responses
- Contract test that validates the fundamental snapshot schema and point-in-time safety
- Structured output for `fundamental_score` and `fundamental_features`
- Clear source status and no-action behavior when a provider key is missing

## Implementation notes

- Prefer provider-backed fetching with explicit env configuration for a real API key
- Maintain deterministic caching for repeated runs
- Keep all score logic explainable and safe by design
- Do not enable execution decisions from this layer alone

## Definition of done

The sprint is complete when a ticket can request a ticker and as-of timestamp, receive a fundamental snapshot with clear source metadata, and the system can prove that no future-dated values are used in the score.
