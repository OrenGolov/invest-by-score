# Sprint 3: Scoring App Requirements

## Goal

Build the first real scoring application workflow for a ticker at an exact timestamp.

## Scope

- Input: ticker and exact date/timestamp
- Output: score out of 10, confidence, explanation, risk notes, action state
- Use a multi-agent architecture with clear responsibilities
- Keep the scoring logic explainable and deterministic
- Prioritize false-positive minimization and risk control

## Required agent roles

1. Market Data Agent
2. Technical Analysis Agent
3. Fundamental Analysis Agent
4. News Intelligence Agent
5. Sentiment Agent
6. Macroeconomic Agent
7. Market Regime Agent
8. Risk Management Agent
9. Performance Auditor Agent
10. Execution Agent

## Design rules

- No single agent should decide the trade alone.
- Risk and performance auditor vetoes are mandatory.
- Live execution remains disabled until validation requirements are passed.
- Favor no-trade over poor trade.
- Every score must be point-in-time and explainable.

## High-priority requirements to carry forward

- exact ticker + exact timestamp scoring
- weighted ensemble of multiple model families
- source reliability scoring
- continuous learning from forecast outcomes
- false-positive minimization
- full backtesting and paper-trading gate before live deployment
- portfolio risk allocation limits

## Implementation path

- Build a scoring contract and API layer
- Add market and technical agent logic
- Add risk and audit gate framework
- Add structured output schema for score explanations
- Prepare for future news, sentiment, macro, and regime agents

## Git rule

This sprint must be committed and pushed as a separate branch named sprint-3.
