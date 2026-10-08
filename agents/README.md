# Agents

This folder defines the multi-agent runtime for the scoring platform.

## Agent roster

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

- Agents are analysis specialists, not autonomous traders.
- Each agent emits structured output with evidence, confidence, and missing-data flags.
- No single agent can create a trade decision.
- Risk and Auditor vetoes are fail-closed.
- Execution is simulation-only until validation gates are satisfied.

## Expected interface

Each agent should return:

- agent_name
- ticker
- as_of
- status
- score
- confidence
- uncertainty
- evidence
- warnings
- model_version

## Planned implementation

- Start with deterministic feature generation and clear data contracts.
- Add each agent in a separate module.
- Connect them through a shared orchestrator.
- Keep all outputs auditable and timestamped.
