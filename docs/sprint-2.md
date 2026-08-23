# Sprint 2 Deliverables

Sprint 2 is design and schema only. It introduces no production ingestion,
model, broker, or execution code.

## Delivered

- Vendor scoring rubric for coverage, reliability, rate limits, cost, and latency.
- Primary and fallback domains for price, fundamentals, news, sentiment, macro,
  FX, and corporate actions, with cadence targets.
- MCP parallel-purpose decision: read-only research tools behind the provider
  boundary; deterministic ingestion and governance remain authoritative.
- Mermaid data-flow, ER, and decision-cycle diagrams.
- TimescaleDB migration for OHLCV, features, predictions, and agent outputs.
- Relational records for portfolios, positions, trades, FX events, tax lots,
  model versions, quality, lineage, and immutable audit events.
- Feature registry ownership and point-in-time enforcement rules.
- Shared backtest contract, walk-forward/embargo policy, metrics, and model
  version requirements.
- Paper A/B and shadow testing, full rationale logging, and release gates.
- Monitoring and future JSON dashboard contract; execution remains disabled.

## Approval Checklist

- [ ] Vendor access, licensing, cost, and rate limits verified.
- [ ] Primary/fallback source contracts approved per domain.
- [ ] TimescaleDB migration reviewed by engineering and compliance/tax owners.
- [ ] Point-in-time tests and model registry contract approved.
- [ ] Agent roster and veto semantics approved.
- [ ] Dashboard fields and alert thresholds approved.
- [ ] Sprint 2 design approved before production implementation begins.