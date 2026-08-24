# Documentation Index

This folder contains the initial architecture and planning documents for the scoring platform.

## Core planning

- [architecture.md](architecture.md) — system boundaries, data flow, and governance
- [data-architecture.md](data-architecture.md) — raw, normalized, feature, prediction, and learning layers
- [database-schema.md](database-schema.md) — schema design for entities, score snapshots, and source registry
- [multi-agent-system.md](multi-agent-system.md) — the ten-agent design and final score flow
- [feature-engineering.md](feature-engineering.md) — data-to-feature conversion and point-in-time rules
- [backtesting-engine.md](backtesting-engine.md) — validation and walk-forward testing framework
- [paper-trading-engine.md](paper-trading-engine.md) — simulation workflow before any live capital
- [monitoring-dashboard.md](monitoring-dashboard.md) — dashboard and drift monitoring design
- [roadmap.md](roadmap.md) — phased delivery plan

## Project folders

- [../agents](../agents) — agent contracts and runtime modules
- [../api](../api) — future scoring API layer
- [../core](../core) — shared config, contracts, and orchestration logic
- [../dashboard](../dashboard) — monitoring and explainability UI
- [../models](../models) — model registry and ensemble logic

## Design principles

- Point-in-time correctness is mandatory.
- Risk and auditor vetoes override all model output.
- Never rely on a single model or a single source.
- Favor robustness and capital preservation over trade frequency.
- Every prediction must be reproducible and explainable.
