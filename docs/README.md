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

## Operating

- **Signup-Guide.docx** — the three accounts/keys only the operator can create,
  step by step, with which network each one works from
- **Quick-Reference.docx** — one page: scheduled jobs, how to run each one by
  hand, and what the common messages mean
- [operator-runbook.md](operator-runbook.md) — what to set up, what to check
  daily, and how to run any job on demand instead of waiting for the scheduler

## Incident reviews

- [news-quota-review-2026-10-06.md](news-quota-review-2026-10-06.md) — why news
  capture stopped for three days, the four compounding defects behind it, and
  the quota model that now holds the arithmetic

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
