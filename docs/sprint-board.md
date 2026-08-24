# Sprint Board: Sprint 1 and Sprint 2

## Baseline status

This project is currently in the early technical scoring stage, not yet in the full governed multi-agent platform described in the design docs.

Current implementation present:
- [fetch_data.py](../fetch_data.py): market OHLCV fetch and Parquet cache layer
- [agents/market_data_agent.py](../agents/market_data_agent.py): market snapshot and validation logic
- [agents/technical_agent.py](../agents/technical_agent.py): technical score formula
- [core/score_engine.py](../core/score_engine.py): execution path that builds a score payload
- [core/schemas.py](../core/schemas.py): result schema
- [web_app.py](../web_app.py): local API server
- [index.html](../index.html): browser UI

Intended architecture described in the design docs:
- [docs/architecture.md](./architecture.md)
- [docs/data-model.md](./data-model.md)
- [docs/agents.md](./agents.md)
- [docs/features-and-models.md](./features-and-models.md)
- [docs/validation.md](./validation.md)
- [docs/roadmap.md](./roadmap.md)

The immediate next step is to harden the foundation and technical feature layer before moving into ML, news, sentiment, and governance.

---

## Sprint 1 — Data foundation and timestamp integrity

### Goal
Lock the system onto a strict point-in-time data contract. Do not add more reasoned intelligence until the data pipeline is proven to be correct at the requested as-of timestamp.

### File targets
- [fetch_data.py](../fetch_data.py)
- [agents/market_data_agent.py](../agents/market_data_agent.py)
- [core/score_engine.py](../core/score_engine.py)
- [docs/architecture.md](./architecture.md)
- [docs/data-model.md](./data-model.md)

### Implementation tasks
- Finalize canonical timestamp policy
  - every record must include event_time, published_time, observed_time, ingested_time, source_id, source_record_id
  - only records whose published_time <= as_of are eligible
- Define provider adapter contract
  - each provider must expose normalized records and source metadata
  - failures must be explicit and traceable
- Add source registry and source reliability model
  - provider name
  - provider type
  - coverage
  - freshness
  - failover status
  - license and usage constraints
- Upgrade data quality checks
  - stale data detection
  - duplicates
  - missing rows
  - invalid schema
  - suspicious gaps
  - impossible values
- Add audit trail for every fetch and every derived feature
- Add deterministic snapshot generation for ticker + as_of requests
- Add explicit as-of filtering to the pipeline before scoring
- Ensure the market snapshot agent does not silently use future or stale data

### Agent and model layer tasks
- Market Data Agent responsibilities
  - validate freshness and completeness
  - attach source metadata
  - emit data quality flags
  - fail closed when critical data is missing
- Technical Agent responsibilities
  - consume only clean, eligible, time-safe market inputs
  - never infer missing values silently
  - surface quality warnings to the caller

### Acceptance criteria
- No future data can be used for a past as-of timestamp.
- Every score can be tied to a valid source record and timestamp.
- Missing or stale critical data degrades confidence or produces ANALYSIS_ONLY.
- Re-running the same ticker + as_of request yields the same normalized feature state.
- Data quality issues are visible in the final API output.

### Definition of done for sprint 1
- timestamp validity is enforced in code and tested
- source lineage exists for all records
- data-quality issues are explicit, not hidden
- score generation depends on validated point-in-time data only

---

## Sprint 2 — Market data and technical feature engine

### Goal
Build the first real quantitative layer: raw price history becomes a disciplined feature set and a technical signal engine.

### File targets
- [fetch_data.py](../fetch_data.py)
- [agents/market_data_agent.py](../agents/market_data_agent.py)
- [agents/technical_agent.py](../agents/technical_agent.py)
- [core/score_engine.py](../core/score_engine.py)
- [core/schemas.py](../core/schemas.py)
- [docs/features-and-models.md](./features-and-models.md)

### Implementation tasks
- Expand price-bar validation
  - OHLCV completeness
  - gaps and missing segments
  - liquidity quality
  - unusual bars or data anomalies
- Add core technical features
  - 1d, 5d, 20d returns
  - 50d, 100d, 150d, 200d moving average features
  - trend slope
  - relative strength versus moving-average baselines
  - RSI
  - realized volatility
  - volume participation vs 20d average
- Add market regime classification
  - bullish
  - bearish
  - range
  - risk-off / stressed regime
- Add short-term and long-term feature separation
  - current-time signal features
  - long-term structural features
- Add feature metadata
  - feature name
  - owner agent
  - lookback window
  - calculation version
  - unit and null policy
- Add explicit engineering boundaries
  - use only bars available at or before as_of
  - no data leakage from future bars or future revisions

### Agent and model layer tasks
- Market Data Agent
  - produce validated OHLCV features and quality conditions
  - compute recent volume, volume ratio, trend context, and gap flags
- Technical Agent
  - compute current-time score inputs
  - compute long-term trend inputs
  - emit technical evidence and risk flags
- Scoring engine
  - combine short-term and long-term technical views into a blended score
  - separate final score from component scores
  - expose current_time_score and long_term_score explicitly

### Acceptance criteria
- Technical features are generated from normalized, valid market data only.
- Current-time score and long-term score are separate and explainable.
- Each score component has a traceable feature and calculation version.
- Data quality problems reduce confidence rather than silently bias the score.
- The feature layer is stable across multiple tickers and multiple timestamps.

### Definition of done for sprint 2
- technical feature generation is stable and deterministic
- the score can explain current-time and long-term drivers
- raw-data quality is visible in the output
- the technical engine is still intentionally narrow but scientifically structured

---

## Recommended execution method for these sprints

### Working rule
We run each sprint in small increments and re-check actual behavior before moving on.

For each sprint:
1. Define exact files to change.
2. Implement one logical layer at a time.
3. Validate with unit tests or direct runtime checks.
4. Review whether the output is explainable and timestamp-safe.
5. Only then move to the next sprint.

### Required validation gates
- Unittest or direct API check after every major step
- confirm point-in-time behavior on a known ticker/date
- confirm no future leakage into historical snapshots
- confirm all score components are visible and explainable
- confirm data quality warnings are surfaced

---

## Immediate next implementation order

For the next working cycle, we do the following in this exact order:

1. Sprint 1 hardening
   - timestamp integrity
   - source registry and quality checks
   - snapshot reproducibility
2. Sprint 2 technical feature expansion
   - current-time and long-term score separation
   - technical feature metadata and output breakdown
3. Stop and review
   - verify the technical model remains explainable and data-safe
   - decide if the next layer should be fundamentals or governance

This avoids prematurely building news, sentiment, or ML layers on top of a weak or unvalidated data foundation.
