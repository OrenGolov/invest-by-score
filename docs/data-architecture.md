# Data Architecture

## 1. Purpose

The system must score a ticker at an exact timestamp using only information available at or before that timestamp. This is the foundation of the entire project.

## 2. Core design principles

- Point-in-time correctness is mandatory.
- All raw data is append-only and never overwritten.
- Every record carries source, publication time, and ingestion time.
- Missing data is not guessed; it is flagged as unavailable.
- Every prediction stores evidence and data lineage.

## 3. Data layers

### 3.1 Raw ingestion layer

This layer stores original provider payloads as received.

Sources include:
- market data
- fundamentals
- SEC filings
- earnings calls
- macro releases
- news and sentiment feeds
- social and alternative data

Each record should include:
- source_id
- source_name
- source_record_id
- payload_hash
- ingested_at
- published_at
- observed_at
- raw_json

### 3.2 Normalized data layer

This layer converts raw data into canonical entity tables.

Examples:
- symbols
- price bars
- trades
- earnings events
- news events
- macro indicators
- sentiment observations

### 3.3 Feature layer

This layer is built for a timestamped scoring request.

A feature is valid only if:
- the source data was published before or at the requested as_of timestamp
- the source has acceptable reliability
- the record is not stale
- the event is not later reversed by a corporate action or restatement

### 3.4 Prediction layer

This layer stores score snapshots, agent outputs, ensemble results, and veto states.

### 3.5 Learning layer

This layer stores:
- realized outcomes
- forecast errors
- false positives
- false negatives
- source contribution scores
- model drift signals

## 4. Data domains

### Market data
- real-time and historical prices
- OHLCV bars
- volume
- spread and liquidity
- volatility metrics
- VWAP
- ATR
- sector/market breadth
- options activity and implied volatility

### Fundamental data
- earnings and guidance
- revenue growth
- free cash flow
- margins
- balance sheet strength
- valuation ratios
- macro-driving business metrics

### Alternative data
- insider transactions
- buybacks
- corporate filings
- regulatory updates
- social chatter
- sentiment signals
- search trends
- developer activity

### Macro data
- inflation
- interest rates
- yield curve
- employment data
- GDP releases
- central bank policy

## 5. Source quality model

Every source must carry a reliability score from 0 to 1.

Score dimensions:
- freshness
- historical accuracy
- latency
- coverage
- conflict rate
- data completeness
- correction frequency

Example formula:

reliability_score = 0.35 * accuracy + 0.25 * freshness + 0.20 * coverage + 0.20 * timeliness

## 6. Recommended architecture

- Raw storage: object store or append-only database
- Structured storage: relational database / warehouse
- Time-series: TimescaleDB or similar time-series database
- Feature store: point-in-time feature cache
- Model registry: versioned tracking metadata
- Orchestration: Python services + agent runtime
- Monitoring: dashboard + alerts + drift detection

## 7. Data contracts

For every record:
- ticker
- as_of / event_time
- published_time
- observed_time
- ingested_at
- source_id
- source_record_id
- quality_score
- is_point_in_time_valid

## 8. Future expansion rule

The system must support unlimited future expansion by using standardized adapters and pluggable source connectors.

No agent should directly call a vendor API. All access must pass through a source adapter and reliability layer.
