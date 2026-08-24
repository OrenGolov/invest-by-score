# Database Schema

## 1. Design goals

- Append-only record history
- Point-in-time alignment
- Clear traceability for every score
- Safe support for replay and audit

## 2. Core tables

### symbols

| Column | Type | Purpose |
| --- | --- | --- |
| symbol_id | UUID | unique symbol identifier |
| ticker | VARCHAR | stock symbol |
| exchange | VARCHAR | exchange code |
| isin | VARCHAR | optional global identifier |
| active_from | TIMESTAMP | start of activity |
| active_to | TIMESTAMP | end of activity |
| metadata | JSONB | extra metadata |

### price_bars

| Column | Type | Purpose |
| --- | --- | --- |
| price_bar_id | UUID | unique id |
| symbol_id | UUID | foreign key |
| event_time | TIMESTAMP | bar timestamp |
| published_time | TIMESTAMP | when published |
| open | FLOAT | OHLCV open |
| high | FLOAT | OHLCV high |
| low | FLOAT | OHLCV low |
| close | FLOAT | OHLCV close |
| volume | BIGINT | traded volume |
| source_id | UUID | data source |
| quality_score | FLOAT | source quality |

### fundamentals

| Column | Type | Purpose |
| --- | --- | --- |
| fundamental_id | UUID | unique id |
| symbol_id | UUID | symbol |
| as_of_date | DATE | reporting date |
| revenue | FLOAT | revenue |
| operating_margin | FLOAT | margin |
| net_income | FLOAT | earnings |
| free_cash_flow | FLOAT | cash generation |
| debt_to_equity | FLOAT | leverage |
| pe_ratio | FLOAT | valuation |
| source_id | UUID | source |

### macro_indicators

| Column | Type | Purpose |
| --- | --- | --- |
| macro_id | UUID | unique id |
| name | VARCHAR | indicator name |
| region | VARCHAR | geography |
| value | FLOAT | indicator value |
| event_time | TIMESTAMP | issue time |
| published_time | TIMESTAMP | publication time |
| source_id | UUID | source |

### news_events

| Column | Type | Purpose |
| --- | --- | --- |
| news_id | UUID | unique id |
| symbol_id | UUID | related symbol |
| title | TEXT | headline |
| summary | TEXT | summary |
| published_time | TIMESTAMP | publication time |
| sentiment_score | FLOAT | sentiment value |
| impact_score | FLOAT | estimated impact |
| source_id | UUID | source |

### sentiment_records

| Column | Type | Purpose |
| --- | --- | --- |
| sentiment_id | UUID | unique id |
| symbol_id | UUID | symbol |
| event_time | TIMESTAMP | observation time |
| score | FLOAT | sentiment score |
| trend | FLOAT | sentiment trend |
| source_id | UUID | provider |

### score_snapshots

| Column | Type | Purpose |
| --- | --- | --- |
| snapshot_id | UUID | unique score id |
| symbol_id | UUID | symbol |
| as_of | TIMESTAMP | scoring timestamp |
| overall_score | FLOAT | final score out of 10 |
| confidence | FLOAT | model confidence |
| action_state | VARCHAR | ANALYSIS_ONLY / NO_TRADE / PAPER |
| vetoed | BOOLEAN | whether vetoed |
| evidence_json | JSONB | reasoning and evidence |
| model_version | VARCHAR | model registry version |
| created_at | TIMESTAMP | creation time |

### agent_outputs

| Column | Type | Purpose |
| --- | --- | --- |
| agent_output_id | UUID | unique output id |
| snapshot_id | UUID | score snapshot |
| agent_name | VARCHAR | agent name |
| output_json | JSONB | agent output |
| status | VARCHAR | OK / UNAVAILABLE / VETO |
| evidence_refs | JSONB | linked evidence |

### model_registry

| Column | Type | Purpose |
| --- | --- | --- |
| model_id | UUID | unique model id |
| family | VARCHAR | xgboost / random_forest / transformer |
| version | VARCHAR | version string |
| active_from | TIMESTAMP | activation |
| active_to | TIMESTAMP | deactivation |
| metrics_json | JSONB | validation stats |

### source_registry

| Column | Type | Purpose |
| --- | --- | --- |
| source_id | UUID | unique source id |
| source_name | VARCHAR | provider name |
| domain | VARCHAR | market/fundamental/news/macro |
| reliability_score | FLOAT | 0-1 score |
| status | VARCHAR | active / deprecated |
| api_limits_json | JSONB | rate limit metadata |

## 3. Non-negotiable rules

- No table should mutate raw data.
- All time-based joins must use publication or event-time constraints.
- A score snapshot must always link back to the exact evidence used.
- A final decision is never recorded without a valid audit trail.
