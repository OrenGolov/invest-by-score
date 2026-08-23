CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE instruments (
    instrument_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    currency CHAR(3) NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    UNIQUE (symbol, exchange, valid_from)
);

CREATE TABLE data_sources (
    source_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    vendor_score NUMERIC(5,4),
    license_ref TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE raw_records (
    raw_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES data_sources,
    source_record_id TEXT NOT NULL,
    event_time TIMESTAMPTZ,
    published_time TIMESTAMPTZ,
    observed_time TIMESTAMPTZ,
    ingested_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_uri TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    UNIQUE (source_id, source_record_id, payload_sha256)
);

CREATE TABLE quality_reports (
    quality_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_record_id UUID NOT NULL REFERENCES raw_records,
    freshness_score NUMERIC(5,4) NOT NULL,
    completeness_score NUMERIC(5,4) NOT NULL,
    validity_score NUMERIC(5,4) NOT NULL,
    point_in_time_eligible BOOLEAN NOT NULL,
    issues JSONB NOT NULL DEFAULT '{}'::jsonb,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_versions (
    model_version TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    artifact_uri TEXT,
    feature_set_version TEXT NOT NULL,
    training_data_cutoff TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'approved', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE price_bars (
    instrument_id UUID NOT NULL REFERENCES instruments,
    bar_time TIMESTAMPTZ NOT NULL,
    interval TEXT NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC,
    currency CHAR(3) NOT NULL,
    source_id UUID REFERENCES data_sources,
    data_version TEXT NOT NULL,
    PRIMARY KEY (instrument_id, bar_time, interval, data_version)
);
SELECT create_hypertable('price_bars', by_range('bar_time'), if_not_exists => TRUE);

CREATE TABLE features (
    instrument_id UUID NOT NULL REFERENCES instruments,
    as_of TIMESTAMPTZ NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value DOUBLE PRECISION,
    missing BOOLEAN NOT NULL DEFAULT FALSE,
    feature_set_version TEXT NOT NULL,
    source_cutoff TIMESTAMPTZ NOT NULL,
    input_hash TEXT NOT NULL,
    PRIMARY KEY (instrument_id, as_of, feature_name, feature_set_version)
);
SELECT create_hypertable('features', by_range('as_of'), if_not_exists => TRUE);
ALTER TABLE features ADD CONSTRAINT features_no_lookahead CHECK (source_cutoff <= as_of);

CREATE TABLE predictions (
    prediction_id UUID NOT NULL DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES instruments,
    as_of TIMESTAMPTZ NOT NULL,
    score NUMERIC(5,2) NOT NULL CHECK (score BETWEEN 0 AND 10),
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    action_state TEXT NOT NULL,
    ensemble_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    rationale JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (prediction_id, as_of)
);
SELECT create_hypertable('predictions', by_range('as_of'), if_not_exists => TRUE);

CREATE TABLE agent_outputs (
    prediction_id UUID NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    agent_name TEXT NOT NULL,
    model_version TEXT REFERENCES model_versions,
    status TEXT NOT NULL,
    output JSONB NOT NULL,
    veto BOOLEAN NOT NULL DEFAULT FALSE,
    rationale JSONB NOT NULL,
    PRIMARY KEY (prediction_id, as_of, agent_name)
);
SELECT create_hypertable('agent_outputs', by_range('as_of'), if_not_exists => TRUE);

CREATE TABLE portfolios (
    portfolio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_ref TEXT NOT NULL,
    base_currency CHAR(3) NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('analysis', 'paper', 'live_disabled', 'live_approved'))
);

CREATE TABLE positions (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios,
    instrument_id UUID NOT NULL REFERENCES instruments,
    quantity NUMERIC NOT NULL,
    average_cost NUMERIC NOT NULL,
    currency CHAR(3) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    UNIQUE (portfolio_id, instrument_id, as_of)
);

CREATE TABLE fx_events (
    fx_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    base_currency CHAR(3) NOT NULL,
    quote_currency CHAR(3) NOT NULL,
    rate NUMERIC NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    source_id UUID REFERENCES data_sources,
    source_record_id TEXT NOT NULL
);

CREATE TABLE trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios,
    instrument_id UUID NOT NULL REFERENCES instruments,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC NOT NULL,
    price NUMERIC,
    currency CHAR(3) NOT NULL,
    fx_event_id UUID REFERENCES fx_events,
    prediction_id UUID,
    decision_time TIMESTAMPTZ NOT NULL,
    fill_time TIMESTAMPTZ,
    status TEXT NOT NULL,
    paper_only BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE tax_lots (
    tax_lot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios,
    instrument_id UUID NOT NULL REFERENCES instruments,
    opening_trade_id UUID REFERENCES trades,
    closing_trade_id UUID REFERENCES trades,
    quantity NUMERIC NOT NULL,
    cost_basis NUMERIC NOT NULL,
    basis_currency CHAR(3) NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL,
    disposed_at TIMESTAMPTZ,
    holding_period TEXT,
    wash_sale_flag BOOLEAN NOT NULL DEFAULT FALSE,
    jurisdiction TEXT NOT NULL
);

CREATE TABLE audit_events (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID,
    portfolio_id UUID REFERENCES portfolios,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL UNIQUE
);