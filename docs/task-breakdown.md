# Engineering Task Breakdown — Governed Multi-Agent Scoring Platform

Audience: senior engineers executing in small, verifiable increments.
Scope: everything between the current baseline (Sprints 1–3 substantially done,
orchestration scaffolding present, governance decorative) and release
readiness. Each task states its objective, design constraints, touched files,
edge cases, and acceptance gates. Nothing here may violate the platform's
non-negotiables: point-in-time eligibility, fail-closed governance, no silent
data invention, deterministic replay.

Baseline reference (do not re-litigate, verify instead):
- PIT filtering + feature provenance: `agents/market_data_agent.py`, `core/schemas.py` (FeatureContract)
- Source registry (2 domains): `fetch_data.py::SOURCE_REGISTRY`
- Typed contracts: `core/agent_contracts.py`, `core/orchestrator.py` (6 agents)
- Deterministic evidence-weighted confidence: `core/score_engine.py::_compute_confidence` (`evidence-confidence-v2`)
- Audit store: `core/audit_store.py` (append-only JSONL + replay lookups)
- Landed data-integrity fixes: commit `4fbb92c` (pct-change anchor, weekend gap false positives, collinear MA-term de-duplication)

---

## Sprint W — Governance Wiring (immediate cycle)

Rationale: the orchestrator runs six agents but the headline score ignores
them; Risk and Auditor are passive relays. Every later sprint (news, ML,
backtest) compounds on this fault line. Close it first.

### W1. Versioned ensemble wiring — agent outputs drive the final score ✓ DONE

- Objective: make the published score the weighted product of live agent
  outputs instead of an independent hand-built blend, without losing the
  current-time / long-term decomposition the dashboard relies on.
- Design constraints:
  - `core/config.py` gains `ENSEMBLE_WEIGHTS_CURRENT` and
    `ENSEMBLE_WEIGHTS_LONG`: dicts keyed by agent name (`market_data`,
    `technical_analysis`, `fundamental_analysis`, `news_intelligence`,
    `sentiment`, `macroeconomic`, `market_regime`), plus `ENSEMBLE_VERSION`
    stamped into every decision. Both sets must sum to 1.0 — assert at import
    time (tolerance 1e-9). Agents not yet implemented hold explicit `0.0`
    weights; presence in the dict is the contract, absence is a startup error.
  - Renormalization rule: when an agent's status is not `OK`, its weight is
    redistributed proportionally across `OK` agents; if none are `OK`, the
    decision is `NO_TRADE` with reason `no_eligible_agents`. Never treat a
    missing agent as silently neutral.
  - Determinism: weights are code constants, not runtime config; no
    wall-clock or randomness in the score path (audit timestamps attach
    outside the engine, as today).
- Implementation notes:
  - Each agent contribution maps to 0–10 before weighting: technical_analysis
    → blended `(current+long)/2` from ONE canonical scorer (see W5);
    fundamental_analysis → `fundamental_score`; market_data →
    data-quality-derived informational score; future agents → contract score.
  - Extend the payload with `ensemble_breakdown`:
    per-agent `{weight, contribution, status, renormalized}` so the UI panel
    and the number can never disagree.
- Edge cases: agent score `None` → excluded from renormalization (not coerced
  to 0); the two weight sets must stay proportional or the blend loses
  meaning (assert the ratio invariant in tests).
- Files: `core/config.py`, `core/score_engine.py`, `core/orchestrator.py`,
  `core/schemas.py` (new field), tests.
- Acceptance: flipping the fundamental weight visibly moves score and
  breakdown; a failing provider renormalizes provably in a unit test; two
  identical calls produce byte-identical `to_dict()`.

### W2. Risk agent becomes a real fail-closed gate ✓ DONE

- Status: **implemented** — `core/risk_policy.py` evaluates `RISK_POLICY_V2`
  (core/config.py); the orchestrator's risk agent carries the full structured
  rule evaluation and `veto_rule_ids` replace `_build_veto_reasons`; mode
  selection is centralized in `_select_mode` with the tested invariant that
  PAPER is unreachable while any veto-severity rule fires.

- Objective: replace the relay (`status:"OK", veto_ready:True`) with a
  deterministic policy evaluation that cannot be bypassed downstream.
- Design constraints:
  - Inputs (all already persisted): `confidence_breakdown`, `governance`,
    `source_quality`, `risk_flags`, source statuses.
  - Policy table `RISK_POLICY_V2` in `core/config.py`: explicit thresholds —
    minimum confidence, maximum total penalty, forbidden statuses (`INVALID`,
    `STALE` on critical inputs), minimum data-quality score, maximum
    volatility-regime penalty. Each rule yields structured
    `{rule_id, severity: veto|warning, triggered, detail}`.
  - Fail-closed semantics: missing/`None` inputs evaluate to triggered veto
    rules, never to pass; policy version stamped into the payload.
- Implementation notes:
  - Move `_build_veto_reasons` threshold literals (quality < 60, source <
    0.7, score < 5.5, analysis-only) into the policy table — one place
    governs gates; the orchestrator consumes rule results instead of
    duplicating comparisons.
  - `risk_agent.payload` carries the full rule evaluation; `veto_reasons`
    become structured (`rule_id` + human text) while keeping the existing
    string keys so `index.html` keeps rendering.
- Edge cases: confidence exactly at floor; only warning-severity rules
  triggered; governance state contradicting policy state.
- Acceptance: table-driven unit tests prove each rule triggers and —
  critically — that no input combination can produce `mode == "PAPER"` while
  any `veto`-severity rule is triggered.

### W3. Auditor agent becomes an evidence-and-replay validator ✓ DONE

- Status: **implemented** — `core/audit_policy.py` (owning canonical hashing)
  evaluates seven checks: evidence sufficiency, agent input-hash integrity,
  snapshot-hash recomputation, a determinism probe (second in-process build
  via `build_score(persist_audit=False)`), calibration sanity, ensemble
  consistency, and evidence-ledger consistency (warning). A failed
  veto-severity check appends the `auditor_veto` reason, blocking PAPER
  through the `_select_mode` invariant.

- Objective: the auditor independently verifies that a decision is provable
  and may veto on audit failure (its reserved severity), per the design docs.
- Checks (each returns pass/fail + detail, assembled into `payload.findings`):
  - Evidence sufficiency: every `OK` agent carries non-empty `evidence` with
    a `source_record_id`; evidence-ledger status consistent with governance.
  - Hash integrity: each agent's `input_hash` recomputes from its payload;
    `snapshot_hash` stable across a second in-process build.
  - Determinism probe: run `build_score` twice on identical inputs and
    compare `to_dict()` modulo audit-only fields; mismatch → veto.
  - Calibration sanity: confidence within `[FLOOR, CAP]`; breakdown value
    agrees with the headline within 0.005.
- Acceptance: tampering a payload or hash flips the decision to `NO_TRADE`
  with an `auditor_veto` reason; the happy path adds no network or I/O.

### W4. Failure-state taxonomy across every contract ✓ DONE

- Status: **implemented** — `AgentStatus` enum (OK/UNAVAILABLE/STALE/
  INCOMPLETE/CONTRADICTORY/INVALID + VETO for governance agents) in
  `core/schemas.py` with `STATUS_POSTURE` propagation and `worst_status`
  combiner; `AgentContract.__post_init__` rejects unknown statuses;
  `_derive_agent_statuses` (orchestrator) maps raw signals onto the
  taxonomy reusing risk-policy thresholds; worst data-agent posture
  propagates (`INVALID/CONTRADICTORY → agent_status_no_trade veto`,
  `STALE/INCOMPLETE → ANALYSIS_ONLY floor`).

- Objective: one vocabulary for degraded data, replacing ad-hoc strings.
- Design: `AgentStatus` (str-Enum) in `core/schemas.py`:
  `OK, UNAVAILABLE, STALE, INCOMPLETE, CONTRADICTORY, INVALID`.
  `AgentContract.status` stays a plain string but must validate against the
  enum at construction (raise on unknown) — JSON shape unchanged.
- Mapping rules (documented in the enum docstring):
  - `STALE`: freshness factor below threshold (surface from
    `confidence_breakdown.factors[freshness]`) or cache older than TTL on read.
  - `INVALID`: `timestamp_valid == False`, future-dated payload, schema violation.
  - `INCOMPLETE`: data-quality below the governance threshold or missing
    critical fields (close, valuation_metrics empty).
  - `CONTRADICTORY`: reserved until ≥2 same-domain sources exist and disagree
    beyond tolerance (lands with Sprint N adapters).
  - `UNAVAILABLE`: provider unconfigured or empty response (news today).
- Propagation: `orchestrate_score` derives posture from the worst agent
  status — `INVALID`/`CONTRADICTORY` on critical agents → `NO_TRADE`;
  `STALE`/`INCOMPLETE` → `ANALYSIS_ONLY` floor. The policy table (W2) consumes
  statuses instead of raw float comparisons.
- Acceptance: table-driven tests map each snapshot corruption to the intended
  status, and each status to the intended decision posture.

### W5. Single technical truth ✓ DONE

- Status: **implemented** — `agents/technical_agent.py` is now a thin adapter
  over the canonical scorers (`_score_current_time`/`_score_long_term`);
  the independent formula is deleted, as is `score_engine`'s dead import of
  it (which had created a circular-import hazard). The orchestrator passes
  the decision's real news snapshot so the agent view matches the ensemble's
  embedded-news view. Invariant test: agent score equals the rounded blend
  of the raw views within the 2-dp rounding envelope.

- Objective: eliminate the split-brain where `agents/technical_agent.py::
  score_technical` feeds the technical AgentContract while the headline score
  comes from `_score_current_time`/`_score_long_term` with different
  coefficients.
- Decision: `score_technical` becomes a thin documented wrapper returning the
  blended technical view `(current + long) / 2` computed by the shared
  functions in `core/score_engine.py`; its independent formula is deleted.
  The disjoint-feature contract (current never reads long-horizon inputs and
  vice versa) is preserved and stays test-enforced.
- Consequence for W1: the technical agent's contribution equals a component
  of the headline by construction — the ensemble becomes coherent.
- Acceptance: separation tests still pass; a new invariant test asserts
  `|agent_technical.score − blend(current, long)| < 1e-9`.

### W6. Append-only raw-record store (Sprint-1 leftover, minimal viable) ✓ DONE

- Status: **implemented** — `core/raw_store.py`: `append_raw_records` (one
  JSONL line per fetch under `data/raw/{source_id}/{YYYY-MM-DD}.jsonl` with
  `ingested_time`/`payload_sha256`/`schema_version`, fail-soft via the
  `raw_store_write_failed` warning), `load_raw_records` (request-key
  filtering, supersede-on-refetch marking older versions with
  `superseded_by`), and `rebuild_price_frame` (latest-version OHLCV frame
  reconstruction). Wired into `fetch_price_history` (bars, before cache
  write) and both `fetch_fundamental_snapshot` return paths (full snapshot
  payload). Acceptance proven: deleting the VOO parquet cache and
  rebuilding from raw yields a value- and index-identical frame; the MSFT
  rebuild-from-raw test reproduces the full snapshot feature-for-feature.

- Objective: satisfy raw immutability / replayability without a database —
  no fetch may leave the system unable to rebuild what it saw.
- Design:
  - Path scheme `data/raw/{source_id}/{YYYY-MM-DD}.jsonl`; one JSON object
    per line: `{ingested_time, source_id, request_key, payload_sha256,
    schema_version, records:[…]}` — records are normalized bars
    (already point-in-time filtered) or raw provider fields (fundamentals).
  - Writes append-only; re-fetching the same bar appends a new version line
    (dedupe key `{request_key, bar_time}`; readers keep both and mark
    `superseded_by` — mirrors the data-model doc's versioning rule).
  - Reader helper `load_raw_records(source_id, request_key, as_of)` rebuilds
    input state for audit replays.
- Constraints: the store write lives inside the adapter before the cache
  write, wrapped so failures log a `raw_store_write_failed` quality warning
  and never corrupt scoring (degraded-but-flagged is acceptable).
- Acceptance: delete the Parquet cache, rebuild a historical snapshot purely
  from `data/raw` for a past as_of, assert feature equality with the original
  run within float tolerance.

### W7. Audit event enrichment ✓ DONE

- Status: **implemented** — events are now `audit-event-v3` (v2 + the M2
  `model_resolutions` block): `schema_version`,
  `ensemble_version`, `model_versions` (agent → version), `agent_statuses`,
  `veto` (risk rule ids + auditor check ids), and
  `confidence_breakdown_digest` (hash, not the bulk payload). The enriched
  write moved to the orchestrator (where statuses/versions/vetoes are all
  known); the determinism probe uses `persist_audit=False` so a decision
  still produces exactly one event. `get_events_since(cursor)` added for the
  timeline API; append-only log with no backfill migration.

- `persist_decision_audit` gains: `schema_version`, `ensemble_version`,
  `model_versions` (agent → version), `veto` object (agent, rule_ids,
  severity), `agent_statuses`, and a `confidence_breakdown_digest` (hash, not
  the bulk payload).
- Keep the log append-only; add `get_events_since(cursor)` for the upcoming
  timeline API. No backfill migration — old lines stay readable via `.get`
  defaults.

Sprint W exit criteria: `python main.py MSFT 2026-08-26` twice → identical
output; tamper any payload → `NO_TRADE` with a named rule; weights visible
and effective in `/api/score`; raw-store rebuild test green; suite ≥ 60
hermetic tests.

---

## Sprint N — News, Sentiment, Macro, and Regime (context layer)

Governing rule inherited from the design docs: every new agent is born wired —
it enters `ENSEMBLE_WEIGHTS` with a nonzero weight and has a veto-path test on
the day it lands. An agent that only decorates the payload is a defect.

### N1. News intelligence agent (real ingestion + classification) ✓ DONE

Implemented 2026-09-01 (commit 5dc7099). Full pipeline end-to-end, all
acceptance gates verified.

- Adapter: `core/news_adapter.py` (~660 lines) behind existing contract
  (`core/news_contract.py`; `fetch_news_snapshot` signature unchanged).
  Provider resolution via `NEWS_PROVIDER_API_KEY_ENV`; no-key path remains
  byte-for-byte UNAVAILABLE legacy stub (tested).
- Query window: PIT filter gates articles by `published_time <= as_of`; 
  window 7d lookback (config). Future-dated or unparseable → rejected INVALID
  (fail-closed); PIT policy enforced before any scoring.
- Per-article record: `{source_id, source_record_id, published_time, headline,
  url, category, tone, tone_derivation, relevance, source_weight,
  included_in_aggregation, exclusion_reason}`.
- Classifier v2 (curated pattern sets, `NEWS_CLASSIFIER_VERSION`; v2 extended
  litigation and regulation coverage — derivative/countersuit phrasing, SEC
  reporting/rules/requirements): earnings,
  guidance, litigation, regulation, product_launch, macro_shock, m_and_a,
  strategic_announcement, management_commentary, other. First match wins;
  taxonomy defined in `NEWS_CATEGORY_PATTERNS` as data constant.
- Tone v1: provider-supplied when in [−1, 1], else lexicon fallback
  (`NEWS_TONE_LEXICON_VERSION`). Lexicon: curated positive/negative terms
  with negation handling (negator within 3 tokens flips sign). Derivation
  stamped per record (`provider` or `lexicon:v1`).
- Relevance: ticker exact match (1.0), company name token subset (0.7),
  off-entity (0.0). Relevance 0 stays in evidence, excluded from aggregation
  (audit trail via `exclusion_reason`).
- Source weighting: `base_confidence × source_quality × recency_decay`
  (exponential, half-life 3d). Zero-quality sources cannot raise confidence
  and are fail-closed from contradiction aggregation.
- Contradiction v1: same-day, same-category cluster with opposite-sign mean
  tone, |Δ| > 0.6 → status `CONTRADICTORY`, both positive/negative sides
  surfaced. Contradictory never averages to neutral; confidence floored to
  0.10. Only credible articles (non-zero weight + relevance) enter clustering.
- Novelty: duplicate headlines (normalized) deduplicated; first occurrence
  wins (reproducible via sorted order). Duplicates stay in evidence with
  `exclusion_reason: duplicate_headline`.
- Aggregation: sentiment = weighted mean tone (weight = relevance × source_weight);
  confidence = f(count/5, mean_weight, 1−dispersion, weakest_source); capped
  at weakest contributor. Empty → `UNAVAILABLE` or `INCOMPLETE` per status.
- Raw immutability: `append_raw_records()` logs all fetches under
  `data/raw/newsapi_news/{YYYY-MM-DD}.jsonl` (one JSONL line per request key).
- Status contract: UNAVAILABLE (no key, fetch failed, no articles), OK
  (credible aggregation), CONTRADICTORY (opposite-sign cluster), INVALID
  (future-dated rejection), INCOMPLETE (no credible articles).
- Acceptance: no-key contract byte-for-byte + tests pass; zero-quality source
  cannot raise confidence (test suite); contradictory cluster yields status
  with both sides + confidence floor (test suite); as_of filtering proven
  (future-dated test). Smoke: 6/6 checks pass.

### N2. Sentiment agent (social/positioning, distinct from news tone) ✓ DONE

- Scope per design docs: retail/institutional positioning and social signals —
  not a re-broadcast of news tone. Until a real provider lands, ship
  `UNAVAILABLE` with a typed placeholder documenting intended inputs
  (mention volume, tone trend, disagreement, manipulation flags).
- Any derivation from news tone must be labeled `derived_from_news` in the
  payload with reduced confidence (×0.5); silent proxying is prohibited by
  the news contract's docstring rule.
- Acceptance: a contract test pins the UNAVAILABLE shape so a future provider
  cannot silently change the public schema.
- Status: **implemented** — `core/sentiment_contract.py::fetch_sentiment_snapshot`
  (signature accepts ONLY `ticker, as_of`, so silent proxying is impossible by
  construction) over the `SentimentSnapshot` schema (`derivation: "none"`,
  `intended_inputs`, anti-proxy reason). `build_score` fetches it, hashes it
  into the replay metadata, and emits a zero-weight `sentiment` ensemble line;
  `orchestrate_score` carries it as the 7th agent (audit event
  `model_versions`/`agent_statuses` included, `expected_input_hashes`
  verified) while deliberately sitting OUTSIDE the data-agent posture loop —
  its absence renormalizes to nothing and cannot floor a healthy decision to
  `ANALYSIS_ONLY`. The `derived_from_news` labeling is now an enforced,
  test-pinned contract: `core/config.py::SENTIMENT_DERIVED_CONFIDENCE_SCALE`
  (0.5) and `core/sentiment_contract.py::derive_sentiment_from_news` — the
  ONLY sanctioned path to news-derived sentiment — is fail-closed (any news
  snapshot that is not status OK with in-range numeric tone/confidence yields
  the UNAVAILABLE placeholder; degraded or CONTRADICTORY evidence is never
  laundered into a sentiment value), stamps
  `derivation: "derived_from_news"` + `source_id: "sentiment_derived_from_news"`,
  and scales `source_confidence` ×0.5 so the single-source dependency is
  priced into confidence, not merely documented. It is deliberately NOT wired
  into the production pipeline; promoting it requires a feature-registry entry
  and an explicit ensemble-weight decision. Anti-proxying is proven three ways
  in `tests/test_sentiment_contract.py`: the pinned UNAVAILABLE shape,
  behavioral invariance (output identical under wildly different patched
  news/technical pipeline states), and an AST import-hygiene guard (the module
  may import only config + schemas; importing news/technical/market code fails
  the suite). A source-scan test also forbids production references to the
  derived path until it is formally registered.
  `tests/test_sentiment_contract.py` pins the shape byte-for-byte.

### N3. Macroeconomic agent ✓ DONE

Implemented 2026-09-01. Full vintage-aware series registry, PIT filtering,
risk regime classification, and sector sensitivity mapping complete.
Vintage-aware v2 (publication-time vintages) completed 2026-09-15.

- Series registry: `core/macro_registry.py` (data constant `MacroSeries` entries
  for fed_funds, cpi_yoy, initial_claims, gdp_growth, 10y_yield). Each series
  carries complete metadata: provider (FRED), series_id, unit, frequency,
  transformation, publication_lag_days, reference_period_field,
  published_time_field, feature_version, lookback_periods, description,
  pit_policy (`published_time_gate_first_release_retained`) and vintage_source
  (`fred_alfred_realtime` for all five series).
- Adapter: `core/macro_adapter.py` (~500 lines) implementing full pipeline:
  FETCH → PROVIDER GATE → PIT FILTER → REVISION HANDLING → INDICATOR ANALYSIS
  → SECTOR-WEIGHTED REGIME → CONFIDENCE CALCULATION.
- PIT policy: articles/releases with published_time > as_of are rejected
  (INVALID, fail-closed). Eligible records are PIT-filtered before analysis.
  Revisions append to raw_store (W6) with request_key-based versioning.
- Sector sensitivity v1: `core/macro_registry.py::SECTOR_MACRO_LOADINGS` maps
  GICS sectors to (rates_sensitivity, energy_beta, usd_beta) loadings (static,
  curated, `MACRO_SENSITIVITY_VERSION`). Symbol→sector mapping table
  (`SYMBOL_TO_SECTOR`) v1 for portfolio holdings; extensible later via
  classification service.
- Risk regime classification: compute_risk_regime() evaluates fed_funds (rates
  level: >4% restrictive, <2% accommodative), cpi_yoy (inflation: >3.5% high,
  <1.5% subdued), initial_claims (labor: >450k elevated, <200k tight),
  gdp_growth (growth: <1% weak, >3% strong), 10y_yield (yields: >3.5%
  elevated, <1.5% depressed). Mean signal → risk_score ∈ [0, 1] (0 = risk-off,
  0.5 = neutral, 1.0 = risk-on). Regime: risk_off if score < 0.3, risk_on if
  score > 0.7, neutral otherwise.
- Status contract: UNAVAILABLE (no provider key), OK (all series available),
  INCOMPLETE (missing series; confidence degraded by MACRO_MISSING_SERIES_PENALTY
  per series), INVALID (future-dated or unparseable publication times).
- Per-series contributions: each series carries source_record_ids, published_time,
  value, credibility status, and a `vintage` block (source, reference_date,
  vintage_date, first_release_value, first_release_vintage_date, revision_count).
  Aggregation uses `resolve_vintage()` as-of-known selection, never FIFO-last.
- Vintage awareness v2 (macro-adapter-v2): FRED's standard endpoint returns only
  the CURRENT vintage of each observation; consuming it for a historical as_of
  would silently apply later revisions (the prohibited revised-data-in-history
  failure). The adapter fetches the AS-OF-KNOWN vintage via the ALFRED realtime
  endpoint (`fetch_fred_vintages`), and `resolve_vintage()` performs
  deterministic VINTAGE SELECTION: latest eligible reference period, selected
  value = the vintage known at as_of, first-release value retained,
  revision_count = revision history where available. A vintage-feed failure
  falls back to the current-vintage endpoint but is DISCLOSED (pipeline
  `vintage_source_used`, `vintage_gaps`, `current_vintage_fallback`; per-series
  `vintage.source = "none"`), never silent. Covered by `VintageTests` in
  `tests/test_macro_registry.py` (as-of-known selection, no revision leak past
  as_of, published-time proxy degradation, fallback disclosure).
- Confidence: base 0.9 for FRED data, penalized by 0.15 per missing series.
  Never silent zero-fill; missing series explicitly flagged.
- Output: `MacroSnapshot` dataclass with ticker, as_of, status, regime,
  regime_score, series_values, series_credibility, sector_loadings,
  per_series_contributions, reason.
- Contract: `core/macro_contract.py::fetch_macro_snapshot(ticker, as_of)` —
  thin entry point, stable signature, returns full MacroSnapshot dict.
- Raw immutability: all fetches logged to `data/raw/fred_macro/{YYYY-MM-DD}.jsonl`.
- Acceptance: no-key contract byte-for-byte + tests pass; missing series
  confirmed visible and penalized; future-dated rejection verified;
  risk-on/off regime classification tested (risk_on score > 0.7, risk_off < 0.3).
  Smoke: 6/6 checks pass.

### N4. Market Regime agent ✓ DONE

Implemented 2026-09-03. Five-state governance classification with STRESS →
NO_TRADE coupling and RISK_OFF momentum dampening complete.

- Classifier: `core/regime_agent.py` — pipeline FETCH → PIT FILTER → FEATURE
  SERIES → RULE CHAIN (per session) → LABEL + PROBABILITY PROXY + TRANSITION
  RISK. States: `bullish, bearish, range, risk_off, stress`
  (`REGIME_LABELS`, `REGIME_CLASSIFIER_VERSION = regime-classifier-v1`,
  `REGIME_PIPELINE_VERSION`, `REGIME_CONTRACT_VERSION`).
- Rule chain v1 (strict precedence stress > risk_off > range > bearish >
  bullish; every comparison strict): `stress` = 30d realized vol strictly
  above its trailing 1y 95th percentile AND drawdown from the 60-session
  high > 15%; `risk_off` = vol strictly above the 1y 80th percentile OR
  (MA50 < MA200 with 20d AND 60d momentum both negative); `range` = both MA
  distances strictly inside ±2%; `bearish` = close < MA200 and MA50 < MA200
  without aligned negative momentum; `bullish` default. Pure scalar core
  `evaluate_rules` is the boundary-testable unit; `classify_regime` labels
  every session (transition risk needs per-session labels).
- Volatility percentiles compare the current session against the PRIOR 252
  sessions only (never part of its own reference distribution); windows
  require `REGIME_REQUIRED_SESSIONS = 283` eligible sessions, else INCOMPLETE
  with `regime` explicitly None (a partial rule evaluation would make the
  label depend on data availability, not the contract).
- Output: `{label, probability_proxy (per-label distance from the deciding
  boundary, scaled by REGIME_TREND_MARGIN_SCALE / REGIME_MOMENTUM_MARGIN_SCALE,
  clipped [0,1]), transition_risk (flips per trailing 20 sessions, flip_rate,
  labels)}` plus full `inputs` and `rule_trace` evidence.
- Failure states are statuses, never neutral labels: UNAVAILABLE (fetch
  failed / no eligible bars), INVALID (schema violation), INCOMPLETE
  (history shorter than the strict windows), OK. Future bars are excluded
  and counted before any feature is computed.
- Contract: `core/regime_contract.py::fetch_regime_snapshot(ticker, as_of)`.
  `RegimeSnapshot` dataclass in `core/schemas.py`; `ScoreResult.
  market_regime_snapshot` carries it; replay metadata adds
  `regime_snapshot_hash`. The legacy 3-state `market_regime` display
  heuristic in the market snapshot is untouched and ungoverned.
- Governance coupling: `market_regime_stress` veto rule (severity `veto`) in
  `RISK_POLICY_V2`, evaluated by `core/risk_policy.py` (the only evaluator) —
  triggers on the STRESS label AND fail-closed on a missing/unknown label;
  the orchestrator couples that veto to `effective_action = "NO_TRADE"`
  (regime stress blocks trades; threshold vetoes keep their ANALYSIS_ONLY
  semantics). `risk_off` multiplies the current-time momentum coefficients
  (momentum_1d/5d/20d, trend_vs_20d_mean) by
  `REGIME_RISKOFF_MOMENTUM_DAMPING = 0.5` inside `_score_current_time`,
  mirrored exactly in the scoring breakdown (`regime_momentum_damping` +
  a score-change driver note) so the explanation cannot contradict the
  number; RSI/MA-distance/volume terms and the long-term view are not
  dampened.
- Born wired: `market_regime` is the ninth decision agent — in
  `_derive_agent_statuses`, the data-agent posture loop (UNAVAILABLE/
  INCOMPLETE floors to ANALYSIS_ONLY), the risk context, audit
  model_versions/agent_statuses, expected_input_hashes, and agent_outputs.
  Ensemble weight stays 0.0 (gate, not a vote) until forecast conditioning.
- Acceptance: boundary tests at each threshold/percentile (strict
  comparisons pinned at and past every boundary); precedence, proxy bounds,
  transition flips, PIT future-bar exclusion, failure-state statuses,
  fail-closed veto coupling, dampening mirror, and an orchestrator test
  proving a stress snapshot cannot reach PAPER regardless of score.
  Full suite: 236/236 pass.

### N5. Narrative vs fundamental attribution ✓ DONE

Implemented 2026-09-03. The published score decomposes into governed
contribution buckets so the system can state whether a thesis is supported
by business reality, market narrative, or both.

- Evaluator: `core/score_engine.py::build_attribution` (pure, deterministic,
  `score-attribution-v1`). It reads the per-agent, per-horizon
  `contribution_current`/`contribution_long` entries from the ensemble
  breakdown — the single source of contribution math — and only classifies
  and aggregates them, so the attribution can never contradict the
  published score.
- Buckets (`ATTRIBUTION_BUCKETS`, versioned in `core/config.py` with an
  import-time partition guard): `operational` = fundamental_analysis +
  technical_analysis (the long-horizon technical view anchors the bucket;
  its current-horizon component is tactical but still technical evidence);
  `narrative` = news_intelligence + sentiment ONLY; `macro_shock` =
  macroeconomic + market_regime. `market_data` is the zero-weight
  informational line and sits outside the buckets
  (`ATTRIBUTION_INFORMATIONAL_LINES`).
- Block shape: `per-agent lines {bucket, contribution_current,
  contribution_long, total, status, eligible_current/long}`; `buckets
  {total, stance, members}` with stance supports/opposes/neutral against
  `ATTRIBUTION_SUPPORT_THRESHOLD = 0.25`; `thesis_support` ∈
  {operational, narrative, operational_and_narrative, neither};
  `attributed_total` + `reconciles` (strict equality against the published
  score via the same rounding path); `summary` sentence.
- Acceptance: with news at zero weight (ineligible) the narrative bucket
  reads exactly `0.0` — no phantom narrative; the summary states WHY
  (no provider vs provider-OK-but-scoreless). Weight renormalization moves
  absolute contributions when a line enters/leaves; the pinned invariant is
  the underlying agent score plus the exact narrative-line equality.
- Explanation assembly: the `explanation` string appends the attribution
  sentence (bucket totals + thesis support + summary) built from the same
  block, so the UI can state why the score moved without ever disagreeing
  with the numbers.
- Semantics note: with the current 0-10 weighted-average ensemble every
  contribution is non-negative, so the `opposes` stance is unreachable
  through the blend — it is reserved for the signed contribution semantics
  of forecast decomposition (Sprint F6) and pinned by a synthetic signed
  breakdown test.
- Location: `scoring_breakdown["attribution"]` on every `ScoreResult`.
- Acceptance: 17 new tests (`tests/test_attribution.py`) covering the
  partition guard, line/bucket math, the zero-narrative criterion, stance
  and thesis classification, exact reconciliation, explanation assembly,
  and determinism. Full suite: 253/253 pass.

### N6. Deferred-but-slotted feature gaps ✓ DONE

Implemented 2026-09-03. Features are registered with full provenance and
DEFERRED — none may enter a scorer without a weight and a test (pinned).

- ATR(14): `agents/market_data_agent.py::_average_true_range` — simple mean
  of true range over the last 14 sessions (not Wilder smoothing; versioned
  with `MARKET_FEATURE_VERSION`). True range = max(high − low,
  |high − prev_close|, |low − prev_close|); the first session's TR is
  explicitly undefined (NaN — high-low never masquerades as a full range).
  Strict window: `None` until 15 sessions exist; a shorter mean is a
  different, noisier quantity and must not be published as `atr_14`.
- 60d trend slope: `_trend_slope` — exact least-squares slope of the last
  60 closes versus session position (price/session, signed); strict window,
  `None` until 60 sessions.
- 50/100/150/200d trailing returns: reuse the anchored `_pct_change` family
  convention (0.0 until the anchor session exists), matching the existing
  `change_20d/60d` behavior.
- All six (`change_50d/100d/150d/200d`, `atr_14`, `trend_slope_60d`) appear
  as top-level snapshot fields AND provenance-complete feature contracts;
  `MarketSnapshot.from_dict` canonicalizes them.
- Scorer neutrality is test-enforced: the names are absent from
  `CURRENT_SCORE_FEATURES`/`LONG_TERM_SCORE_FEATURES` and mutating the new
  snapshot fields cannot move `_score_current_time`/`_score_long_term`.
  Wiring one in later requires a weight and a test by house rule.
- Breadth/participation: explicitly deferred with a `breadth` entry in
  `fetch_data.SOURCE_REGISTRY` (`provider_key_required`,
  `base_confidence: 0.0`, `domain: market_breadth`). Survivorship-safe
  breadth requires an index-constituent adapter and can never be inferred
  from price, volume, or technical indicators.
- Acceptance: 18 hermetic tests (`tests/test_deferred_features.py`) —
  value correctness against hand-computed/independent definitions, strict
  windows, provenance, PIT (future bars cannot change features), scorer
  neutrality, and the breadth placeholder.

---

## Sprint V — Outcome Labels, Walk-Forward Backtest, Paper Engine

Ordering rule: this sprint precedes any ML work. A model trained before a
leakage-safe label and validation harness exists would be unvalidatable by
construction. The shared-research contract in `docs/validation.md` is binding:
the backtester consumes the same feature contracts and scorers as live —
no side research dataset, ever.

### V1. Outcome label builder ✓ DONE

Implemented 2026-09-03. Point-in-time-safe labels for every persisted
decision, computed strictly from bars in `(as_of, as_of + h]` with h in
trading sessions.

- Evaluator: `core/labels.py::build_outcome_labels` (pure with respect to
  the dataset state; `OUTCOME_LABEL_VERSION = outcome-label-v1`).
  Per-horizon labels: `forward_return` (exit close vs entry close),
  `realized_vol` (the h close-to-close returns realizing inside the window,
  the first anchored at the entry close), `risk_adjusted` (return / realized
  vol — `None` for the 1d horizon where dispersion is undefined and for
  zero-vol windows, computed from the PUBLISHED rounded values so the block
  is self-consistent), and for the 20d window: `adverse_excursion` (worst
  low vs entry close within the window) and `label_20d_up` (strictly greater
  than `OUTCOME_LABEL_UP_THRESHOLD = 0.0`).
- Boundary rule: a horizon's label is null until the horizon has fully
  elapsed RELATIVE TO THE DATA'S LATEST BAR — eligibility is derived from
  the fetched frame's newest bar, never wall-clock, so the same dataset
  state always produces the same labels. Partially elapsed horizons are
  null in the snapshot and NEVER persisted (no partial-window leakage).
  Horizons are trading-session based; the window is `bars.index > as_of`
  (an intraday as_of starts the window after the moment).
- Statuses: OK (all four matured), PARTIAL (some — the spec acceptance:
  a decision dated 10 sessions ago has 1d/5d labels and 20d/60d null),
  PENDING (none), UNAVAILABLE (fetch failed / empty / no entry bar at or
  before as_of). A future as_of raises ValueError (timestamp violation,
  mirroring the market data agent).
- Storage: append-only `data/outcomes.jsonl` keyed
  `{ticker, as_of, horizon, label_version}`. Re-appending a byte-identical
  latest record is a no-op (idempotent recompute); a genuinely different
  value is appended with `supersedes` pointing at the previous record hash —
  history preserved, never mutated. `record_hash`/`labels_hash` are
  canonical SHA-256 digests (labels_hash is the future dataset-hash seed for
  M2). Malformed lines raise (integrity is loud, never skipped).
- Single price truth: the same Yahoo close series the scoring path uses; no
  separate adjustment.
- Leakage safety is test-pinned with poisoned-frame probes: bars beyond a
  horizon's window cannot change that horizon's label (or its realized vol);
  interior window bars change realized vol but not the exit-close-based
  return; pre-as_of bars other than the entry close change nothing; the
  labels_hash is stable under all of these.
- Contract: `OutcomeLabelSet` dataclass in `core/schemas.py`;
  `build_and_persist_outcome_labels` for the append flow;
  `latest_outcome_labels` resolves the newest record per horizon.
- Acceptance: 28 hermetic tests (`tests/test_labels.py`). Full suite:
  299/299 pass.

### V2. Walk-forward backtest engine ✓ DONE

Implemented 2026-09-03. Package core/backtest/ — costs.py, metrics.py,
manifest.py, engine.py — validation infrastructure only (never a
production trading path).

- Folding (engine.py::build_walk_forward_folds): anchored train
  [0, t1] → embargo (t1, t1 + e] → validation
  [t1 + e + 1, t1 + e + fold_sessions] → advance. The embargo is
  enforced >= the max label horizon (60 sessions) at import time
  (BACKTEST_EMBARGO_SESSIONS) and again per fold construction; the final
  BACKTEST_HOLDOUT_SESSIONS tail is excluded from every fold and evaluated
  once with the frozen configuration (evaluation: holdout_once).
- Offline replay seam (offline_replay_seam): injects cached price frames
  into the market-data agent, the regime agent, and the V1 label builder,
  and forces news/sentiment/macro to their no-key UNAVAILABLE contracts
  (fundamentals take the documented offline fallback) — the LIVE scoring
  path (uild_score) replays offline and deterministically. Provider
  overrides are recorded in the manifest.
- Execution model (costs.py, COST_TABLE_V1 / acktest-cost-table-v1):
  decisions at bar t act at bar t+1 open; per-side costs = half spread by
  liquidity bucket (fail-closed: unknown liquidity pays the widest spread)
  + square-root market impact (bps = coefficient x sqrt(participation),
  worst-case participation when unknown) + commission. All parameters in
  the versioned table, never inline.
- Metrics (metrics.py, acktest-metrics-v1): pure, deterministic —
  CAGR, Sharpe (annualized, configurable rf), Sortino, Calmar, max
  drawdown, win rate, profit factor, exposure, turnover, rejection rate;
  documented None conventions keep results JSON-safe. Per-fold and
  aggregate (pooled daily returns/trades/decisions).
- Run manifest (manifest.py, acktest-manifest-v1): code commit,
  feature/ensemble/score/label/cost-table/strategy versions, canonical data
  digest, config snapshot (including the cost table), seed, provider
  overrides; 
un_hash is the canonical SHA-256 of all of it.
  alidate_manifest treats an incomplete manifest as an invalid run;
  seed: None is legitimate (no randomness).
- Harness strategy (acktest-strategy-v1): enter long at score >= 6.5
  when the posture is not NO_TRADE; exit below 4.5; hold between. The final
  bar's signal cannot execute inside a window (documented boundary).
- Label alignment: every injected label is verified per decision against
  the canonical V1 recomputation by 
ecord_hash; any mismatch aborts the
  run with BacktestLeakageError (leaked_labels_detected) before any
  metric exists.
- Acceptance: a deliberately leaked variant (labels shifted one bar early)
  is detected and rejected by the harness (test); identical inputs rerun to
  identical results — the full run dict is byte-equal (manifest carries no
  wall-clock); cost parameters demonstrably change final equity and the run
  hash (test). 23 tests in 	ests/test_backtest.py. Full suite: 322/322
  pass.

### V3. Transaction costs and slippage assumptions made explicit ✓ DONE

Implemented 2026-09-03. The cost model evolved to acktest-cost-table-v2
(core/backtest/costs.py) — transaction costs and slippage are now
first-class, versioned, itemized artifacts, never hidden inside a price.

- Volatility-adjusted square-root impact (the classic law made explicit):
  impact_bps = impact_coefficient_bps * vol_factor * sqrt(participation)
  with ol_factor = clamp(daily_vol / impact_vol_baseline_daily, min, max)
  — daily_vol is the realized 20-session close-to-close volatility,
  computed by the engine for every decision (
ealized_vol_daily).
- Explicit floor/cap: 	otal_bps is clamped to
  [min_total_side_cost_bps, max_total_side_cost_bps] — never a
  zero-cost fill, never an absurd-cost fill; every record states
  clamped.
- Fail-closed fallbacks made explicit: unknown liquidity → widest
  spread; unknown participation → worst case 1.0; unknown volatility
  → neutral vol factor 1.0 with ol_available: false stamped on the
  record (the engine always computes it).
- Itemized attribution: every execution emits
  execution_cost_record (side, open/executed price, order notional,
  participation, average dollar volume, daily vol, bucket, half-spread,
  impact, vol factor, commission, raw and clamped totals, cost notional);
  trades carry entry_costs/exit_costs; the aggregate reports
  	otal_cost_notional, vg_execution_cost_bps, and cost_drag
  (cost notional / initial capital).
- Versioning: COST_TABLE_V1 frozen (no vol term, no clamp) so historical
  runs replay under their original assumptions — V1-shaped records carry
  no vol fields at all; every table carries its own
  cost_table_version; the manifest states which table produced the run.
- Binding documentation: docs/validation.md gains the “Transaction
  Costs and Slippage Assumptions” section with the full model, every
  parameter, and the explicitly accepted limitations (no intraday timing,
  no partial fills, no borrow costs, no fees/rebates/taxes, open-gap risk
  borne by the strategy, split-adjusted feed).
- Acceptance: 10 new tests (vol scaling and clip bounds, floor/cap,
  itemized records, unknown-table-version rejection, V1 backward
  compatibility, explicit attribution in the engine, expensive table pays
  more per execution and per unit of capital). Full suite: 332/332 pass.

### V4. Run manifests mandatory for every backtest ✓ DONE

Implemented 2026-09-03. V2 built the manifest builder; this task made
manifests MANDATORY and PERSISTED — a backtest run without a persisted,
valid manifest does not exist.

- Mandatory enforcement (`engine.py::BacktestManifestError`): the manifest
  is validated BEFORE any fold is replayed — a run whose manifest fails
  validation is refused outright (no decisions, no metrics, no partial
  results). `manifest_issues` in a returned result is therefore always
  empty by construction.
- Persistence (`manifest.py::persist_run_manifest`): every completed run
  appends its manifest to the append-only `data/backtest_runs.jsonl`
  (override via `manifest_store_path` for isolation). Idempotent per
  `run_hash`: a byte-identical recompute appends nothing. A same-hash/
  different-content record raises an integrity violation — the run hash is
  the canonical digest of all inputs, so that is corruption, not a
  revision. Aborted runs (e.g. leakage rejection) persist nothing: only
  completed runs exist in the store.
- Integrity on load: `load_run_manifests` / `load_manifest_by_run_hash`
  raise on malformed lines (loud, never skipped), and
  `require_valid_manifest` re-validates any manifest on demand.
- Store mirrors the outcomes/paper-order conventions: JSONL, append-only,
  `Path`-overridable for hermetic tests.
- Acceptance: 6 new tests (`ManifestMandateTests`) — every completed run
  persists a valid manifest; identical reruns are idempotent in the store;
  the engine refuses an invalid manifest before any work and stores
  nothing; same-hash/different-content is an integrity violation;
  malformed lines fail loudly; aborted runs persist nothing. Full suite:
  338/338 pass.

### V5. Shared feature/scoring contracts — research == production ✓ DONE

Implemented 2026-09-03. The Shared Research Contract is now an enforced,
test-pinned invariant — no parallel ad hoc path.

- Toolkit: `core/contract_verification.py`
  (`shared-contract-verification-v1`) — pure, diff-precise problem lists:
  `feature_contract_problems` (every feature contract complete + current
  MARKET_FEATURE_VERSION), `snapshot_field_problems` and
  `score_identity_problems` (identity across paths, including the N6
  slotted features and the ensemble/governance layer),
  `engine_version_problems` (manifest versions must equal the live config
  constants — drift is a failure), and `parallel_path_problems`
  (structural scan: research modules must call the canonical producers and
  must never define `build_score`/scorers/label builders of their own).
- Identity proof: a direct (production-reference) call and the same call
  under the offline research seam produce byte-identical market snapshots,
  score layers, and V1 labels — the seam changes only the data source,
  never the computation.
- Acceptance: 11 tests (`tests/test_shared_contracts.py`), including a
  seeded-violation test proving the parallel-path detector catches a
  rogue scorer and that removal restores a clean scan.

### V6. Paper-trading order engine (simulation only) ✓ DONE

- Objective: simulate the decision → order → fill loop with governance
  intact, producing the trade evidence later sprints consume.
- Flow: accepted decisions (`mode == "PAPER"`) emit an order intent
  `{order_id (deterministic hash of ticker+as_of+intent), side, quantity,
  intent_time}`; fill simulated at next bar open ± slippage; rejections
  carry the governing rule id. `NO_TRADE` decisions are logged too — the
  paper log must show why nothing happened.
- Invariants: idempotent order ids (retry-safe); no order path exists for
  `LIVE_DISABLED`/`ANALYSIS_ONLY` — the live branch is a `NotImplemented`
  hard stop, not a config flag away; every state transition appends to the
  audit store (W7 schema).
- Storage: `data/paper_orders.jsonl` (append-only), mirrors the trades
  concept from the schema doc (paper_only = True by construction).
- Acceptance: duplicate intent submission yields one order; a vetoed
  decision produces an intent-shaped rejection, never a fill; live branch
  raises unconditionally.

**Implemented 2026-09-16** in `core/paper_engine.py` (`paper-engine-v1`,
constants owned by `core/config.py`).

- Numbering note: commit `540cb34` landed the survivorship-safe universe
  (`core/universe.py`) under the "V6" label while this V6 — the paper
  engine — stayed unbuilt. Both are now complete; the universe work is
  tracked under **V6b** below to keep the commit history readable. Sprint V
  ("Labels / Backtest / Paper") is complete only now that Paper exists.
- Governance is upstream and absolute: `submit_order_intent` reads the
  `OrchestrationDecision` it is handed and re-derives nothing, so it cannot
  overturn a veto. Only `mode == "PAPER"` is ACCEPTED; ANALYSIS_ONLY and
  NO_TRADE produce intent-shaped REJECTED records carrying
  `governing_rule_ids` — the log shows why nothing happened.
- Idempotency: `order_id` is a truncated SHA-256 over
  (ticker, as_of, side, kind), so it is stable across processes and
  crash-retries. A resubmission returns the stored record with
  `duplicate: True` and appends nothing.
- Fills price exclusively through the V3 cost table
  (`execution_cost_record`) — no inline spread or slippage constant exists
  in the engine. Fail-closed: unknown liquidity pays the widest (micro)
  spread. A frozen `COST_TABLE_V1` replay is supported.
- No live path: `submit_live_order` raises `NotImplementedError`
  unconditionally, `EXECUTION_MODE` is permanently `LIVE_DISABLED` (guarded
  at import by `_validate_paper_config`), and a test greps `core/` and
  `agents/` to prove `LIVE_APPROVED` has no construction path.
- Acceptance: 29 tests in `tests/test_paper_engine.py`. Full suite:
  552/552 pass.

### V6b. Survivorship-safe historical universe ✓ DONE

Landed in commit `540cb34` as `core/universe.py` (`universe-ledger-v1`),
originally mislabeled V6. Membership intervals `{ticker, listed_from,
listed_to}` keep delisted members after they stop trading;
`survivorship_problems` flags a candidate universe that is missing a member
listed at `as_of`. Covered by `tests/test_universe.py`.

### V7. Monitoring foundations ✓ DONE

Implemented this sprint. Every backtest run now carries its own health
evidence, computed by pure functions — no dashboard dependency.

- Toolkit: `core/monitoring.py` (`monitoring-v1`) — pure, deterministic
  run-health metrics: score-distribution drift (population stability
  index over fixed 0-10 bins; `< 0.1` no change / `0.1-0.25` moderate /
  `> 0.25` significant), confidence drift (mean delta), stale-data rate
  (STALE agent-status fraction; missing status metadata is unknown, never
  clean), veto rate by rule (over decisions with veto metadata), and label
  coverage (fraction with a matured 20d label).
- Engine wiring: `run_walk_forward_backtest` computes a monitoring block
  from the run's own decisions and persists it append-only next to the
  manifest (`data/monitoring.jsonl`, `monitoring_store_path` override,
  idempotent per run hash — same discipline as V4's manifest store). The
  block is exposed as `run["monitoring"]` and merged into the aggregate
  metrics for a single-window read.
- Fail-closed degradation: insufficient observations yield
  `insufficient_data` (never a fabricated verdict); a missing/malformed
  store line raises on load (loud, never skipped); a same-hash/
  different-content record is an integrity violation.
- Acceptance: drift detects a seeded distribution shift (PSI > 0.25 on
  split distributions), identical windows show no change, small windows
  degrade explicitly, corrupt scores raise, engine runs stamp + persist +
  recompute idempotently (4 new engine-integration tests). Full suite:
  389/389 pass.

### V8. Framing — a backtest is historical evidence under explicit assumptions ✓ DONE

Implemented 2026-09-14 (commit bcdec69). The interpretive contract that keeps
validation honest: a backtest is evidence about historical behavior under the
assumptions it actually used — never proof the future behaves the same way.

- Evaluator: `core/framing.py` (pure, deterministic,
  `FRAMING_VERSION = backtest-framing-v1` in `core/config.py`). Every
  completed run carries a versioned framing block built by
  `build_framing_block(manifest, aggregate)`.
- Canonical statement (`FRAMING_STATEMENT`, test-pinned against upgrading
  wording): "This backtest is evidence about historical behavior under the
  explicit assumptions listed in this block. It is not proof that the future
  will behave the same way."
- Assumptions (`assumption_entries`) are sourced FROM THE MANIFEST —
  cost_table, strategy, price_basis, universe, embargo/fold/holdout session
  geometry, enter/exit scores, data_digest — so the framing can never
  disagree with the run it frames. `framing_problems(manifest)` is
  fail-closed: a missing assumption refuses the run. The engine merges
  framing problems into the manifest validation and raises
  `BacktestManifestError` BEFORE any fold is replayed (weak framing is
  refused before replay, never after results exist).
- Limitations (`FRAMING_LIMITATIONS`): the V3/V6-accepted simulation limits
  (no intraday timing, no partial fills, no borrow/rebate costs, open-gap
  risk, price-return basis, no exchange fees/taxes, PIT-only) as an
  explicit, versioned list — never silently rediscovered.
- Non-claims (`FRAMING_NON_CLAIMS`): not_proof_of_future, not_guarantee,
  not_generalization, not_release_gate_waiver — the only framing vocabulary
  a run may use about its own meaning. Strong historical results never
  upgrade the framing (test-pinned): `evidence_status` is
  `historical_evidence` or `historical_evidence_no_decisions`; the
  aggregate decision count only informs the evidence note, never the status
  or the statement.
- Persistence: append-only `data/framing.jsonl`
  (`persist_framing_snapshot`, idempotent per (run_type, run_hash);
  byte-identical recompute appends nothing; same-key/different-content is
  an integrity violation, never a silent revision; malformed lines raise on
  load). The engine default mirrors the manifest store path;
  `run["framing"]` exposes the block next to the manifest and monitoring
  snapshot.
- Binding documentation: `docs/validation.md` gains the "Backtest Framing
  (V8)" section with the full contract.
- Acceptance: canonical statement and non-claims pinned; superb and
  terrible aggregates frame identically; a missing assumption refuses the
  run before replay; idempotent store behavior; every engine run carries
  the block. 17 tests in `tests/test_framing.py`. Full suite: 411/411 pass
  at landing (508/508 as of N3 v2).

## Sprint M — ML Layer, Calibration, Model Registry

> **Numbering.** This board and `NEXT_STEPS_SPRINTS.md` (derived from the
> master context) number Sprint M differently. This board is authoritative
> for execution; the mapping is:
>
> | This board | Master context (`NEXT_STEPS_SPRINTS.md`) |
> |---|---|
> | M1 Feature registry | M1 Canonical Feature Registry |
> | M2 Model registry and artifact tracking | M5 Model Artifact Registry |
> | M3 Training pipeline | M2 Training Dataset Builder + M4 Baseline Suite |
> | M4 Calibration and score mapping | M6 Calibration and Uncertainty |
> | M5 Promotion gates and drift hooks | M7 Champion/Challenger + M8 Reproducibility Gate |
>
> The master context's **M3 Research Trial Registry** has no slot on this
> board yet — it needs one before training runs begin, or experiments will
> bypass the trial registry (a named failure mode, master context section 32).

Preconditions: Sprint V complete (labels exist, harness exists, paper
evidence accumulates). Dependency note: this sprint introduces the project's
first training dependency (scikit-learn is the pragmatic choice) — that is a
requirements.txt change requiring explicit review, plus pinning consistent
with the existing style.

### M1. Feature registry (single source of truth) ✓ DONE

Implemented 2026-09-15 (completed the registry started in commits fe70627 /
4a72757). The registry is the only door into a production model.

- Evaluator: `core/feature_registry.py` (`feature-registry-v1`, owned by
  `core/config.py::FEATURE_REGISTRY_VERSION`). Every feature is declared as a
  `FeatureSpec` with complete, validated metadata — name, owner, domain,
  formula, version, unit, frequency, lookback, minimum history, null policy,
  PIT rule, source dependencies, feature family, model compatibility — with
  closed vocabularies (unknown unit/family/domain/frequency/source/null
  policy/model family is a rejection, never a silent default).
- Default registry: 22 market-data features (owner `market_data_agent`,
  version `MARKET_FEATURE_VERSION`) + the 5 fundamental factors that enter
  the production ensemble via `_build_fundamental_score` (owner
  `fundamental_agent`, version `FUNDAMENTAL_FEATURE_VERSION`); producers
  without registered features are reported informationally by
  `unwired_producers` and wired sprint by sprint.
- Binding rule (three gates): `model_feature_problems` (a model's declared
  feature set), `feature_contract_problems` (the contracts a producer
  emitted — unregistered, PIT-violating, version-drifted, lookback-mismatched
  or undeclared-source contracts are rejected), and the walk-forward engine
  (refuses to start on a non-conformant declared model surface and refuses
  mid-run on a non-conformant consumed surface).
- Live score path: the W3 auditor gained the `feature_registry_conformance`
  veto check (`core/audit_policy.py`) — the decision's market snapshot is
  checked against the registry on every orchestration; an unregistered or
  PIT-violating feature contract forces `auditor_veto` and blocks PAPER.
  Fail-closed: a missing snapshot or empty feature surface vetoes.
- Producer must exist: `PRODUCERS` maps each owner to a real module +
  callable on disk (`producer_problems`); re-registering a definition without
  a version bump is refused (`FeatureRegistry.register`).
- Deterministic identity: spec, registry, named feature set, and consumed
  surface each carry a canonical SHA-256; persistence is append-only at
  `data/feature_registry.jsonl`, idempotent per registry hash, integrity
  violations raise.
- CI drift gate: `scripts/check_feature_registry.py` — registry validity,
  producer resolution, live-snapshot conformance, emitted-vs-registered
  surface drift (diff-precise), hash determinism, persistence round-trip;
  exit 1 on any drift. Removing/renaming a snapshot feature breaks the gate
  with a diff-precise message.
- Acceptance: 95 tests in `tests/test_feature_registry.py` (spec validation,
  hashes, PIT/rejection gates, producer wiring, persistence, engine +
  contract-verifier + auditor + orchestrator integration, fundamental factor
  registration). Full suite: 522/522 pass.

### M1b. Contextual-agent registry coverage ✓ DONE

Closed 2026-09-16. M1's rule was enforced, but only on the market feature
surface: `news_intelligence` (0.10 ensemble weight) and `macroeconomic`
(0.10, both horizons) reached the published score through their own snapshot
shapes, which the registry did not describe. A news or macro input change was
invisible to the drift gate.

Both halves of the gap are now closed:

- **Registration.** Four contextual signals registered in
  `core/feature_registry.py` — `news_sentiment_score` (news-contract-v1),
  `macro_regime_score` (macro-contract-v1), `regime_probability_proxy`
  (regime-contract-v1) and `sentiment_score` (sentiment-contract-v1).
  Exactly the values that reach the production path; no aspirational
  entries. Registry: 22 market + 5 fundamental + 4 contextual = **31
  features**. `unwired_producers` drops from 5 to 1 — only
  `technical_agent`, legitimately, since W5 made it a pure delegator that
  owns no features of its own.
- **Enforcement.** Registration alone changes nothing if no surface is
  built. `contract_verification.contextual_feature_surface` assembles the
  live contextual contracts and `contextual_feature_problems` gates them;
  the W3 auditor's `feature_registry_conformance` check now evaluates that
  surface alongside the market one, so a live news or macro contribution
  that is unregistered, PIT-violating, version-drifted or from an undeclared
  source forces `auditor_veto`.
- **Fail-closed.** `null_policy` is `exclude`, never `default`: a non-OK
  agent yields no feature rather than a neutral substitute. An empty
  contextual surface is conforming — it means no contextual agent is OK,
  which the agent-status taxonomy already governs.
- **The gate is proven non-vacuous.** `scripts/check_feature_registry.py`
  plants a future-dated news contract and fails if it is NOT rejected, and
  verifies every scoring producer owns a registered feature.
- Acceptance: 104 tests in `tests/test_feature_registry.py` (7 new for
  contextual enforcement). Full suite: 593/593 pass.

Implementation note: the regime agent reports `lookback_period` as the
sessions it actually consumed, which varies per run and can never equal a
single registered value. The surface carries the registered lookback (the
definition's window) while the observed count stays visible in the agent's
own payload — caught by the auditor during implementation, not by review.

### M2-DS. Official training dataset builder ✓ DONE

> Master-context M2 ("Official Training Dataset Builder"). Tracked with a
> `-DS` suffix because this board's M2 slot is the model/artifact registry —
> see the numbering table above. Built ahead of the board's M2 because a
> model registry with no dataset to register is premature.

Implemented 2026-09-16 in `core/training_dataset.py` (`training-dataset-v1`,
schema `training-row-v1`, constants owned by `core/config.py`).

- **One door into a training set.** The builder CALLS the canonical
  producers and reimplements nothing: features come from
  `core.score_engine.build_score` replayed through the V2
  `offline_replay_seam`, outcomes from the V1
  `core.labels.build_outcome_labels`. A side extractor that pulls features
  another way is the classic leakage route, so the same parallel-path rule
  that governs research applies here.
- **Row shape:** `prediction_time -> information available at
  prediction_time -> features -> future outcome`. Features and outcome never
  share a code path.
- **Leakage is caught, not assumed absent.** `row_problems` re-verifies every
  contract's `published_time <= prediction_time` independently of the
  registry check, on freshly built rows and on rows loaded back from disk.
  The CI gate plants a future-dated contract and fails if it is NOT rejected
  — a guard that cannot fail is theatre.
- **M1 gate reaches datasets.** An unregistered, PIT-violating,
  version-drifted or undeclared-source feature cannot enter a dataset any
  more than it can enter a production model (`model_feature_problems`).
- **Nothing is imputed.** A missing feature, a null/NaN value, an
  UNAVAILABLE label or an unmatured horizon excludes the row with a recorded
  reason in `report()["exclusion_reasons"]`. Coverage is never silently
  thinned and absence is never a zero.
- **Deterministic `dataset_hash`:** canonical SHA-256 over every row's
  identity (including feature *contracts*, not just values — the same number
  from a different source is different training data) plus the feature-set
  hash, target horizon and label version. Rows are sorted before hashing, so
  build order cannot change identity. Verified by rebuild.
- Append-only manifest at `data/training_datasets.jsonl`, idempotent per
  dataset hash; the manifest carries provenance, not the row payload.
- CI drift gate: `scripts/check_training_dataset.py` — builds from real
  cached history, checks conformance and leakage, proves hash determinism,
  proves both guards non-vacuous, checks persistence idempotency.
- Acceptance: 32 tests in `tests/test_training_dataset.py`. Full suite:
  584/584 pass.

Dataset feature scope: the builder draws the technical scorers' declared
inputs by default. Since M1b the contextual signals are registered too, so a
caller may request them explicitly via `feature_names` once those agents have
live providers connected (news/macro/sentiment are provider-gated and read
UNAVAILABLE offline).

### M2. Model registry and artifact tracking ✓ DONE

Implemented 2026-09-16 in `core/model_registry.py` (`model-registry-v1`,
constants owned by `core/config.py`), persisted at `models/manifest.json`.

- **Entries** carry `{model_version, family, feature_set_version,
  training_data_cutoff, artifact_uri, status, metrics, approved_by,
  approved_at, parent_version}` plus `dataset_hash` (joins an M2-DS dataset
  to the model trained on it), `oos_comparison` and `retired_at/reason`.
  Closed vocabularies: an unknown family or status is a rejection.
- **The live-path rule is enforced.** `require_live_model` resolves a version
  or raises: unregistered, `candidate` and `retired` all refuse. The
  orchestrator resolves `market-data-v1`, `technical-v1` and
  `fundamental-v1` through the registry — the three hardcoded literals are
  gone, and a test plus the CI gate assert they stay gone.
- **The override is audited by construction.** The only way past the gate is
  an explicit `override_reason`, and the returned resolution record names the
  override. Those records flow into the W7 audit event (`model_resolutions`,
  schema bumped to `audit-event-v3`), so a non-approved model can never back
  a decision without leaving a permanent trace. This is the acceptance
  criterion, literally.
- **Promotion is gated.** `promote` refuses without an out-of-sample
  comparison naming the pre-registered primary metric, both sides' values and
  the sample size; refuses when the comparison does not name the real
  incumbent; refuses a candidate that does not beat the incumbent (loss
  metrics declare `higher_is_better: False`); and refuses an automated
  `approved_by` — governance sign-off cannot be a CI job.
- **Retirement never deletes.** `retire` flips status and records when/why;
  the entry and its `artifact_uri` survive, because historical decisions must
  stay explainable.
- Seeded entries are the deterministic rule-based scorers backing today's
  score path. They carry no dataset hash or artifact — the binding rule is
  about governance, not about whether gradient descent was involved — and
  any trained successor must beat them through `promote()`.
- CI drift gate: `scripts/check_model_registry.py` — registry validity,
  committed-manifest-vs-code drift, live-gate refusals proven non-vacuous,
  override recording, all three promotion refusals, retirement preservation,
  and the absence of hardcoded literals.
- Acceptance: 44 tests in `tests/test_model_registry.py`. Full suite:
  637/637 pass.

### M3-TR. Research trial registry ✓ DONE

> Master-context M3 ("Research Trial Registry"). Tracked with a `-TR` suffix
> because this board's M3 slot is the training pipeline — see the numbering
> table above. Built first because the board's M3 runs experiments, and an
> experiment that is not pre-registered is not evidence.

Implemented 2026-09-16 in `core/trial_registry.py` (`trial-registry-v1`,
constants owned by `core/config.py`), ledger at `data/research_trials.jsonl`.

- **Every M3 field recorded and validated:** trial_id, hypothesis,
  feature_set_version, model_family, hyperparameters, label_version,
  horizons, training_window, validation_scheme, costs, seed, dataset_hash,
  metrics. Closed vocabularies for model family and validation scheme;
  `costs` must name its `cost_table_version` (V3); `dataset_hash` ties a
  trial to an exact M2-DS dataset; a stale `label_version` is refused.
- **Pre-registration is structural.** A registered trial carries NO metrics —
  validation refuses them — and `complete_trial` refuses metrics that omit
  the pre-registered `primary_metric`. Choosing the metric after seeing the
  results is the definition of cherry-picking, so the metric is locked at
  registration.
- **Results are immutable.** A completed or abandoned trial cannot be
  re-completed, abandoned or edited. Editing a registered trial's
  configuration breaks its `trial_id` and validation says so.
- **An identical configuration is the SAME experiment.** `trial_id` is a
  deterministic hash over the configuration — hypothesis included, metrics
  and timestamps excluded — so re-running an experiment cannot masquerade as
  a new independent result. This is what makes the X7 multiple-testing count
  honest.
- **Abandoned trials are recorded, never deleted**, and must state why.
  `multiple_testing_report()` counts them alongside completed trials: the
  number of things tried is the number people forget, and it is exactly the
  number a significance correction needs.
- Append-only ledger: registration and completion are separate lines, so the
  file itself evidences that the hypothesis preceded the result.
- CI drift gate: `scripts/check_trial_registry.py` — 19 guards, each
  exercised and required to fire. Verified non-vacuous by sabotage: disabling
  the primary-metric lock makes the gate exit 1.
- Acceptance: 41 tests in `tests/test_trial_registry.py`. Full suite:
  678/678 pass.

### M3. Training pipeline (offline, reproducible) ✓ DONE

Implemented 2026-09-16 in `core/training.py` (`training-pipeline-v1`) with
the `scripts/train.py` entrypoint the board names.

**Dependency:** this sprint added the project's first ML dependency,
`scikit-learn==1.9.1`, reviewed and approved before landing. `joblib==1.6.0`
is pinned alongside it rather than left transitive, because it serialises the
artifacts M8 must hash.

- **Splits come from the V2 harness, never sklearn.** `train_test_split`,
  `KFold` and `cross_val_score` shuffle by default and leak the future into
  training on panel data. The trainer calls `build_walk_forward_folds` — the
  same anchored windows and embargo the backtest engine uses — and both a
  test and the CI gate parse the module's AST to prove no sklearn splitter is
  imported.
- **The registry gates training.** Every dataset feature must be registered
  and declare compatibility with the family being trained; an unregistered
  feature aborts the run before the matrix is built.
- **Baselines first.** Six estimators: `historical_mean` and `momentum`
  (pure numpy, no dependency) plus ridge, elastic_net, random_forest and
  gradient_boosting. Sequence/temporal models stay out of scope until these
  survive validation.
- **Determinism.** Same dataset hash + same seed + same configuration produce
  identical metrics and an identical artifact hash, verified across separate
  processes including RandomForest. The hash is taken over FITTED PARAMETERS,
  not pickle bytes: joblib output embeds library versions and is not
  byte-stable across environments, which would make the M8 guarantee
  untestable in CI. The environment (python/numpy/sklearn/platform) is
  recorded so M8 can say WHY two runs diverged.
- **Metrics are out-of-sample by construction** — pooled observations equal
  the sum of validation rows, and a test asserts it.
- CI drift gate: `scripts/check_training_pipeline.py` (~26s). Verified
  non-vacuous by sabotage: disabling the registry gate makes it fail, and it
  distinguishes a clean refusal from an incidental crash.
- Acceptance: 26 tests in `tests/test_training.py`. Full suite: 705/705 pass.

Found during implementation: the 22 market features declared only
`technical_analysis` compatibility, so the M1 gate correctly refused to let
any ML family consume the very features the M2-DS builder emits. They now
declare the ML families, matching what the fundamental and contextual
features already did — an M1 oversight the gate caught rather than a
deliberate restriction.

First result, recorded honestly: on a 500-row NVDA dataset the pure
`historical_mean` baseline beat every trained model on RMSE. No model has
earned promotion, and `scripts/train.py` says so explicitly when a pure
baseline wins. That is the point of establishing baselines first.

### M4-BL. Baseline model suite ✓ DONE

> Master-context M4 ("Baseline Model Suite"). Tracked with a `-BL` suffix
> because this board's M4 slot is calibration — see the numbering table
> above.

Implemented 2026-09-17 in `core/baseline_suite.py` (`baseline-suite-v1`),
with `mean_reversion` and `logistic` added to `core/training.py` to complete
the family list.

- **Eight baselines across all seven M4 families:** `historical_mean`
  (baseline_mean), `momentum`, `mean_reversion`, `ridge` + `elastic_net`
  (linear), `logistic`, `random_forest` (tree), `gradient_boosting`.
  `mean_reversion` is the deliberate mirror of `momentum` — the opposing
  hypothesis, so if momentum has an edge this must lose by a similar margin.
  `logistic` classifies direction and maps back to a signed magnitude so it
  is comparable with the regressors on the same metrics; a single-class
  training window degrades to that class rather than raising, and records it.
- **The incumbent is the best SIMPLE baseline, not the best model.** A
  trained model must clear historical_mean / momentum / mean_reversion before
  it is interesting at all. Comparing a boosted tree only against ridge would
  let a whole family of complexity in through the side door.
- **Comparisons must be like-for-like.** `comparison_problems` refuses runs
  that differ in dataset hash, target horizon or feature set rather than
  trusting the caller — comparing across different data is the most common
  way a "win" turns out to be an artefact.
- **A win must be real.** Three bars: beat the incumbent, beat it by at least
  `BASELINE_PROMOTION_MARGIN` (0.02), and win at least
  `BASELINE_MIN_WINNING_FOLD_RATIO` (0.6) of paired folds. A hair-thin margin
  or one lucky fold is refused.
- **The rule connects to M2.** `promotion_comparison` emits exactly the
  `oos_comparison` shape `ModelRegistry.promote` demands, and REFUSES to emit
  one for a losing verdict — so the measurement and the gate cannot disagree.
  A test drives the whole path: measured win → comparison → approved model.
- **"Nothing earned promotion" is a first-class result**, returned as a
  populated report with per-candidate reasons — never an exception, never a
  silently chosen best-of.
- CI drift gate: `scripts/check_baseline_suite.py` (0.3s, synthetic runs).
  Verified non-vacuous by sabotage: disabling the fold-consistency check
  makes it fail with "a one-lucky-fold candidate was promoted".
- Acceptance: 28 tests in `tests/test_baseline_suite.py`. Full suite:
  733/733 pass.

Current standing on real data (600-row NVDA, 5 folds): the incumbent is
`historical_mean` at 0.6025 directional accuracy, and **no candidate beats
it** — every trained model scores between 0.41 and 0.51. No model has earned
promotion. That is the expected and correct outcome at this stage, and the
reason M4 exists before M5-M8.

### M5-AR. Model artifact registry ✓ DONE

> Master-context M5 ("Model Artifact Registry"). Tracked with a `-AR` suffix;
> this board's M5 slot is promotion gates and drift hooks — see the numbering
> table above.

Completed 2026-09-17 as an extension of the board's M2 registry rather than a
second registry. Eight of the sixteen M5 fields already existed; this adds
the rest plus the rules that make them mean something.

- **New fields on `ModelEntry`:** `code_commit`, `hyperparameters`, `seed`,
  `artifact_hash`, `calibration_version`, `forecast_target`, `horizon`,
  `universe` (`model-artifact-v1`). All sixteen M5 fields are now recorded,
  asserted by a test that names each one.
- **Trained artifacts must be reproducible.** `is_trained()` distinguishes a
  fitted artifact from a deterministic rule-based scorer. A trained entry
  must carry code_commit, dataset_hash, horizon, universe, a seed and
  hyperparameters — without them nobody can rebuild it or say what produced
  it. The seeded scorers stay exempt because they genuinely have no artifact,
  and a gate check asserts they never claim otherwise.
- **`forecast_target` is a closed vocabulary** (`expected_return`,
  `probability_up`, `expected_volatility`). Two models over the same features
  predicting different things are not interchangeable, and the registry now
  refuses to blur them.
- **`calibration_version` is mandatory and defaults to `"uncalibrated"`** —
  an honest value. A missing field is not, so a blank is refused. M6 will
  build on this rather than having to infer calibration state.
- **Provenance is captured, never retyped.** `entry_from_training_run()`
  reads the dataset hash, feature-set version, artifact hash, seed,
  hyperparameters and horizon off the M3 run that actually produced the
  model, and `code_commit()` reuses the V4 manifest helper so a research run
  and a model entry can never disagree about which commit they came from.
  Hand-copying these is how a registry drifts from reality.
- **Registering is not trusting.** An entry built from a training run is born
  `candidate`, and the live gate refuses it until the M2 promotion gate
  passes — an out-of-sample win against the incumbent (M4) plus a human
  approver.
- **Provenance is part of artifact identity:** the canonical hash now covers
  code_commit, artifact_hash, seed, horizon, universe, forecast_target and
  calibration_version, so two entries differing in any of them cannot share a
  hash.
- CI gate: extended `scripts/check_model_registry.py` rather than adding an
  eighth gate. Verified non-vacuous by sabotage: disabling the trained-artifact
  rule makes it fail with four precise messages.
- Acceptance: 62 tests in `tests/test_model_registry.py` (18 new). Full
  suite: 751/751 pass.

`models/manifest.json` was regenerated — the committed-manifest drift test
caught the hash change immediately, which is the check working as designed.

### M4. Calibration and score mapping ✓ DONE

> Board M4 and master-context M6 are the same task; this entry satisfies
> both. Implemented 2026-09-17 in `core/calibration.py` (`calibration-v1`).

The binding rule: **never expose arbitrary probability numbers as if they
were calibrated.** A raw model score is not a probability — `0.7` out of a
booster means nothing until mapped through a calibration fitted on held-out
data and measured against what happened.

- **Enforced in code, not in a comment.** `calibrated_probability(score,
  None)` raises `UncalibratedProbabilityError`. There is no passthrough, so
  an uncalibrated number cannot reach a caller by accident.
- **Isotonic preferred, Platt below `CALIBRATION_MIN_ISOTONIC_SAMPLES`**
  (100), because isotonic overfits thin samples. The substitution is
  RECORDED in `fallback_reason` so a reader never guesses which ran. Below
  `CALIBRATION_MIN_SAMPLES` (30), or on a single outcome class, calibration
  is refused outright — a map nobody should trust is worse than a refusal.
- **Measured out-of-fold.** This is the part that matters. Measuring a map on
  the data it was fitted to reports in-sample calibration, which is
  near-perfect by construction: the first real run returned ECE exactly
  `0.0`, which was a red flag rather than a success. `_out_of_fold_probabilities`
  now fits on the other folds and scores each held-out fold, so no
  observation is scored by a map that saw it. Runs with fewer than three
  folds fall back to in-sample and say so via `measurement_basis` rather
  than reporting flattering numbers silently.
- **Monotone by construction** in both methods — a higher raw score can never
  yield a lower probability — and isotonic clamps outside its fitted range
  rather than extrapolating into scores it never saw.
- **Every M6 requirement:** `fit_calibration`/`apply` (probability
  calibration), `reliability_curve`, `brier_score`, `log_loss`,
  `expected_calibration_error` + `max_calibration_error`,
  `prediction_interval`, `CalibrationReport.uncertainty`, `fold_dispersion`.
  Empty reliability bins are omitted, not zeroed: no observations is not the
  same as never happened.
- **M3 change:** `FoldResult` now retains each fold's out-of-sample
  predictions and actuals. Calibration cannot be fitted on validation folds
  if the validation predictions were discarded.
- CI drift gate: `scripts/check_calibration.py` (~3s). Verified non-vacuous
  by two sabotages: letting an uncalibrated score pass through, and
  switching the measurement back to in-sample. Both fail the gate with
  precise messages.
- Acceptance: 43 tests in `tests/test_calibration.py`. Full suite: 799/799
  pass.

Honest first numbers on a real gradient_boosting run (400 out-of-fold
observations, 5 folds): Brier 0.351, log loss 6.14, ECE 0.233, MCE 0.748,
fold directional accuracy 0.26-0.58 (std 0.107). The model is badly
calibrated and highly dispersed across folds. That is the correct reading —
no model has beaten the baseline (M4-BL), so there is nothing here that
should be trusted yet, and the numbers now say so plainly.

### M7-CC. Champion / challenger ✓ DONE

> Master-context M7. Implemented 2026-09-17 as an extension of the M2/M5
> registry (`champion-challenger-v1`), not a new module.

**The gap it closed.** The registry had no role at all. `incumbent()`
returned whichever approved model sorted last by version string, so with
three approved models the most important question — *which model is actually
serving?* — was an accident of naming. A model's role is now declared.

- **Role is distinct from lifecycle status.** `approved` means governance
  cleared it; `champion` means it serves. `is_live_eligible()` requires
  BOTH, so an approved challenger is measured rather than trusted, and an
  approved shadow model is not even consulted.
- **Every model is born `shadow`** (`MODEL_DEFAULT_ROLE`, guarded at import).
  No code path registers a model directly into a serving role.
- **shadow → challenger requires evidence:** `SHADOW_MIN_OBSERVATIONS` (100)
  recorded out-of-sample observations. An unmeasured model has earned
  nothing.
- **challenger → champion requires governance approval**, so crowning cannot
  substitute for the M2 promotion gate (an OOS win plus a human approver).
- **Crowning is a SWAP, not an addition.** `crown_champion` demotes the
  incumbent in the same operation, which is what makes "exactly one
  champion" impossible to violate halfway through. The outgoing champion
  keeps its entry and artifact and becomes a challenger — nothing is
  deleted.
- **Championship is per forecast contract** `(target, horizon, universe)`. A
  20d expected-return champion and a 5d direction champion are not rivals —
  they answer different questions. `role_problems()` reports any contract
  with more than one champion, and `champion()` raises rather than picking
  one.
- **The three seeded scorers are champions of their own contracts**
  (`market_quality`, `technical_view`, `fundamental_view`). They are
  complementary ensemble components, not competitors for one slot, so all
  three serve simultaneously without violating the invariant.
- CI: extended `scripts/check_model_registry.py` rather than adding a ninth
  gate. Verified non-vacuous by sabotage — disabling the demotion makes it
  fail with two messages, including the two-champion invariant breach.
- Acceptance: 34 tests in `tests/test_champion_challenger.py`. Full suite:
  833/833 pass.

`models/manifest.json` was regenerated; the drift test caught the hash change
immediately.

### M8-RG. ML reproducibility gate ✓ DONE

> Master-context M8. Implemented 2026-09-17 in `core/reproducibility.py`
> (`reproducibility-v1`).

The rule's last clause — "within explicitly defined reproducibility
guarantees" — carries the weight. A blanket "bit-identical forever" claim
would be false the moment numpy changes a summation order, and a guarantee
that is quietly false is worse than none. So the promise is stated in
config, verified, and its limits are published with every artifact.

**GUARANTEED** (same environment, same inputs): identical fitted parameters
and artifact hash, identical predictions/metrics/run hash, and the same
across separate processes.

**NOT GUARANTEED** (deliberately): bit-identical artifacts across library
versions or platforms, or stability across a formula change. The environment
block records python/numpy/sklearn/platform so such a divergence is
*attributable* rather than mysterious, and `explanation()` reports it as
expected rather than as a bug.

- **Two identities, deliberately separate.** `artifact_hash` identifies the
  FITTED MODEL (estimator + feature surface + parameters); `run_hash`
  identifies the CONFIGURATION, seed included.
- **Bug found and fixed:** the artifact hash previously included the seed, so
  `historical_mean` and `ridge` — deterministic algorithms that never consult
  it — reported a different artifact under a different seed. A false
  difference on every deterministic estimator is exactly the noise that
  trains people to ignore a gate. The seed now lives only in run identity.
- **Zero tolerance.** Within one environment the same inputs must produce the
  same floats, not merely close ones; a tolerance would hide the
  nondeterminism the gate exists to catch. Guarded at import.
- CI gate: `scripts/check_reproducibility.py` (~24s), including a
  **cross-process** check that trains in a fresh interpreter — in-process
  caching or a warm RNG cannot fake it.
- Acceptance: 24 tests in `tests/test_reproducibility.py`. Full suite:
  857/857 pass, nine CI gates green.

**A hole the sabotage run found.** The first version of the gate ran the
same-process check on `ridge` and `gradient_boosting` only, using
`random_forest` purely for divergence checks. Removing RandomForest's seed
therefore left the gate GREEN — the sabotage passed. Fixed by covering every
stochastic estimator in the same-process check; re-sabotaging now fails with
a precise message. Worth recording: the gate was wrong in a way that only an
attempted break could reveal.

### M5. Promotion gates and drift hooks### M5. Promotion gates and drift hooks

- Promotion checklist automated in `scripts/promote.py`: OOS metrics beat
  incumbent on the pre-registered primary metric, no regression on veto-rate
  or false-positive rate beyond tolerance, drift check (V4) clean, manifest
  complete, human approval recorded. Any failure → candidate stays.
- Historical predictions are immutable: promotion never rewrites past
  decisions' model versions (append-only audit guarantees this).
- Acceptance: attempt to promote with a missing manifest field fails loudly;
  the full gate sequence is exercised in a test with a synthetic candidate.

---

## Sprint E — Event Intelligence & Market Reaction

### E1. Canonical event object ✓ DONE

Implemented 2026-09-17 in `core/event_contract.py` (`event-v1`).

Sprint E learns how events move the chart, which requires ONE event shape
every downstream stage agrees on — E4 (event study), E5 (attribution) and E6
(event memory) must all read the same object, or they will quietly disagree
about what an "event" is, and that surfaces as a mysterious modelling result
rather than an error.

- **All 16 E1 fields**, validated: event_id, published_time, effective_time,
  entity, entities_affected, actor, actor_type, event_type, source,
  source_quality, novelty, relevance, direction, magnitude, confidence,
  evidence.
- **Adapts N1; never re-classifies it.** `event_from_article` maps an
  existing pipeline article onto the canonical shape. Re-deriving category,
  tone or source quality here would create a second, divergent opinion about
  the same article.
- **`published_time` and `effective_time` are distinct**, and the PIT filter
  keys on PUBLICATION. A Monday announcement of a Friday plant closure is
  legitimate and common; collapsing the two is how a backtest silently uses
  future information. `is_backdated()` makes the gap visible because an event
  study anchored on the wrong timestamp measures the wrong window.
- **`event_id` is deterministic** over defining content, excluding the
  scores — rescoring an event does not make it a new event, so the same story
  carried by two providers cannot be double-counted in event memory.
- **CONTRADICTORY is a direction**, propagated into every event of a
  contradictory snapshot rather than averaged to neutral.
- **`magnitude` is a bounded unitless claim size, not an expected return** —
  naming it a return would invite treating an unvalidated number as a
  forecast before Sprint F exists to make real ones.
- **Fail-closed:** a non-OK snapshot yields no events; an unsourced event is
  a rumour and is refused; incomplete articles are skipped rather than
  patched with a fabricated timestamp.
- **Conservative actor typing.** Only company-issued event types claim a
  company actor; everything else stays `unknown`. E2/E3 resolve actors
  properly, and guessing now would put unearned confidence into the data
  those sprints consume.
- CI drift gate: `scripts/check_event_contract.py`. Verified non-vacuous by
  sabotage — switching the PIT filter to `effective_time` fails it with the
  leakage message.
- Acceptance: 40 tests in `tests/test_event_contract.py`. Full suite:
  913/913 pass, ten gates green.

### E2. Entity resolution ✓ DONE

Implemented 2026-09-17 in `core/entity_resolution.py` (`entity-resolver-v1`).

**The defect it fixes.** The N1 relevance heuristic returned `0.0` for two
very different situations: an article correctly identified as being about
another company, and an article about the right company that the resolver
simply could not attribute. Collapsing them made coverage gaps invisible —
"we have no news about this company" was indistinguishable from "we have
news we could not attribute".

- **Five match methods with explicit confidence:** ticker (1.0), legal_name
  (0.9), alias (0.8), executive (0.6), plus `ambiguous` and `none` at 0.0.
  Every resolution records WHY it decided, not only what it decided.
- **An executive mention now resolves** ("Jensen Huang announces a GPU" →
  NVDA), but at 0.6 — deliberately weaker than a direct match, because a CEO
  is frequently quoted about the industry, a rival or the economy rather
  than their own company.
- **Ambiguity is a resolution, not an error.** Text naming two registered
  companies resolves to `ambiguous` with zero confidence and names the other
  candidates, rather than being arbitrarily assigned.
- **Short tickers require a name.** A bare "V" or "BE" is an ordinary
  English word, so tickers at or below two characters must match by name or
  alias — the same class of error that produced `QCOMCRWV` in the portfolio
  list.
- **Phrase matching respects word boundaries:** "Metaverse" is not "Meta",
  and "Advanced Micro Devices" does not match those three words scattered
  through a sentence.
- **A registry, not a regex.** Aliases and executives are curated data with
  a visible owner, so a wrong alias is a one-line fix rather than a buried
  pattern. `registry_problems` refuses a shared alias, which would otherwise
  make every mention of it ambiguous.
- **Wired into E1.** Every event carries its `entity_resolution`, and
  `events_from_news_snapshot(..., require_resolved_entity=True)` drops
  unresolved events before they can become training rows. An event with NO
  recorded resolution is treated as unresolved — an absent resolution is the
  most silent failure of all.
- **Rejections are recorded, not discarded.** `resolution_report` separates
  `none` from `ambiguous` because they call for different fixes: a registry
  gap versus genuinely hard source material.
- CI drift gate: `scripts/check_entity_resolution.py`. Verified non-vacuous
  by two sabotages — removing the short-ticker guard and removing the
  ambiguity check both fail it.
- Acceptance: 41 tests in `tests/test_entity_resolution.py`. Full suite:
  954/954 pass, eleven gates green.

### E3. Influential person intelligence ✓ DONE

Implemented 2026-09-17 in `core/actor_intelligence.py` (`actor-registry-v1`).

E3 tracks identity, role, organization, historical relevance, topic
specialization, source credibility, statement frequency, novelty and
historical market impact — the inputs for eventually learning
`actor × topic × company → historical market response`.

**The property that matters most is about NOT knowing things.** E3 exists
before E4 measures any market reaction, so an actor's "historical impact" is
the number the system is most tempted to invent.

- **Impact is `unmeasured` below `ACTOR_MIN_OBSERVATIONS` (20), and reports
  NO numbers there** — not a mean over three events that a consumer could
  mistake for a finding. Statements with no measured reaction do not count
  toward a track record at all.
- **Credibility is labelled a PRIOR.** `credibility()` returns
  `basis: "prior"` with a detail saying it is not a measurement. Presenting a
  guessed number as measured is exactly the failure M6 exists to prevent.
- **Measured impact disclaims causality** — an abnormal return following a
  statement is association, not cause (master context §37).
- **Insider standing is company-specific.** A CEO's guidance IS company
  guidance, but that same CEO commenting on a rival is external there, and
  drops from 0.75 to 0.50 credibility.
- **Topic specialisation is bounded** (+0.15 on-topic, −0.20 off-topic). A
  CEO on their own product line is stronger evidence than the same CEO on
  macro policy, but specialisation informs rather than dominates.
- **Statement frequency is recorded, not scored.** A daily commentator
  plausibly carries less information per statement, but that is a hypothesis
  for E6 to test with evidence rather than a weight to bake in now.
- **Derived from E2, not maintained twice.** 67 actors built from the entity
  registry's executives. Funds contribute none — no officer speaks for an
  ETF, and listing one would be a false claim.
- CI drift gate: `scripts/check_actor_intelligence.py`. Verified non-vacuous
  by two sabotages: removing the impact threshold and removing the
  company-specific insider check both fail it.
- Acceptance: 35 tests in `tests/test_actor_intelligence.py`. Full suite:
  994/994 pass, twelve gates green.

### E4. Event study engine ✓ DONE

Implemented 2026-09-17 in `core/event_study.py` (`event-study-v1`).

The full chain per event — pre-event baseline → stock reaction → benchmark
reaction → sector reaction → abnormal return → volatility response → volume
response — across intraday / 1D / 5D / 20D / 60D. Horizons are drawn from
`LABEL_HORIZON_SESSIONS`, so a study and a training label always describe the
same window.

- **The baseline ends BEFORE the event, with a 2-session gap.** Information
  leaks into prices ahead of an announcement; including those sessions in
  "normal" would fold part of the reaction into the baseline and shrink the
  measured abnormal return toward zero. The study would quietly understate
  exactly what it exists to measure.
- **Anchored on PUBLICATION, not effective time** — the market reacts when it
  is told, which is what E1 kept those timestamps distinct for.
- **An unmatured window is ABSENT, never 0.0.** A zero would enter training
  as a measured absence of reaction. A recent event honestly reports only
  the horizons that elapsed.
- **No benchmark means no abnormal return** — never a stock return
  relabelled as abnormal. Benchmark and sector frames align by TIMESTAMP, so
  a benchmark with different history length cannot shift the window.
- **The model is named on every result** (`market_adjusted`, beta = 1). A
  full market model estimating beta from the baseline is the natural v2;
  calling this "the abnormal return" without naming the model would imply a
  sophistication that is not there.
- **Every result disclaims causality** (master context §37). E5 exists
  precisely because this module cannot make that claim.
- **Supplies E3 the returns it was refusing to invent.**
  `observations_from_studies` joins studied events to actor observations
  with real measured abnormal returns; anonymous events and unmeasured
  studies are skipped rather than recorded with a fabricated reaction.
  Verified end to end: 22 events → 22 studies → an actor track record
  crossing the 20-observation threshold into `measured`.
- CI drift gate: `scripts/check_event_study.py`. Verified non-vacuous by two
  sabotages — removing the baseline gap, and zero-filling an unmeasurable
  abnormal return — both fail with precise diagnoses.
- Acceptance: 34 tests in `tests/test_event_study.py`. Full suite:
  1028/1028 pass, thirteen gates green.

---

## Sprint R — Portfolio Risk Context (completing the fail-closed system)

The W2 policy table gates single-decision quality; this sprint adds the
portfolio dimension the design docs require before any paper posture can be
trusted as more than theater.

### R1. Portfolio state input

- `data/portfolio.json` (versioned schema): holdings
  `{ticker, quantity, average_cost, currency, as_of}`; loaded through a typed
  `PortfolioState` dataclass in `core/schemas.py`; absent file ⇒ empty
  portfolio with `INCOMPLETE` flag on risk payload (analysis proceeds, risk
  context marked degraded).
- All risk computations consume this state through the orchestrator — agents
  never read the file directly.

### R2. Risk checks (each a W2 policy rule with rule_id)

- Per-position cap: proposed/paper exposure vs portfolio notional > 1% → veto
  (`position_cap_exceeded`), per `docs/validation.md` hard limits.
- Total exposure cap 20%; daily loss 1%, weekly 3%, monthly drawdown 8% —
  breach of any halts order generation and reverts mode to `ANALYSIS_ONLY`
  (the doc-specified behavior), surfaced as `risk_halt_active` until a manual
  reset event is appended to the audit log.
- Concentration by correlation proxy: top-3 holdings' pairwise return
  correlation (60d, from cached bars) > 0.8 → warning severity; liquidity
  floor: position notional > X% of 20d ADV → veto.
- All thresholds live in `RISK_POLICY_V2` — no magic numbers in code paths.
- Acceptance: synthetic portfolio fixtures trip each rule exactly at its
  boundary; halt-state persists across decisions until the reset event.

### R3. Kill switches and mode separation hardening

- `KILL_SWITCH` env flag checked at the orchestrator entry — when set, every
  decision returns `NO_TRADE` with reason `kill_switch` regardless of inputs
  (fail-closed by construction, not by policy evaluation).
- Mode lattice enforced in one place: `ANALYSIS_ONLY`/`PAPER` reachable by
  evaluation; `LIVE_DISABLED` is the permanent default state constant;
  `LIVE_APPROVED` exists only as a schema value with no construction path —
  a deliberate absent-code guarantee, documented as such.

---

## Sprint D — Operator Dashboard and Explanation Surface

### D1. Per-agent contribution panel

- Render `ensemble_breakdown` (W1) as the primary panel: agent, weight
  (pre/post renormalization), contribution, status, model version; hover
  detail mirrors the existing confidence-tooltip pattern.
- Acceptance: the panel's sum reconciles with the headline score within
  rounding; a renormalized agent is visually distinguished.

### D2. Historical timeline and confidence drift

- `GET /api/history?ticker=&limit=` served from the enriched audit store
  (W7) — decision score, confidence, mode, veto reasons over time; UI renders
  a sparkline/timeline with drift band from the V4 metrics.
- Acceptance: timeline reflects decisions in strict as_of order regardless of
  insertion order; gaps (missing days) render as gaps, not zeros.

### D3. Veto and data-quality explanation surface

- Veto panel: structured rule id + severity + human text per W2; the
  decision-state card links each veto to the issuing agent.
- Data-quality panel: per-source freshness, quality flags, confidence factor
  values (from the existing breakdown) — stale/missing sources visually
  degrade the card, satisfying the design rule that degraded confidence can
  never hide.

### D4. API completeness contract

- `/api/score` response documented against a schema snapshot test: agent
  outputs, evidence references, model versions, source metadata, risk and
  audit results, ensemble breakdown, confidence breakdown — the full JSON
  contract from `docs/validation.md`, versioned via a top-level
  `api_schema_version`.
- Acceptance: a golden-file test fails on any undeclared response-shape
  change, forcing a deliberate version bump.

---

## Sprint X — Release Readiness and Operational Guardrails

### X1. Reproducible release snapshots

- `scripts/export_snapshot.py`: bundles a decision with its complete input
  state — raw records (W6), feature versions, ensemble/model versions, config
  snapshot, code commit — into a single verifiable archive (sha256 manifest
  inside). Any released score must be replayable from its archive alone.
- Acceptance: archive created on machine A replays bit-identically on
  machine B with only the repo + venv + archive.

### X2. Alerting rules (evaluate-locally, no infrastructure)

- Rule functions over audit/outcome stores: stale critical source > N hours,
  veto-rate spike, score-distribution drift beyond PSI threshold, limit
  breach (R2), audit-write failure. Output: structured alert records to
  `data/alerts.jsonl` + stderr; no external pager dependency at this stage.
- Acceptance: each rule has a positive and negative synthetic test; alerts
  carry the evidence ids that triggered them.

### X3. Human approval workflow

- Approval events are audit-log entries (`event_type: approval`) with actor,
  scope (model promotion / mode change / halt reset), and rationale — no
  separate approval system until one is justified.
- The 6-month/500-trade paper gate (validation doc) is a queried report
  (`scripts/gate_report.py`), not a dashboard claim: it computes the evidence
  or states plainly that it does not exist yet.
- Acceptance: gate report on today's data returns "gate not met" with the
  exact unmet criteria listed — honesty as a tested behavior.

### X4. Final hard gates

- Execution path: remains nonexistent. The release checklist asserts the
  absent-code guarantees (no order construction outside paper engine, no
  broker credentials in config surface, `LIVE_APPROVED` unreachable).
- Every gate from `docs/validation.md` mapped to a test or a script check,
  indexed in a single `RELEASE_GATES.md` with pass/fail provenance.

---

## Cross-Cutting Engineering Standards (bind every sprint above)

- Determinism: the score path contains no wall-clock reads, no randomness,
  no dict-order dependence; all timestamps attached at the orchestration
  boundary. Replay tests enforce per sprint, not once.
- Versioning discipline: any change to a formula, weight, threshold table,
  label definition, or classifier bumps its `*_VERSION` constant in the same
  commit; the old version's behavior stays documented in the audit trail.
- Fail-closed default: every new input, provider, or factor starts in the
  most restrictive posture; loosening requires a test that names the risk
  being accepted.
- No silent defaults: neutral values (e.g., RSI 50, base 4.0) are allowed
  only where the design doc states the neutral semantics; everywhere else,
  absence is a status, not a zero.
- Test conventions: provider adapters are hermetically mocked (recorded
  fixtures); network-touching integration tests are marked and kept few;
  every bug fix lands with the regression test that would have caught it
  (house rule proven by `4fbb92c`).
- Documentation-in-motion: each sprint updates its board section and the
  relevant design doc in the same PR as the code; `docs/sprint-board.md`
  checkboxes reflect reality, not aspiration.
- Performance guardrail: interactive score path stays within the 10s p95
  budget from `docs/architecture.md`; backtest/paper paths have no budget
  but must be offline-replayable.

---

## Recommended Execution Order

1. **Sprint W** (governance wiring) — unblocks and de-risks everything;
   small, fully testable increments.
2. **Sprint N** (news/sentiment/macro/regime) — context layer, agents born
   wired into the now-real ensemble.
3. **Sprint V** (labels → backtest → paper) — evidence engine; independent
   of N3/N4 depth, can start after W.
4. **Sprint M** (ML/calibration/registry) — strictly after V; the first
   dependency addition of the project happens here.
5. **Sprint R + D** (portfolio risk, operator surface) — R rides on W2's
   policy table; D rides on W7's enriched audit store.
6. **Sprint X** (release gates) — last, and mostly verification of what the
   earlier sprints should have made true.

Standing rule between sprints: stop, review actual behavior against the
acceptance criteria, and only then advance — per the working method already
established in `docs/sprint-board.md`.







