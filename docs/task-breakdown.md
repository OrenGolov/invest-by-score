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

### W1. Versioned ensemble wiring — agent outputs drive the final score

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

### W2. Risk agent becomes a real fail-closed gate

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

### W3. Auditor agent becomes an evidence-and-replay validator

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

### W4. Failure-state taxonomy across every contract

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

### W5. Single technical truth

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

### W6. Append-only raw-record store (Sprint-1 leftover, minimal viable)

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

### W7. Audit event enrichment

- Status: **implemented** — events are now `audit-event-v2`: `schema_version`,
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
- Classifier v1 (curated pattern sets, `NEWS_CLASSIFIER_VERSION`): earnings,
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

### N2. Sentiment agent (social/positioning, distinct from news tone)

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
  `ANALYSIS_ONLY`. The `derived_from_news` labeling (×0.5 confidence) is a
  documented, test-pinned contract for any future news-derived design.
  `tests/test_sentiment_contract.py` pins the shape byte-for-byte.

### N3. Macroeconomic agent ✓ DONE

Implemented 2026-09-01. Full vintage-aware series registry, PIT filtering,
risk regime classification, and sector sensitivity mapping complete.

- Series registry: `core/macro_registry.py` (data constant `MacroSeries` entries
  for fed_funds, cpi_yoy, initial_claims, gdp_growth, 10y_yield). Each series
  carries complete metadata: provider (FRED), series_id, unit, frequency,
  transformation, publication_lag_days, reference_period_field,
  published_time_field, feature_version, lookback_periods, description.
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
  value, and credibility status. Aggregation uses latest-version records.
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

### V6. Paper-trading order engine (simulation only)

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

## Sprint M — ML Layer, Calibration, Model Registry

Preconditions: Sprint V complete (labels exist, harness exists, paper
evidence accumulates). Dependency note: this sprint introduces the project's
first training dependency (scikit-learn is the pragmatic choice) — that is a
requirements.txt change requiring explicit review, plus pinning consistent
with the existing style.

### M1. Feature registry (single source of truth)

- `core/feature_registry.py`: every scored feature declared as
  `{name, owner_agent, domain, dtype, unit, lookback, null_policy,
  calculation_version, min_history}`; a `validate_snapshot(snapshot)`
  conformance check runs in the score path (dev/test modes) and in CI —
  registry and snapshot drift is a build failure, not a runtime surprise.
- Registry version `FEATURE_REGISTRY_VERSION` participates in replay hashes
  and run manifests. Adding a feature without registry entry: rejected.
- Acceptance: removing a snapshot feature or renaming one breaks CI with a
  diff-precise message; a registry entry with no producer fails the inverse
  check.

### M2. Model registry and artifact tracking

- `models/manifest.json` (append-friendly, versioned entries):
  `{model_version, family, feature_set_version, training_data_cutoff,
  artifact_uri, status: candidate|approved|retired, metrics: {...},
  approved_by, approved_at, parent_version}`.
- Rules enforced in code: a model referenced by a live decision must be
  `approved`; promotion candidate→approved requires an out-of-sample
  comparison row against the incumbent plus a human `approved_by`; retirement
  never deletes artifacts. `ScoreResult.model_version` fields populate from
  here (currently hard-coded `technical-v1` etc. — those strings become
  registry lookups).
- Acceptance: referencing an unapproved/retired model in the score path is
  impossible without an explicit override that itself is audited.

### M3. Training pipeline (offline, reproducible)

- `scripts/train.py`: loads point-in-time-eligible features (via the same
  snapshot code paths against the raw store — never a parallel extractor),
  joins V1 labels, applies the M1 registry, trains baselines:
  regularized linear (Ridge/ElasticNet), RandomForest, GradientBoosting.
  Sequence/temporal models are explicitly out of scope until the baselines
  survive V2 validation.
- Time-safe splits come from the V2 harness (folds + embargo), not from
  sklearn defaults; class/label imbalance handling documented in the run
  manifest; seeds fixed and recorded.
- Determinism: same seed + same data digests → identical artifact hash.
- Acceptance: two training runs with identical manifests produce identical
  metrics and artifact hashes; a feature added without registry entry aborts
  training.

### M4. Calibration and score mapping

- Calibrate raw model output to probabilities on validation folds only
  (isotonic preferred, Platt fallback for small folds); calibration map is
  versioned and shipped with the artifact.
- Documented, monotone mapping calibrated-probability → 0–10 score with
  confidence/uncertainty band derived from fold-wise dispersion. The 0–10
  score remains *not* a probability of profit (design-doc language).
- The evidence-confidence-v2 model (current) is retained as the
  no-model/uncertainty overlay: final confidence = min(model confidence,
  evidence confidence) until V-series evidence justifies replacement — the
  replacement itself is a gated promotion, not a cutover.
- Acceptance: calibration reliability curve reported per fold; mapping
  monotonicity unit-tested; a model prediction without calibration artifacts
  is rejected.

### M5. Promotion gates and drift hooks

- Promotion checklist automated in `scripts/promote.py`: OOS metrics beat
  incumbent on the pre-registered primary metric, no regression on veto-rate
  or false-positive rate beyond tolerance, drift check (V4) clean, manifest
  complete, human approval recorded. Any failure → candidate stays.
- Historical predictions are immutable: promotion never rewrites past
  decisions' model versions (append-only audit guarantees this).
- Acceptance: attempt to promote with a missing manifest field fails loudly;
  the full gate sequence is exercised in a test with a synthetic candidate.

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







