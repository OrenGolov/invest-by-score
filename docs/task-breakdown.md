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

### N1. News intelligence agent (real ingestion + classification)

- Adapter: new `core/news_adapter.py` behind the existing contract
  (`fetch_news_snapshot` signature unchanged). Registry entry `news` in
  `SOURCE_REGISTRY` with `status: provider_key_required` until
  `NEWS_PROVIDER_API_KEY` resolves — the no-key path must remain the current
  explicit `UNAVAILABLE` contract, byte-for-byte.
- Query window: articles with `published_time <= as_of`; window end at as_of,
  start at as_of − lookback (default 7d, config). Future-dated items are
  rejected `INVALID` per the taxonomy.
- Per-article record: `{source_id, source_record_id, published_time, headline,
  url, category, tone, relevance, source_weight}`.
- Classifier v1 (rule-based, `NEWS_CLASSIFIER_VERSION`): category taxonomy
  exactly — earnings, guidance, litigation, regulation, product_launch,
  macro_shock, m_and_a; curated pattern sets stored as data constants, not
  inline regex sprawl; unmatched → `other`.
- Tone v1: provider-supplied when available, else lexicon scoring (documented
  negation handling); tone ∈ [−1, 1], derivation stamped per record.
- Relevance: entity/ticker match quality; relevance 0 excludes from
  aggregation but keeps the article in evidence.
- Source weighting: registry `base_confidence` × recency decay (exponential,
  configurable half-life, default 3 trading days).
- Contradiction v1: same-day same-category cluster with opposite-sign mean
  tone and |Δ| > 0.6 → agent status `CONTRADICTORY`, both sides surfaced in
  evidence; contradictory ⇒ confidence floor, never a neutral average.
- Aggregation: `sentiment_score` = relevance- and source-weighted,
  recency-decayed tone mean; `confidence` = f(sample count, mean source
  weight, dispersion), capped by the weakest constituent; empty ⇒
  `UNAVAILABLE` exactly as today.
- Acceptance: a positive headline from a zero-quality source cannot raise
  confidence; a contradictory cluster yields the explicit status; as_of
  filtering provable via a future-dated-article test.

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

### N3. Macroeconomic agent

- Series registry `core/macro_registry.py`: logical series (fed_funds,
  cpi_yoy, initial_claims, gdp_growth, 10y_yield) → provider id, unit,
  release cadence, publication-lag convention; FRED-style adapter with
  `provider_key_required` fallback → `UNAVAILABLE`.
- Point-in-time rule is non-negotiable: eligibility uses the release
  `published_time` (first-release vintage), never the reference-period end;
  revisions append a new version line to the raw store (W6).
- Sector sensitivity v1: static GICS-sector → factor-loadings (rates, energy,
  dollar) as versioned code constants (`MACRO_SENSITIVITY_VERSION`); symbol →
  sector via a maintained table.
- Output: `macro_score` (risk-on/off tilt), scenario notes, per-series
  contributions with evidence ids; a missing series degrades confidence
  (`INCOMPLETE`), never zero-fills into neutrality.
- Acceptance: a release published after as_of is provably excluded; missing
  series is visible and penalized.

### N4. Market Regime agent (upgrade from the 3-state heuristic)

- Extend classification to the doc's full set:
  `bullish, bearish, range, risk_off, stress` — versioned
  `REGIME_CLASSIFIER_VERSION`, deterministic rules from existing features:
  - `range`: |price_vs_ma_50| < 0.02 and |price_vs_ma_200| < 0.02 (exists).
  - `stress`: 30d realized vol above its own 1y 95th percentile AND drawdown
    from the 60d high > 15%.
  - `risk_off`: vol above the 1y 80th percentile, or MA50 < MA200 with
    aligned negative 20d/60d momentum.
  - bull/bear as today; `range` takes precedence when flat.
- Output: `{label, probability_proxy (distance from boundary, scaled),
  transition_risk (regime flips per trailing 20 sessions)}`.
- Governance coupling: `stress` forces `NO_TRADE` via a W2 policy rule;
  `risk_off` applies a documented dampening factor to momentum weights in
  config.
- Acceptance: boundary tests at each threshold/percentile; a stress snapshot
  cannot reach `PAPER` regardless of score.

### N5. Narrative vs fundamental attribution

- Attribution block in the scoring breakdown classifying each weighted
  contribution: `operational` (fundamental, long-horizon technical),
  `narrative` (news, sentiment), `macro_shock` (macro, regime transitions).
- The explanation string assembles from this block so the UI can state *why*
  the score moved — narrative vs operational drivers, per the sprint's
  acceptance criterion.
- Acceptance: with news at zero weight the narrative bucket reads exactly
  `0.0` — no phantom narrative.

### N6. Deferred-but-slotted feature gaps (from the Sprint-2 review)

- ATR(14), 60d trend slope (linear regression of closes), 50/100/150/200d
  trailing returns — added to the feature contract with full provenance; none
  may enter a scorer without a weight and a test.
- Breadth/participation: requires an index-constituent adapter; explicitly
  deferred with a `breadth` registry placeholder (`provider_key_required`).

---

## Sprint V — Outcome Labels, Walk-Forward Backtest, Paper Engine

Ordering rule: this sprint precedes any ML work. A model trained before a
leakage-safe label and validation harness exists would be unvalidatable by
construction. The shared-research contract in `docs/validation.md` is binding:
the backtester consumes the same feature contracts and scorers as live —
no side research dataset, ever.

### V1. Outcome label builder

- Objective: produce point-in-time-safe labels for every persisted decision.
- Labels per decision, computed strictly from bars in `(as_of, as_of + h]`:
  `forward_return_1d/5d/20d/60d`, `adverse_excursion` (worst drawdown from
  entry close within the 20d window), `label_20d_up` (boolean at a
  configurable threshold), and a risk-adjusted outcome (return / realized
  vol over the window).
- Boundary rule: a label is `null` until the horizon has fully elapsed
  relative to the data's latest bar; partially-elapsed horizons are never
  emitted (no partial-window leakage). The builder derives eligibility from
  the latest bar timestamp, not wall-clock.
- Storage: append-only `data/outcomes.jsonl` keyed
  `{ticker, as_of, horizon, label_version}`; label version stamped
  (`OUTCOME_LABEL_VERSION`) so threshold changes never rewrite history.
- Acceptance: a decision dated 10 sessions ago has `5d` labels but `20d =
  null`; synthetic-bar tests prove off-by-one safety at horizon boundaries.

### V2. Walk-forward backtest engine

- Objective: replay the live scoring path over history with realistic costs
  and strict temporal hygiene, producing comparable run manifests.
- Folding: train window `[t0, t1]` → embargo (≥ max horizon, i.e. 60 sessions)
  → validation `[t1 + embargo, t2]` → advance; final frozen configuration
  evaluated once on a never-touched tail period. The embargo applies to both
  features (via PIT eligibility, already enforced) and labels (V1).
- Execution model: decisions at bar `t` act at bar `t+1` open with costs —
  spread (bps by liquidity bucket), square-root market-impact slippage as a
  function of participation, commission — all parameters in a versioned cost
  table, never inline.
- Metrics module (pure functions, deterministic): CAGR, Sharpe (annualized,
  configurable rf), Sortino, Calmar, max drawdown, win rate, profit factor,
  exposure, turnover, rejection rate. Emit per-fold and aggregate.
- Run manifest (required, persisted next to results): code commit, feature
  versions, ensemble version, cost table version, data hashes (raw store
  digests), random seed if any, config snapshot. A run without a manifest is
  invalid by definition.
- Implementation shape: `core/backtest/` package — `engine.py`,
  `costs.py`, `metrics.py`, `manifest.py`; reuses `fetch_market_snapshot` via
  a cached-history injection seam so backtests run offline from the raw store.
- Acceptance: a deliberately leaked variant (labels shifted one bar early)
  is detected and rejected by the harness; identical inputs rerun to
  identical metrics; cost parameters demonstrably affect results.

### V3. Paper-trading order engine (simulation only)

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

### V4. Monitoring foundations

- Metrics snapshot per backtest/paper run: score distribution drift
  (population stability index vs trailing window), stale-data rate,
  veto-rate by rule, confidence drift — computed by pure functions over the
  audit/outcome stores, no dashboard dependency.
- Acceptance: drift function detects a seeded distribution shift in
  synthetic data; missing data degrades the report explicitly.

---

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







