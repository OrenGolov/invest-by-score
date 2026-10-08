# INVEST-BY-SCORE — NEXT STEPS: SPRINT BREAKDOWN

> Extracted from `INVEST_BY_SCORE_MASTER_CONTEXT.md`. Sprint **W (Governance Wiring)** and everything before it (foundation, Sprints 1–4, later governance hardening) is already done — the document states the project "is now moving into Sprint N." Everything below is what comes next, in the document's recommended execution order: **N → V → M → E → C → F → L → R → D → A → X** (implementation can overlap where dependencies are clean; validation infrastructure must precede serious ML promotion).

---

## Sprint N — News / Sentiment / Macro / Regime
**Goal:** contextual intelligence — not simply "add a news agent."

- News adapter + classifier taxonomy + contradiction handling. Pipeline: NEWS → PIT FILTER → ENTITY RESOLUTION → EVENT CLASSIFICATION → SOURCE QUALITY → RELEVANCE/NOVELTY → DIRECTION/MAGNITUDE → CONTRADICTION DETECTION → EVIDENCE-BACKED OUTPUT. Taxonomy: earnings, guidance, regulation, litigation, product launch, M&A, macro shock, strategic announcement, management commentary, other. Contradictory credible evidence must produce `CONTRADICTORY`, not silently average to neutral.
- Sentiment placeholder + anti-proxying rule. Sentiment must not be fabricated — `UNAVAILABLE` if no legitimate source. Do not infer sentiment from RSI, price direction, technical indicators, or generic news score unless explicitly designed and documented as a derived feature. If news-derived, label it explicitly (e.g. `derived_from_news`) and reflect that dependency in confidence.
- Macro series registry + publication-time vintages. Each series needs: series identifier, source, publication timestamp, reference period, first-release value, revision history where available, frequency, unit, transformation, PIT policy, feature version. Macro data must be vintage-aware.
- 5-state regime classifier + stress → NO_TRADE coupling. States: `BULLISH`, `BEARISH`, `RANGE`, `RISK_OFF`, `STRESS`. Derived from explicit, versioned inputs. `STRESS` must couple to `NO_TRADE` per governance policy.
- Narrative-vs-fundamental attribution — decompose the score into contribution lines, e.g. Fundamental / Technical / News-narrative / Macro / Regime, so the system can explain whether the thesis is supported by business reality, market narrative, or both.
- Slotted feature gaps (ATR, slope, breadth) — deferred until properly wired, not bolted on ad hoc.

---

## Sprint V — Labels / Backtest / Paper
**Goal:** trustworthy historical validation before ML is layered on top.

- Labels must be leakage-safe; future information excluded.
- Walk-forward validation; embargo used where required.
- Transaction costs and slippage assumptions made explicit.
- Run manifests mandatory for every backtest.
- The same feature/scoring contracts must be used in research and production — no parallel ad hoc path.
- Historical universe construction must avoid survivorship bias.
- Corporate actions must be handled correctly.
- Framing: a backtest is evidence about historical behavior under explicit assumptions — not proof the future behaves the same way.

---

## Sprint M — ML Foundation & Model Registry

- **M1 — Canonical Feature Registry.** Every feature needs: name, owner, domain, formula, version, unit, frequency, lookback, minimum history, null policy, PIT rule, source dependencies, feature family, model compatibility. Acceptance: an unregistered feature cannot enter a production model; producer must exist; future/revised input is rejected; deterministic feature hash.
- **M2 — Official Training Dataset Builder.** The only official dataset generator. Every training row: prediction_time → information available at prediction_time → features → future outcome. Dataset must be deterministically hashed.
- **M3 — Research Trial Registry.** Record trial ID, hypothesis, feature-set version, model family, hyperparameters, label version, horizons, training window, validation scheme, costs, seed, dataset hash, metrics — protects against uncontrolled experimentation and cherry-picking.
- **M4 — Baseline Model Suite.** Establish strong simple baselines first: historical mean, momentum, mean reversion, linear/Ridge, logistic regression, tree/boosting. No advanced model is promoted without beating the incumbent out-of-sample.
- **M5 — Model Artifact Registry.** Store model version, feature-set version, training cutoff, dataset hash, code commit, hyperparameters, seed, artifact hash, calibration version, metrics, lifecycle status, parent model, approval, forecast target, horizon, universe.
- **M6 — Calibration and Uncertainty.** Support probability calibration, reliability curves, Brier score, log loss, calibration error, prediction intervals, uncertainty, fold dispersion. Never expose arbitrary probability numbers as if they were calibrated.
- **M7 — Champion / Challenger.** One Champion, multiple Challengers (A/B/C). New models always begin in `SHADOW`.
- **M8 — ML Reproducibility Gate.** Same data, features, seed, code, configuration must produce identical model artifacts/predictions within explicitly defined reproducibility guarantees.

---

## Sprint E — Event Intelligence & Market Reaction
**Goal:** learn how events actually move the chart.

- **E1 — Canonical Event Object.** Schema: event_id, published_time, effective_time, entity, entities_affected, actor, actor_type, event_type, source, source_quality, novelty, relevance, direction, magnitude, confidence, evidence.
- **E2 — Entity Resolution.** Resolve company names, tickers, executives, influential people, organizations, aliases. Bad resolution must not silently enter training.
- **E3 — Influential Person Intelligence.** Track identity, role, organization, historical relevance, topic specialization, source credibility, statement frequency, novelty, historical market impact. Eventually learn actor × topic × company/sector → historical market response.
- **E4 — Event Study Engine.** For every event: pre-event baseline → stock reaction → benchmark reaction → sector reaction → abnormal return → volatility response → volume response, measured intraday/1D/5D/20D/60D.
- **E5 — Confounder / Attribution Engine.** Never automatically claim causality — decompose observed movement into market component + sector component + stock-specific component + event-associated residual.
- **E6 — Event Memory.** Store event, context, chart state, historical analogs, and 1D/5D/20D/60D response as institutional memory.
- **E7 — Event Revision / Contradiction Learning.** Store event chains (initial claim → correction → confirmation → reversal) — the evolution itself becomes training information.

---

## Sprint C — Chart & Temporal Intelligence

- **C1 — Multi-Timeframe Representation.** Intraday, daily, weekly, monthly, yearly state, all aligned to the same `as_of`.
- **C2 — Price/Volume Feature Expansion.** Returns, momentum, acceleration, volatility, ATR, volume surprise, gaps, slope, drawdown, recovery speed, support/resistance distance, breakout/failure state, volatility expansion/contraction, relative strength.
- **C3 — Market Context Features.** S&P 500, Nasdaq, Russell 2000, VIX, 10Y yield, USD, sector ETF, industry benchmark.
- **C4 — Deterministic Chart Structure Representation.** Trend, consolidation, breakout, failed breakout, gap, reversal, volatility expansion/contraction, higher-high/higher-low, lower-high/lower-low. Avoid starting from screenshot interpretation or vague pattern labels.
- **C5 — Temporal Sequence Dataset.** Point-in-time sequences (T-60 … T0) with price, volume, technical state, market state, sector state, events, macro, sentiment, fundamentals.
- **C6 — Sequence Model Research.** Evaluate only after baseline validation: temporal convolution, LSTM/GRU, transformer/time-series transformer, temporal fusion. Complexity must earn its place through out-of-sample evidence.
- **C7 — Chart Reaction Memory.** Before-event / event / 1h / 1D / 5D / 20D / 60D reaction, retained for historical analog retrieval.

---

## Sprint F — Multi-Horizon Forecasting Engine
**Goal:** the central forecasting product.

- **F1 — Forecast Targets.** Direction P(return > 0), expected return E(return), return distribution P(return in interval), downside/adverse excursion, expected future volatility, relative performance P(stock > benchmark).
- **F2 — Horizons.** At minimum 1D, 5D, 20D, 60D, 120D, 252D.
- **F3 — Joint Forecast.** Report return + P(up) across all horizons together, with calibrated uncertainty.
- **F4 — Conditional Forecasting.** Condition on context, e.g. P(+5% in 20D | bullish regime) vs. P(+5% in 20D | stress regime).
- **F5 — Event-Conditioned Forecast.** New event → event representation → historical event matches → current chart → market regime → forecast.
- **F6 — Forecast Decomposition.** Break the forecast into Technical / Fundamental / News-Event / Macro / Regime / Sentiment / Historical-analog contributions — interpretable, without implying false causal certainty.
- **F7 — Forecast Confidence.** Accounts for sample size, calibration, model agreement, feature completeness, source quality, regime similarity, event similarity, model drift, uncertainty. Confidence is not the same thing as P(up).
- **F8 — Forecast API Contract.** Versioned `ForecastSnapshot`: ticker, as_of, forecast_version, model_versions, horizon, expected_return, probability_up, prediction_interval, confidence, regime, event_context, feature_digest, evidence, model_contributions, warnings.

---

## Sprint L — Continuous Learning & Institutional Memory
**Goal:** controlled learning, not uncontrolled self-modification.

- **L1 — Outcome Closure.** Forecast → horizon expires → actual outcome → error → calibration evaluation.
- **L2 — Forecast Performance Ledger.** Measured by ticker, sector, regime, horizon, event type, source, model, confidence bucket, volatility regime.
- **L3 — Error Memory.** Store forecast, actual, error, context; identify systematic errors.
- **L4 — Source Reliability Learning.** Source quality may depend on source × event type × sector × horizon — one global source score is not assumed sufficient.
- **L5 — Controlled Incremental Learning.** New data → candidate update → shadow evaluation → drift testing → OOS validation → promotion gate → human approval → new champion. Never auto-replace the production champion daily.
- **L6 — Concept Drift Detection.** Feature distribution drift, relationship drift, calibration drift, event-response drift.
- **L7 — Regime-Specific Learning.** Evaluate separate models per regime (bullish/bearish/range/risk-off/stress) only if OOS evidence supports specialization.
- **L8 — Champion Evolution.** Replacement requires credible OOS improvement, acceptable calibration, no unacceptable false-positive degradation, acceptable risk, regime robustness, reproducibility, governance approval. Historical forecasts remain immutable.

---

## Sprint R — Portfolio-Aware Forecasting
**Goal:** move from "is this ticker attractive?" to "does acting on this forecast improve the current portfolio without violating risk constraints?"

- R1 — position exposure
- R2 — correlation-aware sizing
- R3 — sector concentration
- R4 — forecast-adjusted risk
- R5 — expected portfolio impact
- R6 — stress scenarios
- R7 — portfolio-level `NO_TRADE`

---

## Sprint D — Research / Forecast Dashboard
**Goal:** a research workstation, not a black box.

- Show Investment Quality score, multi-horizon forecast (1D/5D/20D/60D with return% and probability), confidence, regime, and risk status together.
- Show a "WHY?" breakdown: Technical / Fundamental / News / Macro / Regime / Sentiment contributions.
- Show event context: recent event, classification, historical analog count, median historical response, current setup similarity.
- The UI must expose provenance, not hide it.

---

## Sprint A — Forecast Alerts
**Goal:** forecast-driven alerts, not simplistic score alerts.

- **A1 — Forecast Change.** Alert when the forecast materially changes.
- **A2 — Confidence Change.** Alert when confidence changes materially.
- **A3 — High-Impact Event.** Alert when a high-impact event is detected.
- **A4 — Regime Change.** E.g. `BULLISH → RISK_OFF`.
- **A5 — Thesis Break.** Alert when evidence materially contradicts the existing thesis.
- **A6 — Forecast Threshold.** E.g. 20D expected return + P(up) + confidence crossing a defined threshold, with no veto active.
- **A7 — Alert Suppression.** Prevent repeated/spam alerts. Alerts must respect risk and governance.

---

## Sprint X — Scientific Release Gate
**Goal:** the gate a model/forecast must pass before it's production-capable.

- **X1 — OOS forecast validation.** Demonstrate performance on unseen data.
- **X2 — Calibration gate.** Probabilities must be demonstrably calibrated.
- **X3 — Regime robustness.** No single regime should explain the entire edge.
- **X4 — Event robustness.** No single viral event/source should explain the apparent edge.
- **X5 — Feature ablation.** Test removing news, technicals, macro, sentiment, fundamentals to prove which sources add incremental value.
- **X6 — Temporal robustness.** Evaluate across 1D/5D/20D/60D/120D.
- **X7 — Multiple-testing protection.** Guard against selection bias (bootstrap/permutation, Reality Check/SPA-style methods, Probability of Backtest Overfitting, Deflated Sharpe where applicable).
- **X8 — Sealed holdout.** Maintain an untouched final period; once opened, never use it for further model selection.
- **X9 — Release snapshot.** Freeze code, data, features, models, weights, calibration, configuration, validation results.
- **X10 — Honest gate report.** If evidence is insufficient, report `FORECASTING RELEASE: NOT APPROVED` with the reason and next action. "Gate not met" is a valid, successful engineering outcome.

---

## Constraints that apply across every sprint above

- Do not add ML before the dataset/label/backtest foundation is trustworthy.
- Do not use future information, or revised macro data in historical snapshots without PIT handling.
- Do not infer sentiment from price unless explicitly defined as a derived feature.
- Do not treat news polarity as causality, or claim an influential person caused a price move without proper attribution methodology.
- Do not silently replace missing values with neutral evidence, or let a failed agent masquerade as `OK`.
- Do not let risk/audit vetoes be overridden by a high score.
- Do not create duplicate technical scoring implementations.
- Do not let models consume unregistered features, or let experiments bypass the trial registry.
- Do not automatically promote a newly trained model, or let continuous learning silently mutate the production champion.
- Do not optimize on the final holdout.
- Do not build a complex neural network before proving simple baselines.
- Do not use screenshots/chart patterns as vague "AI intuition" without a measurable representation.
- Do not add indicators just because they look useful.
- Do not let the dashboard hide uncertainty or provenance.
